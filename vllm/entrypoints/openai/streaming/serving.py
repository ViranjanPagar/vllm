# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSE serving class for continuous video-stream captioning.

For each segment produced by the persistent DeepStream ``stream_uri``
pipeline (running in a subprocess), this class:

1. Builds a chat-template prompt with the model's video placeholder.
2. Attaches decoded frames as ``multi_modal_data``.
3. Runs VLM inference via ``engine_client.generate()``.
4. Yields the caption as an SSE event.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy.typing as npt
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logger import init_logger
from vllm.multimodal.rtsp_stream_manager import RTSPStreamManager
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = init_logger(__name__)


def _extract_user_text(messages) -> str:
    """Pull the concatenated text content from the most recent user message.
    Falls back to the default caption prompt if none is found."""
    default = "Describe what is happening in this video segment."
    for msg in reversed(list(messages)):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = (
            msg.get("content") if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if isinstance(content, str):
            return content or default
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                else:
                    if getattr(part, "type", None) == "text":
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            texts.append(text)
            if texts:
                return "\n".join(texts)
        return default
    return default


class VideoStreamingServing(OpenAIServing):
    """Continuous video-stream captioning via SSE."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )
        self._stream_manager = RTSPStreamManager()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_video_prompt(self, user_text: str) -> str:
        """Apply the model's chat template for a single-turn video+text
        user message.

        The HF Jinja template expands ``{"type": "video"}`` into the
        model-specific video placeholder tokens (e.g. Qwen2-VL uses
        ``<|vision_start|><|video_pad|><|vision_end|>``).
        """
        tokenizer = self.renderer.tokenizer
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def prewarm(
        self,
        sample_path: str = "/data/video/sample_1080p_15s.mp4",
        num_frames: int = 32,
        chunk_duration: float = 10.0,
    ) -> None:
        """Start the DeepStream pipeline early using the sample video.

        Runs the decode in a background task so server startup is not blocked.
        The task drains frames and exits when the file ends.
        """
        if not os.path.exists(sample_path):
            logger.warning(
                "Prewarm sample video not found at %s, skipping pipeline pre-warm",
                sample_path,
            )
            return

        uri = f"file://{sample_path}"
        logger.info("Pre-warming pipeline with %s", sample_path)
        consumer_id, seg_queue = await self._stream_manager.subscribe(
            uri=uri,
            chunk_duration=chunk_duration,
            num_frames=num_frames,
        )

        async def _drain() -> None:
            try:
                while True:
                    item = await seg_queue.get()
                    if item is None:
                        break
            finally:
                await self._stream_manager.unsubscribe(
                    uri=uri,
                    chunk_duration=chunk_duration,
                    num_frames=num_frames,
                    consumer_id=consumer_id,
                )
            logger.info("Pipeline prewarm complete")

        asyncio.create_task(_drain(), name="pipeline-prewarm")

    # ------------------------------------------------------------------
    # Chat-completions-shaped streaming (RTSP via /v1/chat/completions)
    # ------------------------------------------------------------------

    async def create_video_chat_stream(
        self,
        request,  # ChatCompletionRequest from vllm.entrypoints.openai.chat_completion
        raw_request: Request,
        rtsp_url: str,
    ) -> AsyncGenerator[str, None] | ErrorResponse:
        """Validate and return a chat.completion.chunk SSE generator for an
        RTSP source. Reuses the persistent DeepStream pipeline managed by
        :class:`RTSPStreamManager`; emits one
        ``chat.completion.chunk`` per decoded segment caption."""
        if not self._is_model_supported(request.model):
            return self.create_error_response(
                message=f"The model {request.model!r} is not available.",
            )
        return self._stream_chat_segments(request, raw_request, rtsp_url)

    async def _stream_chat_segments(
        self,
        request,
        raw_request: Request,
        rtsp_url: str,
    ) -> AsyncGenerator[str, None]:
        """Yield ``data: {chat.completion.chunk}\\n\\n`` SSE events — one
        per decoded segment caption, terminated by a ``finish_reason: "stop"``
        chunk and a literal ``data: [DONE]``. Mirrors the RTVI wire format:
        same ``id`` / ``created`` across all chunks; per-segment caption goes
        into ``delta.content``; no PTS or segment_index in the JSON.
        """
        # Extract custom extras (chunk_duration, num_frames_per_second_or_fixed_frames_chunk)
        # from request.model_extra — OpenAIBaseModel is configured with extra='allow'.
        extras = getattr(request, "model_extra", None) or {}
        chunk_duration = float(extras.get("chunk_duration") or 10.0)
        num_frames_raw = extras.get("num_frames_per_second_or_fixed_frames_chunk")
        if num_frames_raw is None:
            num_frames_raw = extras.get("num_frames", 8)
        try:
            num_frames = int(num_frames_raw)
        except (TypeError, ValueError):
            num_frames = 8

        prompt_text = _extract_user_text(request.messages)
        request_id = f"chatcmpl-{random_uuid()}"
        created = int(time.time())
        model_name = request.model or ""

        consumer_id: str | None = None
        try:
            consumer_id, segment_queue = await self._stream_manager.subscribe(
                uri=rtsp_url,
                chunk_duration=chunk_duration,
                num_frames=num_frames,
            )

            while True:
                if await raw_request.is_disconnected():
                    break

                segment = await segment_queue.get()
                if segment is None:
                    break

                frames: npt.NDArray = segment[0]
                metadata: dict[str, Any] = segment[1]
                seg_request_id = f"vseg-{random_uuid()}"

                try:
                    if os.environ.get("VLLM_STUB_RESPONSE", "0").strip() in ("1", "true", "True"):
                        caption = os.environ.get(
                            "VLLM_STUB_CAPTION",
                            "This is a stub response. Model inference is disabled.",
                        )
                    else:
                        caption = await self._infer_chat_caption(
                            frames, request, prompt_text, seg_request_id,
                        )
                except Exception:
                    logger.exception(
                        "Error inferring segment %d",
                        metadata.get("segment_index", -1),
                    )
                    continue

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": caption},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            terminal = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(terminal)}\n\n"
            yield "data: [DONE]\n\n"

        finally:
            if consumer_id is not None:
                await self._stream_manager.unsubscribe(
                    uri=rtsp_url,
                    chunk_duration=chunk_duration,
                    num_frames=num_frames,
                    consumer_id=consumer_id,
                )

    async def _infer_chat_caption(
        self,
        frames: npt.NDArray,
        request,
        prompt_text: str,
        request_id: str,
    ) -> str:
        """Run VLM inference on one segment and return the caption text."""
        full_prompt = self._build_video_prompt(prompt_text)

        sampling_params = SamplingParams(
            temperature=request.temperature if request.temperature is not None else 0.0,
            max_tokens=(
                request.max_completion_tokens
                or request.max_tokens
                or 256
            ),
        )
        engine_prompt: dict[str, Any] = {
            "prompt": full_prompt,
            "multi_modal_data": {"video": frames},
        }
        caption = ""
        async for output in self.engine_client.generate(
            engine_prompt,
            sampling_params,
            request_id,
        ):
            if output.outputs:
                caption = output.outputs[0].text
        return caption
