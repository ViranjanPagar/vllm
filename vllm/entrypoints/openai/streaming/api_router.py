# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State setup for the video streaming serving.

The streaming feature is exposed via ``/v1/chat/completions`` — when a
request contains an ``rtsp://`` (or rtsps/rtmp) ``video_url`` and
``stream=true``, the standard chat-completions handler dispatches to
:class:`VideoStreamingServing` for continuous segment captioning. This
module no longer registers a dedicated route.
"""

from typing import TYPE_CHECKING

from fastapi import FastAPI

from vllm.entrypoints.openai.streaming.serving import VideoStreamingServing
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object


def attach_router(app: FastAPI) -> None:
    """No-op kept for symmetry with other entrypoint modules.

    RTSP streaming reuses the existing ``/v1/chat/completions`` route via
    a dispatch in :class:`OpenAIServingChat.create_chat_completion`, so
    no separate FastAPI route registration is needed here.
    """
    del app
    logger.debug(
        "Streaming module attached: RTSP delegation handled inside "
        "/v1/chat/completions; no dedicated route registered."
    )


def init_streaming_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    state.video_streaming_serving = (
        VideoStreamingServing(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if "generate" in supported_tasks
        else None
    )
