# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manages persistent DeepStream RTSP pipelines in daemon threads.

Each unique ``(uri, num_frames)`` tuple maps to one daemon thread running
:pymeth:`DeepStreamVideoBackend.stream_uri`.  Frames stay GPU-resident
throughout — the CUDA tensor yielded by ``stream_uri`` is passed directly
across the thread boundary without any CPU copy.

Frame data crosses the thread boundary via a sync ``queue.Queue`` — the
worker puts ``(frames_tensor, metadata)`` there; an asyncio bridge task reads
it with ``run_in_executor`` and fans out to per-consumer ``asyncio.Queue``s.

No subprocess, no ``SharedMemory``, no subprocess lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt

logger = logging.getLogger(__name__)

_SENTINEL = None


# ------------------------------------------------------------------
# Per-stream bookkeeping
# ------------------------------------------------------------------

@dataclass
class _StreamState:
    thread: threading.Thread
    sync_queue: queue.Queue  # type: ignore[type-arg]
    stop_event: threading.Event
    consumers: dict[str, asyncio.Queue]
    bridge_task: asyncio.Task[None] | None = None
    _closed: bool = field(default=False, init=False)

    def add_consumer(self) -> tuple[str, asyncio.Queue]:
        cid = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.consumers[cid] = q
        return cid, q

    def remove_consumer(self, cid: str) -> int:
        self.consumers.pop(cid, None)
        return len(self.consumers)


# ------------------------------------------------------------------
# Worker thread target
# ------------------------------------------------------------------

def _worker(
    uri: str,
    num_frames: int,
    chunk_duration: float,
    sync_q: queue.Queue,  # type: ignore[type-arg]
    stop_event: threading.Event,
) -> None:
    logger.info(
        "[stream worker] started for %s  num_frames=%d buffer_sec=%.1f",
        uri, num_frames, chunk_duration,
    )
    try:
        from vllm.multimodal.video import DeepStreamVideoBackend

        is_file = uri.startswith("file://")
        if is_file:
            file_path = uri[len("file://"):]
            stream_gen = DeepStreamVideoBackend.stream_file_chunked(
                file_path,
                num_frames=num_frames,
                chunk_duration_sec=chunk_duration,
            )
        else:
            stream_gen = DeepStreamVideoBackend.stream_uri(
                uri,
                num_frames=num_frames,
                chunk_duration=chunk_duration,
            )

        for frames_tensor, metadata in stream_gen:
            if stop_event.is_set():
                break
            try:
                sync_q.put((frames_tensor, metadata), timeout=5)
            except queue.Full:
                logger.warning("[stream worker] sync queue full, dropping segment")
    except Exception as exc:
        logger.error("[stream worker] error for %s: %s", uri, exc, exc_info=True)
    finally:
        sync_q.put(_SENTINEL)
    logger.info("[stream worker] exiting for %s", uri)


# ------------------------------------------------------------------
# Singleton manager
# ------------------------------------------------------------------

class RTSPStreamManager:
    _instance: RTSPStreamManager | None = None

    def __new__(cls) -> RTSPStreamManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._streams = {}
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    _streams: dict[tuple[str, float, int], _StreamState]
    _lock: asyncio.Lock

    async def subscribe(
        self,
        uri: str,
        chunk_duration: float = 10.0,
        num_frames: int = 32,
    ) -> tuple[str, asyncio.Queue]:
        key = (uri, chunk_duration, num_frames)

        async with self._lock:
            state = self._streams.get(key)
            if state is None:
                state = self._spawn(uri, num_frames, chunk_duration)
                self._streams[key] = state
                loop = asyncio.get_running_loop()
                state.bridge_task = loop.create_task(
                    self._bridge(key, state),
                    name=f"rtsp-bridge-{uri}",
                )
            cid, q = state.add_consumer()

        logger.info(
            "RTSP subscribe: consumer=%s uri=%s consumers=%d",
            cid, uri, len(state.consumers),
        )
        return cid, q

    async def unsubscribe(
        self,
        uri: str,
        chunk_duration: float,
        num_frames: int,
        consumer_id: str,
    ) -> None:
        key = (uri, chunk_duration, num_frames)
        async with self._lock:
            state = self._streams.get(key)
            if state is None:
                return
            remaining = state.remove_consumer(consumer_id)
            logger.info(
                "RTSP unsubscribe: consumer=%s remaining=%d",
                consumer_id, remaining,
            )
            if remaining == 0:
                self._teardown(key, state)

    # ---- internals -------------------------------------------------

    def _spawn(
        self,
        uri: str,
        num_frames: int,
        chunk_duration: float,
    ) -> _StreamState:
        stop_ev = threading.Event()
        sync_q: queue.Queue = queue.Queue(maxsize=8)
        t = threading.Thread(
            target=_worker,
            args=(uri, num_frames, chunk_duration, sync_q, stop_ev),
            daemon=True,
            name=f"rtsp-{uri[:40]}",
        )
        t.start()
        logger.info("Spawned RTSP thread for %s", uri)
        return _StreamState(
            thread=t,
            sync_queue=sync_q,
            stop_event=stop_ev,
            consumers={},
        )

    async def _bridge(
        self,
        key: tuple[str, float, int],
        state: _StreamState,
    ) -> None:
        loop = asyncio.get_running_loop()
        uri = key[0]
        seg = 0
        logger.info("[RTSP bridge] started for %s", uri)

        cumulative_frames: int = 0
        while True:
            item = await loop.run_in_executor(None, state.sync_queue.get)
            if item is _SENTINEL:
                logger.info("[RTSP bridge] sentinel received for %s", uri)
                for q in list(state.consumers.values()):
                    await q.put(None)
                break

            frames_tensor, raw_metadata = item
            fps = raw_metadata.get("fps", 0.0)
            n_frames = raw_metadata.get("total_num_frames", 0)
            pts_start = cumulative_frames / fps if fps > 0 else 0.0
            pts_end = (cumulative_frames + n_frames) / fps if fps > 0 else 0.0
            metadata = {
                **raw_metadata,
                "segment_index": seg,
                "pts_start": pts_start,
                "pts_end": pts_end,
                "duration": pts_end - pts_start,
            }
            cumulative_frames += n_frames

            for q in list(state.consumers.values()):
                try:
                    q.put_nowait((frames_tensor, metadata))
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait((frames_tensor, metadata))
                    except asyncio.QueueFull:
                        pass
            seg += 1

        logger.info("[RTSP bridge] exiting for %s", uri)
        async with self._lock:
            if key in self._streams:
                self._teardown(key, state)

    def _teardown(self, key: tuple, state: _StreamState) -> None:
        if state._closed:
            return
        state._closed = True
        state.stop_event.set()
        if state.bridge_task and not state.bridge_task.done():
            state.bridge_task.cancel()
        state.bridge_task = None  # break reference cycle: task -> state -> task
        state.thread.join(timeout=5)
        if state.thread.is_alive():
            logger.warning("RTSP worker thread still alive after 5s for %s", key[0])
        self._streams.pop(key, None)
        logger.info("Torn down RTSP stream for %s", key[0])

    async def shutdown_all(self) -> None:
        async with self._lock:
            for key, state in list(self._streams.items()):
                self._teardown(key, state)
