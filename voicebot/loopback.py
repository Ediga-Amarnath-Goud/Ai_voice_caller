"""WASAPI loopback input transport with debounced call detection.

Captures the Realtek speakers' loopback twin and pushes 24 kHz PCM into the
pipeline. No DSP: the caller's voice is passed through unchanged, which the
audit confirmed is already clear enough for Gemini's speech-to-speech.

Call detection is edge-triggered:
  * ARM after N consecutive frames above threshold (so a single click won't fire)
  * clear only after CALL_OFF_MS of continuous quiet (so mid-word dips don't
    reset it). The old build cleared after 3 quiet frames and chattered badly.
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import soundcard as sc
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_transport import TransportParams

from . import config


def _find_loopback(name: str):
    for d in sc.all_microphones(include_loopback=True):
        try:
            if getattr(d, "isloopback", False) and str(d.name) == name:
                return d
        except Exception:
            continue
    return None


class LoopbackInputTransport(BaseInputTransport):
    def __init__(self, params: TransportParams) -> None:
        super().__init__(params)
        self._recorder = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._block = int(config.RATE_IN * config.BLOCK_MS / 1000)
        self._threshold = 10.0 ** (config.CALL_THRESHOLD_DBFS / 20.0)
        self._call_active = threading.Event()
        self._loud_streak = 0
        self._last_loud = 0.0
        self._lock = threading.Lock()
        self._pushed = 0
        self._reported = time.monotonic()

    @property
    def call_active(self) -> threading.Event:
        return self._call_active

    def last_audio_age(self) -> float:
        """Seconds since a frame was above threshold (inf if never)."""
        with self._lock:
            if not self._last_loud:
                return float("inf")
            return time.monotonic() - self._last_loud

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        device = _find_loopback(config.LOOPBACK_DEVICE)
        if device is None:
            available = [
                str(m.name) for m in sc.all_microphones(include_loopback=True)
                if getattr(m, "isloopback", False)
            ]
            raise RuntimeError(
                f"loopback endpoint {config.LOOPBACK_DEVICE!r} not found; available: {available}"
            )
        self._recorder = device.recorder(
            samplerate=config.RATE_IN, channels=1, blocksize=self._block
        )
        self._recorder.__enter__()
        logger.info(f"loopback open: {config.LOOPBACK_DEVICE} @ {config.RATE_IN} Hz")

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        await self.set_transport_ready(frame)

    def _capture_loop(self) -> None:
        loop = self.get_event_loop()
        scale = 32767.0
        off_seconds = config.CALL_OFF_MS / 1000.0
        while not self._stop.is_set() and self._recorder is not None:
            try:
                data = np.asarray(
                    self._recorder.record(numframes=self._block), dtype=np.float32
                ).reshape(-1)
            except Exception as exc:
                logger.warning(f"loopback read failed: {type(exc).__name__}: {exc}")
                continue
            if data.size == 0:
                continue

            peak = float(np.abs(data).max())
            db = 20.0 * np.log10(peak) if peak > 0 else -120.0
            loud = peak > self._threshold
            now = time.monotonic()

            with self._lock:
                if loud:
                    self._last_loud = now
                    self._loud_streak += 1
                    if not self._call_active.is_set() and self._loud_streak >= config.CALL_ON_FRAMES:
                        self._call_active.set()
                        logger.info(f"call active: caller audio at {db:.1f} dBFS")
                else:
                    self._loud_streak = 0
                    if self._call_active.is_set() and (now - self._last_loud) >= off_seconds:
                        self._call_active.clear()
                        logger.info("call inactive: caller audio went quiet")

            pcm = (np.clip(data, -1.0, 1.0) * scale).astype("<i2").tobytes()
            frame = InputAudioRawFrame(
                audio=pcm, sample_rate=config.RATE_IN, num_channels=1
            )
            asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), loop)
            self._pushed += 1

            if now - self._reported >= 10:
                logger.info(f"loopback: {self._pushed} frames pushed, peak {db:.1f} dBFS")
                self._pushed = 0
                self._reported = now

    async def wait_for_call_active(self, timeout: float | None = None) -> bool:
        loop = asyncio.get_running_loop()
        if self._call_active.is_set():
            return True
        return await loop.run_in_executor(None, self._call_active.wait, timeout)

    async def cleanup(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self._recorder is not None:
            try:
                self._recorder.__exit__(None, None, None)
            except Exception:
                pass
            self._recorder = None
        await super().cleanup()
