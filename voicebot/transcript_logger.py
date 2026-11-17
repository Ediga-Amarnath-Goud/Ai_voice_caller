"""Log caller and bot speech as human-readable transcription lines."""
from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputTextRawFrame,
    TTSTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TranscriptLogger(FrameProcessor):
    """Emits `caller: ...` for input transcriptions and `agent : ...` for TTS."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                logger.info(f"caller: {frame.text.strip()}")
        elif isinstance(frame, TTSTextFrame):
            pass  # logged once per turn by the assistant aggregator
        elif isinstance(frame, InputTextRawFrame):
            if frame.text and frame.text.strip() == "Hello":
                logger.info("caller: (greeting trigger)")
        await self.push_frame(frame, direction)
