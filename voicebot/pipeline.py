"""Gemini Live speech-to-speech pipeline wired to the loopback + virtual cable."""
from __future__ import annotations

import contextlib

import pyaudio
from loguru import logger
from pipecat.frames.frames import InputTextRawFrame, LLMContextFrame, LLMRunFrame
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    UserTurnMessageAddedMessage,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.pipeline.worker import (
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import (
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from . import config, audio_devices
from .loopback import LoopbackInputTransport
from .transcript_logger import TranscriptLogger


class VoicePipeline:
    """Owns the worker, runner, and input transport for one process lifetime."""

    def __init__(self, output_device_index: int, serial: str | None = None) -> None:
        self._pa = pyaudio.PyAudio()
        dev = audio_devices.device_info(output_device_index)
        dev_name = dev["name"] if dev else str(output_device_index)
        logger.info(f"audio out: [{output_device_index}] {dev_name} @ {config.RATE_OUT} Hz")
        self.input = LoopbackInputTransport(TransportParams(audio_in_enabled=True))
        self._output = LocalAudioOutputTransport(
            self._pa,
            LocalAudioTransportParams(
                audio_out_enabled=True,
                output_device_index=output_device_index,
            ),
        )
        self.llm = GeminiLiveLLMService(
            api_key=config.api_key(),
            system_instruction=config.SYSTEM_PROMPT,
            settings=GeminiLiveLLMService.Settings(model=config.MODEL, voice=config.VOICE),
            inference_on_context_initialization=False,
        )
        self.context = LLMContext([])
        user_agg, assistant_agg = LLMContextAggregatorPair(self.context)
        transcript = TranscriptLogger()

        @user_agg.event_handler("on_user_turn_message_added")
        async def _on_user(agg, message: UserTurnMessageAddedMessage):
            if message.content and message.content.strip():
                logger.info(f"caller: {message.content.strip()}")

        @assistant_agg.event_handler("on_assistant_turn_stopped")
        async def _on_assistant(agg, message: AssistantTurnStoppedMessage):
            if message.content and message.content.strip():
                logger.info(f"agent : {message.content.strip()}")

        # Aggregators are required: they turn the greeting InputTextRawFrame into
        # a user turn (triggering inference) and surface TTS text for logging.
        pipeline = Pipeline(
            [self.input, user_agg, self.llm, transcript, self._output, assistant_agg]
        )
        self.worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=config.RATE_IN,
                audio_out_sample_rate=config.RATE_OUT,
                enable_metrics=False,
            ),
            idle_timeout_secs=None,
            processor_unusable_policy=ProcessorUnusablePolicy.END,
        )
        self._started = __import__("asyncio").Event()

        @self.worker.event_handler("on_pipeline_started")
        async def _on_started(worker, frame):
            self._started.set()

        self.runner = WorkerRunner(handle_sigint=True)

    async def prepare(self) -> None:
        # Only register the worker. on_pipeline_started fires from runner.run(),
        # which the caller starts separately; wait_ready() then blocks on it.
        await self.runner.add_workers(self.worker)

    async def wait_ready(self, timeout: float = 25.0) -> None:
        import asyncio
        # 1. Wait for pipeline start frame
        await asyncio.wait_for(self._started.wait(), timeout=timeout)

        # 2. Wait for Gemini Live WebSocket session to connect
        deadline = asyncio.get_running_loop().time() + timeout
        while self.llm._session is None:
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError("Timed out waiting for Gemini Live WebSocket connection")
            await asyncio.sleep(0.05)

        # 3. Push context frame so Gemini Live enters realtime mode
        if not self.llm._ready_for_realtime_input:
            await self.worker.queue_frames([LLMContextFrame(self.context)])
            while not self.llm._ready_for_realtime_input:
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError("Timed out waiting for Gemini Live realtime readiness")
                await asyncio.sleep(0.05)

        logger.info("pipeline ready (Gemini Live 3.8 connected & accepting realtime audio)")

    async def greet(self) -> None:
        logger.info("greeting caller")
        await self.worker.queue_frames([InputTextRawFrame("Hello")])

    async def run(self):
        return await self.runner.run()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._pa.terminate()
