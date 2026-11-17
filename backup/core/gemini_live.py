"""
Gemini Live 3.8 Real-Time Client Core.
Manages bidirectional WebSocket sessions with Gemini Live (model 'gemini-3.8-live'),
Aoede voice persona, streaming 16kHz audio input, 24kHz audio output extraction,
barge-in interruption detection, operator whisper injection, and tool call dispatching.
"""

import os
import inspect
import asyncio
import logging
from typing import Optional, Callable, Awaitable, List, Dict, Any, Union
from google import genai
from google.genai import types

from core.tools import ToolRegistry, get_tool_declarations

logger = logging.getLogger("core.gemini_live")

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an AI call screening assistant answering phone calls on behalf of the user. "
    "You speak with a warm, soft-spoken, and professional female voice (Aoede). "
    "Your role is to politely greet the caller, inquire about the purpose of their call, "
    "take messages if the user is unavailable using the take_message tool, "
    "transfer important calls to a human using transfer_to_human, or gracefully end the call using end_call. "
    "Be concise, courteous, and natural as in a real phone conversation."
)


class GeminiLiveClient:
    """
    Client for managing real-time bidirectional audio streaming sessions with Gemini Live 3.8.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_name: str = "Aoede",
        system_instruction: Optional[str] = None,
        tools: Optional[List[types.Tool]] = None,
        model: str = "gemini-3.8-live",
        client: Optional[genai.Client] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.voice_name = voice_name
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self.tools = tools if tools is not None else get_tool_declarations()
        self.model = model
        self.client = client
        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()

        # Callbacks
        self._on_audio_received: Optional[Callable[[bytes], Any]] = None
        self._on_transcript: Optional[Callable[[str, str, bool], Any]] = None
        self._on_interrupted: Optional[Callable[[], Any]] = None

        # Session state
        self._session_cm = None
        self._session = None
        self._receive_task: Optional[asyncio.Task] = None
        self._is_connected: bool = False

    @property
    def is_connected(self) -> bool:
        """Returns True if the live session is connected and active."""
        return self._is_connected and self._session is not None

    def set_on_audio_received(self, callback: Callable[[bytes], Any]) -> None:
        """Registers callback for receiving 24kHz 16-bit PCM output audio chunks."""
        self._on_audio_received = callback

    def set_on_transcript(self, callback: Callable[[str, str, bool], Any]) -> None:
        """Registers callback for live transcripts: callback(speaker, text, is_final)."""
        self._on_transcript = callback

    def set_on_interrupted(self, callback: Callable[[], Any]) -> None:
        """Registers callback for barge-in speech interruption detection."""
        self._on_interrupted = callback

    def register_tool(
        self,
        name: str,
        func: Callable,
        schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Registers a callable tool handler into the active tool registry."""
        self.tool_registry.register_handler(name, func)

    def _build_config(self) -> types.LiveConnectConfig:
        """Builds the LiveConnectConfig for Gemini Live 3.8."""
        modalities = [types.Modality.AUDIO] if hasattr(types, "Modality") else ["AUDIO"]

        system_content = None
        if self.system_instruction:
            system_content = types.Content(
                parts=[types.Part.from_text(text=self.system_instruction)]
            )

        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=self.voice_name
                )
            )
        )

        return types.LiveConnectConfig(
            response_modalities=modalities,
            system_instruction=system_content,
            speech_config=speech_config,
            tools=self.tools,
        )

    async def connect(self) -> None:
        """
        Establishes bidirectional live WebSocket connection to Gemini Live 3.8
        and starts the background receive loop.
        """
        if self._is_connected:
            logger.warning("GeminiLiveClient is already connected.")
            return

        # Clean up any residual session context manager or task from a prior dropped connection
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Exception cleaning up previous session context manager: {e}")
            finally:
                self._session_cm = None

        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Exception cancelling prior receive task: {e}")
        self._receive_task = None

        if self.client is None:
            self.client = genai.Client(api_key=self.api_key)

        config = self._build_config()
        logger.info(f"Connecting to Gemini Live model={self.model} with voice={self.voice_name}")

        self._session_cm = self.client.aio.live.connect(model=self.model, config=config)
        self._session = await self._session_cm.__aenter__()
        self._is_connected = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Connected to Gemini Live session successfully.")

    async def disconnect(self) -> None:
        """
        Closes the active live session gracefully, cancels the receive loop,
        and cleans up background tasks.
        """
        if not self._is_connected and self._session is None and self._receive_task is None:
            return

        logger.info("Disconnecting Gemini Live session...")
        self._is_connected = False

        if self._receive_task is not None:
            if not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"Exception during receive task cancellation: {e}")
            self._receive_task = None

        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Exception closing session context manager: {e}")
            finally:
                self._session_cm = None
                self._session = None
        else:
            self._session = None

        logger.info("Gemini Live session disconnected cleanly.")

    async def send_audio_chunk(self, pcm_16k_mono: bytes) -> None:
        """
        Streams a 16kHz 16-bit signed mono PCM chunk into the active Gemini Live session.
        MIME type: 'audio/pcm;rate=16000'.
        """
        if not self.is_connected or not self._session:
            logger.warning("Attempted to send audio chunk while session is not connected.")
            return

        if not pcm_16k_mono:
            return

        blob = types.Blob(data=pcm_16k_mono, mime_type="audio/pcm;rate=16000")
        await self._session.send_realtime_input(media=blob)

    async def send_whisper(self, instruction: str) -> None:
        """
        Injects a silent operator instruction into the active session without interrupting
        ongoing audio streaming. Formatted as [OPERATOR INSTRUCTION: ...].
        """
        if not self.is_connected or not self._session:
            logger.warning("Attempted to send whisper while session is not connected.")
            return

        if not instruction:
            return

        formatted_instruction = f"[OPERATOR INSTRUCTION: {instruction.strip()}]"
        logger.info(f"Injecting operator whisper: {formatted_instruction}")
        await self._session.send_realtime_input(text=formatted_instruction)

    async def _invoke_callback(self, callback: Optional[Callable], *args) -> None:
        """Safely invokes a registered callback, supporting both sync and async callables."""
        if not callback:
            return
        try:
            res = callback(*args)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            logger.error(f"Error executing callback {callback}: {e}", exc_info=True)

    async def _handle_tool_call(self, tool_call: Any) -> None:
        """Handles tool call requests from Gemini Live and sends back structured responses."""
        function_calls = getattr(tool_call, "function_calls", []) or []
        responses = []

        for fc in function_calls:
            fn_name = getattr(fc, "name", "")
            fn_args = getattr(fc, "args", {}) or {}
            call_id = getattr(fc, "id", None)

            logger.info(f"Handling tool call: id={call_id}, name={fn_name}, args={fn_args}")
            result = await self.tool_registry.execute_tool(fn_name, fn_args)

            fr = types.FunctionResponse(
                name=fn_name,
                id=call_id,
                response={"result": result}
            )
            responses.append(fr)

        if responses and self._session:
            logger.info(f"Dispatching {len(responses)} tool responses to Gemini Live")
            await self._session.send_tool_response(function_responses=responses)

    async def _receive_loop(self) -> None:
        """
        Background worker receiving server events from Gemini Live:
        - Interruption (barge-in): invokes on_interrupted
        - Model turn audio / text: unpacks 24kHz PCM and transcript tokens
        - Turn completion: flags transcript finalization
        - Tool call: executes tool handler and responds
        """
        try:
            async for message in self._session.receive():
                if not self._is_connected:
                    break

                # 1. Server Content (speech, transcript, barge-in, turn_complete)
                sc = getattr(message, "server_content", None)
                if sc is not None:
                    # Speech barge-in detected
                    if getattr(sc, "interrupted", False):
                        logger.info("Speech barge-in detected by Gemini Live.")
                        await self._invoke_callback(self._on_interrupted)

                    # Model turn content
                    model_turn = getattr(sc, "model_turn", None)
                    if model_turn is not None:
                        parts = getattr(model_turn, "parts", []) or []
                        for part in parts:
                            # Text transcript
                            text = getattr(part, "text", None)
                            if text:
                                await self._invoke_callback(self._on_transcript, "agent", text, False)

                            # 24kHz PCM Audio Chunk
                            inline_data = getattr(part, "inline_data", None)
                            if inline_data is not None:
                                audio_data = getattr(inline_data, "data", None)
                                if audio_data:
                                    await self._invoke_callback(self._on_audio_received, audio_data)

                    # Turn completed
                    if getattr(sc, "turn_complete", False):
                        await self._invoke_callback(self._on_transcript, "agent", "", True)

                # 2. Tool Calls
                tool_call = getattr(message, "tool_call", None)
                if tool_call is not None:
                    await self._handle_tool_call(tool_call)

        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled.")
        except Exception as e:
            if self._is_connected:
                logger.error(f"Error in Gemini Live receive loop: {e}", exc_info=True)
        finally:
            self._is_connected = False
            self._session = None
