"""
Base Telephony Driver Abstract Interface for AI Voice Calling Assistant.
Defines the standard contract for telephony bridges, audio hardware routing,
call lifecycle management, and barge-in buffer handling.
"""

from abc import ABC, abstractmethod
import asyncio
import inspect
import logging
from typing import Callable, Optional, Dict, Any, Awaitable, Union

logger = logging.getLogger("telephony.base")


class BaseTelephonyDriver(ABC):
    """
    Abstract Base Class for all telephony and audio bridge drivers.
    Manages audio streaming endpoints, call state machines, and hardware routing.
    """

    def __init__(self) -> None:
        self._on_audio_in: Optional[Callable[[bytes], Union[None, Awaitable[None]]]] = None
        self._on_state_change: Optional[Callable[[str, Dict[str, Any]], Union[None, Awaitable[None]]]] = None
        self._call_state: str = "IDLE"
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def driver_name(self) -> str:
        """Human-readable driver identifier."""
        return self.__class__.__name__

    @property
    def call_state(self) -> str:
        """Current telephony call state (e.g. IDLE, DIALING, RINGING, ACTIVE, COMPLETED)."""
        return self._call_state

    @abstractmethod
    async def start(self) -> None:
        """
        Start the telephony driver, opening audio streams and initializing subsystems.
        Must be idempotent.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the telephony driver, closing active streams and freeing hardware resources.
        Must be idempotent.
        """
        pass

    @abstractmethod
    async def dial(self, phone_number: str) -> bool:
        """
        Initiate an outbound call to the target phone number.
        Returns True if dial command was accepted/initiated, False otherwise.
        """
        pass

    @abstractmethod
    async def answer_call(self) -> bool:
        """
        Answer an incoming ringing call.
        Returns True if call was answered successfully, False otherwise.
        """
        pass

    @abstractmethod
    async def hang_up(self, reason: str = "normal") -> bool:
        """
        Terminate the active or ringing call.
        Returns True if call termination succeeded, False otherwise.
        """
        pass

    def set_audio_callbacks(
        self,
        on_audio_in: Callable[[bytes], Union[None, Awaitable[None]]],
        on_state_change: Optional[Callable[[str, Dict[str, Any]], Union[None, Awaitable[None]]]] = None,
    ) -> None:
        """
        Register callbacks for incoming raw PCM audio chunks and call state transitions.
        Callbacks may be standard synchronous functions or asynchronous coroutine functions.
        """
        self._on_audio_in = on_audio_in
        self._on_state_change = on_state_change

    @abstractmethod
    def play_audio_chunk(self, pcm_data: bytes) -> None:
        """
        Enqueue an outgoing PCM audio chunk for transmission or speaker playback.
        Thread-safe and non-blocking.
        """
        pass

    @abstractmethod
    def clear_audio_buffer(self) -> None:
        """
        Immediately clear and flush all queued outgoing audio buffers.
        Crucial for instant barge-in speech interruption.
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """
        Return True if driver is actively running, streams are open, or in a call session.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Return operational telemetry: driver name, active devices, sample rates,
        buffer state, queue depth, and current call state.
        """
        pass

    def _dispatch_callback(self, callback: Optional[Callable], *args: Any, **kwargs: Any) -> None:
        """
        Internal helper to safely dispatch callbacks from background threads or async tasks.
        Handles both sync callables and async coroutines across thread boundaries.
        """
        if callback is None:
            return

        try:
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                loop = self._loop
                if loop is None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                if loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(result, loop)
                else:
                    # Close coroutine to avoid un-awaited coroutine warning if no loop
                    try:
                        asyncio.run(result)
                    except Exception as loop_err:
                        logger.debug(f"Unable to execute async callback with asyncio.run: {loop_err}")
                        result.close()
        except Exception as e:
            logger.error(f"Error executing telephony driver callback {callback}: {e}", exc_info=True)

    def _set_state(self, new_state: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Update internal call state and notify registered state change listener."""
        self._call_state = new_state
        payload = details.copy() if details else {}
        payload["state"] = new_state
        payload["driver"] = self.driver_name
        self._dispatch_callback(self._on_state_change, new_state, payload)
