"""
server/call_state.py - Call State Machine & Audio Orchestrator
Coordinates active call sessions, BaseTelephonyDriver, GeminiLiveClient,
SQLite persistence via DatabaseManager, and 25 Hz WebSocket audio telemetry.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from core.audio_processor import AudioProcessor
from core.gemini_live import DEFAULT_SYSTEM_INSTRUCTION, GeminiLiveClient
from server.database import DatabaseManager
from telephony.base_driver import BaseTelephonyDriver

logger = logging.getLogger("server.call_state")


class CallState(str, Enum):
    """
    Call session lifecycle states.
    Supports case-insensitive string comparisons and aliases (ON_HOLD == ON_HOLD_TRANSFER).
    """
    IDLE = "IDLE"
    RINGING = "RINGING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    DECLINED = "DECLINED"
    ON_HOLD_TRANSFER = "ON_HOLD_TRANSFER"
    ON_HOLD = "ON_HOLD"
    DIALING = "DIALING"
    OPERATOR_TALKING = "OPERATOR_TALKING"
    FAILED = "FAILED"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            s = self.value.upper()
            o = other.upper()
            if s == o:
                return True
            if s in ("ON_HOLD", "ON_HOLD_TRANSFER") and o in ("ON_HOLD", "ON_HOLD_TRANSFER"):
                return True
            return False
        if isinstance(other, CallState):
            s = self.value.upper()
            o = other.value.upper()
            if s == o:
                return True
            if s in ("ON_HOLD", "ON_HOLD_TRANSFER") and o in ("ON_HOLD", "ON_HOLD_TRANSFER"):
                return True
            return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        norm = "ON_HOLD" if self.value in ("ON_HOLD", "ON_HOLD_TRANSFER") else self.value.upper()
        return hash(norm)


class CallStateManager:
    """
    Central hub orchestrating the telephony driver, Gemini Live session,
    database persistence, operator whisper, and 25 Hz telemetry.
    """

    def __init__(
        self,
        db: DatabaseManager,
        driver: BaseTelephonyDriver,
        gemini_client: Optional[GeminiLiveClient] = None,
        gemini_client_factory: Optional[Callable[..., GeminiLiveClient]] = None,
        broadcast_func: Optional[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]] = None,
    ) -> None:
        self.db: DatabaseManager = db
        self.driver: BaseTelephonyDriver = driver
        self.broadcast_func: Optional[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]] = broadcast_func
        self._gemini_client_factory: Optional[Callable[..., GeminiLiveClient]] = gemini_client_factory

        # Instantiate or assign Gemini Live Client
        if gemini_client is not None:
            self.gemini_client: GeminiLiveClient = gemini_client
        elif self._gemini_client_factory is not None:
            self.gemini_client = self._gemini_client_factory()
        else:
            self.gemini_client = GeminiLiveClient()

        # Session tracking
        self.state: CallState = CallState.IDLE
        self.current_call_id: Optional[str] = None
        self.start_time: float = 0.0
        self.pending_whispers: List[str] = []
        self.sent_whispers: List[str] = []

        # Audio telemetry (25 Hz / 40ms)
        self._last_telemetry_time: float = 0.0
        self._latest_caller_levels: Dict[str, Any] = {"rms": 0.0, "peak": 0.0, "spectrum": [0.0] * 8}
        self._latest_gemini_levels: Dict[str, Any] = {"rms": 0.0, "peak": 0.0, "spectrum": [0.0] * 8}
        self._telemetry_task: Optional[asyncio.Task] = None

        # Transcripts tracking
        self._transcript_sequence: int = 0
        self._current_turn_id: str = "turn_001"

        # Concurrency & loop capture
        self._state_lock: asyncio.Lock = asyncio.Lock()
        try:
            self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Wire up callbacks
        self._wire_gemini_callbacks()
        self._wire_driver_callbacks()

    def _wire_driver_callbacks(self) -> None:
        """Wires up telephony driver audio and state change callbacks."""
        if self.driver and hasattr(self.driver, "set_audio_callbacks"):
            self.driver.set_audio_callbacks(
                on_audio_in=self.feed_caller_audio,
                on_state_change=self._on_driver_state_change,
            )

    async def _on_driver_state_change(self, new_state: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Handles state transition dispatched from the underlying telephony driver."""
        det = details or {}
        norm_state = str(new_state).upper()
        if norm_state == "RINGING" and self.state == CallState.IDLE:
            caller_id = det.get("caller_id", "Inbound Call")
            caller_name = det.get("caller_name", "Phone Link")
            await self.handle_inbound_ring(caller_id, caller_name)
        elif norm_state == "ACTIVE" and self.state == CallState.RINGING:
            await self.approve_inbound_call("ai_takeover")
        elif norm_state == "COMPLETED" and self.state == CallState.ACTIVE:
            await self.terminate_call(det.get("reason", "remote_hangup"))

    def _wire_gemini_callbacks(self) -> None:
        """Wires up Gemini Live client callbacks and registers tool handlers."""
        if hasattr(self.gemini_client, "set_on_audio_received"):
            self.gemini_client.set_on_audio_received(self._on_assistant_audio)
        if hasattr(self.gemini_client, "set_on_transcript"):
            self.gemini_client.set_on_transcript(self._on_transcript)
        if hasattr(self.gemini_client, "set_on_interrupted"):
            self.gemini_client.set_on_interrupted(self.handle_barge_in)

        # Register custom tool handlers with Gemini Live
        if hasattr(self.gemini_client, "register_tool"):
            self.gemini_client.register_tool(
                "take_message",
                lambda **kwargs: self.handle_tool_call("take_message", kwargs)
            )
            self.gemini_client.register_tool(
                "end_call",
                lambda **kwargs: self.handle_tool_call("end_call", kwargs)
            )
            self.gemini_client.register_tool(
                "transfer_to_human",
                lambda **kwargs: self.handle_tool_call("transfer_to_human", kwargs)
            )
            self.gemini_client.register_tool(
                "log_call_summary",
                lambda **kwargs: self.handle_tool_call("log_call_summary", kwargs)
            )

    # -------------------------------------------------------------------------
    # Broadcast and Telemetry Helpers
    # -------------------------------------------------------------------------

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        """Thread-safe event broadcast to connected WebSocket clients."""
        if not self.broadcast_func:
            return
        try:
            res = self.broadcast_func(event)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            logger.debug(f"Broadcast handler failed: {e}")

    def _schedule_broadcast(self, event: Dict[str, Any]) -> None:
        """Safely schedule broadcast from synchronous or thread-bound contexts."""
        if not self.broadcast_func:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._broadcast(event))
        except RuntimeError:
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(self._broadcast(event), self._loop)

    async def _telemetry_loop(self) -> None:
        """Periodic 25 Hz (40ms) telemetry tick broadcasting live audio levels."""
        try:
            while self.state == CallState.ACTIVE:
                await asyncio.sleep(0.040)  # 25 Hz rate limit
                payload = {
                    "type": "audio_telemetry",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_id": self.current_call_id,
                    "data": {
                        "caller": self._latest_caller_levels,
                        "gemini": self._latest_gemini_levels,
                    }
                }
                await self._broadcast(payload)
                # Exponential decay for visual smoothing when no new audio chunk arrives
                if isinstance(self._latest_caller_levels, dict):
                    self._latest_caller_levels["rms"] = self._latest_caller_levels.get("rms", 0.0) * 0.8
                    self._latest_caller_levels["peak"] = self._latest_caller_levels.get("peak", 0.0) * 0.8
                if isinstance(self._latest_gemini_levels, dict):
                    self._latest_gemini_levels["rms"] = self._latest_gemini_levels.get("rms", 0.0) * 0.8
                    self._latest_gemini_levels["peak"] = self._latest_gemini_levels.get("peak", 0.0) * 0.8
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Telemetry loop terminated: {e}")

    # -------------------------------------------------------------------------
    # Audio Routing & Callbacks
    # -------------------------------------------------------------------------

    def feed_caller_audio(self, pcm_bytes: bytes) -> None:
        """
        Feeds incoming caller audio from the telephony driver to Gemini Live.
        Computes 16kHz audio levels and streams PCM asynchronously.
        """
        if not pcm_bytes:
            return

        # Compute Caller levels (16kHz PCM)
        try:
            self._latest_caller_levels = AudioProcessor.calculate_levels(pcm_bytes, 16000)
        except Exception as e:
            logger.debug(f"Error calculating caller audio levels: {e}")

        # Stream chunk to Gemini Live if session is active
        is_conn = getattr(self.gemini_client, "is_connected", False)
        if self.state == CallState.ACTIVE and is_conn:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.gemini_client.send_audio_chunk(pcm_bytes))
            except RuntimeError:
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.gemini_client.send_audio_chunk(pcm_bytes),
                        self._loop
                    )

    def _on_assistant_audio(self, pcm_24k: bytes) -> None:
        """
        Routes 24kHz assistant audio from Gemini Live to telephony driver.
        Computes 24kHz audio levels.
        """
        if not pcm_24k:
            return

        # Compute Gemini levels (24kHz PCM)
        try:
            self._latest_gemini_levels = AudioProcessor.calculate_levels(pcm_24k, 24000)
        except Exception as e:
            logger.debug(f"Error calculating gemini audio levels: {e}")

        # Enqueue audio to driver playback queue
        if self.driver:
            self.driver.play_audio_chunk(pcm_24k)

    async def _on_transcript(self, speaker: str, text: str, is_final: bool) -> None:
        """Persists streaming and finalized transcripts to database and broadcasts WebSocket event."""
        if not self.current_call_id or not text:
            return

        self._transcript_sequence += 1
        normalized_speaker = "gemini" if speaker in ("agent", "gemini") else speaker

        # Commit to SQLite
        try:
            await self.db.append_transcript(
                call_id=self.current_call_id,
                speaker=normalized_speaker,
                text=text,
                is_final=is_final,
            )
        except Exception as e:
            logger.error(f"Failed to persist transcript: {e}")

        # Broadcast transcript chunk
        await self._broadcast({
            "type": "transcript_chunk",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": self.current_call_id,
            "data": {
                "turn_id": self._current_turn_id,
                "speaker": normalized_speaker,
                "text": text,
                "is_final": is_final,
                "sequence": self._transcript_sequence,
            }
        })

        if is_final:
            self._current_turn_id = f"turn_{self._transcript_sequence + 1:03d}"

    async def handle_barge_in(self) -> None:
        """Instantly flushes telephony audio buffer upon speech barge-in interruption."""
        logger.info("Handling barge-in speech interruption.")
        if self.driver and hasattr(self.driver, "clear_audio_buffer"):
            self.driver.clear_audio_buffer()
        self._latest_gemini_levels = {"rms": 0.0, "peak": 0.0, "spectrum": [0.0] * 8}

    # -------------------------------------------------------------------------
    # Call Lifecycle & Inbound Approval Workflow
    # -------------------------------------------------------------------------

    async def handle_inbound_ring(self, caller_id: str, caller_name: str = "") -> str:
        """
        Handles inbound phone call arrival.
        Transitions state: IDLE -> RINGING.
        Returns the new call_id string.
        """
        async with self._state_lock:
            if self.state not in (CallState.IDLE, CallState.COMPLETED, CallState.FAILED, CallState.DECLINED):
                raise RuntimeError(f"Cannot accept inbound call in state: {self.state}")

            self.current_call_id = f"call_{int(time.time() * 1000)}"
            self.start_time = time.time()
            self.state = CallState.RINGING
            self._transcript_sequence = 0
            self._current_turn_id = "turn_001"

            if hasattr(self.driver, "play_ring_chime"):
                self.driver.play_ring_chime()

            driver_name = getattr(self.driver, "driver_name", "virtual")

            # Insert call record in SQLite
            await self.db.create_call({
                "id": self.current_call_id,
                "direction": "inbound",
                "caller_id": caller_id,
                "caller_name": caller_name,
                "status": "ringing",
                "approval_action": "pending",
                "driver_type": driver_name,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })

            # Broadcast state changes and approval prompt
            await self._broadcast({
                "type": "call_state_change",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"state": "ringing", "driver": driver_name}
            })
            await self._broadcast({
                "type": "inbound_call_prompt",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {
                    "caller_id": caller_id,
                    "caller_name": caller_name,
                    "actions": ["ai_takeover", "answer_personally", "decline"]
                }
            })

            return self.current_call_id

    async def approve_inbound_call(self, action: str) -> Dict[str, Any]:
        """
        Approves an inbound ringing call with one of:
        - 'ai_takeover': Connects Gemini Live and answers call.
        - 'answer_personally': Answers call on operator headset.
        - 'decline': Drops call and purges buffered whispers.
        """
        async with self._state_lock:
            if self.state not in (CallState.RINGING, CallState.IDLE, CallState.COMPLETED, CallState.DECLINED):
                raise RuntimeError(f"Cannot approve call in state: {self.state}")

            if hasattr(self.driver, "stop_ring_chime"):
                self.driver.stop_ring_chime()

            # If call was not already ringing (e.g. user manually took over live call from IDLE)
            if not self.current_call_id or self.state != CallState.RINGING:
                self.current_call_id = f"call_{int(time.time() * 1000)}"
                self.start_time = time.time()
                self._transcript_sequence = 0
                self._current_turn_id = "turn_001"
                driver_name = getattr(self.driver, "driver_name", "windows")
                await self.db.create_call({
                    "id": self.current_call_id,
                    "direction": "inbound",
                    "caller_id": "Phone Call",
                    "caller_name": "Inbound Caller",
                    "status": "ringing",
                    "approval_action": "pending",
                    "driver_type": driver_name,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                })

            if action == "ai_takeover":
                await self.driver.answer_call()
                self.state = CallState.ACTIVE
                await self.db.update_call_status(
                    self.current_call_id,
                    "active",
                    approval_action="ai_takeover",
                    answered_at=datetime.now(timezone.utc).isoformat()
                )

                # Connect Gemini Live
                is_conn = getattr(self.gemini_client, "is_connected", False)
                if not is_conn:
                    await self.gemini_client.connect()

                # Register audio callback from driver to feed caller audio
                self.driver.set_audio_callbacks(self.feed_caller_audio)

                # Deliver any pre-buffered operator whispers
                while self.pending_whispers:
                    w = self.pending_whispers.pop(0)
                    if hasattr(self.gemini_client, "send_whisper"):
                        await self.gemini_client.send_whisper(w)
                    self.sent_whispers.append(w)

                # Start 25Hz telemetry broadcast task
                if self._telemetry_task is None or self._telemetry_task.done():
                    self._telemetry_task = asyncio.create_task(self._telemetry_loop())

                await self._broadcast({
                    "type": "call_state_change",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_id": self.current_call_id,
                    "data": {"state": "active", "action": "ai_takeover"}
                })
                return {"status": "approved", "action": "ai_takeover", "state": "active"}

            elif action == "answer_personally":
                await self.driver.answer_call()
                self.state = CallState.OPERATOR_TALKING
                await self.db.update_call_status(
                    self.current_call_id,
                    "operator_talking",
                    approval_action="answered_personally",
                    answered_at=datetime.now(timezone.utc).isoformat()
                )
                await self._broadcast({
                    "type": "call_state_change",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_id": self.current_call_id,
                    "data": {"state": "operator_talking", "action": "answer_personally"}
                })
                return {"status": "approved", "action": "answer_personally", "state": "operator_talking"}

            elif action == "decline":
                await self.driver.hang_up(reason="declined")
                self.state = CallState.COMPLETED
                self.pending_whispers.clear()  # Purge buffered whispers immediately
                await self.db.update_call_status(
                    self.current_call_id,
                    "completed",
                    hangup_reason="declined",
                    approval_action="declined",
                    ended_at=datetime.now(timezone.utc).isoformat()
                )
                await self._broadcast({
                    "type": "call_state_change",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_id": self.current_call_id,
                    "data": {"state": "completed", "action": "decline"}
                })
                return {"status": "approved", "action": "decline", "state": "completed"}

            else:
                raise ValueError(f"Invalid approval action: '{action}'")

    async def initiate_outbound(self, phone_number: str, goal_prompt: str = "") -> bool:
        """
        Initiates an outbound phone call with an optional customized goal prompt.
        Transitions state: IDLE -> DIALING -> ACTIVE.
        Returns True on successful dial and connection, False otherwise.
        """
        async with self._state_lock:
            # Collision guard: do not disrupt an inbound ringing call
            if self.state == CallState.RINGING:
                logger.warning("Cannot initiate outbound call while an inbound call is ringing.")
                return False

            if self.state not in (CallState.IDLE, CallState.COMPLETED, CallState.FAILED, CallState.DECLINED):
                logger.warning(f"Cannot initiate outbound call in active state: {self.state}")
                return False

            self.current_call_id = f"call_{int(time.time() * 1000)}"
            self.start_time = time.time()
            self.state = CallState.DIALING
            self._transcript_sequence = 0
            self._current_turn_id = "turn_001"

            driver_name = getattr(self.driver, "driver_name", "virtual")

            await self.db.create_call({
                "id": self.current_call_id,
                "direction": "outbound",
                "caller_id": phone_number,
                "status": "dialing",
                "goal_prompt": goal_prompt or "",
                "driver_type": driver_name,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })

            await self._broadcast({
                "type": "call_state_change",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"state": "dialing", "phone_number": phone_number}
            })

            # Execute dial on driver
            dial_success = await self.driver.dial(phone_number)
            if not dial_success:
                self.state = CallState.FAILED
                await self.db.update_call_status(
                    self.current_call_id,
                    "failed",
                    hangup_reason="dial_failed",
                    ended_at=datetime.now(timezone.utc).isoformat()
                )
                await self._broadcast({
                    "type": "call_state_change",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_id": self.current_call_id,
                    "data": {"state": "failed", "reason": "dial_failed"}
                })
                return False

            # Configure custom goal prompt in Gemini Live if provided
            if goal_prompt and hasattr(self.gemini_client, "system_instruction"):
                self.gemini_client.system_instruction = (
                    f"{DEFAULT_SYSTEM_INSTRUCTION} OUTBOUND GOAL: {goal_prompt}"
                )

            # Connect Gemini Live
            is_conn = getattr(self.gemini_client, "is_connected", False)
            if not is_conn:
                await self.gemini_client.connect()

            # Set audio callback to route caller audio to Gemini
            self.driver.set_audio_callbacks(self.feed_caller_audio)

            self.state = CallState.ACTIVE
            await self.db.update_call_status(
                self.current_call_id,
                "active",
                answered_at=datetime.now(timezone.utc).isoformat()
            )

            # Start 25Hz telemetry broadcast task
            if self._telemetry_task is None or self._telemetry_task.done():
                self._telemetry_task = asyncio.create_task(self._telemetry_loop())

            await self._broadcast({
                "type": "call_state_change",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"state": "active", "phone_number": phone_number}
            })
            return True

    async def inject_whisper(self, instruction: str) -> Dict[str, Any]:
        """
        Injects a silent operator instruction into the active Gemini Live session.
        If call is ringing, buffers the instruction until approval.
        """
        if not instruction or not str(instruction).strip():
            return {"status": "ignored", "reason": "empty_instruction"}

        clean_instruction = str(instruction).strip()

        if self.state == CallState.RINGING:
            self.pending_whispers.append(clean_instruction)
            await self._broadcast({
                "type": "operator_whisper_ack",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"instruction": clean_instruction, "status": "buffered"}
            })
            return {"status": "buffered", "instruction": clean_instruction}

        elif self.state in (CallState.ACTIVE, CallState.ON_HOLD, CallState.ON_HOLD_TRANSFER):
            if self.gemini_client and hasattr(self.gemini_client, "send_whisper"):
                await self.gemini_client.send_whisper(clean_instruction)
                self.sent_whispers.append(clean_instruction)
                await self._broadcast({
                    "type": "operator_whisper_ack",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "call_id": self.current_call_id,
                    "data": {"instruction": clean_instruction, "status": "sent"}
                })
                return {"status": "sent", "instruction": clean_instruction}

        return {"status": "ignored", "reason": f"invalid_state_{self.state}"}

    async def terminate_call(self, reason: str = "operator_hangup") -> Dict[str, Any]:
        """
        Terminates the call cleanly, disconnects Gemini Live, stops audio streams,
        and records final metrics in SQLite. Idempotent.
        """
        async with self._state_lock:
            if self.state in (CallState.COMPLETED, CallState.IDLE) and not self.current_call_id:
                return {"status": "already_terminated", "state": str(self.state)}

            duration = round(time.time() - self.start_time, 2) if self.start_time > 0 else 0.0

            # Cancel telemetry loop
            if self._telemetry_task and not self._telemetry_task.done():
                self._telemetry_task.cancel()
                self._telemetry_task = None

            # Disconnect Gemini Live (guarantee telephony hangup even if disconnect raises)
            try:
                if self.gemini_client and hasattr(self.gemini_client, "disconnect"):
                    await self.gemini_client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting Gemini Live: {e}")

            # Hang up telephony driver
            try:
                if self.driver:
                    await self.driver.hang_up(reason=reason)
            except Exception as e:
                logger.warning(f"Error hanging up driver: {e}")

            # Persist completion to SQLite
            if self.current_call_id:
                try:
                    await self.db.update_call_status(
                        self.current_call_id,
                        "completed",
                        hangup_reason=reason,
                        end_reason=reason,
                        duration=duration,
                        duration_seconds=int(duration),
                        ended_at=datetime.now(timezone.utc).isoformat(),
                    )
                except Exception as e:
                    logger.error(f"Error updating call status in DB: {e}")

            last_call_id = self.current_call_id
            self.state = CallState.COMPLETED
            self.pending_whispers.clear()
            self._latest_caller_levels = {"rms": 0.0, "peak": 0.0, "spectrum": [0.0] * 8}
            self._latest_gemini_levels = {"rms": 0.0, "peak": 0.0, "spectrum": [0.0] * 8}

            await self._broadcast({
                "type": "call_state_change",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": last_call_id,
                "data": {"state": "completed", "reason": reason, "duration": duration}
            })

            return {"status": "terminated", "call_id": last_call_id, "duration": duration, "reason": reason}

    # -------------------------------------------------------------------------
    # Hold & Escalation Transitions
    # -------------------------------------------------------------------------

    async def hold_call(self) -> None:
        """Puts active call on hold."""
        self.state = CallState.ON_HOLD
        if hasattr(self.driver, "play_hold_audio"):
            self.driver.play_hold_audio()
        if hasattr(self.driver, "clear_audio_buffer"):
            self.driver.clear_audio_buffer()
        if self.current_call_id:
            await self.db.update_call_status(self.current_call_id, "on_hold")
        await self._broadcast({
            "type": "call_state_change",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": self.current_call_id,
            "data": {"state": "on_hold"}
        })

    async def resume_call(self) -> None:
        """Resumes call from hold."""
        self.state = CallState.ACTIVE
        if hasattr(self.driver, "stop_hold_audio"):
            self.driver.stop_hold_audio()
        if self.current_call_id:
            await self.db.update_call_status(self.current_call_id, "active")
        await self._broadcast({
            "type": "call_state_change",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": self.current_call_id,
            "data": {"state": "active"}
        })

    async def operator_pickup(self) -> None:
        """Operator picks up call from hold or AI takeover."""
        self.state = CallState.OPERATOR_TALKING
        if hasattr(self.driver, "stop_hold_audio"):
            self.driver.stop_hold_audio()
        if self.current_call_id:
            await self.db.update_call_status(self.current_call_id, "operator_talking")
        await self._broadcast({
            "type": "call_state_change",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": self.current_call_id,
            "data": {"state": "operator_talking"}
        })

    async def handle_remote_hangup(self) -> None:
        """Handles remote disconnection by caller."""
        await self.terminate_call("remote_hung_up")

    async def handle_driver_error(self, exc: Exception) -> None:
        """Handles driver hardware exception during audio streaming."""
        logger.error(f"Telephony hardware error: {exc}")
        await self.terminate_call("hardware_error")
        self.state = CallState.FAILED

    async def handle_socket_disconnect(self, code: int, reason: str) -> None:
        """Handles remote Gemini WebSocket drop."""
        logger.warning(f"Gemini WebSocket disconnected code={code}, reason={reason}")
        await self.terminate_call("gemini_socket_error")

    # -------------------------------------------------------------------------
    # Agent Function Calling Handlers & DB Persistence
    # -------------------------------------------------------------------------

    async def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes tools triggered by Gemini Live and persists records to SQLite.
        """
        clean_args = args if isinstance(args, dict) else {}

        if name == "take_message":
            caller_name = str(clean_args.get("caller_name") or "Unknown")
            phone_number = str(clean_args.get("phone_number") or "")
            message = str(clean_args.get("message") or "")
            urgency = str(clean_args.get("urgency") or "medium")
            if urgency not in ("low", "normal", "medium", "high", "critical"):
                urgency = "medium"

            msg_id = await self.db.save_message(
                call_id=self.current_call_id,
                caller_name=caller_name,
                phone_number=phone_number,
                message=message,
                urgency=urgency,
            )

            result = {
                "status": "saved",
                "message_id": msg_id,
                "caller_name": caller_name,
                "info": "Message saved successfully."
            }
            await self._broadcast({
                "type": "tool_execution",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"tool": "take_message", "result": result}
            })
            return result

        elif name == "end_call":
            reason = str(clean_args.get("reason") or "normal")
            await self.terminate_call(reason)
            result = {"status": "call_ended", "reason": reason}
            await self._broadcast({
                "type": "tool_execution",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"tool": "end_call", "result": result}
            })
            return result

        elif name == "transfer_to_human":
            reason = str(clean_args.get("reason") or "escalation")
            self.state = CallState.ON_HOLD
            if hasattr(self.driver, "play_hold_audio"):
                self.driver.play_hold_audio()
            if hasattr(self.driver, "clear_audio_buffer"):
                self.driver.clear_audio_buffer()
            if self.current_call_id:
                await self.db.update_call_status(self.current_call_id, "on_hold")
            result = {"status": "transfer_initiated", "reason": reason, "on_hold": True}
            await self._broadcast({
                "type": "tool_execution",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"tool": "transfer_to_human", "result": result}
            })
            await self._broadcast({
                "type": "call_state_change",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"state": "on_hold", "reason": reason}
            })
            return result

        elif name == "log_call_summary":
            summary = str(clean_args.get("summary") or "")
            action_items = clean_args.get("action_items")
            if isinstance(action_items, str):
                items = [action_items]
            elif isinstance(action_items, list):
                items = [str(i) for i in action_items]
            else:
                items = []

            if self.current_call_id:
                await self.db.save_summary(self.current_call_id, summary, items)

            result = {"status": "summary_logged", "summary": summary, "action_items": items}
            await self._broadcast({
                "type": "tool_execution",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"tool": "log_call_summary", "result": result}
            })
            await self._broadcast({
                "type": "call_summary_generated",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": self.current_call_id,
                "data": {"summary": summary, "action_items": items}
            })
            return result

        return {"status": "error", "error": f"Unknown tool: '{name}'"}

    async def safe_handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Defensive wrapper catching any tool execution exception."""
        try:
            return await self.handle_tool_call(name, args)
        except Exception as e:
            logger.error(f"Exception handling tool {name}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------------
    # Status Telemetry
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Returns operational status of the call manager and active call."""
        driver_name = getattr(self.driver, "driver_name", "none") if self.driver else "none"
        driver_status = self.driver.get_status() if self.driver and hasattr(self.driver, "get_status") else {}
        gemini_conn = getattr(self.gemini_client, "is_connected", False) if self.gemini_client else False

        return {
            "status": "ok",
            "state": self.state.value.lower() if hasattr(self.state, "value") else str(self.state).lower(),
            "raw_state": str(self.state),
            "current_call_id": self.current_call_id,
            "driver": driver_name,
            "driver_status": driver_status,
            "gemini_connected": gemini_conn,
            "pending_whispers_count": len(self.pending_whispers),
            "start_time": self.start_time,
        }
