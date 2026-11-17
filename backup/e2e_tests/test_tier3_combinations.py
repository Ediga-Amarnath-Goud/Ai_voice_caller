"""
e2e_tests/test_tier3_combinations.py

Tier 3: Pairwise Cross-Feature Interactions Testing (>=15 tests).
Tests interactions between concurrent subsystems:
- Speech barge-in during active tool call execution
- Operator whisper injection during incoming ringing state
- Mid-call whisper injection while AI is actively speaking
- Tool take_message invoked immediately prior to tool end_call
- Outbound dialer triggering active state followed by post-call summary logging
- Driver audio error during active Gemini Live WebSocket session
- Inbound call approval ("AI Take Over") concurrently with operator whisper pre-load
- Rapid state transitions (IDLE -> RINGING -> ACTIVE -> ON_HOLD -> ACTIVE -> COMPLETED)
- Regression test: whisper state leakage prevention on decline
- Regression test: telephony hangup guarantee on disconnect exception
- Plus 10 additional pairwise interaction tests.

Total: 20 tests.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch
import pytest

from core.audio_processor import AudioProcessor

# Dynamic imports with conftest fallback
try:
    from telephony.base_driver import BaseTelephonyDriver
except (ImportError, ModuleNotFoundError):
    from e2e_tests.conftest import BaseTelephonyDriver

try:
    from telephony.windows_driver import WindowsAudioDriver
except (ImportError, ModuleNotFoundError):
    from e2e_tests.conftest import WindowsAudioDriver

try:
    from telephony.virtual_driver import VirtualSimulatorDriver
except (ImportError, ModuleNotFoundError):
    from e2e_tests.conftest import VirtualSimulatorDriver

try:
    from telephony.linux_driver import LinuxDriver
except (ImportError, ModuleNotFoundError):
    from e2e_tests.conftest import LinuxDriver

try:
    from server.database import DatabaseManager
except (ImportError, ModuleNotFoundError):
    from e2e_tests.conftest import DatabaseManager

try:
    from server.call_state import CallStateManager
except (ImportError, ModuleNotFoundError):
    from e2e_tests.conftest import CallStateManager

try:
    from server.app import create_app
except (ImportError, ModuleNotFoundError):
    from e2e_tests.conftest import create_app

from e2e_tests.conftest import MockTelephonyDriver


@pytest.mark.asyncio
async def test_comb_barge_in_during_active_tool_call(call_manager):
    """Test 1: Caller interrupts with speech while a tool call is in-flight."""
    # 1. Start active call
    await call_manager.handle_inbound_ring("+15551112222", "Alice")
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"

    # Queue some audio to simulate assistant speaking
    call_manager.driver.play_audio_chunk(b"\x00\x10" * 480)
    assert call_manager.driver.get_buffer_size() > 0

    # 2. Trigger slow tool call execution concurrently with barge-in
    tool_executed = asyncio.Event()

    async def slow_take_message():
        await asyncio.sleep(0.04)
        res = await call_manager.handle_tool_call(
            "take_message",
            {"caller_name": "Alice", "phone_number": "+15551112222", "message": "Quick note", "urgency": "medium"}
        )
        tool_executed.set()
        return res

    tool_task = asyncio.create_task(slow_take_message())
    await asyncio.sleep(0.01)  # Mid-flight

    # Caller interrupts
    await call_manager.handle_barge_in()

    # Buffer must be cleared immediately (<50ms)
    assert call_manager.driver.get_buffer_size() == 0
    await tool_task
    assert tool_executed.is_set()

    # Verify message was saved despite interruption
    msgs = await call_manager.db.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["caller_name"] == "Alice"
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_whisper_injection_during_ringing_state(call_manager):
    """Test 2: Whisper guidance injected while phone is ringing before approval."""
    await call_manager.handle_inbound_ring("+15552223333", "Bob VIP")
    assert call_manager.state == "RINGING"

    # Operator whispers before answering
    await call_manager.inject_whisper("Caller is VIP, offer immediate appointment")
    assert "Caller is VIP, offer immediate appointment" in call_manager.pending_whispers

    # Approve call
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"
    # Verify whisper was flushed into session
    assert len(call_manager.pending_whispers) == 0
    assert len(call_manager.sent_whispers) == 1
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_midcall_whisper_while_ai_speaking(call_manager):
    """Test 3: Whisper injected while assistant is actively outputting audio."""
    await call_manager.handle_inbound_ring("+15553334444", "Charlie")
    await call_manager.approve_inbound_call("ai_takeover")

    # AI streams 24kHz audio chunk into driver buffer
    call_manager.driver.play_audio_chunk(b"\x00\x20" * 480)
    buf_size_before = call_manager.driver.get_buffer_size()
    assert buf_size_before > 0

    # Operator whispers
    await call_manager.inject_whisper("Keep response brief")

    # Whisper must NOT clear playback buffer (unlike barge-in)
    assert call_manager.driver.get_buffer_size() == buf_size_before
    assert len(call_manager.sent_whispers) == 1
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_take_message_immediately_prior_to_end_call(call_manager):
    """Test 4: take_message invoked immediately prior to end_call."""
    await call_manager.handle_inbound_ring("+15554445555", "Dave")
    await call_manager.approve_inbound_call("ai_takeover")
    call_id = call_manager.current_call_id

    # Rapid sequential execution of both tools
    await call_manager.handle_tool_call(
        "take_message",
        {"caller_name": "Dave", "phone_number": "+15554445555", "message": "Call back tomorrow", "urgency": "low"}
    )
    await call_manager.handle_tool_call(
        "end_call",
        {"reason": "message_recorded"}
    )

    assert call_manager.state == "COMPLETED"

    # Check DB
    msgs = await call_manager.db.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["message"] == "Call back tomorrow"
    call_details = await call_manager.db.get_call_details(call_id)
    assert call_details["status"] == "completed"


@pytest.mark.asyncio
async def test_comb_outbound_dialer_active_state_then_summary_logging(call_manager):
    """Test 5: Outbound dial initiates, transitions to ACTIVE, logs summary on completion."""
    success = await call_manager.initiate_outbound("+15556667777", "Book dental checkup for Friday")
    assert success is True
    assert call_manager.state == "ACTIVE"
    call_id = call_manager.current_call_id

    # Simulate conversation turn & summary tool
    await call_manager.handle_tool_call(
        "log_call_summary",
        {"summary": "Checkup booked for Friday 10 AM", "action_items": ["Add to calendar"]}
    )
    await call_manager.terminate_call("objective_completed")

    assert call_manager.state == "COMPLETED"
    call_details = await call_manager.db.get_call_details(call_id)
    assert call_details["direction"] == "outbound"
    assert "Friday 10 AM" in call_details["summary"]["summary"]


@pytest.mark.asyncio
async def test_comb_driver_audio_error_during_gemini_session(call_manager):
    """Test 6: Driver raises audio error during active session."""
    await call_manager.handle_inbound_ring("+15557778888", "Frank")
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"

    # Simulate hardware exception in audio callback
    await call_manager.handle_driver_error(RuntimeError("Audio device unplugged"))

    # State must recover cleanly to COMPLETED or FAILED without hanging
    assert call_manager.state in ["COMPLETED", "FAILED"]


@pytest.mark.asyncio
async def test_comb_inbound_approval_concurrent_with_whisper_preload(call_manager):
    """Test 7: Race between approval and whisper pre-load via asyncio.gather."""
    await call_manager.handle_inbound_ring("+15558889999", "Grace")

    await asyncio.gather(
        call_manager.approve_inbound_call("ai_takeover"),
        call_manager.inject_whisper("Offer 10% discount")
    )
    assert call_manager.state == "ACTIVE"
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_rapid_state_transitions_full_cycle(call_manager):
    """Test 8: Rapid IDLE -> RINGING -> ACTIVE -> ON_HOLD -> ACTIVE -> COMPLETED transitions."""
    assert call_manager.state == "IDLE"
    await call_manager.handle_inbound_ring("+15550001111")
    assert call_manager.state == "RINGING"
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"
    await call_manager.hold_call()
    assert call_manager.state == "ON_HOLD"
    await call_manager.resume_call()
    assert call_manager.state == "ACTIVE"
    await call_manager.terminate_call("normal_hangup")
    assert call_manager.state == "COMPLETED"


@pytest.mark.asyncio
async def test_comb_barge_in_immediately_after_whisper(call_manager):
    """Test 9: Barge-in occurs right after whisper modified turn begins."""
    await call_manager.handle_inbound_ring("+15551239999")
    await call_manager.approve_inbound_call("ai_takeover")

    await call_manager.inject_whisper("Speak softly")
    call_manager.driver.play_audio_chunk(b"\x00\x10" * 480)
    assert call_manager.driver.get_buffer_size() > 0

    # Caller interrupts immediately
    await call_manager.handle_barge_in()
    assert call_manager.driver.get_buffer_size() == 0
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_inbound_decline_during_ring_audio(call_manager):
    """Test 10: Inbound decline stops ringing audio and skips Gemini session."""
    await call_manager.handle_inbound_ring("+15559998888")
    assert call_manager.driver.is_ringing is True

    await call_manager.approve_inbound_call("decline")
    assert call_manager.state == "COMPLETED"
    assert call_manager.driver.is_ringing is False
    assert not call_manager.gemini_client.is_connected


@pytest.mark.asyncio
async def test_comb_transfer_to_human_then_caller_hangup_on_hold(call_manager):
    """Test 11: Caller disconnects while on hold waiting for human."""
    await call_manager.handle_inbound_ring("+15553331111")
    await call_manager.approve_inbound_call("ai_takeover")

    await call_manager.handle_tool_call("transfer_to_human", {"reason": "Complex inquiry"})
    assert call_manager.state == "ON_HOLD"
    assert call_manager.driver.is_on_hold is True

    # Remote disconnect
    await call_manager.handle_remote_hangup()
    assert call_manager.state == "COMPLETED"
    assert call_manager.driver.is_on_hold is False


@pytest.mark.asyncio
async def test_comb_dual_tool_calls_in_single_turn(call_manager):
    """Test 12: Model invokes take_message and log_call_summary in same turn."""
    call_id = await call_manager.handle_inbound_ring("+15557776666")
    await call_manager.approve_inbound_call("ai_takeover")

    res1 = await call_manager.handle_tool_call("take_message", {
        "caller_name": "Sam", "phone_number": "+15557776666", "message": "Project update", "urgency": "high"
    })
    res2 = await call_manager.handle_tool_call("log_call_summary", {
        "summary": "Sam provided update", "action_items": ["Review document"]
    })

    assert res1["status"] == "saved"
    assert res2["status"] == "summary_logged"

    msgs = await call_manager.db.get_messages()
    assert len(msgs) == 1
    call_details = await call_manager.db.get_call_details(call_id)
    assert call_details["summary"] is not None
    await call_manager.terminate_call("test_done")


def test_comb_dynamic_resampling_rate_switch():
    """Test 13: PCM resampler handles dynamic rate change mid-stream."""
    pcm_16k = b"\x00\x00\x7f\x00" * 400
    res_24k = AudioProcessor.resample_pcm(pcm_16k, 16000, 24000)
    assert len(res_24k) > 0
    # Dynamic switch to 48k
    res_48k = AudioProcessor.resample_pcm(pcm_16k, 16000, 48000)
    assert len(res_48k) == len(res_24k) * 2


@pytest.mark.asyncio
async def test_comb_websocket_reconnect_during_active_call(live_server):
    """Test 14: WS client reconnects during active call and receives state snapshot."""
    await live_server.call_manager.handle_inbound_ring("+15554441111")
    await live_server.call_manager.approve_inbound_call("ai_takeover")
    assert live_server.call_manager.state == "ACTIVE"

    # Connect WS 1
    async with live_server.connect_ws("/ws/live") as ws1:
        data1 = json.loads(await ws1.recv())
        assert data1["type"] == "initial_state"
        assert data1["state"] == "ACTIVE"

    # Disconnect and Reconnect WS 2
    async with live_server.connect_ws("/ws/live") as ws2:
        data2 = json.loads(await ws2.recv())
        assert data2["type"] == "initial_state"
        assert data2["state"] == "ACTIVE"

    await live_server.call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_outbound_dial_busy_rejection(call_manager):
    """Test 15: Outbound dial returns busy/rejected."""
    with patch.object(call_manager.driver, "dial", new=AsyncMock(return_value=False)):
        success = await call_manager.initiate_outbound("+15550000000", "Goal")
        assert success is False
        assert call_manager.state in ["FAILED", "IDLE"]


@pytest.mark.asyncio
async def test_comb_inbound_ring_and_outbound_dial_collision(call_manager):
    """Test 16: Outbound dial rejected when inbound ring is active."""
    await call_manager.handle_inbound_ring("+15551112222")
    assert call_manager.state == "RINGING"

    success = await call_manager.initiate_outbound("+15559990000", "Goal")
    assert success is False
    assert call_manager.state == "RINGING"
    await call_manager.approve_inbound_call("decline")


@pytest.mark.asyncio
async def test_comb_tool_execution_timeout_recovery(call_manager):
    """Test 17: Tool execution error returns structured error response without crashing session."""
    with patch.object(call_manager, "handle_tool_call", side_effect=TimeoutError("DB lock timeout")):
        response = await call_manager.safe_handle_tool_call("take_message", {})
        assert "error" in response
        assert "DB lock timeout" in response["error"]


@pytest.mark.asyncio
async def test_comb_whisper_rapid_burst_queueing(call_manager):
    """Test 18: Burst of 5 whispers queued and sent in order."""
    await call_manager.handle_inbound_ring("+15551234567")
    await call_manager.approve_inbound_call("ai_takeover")

    whispers = [f"Steering whisper #{i}" for i in range(5)]
    await asyncio.gather(*(call_manager.inject_whisper(w) for w in whispers))
    assert len(call_manager.sent_whispers) == 5
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_whisper_cleared_on_inbound_decline(call_manager):
    """Test 19: Operator whisper buffered during ringing is purged on decline and never leaks to next call."""
    # Call 1 rings
    await call_manager.handle_inbound_ring("+15551110001", "Spammer")
    assert call_manager.state == "RINGING"

    # Operator injects private steering guidance meant for Call 1
    await call_manager.inject_whisper("Confidential steering: Decline this caller immediately")
    assert len(call_manager.pending_whispers) == 1

    # Operator declines Call 1
    await call_manager.approve_inbound_call("decline")
    assert call_manager.state == "COMPLETED"
    assert len(call_manager.pending_whispers) == 0, "Pending whispers must be cleared on decline!"

    # Call 2 arrives from VIP client
    await call_manager.handle_inbound_ring("+15552220002", "VIP Client")
    assert call_manager.state == "RINGING"
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"

    # Verify Call 1's whisper was NOT delivered to Call 2
    assert len(call_manager.sent_whispers) == 0, "Confidential whisper from Call 1 leaked into Call 2!"
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_comb_terminate_call_disconnect_exception_hangup_guarantee(call_manager):
    """Test 20: Telephony hangup and state flush are guaranteed even if Gemini client disconnect raises an exception."""
    await call_manager.handle_inbound_ring("+15553334444", "Active Caller")
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"
    is_active_initial = getattr(call_manager.driver, "active_call", False) or getattr(call_manager.driver, "call_state", "") == "ACTIVE"
    assert is_active_initial is True

    # Simulate socket drop / ConnectionResetError during disconnect
    with patch.object(call_manager.gemini_client, "disconnect", side_effect=ConnectionResetError("Socket forced close")):
        await call_manager.terminate_call("emergency_network_drop")

    # State must transition to COMPLETED, and driver must be hung up
    assert call_manager.state == "COMPLETED"
    is_active_final = getattr(call_manager.driver, "active_call", False) or getattr(call_manager.driver, "call_state", "") == "ACTIVE"
    assert is_active_final is False
    assert len(call_manager.pending_whispers) == 0

