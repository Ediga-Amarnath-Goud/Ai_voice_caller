"""
e2e_tests/test_tier4_scenarios.py

Tier 4: Real-World Application Workloads.
Comprehensive multi-turn simulated telephone call workflows:
- Scenario 1: Complete Inbound Call Screening & Message Taking
- Scenario 2: Outbound Goal-Driven Appointment Dialing
- Scenario 3: Operator Whisper Live Steering
- Scenario 4: Natural Caller Barge-In During Assistant Speech
- Scenario 5: Complex Inquiry Escalation and Transfer to Human
- Scenario 6: Network Disconnect and Graceful Telephony Teardown

Total: 6 scenarios.
"""

import asyncio
import time
import pytest

from e2e_tests.conftest import MockTelephonyDriver, DatabaseManager


@pytest.mark.asyncio
async def test_scenario_1_inbound_screening_and_message_taking(call_manager, mock_gemini, audio_synth):
    """
    Scenario 1: Complete Inbound Call Screening & Message Taking.
    Flow:
    1. Incoming ring (+15550192834, FedEx Express) -> RINGING.
    2. Web console approval ('ai_takeover') -> ACTIVE.
    3. Gemini greeting in Aoede voice -> assistant speech played.
    4. Caller states drop-off note -> caller audio streamed.
    5. Gemini invokes take_message -> persisted in SQLite.
    6. Gemini speaks farewell & invokes end_call -> COMPLETED.
    7. All DB records verified.
    """
    # 1. Incoming call rings
    call_id = await call_manager.handle_inbound_ring("+15550192834", "FedEx Express")
    assert call_manager.state == "RINGING"
    assert call_manager.current_call_id == call_id

    # 2. Operator approves via web console
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"
    assert call_manager.driver.active_call is True

    # 3. Gemini Live Aoede voice greeting
    greeting_pcm = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.1, sample_rate=24000)
    await mock_gemini.emit_speech(
        text="Hello! Amar is currently unavailable, I'm his AI assistant. How can I help you today?",
        pcm_24k=greeting_pcm,
        turn_complete=True
    )
    await asyncio.sleep(0.05)
    assert call_manager.driver.played_chunks_count > 0

    # 4. Caller states message
    caller_audio = audio_synth.generate_sine_pcm(freq_hz=300.0, duration_sec=0.1, sample_rate=16000)
    call_manager.feed_caller_audio(caller_audio)
    await call_manager.db.append_transcript(
        call_id, "caller", "Leaving package by garage doors, sign online at fedex.com", is_final=True
    )

    # 5. Gemini takes structured message
    tool_resp = await call_manager.handle_tool_call("take_message", {
        "caller_name": "John - FedEx",
        "phone_number": "+15550192834",
        "message": "Leaving package by garage doors, sign online at fedex.com",
        "urgency": "medium"
    })
    assert tool_resp["status"] == "saved"

    # 6. Gemini says goodbye and terminates
    farewell_pcm = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.05, sample_rate=24000)
    await mock_gemini.emit_speech(
        text="Thank you John, I have logged your message. Have a great day!",
        pcm_24k=farewell_pcm,
        turn_complete=True
    )
    await call_manager.handle_tool_call("end_call", {"reason": "completed_message"})
    assert call_manager.state == "COMPLETED"
    assert call_manager.driver.active_call is False

    # 7. Verification of SQLite records
    call_rec = await call_manager.db.get_call_details(call_id)
    assert call_rec["status"] == "completed"
    assert call_rec["caller_id"] == "+15550192834"
    assert call_rec["caller_name"] == "FedEx Express"
    assert call_rec["hangup_reason"] == "completed_message"

    msgs = await call_manager.db.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["caller_name"] == "John - FedEx"
    assert msgs[0]["urgency"] == "medium"
    assert "garage doors" in msgs[0]["message"]

    transcripts = await call_manager.db.get_transcripts(call_id)
    assert len(transcripts) >= 1
    assert any("fedex.com" in t["text"] for t in transcripts)


@pytest.mark.asyncio
async def test_scenario_2_outbound_goal_driven_appointment_dialing(call_manager, mock_gemini, audio_synth):
    """
    Scenario 2: Outbound Goal-Driven Appointment Dialing.
    Flow:
    1. Initiate outbound call with goal prompt.
    2. Driver dials, call transitions to ACTIVE.
    3. Multi-turn conversation simulated.
    4. Gemini invokes log_call_summary with action items.
    5. Call terminates cleanly, DB records verified.
    """
    goal = "Book dental cleaning with Dr. Smith for next Tuesday afternoon. Insurance: Delta Dental."

    # 1. Trigger outbound call
    success = await call_manager.initiate_outbound("+15559876543", goal)
    assert success is True
    assert call_manager.state == "ACTIVE"
    call_id = call_manager.current_call_id

    # 2. Assistant introduces goal
    await mock_gemini.emit_speech(
        text="Hello, I'm calling on behalf of Amar to schedule a dental cleaning for next Tuesday afternoon.",
        turn_complete=True
    )
    await call_manager.db.append_transcript(
        call_id, "agent", "Hello, I'm calling on behalf of Amar to schedule a dental cleaning for next Tuesday afternoon."
    )

    # Receptionist confirms
    receptionist_pcm = audio_synth.generate_sine_pcm(duration_sec=0.05, sample_rate=16000)
    call_manager.feed_caller_audio(receptionist_pcm)
    await call_manager.db.append_transcript(
        call_id, "caller", "We have Tuesday at 2:30 PM available with Dr. Smith."
    )

    # 3. Gemini logs summary and hangs up
    await call_manager.handle_tool_call("log_call_summary", {
        "summary": "Booked dental cleaning with Dr. Smith for Tuesday at 2:30 PM.",
        "action_items": ["Add to calendar", "Bring Delta Dental insurance card"]
    })
    await call_manager.handle_tool_call("end_call", {"reason": "appointment_booked"})

    # 4. Assertions
    assert call_manager.state == "COMPLETED"
    call_rec = await call_manager.db.get_call_details(call_id)
    assert call_rec["direction"] == "outbound"
    assert call_rec["status"] == "completed"
    assert call_rec["goal_prompt"] == goal
    assert call_rec["summary"] is not None
    assert "Dr. Smith" in call_rec["summary"]["summary"]
    assert len(call_rec["summary"]["action_items"]) == 2


@pytest.mark.asyncio
async def test_scenario_3_operator_whisper_live_steering(call_manager, mock_gemini, audio_synth):
    """
    Scenario 3: Operator Whisper Live Steering.
    Flow:
    1. Recruiter pitches hybrid role.
    2. Operator injects whisper instructions.
    3. Gemini incorporates instructions into next speech turn.
    4. Steered dialogue logged in DB.
    """
    call_id = await call_manager.handle_inbound_ring("+15554443333", "Acme Recruiting")
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"

    # Recruiter pitch
    pitch_pcm = audio_synth.generate_sine_pcm(duration_sec=0.05, sample_rate=16000)
    call_manager.feed_caller_audio(pitch_pcm)
    await call_manager.db.append_transcript(
        call_id, "caller", "We have a hybrid role in Seattle with $180k compensation."
    )

    # Operator injects steering whisper
    await call_manager.inject_whisper("Amar is only interested in 100% remote roles with minimum $220k base.")
    assert len(call_manager.sent_whispers) == 1

    # Assistant incorporates steering
    steered_text = "Amar is exclusively considering 100% remote opportunities with a minimum base of $220k."
    await mock_gemini.emit_speech(text=steered_text, turn_complete=True)
    await call_manager.db.append_transcript(call_id, "agent", steered_text)

    # Verify transcripts
    transcripts = await call_manager.db.get_transcripts(call_id)
    assert any("220k" in t["text"] for t in transcripts)
    assert any("remote" in t["text"].lower() for t in transcripts)

    await call_manager.terminate_call("steering_successful")
    assert call_manager.state == "COMPLETED"


@pytest.mark.asyncio
async def test_scenario_4_natural_caller_barge_in_playback_flush(call_manager, audio_synth):
    """
    Scenario 4: Natural Caller Barge-In During Assistant Speech.
    Flow:
    1. Assistant speaks lengthy monologue, buffering audio.
    2. Caller interrupts.
    3. Buffer flushed immediately (<50ms target).
    4. Stale audio discarded; state remains ACTIVE.
    """
    await call_manager.handle_inbound_ring("+15555556666", "Talkative Caller")
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"

    # Assistant begins long monologue
    speech_chunks = [audio_synth.generate_sine_pcm(duration_sec=0.1, sample_rate=24000) for _ in range(5)]
    call_manager.driver.queue_audio_chunks(speech_chunks)
    assert call_manager.driver.get_buffer_size() > 0

    # Caller interrupts loudly
    t0 = time.perf_counter()
    await call_manager.handle_barge_in()
    t_flush_ms = (time.perf_counter() - t0) * 1000.0

    # Assertions
    assert t_flush_ms < 50.0  # Flushed under 50ms
    assert call_manager.driver.get_buffer_size() == 0
    assert call_manager.state == "ACTIVE"
    await call_manager.terminate_call("test_done")


@pytest.mark.asyncio
async def test_scenario_5_complex_inquiry_escalation_transfer_to_human(call_manager):
    """
    Scenario 5: Complex Inquiry Escalation and Transfer to Human.
    Flow:
    1. Inbound critical call arrives.
    2. Gemini invokes transfer_to_human tool.
    3. State transitions to ON_HOLD; hold chime starts.
    4. Operator takes over call -> OPERATOR_TALKING.
    5. Operator finishes and hangs up -> COMPLETED.
    """
    call_id = await call_manager.handle_inbound_ring("+15559110000", "Police Dept")
    await call_manager.approve_inbound_call("ai_takeover")

    # Gemini recognizes urgency and transfers
    res = await call_manager.handle_tool_call("transfer_to_human", {
        "reason": "Urgent police inquiry regarding residence"
    })
    assert res["status"] == "transfer_initiated"
    assert call_manager.state == "ON_HOLD"
    assert call_manager.driver.is_on_hold is True

    # Operator takes over
    await call_manager.operator_pickup()
    assert call_manager.state == "OPERATOR_TALKING"
    assert call_manager.driver.is_on_hold is False

    # Operator finishes call
    await call_manager.terminate_call("operator_finished")
    assert call_manager.state == "COMPLETED"

    details = await call_manager.db.get_call_details(call_id)
    assert details["status"] == "completed"


@pytest.mark.asyncio
async def test_scenario_6_network_disconnect_and_graceful_teardown(call_manager):
    """
    Scenario 6: Network Disconnect and Graceful Telephony Teardown.
    Flow:
    1. Active call with partial transcripts logged.
    2. Unexpected network disconnect (1011).
    3. Safe recovery: partial transcripts saved, telephony driver hung up, state COMPLETED.
    4. Subsequent call starts and succeeds cleanly without system deadlock.
    """
    call1_id = await call_manager.handle_inbound_ring("+15558881234")
    await call_manager.approve_inbound_call("ai_takeover")
    assert call_manager.state == "ACTIVE"

    # Simulate conversation progress
    await call_manager.db.append_transcript(call1_id, "caller", "Hello? Are you there?", is_final=True)
    await call_manager.db.append_transcript(call1_id, "agent", "Yes, I am listening.", is_final=True)

    # Abrupt socket drop
    await call_manager.handle_socket_disconnect(1011, "Abnormal socket closure")

    # Verify graceful teardown
    assert call_manager.state in ["COMPLETED", "IDLE"]
    assert call_manager.driver.active_call is False

    # Verify transcripts preserved
    saved_trans = await call_manager.db.get_transcripts(call1_id)
    assert len(saved_trans) == 2

    # Subsequent call starts cleanly
    call2_id = await call_manager.handle_inbound_ring("+15558885678")
    assert call_manager.state == "RINGING"
    assert call_manager.current_call_id == call2_id
    await call_manager.approve_inbound_call("decline")
    assert call_manager.state == "COMPLETED"
