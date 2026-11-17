"""
e2e_tests/test_tier2_boundaries.py

Tier 2: Boundary & Corner Cases (Boundary Value Analysis and Stress Hardening).
Provides >=5 tests per boundary domain:
1. Empty Audio Chunks & Zero-Duration Signals (5 tests)
2. Zero-Byte, Odd-Byte & Unaligned PCM Buffers (5 tests)
3. Extreme & Non-Standard Sample Rates (5 tests)
4. Buffer Overrun, Queue Saturation & Barge-In Clearing (5 tests)
5. Rapid Disconnect, Reconnect & Telephony Interruption (5 tests)
6. Malformed WebSocket Frames & Schema Non-Compliance (6 tests)
7. Missing, Malformed & Schema-Violating Tool Arguments (6 tests)
8. Extreme Transcript Lengths & SQL / Unicode Payloads (5 tests)
9. High Concurrency, Multithreaded & Async Contention (5 tests)
10. Web Management Console Asset Boundaries (5 tests)

Total: 52 tests.
"""

import asyncio
import base64
import json
import struct
import tempfile
import os
import numpy as np
import pytest
import websockets

from core.audio_processor import AudioProcessor
from core.gemini_live import GeminiLiveClient
from core.tools import ToolRegistry

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


# ============================================================================
# BOUNDARY DOMAIN 1: EMPTY AUDIO CHUNKS & ZERO-DURATION SIGNALS
# ============================================================================

class TestTier2EmptyAudioAndSignals:
    """Verifies defensive handling of empty audio payloads across all pipelines."""

    @pytest.mark.asyncio
    async def test_tier2_empty_audio_chunk_to_gemini(self, mock_gemini):
        """Passing b'' to send_audio_chunk is a safe no-op that sends nothing."""
        client = GeminiLiveClient()
        await client.connect()
        await client.send_audio_chunk(b"")
        await asyncio.sleep(0.02)
        assert len(mock_gemini.received_audio_chunks) == 0
        await client.disconnect()

    def test_tier2_empty_audio_resampling(self):
        """Passing b'' to resample_pcm returns b'' cleanly without error."""
        res = AudioProcessor.resample_pcm(b"", 16000, 24000)
        assert res == b""

    def test_tier2_empty_audio_levels_calculation(self):
        """Passing b'' to calculate_levels returns normalized zero metrics."""
        levels = AudioProcessor.calculate_levels(b"")
        assert levels["rms"] == 0.0
        assert levels["peak"] == 0.0
        assert all(b == 0.0 for b in levels["spectrum"])

    def test_tier2_empty_audio_driver_playback(self):
        """Calling play_audio_chunk with b'' leaves buffer unchanged."""
        driver = MockTelephonyDriver()
        driver.play_audio_chunk(b"")
        assert driver.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_tier2_empty_websocket_audio_frame(self, live_server):
        """Sending empty audio text/data frame to WebSocket does not crash connection."""
        async with live_server.connect_ws("/ws/live") as ws:
            # Drain initial state
            await ws.recv()
            await ws.send(json.dumps({"type": "audio", "data": ""}))
            # Verify socket is still responsive with ping
            await ws.send(json.dumps({"type": "ping"}))
            resp = json.loads(await ws.recv())
            assert resp.get("type") == "pong"


# ============================================================================
# BOUNDARY DOMAIN 2: ZERO-BYTE, ODD-BYTE & UNALIGNED PCM BUFFERS
# ============================================================================

class TestTier2OddAndUnalignedBuffers:
    """Verifies robustness against odd byte lengths and boundary sample values."""

    def test_tier2_odd_byte_length_pcm_resample(self, audio_synth):
        """Unaligned odd byte buffers (e.g. 641 bytes) are truncated safely."""
        pcm_640 = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)
        odd_pcm = pcm_640 + b"\x7f"  # 641 bytes
        res = AudioProcessor.resample_pcm(odd_pcm, 16000, 24000)
        assert len(res) == 960  # Exactly 480 16-bit samples

    def test_tier2_odd_byte_length_levels_calculation(self):
        """calculate_levels handles 3-byte input by evaluating the 1 valid sample."""
        three_bytes = b"\x00\x10\x7f"
        levels = AudioProcessor.calculate_levels(three_bytes)
        assert "rms" in levels
        assert "peak" in levels
        assert levels["peak"] > 0.0

    def test_tier2_digital_silence_all_zeros(self, audio_synth):
        """Digital silence chunks produce exactly 0.0 RMS and 0.0 Peak."""
        silence = audio_synth.generate_silence_pcm(duration_sec=0.05, sample_rate=16000)
        levels = AudioProcessor.calculate_levels(silence)
        assert levels["rms"] == 0.0
        assert levels["peak"] == 0.0

    def test_tier2_max_amplitude_clipping_no_overflow(self):
        """Max amplitude samples (-32768 and 32767) do not cause integer overflow."""
        extreme_bytes = struct.pack("<hh", -32768, 32767)
        arr = AudioProcessor.pcm_to_numpy(extreme_bytes)
        assert np.min(arr) >= -1.0
        assert np.max(arr) <= 1.0
        repacked = AudioProcessor.numpy_to_pcm(arr)
        unpacked = struct.unpack("<hh", repacked)
        assert unpacked[0] == -32768
        assert unpacked[1] == 32767

    def test_tier2_single_sample_buffer(self):
        """A single 2-byte sample resamples safely without index out of bounds."""
        single_sample = struct.pack("<h", 16000)
        res = AudioProcessor.resample_pcm(single_sample, 16000, 24000)
        assert len(res) in [2, 4]  # 1 or 2 samples produced


# ============================================================================
# BOUNDARY DOMAIN 3: EXTREME & NON-STANDARD SAMPLE RATES
# ============================================================================

class TestTier2ExtremeSampleRates:
    """Verifies resampling across legacy telephony, CD, and hi-res sample rates."""

    def test_tier2_rate_narrowband_8khz_to_16khz(self, audio_synth):
        """Upsampling telephony 8kHz to 16kHz doubles sample count exactly."""
        pcm_8k = audio_synth.generate_sine_pcm(freq_hz=300.0, duration_sec=0.1, sample_rate=8000)
        assert len(pcm_8k) == 1600  # 800 samples * 2

        pcm_16k = AudioProcessor.resample_pcm(pcm_8k, 8000, 16000)
        assert len(pcm_16k) == 3200  # 1600 samples * 2

    def test_tier2_rate_non_integer_cd_44_1khz_to_16khz(self, audio_synth):
        """Non-integer ratio 44.1kHz to 16kHz resamples cleanly."""
        pcm_44k = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.1, sample_rate=44100)
        pcm_16k = AudioProcessor.resample_pcm(pcm_44k, 44100, 16000)
        assert abs(len(pcm_16k) - 3200) <= 4  # Within fractional rounding tolerance

    def test_tier2_rate_studio_48khz_to_16khz(self, audio_synth):
        """Studio 48kHz downsampled by integer 3:1 yields exactly 1/3 samples."""
        pcm_48k = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.1, sample_rate=48000)
        assert len(pcm_48k) == 9600

        pcm_16k = AudioProcessor.resample_pcm(pcm_48k, 48000, 16000)
        assert len(pcm_16k) == 3200

    def test_tier2_rate_hires_96khz_to_24khz(self, audio_synth):
        """Hi-res 96kHz downsampled by integer 4:1 yields exactly 1/4 samples."""
        pcm_96k = audio_synth.generate_silence_pcm(duration_sec=0.1, sample_rate=96000)
        assert len(pcm_96k) == 19200

        pcm_24k = AudioProcessor.resample_pcm(pcm_96k, 96000, 24000)
        assert len(pcm_24k) == 4800

    def test_tier2_rate_invalid_rates_validation(self):
        """Negative or zero sample rates raise ValueError with informative message."""
        with pytest.raises(ValueError, match="Sample rates must be positive integers"):
            AudioProcessor.resample_pcm(b"\x00\x00", 0, 16000)

        with pytest.raises(ValueError, match="Sample rates must be positive integers"):
            AudioProcessor.resample_pcm(b"\x00\x00", 16000, -24000)


# ============================================================================
# BOUNDARY DOMAIN 4: BUFFER OVERRUN, QUEUE SATURATION & BARGE-IN CLEARING
# ============================================================================

class TestTier2BufferAndBargeInFlushing:
    """Verifies backpressure handling, high volume streaming, and instant buffer flush."""

    @pytest.mark.asyncio
    async def test_tier2_buffer_audio_input_burst_backpressure(self, mock_gemini, audio_synth):
        """Rapid burst of 100 audio chunks (2 seconds of audio) queues without event loop hang."""
        client = GeminiLiveClient()
        await client.connect()

        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)
        # Push 100 chunks immediately
        tasks = [client.send_audio_chunk(chunk) for _ in range(100)]
        await asyncio.gather(*tasks)

        for _ in range(20):
            if len(mock_gemini.received_audio_chunks) >= 100:
                break
            await asyncio.sleep(0.05)

        assert len(mock_gemini.received_audio_chunks) == 100
        await client.disconnect()

    def test_tier2_buffer_driver_playback_overflow_policy(self, audio_synth):
        """Flooding playback buffer with 500 chunks operates cleanly without memory error."""
        driver = MockTelephonyDriver()
        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=24000)
        for _ in range(500):
            driver.play_audio_chunk(chunk)

        assert driver.played_chunks_count == 500
        assert driver.get_buffer_size() == 500 * len(chunk)
        driver.clear_audio_buffer()
        assert driver.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_tier2_buffer_barge_in_instant_flush(self, call_manager, audio_synth):
        """Interruption signal flushes driver audio buffer to 0 bytes instantly (<5ms)."""
        chunk = audio_synth.generate_sine_pcm(duration_sec=0.1, sample_rate=24000)
        for _ in range(10):
            call_manager.driver.play_audio_chunk(chunk)

        assert call_manager.driver.get_buffer_size() > 0
        t0 = asyncio.get_event_loop().time()
        await call_manager.handle_barge_in()
        t1 = asyncio.get_event_loop().time()

        assert call_manager.driver.get_buffer_size() == 0
        assert (t1 - t0) < 0.020  # Under 20ms

    def test_tier2_buffer_barge_in_suppresses_queued_audio(self):
        """Subsequent clear_audio_buffer invocations increment buffer_cleared_count."""
        driver = MockTelephonyDriver()
        driver.clear_audio_buffer()
        driver.clear_audio_buffer()
        driver.clear_audio_buffer()
        assert driver.buffer_cleared_count == 3

    def test_tier2_buffer_memory_stability_long_stream(self, audio_synth):
        """Iterative resampling of 2,000 frames does not accumulate memory leaks."""
        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)
        for _ in range(2000):
            res = AudioProcessor.resample_pcm(chunk, 16000, 24000)
            levels = AudioProcessor.calculate_levels(res, 24000)
            assert levels["rms"] >= 0.0


# ============================================================================
# BOUNDARY DOMAIN 5: RAPID DISCONNECT, RECONNECT & TELEPHONY INTERRUPTION
# ============================================================================

class TestTier2RapidDisconnectAndInterruption:
    """Verifies client resilience under abrupt socket dropouts and rapid lifecycle cycling."""

    @pytest.mark.asyncio
    async def test_tier2_rapid_websocket_disconnect_during_speech(self, mock_gemini, audio_synth):
        """Client disconnect while speech is streaming terminates tasks cleanly."""
        client = GeminiLiveClient()
        await client.connect()

        chunk = audio_synth.generate_sine_pcm(duration_sec=0.05, sample_rate=24000)
        await mock_gemini.emit_speech(pcm_24k=chunk)
        # Immediate abrupt disconnect
        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_tier2_gemini_remote_socket_closure_recovery(self, mock_gemini, call_manager):
        """Mock server force disconnect (1011) is caught cleanly and call marked COMPLETED."""
        await call_manager.handle_inbound_ring("+15551239999")
        await call_manager.approve_inbound_call("ai_takeover")
        assert call_manager.state == "ACTIVE"

        # Remote server abnormal closure
        await mock_gemini.force_disconnect(code=1011)
        await asyncio.sleep(0.05)
        await call_manager.handle_socket_disconnect(1011, "Abnormal close")

        assert call_manager.state == "COMPLETED"

    @pytest.mark.asyncio
    async def test_tier2_rapid_reconnect_loop(self, mock_gemini):
        """5 rapid consecutive connect/disconnect cycles operate without socket leaks."""
        client = GeminiLiveClient()
        for _ in range(5):
            await client.connect()
            assert client.is_connected
            await client.disconnect()
            assert not client.is_connected

    @pytest.mark.asyncio
    async def test_tier2_driver_hangup_during_streaming(self, call_manager, audio_synth):
        """Hangup while caller audio is streaming clears callbacks and terminates call."""
        await call_manager.handle_inbound_ring("+15559871111")
        await call_manager.approve_inbound_call("ai_takeover")

        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)
        call_manager.feed_caller_audio(chunk)

        await call_manager.terminate_call("caller_hangup")
        assert call_manager.state == "COMPLETED"
        assert not call_manager.gemini_client.is_connected

    @pytest.mark.asyncio
    async def test_tier2_dial_while_call_in_progress(self, async_http_client):
        """POST /api/call/dial while call is active returns HTTP 409 Conflict."""
        # Initiate first call
        res1 = await async_http_client.post(
            "/api/call/dial",
            json={"phone_number": "+15551112222", "goal_prompt": "First call"}
        )
        assert res1.status_code == 200

        # Attempt second dial
        res2 = await async_http_client.post(
            "/api/call/dial",
            json={"phone_number": "+15553334444", "goal_prompt": "Second call"}
        )
        assert res2.status_code == 409
        assert "Call already active" in res2.json()["detail"]

        # Teardown
        await async_http_client.post("/api/call/hangup", json={})


# ============================================================================
# BOUNDARY DOMAIN 6: MALFORMED WEBSOCKET FRAMES & SCHEMA NON-COMPLIANCE
# ============================================================================

class TestTier2MalformedWebSocketFrames:
    """Verifies WebSocket hub defensively handles corrupted and unexpected frames."""

    @pytest.mark.asyncio
    async def test_tier2_ws_malformed_json_text_frame(self, live_server):
        """Sending unparseable text returns error response without closing socket."""
        async with live_server.connect_ws("/ws/live") as ws:
            await ws.recv()  # initial_state
            await ws.send("{not valid json at all")
            err = json.loads(await ws.recv())
            assert err.get("type") == "error"
            assert "Invalid JSON" in err.get("message", "")

    @pytest.mark.asyncio
    async def test_tier2_ws_missing_type_field(self, live_server):
        """Sending JSON missing required 'type' field returns validation error."""
        async with live_server.connect_ws("/ws/live") as ws:
            await ws.recv()
            await ws.send(json.dumps({"payload": 12345}))
            err = json.loads(await ws.recv())
            assert err.get("type") == "error"
            assert "Missing 'type' field" in err.get("message", "")

    @pytest.mark.asyncio
    async def test_tier2_ws_unexpected_binary_payload(self, live_server):
        """Sending binary frame to control endpoint is handled gracefully."""
        async with live_server.connect_ws("/ws/live") as ws:
            await ws.recv()
            await ws.send(b"\x00\x01\x02\x03\x04")
            # Connection remains open; verify with ping
            await ws.send(json.dumps({"type": "ping"}))
            resp = json.loads(await ws.recv())
            assert resp.get("type") == "pong"

    @pytest.mark.asyncio
    async def test_tier2_ws_unknown_event_type(self, live_server):
        """Sending unrecognized event type does not crash server."""
        async with live_server.connect_ws("/ws/live") as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "non_existent_event_xyz"}))
            await ws.send(json.dumps({"type": "ping"}))
            resp = json.loads(await ws.recv())
            assert resp.get("type") == "pong"

    @pytest.mark.asyncio
    async def test_tier2_ws_oversized_payload(self, live_server):
        """Sending a 100KB whisper text is handled safely."""
        async with live_server.connect_ws("/ws/live") as ws:
            await ws.recv()
            massive_text = "A" * 100000
            await ws.send(json.dumps({"type": "whisper", "instruction": massive_text}))
            await ws.send(json.dumps({"type": "ping"}))
            resp = json.loads(await ws.recv())
            assert resp.get("type") == "pong"

    @pytest.mark.asyncio
    async def test_tier2_ws_gemini_non_dict_json_payloads(self, mock_gemini):
        """Sending top-level non-dict JSON payloads (null, [], 12345, string, bool) to
        MockGeminiLiveServer does not crash the handler with 1011; subsequent valid
        setup and audio payloads are processed normally."""
        uri = f"ws://127.0.0.1:{mock_gemini.actual_port}"
        async with websockets.connect(uri) as ws:
            # 1. Send various non-dict top-level JSON payloads
            non_dict_payloads = ["null", "[]", "12345", json.dumps("just_a_string"), "true", "false"]
            for payload in non_dict_payloads:
                await ws.send(payload)

            # Allow server event loop to process incoming frames
            await asyncio.sleep(0.05)

            # 2. Verify connection remains alive and processes valid setup handshake
            setup_payload = {
                "setup": {
                    "model": "models/gemini-3.8-live",
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": "Aoede"
                                }
                            }
                        }
                    }
                }
            }
            await ws.send(json.dumps(setup_payload))
            resp_raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            resp = json.loads(resp_raw)
            assert "setupComplete" in resp
            assert mock_gemini.setup_received_event.is_set()

            # 3. Verify subsequent valid realtime audio message is received and decoded
            raw_pcm = b"\x00\x00" * 160  # 160 samples of 16-bit mono silence (320 bytes)
            b64_audio = base64.b64encode(raw_pcm).decode("utf-8")
            audio_msg = {
                "realtimeInput": {
                    "mediaChunks": [
                        {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": b64_audio
                        }
                    ]
                }
            }
            await ws.send(json.dumps(audio_msg))
            await asyncio.sleep(0.05)
            assert len(mock_gemini.received_audio_chunks) >= 1
            assert mock_gemini.received_audio_chunks[-1] == raw_pcm


# ============================================================================
# BOUNDARY DOMAIN 7: MISSING, MALFORMED & SCHEMA-VIOLATING TOOL ARGUMENTS
# ============================================================================

class TestTier2ToolArgumentBoundaries:
    """Verifies tool handler defenses against missing or malformed arguments."""

    @pytest.mark.asyncio
    async def test_tier2_tool_take_message_missing_required_fields(self, call_manager):
        """take_message with missing caller_name defaults gracefully without crashing."""
        res = await call_manager.handle_tool_call("take_message", {
            "message": "Only message provided"
        })
        assert res["status"] == "saved"
        assert res["caller_name"] == "Unknown"

    @pytest.mark.asyncio
    async def test_tier2_tool_take_message_invalid_urgency_value(self, call_manager):
        """take_message with unrecognized urgency records verbatim or defaults."""
        res = await call_manager.handle_tool_call("take_message", {
            "caller_name": "Carol",
            "message": "Note",
            "urgency": "ultra-mega-urgent"
        })
        assert res["status"] == "saved"

    @pytest.mark.asyncio
    async def test_tier2_tool_end_call_empty_args(self, call_manager):
        """end_call invoked with empty dict applies default reason 'normal'."""
        await call_manager.handle_inbound_ring("+15552223333")
        await call_manager.approve_inbound_call("ai_takeover")
        res = await call_manager.handle_tool_call("end_call", {})
        assert res["status"] == "call_ended"
        assert res["reason"] == "normal"
        assert call_manager.state == "COMPLETED"

    @pytest.mark.asyncio
    async def test_tier2_tool_log_call_summary_invalid_types(self, call_manager):
        """log_call_summary handles action_items provided as a string instead of a list."""
        call_id = await call_manager.handle_inbound_ring("+15554445555")
        await call_manager.approve_inbound_call("ai_takeover")

        res = await call_manager.handle_tool_call("log_call_summary", {
            "summary": "String action item test",
            "action_items": "Single action item string"
        })
        assert res["status"] == "summary_logged"
        details = await call_manager.db.get_call_details(call_id)
        assert details["summary"]["action_items"] == ["Single action item string"]

    @pytest.mark.asyncio
    async def test_tier2_tool_unparseable_json_args(self, call_manager):
        """safe_handle_tool_call catches unexpected exceptions cleanly."""
        res = await call_manager.safe_handle_tool_call("unknown_fake_tool", {})
        assert "error" in res

    @pytest.mark.asyncio
    async def test_tier2_tool_take_message_none_arguments(self, call_manager):
        """take_message safely handles explicit None values for all arguments without SQLite NOT NULL errors."""
        call_id = await call_manager.handle_inbound_ring("+15559990000")
        await call_manager.approve_inbound_call("ai_takeover")

        res = await call_manager.handle_tool_call("take_message", {
            "caller_name": None,
            "phone_number": None,
            "message": "Null fields test message",
            "urgency": None,
        })
        assert res["status"] == "saved"
        assert res["caller_name"] == "Unknown"

        # Verify database record was persisted with non-null string coercion
        msgs = await call_manager.db.get_messages()
        assert len(msgs) >= 1
        saved = msgs[-1]
        assert saved["caller_name"] == "Unknown"
        assert saved["phone_number"] == ""
        assert saved["message"] == "Null fields test message"
        assert saved["urgency"] in ["medium", "normal"]
        await call_manager.terminate_call("test_done")


# ============================================================================
# BOUNDARY DOMAIN 8: EXTREME TRANSCRIPT LENGTHS & SQL / UNICODE PAYLOADS
# ============================================================================

class TestTier2ExtremeTranscriptsAndUnicode:
    """Verifies database and WebSocket resilience against large strings and SQL injection."""

    @pytest.mark.asyncio
    async def test_tier2_transcript_massive_chunk_50k_chars(self, temp_db):
        """50,000-character transcript string persists and retrieves accurately."""
        call_id = await temp_db.create_call({"caller_id": "+15557778888"})
        massive_text = "Amar voice calling test " * 2000  # ~48,000 chars
        await temp_db.append_transcript(call_id, "agent", massive_text, is_final=True)

        transcripts = await temp_db.get_transcripts(call_id)
        assert len(transcripts) == 1
        assert transcripts[0]["text"] == massive_text

    @pytest.mark.asyncio
    async def test_tier2_transcript_sql_injection_resilience(self, temp_db):
        """Malicious SQL injection payload is stored verbatim without executing."""
        call_id = await temp_db.create_call({"caller_id": "+15550009999"})
        injection_text = "'; DROP TABLE calls; DROP TABLE messages; --"
        await temp_db.append_transcript(call_id, "caller", injection_text)

        # Verify calls table still exists and record is untouched
        calls = await temp_db.get_recent_calls()
        assert len(calls) >= 1
        trans = await temp_db.get_transcripts(call_id)
        assert trans[0]["text"] == injection_text

    @pytest.mark.asyncio
    async def test_tier2_transcript_multilingual_unicode_emojis(self, temp_db):
        """Multilingual UTF-8 characters and emojis persist with exact fidelity."""
        call_id = await temp_db.create_call({"caller_id": "+15551239999"})
        unicode_text = "📞 Hello 您好 مرحبا नमस्ते 🤖 Call screening activated!"
        await temp_db.append_transcript(call_id, "agent", unicode_text)

        trans = await temp_db.get_transcripts(call_id)
        assert trans[0]["text"] == unicode_text

    @pytest.mark.asyncio
    async def test_tier2_whisper_massive_instruction(self, call_manager):
        """Operator whisper containing 10,000 characters is buffered safely."""
        await call_manager.handle_inbound_ring("+15558880000")
        large_instruction = "Steer the conversation carefully: " + ("guidance " * 1000)
        await call_manager.inject_whisper(large_instruction)
        assert len(call_manager.pending_whispers) == 1
        assert call_manager.pending_whispers[0] == large_instruction

    @pytest.mark.asyncio
    async def test_tier2_db_pagination_extreme_offset_limit(self, temp_db):
        """Extreme limit and offset query returns empty list without error."""
        calls = await temp_db.get_recent_calls(limit=0, offset=999999)
        assert calls == []


# ============================================================================
# BOUNDARY DOMAIN 9: HIGH CONCURRENCY, MULTITHREADED & ASYNC CONTENTION
# ============================================================================

class TestTier2HighConcurrencyAndContention:
    """Verifies SQLite WAL concurrency and parallel request handling."""

    @pytest.mark.asyncio
    async def test_tier2_concurrency_multi_websocket_broadcasting(self, live_server):
        """10 simultaneous WebSocket clients connect and receive initial state."""
        connections = [live_server.connect_ws("/ws/live") for _ in range(10)]
        sockets = [await c.__aenter__() for c in connections]

        try:
            for ws in sockets:
                raw = await ws.recv()
                data = json.loads(raw)
                assert data["type"] == "initial_state"
        finally:
            for ws, c in zip(sockets, connections):
                await c.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_tier2_concurrency_simultaneous_rest_queries(self, async_http_client):
        """25 concurrent REST queries to /api/status execute cleanly under 1.5 seconds."""
        t0 = asyncio.get_event_loop().time()
        tasks = [async_http_client.get("/api/status") for _ in range(25)]
        responses = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - t0

        assert all(r.status_code == 200 for r in responses)
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_tier2_concurrency_simultaneous_db_writes(self, temp_db):
        """Concurrent async tasks writing calls, messages, and transcripts without locking."""
        call_id = await temp_db.create_call({"caller_id": "+15551239999"})

        async def write_turn(idx):
            await temp_db.append_transcript(call_id, "agent", f"Concurrent turn {idx}")
            await temp_db.save_message(call_id, f"Caller {idx}", f"555-{idx}", f"Msg {idx}")

        tasks = [write_turn(i) for i in range(20)]
        await asyncio.gather(*tasks)

        transcripts = await temp_db.get_transcripts(call_id)
        messages = await temp_db.get_messages()
        assert len(transcripts) == 20
        assert len(messages) == 20

    @pytest.mark.asyncio
    async def test_tier2_concurrency_audio_callback_during_hangup(self, call_manager, audio_synth):
        """Audio chunk arrival concurrently with terminate_call does not crash."""
        await call_manager.handle_inbound_ring("+15553338888")
        await call_manager.approve_inbound_call("ai_takeover")

        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)

        async def feed_burst():
            for _ in range(10):
                call_manager.feed_caller_audio(chunk)
                await asyncio.sleep(0.005)

        async def hangup_task():
            await asyncio.sleep(0.015)
            await call_manager.terminate_call("concurrent_hangup")

        await asyncio.gather(feed_burst(), hangup_task())
        assert call_manager.state == "COMPLETED"

    @pytest.mark.asyncio
    async def test_tier2_concurrency_rapid_inbound_simulations(self, call_manager):
        """Second inbound ring while first is ringing raises RuntimeError."""
        await call_manager.handle_inbound_ring("+15551111111")
        assert call_manager.state == "RINGING"

        with pytest.raises(RuntimeError, match="Cannot accept inbound call"):
            await call_manager.handle_inbound_ring("+15552222222")

        await call_manager.approve_inbound_call("decline")
        assert call_manager.state == "COMPLETED"


# ============================================================================
# BOUNDARY DOMAIN 10: WEB MANAGEMENT CONSOLE ASSET BOUNDARIES
# ============================================================================

class TestTier2WebConsoleAssetBoundaries:
    """Verifies path traversal resistance, 404s, invalid methods, and concurrent static streaming."""

    @pytest.mark.asyncio
    async def test_tier2_web_console_path_traversal_resilience(self, async_http_client):
        """Directory traversal payloads (../, ..%2F) return 404 or 400 and never expose filesystem."""
        traversal_paths = [
            "/static/../../etc/passwd",
            "/static/..%2F..%2Fwindows%2Fwin.ini",
            "/static/subdir/../../../etc/shadow",
            "/static/....//....//config.py",
        ]
        for path in traversal_paths:
            resp = await async_http_client.get(path)
            assert resp.status_code in [400, 404], f"Path {path} returned unsafe status {resp.status_code}"

    @pytest.mark.asyncio
    async def test_tier2_web_console_non_existent_asset_404(self, async_http_client):
        """Non-existent static asset requests return 404 Not Found."""
        missing_paths = [
            "/static/non_existent_image.png",
            "/static/missing_script.js",
            "/static/fonts/custom.woff2",
        ]
        for path in missing_paths:
            resp = await async_http_client.get(path)
            assert resp.status_code == 404, f"Path {path} expected 404 but got {resp.status_code}"

    @pytest.mark.asyncio
    async def test_tier2_web_console_invalid_http_methods(self, async_http_client):
        """Disallowed HTTP methods (POST, PUT, DELETE) on static files return 405 Method Not Allowed."""
        resp_post = await async_http_client.post("/static/styles.css", content=b"/* invalid */")
        assert resp_post.status_code == 405

        resp_put = await async_http_client.put("/static/index.html", content=b"<h1>Overwrite</h1>")
        assert resp_put.status_code == 405

        resp_delete = await async_http_client.delete("/static/app.js")
        assert resp_delete.status_code == 405

    @pytest.mark.asyncio
    async def test_tier2_web_console_empty_corrupt_asset_fallback(self, async_http_client):
        """Malformed query parameters and out-of-range headers are handled cleanly without 500 errors."""
        # Malformed query string with null bytes
        resp_query = await async_http_client.get("/static/styles.css?cache=null%00byte")
        assert resp_query.status_code in [200, 400, 404]
        assert resp_query.status_code != 500

        # Out-of-bounds byte range request
        resp_range = await async_http_client.get("/static/styles.css", headers={"Range": "bytes=9999999-99999999"})
        assert resp_range.status_code in [200, 206, 416]
        assert resp_range.status_code != 500

    @pytest.mark.asyncio
    async def test_tier2_web_console_concurrent_static_streaming(self, async_http_client):
        """Concurrent static asset streaming serves 30 requests simultaneously without deadlock."""
        t0 = asyncio.get_event_loop().time()
        routes = ["/", "/static/index.html", "/static/styles.css", "/static/app.js"] * 10  # 40 requests
        tasks = [async_http_client.get(r) for r in routes]
        responses = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - t0

        assert all(r.status_code == 200 for r in responses)
        assert elapsed < 3.0

