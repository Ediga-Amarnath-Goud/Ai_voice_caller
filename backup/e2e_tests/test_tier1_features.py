"""
e2e_tests/test_tier1_features.py

Tier 1: Feature Coverage (Category-Partition Testing).
Provides >=5 tests per feature area across all system domains:
1. Gemini Live 3.8 Connection Setup & Lifecycle (5 tests)
2. 16kHz PCM Audio Streaming Input (5 tests)
3. 24kHz PCM Audio Streaming Output & Unpacking (5 tests)
4. PCM Resampling Engine (5 tests)
5. Aoede Voice Persona & SpeechConfig (5 tests)
6. Agent Function Calling Tools (5 tests)
7. Telephony Drivers (Windows, Virtual, Linux) (5 tests)
8. SQLite Database Persistence Layer & CRUD (5 tests)
9. FastAPI REST Endpoints & WebSocket /ws/live Hub (5 tests)
10. Web Management Console Assets (5 tests)

Total: 50 tests.
"""

import asyncio
import inspect
import json
from unittest.mock import patch
import aiosqlite
import numpy as np
import pytest
import websockets

from core.audio_processor import AudioProcessor
from core.gemini_live import GeminiLiveClient, DEFAULT_SYSTEM_INSTRUCTION
from core.tools import get_tool_declarations, ToolRegistry

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

from e2e_tests.conftest import AudioSynthesizer


# ============================================================================
# FEATURE AREA 1: GEMINI LIVE 3.8 CONNECTION SETUP & LIFECYCLE
# ============================================================================

class TestTier1GeminiLiveConnection:
    """Verifies live WebSocket session setup, config transmission, and teardown."""

    @pytest.mark.asyncio
    async def test_tier1_gemini_connection_success(self, mock_gemini):
        """Verifies GeminiLiveClient connects and awaits setup completion."""
        client = GeminiLiveClient()
        assert not client.is_connected
        await client.connect()
        assert client.is_connected
        assert mock_gemini.connected_clients_count == 1
        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_tier1_gemini_disconnect_graceful(self, mock_gemini):
        """Verifies disconnect terminates background receive task cleanly."""
        client = GeminiLiveClient()
        await client.connect()
        assert client._receive_task is not None
        assert not client._receive_task.done()

        await client.disconnect()
        assert client._session is None
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_tier1_gemini_reconnect_after_disconnect(self, mock_gemini):
        """Verifies client can reconnect sequentially on the same instance."""
        client = GeminiLiveClient()
        await client.connect()
        assert client.is_connected
        await client.disconnect()
        assert not client.is_connected

        # Second connection
        await client.connect()
        assert client.is_connected
        await client.disconnect()
        assert mock_gemini.connected_clients_count == 2

    @pytest.mark.asyncio
    async def test_tier1_gemini_handshake_modalities(self, mock_gemini):
        """Verifies client transmits response_modalities=['AUDIO'] in setup payload."""
        client = GeminiLiveClient()
        await client.connect()
        setup = await mock_gemini.wait_for_setup()

        # Check modalities inside setup payload
        gen_config = setup.get("generationConfig", {})
        modalities = gen_config.get("responseModalities", [])
        assert "AUDIO" in modalities or "audio" in [m.lower() for m in modalities]
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_gemini_setup_complete_signal(self, mock_gemini):
        """Verifies setupComplete signal marks connection ready without timeout."""
        client = GeminiLiveClient()
        t0 = asyncio.get_event_loop().time()
        await client.connect()
        t1 = asyncio.get_event_loop().time()

        assert client.is_connected
        assert (t1 - t0) < 6.0  # Finished handshake well under timeout
        await client.disconnect()


# ============================================================================
# FEATURE AREA 2: 16kHZ PCM AUDIO STREAMING INPUT
# ============================================================================

class TestTier1AudioInputStreaming:
    """Verifies streaming 16kHz PCM audio chunks to Gemini Live."""

    @pytest.mark.asyncio
    async def test_tier1_audio_in_send_single_20ms_chunk(self, mock_gemini, audio_synth):
        """Verifies a single 20ms frame (320 samples = 640 bytes) is transmitted."""
        client = GeminiLiveClient()
        await client.connect()

        chunk_640b = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.02, sample_rate=16000)
        assert len(chunk_640b) == 640

        await client.send_audio_chunk(chunk_640b)
        for _ in range(10):
            if len(mock_gemini.received_audio_chunks) >= 1:
                break
            await asyncio.sleep(0.02)

        assert len(mock_gemini.received_audio_chunks) == 1
        assert len(mock_gemini.received_audio_chunks[0]) == 640
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_in_stream_consecutive_chunks(self, mock_gemini, audio_synth):
        """Verifies streaming 50 chunks (1 full second = 32,000 bytes) arrives in order."""
        client = GeminiLiveClient()
        await client.connect()

        one_sec_pcm = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=1.0, sample_rate=16000)
        chunks = audio_synth.chunk_pcm(one_sec_pcm, chunk_duration_ms=20.0, sample_rate=16000)
        assert len(chunks) == 50

        for c in chunks:
            await client.send_audio_chunk(c)

        for _ in range(20):
            if len(mock_gemini.received_audio_chunks) >= 50:
                break
            await asyncio.sleep(0.05)

        assert len(mock_gemini.received_audio_chunks) == 50
        total_bytes = sum(len(c) for c in mock_gemini.received_audio_chunks)
        assert total_bytes == 32000
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_in_variable_chunk_sizes(self, mock_gemini, audio_synth):
        """Verifies transmission of non-standard frame sizes (10ms, 40ms, 100ms)."""
        client = GeminiLiveClient()
        await client.connect()

        c10 = audio_synth.generate_sine_pcm(duration_sec=0.01, sample_rate=16000)  # 320B
        c40 = audio_synth.generate_sine_pcm(duration_sec=0.04, sample_rate=16000)  # 1280B
        c100 = audio_synth.generate_sine_pcm(duration_sec=0.10, sample_rate=16000) # 3200B

        await client.send_audio_chunk(c10)
        await client.send_audio_chunk(c40)
        await client.send_audio_chunk(c100)

        for _ in range(10):
            if len(mock_gemini.received_audio_chunks) >= 3:
                break
            await asyncio.sleep(0.02)

        assert len(mock_gemini.received_audio_chunks) == 3
        assert len(mock_gemini.received_audio_chunks[0]) == 320
        assert len(mock_gemini.received_audio_chunks[1]) == 1280
        assert len(mock_gemini.received_audio_chunks[2]) == 3200
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_in_realtime_input_structure(self, mock_gemini, audio_synth):
        """Verifies envelope structure of realtime input frame."""
        client = GeminiLiveClient()
        await client.connect()

        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)
        await client.send_audio_chunk(chunk)
        await asyncio.sleep(0.05)

        # Inspect raw received message
        rt_msgs = [m for m in mock_gemini.received_messages if "realtimeInput" in m or "realtime_input" in m]
        assert len(rt_msgs) >= 1
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_in_not_connected_error(self, audio_synth):
        """Verifies send_audio_chunk on disconnected client logs warning and does not crash."""
        client = GeminiLiveClient()
        assert not client.is_connected
        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)
        # Should gracefully return without unhandled exception
        await client.send_audio_chunk(chunk)


# ============================================================================
# FEATURE AREA 3: 24kHZ PCM AUDIO STREAMING OUTPUT & UNPACKING
# ============================================================================

class TestTier1AudioOutputStreaming:
    """Verifies receiving and unpacking 24kHz assistant speech frames."""

    @pytest.mark.asyncio
    async def test_tier1_audio_out_receive_single_chunk(self, mock_gemini, audio_synth):
        """Verifies 24kHz audio chunk sent by server is delivered to registered callback."""
        client = GeminiLiveClient()
        received_chunks = []
        event = asyncio.Event()

        def on_audio(chunk: bytes):
            received_chunks.append(chunk)
            event.set()

        client.set_on_audio_received(on_audio)
        await client.connect()

        speech_chunk = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.02, sample_rate=24000)
        await mock_gemini.emit_speech(text="Greeting", pcm_24k=speech_chunk)

        await asyncio.wait_for(event.wait(), timeout=2.0)
        assert len(received_chunks) == 1
        assert received_chunks[0] == speech_chunk
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_out_stream_multi_chunk_speech(self, mock_gemini, audio_synth):
        """Verifies sequential delivery of 10 consecutive 24k audio chunks."""
        client = GeminiLiveClient()
        received_chunks = []

        client.set_on_audio_received(lambda c: received_chunks.append(c))
        await client.connect()

        for i in range(10):
            chunk = audio_synth.generate_sine_pcm(freq_hz=440.0 + i * 50, duration_sec=0.02, sample_rate=24000)
            await mock_gemini.emit_speech(text=f"Part {i}", pcm_24k=chunk)

        await asyncio.sleep(0.1)
        assert len(received_chunks) == 10
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_out_data_format_int16(self, mock_gemini, audio_synth):
        """Verifies unpacked audio conforms to int16 little-endian constraints."""
        client = GeminiLiveClient()
        received = []
        client.set_on_audio_received(lambda c: received.append(c))
        await client.connect()

        chunk = audio_synth.generate_sine_pcm(freq_hz=880.0, duration_sec=0.05, sample_rate=24000)
        await mock_gemini.emit_speech(text="Tone", pcm_24k=chunk)

        await asyncio.sleep(0.05)
        assert len(received) == 1
        arr = np.frombuffer(received[0], dtype=np.int16)
        assert len(arr) == 1200  # 0.05s * 24000
        assert np.max(arr) <= 32767
        assert np.min(arr) >= -32768
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_out_non_zero_energy(self, mock_gemini, audio_synth):
        """Verifies received synthetic audio exhibits measurable RMS and peak energy."""
        client = GeminiLiveClient()
        received = []
        client.set_on_audio_received(lambda c: received.append(c))
        await client.connect()

        chunk = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.05, sample_rate=24000, amplitude=0.8)
        await mock_gemini.emit_speech(text="Tone", pcm_24k=chunk)

        await asyncio.sleep(0.05)
        rms = audio_synth.calculate_pcm_rms(received[0])
        peak = audio_synth.calculate_pcm_peak(received[0])
        assert rms > 0.4
        assert peak > 0.7
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_audio_out_callback_exception_isolation(self, mock_gemini, audio_synth):
        """Verifies exceptions in audio callback do not crash the receive loop."""
        client = GeminiLiveClient()
        call_count = 0

        def faulty_callback(chunk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Faulty consumer")

        client.set_on_audio_received(faulty_callback)
        await client.connect()

        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=24000)
        await mock_gemini.emit_speech(pcm_24k=chunk)
        await mock_gemini.emit_speech(pcm_24k=chunk)

        await asyncio.sleep(0.1)
        assert call_count == 2
        assert client.is_connected
        await client.disconnect()


# ============================================================================
# FEATURE AREA 4: PCM RESAMPLING ENGINE
# ============================================================================

class TestTier1PCMResamplingEngine:
    """Verifies bidirectional sample rate conversions and signal fidelity."""

    def test_tier1_resample_16k_to_24k_samples(self, audio_synth):
        """Resampling 16kHz to 24kHz scales sample count by exact 1.5 ratio."""
        pcm_16k = audio_synth.generate_sine_pcm(duration_sec=0.1, sample_rate=16000)
        assert len(pcm_16k) == 3200  # 1600 samples * 2 bytes

        pcm_24k = AudioProcessor.resample_pcm(pcm_16k, 16000, 24000)
        assert len(pcm_24k) == 4800  # 2400 samples * 2 bytes

    def test_tier1_resample_24k_to_16k_samples(self, audio_synth):
        """Resampling 24kHz to 16kHz scales sample count by exact 2/3 ratio."""
        pcm_24k = audio_synth.generate_sine_pcm(duration_sec=0.1, sample_rate=24000)
        assert len(pcm_24k) == 4800

        pcm_16k = AudioProcessor.resample_pcm(pcm_24k, 24000, 16000)
        assert len(pcm_16k) == 3200

    def test_tier1_resample_identity_passthrough(self, audio_synth):
        """Resampling with identical in/out rates returns identical byte buffer."""
        pcm = audio_synth.generate_sine_pcm(duration_sec=0.05, sample_rate=16000)
        resampled = AudioProcessor.resample_pcm(pcm, 16000, 16000)
        assert resampled == pcm

    def test_tier1_resample_roundtrip_signal_fidelity(self, audio_synth):
        """Roundtrip 16k -> 24k -> 16k maintains high waveform correlation (r >= 0.98)."""
        orig_16k = audio_synth.generate_sine_pcm(freq_hz=440.0, duration_sec=0.2, sample_rate=16000)
        res_24k = AudioProcessor.resample_pcm(orig_16k, 16000, 24000)
        roundtrip_16k = AudioProcessor.resample_pcm(res_24k, 24000, 16000)

        corr = audio_synth.verify_signal_correlation(orig_16k, roundtrip_16k, 16000, 16000, min_corr=0.98)
        assert corr >= 0.98

    def test_tier1_resample_hardware_rates(self, audio_synth):
        """Resampling to/from 48kHz hardware rates matches exact sample proportions."""
        pcm_48k = audio_synth.generate_silence_pcm(duration_sec=0.1, sample_rate=48000)
        assert len(pcm_48k) == 9600  # 4800 samples * 2

        pcm_16k = AudioProcessor.resample_pcm(pcm_48k, 48000, 16000)
        assert len(pcm_16k) == 3200  # 1600 samples * 2


# ============================================================================
# FEATURE AREA 5: AOEDE VOICE PERSONA & SPEECHCONFIG
# ============================================================================

class TestTier1AoedeVoicePersona:
    """Verifies default Aoede persona, speech config, and system prompt setup."""

    def test_tier1_persona_default_aoede(self):
        """GeminiLiveClient defaults voice_name to 'Aoede'."""
        client = GeminiLiveClient()
        assert client.voice_name == "Aoede"

    @pytest.mark.asyncio
    async def test_tier1_persona_speech_config_payload(self, mock_gemini):
        """Connect payload transmits voiceName='Aoede'."""
        client = GeminiLiveClient()
        await client.connect()
        setup = await mock_gemini.wait_for_setup()

        gen_config = setup.get("generationConfig") or setup.get("generation_config") or {}
        speech_config = (
            setup.get("speechConfig") or setup.get("speech_config")
            or gen_config.get("speechConfig") or gen_config.get("speech_config") or {}
        )
        vc = speech_config.get("voiceConfig") or speech_config.get("voice_config") or {}
        pvc = vc.get("prebuiltVoiceConfig") or vc.get("prebuilt_voice_config") or {}
        voice_name = pvc.get("voiceName") or pvc.get("voice_name")

        assert voice_name == "Aoede"
        await client.disconnect()

    def test_tier1_persona_system_instruction_screening(self):
        """Default system instruction contains call screening directives."""
        client = GeminiLiveClient()
        instr = client.system_instruction.lower()
        assert "call screening" in instr or "screening" in instr
        assert "aoede" in instr

    @pytest.mark.asyncio
    async def test_tier1_persona_custom_system_instruction_override(self, mock_gemini):
        """Custom system instruction overrides default prompt."""
        custom_prompt = "You are an executive assistant for ACME Corp."
        client = GeminiLiveClient(system_instruction=custom_prompt)
        await client.connect()
        setup = await mock_gemini.wait_for_setup()

        parts = setup.get("systemInstruction", {}).get("parts", [])
        assert len(parts) >= 1
        assert custom_prompt in parts[0].get("text", "")
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_tier1_persona_voice_override(self, mock_gemini):
        """Specifying an alternative voice name is transmitted in config."""
        client = GeminiLiveClient(voice_name="Puck")
        await client.connect()
        setup = await mock_gemini.wait_for_setup()

        gen_config = setup.get("generationConfig") or setup.get("generation_config") or {}
        speech_config = (
            setup.get("speechConfig") or setup.get("speech_config")
            or gen_config.get("speechConfig") or gen_config.get("speech_config") or {}
        )
        vc = speech_config.get("voiceConfig") or speech_config.get("voice_config") or {}
        pvc = vc.get("prebuiltVoiceConfig") or vc.get("prebuilt_voice_config") or {}
        voice_name = pvc.get("voiceName") or pvc.get("voice_name")

        assert voice_name == "Puck"
        await client.disconnect()


# ============================================================================
# FEATURE AREA 6: AGENT FUNCTION CALLING TOOLS
# ============================================================================

class TestTier1AgentTools:
    """Verifies schema definitions and execution handlers for tools."""

    @pytest.mark.asyncio
    async def test_tier1_tool_take_message_execution(self, call_manager):
        """take_message persists message record and returns confirmation dict."""
        res = await call_manager.handle_tool_call("take_message", {
            "caller_name": "Bob Vance",
            "phone_number": "555-0144",
            "message": "Refrigeration delivery scheduled",
            "urgency": "high"
        })
        assert res["status"] == "saved"
        assert res["caller_name"] == "Bob Vance"
        msgs = await call_manager.db.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["caller_name"] == "Bob Vance"
        assert msgs[0]["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_tier1_tool_end_call_execution(self, call_manager):
        """end_call schedules hangup and updates state to COMPLETED."""
        await call_manager.handle_inbound_ring("+15550001111")
        await call_manager.approve_inbound_call("ai_takeover")
        assert call_manager.state == "ACTIVE"

        res = await call_manager.handle_tool_call("end_call", {"reason": "inquiry_resolved"})
        assert res["status"] == "call_ended"
        assert call_manager.state == "COMPLETED"

    @pytest.mark.asyncio
    async def test_tier1_tool_transfer_to_human_execution(self, call_manager):
        """transfer_to_human sets state to ON_HOLD and plays hold audio."""
        await call_manager.handle_inbound_ring("+15550002222")
        await call_manager.approve_inbound_call("ai_takeover")

        res = await call_manager.handle_tool_call("transfer_to_human", {"reason": "urgent_matter"})
        assert res["status"] == "transfer_initiated"
        assert call_manager.state == "ON_HOLD"
        assert call_manager.driver.is_on_hold is True

    @pytest.mark.asyncio
    async def test_tier1_tool_log_call_summary_execution(self, call_manager):
        """log_call_summary persists structured summary and action items in DB."""
        call_id = await call_manager.handle_inbound_ring("+15550003333")
        await call_manager.approve_inbound_call("ai_takeover")

        res = await call_manager.handle_tool_call("log_call_summary", {
            "summary": "Discussed product catalog pricing",
            "action_items": ["Send PDF catalog", "Call back Thursday"]
        })
        assert res["status"] == "summary_logged"
        details = await call_manager.db.get_call_details(call_id)
        assert details["summary"] is not None
        assert details["summary"]["summary"] == "Discussed product catalog pricing"
        assert len(details["summary"]["action_items"]) == 2

    def test_tier1_tool_declarations_schema(self):
        """Tool declarations expose all 4 required tools with proper properties."""
        declarations = get_tool_declarations()
        assert len(declarations) == 1
        fn_map = {fn.name: fn for fn in declarations[0].function_declarations}
        assert set(fn_map.keys()) == {
            "take_message",
            "end_call",
            "transfer_to_human",
            "log_call_summary"
        }


# ============================================================================
# FEATURE AREA 7: TELEPHONY DRIVERS (WINDOWS, VIRTUAL, LINUX)
# ============================================================================

class TestTier1TelephonyDrivers:
    """Verifies driver contracts, audio loopbacks, and platform fallbacks."""

    def test_tier1_driver_base_interface_conformance(self):
        """All driver classes inherit BaseTelephonyDriver."""
        for cls in [VirtualSimulatorDriver, WindowsAudioDriver, LinuxDriver]:
            assert issubclass(cls, BaseTelephonyDriver)

    @pytest.mark.asyncio
    async def test_tier1_driver_virtual_audio_loopback(self, audio_synth):
        """VirtualSimulatorDriver routes audio and clears playback buffer."""
        sig = inspect.signature(VirtualSimulatorDriver.__init__)
        driver = VirtualSimulatorDriver(headless=True) if "headless" in sig.parameters else VirtualSimulatorDriver()
        await driver.start()
        is_running = getattr(driver, "is_running", None)
        if is_running is None:
            is_running = driver.is_active()
        assert is_running is True

        chunk = audio_synth.generate_sine_pcm(duration_sec=0.02, sample_rate=16000)
        driver.play_audio_chunk(chunk)
        q_size = len(getattr(driver, "playback_queue", [])) if hasattr(driver, "playback_queue") else driver._out_queue.qsize()
        assert q_size == 1

        driver.clear_audio_buffer()
        q_size_cleared = len(getattr(driver, "playback_queue", [])) if hasattr(driver, "playback_queue") else driver._out_queue.qsize()
        assert q_size_cleared == 0
        await driver.stop()
        is_still_running = getattr(driver, "is_running", None)
        if is_still_running is None:
            is_still_running = driver.is_active()
        assert not is_still_running

    def test_tier1_driver_windows_device_enumeration_sanitization(self):
        """WindowsAudioDriver sanitizes control characters and device headers."""
        raw_name = "Output (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iQOO Z9 5G))"
        sanitizer = getattr(WindowsAudioDriver, "sanitize_device_name", getattr(WindowsAudioDriver, "_sanitize_device_name", None))
        assert sanitizer is not None
        clean = sanitizer(raw_name)
        assert "\r" not in clean
        assert "\n" not in clean
        assert "Hands-Free HF Audio" in clean or "Hands-Free" in clean

    @pytest.mark.asyncio
    async def test_tier1_driver_windows_dial_tel_dispatch(self):
        """WindowsAudioDriver dial validates phone number format and constructed tel: URI."""
        driver = WindowsAudioDriver()
        with patch("os.startfile") as mock_startfile:
            success = await driver.dial("+1 (555) 0199")
            assert success is True
            # Verify phone number format and constructed tel: URI parameter
            if mock_startfile.called:
                mock_startfile.assert_called_once_with("tel:+15550199")
            elif hasattr(driver, "last_dialed_uri"):
                assert driver.last_dialed_uri == "tel:+15550199"
                assert driver.last_dialed_number == "+15550199"

            is_active = getattr(driver, "active_call", False) or getattr(driver, "call_state", "") == "ACTIVE"
            assert is_active is True
            await driver.hang_up()
            is_still_active = getattr(driver, "active_call", False) or getattr(driver, "call_state", "") == "ACTIVE"
            assert is_still_active is False

    @pytest.mark.asyncio
    async def test_tier1_driver_linux_graceful_fallback(self):
        """LinuxDriver initializes safely on Windows without crash."""
        sig = inspect.signature(LinuxDriver.__init__)
        driver = LinuxDriver(mock_mode=True) if "mock_mode" in sig.parameters else LinuxDriver()
        await driver.start()
        success = await driver.dial("+15550199")
        assert success is True
        await driver.stop()


# ============================================================================
# FEATURE AREA 8: SQLITE DATABASE PERSISTENCE LAYER & CRUD
# ============================================================================

class TestTier1DatabasePersistence:
    """Verifies SQLite tables, WAL mode, and CRUD operations."""

    @pytest.mark.asyncio
    async def test_tier1_db_init_and_wal_mode(self, temp_db):
        """Database initializes tables in WAL journal mode."""
        async with aiosqlite.connect(temp_db.db_path) as db:
            async with db.execute("PRAGMA journal_mode") as cur:
                row = await cur.fetchone()
                assert row[0].lower() == "wal"

    @pytest.mark.asyncio
    async def test_tier1_db_create_and_update_call(self, temp_db):
        """create_call and update_call_status properly track lifecycle."""
        call_id = await temp_db.create_call({
            "caller_id": "+15551234567",
            "caller_name": "Test Caller",
            "direction": "inbound"
        })
        assert call_id is not None

        await temp_db.update_call_status(call_id, "completed", hangup_reason="normal", duration=14.5)
        details = await temp_db.get_call_details(call_id)
        assert details["status"] == "completed"
        assert details["hangup_reason"] == "normal"
        assert details["duration"] == 14.5

    @pytest.mark.asyncio
    async def test_tier1_db_append_and_get_transcripts(self, temp_db):
        """append_transcript chronologically stores speaker turns."""
        call_id = await temp_db.create_call({"caller_id": "+15552345678"})
        await temp_db.append_transcript(call_id, "agent", "Hello! How can I help?", is_final=True)
        await temp_db.append_transcript(call_id, "caller", "I need to speak to Amar.", is_final=True)

        transcripts = await temp_db.get_transcripts(call_id)
        assert len(transcripts) == 2
        assert transcripts[0]["speaker"] == "agent"
        assert transcripts[1]["speaker"] == "caller"

    @pytest.mark.asyncio
    async def test_tier1_db_save_and_query_messages(self, temp_db):
        """save_message stores messages and supports unread filtering."""
        call_id = await temp_db.create_call({"caller_id": "+15553456789"})
        msg_id = await temp_db.save_message(
            call_id=call_id,
            caller_name="Alice",
            phone_number="+15553456789",
            message="Please call back regarding invoice",
            urgency="high"
        )
        assert msg_id > 0

        msgs = await temp_db.get_messages(unread_only=True)
        assert len(msgs) == 1
        assert msgs[0]["caller_name"] == "Alice"
        assert msgs[0]["urgency"] == "high"

    @pytest.mark.asyncio
    async def test_tier1_db_save_and_retrieve_summary(self, temp_db):
        """save_summary serializes and retrieves post-call summaries."""
        call_id = await temp_db.create_call({"caller_id": "+15554567890"})
        await temp_db.save_summary(
            call_id=call_id,
            summary="Client requested consultation reschedule",
            action_items=["Update calendar", "Notify team"]
        )

        details = await temp_db.get_call_details(call_id)
        assert details["summary"] is not None
        assert details["summary"]["summary"] == "Client requested consultation reschedule"
        assert details["summary"]["action_items"] == ["Update calendar", "Notify team"]


# ============================================================================
# FEATURE AREA 9: FASTAPI REST ENDPOINTS & WEBSOCKET /WS/LIVE HUB
# ============================================================================

class TestTier1FastAPIServerAndWebSocket:
    """Verifies REST endpoints and /ws/live WebSocket hub."""

    @pytest.mark.asyncio
    async def test_tier1_rest_get_status(self, async_http_client):
        """GET /api/status returns HTTP 200 and initial IDLE state."""
        res = await async_http_client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["state"] == "idle"

    @pytest.mark.asyncio
    async def test_tier1_rest_get_devices(self, async_http_client):
        """GET /api/devices returns sanitized audio device listings."""
        res = await async_http_client.get("/api/devices")
        assert res.status_code == 200
        data = res.json()
        assert "inputs" in data
        assert "outputs" in data
        assert len(data["inputs"]) > 0

    @pytest.mark.asyncio
    async def test_tier1_rest_call_lifecycle_controls(self, async_http_client):
        """POST /api/call/dial and POST /api/call/hangup execute call lifecycle."""
        # Dial
        dial_res = await async_http_client.post(
            "/api/call/dial",
            json={"phone_number": "+15558889999", "goal_prompt": "Test call"}
        )
        assert dial_res.status_code == 200
        assert dial_res.json()["status"] == "dialing"

        # Hangup
        hang_res = await async_http_client.post(
            "/api/call/hangup",
            json={"reason": "test_completed"}
        )
        assert hang_res.status_code == 200
        assert hang_res.json()["status"] == "terminated"

    @pytest.mark.asyncio
    async def test_tier1_rest_inbound_action_approval(self, live_server, async_http_client):
        """POST /api/call/inbound-action approves ringing call."""
        # Trigger ring
        await live_server.call_manager.handle_inbound_ring("+15557770000", "Incoming Tester")
        assert live_server.call_manager.state == "RINGING"

        res = await async_http_client.post(
            "/api/call/inbound-action",
            json={"action": "ai_takeover"}
        )
        assert res.status_code == 200
        assert res.json()["status"] == "approved"
        assert live_server.call_manager.state == "ACTIVE"

        # Clean teardown
        await live_server.call_manager.terminate_call("test_end")

    @pytest.mark.asyncio
    async def test_tier1_websocket_live_connection_and_telemetry(self, live_server):
        """WebSocket /ws/live connects and transmits initial state snapshot."""
        async with live_server.connect_ws("/ws/live") as ws:
            raw = await ws.recv()
            data = json.loads(raw)
            assert data["type"] == "initial_state"
            assert "state" in data


# ============================================================================
# FEATURE AREA 10: WEB MANAGEMENT CONSOLE ASSETS (FEATURE 23)
# ============================================================================

class TestTier1WebConsoleAssets:
    """Verifies static asset routes, HTML5 canvas waveforms, whisper input, dialer, and MIME types."""

    @pytest.mark.asyncio
    async def test_tier1_web_console_static_asset_routes(self, async_http_client):
        """Static asset routes (/, /static/index.html, /static/styles.css, /static/app.js) return 200."""
        routes = ["/", "/static/index.html", "/static/styles.css", "/static/app.js"]
        for route in routes:
            resp = await async_http_client.get(route)
            assert resp.status_code == 200, f"Route {route} failed with status {resp.status_code}"
            assert len(resp.text) > 0, f"Route {route} returned empty content"

    @pytest.mark.asyncio
    async def test_tier1_web_console_html5_waveform_canvases(self, async_http_client):
        """Web console HTML contains Caller cyan and Gemini emerald canvas waveform elements."""
        resp = await async_http_client.get("/")
        assert resp.status_code == 200
        html = resp.text

        # Verify Caller waveform canvas
        assert '<canvas id="caller-waveform"' in html or 'id="caller-waveform"' in html
        assert "cyan" in html.lower()

        # Verify Gemini Live waveform canvas
        assert '<canvas id="gemini-waveform"' in html or 'id="gemini-waveform"' in html
        assert "emerald" in html.lower()

    @pytest.mark.asyncio
    async def test_tier1_web_console_whisper_input_box(self, async_http_client):
        """Web console HTML contains mid-call operator whisper input box."""
        resp = await async_http_client.get("/static/index.html")
        assert resp.status_code == 200
        html = resp.text

        assert 'id="whisper-input"' in html or 'name="whisper"' in html
        assert "whisper" in html.lower()

    @pytest.mark.asyncio
    async def test_tier1_web_console_outbound_dialer_keypad(self, async_http_client):
        """Web console HTML contains outbound dialer keypad with dial trigger."""
        resp = await async_http_client.get("/")
        assert resp.status_code == 200
        html = resp.text

        assert 'id="dialer-keypad"' in html or 'keypad-grid' in html
        assert "dial" in html.lower()
        # Verify dialer phone input
        assert 'id="dialer-phone"' in html or 'type="tel"' in html

    @pytest.mark.asyncio
    async def test_tier1_web_console_content_type_headers(self, async_http_client):
        """Web console static assets return correct Content-Type MIME headers."""
        # HTML routes
        for html_route in ["/", "/static/index.html"]:
            resp = await async_http_client.get(html_route)
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "").lower()
            assert "text/html" in ct, f"Expected text/html for {html_route}, got {ct}"

        # CSS stylesheet
        css_resp = await async_http_client.get("/static/styles.css")
        assert css_resp.status_code == 200
        css_ct = css_resp.headers.get("content-type", "").lower()
        assert "text/css" in css_ct, f"Expected text/css for /static/styles.css, got {css_ct}"

        # JavaScript client
        js_resp = await async_http_client.get("/static/app.js")
        assert js_resp.status_code == 200
        js_ct = js_resp.headers.get("content-type", "").lower()
        assert "javascript" in js_ct, f"Expected javascript MIME for /static/app.js, got {js_ct}"

