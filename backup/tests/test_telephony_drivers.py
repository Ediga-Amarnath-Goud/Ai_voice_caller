"""
Unit Tests for Telephony Drivers Layer:
- BaseTelephonyDriver interface contracts & callback dispatching.
- telephony package exports.
- VirtualSimulatorDriver simulated lifecycle, audio routing, barge-in, resampling, headless mode.
- WindowsAudioDriver dynamic regex device discovery, tel: dialing, sounddevice streaming, resampling.
- LinuxDriver cross-platform import safety, automatic mock fallback, and oFono D-Bus execution.
- Concurrency, boundary conditions, and edge-case handling.
"""

import asyncio
import os
import queue
import re
import sys
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.audio_processor import AudioProcessor
import telephony
from telephony.base_driver import BaseTelephonyDriver
from telephony.virtual_driver import VirtualSimulatorDriver
from telephony.windows_driver import WindowsAudioDriver, DEFAULT_HFP_REGEX_PATTERNS
from telephony.linux_driver import LinuxDriver


# ============================================================================
# 1. BaseTelephonyDriver & Package Export Tests
# ============================================================================

class TestBaseDriverAndExports:
    def test_package_exports(self):
        """Verify that telephony/__init__.py exports all required driver classes."""
        assert hasattr(telephony, "BaseTelephonyDriver")
        assert hasattr(telephony, "WindowsAudioDriver")
        assert hasattr(telephony, "VirtualSimulatorDriver")
        assert hasattr(telephony, "LinuxDriver")
        assert telephony.BaseTelephonyDriver is BaseTelephonyDriver
        assert telephony.WindowsAudioDriver is WindowsAudioDriver
        assert telephony.VirtualSimulatorDriver is VirtualSimulatorDriver
        assert telephony.LinuxDriver is LinuxDriver

    def test_base_driver_cannot_be_instantiated_directly(self):
        """Verify that BaseTelephonyDriver is an ABC and cannot be directly instantiated."""
        with pytest.raises(TypeError) as excinfo:
            BaseTelephonyDriver()  # type: ignore
        assert "Can't instantiate abstract class" in str(excinfo.value)

    def test_concrete_subclass_must_implement_all_abstract_methods(self):
        """Verify that missing any abstract method prevents subclass instantiation."""
        class IncompleteDriver(BaseTelephonyDriver):
            async def start(self) -> None:
                pass

        with pytest.raises(TypeError) as excinfo:
            IncompleteDriver()  # type: ignore
        assert "Can't instantiate abstract class" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_base_driver_callback_dispatch_sync_and_async(self):
        """Verify _dispatch_callback safely handles both sync callables and async coroutines."""
        driver = VirtualSimulatorDriver(headless=True)
        driver._loop = asyncio.get_running_loop()

        sync_called = []
        def sync_cb(msg: str):
            sync_called.append(msg)

        async_called = []
        async def async_cb(msg: str):
            async_called.append(msg)

        driver._dispatch_callback(sync_cb, "hello_sync")
        assert sync_called == ["hello_sync"]

        driver._dispatch_callback(async_cb, "hello_async")
        # Allow loop to cycle to execute the coroutine
        await asyncio.sleep(0.01)
        assert async_called == ["hello_async"]

    def test_base_driver_dispatch_callback_handles_exceptions_gracefully(self):
        """Verify _dispatch_callback catches callback errors without raising to caller."""
        driver = VirtualSimulatorDriver(headless=True)
        def faulty_cb():
            raise RuntimeError("Callback explosion")

        # Must not raise
        driver._dispatch_callback(faulty_cb)


# ============================================================================
# 2. VirtualSimulatorDriver Tests
# ============================================================================

class TestVirtualSimulatorDriver:
    @pytest.mark.asyncio
    async def test_virtual_driver_lifecycle(self):
        """Verify VirtualSimulatorDriver starts, reports active, and stops cleanly."""
        driver = VirtualSimulatorDriver(headless=True)
        assert not driver.is_active()
        assert driver.call_state == "IDLE"

        await driver.start()
        assert driver.is_active()

        # Idempotent start
        await driver.start()
        assert driver.is_active()

        await driver.stop()
        assert not driver.is_active()
        assert driver.call_state == "IDLE"

        # Idempotent stop
        await driver.stop()
        assert not driver.is_active()

    @pytest.mark.asyncio
    async def test_virtual_driver_dialing_and_state_transitions(self):
        """Verify simulated outbound dialing, state callbacks, and hang up."""
        driver = VirtualSimulatorDriver(headless=True)
        await driver.start()

        state_history = []
        def on_state(state, details):
            state_history.append((state, details))

        driver.set_audio_callbacks(on_audio_in=lambda pcm: None, on_state_change=on_state)

        # Dial valid phone number
        success = await driver.dial("+1 (555) 987-6543")
        assert success is True
        assert driver.call_state == "ACTIVE"
        assert len(state_history) == 2
        assert state_history[0][0] == "DIALING"
        assert state_history[0][1]["phone_number"] == "+15559876543"
        assert state_history[1][0] == "ACTIVE"

        # Hang up
        hang_success = await driver.hang_up(reason="call_finished")
        assert hang_success is True
        assert driver.call_state == "COMPLETED"
        assert state_history[-1][0] == "COMPLETED"
        assert state_history[-1][1]["reason"] == "call_finished"

        await driver.stop()

    @pytest.mark.asyncio
    async def test_virtual_driver_invalid_phone_number_dial(self):
        """Verify dialing rejection on invalid, empty, or non-numeric inputs."""
        driver = VirtualSimulatorDriver(headless=True)
        await driver.start()

        assert await driver.dial("") is False
        assert await driver.dial("   ") is False
        assert await driver.dial("letters_only") is False
        assert await driver.dial(None) is False  # type: ignore

        await driver.stop()

    @pytest.mark.asyncio
    async def test_virtual_driver_answer_call(self):
        """Verify answering simulated incoming calls."""
        driver = VirtualSimulatorDriver(headless=True)
        await driver.start()

        answered = await driver.answer_call()
        assert answered is True
        assert driver.call_state == "ACTIVE"

        await driver.stop()

    def test_virtual_driver_playback_queue_and_barge_in(self):
        """Verify audio chunk queueing, status reporting, and immediate buffer flush."""
        driver = VirtualSimulatorDriver(headless=True)
        assert driver.get_status()["queue_depth"] == 0

        # Create 20ms chunk of 24kHz audio (480 samples = 960 bytes)
        chunk = b"\x01\x00" * 480
        driver.play_audio_chunk(chunk, sample_rate=24000)
        driver.play_audio_chunk(chunk, sample_rate=24000)

        status = driver.get_status()
        assert status["queue_depth"] == 2
        assert status["driver"] == "VirtualSimulatorDriver"

        # Barge-in: flush buffer
        driver.clear_audio_buffer()
        assert driver.get_status()["queue_depth"] == 0

    def test_virtual_driver_simulate_inbound_audio(self):
        """Verify test helper injects inbound audio into registered callback."""
        driver = VirtualSimulatorDriver(headless=True)
        received_chunks = []

        driver.set_audio_callbacks(on_audio_in=lambda pcm: received_chunks.append(pcm))

        dummy_pcm = b"\x20\x00" * 160
        driver.simulate_inbound_audio(dummy_pcm)
        assert len(received_chunks) == 1
        assert received_chunks[0] == dummy_pcm

    @pytest.mark.asyncio
    async def test_virtual_driver_resampling_on_playback(self):
        """Verify outbound audio resamples when input sample_rate != output_sample_rate."""
        driver = VirtualSimulatorDriver(output_sample_rate=16000, headless=True)
        # Play 24kHz audio (480 samples = 960 bytes, 20ms)
        chunk_24k = b"\x10\x00" * 480
        driver.play_audio_chunk(chunk_24k, sample_rate=24000)

        # The queued chunk should be resampled to 16kHz (320 samples = 640 bytes)
        assert driver._out_queue.qsize() == 1
        queued_chunk = driver._out_queue.get_nowait()
        assert len(queued_chunk) == 640


# ============================================================================
# 3. WindowsAudioDriver Tests
# ============================================================================

class TestWindowsAudioDriver:
    @pytest.fixture
    def mock_devices(self):
        """Mock list of sounddevice devices including Bluetooth HFP and default sound devices."""
        return [
            {
                "name": "Microsoft Sound Mapper - Input",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Microphone Array (AMD Audio Device)",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Microsoft Sound Mapper - Output",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Speakers (Realtek(R) Audio)",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Output (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iQOO Z9 5G))",
                "hostapi": 3,
                "max_input_channels": 0,
                "max_output_channels": 1,
                "default_samplerate": 16000.0,
            },
            {
                "name": "Input (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iQOO Z9 5G))",
                "hostapi": 3,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            },
        ]

    def test_device_name_sanitizer(self):
        """Verify _sanitize_device_name removes CRLF from Windows PortAudio names."""
        raw = "Output (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iQOO Z9 5G))\r\n"
        cleaned = WindowsAudioDriver._sanitize_device_name(raw)
        assert "\r" not in cleaned
        assert "\n" not in cleaned
        assert "Hands-Free HF Audio" in cleaned

    def test_resolve_devices_with_bluetooth_hfp(self, mock_devices):
        """Verify dynamic regex discovery matches iQOO Z9 5G / Hands-Free endpoints."""
        with patch("sounddevice.query_devices", return_value=mock_devices):
            driver = WindowsAudioDriver()
            in_idx, out_idx = driver.resolve_devices()

            assert in_idx == 5  # Index 5 is Input Hands-Free HF Audio
            assert out_idx == 4  # Index 4 is Output Hands-Free HF Audio
            assert "Hands-Free HF Audio" in driver._input_device_name
            assert "Hands-Free HF Audio" in driver._output_device_name

    def test_resolve_devices_with_custom_regex(self, mock_devices):
        """Verify custom device_regex pattern matches targeted device."""
        with patch("sounddevice.query_devices", return_value=mock_devices):
            driver = WindowsAudioDriver(device_regex=r"AMD Audio")
            in_idx, out_idx = driver.resolve_devices()

            assert in_idx == 1  # AMD Audio Device
            # Out_idx falls back to default HFP pattern since AMD is input-only
            assert out_idx == 4

    def test_resolve_devices_fallback_to_defaults(self):
        """Verify fallback to default devices if no Bluetooth HFP devices exist."""
        plain_devices = [
            {"name": "Mic 1", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0},
            {"name": "Speaker 1", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
        ]
        with patch("sounddevice.query_devices", return_value=plain_devices), \
             patch("sounddevice.default.device", [0, 1]):
            driver = WindowsAudioDriver()
            in_idx, out_idx = driver.resolve_devices()
            assert in_idx == 0
            assert out_idx == 1

    @pytest.mark.asyncio
    async def test_windows_driver_dial_sanitization_and_execution(self):
        """Verify dial sanitizes number and executes os.startfile with tel: protocol."""
        driver = WindowsAudioDriver()
        with patch("os.startfile") as mock_startfile:
            success = await driver.dial("+1 (800) 555-0199")
            assert success is True
            mock_startfile.assert_called_once_with("tel:+18005550199")
            assert driver.call_state == "ACTIVE"

    @pytest.mark.asyncio
    async def test_windows_driver_dial_subprocess_fallback(self):
        """Verify fallback to cmd /c start tel: when os.startfile raises an error."""
        driver = WindowsAudioDriver()
        with patch("os.startfile", side_effect=OSError("No association")), \
             patch("subprocess.run") as mock_subproc:
            success = await driver.dial("+91 98765 43210")
            assert success is True
            mock_subproc.assert_called_once_with(["cmd", "/c", "start", "", "tel:+919876543210"], shell=True, check=False)
            assert driver.call_state == "ACTIVE"

    @pytest.mark.asyncio
    async def test_windows_driver_dial_invalid_number(self):
        """Verify dial returns False on invalid input without executing OS command."""
        driver = WindowsAudioDriver()
        with patch("os.startfile") as mock_startfile:
            assert await driver.dial("") is False
            assert await driver.dial("abc") is False
            mock_startfile.assert_not_called()

    @pytest.mark.asyncio
    async def test_windows_driver_answer_and_hangup(self):
        """Verify answer_call and hang_up state management and buffer flush."""
        driver = WindowsAudioDriver()
        await driver.answer_call()
        assert driver.call_state == "ACTIVE"

        driver.play_audio_chunk(b"\x01\x00" * 320, sample_rate=16000)
        assert driver._out_queue.qsize() == 1

        await driver.hang_up(reason="operator_hangup")
        assert driver.call_state == "COMPLETED"
        assert driver._out_queue.qsize() == 0

    def test_windows_driver_outbound_resampling_24k_to_16k(self):
        """Verify Gemini Live 24kHz speech is downsampled to 16kHz for Bluetooth HFP output."""
        driver = WindowsAudioDriver(output_sample_rate=16000)
        # 20ms of 24kHz audio = 480 samples = 960 bytes
        chunk_24k = b"\x05\x00" * 480
        driver.play_audio_chunk(chunk_24k, sample_rate=24000)

        assert driver._out_queue.qsize() == 1
        queued = driver._out_queue.get_nowait()
        # 16kHz audio for 20ms = 320 samples = 640 bytes
        assert len(queued) == 640

    def test_windows_driver_inbound_stream_callback_with_resampling(self):
        """Verify _in_stream_callback resamples input to 16kHz before dispatching to on_audio_in."""
        driver = WindowsAudioDriver(input_sample_rate=48000)
        received = []
        driver.set_audio_callbacks(on_audio_in=lambda pcm: received.append(pcm))

        # 20ms of 48kHz audio = 960 samples = 1920 bytes
        chunk_48k = b"\x10\x00" * 960
        driver._in_stream_callback(indata=chunk_48k, frames=960, time_info=None, status=None)

        assert len(received) == 1
        # Resampled output must be 16kHz = 320 samples = 640 bytes
        assert len(received[0]) == 640

    def test_windows_driver_out_stream_callback_staging_and_silence(self):
        """Verify _out_stream_callback correctly pulls from queue, handles staging remainder and silence."""
        driver = WindowsAudioDriver(output_sample_rate=16000)

        # Enqueue 40ms of 16kHz audio = 640 samples = 1280 bytes
        driver.play_audio_chunk(b"\x20\x00" * 640, sample_rate=16000)

        # PortAudio asks for 20ms (320 samples = 640 bytes)
        outdata = bytearray(640)
        driver._out_stream_callback(outdata, frames=320, time_info=None, status=None)
        assert bytes(outdata) == b"\x20\x00" * 320
        # Staging buffer should have the remaining 640 bytes
        assert len(driver._playback_staging) == 640

        # Next callback pulls remaining 640 bytes
        outdata2 = bytearray(640)
        driver._out_stream_callback(outdata2, frames=320, time_info=None, status=None)
        assert bytes(outdata2) == b"\x20\x00" * 320
        assert len(driver._playback_staging) == 0

        # When queue is empty, fills with silence
        outdata3 = bytearray(640)
        driver._out_stream_callback(outdata3, frames=320, time_info=None, status=None)
        assert bytes(outdata3) == b"\x00" * 640

    def test_windows_driver_barge_in_clears_queue_and_staging(self):
        """Verify clear_audio_buffer flushes both queued items and staging buffer instantly."""
        driver = WindowsAudioDriver(output_sample_rate=16000)
        driver.play_audio_chunk(b"\x11\x00" * 1000, sample_rate=16000)
        driver._playback_staging.extend(b"\x22\x00" * 500)

        assert driver._out_queue.qsize() == 1
        assert len(driver._playback_staging) == 1000

        driver.clear_audio_buffer()
        assert driver._out_queue.qsize() == 0
        assert len(driver._playback_staging) == 0

    @pytest.mark.asyncio
    async def test_windows_driver_stream_start_error_handling(self, mock_devices):
        """Verify PortAudio exception during stream creation is handled gracefully."""
        with patch("sounddevice.query_devices", return_value=mock_devices), \
             patch("sounddevice.RawInputStream", side_effect=RuntimeError("PortAudio Device Busy")):
            driver = WindowsAudioDriver()
            with pytest.raises(RuntimeError):
                await driver.start()
            assert not driver.is_active()


# ============================================================================
# 4. LinuxDriver Tests
# ============================================================================

class TestLinuxDriver:
    def test_linux_driver_import_safety_and_mock_detection(self):
        """Verify LinuxDriver initializes safely on Windows/non-Linux without D-Bus."""
        driver = LinuxDriver()
        assert driver.mock_mode is True
        assert driver.driver_name == "LinuxDriver"
        status = driver.get_status()
        assert status["mock_mode"] is True
        assert status["call_state"] == "IDLE"

    @pytest.mark.asyncio
    async def test_linux_driver_mock_lifecycle_and_dialing(self):
        """Verify mock mode full lifecycle, dialing, answering, and hang up."""
        driver = LinuxDriver(mock_mode=True)
        await driver.start()
        assert driver.is_active()

        state_history = []
        driver.set_audio_callbacks(
            on_audio_in=lambda pcm: None,
            on_state_change=lambda state, details: state_history.append((state, details)),
        )

        success = await driver.dial("+44 7911 123456")
        assert success is True
        assert driver.call_state == "ACTIVE"

        await driver.answer_call()
        assert driver.call_state == "ACTIVE"

        await driver.hang_up(reason="normal")
        assert driver.call_state == "COMPLETED"

        await driver.stop()
        assert not driver.is_active()
        assert driver.call_state == "IDLE"

    def test_linux_driver_audio_and_barge_in_in_mock_mode(self):
        """Verify mock mode handles play_audio_chunk and barge-in clear_audio_buffer."""
        driver = LinuxDriver(mock_mode=True, sample_rate=16000)
        chunk_24k = b"\x30\x00" * 480
        driver.play_audio_chunk(chunk_24k, sample_rate=24000)

        assert driver._out_queue.qsize() == 1
        driver.clear_audio_buffer()
        assert driver._out_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_linux_driver_dbus_execution_paths(self):
        """Verify Linux D-Bus mode execution when mocked D-Bus objects are present."""
        driver = LinuxDriver(mock_mode=False, modem_path="/modem0")
        driver.mock_mode = False  # Explicitly force mock_mode to False

        mock_vcm = MagicMock()
        mock_vcm.Dial.return_value = "/call0"
        mock_call = MagicMock()

        driver._vcm = mock_vcm
        driver._active_call = mock_call

        # Test Dial via oFono VoiceCallManager
        success = await driver.dial("+15551234567")
        assert success is True
        mock_vcm.Dial.assert_called_once_with("+15551234567", "default")
        assert driver._active_call_path == "/call0"

        # Test Answer via oFono VoiceCall
        assert await driver.answer_call() is True
        mock_call.Answer.assert_called_once()

        # Test Hangup via oFono VoiceCall
        assert await driver.hang_up(reason="user_ended") is True
        mock_call.Hangup.assert_called_once()
        assert driver._active_call is None


# ============================================================================
# 5. Concurrency, Resampling & Boundary Condition Tests
# ============================================================================

class TestConcurrencyAndBoundaries:
    def test_concurrent_audio_playback_and_clear_buffer(self):
        """Verify thread safety during high-frequency concurrent play and clear operations."""
        driver = VirtualSimulatorDriver(headless=True)
        stop_event = threading.Event()
        errors = []

        def worker_play():
            chunk = b"\x01\x00" * 160
            while not stop_event.is_set():
                try:
                    driver.play_audio_chunk(chunk)
                except Exception as e:
                    errors.append(e)

        def worker_clear():
            while not stop_event.is_set():
                try:
                    driver.clear_audio_buffer()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=worker_play),
            threading.Thread(target=worker_play),
            threading.Thread(target=worker_clear),
        ]
        for t in threads:
            t.start()

        import time
        time.sleep(0.15)
        stop_event.set()

        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_audio_chunk_odd_byte_length_truncation(self):
        """Verify odd-length byte chunks are safely truncated to even sample alignment."""
        driver = VirtualSimulatorDriver(headless=True)
        odd_chunk = b"\x01\x02\x03"
        # Must not raise
        driver.play_audio_chunk(odd_chunk)
        assert driver._out_queue.qsize() == 1
        queued = driver._out_queue.get_nowait()
        assert len(queued) % 2 == 0

    def test_empty_audio_chunk_ignored(self):
        """Verify empty audio chunk is safely ignored without enqueueing."""
        driver = VirtualSimulatorDriver(headless=True)
        driver.play_audio_chunk(b"")
        assert driver._out_queue.qsize() == 0

    def test_clear_buffer_on_empty_queue_is_safe(self):
        """Verify clear_audio_buffer on empty queue does not raise."""
        driver = WindowsAudioDriver()
        driver.clear_audio_buffer()
        assert driver.get_status()["queue_depth"] == 0
