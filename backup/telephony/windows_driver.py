"""
Windows Audio Telephony Driver for AI Voice Calling Assistant.
Routes audio between phone call endpoints (Bluetooth HFP / Hands-Free Profile,
Windows communication devices, or Virtual Audio Cable) and the Gemini Live engine.
Supports dynamic regex device discovery, sample rate resampling, barge-in buffer
flushing, and tel: protocol outbound dialing.
"""

import asyncio
import logging
import os
import queue
import re
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False

from core.audio_processor import AudioProcessor
from telephony.base_driver import BaseTelephonyDriver

logger = logging.getLogger("telephony.windows")

DEFAULT_HFP_REGEX_PATTERNS = [
    r"Virtual.*AudioRelay",
    r"AudioRelay",
    r"Hands-Free.*iQOO",
    r"iQOO.*Hands-Free",
    r"iQOO",
    r"Hands-Free.*Audio",
    r"Hands-Free",
    r"Handsfree",
    r"bthhfenum",
]


class WindowsAudioDriver(BaseTelephonyDriver):
    """
    Telephony driver for Windows 11 using sounddevice / PortAudio.
    Interacts with Bluetooth Hands-Free Profile (HFP/SCO) endpoints.
    """

    def __init__(
        self,
        device_regex: Optional[str] = None,
        input_device: Optional[Union[int, str]] = None,
        output_device: Optional[Union[int, str]] = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.device_regex = device_regex
        self.preferred_input_device = input_device
        self.preferred_output_device = output_device
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        self._input_device_idx: Optional[int] = None
        self._output_device_idx: Optional[int] = None
        self._input_device_name: str = "Unresolved"
        self._output_device_name: str = "Unresolved"

        self._in_stream: Optional[Any] = None
        self._out_stream: Optional[Any] = None
        self._is_active: bool = False

        self._out_queue: queue.Queue = queue.Queue()
        self._playback_staging: bytearray = bytearray()
        self._buffer_lock = threading.Lock()

    @staticmethod
    def _sanitize_device_name(name: str) -> str:
        """Strip raw carriage returns and extra whitespace from PortAudio device names."""
        if not name:
            return ""
        return re.sub(r"[\r\n]+", " ", name).strip()

    def resolve_devices(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Dynamically discover and resolve PortAudio device indices for input and output endpoints.
        Prioritizes user regex, known Bluetooth HFP patterns, and falls back to system defaults.
        """
        if not HAS_SOUNDDEVICE or sd is None:
            logger.warning("sounddevice is not available; unable to query audio devices.")
            return None, None

        try:
            device_list = sd.query_devices()
        except Exception as e:
            logger.error(f"Failed to query sounddevice audio devices: {e}")
            return None, None

        in_idx: Optional[int] = None
        out_idx: Optional[int] = None

        # 1. Check explicit preferred device indices/names
        if self.preferred_input_device is not None:
            if isinstance(self.preferred_input_device, int):
                in_idx = self.preferred_input_device
            else:
                for idx, dev in enumerate(device_list):
                    if dev["max_input_channels"] > 0 and self.preferred_input_device.lower() in dev["name"].lower():
                        in_idx = idx
                        break

        if self.preferred_output_device is not None:
            if isinstance(self.preferred_output_device, int):
                out_idx = self.preferred_output_device
            else:
                for idx, dev in enumerate(device_list):
                    if dev["max_output_channels"] > 0 and self.preferred_output_device.lower() in dev["name"].lower():
                        out_idx = idx
                        break

        # 2. Search by custom device_regex if provided
        patterns_to_try = []
        if self.device_regex:
            patterns_to_try.append(self.device_regex)
        patterns_to_try.extend(DEFAULT_HFP_REGEX_PATTERNS)

        if in_idx is None:
            for pattern in patterns_to_try:
                regex = re.compile(pattern, re.IGNORECASE)
                for idx, dev in enumerate(device_list):
                    clean_name = self._sanitize_device_name(dev["name"])
                    if dev["max_input_channels"] > 0 and regex.search(clean_name):
                        in_idx = idx
                        break
                if in_idx is not None:
                    break

        if out_idx is None:
            for pattern in patterns_to_try:
                regex = re.compile(pattern, re.IGNORECASE)
                for idx, dev in enumerate(device_list):
                    clean_name = self._sanitize_device_name(dev["name"])
                    if dev["max_output_channels"] > 0 and regex.search(clean_name):
                        out_idx = idx
                        break
                if out_idx is not None:
                    break

        # 3. Fallback to system defaults if still unresolved
        try:
            defaults = sd.default.device
            if in_idx is None:
                if isinstance(defaults, (list, tuple)) and len(defaults) > 0 and defaults[0] >= 0:
                    in_idx = defaults[0]
                else:
                    for idx, dev in enumerate(device_list):
                        if dev["max_input_channels"] > 0:
                            in_idx = idx
                            break

            if out_idx is None:
                if isinstance(defaults, (list, tuple)) and len(defaults) > 1 and defaults[1] >= 0:
                    out_idx = defaults[1]
                else:
                    for idx, dev in enumerate(device_list):
                        if dev["max_output_channels"] > 0:
                            out_idx = idx
                            break
        except Exception as e:
            logger.debug(f"Unable to read sounddevice default devices: {e}")

        self._input_device_idx = in_idx
        self._output_device_idx = out_idx

        if in_idx is not None and in_idx < len(device_list):
            self._input_device_name = self._sanitize_device_name(device_list[in_idx]["name"])
        else:
            self._input_device_name = "Default Input"

        if out_idx is not None and out_idx < len(device_list):
            self._output_device_name = self._sanitize_device_name(device_list[out_idx]["name"])
        else:
            self._output_device_name = "Default Output"

        logger.info(
            f"Resolved Windows audio endpoints -> Input: [{self._input_device_idx}] {self._input_device_name}, "
            f"Output: [{self._output_device_idx}] {self._output_device_name}"
        )
        return in_idx, out_idx

    def _in_stream_callback(self, indata: bytes, frames: int, time_info: Any, status: Any) -> None:
        """
        PortAudio callback for inbound microphone/HFP audio stream.
        Executes on a background PortAudio thread.
        """
        if status:
            logger.warning(f"WindowsAudioDriver input stream status: {status}")

        if not indata:
            return

        pcm_bytes = bytes(indata)

        # Inbound audio to Gemini Live must be 16kHz 16-bit signed mono PCM
        if self.input_sample_rate != 16000:
            try:
                pcm_bytes = AudioProcessor.resample_pcm(
                    pcm_bytes,
                    in_rate=self.input_sample_rate,
                    out_rate=16000,
                )
            except Exception as e:
                logger.error(f"Inbound audio resampling failed: {e}")
                return

        if self._on_audio_in is not None:
            self._dispatch_callback(self._on_audio_in, pcm_bytes)

    def _out_stream_callback(self, outdata: bytearray, frames: int, time_info: Any, status: Any) -> None:
        """
        PortAudio callback for outbound speaker/HFP audio stream.
        Fills outdata with 16-bit mono PCM samples from playback staging buffer or queue.
        """
        if status:
            logger.warning(f"WindowsAudioDriver output stream status: {status}")

        bytes_needed = frames * 2  # 16-bit mono = 2 bytes per sample

        with self._buffer_lock:
            # Drain queue into staging buffer if needed
            while len(self._playback_staging) < bytes_needed:
                try:
                    chunk = self._out_queue.get_nowait()
                    if chunk:
                        self._playback_staging.extend(chunk)
                except queue.Empty:
                    break

            available = len(self._playback_staging)
            if available >= bytes_needed:
                outdata[:] = self._playback_staging[:bytes_needed]
                del self._playback_staging[:bytes_needed]
            elif available > 0:
                silence_padding = bytes_needed - available
                outdata[:] = self._playback_staging + (b"\x00" * silence_padding)
                self._playback_staging.clear()
            else:
                outdata[:] = b"\x00" * bytes_needed

    async def start(self) -> None:
        """
        Open PortAudio RawInputStream and RawOutputStream and begin streaming.
        Idempotent: does nothing if already active.
        """
        if self._is_active:
            logger.debug("WindowsAudioDriver start called but driver is already active.")
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        if not HAS_SOUNDDEVICE or sd is None:
            raise RuntimeError("sounddevice is required for WindowsAudioDriver but is not installed.")

        in_idx, out_idx = self.resolve_devices()

        in_blocksize = int(self.input_sample_rate * 0.02)  # 20ms blocksize (320 samples @ 16kHz)
        out_blocksize = int(self.output_sample_rate * 0.02)

        try:
            self._in_stream = sd.RawInputStream(
                device=in_idx,
                channels=1,
                samplerate=self.input_sample_rate,
                dtype="int16",
                blocksize=in_blocksize,
                callback=self._in_stream_callback,
            )
            self._out_stream = sd.RawOutputStream(
                device=out_idx,
                channels=1,
                samplerate=self.output_sample_rate,
                dtype="int16",
                blocksize=out_blocksize,
                callback=self._out_stream_callback,
            )

            self._in_stream.start()
            self._out_stream.start()
            self._is_active = True
            logger.info("WindowsAudioDriver streams started successfully.")
        except Exception as e:
            logger.warning(
                f"Could not open specific device indices [{in_idx}, {out_idx}] ({e}); "
                f"falling back to system default communication audio endpoints."
            )
            if self._in_stream:
                try:
                    self._in_stream.close()
                except Exception:
                    pass
                self._in_stream = None
            if self._out_stream:
                try:
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None

            try:
                def_in, def_out = sd.default.device
                self._in_stream = sd.RawInputStream(
                    device=def_in,
                    channels=1,
                    samplerate=self.input_sample_rate,
                    dtype="int16",
                    blocksize=in_blocksize,
                    callback=self._in_stream_callback,
                )
                self._out_stream = sd.RawOutputStream(
                    device=def_out,
                    channels=1,
                    samplerate=self.output_sample_rate,
                    dtype="int16",
                    blocksize=out_blocksize,
                    callback=self._out_stream_callback,
                )
                self._in_stream.start()
                self._out_stream.start()
                self._is_active = True
                self._input_device_idx = def_in
                self._output_device_idx = def_out
                logger.info(f"WindowsAudioDriver fallback streams started on devices [{def_in}, {def_out}].")
            except Exception as e2:
                self._is_active = False
                logger.error(f"Failed to start fallback WindowsAudioDriver audio streams: {e2}")
                raise

        # Start background inbound call detection task
        self._detector_task: Optional[asyncio.Task] = asyncio.create_task(self._monitor_incoming_calls())

    async def _monitor_incoming_calls(self) -> None:
        """Polls Windows audio sessions to automatically detect incoming phone call audio."""
        while self._is_active:
            await asyncio.sleep(1.0)
            try:
                from pycaw.pycaw import AudioUtilities
                sessions = AudioUtilities.GetAllSessions()
                is_call_session = False
                for s in sessions:
                    pname = s.Process.name().lower() if s.Process else ""
                    dname = (s.DisplayName or "").lower()
                    if ("phone" in pname or "iqoo" in dname or "hands-free" in dname) and s.State == 1:
                        is_call_session = True
                        break

                if is_call_session and self._call_state == "IDLE":
                    logger.info("Detected active incoming phone call audio session from Windows Phone Link!")
                    self._set_state("RINGING", {"caller_id": "Inbound Cellular Call", "caller_name": "Phone Link"})

            except Exception as e:
                logger.debug(f"Audio session poll error: {e}")

    async def stop(self) -> None:
        """
        Stop and close active audio streams and flush buffers.
        Idempotent: does nothing if not active.
        """
        if not self._is_active:
            logger.debug("WindowsAudioDriver stop called but driver is not active.")
            return

        self._is_active = False

        if hasattr(self, "_detector_task") and self._detector_task and not self._detector_task.done():
            self._detector_task.cancel()
            self._detector_task = None

        if self._in_stream is not None:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception as e:
                logger.debug(f"Error stopping input stream: {e}")
            self._in_stream = None

        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception as e:
                logger.debug(f"Error stopping output stream: {e}")
            self._out_stream = None

        self.clear_audio_buffer()
        self._set_state("IDLE")
        logger.info("WindowsAudioDriver stopped.")

    def play_audio_chunk(self, pcm_data: bytes, sample_rate: int = 24000) -> None:
        """
        Enqueue an outgoing PCM audio chunk for transmission/playback.
        If received audio rate (default 24kHz Gemini speech) differs from output endpoint rate
        (e.g. 16kHz Bluetooth HFP), resamples using AudioProcessor.resample_pcm.
        """
        if not pcm_data:
            return

        # Ensure even alignment for 16-bit PCM
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
        if not pcm_data:
            return

        if sample_rate != self.output_sample_rate:
            try:
                pcm_data = AudioProcessor.resample_pcm(
                    pcm_data,
                    in_rate=sample_rate,
                    out_rate=self.output_sample_rate,
                )
            except Exception as e:
                logger.error(f"Outbound audio resampling failed ({sample_rate} -> {self.output_sample_rate}): {e}")
                return

        self._out_queue.put(pcm_data)

    def clear_audio_buffer(self) -> None:
        """
        Immediately clear all queued outgoing audio buffers for instant barge-in speech interruption.
        Flushes both the waiting queue and the active callback staging buffer.
        """
        with self._buffer_lock:
            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except queue.Empty:
                    break
            self._playback_staging.clear()

    async def dial(self, phone_number: str) -> bool:
        """
        Initiate outbound dialing via Windows tel: protocol URL.
        Sanitizes phone number (strips characters, preserves digits and leading +).
        """
        if not phone_number or not isinstance(phone_number, str):
            logger.warning(f"Invalid phone number provided for dialing: {phone_number}")
            return False

        cleaned = phone_number.strip()
        has_plus = cleaned.startswith("+")
        digits = re.sub(r"\D", "", cleaned)
        if not digits:
            logger.warning(f"Phone number contains no numeric digits: {phone_number}")
            return False

        clean_number = f"+{digits}" if has_plus else digits
        tel_url = f"tel:{clean_number}"
        logger.info(f"Initiating outbound dial via protocol URL: {tel_url}")

        dial_success = False
        if sys.platform == "win32" and hasattr(os, "startfile"):
            try:
                os.startfile(tel_url)
                dial_success = True
            except Exception as e:
                logger.warning(f"os.startfile failed for {tel_url} ({e}); attempting subprocess fallback")

        if not dial_success:
            try:
                if sys.platform == "win32":
                    subprocess.run(["cmd", "/c", "start", "", tel_url], shell=True, check=False)
                    dial_success = True
                else:
                    # Non-Windows fallback (e.g. xdg-open)
                    subprocess.run(["xdg-open", tel_url], check=False)
                    dial_success = True
            except Exception as e:
                logger.error(f"Fallback dialing command failed for {tel_url}: {e}")
                return False

        self._set_state("ACTIVE", {"phone_number": clean_number, "direction": "outbound"})
        return dial_success

    async def answer_call(self) -> bool:
        """Answer an incoming ringing call and transition driver to ACTIVE state."""
        self._set_state("ACTIVE", {"direction": "inbound"})
        return True

    async def hang_up(self, reason: str = "normal") -> bool:
        """Terminate the active call and flush all audio buffers."""
        self.clear_audio_buffer()
        self._set_state("COMPLETED", {"reason": reason})
        return True

    def is_active(self) -> bool:
        """Return True if driver audio streams are active or in call session."""
        return self._is_active

    def get_status(self) -> Dict[str, Any]:
        """Return driver operational telemetry, active endpoints, and queue metrics."""
        with self._buffer_lock:
            staging_bytes = len(self._playback_staging)
        return {
            "driver": self.driver_name,
            "is_active": self._is_active,
            "call_state": self._call_state,
            "input_device": self._input_device_name,
            "input_device_index": self._input_device_idx,
            "input_sample_rate": self.input_sample_rate,
            "output_device": self._output_device_name,
            "output_device_index": self._output_device_idx,
            "output_sample_rate": self.output_sample_rate,
            "queue_depth": self._out_queue.qsize(),
            "staging_buffer_bytes": staging_bytes,
        }
