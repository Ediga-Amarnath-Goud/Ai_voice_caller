"""
Virtual Simulator Telephony Driver for AI Voice Calling Assistant.
Routes audio directly through the host laptop's microphone and speakers for
offline conversational testing, latency profiling, barge-in evaluation, and
tool-calling verification without a physical cellular phone or Bluetooth connection.
Includes headless / simulated mode support for CI environments without physical audio hardware.
"""

import asyncio
import logging
import queue
import re
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

logger = logging.getLogger("telephony.virtual")


class VirtualSimulatorDriver(BaseTelephonyDriver):
    """
    Simulated telephony bridge using local sound hardware (microphone & speakers)
    or non-blocking in-memory simulation for headless testing.
    """

    def __init__(
        self,
        input_device: Optional[Union[int, str]] = None,
        output_device: Optional[Union[int, str]] = None,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        headless: bool = False,
    ) -> None:
        super().__init__()
        self.preferred_input_device = input_device
        self.preferred_output_device = output_device
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.headless = headless

        self._input_device_idx: Optional[int] = None
        self._output_device_idx: Optional[int] = None
        self._input_device_name: str = "Simulated Mic" if headless else "Unresolved"
        self._output_device_name: str = "Simulated Speaker" if headless else "Unresolved"

        self._in_stream: Optional[Any] = None
        self._out_stream: Optional[Any] = None
        self._is_active: bool = False
        self._simulated_mode: bool = headless

        self._out_queue: queue.Queue = queue.Queue()
        self._playback_staging: bytearray = bytearray()
        self._buffer_lock = threading.Lock()

    @staticmethod
    def _sanitize_device_name(name: str) -> str:
        if not name:
            return ""
        return re.sub(r"[\r\n]+", " ", name).strip()

    def resolve_devices(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Resolve PortAudio device indices for laptop microphone and speakers.
        """
        if self.headless or not HAS_SOUNDDEVICE or sd is None:
            self._simulated_mode = True
            self._input_device_name = "Simulated Mic"
            self._output_device_name = "Simulated Speaker"
            return None, None

        try:
            device_list = sd.query_devices()
        except Exception as e:
            logger.warning(f"Could not query audio devices ({e}); falling back to simulated mode.")
            self._simulated_mode = True
            return None, None

        in_idx: Optional[int] = None
        out_idx: Optional[int] = None

        # 1. Preferred devices
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

        # 2. Defaults from sounddevice
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
            self._input_device_name = "Default Laptop Mic"

        if out_idx is not None and out_idx < len(device_list):
            self._output_device_name = self._sanitize_device_name(device_list[out_idx]["name"])
        else:
            self._output_device_name = "Default Laptop Speaker"

        logger.info(
            f"Virtual simulator endpoints -> Input: [{self._input_device_idx}] {self._input_device_name}, "
            f"Output: [{self._output_device_idx}] {self._output_device_name}"
        )
        return in_idx, out_idx

    def _in_stream_callback(self, indata: bytes, frames: int, time_info: Any, status: Any) -> None:
        """Inbound stream callback routing laptop microphone audio to on_audio_in."""
        if status:
            logger.warning(f"VirtualSimulatorDriver input stream status: {status}")

        if not indata:
            return

        pcm_bytes = bytes(indata)

        if self.input_sample_rate != 16000:
            try:
                pcm_bytes = AudioProcessor.resample_pcm(
                    pcm_bytes,
                    in_rate=self.input_sample_rate,
                    out_rate=16000,
                )
            except Exception as e:
                logger.error(f"Inbound audio resampling failed in VirtualSimulatorDriver: {e}")
                return

        if self._on_audio_in is not None:
            self._dispatch_callback(self._on_audio_in, pcm_bytes)

    def _out_stream_callback(self, outdata: bytearray, frames: int, time_info: Any, status: Any) -> None:
        """Outbound stream callback playing assistant speech through laptop speakers."""
        if status:
            logger.warning(f"VirtualSimulatorDriver output stream status: {status}")

        bytes_needed = frames * 2  # 16-bit mono

        with self._buffer_lock:
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
        Open audio streams on laptop microphone and speakers, or engage simulated mode.
        Idempotent: does nothing if already active.
        """
        if self._is_active:
            logger.debug("VirtualSimulatorDriver start called but driver is already active.")
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        if self.headless or not HAS_SOUNDDEVICE or sd is None:
            self._simulated_mode = True
            self._is_active = True
            logger.info("VirtualSimulatorDriver started in simulated / headless mode.")
            return

        in_idx, out_idx = self.resolve_devices()

        in_blocksize = int(self.input_sample_rate * 0.02)
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
            self._simulated_mode = False
            logger.info("VirtualSimulatorDriver hardware audio streams started successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to open hardware audio streams ({e}); switching VirtualSimulatorDriver to simulated mode."
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
            self._simulated_mode = True
            self._is_active = True

    async def stop(self) -> None:
        """
        Stop and close active audio streams and flush buffers.
        Idempotent: does nothing if not active.
        """
        if not self._is_active:
            logger.debug("VirtualSimulatorDriver stop called but driver is not active.")
            return

        self._is_active = False

        if self._in_stream is not None:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception as e:
                logger.debug(f"Error stopping simulator input stream: {e}")
            self._in_stream = None

        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception as e:
                logger.debug(f"Error stopping simulator output stream: {e}")
            self._out_stream = None

        self.clear_audio_buffer()
        self._set_state("IDLE")
        logger.info("VirtualSimulatorDriver stopped.")

    def play_audio_chunk(self, pcm_data: bytes, sample_rate: int = 24000) -> None:
        """
        Enqueue an outgoing PCM audio chunk for laptop speaker playback.
        Resamples if incoming rate differs from output rate.
        """
        if not pcm_data:
            return

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
                logger.error(f"Outbound audio resampling failed in VirtualSimulatorDriver: {e}")
                return

        self._out_queue.put(pcm_data)

    def clear_audio_buffer(self) -> None:
        """
        Immediately clear all queued outgoing audio buffers for instant barge-in speech interruption.
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
        Simulate outbound call initiation to a phone number.
        Transitions state: IDLE -> DIALING -> ACTIVE.
        """
        if not phone_number or not isinstance(phone_number, str):
            logger.warning(f"Invalid phone number provided to virtual dial: {phone_number}")
            return False

        cleaned = phone_number.strip()
        has_plus = cleaned.startswith("+")
        digits = re.sub(r"\D", "", cleaned)
        if not digits:
            logger.warning(f"Phone number contains no numeric digits: {phone_number}")
            return False

        clean_number = f"+{digits}" if has_plus else digits
        self._set_state("DIALING", {"phone_number": clean_number, "direction": "outbound"})
        self._set_state("ACTIVE", {"phone_number": clean_number, "direction": "outbound"})
        return True

    async def answer_call(self) -> bool:
        """Simulate answering an incoming ringing call."""
        self._set_state("ACTIVE", {"direction": "inbound"})
        return True

    async def hang_up(self, reason: str = "normal") -> bool:
        """Simulate terminating the active call."""
        self.clear_audio_buffer()
        self._set_state("COMPLETED", {"reason": reason})
        return True

    def simulate_inbound_audio(self, pcm_data: bytes) -> None:
        """
        Test helper to inject simulated caller audio into the inbound audio callback.
        Useful for automated and CI tests.
        """
        if not pcm_data:
            return
        if self._on_audio_in is not None:
            self._dispatch_callback(self._on_audio_in, pcm_data)

    def is_active(self) -> bool:
        """Return True if driver is active."""
        return self._is_active

    def get_status(self) -> Dict[str, Any]:
        """Return driver operational telemetry, endpoints, and queue metrics."""
        with self._buffer_lock:
            staging_bytes = len(self._playback_staging)
        return {
            "driver": self.driver_name,
            "is_active": self._is_active,
            "simulated_mode": self._simulated_mode,
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
