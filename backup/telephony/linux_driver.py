"""
Linux Standalone Telephony Driver for AI Voice Calling Assistant.
Implements telephony integration on Linux using BlueZ (Bluetooth stack),
oFono (telephony daemon for call controls), and PipeWire / ALSA (SCO audio routing).
Provides guarded imports and automatic mock fallback on non-Linux hosts or when
D-Bus libraries are unavailable.
"""

import asyncio
import logging
import queue
import re
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("telephony.linux")

HAS_DBUS = False
dbus_module: Any = None

if sys.platform.startswith("linux"):
    try:
        import dbus  # type: ignore
        import dbus.mainloop.glib  # type: ignore
        dbus_module = dbus
        HAS_DBUS = True
    except ImportError:
        HAS_DBUS = False
        logger.debug("Linux D-Bus modules not found; LinuxDriver will use mock fallback.")

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False

from core.audio_processor import AudioProcessor
from telephony.base_driver import BaseTelephonyDriver


class LinuxDriver(BaseTelephonyDriver):
    """
    Linux Telephony Driver interfacing with BlueZ and oFono via D-Bus,
    with automatic mock fallback for cross-platform stability.
    """

    OFONO_SERVICE = "org.ofono"
    OFONO_MANAGER_INTERFACE = "org.ofono.Manager"
    OFONO_VCM_INTERFACE = "org.ofono.VoiceCallManager"
    OFONO_CALL_INTERFACE = "org.ofono.VoiceCall"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mock_mode: bool = False,
        modem_path: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.modem_path = modem_path
        self.sample_rate = sample_rate

        self.is_linux = sys.platform.startswith("linux")
        self.mock_mode = mock_mode or (not self.is_linux) or (not HAS_DBUS)

        if self.mock_mode:
            logger.info(
                f"LinuxDriver initialized in MOCK mode (platform={sys.platform}, has_dbus={HAS_DBUS}, mock_flag={mock_mode})"
            )

        self._is_active: bool = False
        self._out_queue: queue.Queue = queue.Queue()
        self._playback_staging: bytearray = bytearray()
        self._buffer_lock = threading.Lock()

        # D-Bus & oFono handles
        self._bus: Optional[Any] = None
        self._manager: Optional[Any] = None
        self._vcm: Optional[Any] = None
        self._active_call_path: Optional[str] = None
        self._active_call: Optional[Any] = None

        # Audio stream handles (PipeWire / ALSA via sounddevice)
        self._in_stream: Optional[Any] = None
        self._out_stream: Optional[Any] = None

    def _setup_dbus(self) -> bool:
        """Initialize D-Bus connection and locate active oFono modem."""
        if self.mock_mode or not HAS_DBUS or dbus_module is None:
            return False

        try:
            dbus_module.mainloop.glib.DBusGMainLoop(set_as_default=True)
            self._bus = dbus_module.SystemBus()
            self._manager = dbus_module.Interface(
                self._bus.get_object(self.OFONO_SERVICE, "/"),
                self.OFONO_MANAGER_INTERFACE,
            )

            # Discover first powered modem if not specified
            if not self.modem_path:
                modems = self._manager.GetModems()
                for path, properties in modems:
                    if properties.get("Powered", False) and properties.get("Online", False):
                        self.modem_path = path
                        break
                if not self.modem_path and modems:
                    self.modem_path = modems[0][0]

            if self.modem_path:
                self._vcm = dbus_module.Interface(
                    self._bus.get_object(self.OFONO_SERVICE, self.modem_path),
                    self.OFONO_VCM_INTERFACE,
                )
                logger.info(f"Connected to oFono VoiceCallManager on modem {self.modem_path}")
                return True
            else:
                logger.warning("No active oFono modem found on D-Bus system bus.")
                return False
        except Exception as e:
            logger.warning(f"Failed to initialize Linux D-Bus / oFono ({e}); switching to mock mode.")
            self.mock_mode = True
            return False

    def _in_stream_callback(self, indata: bytes, frames: int, time_info: Any, status: Any) -> None:
        """Inbound audio stream callback on Linux."""
        if not indata:
            return
        pcm_bytes = bytes(indata)
        if self.sample_rate != 16000:
            try:
                pcm_bytes = AudioProcessor.resample_pcm(pcm_bytes, in_rate=self.sample_rate, out_rate=16000)
            except Exception as e:
                logger.error(f"LinuxDriver inbound resampling failed: {e}")
                return
        if self._on_audio_in is not None:
            self._dispatch_callback(self._on_audio_in, pcm_bytes)

    def _out_stream_callback(self, outdata: bytearray, frames: int, time_info: Any, status: Any) -> None:
        """Outbound audio stream callback on Linux."""
        bytes_needed = frames * 2
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
                padding = bytes_needed - available
                outdata[:] = self._playback_staging + (b"\x00" * padding)
                self._playback_staging.clear()
            else:
                outdata[:] = b"\x00" * bytes_needed

    async def start(self) -> None:
        """
        Start Linux telephony driver. In Linux mode, sets up D-Bus and PipeWire/ALSA streams.
        In mock mode, transitions operational state.
        Idempotent.
        """
        if self._is_active:
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        if self.mock_mode:
            self._is_active = True
            logger.info("LinuxDriver started in mock mode.")
            return

        # Attempt D-Bus connection
        self._setup_dbus()

        # Start audio streams via sounddevice / PipeWire
        if HAS_SOUNDDEVICE and sd is not None:
            try:
                blocksize = int(self.sample_rate * 0.02)
                self._in_stream = sd.RawInputStream(
                    channels=1,
                    samplerate=self.sample_rate,
                    dtype="int16",
                    blocksize=blocksize,
                    callback=self._in_stream_callback,
                )
                self._out_stream = sd.RawOutputStream(
                    channels=1,
                    samplerate=self.sample_rate,
                    dtype="int16",
                    blocksize=blocksize,
                    callback=self._out_stream_callback,
                )
                self._in_stream.start()
                self._out_stream.start()
            except Exception as e:
                logger.warning(f"Could not open Linux hardware audio streams ({e}); using queue-only mode.")

        self._is_active = True
        logger.info("LinuxDriver started.")

    async def stop(self) -> None:
        """
        Stop Linux telephony driver, close streams, and release D-Bus resources.
        Idempotent.
        """
        if not self._is_active:
            return

        self._is_active = False

        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

        if self._out_stream:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

        self.clear_audio_buffer()
        self._set_state("IDLE")
        logger.info("LinuxDriver stopped.")

    async def dial(self, phone_number: str) -> bool:
        """
        Initiate outbound dialing. On Linux with oFono, calls VoiceCallManager.Dial.
        In mock mode, simulates state transitions.
        """
        if not phone_number or not isinstance(phone_number, str):
            logger.warning(f"Invalid phone number provided for Linux dial: {phone_number}")
            return False

        cleaned = phone_number.strip()
        has_plus = cleaned.startswith("+")
        digits = re.sub(r"\D", "", cleaned)
        if not digits:
            logger.warning(f"Phone number contains no numeric digits: {phone_number}")
            return False

        clean_number = f"+{digits}" if has_plus else digits

        if not self.mock_mode and self._vcm is not None:
            try:
                self._set_state("DIALING", {"phone_number": clean_number, "direction": "outbound"})
                call_path = self._vcm.Dial(clean_number, "default")
                self._active_call_path = str(call_path)
                if self._bus and dbus_module:
                    self._active_call = dbus_module.Interface(
                        self._bus.get_object(self.OFONO_SERVICE, self._active_call_path),
                        self.OFONO_CALL_INTERFACE,
                    )
                self._set_state("ACTIVE", {"phone_number": clean_number, "call_path": self._active_call_path})
                return True
            except Exception as e:
                logger.error(f"oFono VoiceCallManager.Dial failed: {e}")
                return False

        # Mock mode fallback
        self._set_state("DIALING", {"phone_number": clean_number, "direction": "outbound"})
        self._set_state("ACTIVE", {"phone_number": clean_number, "direction": "outbound"})
        return True

    async def answer_call(self) -> bool:
        """Answer an incoming ringing call."""
        if not self.mock_mode and self._active_call is not None:
            try:
                self._active_call.Answer()
                self._set_state("ACTIVE", {"direction": "inbound"})
                return True
            except Exception as e:
                logger.error(f"oFono VoiceCall.Answer failed: {e}")
                return False

        self._set_state("ACTIVE", {"direction": "inbound"})
        return True

    async def hang_up(self, reason: str = "normal") -> bool:
        """Terminate the active or ringing call."""
        if not self.mock_mode and self._active_call is not None:
            try:
                self._active_call.Hangup()
            except Exception as e:
                logger.warning(f"oFono VoiceCall.Hangup error ({e}); proceeding with cleanup.")

        self._active_call = None
        self._active_call_path = None
        self.clear_audio_buffer()
        self._set_state("COMPLETED", {"reason": reason})
        return True

    def play_audio_chunk(self, pcm_data: bytes, sample_rate: int = 24000) -> None:
        """Enqueue an outgoing PCM audio chunk for transmission or speaker playback."""
        if not pcm_data:
            return
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
        if not pcm_data:
            return

        if sample_rate != self.sample_rate:
            try:
                pcm_data = AudioProcessor.resample_pcm(
                    pcm_data,
                    in_rate=sample_rate,
                    out_rate=self.sample_rate,
                )
            except Exception as e:
                logger.error(f"LinuxDriver outbound resampling failed: {e}")
                return

        self._out_queue.put(pcm_data)

    def clear_audio_buffer(self) -> None:
        """Flush playback queue and active buffer for instant speech barge-in."""
        with self._buffer_lock:
            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except queue.Empty:
                    break
            self._playback_staging.clear()

    def simulate_inbound_audio(self, pcm_data: bytes) -> None:
        """Test helper to inject simulated inbound audio."""
        if not pcm_data:
            return
        if self._on_audio_in is not None:
            self._dispatch_callback(self._on_audio_in, pcm_data)

    def is_active(self) -> bool:
        """Return True if driver is active."""
        return self._is_active

    def get_status(self) -> Dict[str, Any]:
        """Return operational status of Linux driver."""
        with self._buffer_lock:
            staging_bytes = len(self._playback_staging)
        return {
            "driver": self.driver_name,
            "is_active": self._is_active,
            "mock_mode": self.mock_mode,
            "is_linux": self.is_linux,
            "has_dbus": HAS_DBUS,
            "call_state": self._call_state,
            "modem_path": self.modem_path,
            "sample_rate": self.sample_rate,
            "queue_depth": self._out_queue.qsize(),
            "staging_buffer_bytes": staging_bytes,
        }
