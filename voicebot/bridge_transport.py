"""Pipecat transport for the Phone Link audio bridge.

Phone Link plays the caller through the ordinary Realtek speakers and records
its microphone from the Windows default capture device. That gives us a purely
software bridge, with no Bluetooth hands-free endpoint and no hardware:

    caller -> Phone Link -> Realtek speakers -> WASAPI loopback -> Pipecat
    Pipecat -> CABLE Input -> CABLE Output -> Windows default mic -> Phone Link

WASAPI loopback is not exposed through PyAudio/MME, so the input side uses the
``soundcard`` package (MediaFoundation) in a capture thread. The output side is
the stock Pipecat PyAudio transport, which writes to CABLE Input like any other
playback device.

Inbound audio chain (24 kHz native):
    DC blocker -> 120 Hz Butterworth HPF -> +3 dB @ 3.2 kHz Presence EQ
        -> Smooth Envelope AGC -> soft clip

Outbound: Gemini Live 24 kHz -> CABLE Input (MME, 44.1 kHz), resampled by Pipecat.
"""
from __future__ import annotations

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time

import numpy as np
import soundcard as sc
from pipecat.frames.frames import InputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.local.audio import LocalAudioOutputTransport

try:
    import pyaudio
except ModuleNotFoundError as exc:  # pragma: no cover - environment check
    raise ImportError("pyaudio is required for the Phone Link bridge") from exc

DEFAULT_LOOPBACK = "Speakers (Realtek(R) Audio)"


def _safe_name(device) -> str | None:
    try:
        return str(device.name)
    except Exception:  # noqa: BLE001 - a dead endpoint throws on .name
        return None


def _loopback_device(name: str):
    for device in sc.all_microphones(include_loopback=True):
        if bool(getattr(device, "isloopback", False)) and _safe_name(device) == name:
            return device
    return None


# ---------------------------------------------------------------------------
# Biquad IIR filter (direct form II transposed)
# ---------------------------------------------------------------------------
class Biquad:
    """Second-order IIR filter.

    Coefficients are normalised so that ``a0 == 1``.  State is kept per
    instance so the filter can be called block-by-block without discontinuity.
    """

    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float) -> None:
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.a1 = a1
        self.a2 = a2
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def process(self, x: float) -> float:
        """Filter one sample."""
        y = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y

    def process_block(self, data: np.ndarray) -> np.ndarray:
        """Filter a whole block, preserving state across calls."""
        out = np.empty(len(data), dtype=np.float32)
        for i in range(len(data)):
            out[i] = self.process(float(data[i]))
        return out

    def reset(self) -> None:
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0


def _butterworth_hp(f0: float, fs: float) -> tuple[float, float, float, float, float]:
    """2nd-order Butterworth high-pass coefficients (normalised, a0 == 1).

    Q = 0.7071 for maximally flat response.
    """
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * 0.7071067811865476)
    cos_w0 = math.cos(w0)

    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _peaking_eq(f0: float, fs: float, gain_db: float, q: float) -> tuple[float, float, float, float, float]:
    """Parametric peaking EQ coefficients (normalised, a0 == 1).

    Positive gain_db = boost, negative gain_db = cut.
    """
    w0 = 2.0 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2.0 * q)
    a = 10.0 ** (gain_db / 40.0)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a

    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


# ---------------------------------------------------------------------------
# Transport parameters
# ---------------------------------------------------------------------------
class BridgeTransportParams(TransportParams):
    """Transport parameters with the loopback endpoint name added."""

    loopback_device: str = DEFAULT_LOOPBACK
    output_device_index: int | None = None

    #: Native sample rate — 24 kHz matches Gemini Live's output rate and avoids
    #: resampling artifacts on both input and output.
    sample_rate: int = 24000

    #: Inbound 120 Hz Butterworth HPF — removes cell-phone rumble / fan noise.
    enable_hpf: bool = False
    hpf_cutoff_hz: float = 120.0

    #: Inbound presence EQ — +3 dB @ 3.2 kHz, Q = 1.2 for crisp consonants.
    enable_presence_eq: bool = False
    presence_eq_freq_hz: float = 3200.0
    presence_eq_gain_db: float = 3.0
    presence_eq_q: float = 1.2

    #: Smooth envelope AGC — fast attack catches peaks, slow release keeps
    #: natural cadence without 20 ms gain pumping.
    enable_agc: bool = False
    agc_target_rms: float = 0.10          # about -20 dBFS RMS
    agc_attack_ms: float = 15.0
    agc_release_ms: float = 350.0
    agc_max_gain: float = 10.0            # max +20 dB boost
    agc_min_gain: float = 0.1             # max -20 dB cut

    #: Soft-clip ceiling (linear, 0..1). Prevents harsh digital distortion.
    audio_in_soft_clip_ceiling: float = 0.95

    #: Hysteresis frames for call-active detection.
    call_active_hysteresis_frames: int = 3


# ---------------------------------------------------------------------------
# Smooth envelope AGC
# ---------------------------------------------------------------------------
class SmoothAGC:
    """RMS envelope follower with independent attack/release smoothing.

    Gain changes are slew-limited so consecutive 20 ms blocks never jump in
    volume — this eliminates the "robotic" / choppy sound caused by
    instantaneous per-block normalisation.
    """

    def __init__(
        self,
        fs: int,
        target_rms: float = 0.10,
        attack_ms: float = 15.0,
        release_ms: float = 350.0,
        max_gain: float = 10.0,
        min_gain: float = 0.1,
    ) -> None:
        self.fs = fs
        self.target_rms = target_rms
        self.min_gain = min_gain
        self.max_gain = max_gain

        # One-pole smoothing coefficients (envelope follower).
        self._attack_coef = math.exp(-1.0 / (fs * attack_ms / 1000.0))
        self._release_coef = math.exp(-1.0 / (fs * release_ms / 1000.0))

        self._envelope = 0.0
        self._current_gain = 1.0

    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothed gain to one block."""
        rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0

        # Update envelope with attack/release smoothing.
        if rms > self._envelope:
            coef = self._attack_coef
        else:
            coef = self._release_coef
        self._envelope = coef * self._envelope + (1.0 - coef) * rms

        # Target gain from envelope, clamped.
        if self._envelope > 1e-6:
            target_gain = self.target_rms / self._envelope
        else:
            target_gain = self.max_gain
        target_gain = float(np.clip(target_gain, self.min_gain, self.max_gain))

        # Smooth the gain itself.
        gain_delta = target_gain - self._current_gain
        if abs(gain_delta) > 0.001:
            self._current_gain += gain_delta * 0.2

        return data * self._current_gain


# ---------------------------------------------------------------------------
# Input transport
# ---------------------------------------------------------------------------
class LoopbackInputTransport(BaseInputTransport):
    """Capture a render endpoint's loopback twin as pipeline audio input.

    Inbound chain (24 kHz):

        120 Hz Butterworth HPF -> +3 dB @ 3.2 kHz Presence EQ
            -> Smooth Envelope AGC -> soft clip

    Also monitors the loopback for caller audio and signals ``call_active``
    when the level rises above threshold.
    """

    _params: BridgeTransportParams

    def __init__(self, params: BridgeTransportParams) -> None:
        super().__init__(params)
        self._recorder = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._block = 0
        self._call_active = threading.Event()
        self._call_active_threshold = -45.0  # dBFS peak
        self._last_audio_time: float | None = None
        self._lock = threading.Lock()
        # Hysteresis counters for stable call-active detection.
        self._above_count = 0
        self._below_count = 0

        # Build the DSP chain from params.
        p = params
        self._hpf = Biquad(*_butterworth_hp(p.hpf_cutoff_hz, p.sample_rate)) if p.enable_hpf else None
        self._presence_eq = (
            Biquad(*_peaking_eq(p.presence_eq_freq_hz, p.sample_rate,
                                p.presence_eq_gain_db, p.presence_eq_q))
            if p.enable_presence_eq
            else None
        )
        self._agc = (
            SmoothAGC(
                fs=p.sample_rate,
                target_rms=p.agc_target_rms,
                attack_ms=p.agc_attack_ms,
                release_ms=p.agc_release_ms,
                max_gain=p.agc_max_gain,
                min_gain=p.agc_min_gain,
            )
            if p.enable_agc
            else None
        )
        self._clip_ceiling = p.audio_in_soft_clip_ceiling

    @property
    def call_active(self) -> threading.Event:
        return self._call_active

    @property
    def call_active_threshold(self) -> float:
        return self._call_active_threshold

    @call_active_threshold.setter
    def call_active_threshold(self, value: float) -> None:
        self._call_active_threshold = value

    @property
    def last_audio_time(self) -> float | None:
        with self._lock:
            return self._last_audio_time

    def is_silent_for(self, seconds: float) -> bool:
        """True if no caller audio has been heard for at least ``seconds``."""
        with self._lock:
            if self._last_audio_time is None:
                return True
        return time.monotonic() - self._last_audio_time >= seconds

    def _process_block(self, data: np.ndarray) -> np.ndarray:
        """Apply HPF -> Presence EQ -> Smooth AGC -> soft clip.

        No instantaneous per-block gain jumps — AGC uses a smoothed envelope
        so the audio stays natural for Gemini Live's speech-to-speech path.
        """
        data = data.astype(np.float32)

        if self._hpf is not None:
            data = self._hpf.process_block(data).astype(np.float32)
        if self._presence_eq is not None:
            data = self._presence_eq.process_block(data).astype(np.float32)
        if self._agc is not None:
            data = self._agc.process(data).astype(np.float32)

        # Soft clip to prevent harsh digital distortion.
        return np.tanh(data / self._clip_ceiling) * self._clip_ceiling

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)

        device = _loopback_device(self._params.loopback_device)
        if device is None:
            available = [
                _safe_name(m)
                for m in sc.all_microphones(include_loopback=True)
                if getattr(m, "isloopback", False)
            ]
            raise RuntimeError(
                f"loopback endpoint {self._params.loopback_device!r} not found; "
                f"available: {available}"
            )

        self._block = int(self.sample_rate / 100)  # 10 ms frames for lower latency
        self._recorder = device.recorder(
            samplerate=self.sample_rate,
            channels=self._params.audio_in_channels,
            blocksize=self._block,
        )
        self._recorder.__enter__()  # arm the stream before record()

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        await self.set_transport_ready(frame)

    def _capture_loop(self) -> None:
        """Pull 20 ms of loopback audio, lightly process it, and push into the pipeline.

        Processing is limited to a DC blocker + HPF + EQ + gentle AGC + soft clip
        — no per-block gain jumps, so the audio stays natural for Gemini Live's
        native speech-to-speech path.
        """
        loop = self.get_event_loop()
        scale = 32767.0
        p = self._params
        log = logging.getLogger("agent.bridge")
        pushed = 0
        reported = time.monotonic()

        while not self._stop.is_set() and self._recorder is not None:
            try:
                data = np.asarray(
                    self._recorder.record(numframes=self._block), dtype=np.float32
                ).reshape(-1)
            except Exception as exc:  # noqa: BLE001 - a dropped block must not kill the call
                log.warning(f"loopback read failed: {type(exc).__name__}: {exc}")
                continue

            if len(data) == 0:
                continue

            peak = float(np.abs(data).max())
            db = 20.0 * np.log10(peak) if peak > 0 else -120.0

            if self._hpf is not None or self._presence_eq is not None or self._agc is not None:
                processed = self._process_block(data)
            else:
                processed = data

            # Track call-active state with hysteresis (uses raw peak).
            now = time.monotonic()
            above = peak > (10.0 ** (self._call_active_threshold / 20.0))

            with self._lock:
                if above:
                    self._above_count += 1
                    self._below_count = 0
                    if (
                        not self._call_active.is_set()
                        and self._above_count >= p.call_active_hysteresis_frames
                    ):
                        self._call_active.set()
                        self._last_audio_time = now
                        log.info(f"call active: caller audio at {db:.1f} dBFS")
                    elif self._call_active.is_set():
                        self._last_audio_time = now
                else:
                    self._below_count += 1
                    self._above_count = 0
                    if (
                        self._call_active.is_set()
                        and self._below_count >= p.call_active_hysteresis_frames
                    ):
                        self._call_active.clear()
                        log.info("call inactive: caller audio dropped below threshold")

            pcm = (np.clip(processed, -1.0, 1.0) * scale).astype("<i2").tobytes()
            frame = InputAudioRawFrame(
                audio=pcm,
                sample_rate=self.sample_rate,
                num_channels=self._params.audio_in_channels,
            )
            asyncio.run_coroutine_threadsafe(self.push_audio_frame(frame), loop)
            pushed += 1

            if time.monotonic() - reported >= 5:
                log.info(f"loopback: {pushed} pushed, peak {db:.1f} dBFS")
                pushed = 0
                reported = time.monotonic()

    async def wait_for_call_active(self, timeout: float | None = None) -> bool:
        """Wait until caller audio is detected on the loopback."""
        loop = asyncio.get_running_loop()

        def _check() -> bool:
            return self._call_active.is_set()

        if _check():
            return True
        return await loop.run_in_executor(None, self._call_active.wait, timeout)

    async def cleanup(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self._recorder is not None:
            try:
                self._recorder.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - teardown only
                pass
            self._recorder = None
        await super().cleanup()


# ---------------------------------------------------------------------------
# Output transport wrapper
# ---------------------------------------------------------------------------
class BridgeTransport(BaseTransport):
    """Loopback input plus PyAudio output into the virtual cable.

    Output uses the resolved PyAudio device's native rate (MME CABLE Input is
    44.1 kHz); Pipecat resamples Gemini Live's 24 kHz to match.
    """

    def __init__(self, params: BridgeTransportParams) -> None:
        super().__init__()
        self._params = params
        self._py_audio = pyaudio.PyAudio()
        self._input: LoopbackInputTransport | None = None
        self._output: BaseOutputTransport | None = None

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = LoopbackInputTransport(self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = LocalAudioOutputTransport(self._py_audio, self._params)
        return self._output