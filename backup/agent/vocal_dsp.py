"""Real-time vocal mastering and telephony DSP for broadcast-quality calls.

Solves the three core issues causing poor in-call audio:
1. Low Output Gain: Gemini Live generates audio at ~-32 dBFS RMS (faint whisper).
   This boosts it to the broadcast telephony standard (-14 dBFS RMS) so the
   caller hears the AI loudly and clearly without raising phone volume.
2. Boxy / Muffled Frequency Response: Mobile phone earpieces muffle frequencies
   below 300 Hz and compress mid-tones. A 130 Hz low-cut filter removes low mud,
   a 450 Hz dip removes hollow boxiness, and a +4.5 dB presence boost at 3.4 kHz
   gives the voice sharp, crystal-clear consonant diction (s, t, k, p).
3. Stateful Processing: Uses continuous IIR biquad filter state registers (zi)
   across 20ms chunks so there are ZERO border clicks or phase discontinuities.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from pipecat.frames.frames import Frame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TelephonyVocalDSP:
    """Continuous, stateful DSP processor for sharp telephony voice."""

    def __init__(self, sample_rate: int = 24000, is_outbound: bool = True) -> None:
        self.rate = sample_rate
        self.is_outbound = is_outbound
        nyq = sample_rate / 2.0

        # 1. 2nd-order Butterworth Highpass @ 130 Hz (De-mudding)
        self.b_hp, self.a_hp = signal.butter(2, 130.0 / nyq, btype="high")
        self.zi_hp = signal.lfilter_zi(self.b_hp, self.a_hp) * 0.0

        # 2. De-box notch/dip @ 450 Hz (-2.5 dB, Q=1.0)
        self.b_box, self.a_box = self._peaking_biquad(
            f0=450.0, Q=1.0, gain_db=-2.5, sr=sample_rate
        )
        self.zi_box = signal.lfilter_zi(self.b_box, self.a_box) * 0.0

        # 3. Telephony Vocal Presence Boost @ 3400 Hz (+4.5 dB, Q=1.2)
        # Inbound uses +3 dB, outbound uses +4.5 dB for crisp earpiece projection
        presence_gain = 4.5 if is_outbound else 3.0
        self.b_pres, self.a_pres = self._peaking_biquad(
            f0=3400.0, Q=1.2, gain_db=presence_gain, sr=sample_rate
        )
        self.zi_pres = signal.lfilter_zi(self.b_pres, self.a_pres) * 0.0

        # 4. Smooth Envelope Follower for Leveling & AGC
        self.env = 0.02
        # Fast attack (15ms), smooth decay (350ms)
        self.attack_alpha = float(np.exp(-1.0 / (0.015 * sample_rate / 480)))
        self.release_alpha = float(np.exp(-1.0 / (0.350 * sample_rate / 480)))
        # Target: -14 dBFS for outbound, -18 dBFS for inbound
        self.target_level = 0.20 if is_outbound else 0.125
        self.max_gain = 6.0 if is_outbound else 3.0

    @staticmethod
    def _peaking_biquad(
        f0: float, Q: float, gain_db: float, sr: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate coefficients for a 2nd-order peaking EQ biquad filter."""
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2.0 * Q)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A
        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
        return b, a

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process a float32 audio chunk (-1.0 to 1.0)."""
        if len(chunk) == 0:
            return chunk

        # Filter cascade: Highpass -> De-box -> Presence Boost
        y, self.zi_hp = signal.lfilter(self.b_hp, self.a_hp, chunk, zi=self.zi_hp)
        y, self.zi_box = signal.lfilter(self.b_box, self.a_box, y, zi=self.zi_box)
        y, self.zi_pres = signal.lfilter(self.b_pres, self.a_pres, y, zi=self.zi_pres)

        # Smooth Envelope Follower & Telephony AGC
        peak = float(np.abs(y).max())
        alpha = self.attack_alpha if peak > self.env else self.release_alpha
        self.env = alpha * self.env + (1.0 - alpha) * max(peak, 0.005)
        gain = min(self.target_level / self.env, self.max_gain)
        y = y * gain

        # Soft-knee limiter (tanh curve prevents digital clipping)
        return np.tanh(y / 0.95) * 0.95

    def process_pcm16(self, pcm_bytes: bytes) -> bytes:
        """Process raw 16-bit PCM bytes and return enhanced 16-bit PCM bytes."""
        if not pcm_bytes:
            return pcm_bytes
        data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        enhanced = self.process(data)
        out_int16 = (np.clip(enhanced, -1.0, 1.0) * 32767.0).astype(np.int16)
        return out_int16.tobytes()


class VocalMasteringProcessor(FrameProcessor):
    """Pipecat pipeline processor that masters AI audio in real-time."""

    def __init__(self, sample_rate: int = 24000, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dsp = TelephonyVocalDSP(sample_rate=sample_rate, is_outbound=True)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame):
            frame.audio = self._dsp.process_pcm16(frame.audio)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
