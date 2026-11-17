"""
Audio Processor Engine for AI Voice Calling Assistant.
Provides high-fidelity resampling using soxr with scipy fallback,
PCM conversions, and audio signal analysis (RMS, Peak, 8-band FFT spectrum).
"""

import math
import logging
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger("audio_processor")

try:
    import soxr
    HAS_SOXR = True
except ImportError:
    HAS_SOXR = False
    logger.warning("soxr library not found; falling back to scipy.signal.resample_poly")

from scipy import signal


class AudioProcessor:
    """
    Handles audio conversions, sample rate resampling, and real-time level analysis.
    Operates on 16-bit signed mono PCM little-endian format.
    """

    @staticmethod
    def pcm_to_numpy(pcm_bytes: bytes) -> np.ndarray:
        """
        Converts 16-bit signed mono little-endian PCM bytes to float32 numpy array in [-1.0, 1.0].
        Handles odd-length byte arrays by truncating trailing byte.
        """
        if not pcm_bytes:
            return np.array([], dtype=np.float32)
        
        # Ensure even byte count for 16-bit samples
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        
        if not pcm_bytes:
            return np.array([], dtype=np.float32)

        int16_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_samples.astype(np.float32) / 32768.0

    @staticmethod
    def numpy_to_pcm(audio_arr: np.ndarray) -> bytes:
        """
        Converts float32 numpy array in [-1.0, 1.0] to 16-bit signed mono little-endian PCM bytes.
        Clips out-of-range values to [-32768, 32767].
        """
        if audio_arr is None or len(audio_arr) == 0:
            return b""
        
        clipped = np.clip(np.round(audio_arr * 32768.0), -32768.0, 32767.0)
        return clipped.astype(np.int16).tobytes()

    @classmethod
    def resample_pcm(cls, pcm_bytes: bytes, in_rate: int, out_rate: int) -> bytes:
        """
        Resamples 16-bit signed mono little-endian PCM from in_rate to out_rate.
        Uses soxr for high-speed, high-fidelity conversion with scipy.signal.resample_poly fallback.

        Handles edge cases:
        - Empty bytes -> returns b""
        - in_rate == out_rate -> returns aligned pcm_bytes
        - Odd-length bytes -> truncated to even sample alignment
        """
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError(f"Sample rates must be positive integers: in_rate={in_rate}, out_rate={out_rate}")

        if not pcm_bytes:
            return b""

        # Ensure even alignment
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]

        if not pcm_bytes:
            return b""

        if in_rate == out_rate:
            return pcm_bytes

        audio_float = cls.pcm_to_numpy(pcm_bytes)
        if len(audio_float) == 0:
            return b""

        resampled = None
        if HAS_SOXR:
            try:
                resampled = soxr.resample(audio_float, in_rate, out_rate)
            except Exception as e:
                logger.debug(f"soxr resampling failed ({e}); attempting scipy fallback")

        if resampled is None:
            # Fallback using scipy.signal.resample_poly
            gcd = math.gcd(in_rate, out_rate)
            up = out_rate // gcd
            down = in_rate // gcd
            resampled = signal.resample_poly(audio_float, up, down).astype(np.float32)

        return cls.numpy_to_pcm(resampled)

    @classmethod
    def calculate_levels(cls, pcm_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Calculates real-time audio levels from 16-bit signed mono PCM.
        Returns:
            {
                "rms": float,            # Root mean square energy (0.0 to 1.0)
                "peak": float,           # Peak absolute amplitude (0.0 to 1.0)
                "spectrum": list[float]  # 8 normalized frequency bands (0.0 to 1.0)
            }
        """
        default_result = {
            "rms": 0.0,
            "peak": 0.0,
            "spectrum": [0.0] * 8,
        }

        if not pcm_bytes or len(pcm_bytes) < 2:
            return default_result

        samples = cls.pcm_to_numpy(pcm_bytes)
        n = len(samples)
        if n == 0:
            return default_result

        # Peak amplitude (0.0 to 1.0)
        peak = float(np.max(np.abs(samples)))
        peak = min(1.0, max(0.0, peak))

        # RMS level (0.0 to 1.0)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        rms = min(1.0, max(0.0, rms))

        # 8-band FFT spectrum
        if n < 8:
            spectrum = [peak] * 8
        else:
            # Apply Hanning window to reduce spectral leakage
            window = np.hanning(n)
            windowed = samples * window
            fft_mag = np.abs(np.fft.rfft(windowed))

            # Hanning window coherent gain is 0.5; max sine peak amplitude in rfft is n / 4.0
            norm_factor = max(n / 4.0, 1.0)
            norm_mag = fft_mag / norm_factor

            # Partition frequency bins into 8 bands
            band_splits = np.array_split(norm_mag, 8)
            spectrum = []
            for b in band_splits:
                if len(b) > 0:
                    val = float(np.mean(b))
                else:
                    val = 0.0
                spectrum.append(min(1.0, max(0.0, val)))

        return {
            "rms": rms,
            "peak": peak,
            "spectrum": spectrum,
        }
