"""
Unit tests for AudioProcessor:
Verifies resampling (16k -> 24k, 24k -> 16k, 48k -> 16k, 16k -> 8k, identity, empty),
format conversions (pcm_to_numpy, numpy_to_pcm), and level calculations (rms, peak, 8-band spectrum).
"""

import math
import pytest
import numpy as np

import core.audio_processor as ap_module
from core.audio_processor import AudioProcessor


def generate_sine_pcm(freq: float, sample_rate: int, duration_sec: float, amplitude: float = 0.9) -> bytes:
    """Generates 16-bit mono PCM bytes for a pure sine wave."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    waveform = np.sin(2 * np.pi * freq * t) * amplitude
    int16_samples = np.clip(waveform * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return int16_samples.tobytes()


class TestAudioProcessorResampling:
    """Tests sample rate conversion fidelity and edge cases."""

    def test_resample_pcm_16k_to_24k(self):
        """100ms at 16kHz (1600 samples) resampled to 24kHz should yield 2400 samples."""
        in_pcm = generate_sine_pcm(freq=440.0, sample_rate=16000, duration_sec=0.1)
        assert len(in_pcm) == 1600 * 2

        out_pcm = AudioProcessor.resample_pcm(in_pcm, 16000, 24000)
        assert len(out_pcm) == 2400 * 2

        # Check amplitude preservation
        out_samples = AudioProcessor.pcm_to_numpy(out_pcm)
        assert np.max(np.abs(out_samples)) > 0.8

    def test_resample_pcm_24k_to_16k(self):
        """100ms at 24kHz (2400 samples) resampled to 16kHz should yield 1600 samples."""
        in_pcm = generate_sine_pcm(freq=1000.0, sample_rate=24000, duration_sec=0.1)
        assert len(in_pcm) == 2400 * 2

        out_pcm = AudioProcessor.resample_pcm(in_pcm, 24000, 16000)
        assert len(out_pcm) == 1600 * 2

        out_samples = AudioProcessor.pcm_to_numpy(out_pcm)
        assert np.max(np.abs(out_samples)) > 0.8

    def test_resample_pcm_48k_to_16k(self):
        """100ms at 48kHz (4800 samples) resampled to 16kHz should yield 1600 samples."""
        in_pcm = generate_sine_pcm(freq=500.0, sample_rate=48000, duration_sec=0.1)
        assert len(in_pcm) == 4800 * 2

        out_pcm = AudioProcessor.resample_pcm(in_pcm, 48000, 16000)
        assert len(out_pcm) == 1600 * 2

    def test_resample_pcm_16k_to_8k(self):
        """100ms at 16kHz (1600 samples) downsampled to 8kHz should yield 800 samples."""
        in_pcm = generate_sine_pcm(freq=300.0, sample_rate=16000, duration_sec=0.1)
        out_pcm = AudioProcessor.resample_pcm(in_pcm, 16000, 8000)
        assert len(out_pcm) == 800 * 2

    def test_resample_identity(self):
        """When in_rate == out_rate, returns original bytes unchanged."""
        in_pcm = b"\x01\x02\x03\x04\x05\x06"
        out_pcm = AudioProcessor.resample_pcm(in_pcm, 16000, 16000)
        assert out_pcm == in_pcm

    def test_resample_empty_and_odd_bytes(self):
        """Tests handling of empty, 1-byte, and odd-length PCM buffers."""
        assert AudioProcessor.resample_pcm(b"", 16000, 24000) == b""
        assert AudioProcessor.resample_pcm(b"\x00", 16000, 24000) == b""

        # Odd byte count (e.g. 3201 bytes -> 3200 bytes aligned -> 2400 samples out)
        sine_pcm = generate_sine_pcm(freq=440.0, sample_rate=16000, duration_sec=0.1)
        odd_pcm = sine_pcm + b"\x55"
        assert len(odd_pcm) == 3201

        out_pcm = AudioProcessor.resample_pcm(odd_pcm, 16000, 24000)
        assert len(out_pcm) == 2400 * 2

    def test_resample_invalid_sample_rates(self):
        """Zero or negative rates should raise ValueError."""
        with pytest.raises(ValueError):
            AudioProcessor.resample_pcm(b"\x00\x00", 0, 16000)
        with pytest.raises(ValueError):
            AudioProcessor.resample_pcm(b"\x00\x00", 16000, -100)

    def test_scipy_fallback_branch(self, monkeypatch):
        """Verifies that when HAS_SOXR is False, scipy.signal.resample_poly is used seamlessly."""
        monkeypatch.setattr(ap_module, "HAS_SOXR", False)

        in_pcm = generate_sine_pcm(freq=440.0, sample_rate=16000, duration_sec=0.1)
        out_pcm = AudioProcessor.resample_pcm(in_pcm, 16000, 24000)
        assert len(out_pcm) == 2400 * 2
        out_samples = AudioProcessor.pcm_to_numpy(out_pcm)
        assert np.max(np.abs(out_samples)) > 0.8


class TestAudioFormatConversions:
    """Tests conversions between PCM byte buffers and numpy float32 arrays."""

    def test_pcm_to_numpy_and_numpy_to_pcm_roundtrip(self):
        """Int16 PCM converted to numpy float32 and back should match exactly."""
        original_samples = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        pcm_bytes = original_samples.tobytes()

        float_arr = AudioProcessor.pcm_to_numpy(pcm_bytes)
        assert len(float_arr) == 5
        assert np.allclose(float_arr[0], -1.0, atol=1e-4)
        assert np.allclose(float_arr[2], 0.0, atol=1e-4)
        assert np.allclose(float_arr[4], 32767.0 / 32768.0, atol=1e-4)

        roundtrip_pcm = AudioProcessor.numpy_to_pcm(float_arr)
        roundtrip_samples = np.frombuffer(roundtrip_pcm, dtype=np.int16)
        assert np.array_equal(original_samples, roundtrip_samples)

    def test_pcm_to_numpy_edge_cases(self):
        """Tests empty and odd-length PCM conversion."""
        empty_arr = AudioProcessor.pcm_to_numpy(b"")
        assert len(empty_arr) == 0
        assert empty_arr.dtype == np.float32

        # Odd byte truncated
        arr_odd = AudioProcessor.pcm_to_numpy(b"\x00\x10\xff")
        assert len(arr_odd) == 1

    def test_numpy_to_pcm_edge_cases(self):
        """Tests None, empty array, and clipping behavior."""
        assert AudioProcessor.numpy_to_pcm(None) == b""
        assert AudioProcessor.numpy_to_pcm(np.array([], dtype=np.float32)) == b""

        # Values outside [-1.0, 1.0] should be clipped safely without overflow
        extreme_arr = np.array([-2.5, 2.5], dtype=np.float32)
        pcm = AudioProcessor.numpy_to_pcm(extreme_arr)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == -32768
        assert samples[1] == 32767


class TestAudioLevelCalculations:
    """Tests RMS, peak amplitude, and 8-band FFT spectrum calculations."""

    def test_calculate_levels_silence(self):
        """Silence should produce 0.0 for RMS, peak, and spectrum."""
        silence_pcm = np.zeros(800, dtype=np.int16).tobytes()
        levels = AudioProcessor.calculate_levels(silence_pcm, sample_rate=16000)

        assert levels["rms"] == 0.0
        assert levels["peak"] == 0.0
        assert len(levels["spectrum"]) == 8
        assert all(val == 0.0 for val in levels["spectrum"])

    def test_calculate_levels_empty_and_short(self):
        """Empty, 1-byte, and ultra-short buffers should return valid 8-band zeros."""
        res_empty = AudioProcessor.calculate_levels(b"")
        assert res_empty["rms"] == 0.0
        assert res_empty["peak"] == 0.0
        assert len(res_empty["spectrum"]) == 8

        res_one = AudioProcessor.calculate_levels(b"\x12")
        assert res_one["rms"] == 0.0
        assert len(res_one["spectrum"]) == 8

        # Short buffer (<8 samples)
        short_pcm = np.array([1000, -1000, 2000, -2000], dtype=np.int16).tobytes()
        res_short = AudioProcessor.calculate_levels(short_pcm, sample_rate=16000)
        assert len(res_short["spectrum"]) == 8
        assert res_short["peak"] > 0.0

    def test_calculate_levels_sine_metrics(self):
        """Pure 440Hz sine wave should yield theoretical RMS (~0.707) and peak (~1.0)."""
        sine_pcm = generate_sine_pcm(freq=440.0, sample_rate=16000, duration_sec=0.05, amplitude=1.0)
        levels = AudioProcessor.calculate_levels(sine_pcm, sample_rate=16000)

        assert pytest.approx(levels["peak"], rel=1e-2) == 1.0
        assert pytest.approx(levels["rms"], rel=0.1) == 0.707
        assert len(levels["spectrum"]) == 8
        assert all(0.0 <= val <= 1.0 for val in levels["spectrum"])

    def test_spectrum_frequency_selectivity(self):
        """
        440Hz tone at 16kHz sample rate (Nyquist = 8000 Hz) falls in Band 0 (0-1000 Hz).
        Band 0 should have substantially higher magnitude than higher bands (Bands 4-7).
        """
        sine_440_pcm = generate_sine_pcm(freq=440.0, sample_rate=16000, duration_sec=0.1, amplitude=0.9)
        levels_440 = AudioProcessor.calculate_levels(sine_440_pcm, sample_rate=16000)
        spec_440 = levels_440["spectrum"]

        assert spec_440[0] > spec_440[4]
        assert spec_440[0] > spec_440[7]

        # Conversely, 6000Hz tone falls in Band 6 (6000-7000 Hz)
        sine_6000_pcm = generate_sine_pcm(freq=6000.0, sample_rate=16000, duration_sec=0.1, amplitude=0.9)
        levels_6000 = AudioProcessor.calculate_levels(sine_6000_pcm, sample_rate=16000)
        spec_6000 = levels_6000["spectrum"]

        assert spec_6000[6] > spec_6000[0]
        assert spec_6000[6] > spec_6000[1]
