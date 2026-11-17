"""Continuous-tone loopback test: removes short-WAV timing as a variable."""
import soundcard as sc, numpy as np, time

spk = next(s for s in sc.all_speakers() if s.name == 'Speakers (Realtek(R) Audio)')
mic = next(m for m in sc.all_microphones(include_loopback=True)
           if m.name == 'Speakers (Realtek(R) Audio)' and m.isloopback)

PLAY_RATE = 48000
t = np.arange(PLAY_RATE) / PLAY_RATE
tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # 1 s of 440 Hz

for cap_rate in (16000, 24000, 48000):
    block = int(cap_rate / 100)
    try:
        rec = mic.recorder(samplerate=cap_rate, channels=1, blocksize=block)
        rec.__enter__()
        chunks = []
        with spk.player(samplerate=PLAY_RATE, channels=1, blocksize=4800) as p:
            # play continuously for the capture window
            for _ in range(20):          # ~1 s worth of 10 ms capture blocks
                p.play(tone[:block * 3])  # enough to cover the read
                chunks.append(np.asarray(rec.record(numframes=block), np.float32).reshape(-1))
        rec.__exit__(None, None, None)
        cap = np.concatenate(chunks)
        peak = float(np.abs(cap).max())
        db = 20 * np.log10(peak) if peak > 0 else -240
        print(f'CONTINUOUS loopback @ {cap_rate:>5d} Hz: {db:7.1f} dBFS -> {"OK" if db>-60 else "SILENT"}')
    except Exception as exc:
        print(f'CONTINUOUS loopback @ {cap_rate:>5d} Hz: ERROR {type(exc).__name__}: {exc}')
