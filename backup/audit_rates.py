"""Audit: does Realtek loopback capture work at the daemon's 24 kHz rate?"""
import soundcard as sc, numpy as np, wave, time

with wave.open('logs/caller_sim.wav', 'rb') as w:
    rate = w.getframerate(); ch = w.getnchannels()
    pcm = w.readframes(w.getnframes())
data = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
if ch == 2:
    data = data.reshape(-1, 2).mean(axis=1)

spk = next(s for s in sc.all_speakers() if s.name == 'Speakers (Realtek(R) Audio)')
mic = next(m for m in sc.all_microphones(include_loopback=True)
           if m.name == 'Speakers (Realtek(R) Audio)' and m.isloopback)

for cap_rate in (16000, 24000, 44100, 48000):
    block = int(cap_rate / 100)  # 10 ms, same as daemon
    try:
        rec = mic.recorder(samplerate=cap_rate, channels=1, blocksize=block)
        rec.__enter__()
        with spk.player(samplerate=rate, channels=1, blocksize=1600) as p:
            p.play(data)
            time.sleep(0.15)
            chunks = [np.asarray(rec.record(numframes=block), np.float32).reshape(-1)
                      for _ in range(60)]
        rec.__exit__(None, None, None)
        cap = np.concatenate(chunks)
        peak = float(np.abs(cap).max())
        db = 20 * np.log10(peak) if peak > 0 else -240
        print(f'loopback @ {cap_rate:>5d} Hz: {db:7.1f} dBFS  -> {"OK" if db>-60 else "SILENT"}')
    except Exception as exc:
        print(f'loopback @ {cap_rate:>5d} Hz: ERROR {type(exc).__name__}: {exc}')
