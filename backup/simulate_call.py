"""End-to-end call simulation with no phone required.

Injects a recorded caller into the REAL endpoint the daemon captures
(Realtek speakers loopback) while recording CABLE Output -- the stream that
would travel to the caller through Phone Link's microphone.

Start the daemon first, then run:

    python simulate_call.py 30 logs/caller_sim.wav

A healthy run shows "call active" + "Triggering greeting" + "agent :" in the
daemon log and "AI AUDIO PRESENT" here.
"""
import sys, time, wave
import numpy as np
import soundcard as sc

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
WAV_IN = sys.argv[2] if len(sys.argv) > 2 else "logs/caller_sim.wav"
WAV_OUT = "logs/sim_cable_out.wav"

with wave.open(WAV_IN, "rb") as w:
    src_rate = w.getframerate()
    ch = w.getnchannels()
    pcm = w.readframes(w.getnframes())
data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
if ch == 2:
    data = data.reshape(-1, 2).mean(axis=1)
fixture_db = 20 * np.log10(float(np.abs(data).max()))
print(f"caller fixture: {src_rate} Hz, {len(data)/src_rate:.2f}s, peak {fixture_db:.1f} dBFS")

spk = next(s for s in sc.all_speakers() if s.name == "Speakers (Realtek(R) Audio)")
cable = next(m for m in sc.all_microphones(include_loopback=True)
             if m.name == "CABLE Output (VB-Audio Virtual Cable)" and not m.isloopback)

CAP = 48000
rec_block = int(CAP / 20)          # 50 ms capture blocks
play_block = int(src_rate / 20)

rec = cable.recorder(samplerate=CAP, channels=1, blocksize=rec_block)
rec.__enter__()
chunks = []
start = time.monotonic()
idx = 0
with spk.player(samplerate=src_rate, channels=1, blocksize=play_block) as p:
    while time.monotonic() - start < DURATION:
        if idx + play_block > len(data):
            idx = 0
        p.play(data[idx:idx + play_block])
        idx += play_block
        chunks.append(np.asarray(rec.record(numframes=rec_block), np.float32).reshape(-1))
rec.__exit__(None, None, None)

cap = np.concatenate(chunks)
with wave.open(WAV_OUT, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(CAP)
    w.writeframes((np.clip(cap, -1.0, 1.0) * 32767).astype("<i2").tobytes())

peak = float(np.abs(cap).max())
db = 20 * np.log10(peak) if peak > 0 else -240
print(f"CABLE Output recorded: peak {db:.1f} dBFS, {len(cap)/CAP:.1f}s -> {WAV_OUT}")
print("AI AUDIO PRESENT" if db > -50 else "AI AUDIO MISSING")
