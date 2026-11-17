"""Caller simulation for the fresh build.

Injects a recorded caller into the REAL endpoint the daemon captures
(Realtek speakers loopback) and records CABLE Output -- the stream that
would reach the caller through Phone Link. Run the daemon first.

    python -m voicebot.simulate 45
"""
import sys, time, wave
from pathlib import Path
import numpy as np
import soundcard as sc

from . import config

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
WAV_IN = Path(__file__).resolve().parent.parent / "backup" / "logs" / "caller_sim.wav"
WAV_OUT = Path(__file__).resolve().parent.parent / "voicebot_sim_out.wav"

with wave.open(str(WAV_IN), "rb") as w:
    src_rate = w.getframerate()
    ch = w.getnchannels()
    pcm = w.readframes(w.getnframes())
data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
if ch == 2:
    data = data.reshape(-1, 2).mean(axis=1)
print(f"caller fixture: {src_rate} Hz, {len(data)/src_rate:.2f}s, "
      f"peak {20*np.log10(float(np.abs(data).max())):.1f} dBFS", flush=True)

spk = next(s for s in sc.all_speakers() if s.name == config.LOOPBACK_DEVICE)
cable = next(m for m in sc.all_microphones(include_loopback=True)
             if m.name == "CABLE Output (VB-Audio Virtual Cable)" and not m.isloopback)

CAP = 44100
rec_block = int(CAP / 20)
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
with wave.open(str(WAV_OUT), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(CAP)
    w.writeframes((np.clip(cap, -1.0, 1.0) * 32767).astype("<i2").tobytes())

peak = float(np.abs(cap).max())
db = 20 * np.log10(peak) if peak > 0 else -240
# Measure only the back half (after greeting) to see if AI responded to speech
half = len(cap) // 2
hpeak = float(np.abs(cap[half:]).max())
hdb = 20 * np.log10(hpeak) if hpeak > 0 else -240
print(f"AI output recorded: {len(cap)/CAP:.1f}s -> {WAV_OUT.name}", flush=True)
print(f"  overall peak : {db:.1f} dBFS", flush=True)
print(f"  back-half peak: {hdb:.1f} dBFS", flush=True)
print("AI AUDIO PRESENT" if hdb > -50 else "AI AUDIO (back half) QUIET", flush=True)
