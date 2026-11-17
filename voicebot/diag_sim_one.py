"""Single-turn diagnostic: play ONE sentence, then record AI response.

1. Plays first ~2s of caller_sim.wav (one greeting/sentence)
2. Stops playback
3. Records CABLE Output for ~15s to capture AI response
"""
import sys, time, wave
from pathlib import Path
import numpy as np
import soundcard as sc

DURATION_AFTER = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
WAV_IN = Path(__file__).resolve().parent.parent / "backup" / "logs" / "caller_sim.wav"
WAV_OUT = Path(__file__).resolve().parent.parent / "voicebot_diag_one_turn.wav"

with wave.open(str(WAV_IN), "rb") as w:
    src_rate = w.getframerate(); ch = w.getnchannels()
    pcm = w.readframes(w.getnframes())
data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
if ch == 2:
    data = data.reshape(-1, 2).mean(axis=1)

# Take only first 2 seconds (first sentence)
sentence = data[:int(src_rate * 2.0)]
print(f"caller fixture: {src_rate} Hz, sentence {len(sentence)/src_rate:.2f}s, "
      f"peak {20*np.log10(float(np.abs(sentence).max())):.1f} dBFS", flush=True)

spk = next(s for s in sc.all_speakers() if s.name == "Speakers (Realtek(R) Audio)")
cable = next(m for m in sc.all_microphones(include_loopback=True)
             if m.name == "CABLE Output (VB-Audio Virtual Cable)" and not m.isloopback)

CAP = 44100
block = int(CAP / 20)
play_block = int(src_rate / 20)

rec = cable.recorder(samplerate=CAP, channels=1, blocksize=block)
rec.__enter__()

chunks = []
# 1. Play the sentence
with spk.player(samplerate=src_rate, channels=1, blocksize=play_block) as p:
    p.play(sentence)
    time.sleep(0.2)  # let it finish playing

# 2. Wait and record AI response
print(f"Recording AI response for {DURATION_AFTER}s...", flush=True)
start = time.monotonic()
while time.monotonic() - start < DURATION_AFTER:
    chunks.append(np.asarray(rec.record(numframes=block), np.float32).reshape(-1))

rec.__exit__(None, None, None)

cap = np.concatenate(chunks)
with wave.open(str(WAV_OUT), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(CAP)
    w.writeframes((np.clip(cap, -1.0, 1.0) * 32767).astype("<i2").tobytes())

peak = float(np.abs(cap).max())
db = 20 * np.log10(peak) if peak > 0 else -240
rms = float(np.sqrt((cap**2).mean())) if cap.size else 0.0
rms_db = 20 * np.log10(rms) if rms > 0 else -240
print(f"AI response: {len(cap)/CAP:.1f}s peak {db:.1f} dBFS RMS {rms_db:.1f} dBFS -> {WAV_OUT.name}", flush=True)
print("AI AUDIO PRESENT" if db > -50 else "AI AUDIO SILENT", flush=True)