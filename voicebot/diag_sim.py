"""Diagnostic simulation: inject caller into loopback, record AI output from CABLE,
and dump both WAVs plus a transcript of what Gemini heard/spoke."""
import sys, time, wave
from pathlib import Path
import numpy as np
import soundcard as sc

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
WAV_IN = Path(__file__).resolve().parent.parent / "backup" / "logs" / "caller_sim.wav"
WAV_LOOP = Path(__file__).resolve().parent.parent / "voicebot_diag_loopback.wav"
WAV_OUT = Path(__file__).resolve().parent.parent / "voicebot_diag_cable_out.wav"

with wave.open(str(WAV_IN), "rb") as w:
    src_rate = w.getframerate(); ch = w.getnchannels()
    pcm = w.readframes(w.getnframes())
data = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
if ch == 2:
    data = data.reshape(-1, 2).mean(axis=1)
print(f"caller fixture: {src_rate} Hz, {len(data)/src_rate:.2f}s, peak {20*np.log10(float(np.abs(data).max())):.1f} dBFS", flush=True)

spk = next(s for s in sc.all_speakers() if s.name == "Speakers (Realtek(R) Audio)")
loop = next(m for m in sc.all_microphones(include_loopback=True)
            if m.name == "Speakers (Realtek(R) Audio)" and m.isloopback)
cable = next(m for m in sc.all_microphones(include_loopback=True)
             if m.name == "CABLE Output (VB-Audio Virtual Cable)" and not m.isloopback)

CAP_LOOP = 24000
CAP_OUT = 44100
loop_block = int(CAP_LOOP / 20)
out_block = int(CAP_OUT / 20)
play_block = int(src_rate / 20)

loop_rec = loop.recorder(samplerate=CAP_LOOP, channels=1, blocksize=loop_block)
loop_rec.__enter__()
out_rec = cable.recorder(samplerate=CAP_OUT, channels=1, blocksize=out_block)
out_rec.__enter__()

loop_chunks = []
out_chunks = []
start = time.monotonic()
idx = 0
with spk.player(samplerate=src_rate, channels=1, blocksize=play_block) as p:
    while time.monotonic() - start < DURATION:
        if idx + play_block > len(data):
            idx = 0
        p.play(data[idx:idx + play_block])
        idx += play_block
        loop_chunks.append(np.asarray(loop_rec.record(numframes=loop_block), np.float32).reshape(-1))
        out_chunks.append(np.asarray(out_rec.record(numframes=out_block), np.float32).reshape(-1))
loop_rec.__exit__(None, None, None)
out_rec.__exit__(None, None, None)

loop_cap = np.concatenate(loop_chunks)
out_cap = np.concatenate(out_chunks)

for name, cap, rate, path in (
    ("LOOPBACK", loop_cap, CAP_LOOP, WAV_LOOP),
    ("CABLE OUT", out_cap, CAP_OUT, WAV_OUT),
):
    peak = float(np.abs(cap).max())
    db = 20 * np.log10(peak) if peak > 0 else -240
    rms = float(np.sqrt((cap**2).mean())) if cap.size else 0.0
    rms_db = 20 * np.log10(rms) if rms > 0 else -240
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(cap, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    print(f"{name}: {len(cap)/rate:.1f}s peak {db:.1f} dBFS RMS {rms_db:.1f} dBFS -> {path.name}", flush=True)

print("Done.", flush=True)