"""Record CABLE Output for a few seconds and report the peak level.

Used to confirm the AI's audio actually reaches the virtual mic during a
greeting, before spending a real phone call on the test.
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np
import soundcard as sc

NAME = "CABLE Output (VB-Audio Virtual Cable)"
RATE = 16000


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.record_cable")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--save", default="logs/cable_greeting.wav")
    args = parser.parse_args()

    mic = next(
        (
            m
            for m in sc.all_microphones(include_loopback=True)
            if m.name == NAME and not getattr(m, "isloopback", False)
        ),
        None,
    )
    if mic is None:
        print("CABLE Output not found", flush=True)
        return 1

    chunks: list[np.ndarray] = []
    print(f"recording {NAME} for {args.seconds:.0f}s ...", flush=True)
    started = time.monotonic()
    with mic.recorder(samplerate=RATE, channels=1, blocksize=1600) as rec:
        while time.monotonic() - started < args.seconds:
            chunks.append(np.asarray(rec.record(numframes=1600), dtype=np.float32).reshape(-1))

    data = np.concatenate(chunks)
    peak = float(np.abs(data).max())
    db = 20 * np.log10(peak) if peak > 0 else -240.0
    print(f"peak {db:.1f} dBFS  -> {'AUDIO PRESENT' if db > -50 else 'SILENT'}", flush=True)

    if args.save:
        from pathlib import Path

        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm.tobytes())
        print(f"saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
