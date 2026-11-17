"""Prove the outbound half of the Phone Link bridge with an audible beep.

Plays a 440 Hz tone into CABLE Input in 2-second bursts, while recording the
Realtek loopback (what the caller says) and CABLE Output (what we inject). If
the person on the other end hears a repeating beep, Phone Link is taking its
microphone from the virtual cable and the AI's voice will reach callers.

    python -m agent.probe_outbound --seconds 180
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np
import soundcard as sc

RATE = 16000
CABLE_IN = "CABLE Input (VB-Audio Virtual Cable)"
CABLE_OUT = "CABLE Output (VB-Audio Virtual Cable)"
LOOPBACK = "Speakers (Realtek(R) Audio)"


def _safe_name(device) -> str | None:
    try:
        return str(device.name)
    except Exception:  # noqa: BLE001
        return None


def _find(kind: str, name: str, loopback: bool | None = None):
    collection = (
        sc.all_speakers()
        if kind == "playback"
        else sc.all_microphones(include_loopback=True)
    )
    for device in collection:
        if _safe_name(device) != name:
            continue
        if loopback is not None and bool(getattr(device, "isloopback", False)) != loopback:
            continue
        return device
    return None


def _dbfs(data: np.ndarray) -> float:
    peak = float(np.abs(np.asarray(data, dtype=np.float64)).max()) if len(data) else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else -120.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.probe_outbound")
    parser.add_argument("--seconds", type=float, default=180.0)
    args = parser.parse_args(argv)

    speaker = _find("playback", CABLE_IN)
    caller = _find("capture", LOOPBACK, loopback=True)
    injected = _find("capture", CABLE_OUT, loopback=False)
    if not speaker or not caller or not injected:
        print("missing device(s):", bool(speaker), bool(caller), bool(injected), flush=True)
        return 1

    t = np.arange(RATE, dtype=np.float32) / RATE
    tone = (0.20 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence = np.zeros(RATE, dtype=np.float32)
    stop = threading.Event()

    def _play() -> None:
        try:
            with speaker.player(samplerate=RATE, channels=1, blocksize=RATE) as player:
                while not stop.is_set():
                    player.play(tone)
                    if stop.wait(2.0):
                        break
                    player.play(silence)
                    if stop.wait(2.0):
                        break
        except Exception as exc:  # noqa: BLE001
            print(f"playback error: {type(exc).__name__}: {exc}", flush=True)

    print("440 Hz beep: 2 s on / 2 s off into CABLE Input", flush=True)
    print("place a Phone Link call and ask the other person if they hear a repeating beep\n", flush=True)
    threading.Thread(target=_play, daemon=True).start()

    started = time.monotonic()
    try:
        with caller.recorder(samplerate=RATE, channels=1, blocksize=RATE) as rec_in, \
             injected.recorder(samplerate=RATE, channels=1, blocksize=RATE) as rec_out:
            print(f"{'':>9}  {'caller':>8} {'inject':>8}  phase", flush=True)
            while time.monotonic() - started < args.seconds:
                elapsed = time.monotonic() - started
                caller_level = _dbfs(rec_in.record(numframes=RATE))
                inject_level = _dbfs(rec_out.record(numframes=RATE))
                phase = "BEEP" if int(elapsed) % 4 < 2 else "----"
                print(
                    f"[{elapsed:6.1f}s]  {caller_level:8.1f} {inject_level:8.1f}  {phase}",
                    flush=True,
                )
    finally:
        stop.set()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
