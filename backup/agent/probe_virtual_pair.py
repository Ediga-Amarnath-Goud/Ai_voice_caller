"""Check whether an existing virtual speaker/mic pair can carry the AI voice.

A virtual cable is just a playback device whose output appears as a recording
device. AudioRelay already installed a "Virtual Speakers" / "Virtual Mic" pair;
if that pair is intact we get the outbound leg of the bridge for free. This
plays a 1 kHz tone into Virtual Speakers and records from Virtual Mic.

    python -m agent.probe_virtual_pair
"""

from __future__ import annotations

import argparse
import math
import threading
import time

import numpy as np
import soundcard as sc

SPEAKER = "CABLE Input (VB-Audio Virtual Cable)"
MIC = "CABLE Output (VB-Audio Virtual Cable)"
RATE = 16000


def _safe_name(device) -> str | None:
    try:
        return str(device.name)
    except Exception:  # noqa: BLE001 - dead endpoints throw on .name
        return None


def _find(kind: str, target: str):
    collection = sc.all_speakers() if kind == "playback" else sc.all_microphones(
        include_loopback=True
    )
    for device in collection:
        if _safe_name(device) == target:
            return device
    return None


def _dbfs(data: np.ndarray) -> float:
    peak = float(np.abs(np.asarray(data, dtype=np.float64)).max()) if len(data) else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else -120.0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.probe_virtual_pair")
    parser.add_argument("--speaker", default=SPEAKER)
    parser.add_argument("--mic", default=MIC)
    args = parser.parse_args()

    print("playback devices:")
    for device in sc.all_speakers():
        print(f"  {_safe_name(device) or '<unreadable>'}")
    print("recording devices:")
    for device in sc.all_microphones(include_loopback=True):
        flag = "LOOPBACK" if getattr(device, "isloopback", False) else "MIC"
        print(f"  {flag:<8} {_safe_name(device) or '<unreadable>'}")
    print()

    speaker = _find("playback", args.speaker)
    mic = _find("capture", args.mic)
    if speaker is None or mic is None:
        print(f"pair unavailable: speaker={speaker is not None} mic={mic is not None}")
        return 1
    print(f"testing: {args.speaker}  ->  {args.mic}")

    t = np.arange(RATE, dtype=np.float32) / RATE
    tone = (0.25 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    def _play() -> None:
        try:
            with speaker.player(samplerate=RATE, channels=1, blocksize=800) as player:
                for _ in range(6):
                    player.play(tone)
                    time.sleep(0.12)
        except Exception as exc:  # noqa: BLE001
            print(f"  playback failed: {type(exc).__name__}: {exc}", flush=True)

    print("playing 1 kHz tone into Virtual Speakers, recording Virtual Mic...")
    player_thread = threading.Thread(target=_play, daemon=True)
    player_thread.start()
    time.sleep(0.25)
    try:
        with mic.recorder(samplerate=RATE, channels=1, blocksize=800) as recorder:
            level = _dbfs(recorder.record(numframes=RATE // 2))
    except Exception as exc:  # noqa: BLE001
        print(f"  capture failed: {type(exc).__name__}: {exc}", flush=True)
        return 1
    player_thread.join(timeout=10)

    verdict = "WORKS - usable as a virtual cable" if level > -50 else "DEAD - no signal"
    print(f"  Virtual Mic level: {level:7.1f} dBFS  -> {verdict}")
    return 0 if level > -50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
