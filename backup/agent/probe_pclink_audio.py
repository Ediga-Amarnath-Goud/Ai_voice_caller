"""Find which Windows endpoint actually carries Phone Link's call audio.

Bluetooth is a dead end on this machine: the hands-free endpoint never
materialises as a Core Audio device, yet a Phone Link call has working audio in
both directions. So Phone Link renders the caller into *some* ordinary Windows
playback device and reads an ordinary capture device for our voice.

Every render endpoint has a loopback twin, so the caller's voice can be measured
without any virtual cable. This arms every loopback on the machine and prints
RMS (the speech discriminator) plus peak, once a second, with the phone's own
audio route. It also saves the raw audio so the trace can be re-examined.

    python -m agent.probe_pclink_audio --seconds 240

Read it like this:

  * one endpoint's RMS sits around -20..-40 dBFS for the whole call -> that is
    the endpoint Phone Link uses, and the caller's voice is capturable in
    software with no extra hardware;
  * every endpoint stays at -96 dBFS while you hear them -> the call audio never
    enters Windows, and Phone Link is not the bridge.
"""

from __future__ import annotations

import argparse
import math
import time
import wave
from pathlib import Path

import numpy as np

from .watch_endpoint import _adb_path, _route, _serial

RATE = 16000


def _stats(frames: np.ndarray) -> tuple[float, float]:
    """(peak dBFS, RMS dBFS) of a float32 recording."""
    if frames is None or not len(frames):
        return -120.0, -120.0
    data = np.asarray(frames, dtype=np.float64)
    peak = float(np.abs(data).max())
    rms = float(np.sqrt(np.mean(np.square(data))))
    to_db = lambda value: 20.0 * math.log10(value) if value > 0 else -120.0
    return to_db(peak), to_db(rms)


def _safe_name(device) -> str | None:
    """A dead endpoint can throw just from having its name read."""
    try:
        return str(device.name)
    except Exception:  # noqa: BLE001 - a broken device is not fatal
        return None


def _short(name: str, width: int = 16) -> str:
    """'Speakers (Realtek(R) Audio)' -> 'Speakers'."""
    head = name.split(" (")[0].strip() or name
    return head[:width]


def _phone_epoch() -> str:
    """The phone's clock, for correlating with its call log (may be set wrong)."""
    import subprocess

    try:
        serial = _serial()
        if not serial:
            return "offline"
        result = subprocess.run(
            [_adb_path(), "-s", serial, "shell", "date", "+%s"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return result.stdout.strip() or "offline"
    except Exception:  # noqa: BLE001 - ADB is optional for the level measurement
        return "offline"


class Captures:
    """Every loopback twin on the machine, opened once and kept armed."""

    def __init__(self, samplerate: int = RATE) -> None:
        import soundcard as sc

        self._recorders: dict[str, object] = {}
        self._names: dict[str, str] = {}

        try:
            raw = sc.all_microphones(include_loopback=True)
        except Exception as exc:  # noqa: BLE001 - enumeration can fail wholesale
            print(f"  microphone enumeration failed: {type(exc).__name__}: {exc}", flush=True)
            raw = []

        for device in raw:
            if not bool(getattr(device, "isloopback", False)):
                continue
            name = _safe_name(device)
            if name is None:
                continue
            label = _short(name)
            if label in self._recorders:
                label = f"{label}{len(self._recorders) + 1}"
            try:
                recorder = device.recorder(samplerate=samplerate, channels=1, blocksize=800)
                recorder.__enter__()  # allocates the stream; required before record()
            except Exception as exc:  # noqa: BLE001 - a dead endpoint is not fatal
                print(f"  cannot open {name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            self._recorders[label] = recorder
            self._names[label] = name

    def __bool__(self) -> bool:
        return bool(self._recorders)

    @property
    def labels(self) -> list[str]:
        return list(self._recorders)

    def name(self, label: str) -> str:
        return self._names[label]

    def read(self, label: str, frames: int) -> np.ndarray:
        try:
            return np.asarray(
                self._recorders[label].record(numframes=frames),  # type: ignore[union-attr]
                dtype=np.float32,
            ).reshape(-1)
        except Exception:  # noqa: BLE001 - a hiccup must not end the run
            return np.zeros(frames, dtype=np.float32)

    def close(self) -> None:
        for recorder in self._recorders.values():
            try:
                recorder.__exit__(None, None, None)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - teardown only
                pass


def _save_wav(path: Path, chunks: list[np.ndarray]) -> None:
    if not chunks:
        return
    pcm = np.concatenate(chunks)
    pcm = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    print(f"saved {path} ({len(pcm) / RATE:.1f}s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.probe_pclink_audio")
    parser.add_argument("--seconds", type=float, default=240.0)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--save", default="logs", help="directory for the raw wav files")
    args = parser.parse_args(argv)

    caps = Captures()
    if not caps:
        print("no loopback endpoints could be opened", flush=True)
        return 1

    print(f"PC epoch: {int(time.time())}   phone epoch: {_phone_epoch()}", flush=True)
    print(f"armed {len(caps.labels)} loopback(s):", flush=True)
    for label in caps.labels:
        print(f"  {label:<16} <- {caps.name(label)}", flush=True)
    print("\nplace the call from Phone Link and talk normally\n", flush=True)

    frames = int(RATE * args.window)
    chunks: dict[str, list[np.ndarray]] = {label: [] for label in caps.labels}
    active_seconds: dict[str, list[float]] = {label: [] for label in caps.labels}
    peak_rms: dict[str, tuple[float, float]] = {label: (-120.0, 0.0) for label in caps.labels}

    print(f"{'':>9}  {'RMS dBFS':>9} {'peak dBFS':>10}   state   route", flush=True)
    started = time.monotonic()
    last_route = ""
    last_route_at = [-999.0]
    while time.monotonic() - started < args.seconds:
        elapsed = time.monotonic() - started
        samples = {label: caps.read(label, frames) for label in caps.labels}
        stats = {label: _stats(samples[label]) for label in caps.labels}

        for label in caps.labels:
            chunks[label].append(samples[label])
            rms = stats[label][1]
            if rms > -45.0:
                active_seconds[label].append(elapsed)
            if rms > peak_rms[label][0]:
                peak_rms[label] = (rms, elapsed)

        # Report the loudest endpoint; with one loopback this is simply it.
        label = max(caps.labels, key=lambda key: stats[key][1])
        peak, rms = stats[label]
        state = "AUDIO" if rms > -45.0 else ("quiet" if rms > -70.0 else "silent")

        route = ""
        if elapsed < 3 or (elapsed - last_route_at[0] >= 5):
            last_route = _route()
            last_route_at[0] = elapsed
            route = f"  {last_route}"
        print(f"[{elapsed:6.1f}s]  {rms:9.1f} {peak:10.1f}   {state:<6}{route}", flush=True)

    caps.close()

    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        for label in caps.labels:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
            _save_wav(out_dir / f"pclink_{safe}.wav", chunks[label])

    print("\nsummary:", flush=True)
    for label in caps.labels:
        active = active_seconds[label]
        span = f"{active[0]:.1f}-{active[-1]:.1f}s" if active else "none"
        best_rms, best_at = peak_rms[label]
        print(
            f"  {label:<16} active {len(active):>4}s  span {span:<16} "
            f"peak RMS {best_rms:6.1f} dBFS at {best_at:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
