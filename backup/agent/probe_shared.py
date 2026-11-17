"""Watch the hands-free endpoints through Core Audio while a call is live.

Every earlier attempt used PortAudio's WDM-KS host API, which opens devices
*exclusively*. If another client (Phone Link, the audio engine) holds the
hands-free endpoint in shared mode, an exclusive open fails with the same -9999
regardless of link state -- so that error can never distinguish "device is dead"
from "device is busy".

This polls Core Audio during the call instead: does the endpoint materialise at
all, and does a *shared* stream open?

    python -m agent.probe_shared --seconds 900
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import soundcard as sc

from .phone import Phone
from .watch_call import _active_route, _overall, _precise_states


def _default_adb() -> str:
    candidate = Path(__file__).resolve().parent.parent / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


def _sweep(match: str) -> dict[str, object]:
    speakers = sc.all_speakers()
    mics = sc.all_microphones(include_loopback=True)
    hit_speakers = [s for s in speakers if match.lower() in s.name.lower()]
    hit_mics = [m for m in mics if match.lower() in m.name.lower()]

    out: dict[str, object] = {
        "render_total": len(speakers),
        "capture_total": len(mics),
        "render_hits": [s.name for s in hit_speakers],
        "capture_hits": [m.name for m in hit_mics],
    }

    for speaker in hit_speakers:
        try:
            with speaker.player(samplerate=16000, channels=1, blocksize=320) as player:
                player.play(b"\x00\x00" * 800)
            out["render_open"] = "OK (shared)"
        except Exception as exc:  # noqa: BLE001 - report what Core Audio says
            out["render_open"] = f"FAILED -> {type(exc).__name__}: {exc}"

    for mic in hit_mics:
        if "loopback" in mic.name.lower():
            continue
        try:
            with mic.recorder(samplerate=16000, channels=1, blocksize=320) as recorder:
                frames = recorder.record(numframes=800)
            peak = float(abs(frames).max()) if len(frames) else 0.0
            out["capture_open"] = f"OK (shared), peak {peak:.3f}"
        except Exception as exc:  # noqa: BLE001
            out["capture_open"] = f"FAILED -> {type(exc).__name__}: {exc}"

    return out


async def _main(args: argparse.Namespace) -> int:
    phone = Phone(args.adb, args.serial)
    print(f"phone  : {await phone.ensure_connected()}", flush=True)
    print(f"state  : {_overall(await _precise_states(phone))}", flush=True)
    print("waiting for a call to become active ...", flush=True)

    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            if _overall(await _precise_states(phone)) == "ACTIVE":
                break
        except Exception:  # noqa: BLE001
            await phone.reconnect()
        await asyncio.sleep(0.4)
    else:
        print("no active call before the timeout", flush=True)
        return 1

    print("active -- sweeping Core Audio\n", flush=True)
    for index in range(args.rounds):
        route = await _active_route(phone)
        print(f"[{index}] route: {route}", flush=True)
        result = await asyncio.to_thread(_sweep, args.match)
        for key, value in result.items():
            print(f"     {key:<14} {value}", flush=True)
        print(flush=True)
        await asyncio.sleep(args.interval)

    print(f"final route: {await _active_route(phone)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.probe_shared")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--match", default="iQOO")
    parser.add_argument("--adb", default=_default_adb())
    parser.add_argument("--serial")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
