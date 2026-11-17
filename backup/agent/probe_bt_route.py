"""Inspect the hands-free endpoints only while the phone is actually on Bluetooth.

Every previous probe opened the endpoints at a fixed offset from "call active",
which is the wrong trigger: the phone spends the first part of a call on its own
earpiece and only moves to Bluetooth later, sometimes only when the operator
forces it. Measuring before that moment says nothing about the endpoint.

This waits for the *route*, then measures.

    python -m agent.probe_bt_route --window 120
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from .phone import Phone
from .probe_shared import _sweep
from .watch_call import _active_route, _overall, _precise_states, _try_open


def _default_adb() -> str:
    candidate = Path(__file__).resolve().parent.parent / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


async def _main(args: argparse.Namespace) -> int:
    phone = Phone(args.adb, args.serial)
    print(f"phone  : {await phone.ensure_connected()}", flush=True)
    print(f"state  : {_overall(await _precise_states(phone))}", flush=True)
    print("waiting for a call ...", flush=True)

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

    print("active -- waiting for the route to become BT_SCO", flush=True)
    print("(switch the call audio to Bluetooth on the phone if it stays on earpiece)", flush=True)

    previous_route = ""
    sweeps = 0
    started = time.monotonic()
    while time.monotonic() - started < args.window:
        try:
            route = await _active_route(phone)
        except Exception:  # noqa: BLE001
            await phone.reconnect()
            await asyncio.sleep(1.0)
            continue

        if route != previous_route:
            elapsed = time.monotonic() - started
            print(f"\n[{elapsed:6.1f}s] route: {route}", flush=True)
            previous_route = route

        if "BT_SCO" not in route:
            await asyncio.sleep(0.5)
            continue

        elapsed = time.monotonic() - started
        print(f"\n[{elapsed:6.1f}s] BT_SCO detected -- sweeping", flush=True)

        core = await asyncio.to_thread(_sweep, args.match)
        for key, value in core.items():
            print(f"     core   {key:<14} {value}", flush=True)

        exclusive = await asyncio.to_thread(
            _try_open, "render", args.output_device, is_input=False
        )
        print(f"     wdmks  {exclusive}", flush=True)
        exclusive = await asyncio.to_thread(
            _try_open, "capture", args.input_device, is_input=True
        )
        print(f"     wdmks  {exclusive}", flush=True)

        sweeps += 1
        if sweeps >= args.rounds:
            break
        await asyncio.sleep(args.interval)

    print(f"\nfinal route: {await _active_route(phone)}", flush=True)
    print(f"sweeps on BT: {sweeps}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.probe_bt_route")
    parser.add_argument("--seconds", type=float, default=600.0, help="wait for a call")
    parser.add_argument("--window", type=float, default=120.0, help="watch the call")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--match", default="iQOO")
    parser.add_argument("--input-device", default="iQOO Z9 5G")
    parser.add_argument("--output-device", default="iQOO Z9 5G")
    parser.add_argument("--adb", default=_default_adb())
    parser.add_argument("--serial")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
