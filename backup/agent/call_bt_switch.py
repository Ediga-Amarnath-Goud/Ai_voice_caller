"""Dial a call, force its audio onto Bluetooth from ADB, and measure the endpoint.

The phone can be pushed onto BT_SCO by tapping the in-call audio-route control.
If that alone produced working call audio, the fallback would stop mattering --
so this test does exactly that and then asks Windows whether an endpoint exists.

    python -m agent.call_bt_switch                 # wait for you to dial
    python -m agent.call_bt_switch --number 98...  # dial it here
    python -m agent.call_bt_switch --number ... --hangup
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path
from xml.etree import ElementTree

from .phone import Phone
from .probe_shared import _sweep
from .watch_call import _active_route, _overall, _precise_states, _try_open

_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def _default_adb() -> str:
    candidate = Path(__file__).resolve().parent.parent / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


async def _nodes(phone: Phone) -> list[dict[str, str]]:
    """Dump the visible UI hierarchy and return every node's identifying fields."""
    out = await phone._adb("shell", "uiautomator dump /sdcard/ui.xml", timeout=30)
    if "dumped" not in out.lower() and "error" in out.lower():
        return []
    xml = await phone._adb("shell", "cat /sdcard/ui.xml", timeout=30)
    start = xml.find("<?xml")
    if start < 0:
        return []
    try:
        root = ElementTree.fromstring(xml[start:])
    except ElementTree.ParseError:
        return []

    nodes: list[dict[str, str]] = []
    for node in root.iter("node"):
        nodes.append(
            {
                "text": node.get("text", ""),
                "desc": node.get("content-desc", ""),
                "id": node.get("resource-id", ""),
                "clickable": node.get("clickable", "false"),
                "bounds": node.get("bounds", ""),
            }
        )
    return nodes


def _center(bounds: str) -> tuple[int, int] | None:
    match = _BOUNDS_RE.match(bounds)
    if not match:
        return None
    x1, y1, x2, y2 = (int(value) for value in match.groups())
    return (x1 + x2) // 2, (y1 + y2) // 2


def _blob(node: dict[str, str]) -> str:
    return f"{node['text']} {node['desc']} {node['id']}".lower()


def _find_route_control(nodes: list[dict[str, str]]) -> dict[str, str] | None:
    """Prefer an obvious Bluetooth control, fall back to the audio-route menu."""
    candidates = [node for node in nodes if "bluetooth" in _blob(node)]
    for node in candidates:
        if node["clickable"] == "true":
            return node
    if candidates:
        return candidates[0]
    for keyword in ("audio route", "route", "speaker", "audio", "more", "overflow"):
        for node in nodes:
            if keyword in _blob(node) and node["clickable"] == "true":
                return node
    return None


async def _tap(phone: Phone, node: dict[str, str]) -> bool:
    centre = _center(node["bounds"])
    if centre is None:
        return False
    await phone._adb("shell", "input", "tap", str(centre[0]), str(centre[1]))
    return True


async def _measure(phone: Phone, args: argparse.Namespace, label: str) -> None:
    route = await _active_route(phone)
    print(f"     {label:<10} route: {route}", flush=True)
    core = await asyncio.to_thread(_sweep, args.match)
    for key, value in core.items():
        print(f"                core {key:<13} {value}", flush=True)
    render = await asyncio.to_thread(
        _try_open, "render", args.output_device, is_input=False
    )
    print(f"                wdmks {render}", flush=True)
    capture = await asyncio.to_thread(
        _try_open, "capture", args.input_device, is_input=True
    )
    print(f"                wdmks {capture}", flush=True)


async def _main(args: argparse.Namespace) -> int:
    phone = Phone(args.adb, args.serial)
    print(f"phone  : {await phone.ensure_connected()}", flush=True)

    if args.number:
        print(f"dialing {args.number} ...", flush=True)
        await phone.dial(args.number, args.sim_slot)
    else:
        print("dial a number from the phone now", flush=True)

    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            state = _overall(await _precise_states(phone))
        except Exception:  # noqa: BLE001
            await phone.reconnect()
            state = "UNKNOWN"
        if state == "ACTIVE":
            break
        await asyncio.sleep(0.5)
    else:
        print("call never became active -- was it answered?", flush=True)
        return 1

    print("call active", flush=True)
    await _measure(phone, args, "baseline")

    await phone.wake()
    await asyncio.sleep(0.6)

    for attempt in range(1, args.attempts + 1):
        nodes = await _nodes(phone)
        node = _find_route_control(nodes)
        if node is None:
            print(f"\n[attempt {attempt}] no route control found; visible nodes:", flush=True)
            for item in nodes:
                label = (item["text"] or item["desc"]).strip()
                if label:
                    print(f"               '{label}' clickable={item['clickable']}", flush=True)
            break

        label = (node["text"] or node["desc"] or node["id"]).strip()
        print(f"\n[attempt {attempt}] tapping '{label}' at {node['bounds']}", flush=True)
        if not await _tap(phone, node):
            print("               tap skipped: unparsable bounds", flush=True)
            break

        await asyncio.sleep(1.5)
        await _measure(phone, args, f"after #{attempt}")

        route = await _active_route(phone)
        if "BT_SCO" in route:
            print("               route is BT_SCO -- holding and re-measuring", flush=True)
            await asyncio.sleep(3.0)
            await _measure(phone, args, "settled")
            break

    if args.keep:
        print(f"\nholding the call for {args.keep:.0f}s", flush=True)
        await asyncio.sleep(args.keep)
        await _measure(phone, args, "final")

    if args.hangup:
        print("hanging up", flush=True)
        await phone.hangup()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.call_bt_switch")
    parser.add_argument("--number", help="dial this number; omit to dial manually")
    parser.add_argument("--sim-slot", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=120.0, help="wait for answer")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--keep", type=float, default=0.0, help="hold the call this long at the end")
    parser.add_argument("--hangup", action="store_true")
    parser.add_argument("--match", default="iQOO")
    parser.add_argument("--input-device", default="iQOO Z9 5G")
    parser.add_argument("--output-device", default="iQOO Z9 5G")
    parser.add_argument("--adb", default=_default_adb())
    parser.add_argument("--serial")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
