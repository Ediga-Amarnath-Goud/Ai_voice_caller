"""Watch a call and record what the audio stack actually does.

Direction-agnostic by design: incoming and outgoing calls go through the same
state machine, because the underlying telephony states differ but the question
does not -- when the call becomes active, does Android route it to the Bluetooth
hands-free device, and can Windows open that endpoint?

    python -m agent.watch_call --seconds 480 --probe-audio

States come from ``mPreciseCallState`` rather than ``mCallState``. ``mCallState``
reports 2 for both "dialing" and "connected", which is useless for an outgoing
call; the precise state separates DIALING, ALERTING and ACTIVE. That distinction
is the whole point of this tool.

Passive by design: it never dials, never answers, and never holds a stream open
for more than a moment.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path

from .audio_devices import device_info, find_device
from .phone import Phone

# android.telephony.PreciseCallState
PRECISE_NAMES = {
    0: "IDLE",
    1: "ACTIVE",
    2: "HOLDING",
    3: "DIALING",
    4: "ALERTING",
    5: "INCOMING",
    6: "WAITING",
    7: "DISCONNECTED",
    8: "DISCONNECTING",
}

# AOSP audio device type bitmasks that matter for a call.
_DEVICE_NAMES = {
    0x1: "EARPIECE",
    0x2: "SPEAKER",
    0x4: "WIRED_HEADSET",
    0x8: "WIRED_HEADPHONE",
    0x10: "BT_SCO",
    0x20: "BT_SCO_HEADSET",
    0x40: "BT_SCO_CARKIT",
    0x80: "BT_A2DP",
    0x2000: "USB_ACCESSORY",
    0x4000: "USB_DEVICE",
    0x4000000: "USB_HEADSET",
    0x10000: "TELEPHONY_TX",
}

# One label for the whole handset, derived from the per-SIM precise states.
# Order is priority: the most "engaged" call wins.
_PRIORITY = ["ACTIVE", "HOLDING", "INCOMING", "WAITING", "ALERTING", "DIALING", "ENDED"]

_PHONE_ID_RE = re.compile(r"Phone Id=(\d+)")
_PRECISE_RE = re.compile(
    r"mPreciseCallState=Ringing call state: (\d+), Foreground call state: (\d+)"
)


def _default_adb() -> str:
    candidate = Path(__file__).resolve().parent.parent / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


def _describe(text: str) -> str:
    """Decode raw bitmasks into device names."""
    for code, name in _DEVICE_NAMES.items():
        text = text.replace(f"0x{code:x} (", f"0x{code:x} {name} (")
    return text


def _sim_label(ringing: int, foreground: int) -> str:
    if ringing in (5, 6):
        return "INCOMING"
    if foreground in (3,):
        return "DIALING"
    if foreground == 4:
        return "ALERTING"
    if foreground == 1:
        return "ACTIVE"
    if foreground == 2:
        return "HOLDING"
    if foreground in (7, 8):
        return "ENDED"
    return "IDLE"


async def _precise_states(phone: Phone) -> dict[int, tuple[int, int, str]]:
    """Return {phone_id: (ringing, foreground, label)} for every SIM."""
    out = await phone._adb(
        "shell",
        "dumpsys telephony.registry | grep -E 'Phone Id=|mPreciseCallState='",
        timeout=8,
    )
    states: dict[int, tuple[int, int, str]] = {}
    current: int | None = None
    for line in out.splitlines():
        header = _PHONE_ID_RE.search(line)
        if header:
            current = int(header.group(1))
            continue
        match = _PRECISE_RE.search(line)
        if match and current is not None:
            ringing, foreground = int(match.group(1)), int(match.group(2))
            states[current] = (ringing, foreground, _sim_label(ringing, foreground))
    return states


def _overall(states: dict[int, tuple[int, int, str]]) -> str:
    labels = {label for _, _, label in states.values()}
    for candidate in _PRIORITY:
        if candidate in labels:
            return candidate
    return "IDLE"


async def _patch_events(phone: Phone) -> list[str]:
    out = await phone._adb(
        "shell", "dumpsys media.audio_flinger | grep CFG_EVENT", timeout=20
    )
    return [_describe(line.strip()) for line in out.splitlines() if "CFG_EVENT" in line]


async def _active_route(phone: Phone) -> str:
    """Routing of the MIXER thread that carries call audio, not every thread."""
    out = await phone._adb(
        "shell",
        "dumpsys media.audio_flinger | grep -A 20 'name AudioOut_D' "
        "| grep -E 'devices:|Input device:'",
        timeout=20,
    )
    parts = [
        _describe(line.strip())
        for line in out.splitlines()
        if line.strip().startswith(("Output devices:", "Input device:"))
    ]
    return " | ".join(parts) if parts else "no AudioOut_D thread"


def _try_open(label: str, device: str, *, is_input: bool) -> str:
    import pyaudio

    try:
        index = find_device(device, need_input=is_input, need_output=not is_input)
    except RuntimeError as exc:
        return f"{label}: resolve failed ({exc})"

    rate = int(device_info(index)["rate"])
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pa.get_format_from_width(2),
            channels=1,
            rate=rate,
            input=is_input,
            output=not is_input,
            input_device_index=index if is_input else None,
            output_device_index=None if is_input else index,
            frames_per_buffer=320,
        )
        stream.close()
        return f"{label}: OPENED ok (idx {index}, {rate} Hz)"
    except OSError as exc:
        return f"{label}: FAILED (idx {index}, {rate} Hz) -> {exc}"
    finally:
        pa.terminate()


def _explain(previous: str, current: str) -> str:
    """Say what a transition means in plain terms."""
    if previous == "IDLE" and current == "DIALING":
        return "outgoing call started"
    if previous == "IDLE" and current == "INCOMING":
        return "incoming call ringing"
    if current == "ALERTING":
        return "far end is ringing"
    if current == "ACTIVE" and previous in {"DIALING", "ALERTING", "INCOMING"}:
        return "ANSWERED -- audio path is now live"
    if current in {"IDLE", "ENDED"} and previous in {"ACTIVE", "DIALING", "ALERTING", "INCOMING"}:
        return "call ended"
    if current == "IDLE" and previous == "ENDED":
        return "call released, handset idle"
    if current == "HOLDING":
        return "call on hold"
    if current == "WAITING":
        return "second call waiting"
    return f"{previous} -> {current}"


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


async def _main(args: argparse.Namespace) -> int:
    phone = Phone(args.adb, args.serial)
    model = await phone.ensure_connected()
    print(f"phone  : {model}  (transport {phone._serial})", flush=True)

    baseline = await _patch_events(phone)
    seen: set[str] = set(baseline)
    states = await _precise_states(phone)
    previous = _overall(states)
    previous_route = await _active_route(phone)
    print(f"state  : {previous}  {states}", flush=True)
    print(f"route  : {previous_route}", flush=True)
    print(f"watching for {args.seconds:.0f}s -- place or take the call now\n", flush=True)

    active_since: float | None = None
    probe_schedule = [0.5, 3.0, 6.0, 10.0]
    pending: list[float] = []
    ended_at: float | None = None
    # One record per call, so ringing / answered / ended read as a single event
    # instead of three unrelated lines the operator has to stitch together.
    call: dict[str, float | str | None] | None = None
    started = time.monotonic()
    deadline = started + args.seconds

    while time.monotonic() < deadline:
        try:
            states = await _precise_states(phone)
            if not states:
                raise RuntimeError("empty telephony dump")
        except Exception as exc:  # noqa: BLE001 - observer must survive ADB blips
            print(f"[{time.monotonic() - started:7.1f}s] poll failed: {exc}", flush=True)
            if await phone.reconnect():
                print("           reconnected to the phone", flush=True)
            previous = "UNKNOWN"
            await asyncio.sleep(1.0)
            continue

        current = _overall(states)
        stamp = time.monotonic() - started

        if current != previous:
            per_sim = ", ".join(
                f"SIM{pid}={label}" for pid, (_, _, label) in states.items()
            )
            print(f"\n[{stamp:7.1f}s] {current:<8} {_explain(previous, current)}", flush=True)
            print(f"           {per_sim}", flush=True)

            events = await _patch_events(phone)
            fresh = [event for event in events if event not in seen]
            seen.update(events)
            for event in fresh[-8:]:
                print(f"           {event}", flush=True)
            if not fresh:
                print("           (no audio patch change)", flush=True)

            now = time.monotonic()

            if current in {"DIALING", "INCOMING", "ALERTING", "ACTIVE"} and call is None:
                # Also covers a watcher that starts mid-call: the first state it
                # sees opens the record, so a call is never left unaccounted for.
                call = {
                    "direction": {
                        "DIALING": "outgoing",
                        "ALERTING": "outgoing",
                        "INCOMING": "incoming",
                    }.get(current, "unknown"),
                    "start": now,
                    "ringing": now if current in {"INCOMING", "ALERTING"} else None,
                    "answer": now if current == "ACTIVE" else None,
                }
            elif call is not None:
                if current in {"DIALING", "ALERTING"} and call["ringing"] is None:
                    call["ringing"] = now
                if current == "ACTIVE" and call["answer"] is None:
                    call["answer"] = now

            if current == "ACTIVE":
                active_since = time.monotonic()
                pending = list(probe_schedule)
                print("           probing audio at +0.5/+3/+6/+10 s", flush=True)
            elif current in {"IDLE", "ENDED"}:
                if active_since is not None:
                    print(
                        f"           active for {time.monotonic() - active_since:.1f}s",
                        flush=True,
                    )
                active_since = None
                pending = []
                ended_at = time.monotonic()

                if call is not None:
                    answered_at = call["answer"]
                    total = now - float(call["start"])
                    if isinstance(answered_at, float):
                        talk = now - answered_at
                        detail = (
                            f"answered after {_clock(answered_at - float(call['start']))}, "
                            f"talked {_clock(talk)}"
                        )
                    else:
                        detail = "never answered"
                    print(
                        f"           summary: {call['direction']} call, {detail}, "
                        f"total {_clock(total)}",
                        flush=True,
                    )
                    call = None
            previous = current

        route = await _active_route(phone)
        if route != previous_route:
            print(f"[{stamp:7.1f}s] route  -> {route}", flush=True)
            previous_route = route

        if active_since is not None and pending:
            delta = time.monotonic() - active_since
            if delta >= pending[0]:
                pending.pop(0)
                print(f"[{stamp:7.1f}s] probe  (+{delta:.1f}s)", flush=True)
                print(f"           route: {route}", flush=True)
                if args.probe_audio:
                    for label, device, is_input in (
                        ("output", args.output_device, False),
                        ("input", args.input_device, True),
                    ):
                        result = await asyncio.to_thread(
                            _try_open, label, device, is_input=is_input
                        )
                        print(f"           {result}", flush=True)

        if ended_at is not None and time.monotonic() - ended_at > 2.0:
            ended_at = None
            try:
                duration = await phone.last_call_duration()
                verdict = "far end never answered" if duration == 0 else f"{duration}s"
                print(f"           call log: duration={duration} ({verdict})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"           call log unavailable: {exc}", flush=True)

        await asyncio.sleep(args.interval)

    print("\nfinal route:", await _active_route(phone), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.watch_call")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=0.6)
    parser.add_argument("--input-device", default="iQOO Z9 5G")
    parser.add_argument("--output-device", default="iQOO Z9 5G")
    parser.add_argument("--probe-audio", action="store_true")
    parser.add_argument("--adb", default=_default_adb())
    parser.add_argument("--serial")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
