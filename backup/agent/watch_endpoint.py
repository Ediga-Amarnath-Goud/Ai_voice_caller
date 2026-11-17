"""Watch the Windows audio stack for the hands-free endpoint while a call is live.

The endpoint question has been asked five different ways and given two
contradictory answers:

  * ``IMMDeviceEnumerator`` (via soundcard) never lists the phone, ever
  * but a started ``SWD\\MMDEVAPI\\{0.0.0...}.{guid}`` AudioEndpoint devnode exists

Those cannot both be true of a healthy endpoint, so one of them is describing a
stale object. The missing variable is *which* call is up: the operator reports
that a call placed from Phone Link stays on the PC, while a call placed from the
phone falls back. So the endpoint may only exist while Phone Link owns the call.

This watches both views, side by side, and reports the moment they agree.

    python -m agent.watch_endpoint --seconds 600
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

_MMDEV = r"HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio"
_NAME_PROP = "{b3f8fa53-0004-438e-9003-51a46e139bfc},6"
_STATE = {1: "ACTIVE", 2: "DISABLED", 4: "NOTPRESENT", 8: "UNPLUGGED"}
_DEVNODE_RE = re.compile(r"SWD\\MMDEVAPI\\\{0\.0\.(\d)\.00000000\}\.(\{[0-9a-fA-F-]+\})")


def _adb_path() -> str:
    candidate = Path(__file__).resolve().parent.parent / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


def _serial() -> str | None:
    """Return a transport that actually answers.

    The saved address goes stale whenever wireless debugging restarts under a
    new port, and `adb devices` then lists the same handset under its mDNS name
    while `-s <old address>` fails with "device not found". Health-check every
    candidate instead of trusting the file.
    """
    endpoint = Path(__file__).resolve().parent.parent / ".adb_endpoint"
    try:
        saved = endpoint.read_text(encoding="utf-8").strip()
    except OSError:
        saved = ""

    try:
        listing = subprocess.run(
            [_adb_path(), "devices"], capture_output=True, text=True, timeout=15
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return saved or None

    candidates = [saved] if saved else []
    for line in listing.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device" and parts[0] not in candidates:
            candidates.append(parts[0])

    for candidate in candidates:
        try:
            state = subprocess.run(
                [_adb_path(), "-s", candidate, "get-state"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            continue
        if state == "device":
            if candidate != saved:
                try:
                    endpoint.write_text(candidate, encoding="utf-8")
                except OSError:
                    pass
            return candidate
    return None


def _route() -> str:
    serial = _serial()
    if not serial:
        return "no adb serial"
    try:
        out = subprocess.run(
            [
                _adb_path(),
                "-s",
                serial,
                "shell",
                "dumpsys media.audio_flinger | grep -A 20 'name AudioOut_D' "
                "| grep -E 'devices:|Input device:'",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"route read failed ({exc})"
    return " ".join(out.split())


def _registry() -> list[str]:
    """Registry view: what the audio service has registered, with its state."""
    script = (
        f"Get-ChildItem '{_MMDEV}\\Render','{_MMDEV}\\Capture' -ErrorAction SilentlyContinue "
        "| ForEach-Object { $k=$_; "
        f"$n=(Get-ItemProperty \"$($k.PSPath)\\Properties\" -ErrorAction SilentlyContinue).'{_NAME_PROP}'; "
        "if ($n -match 'iQOO') { "
        "$s=(Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue).DeviceState; "
        "$f=if ($k.PSPath -match '\\\\Render\\\\') {'render'} else {'capture'}; "
        "\"$f|$s|$($k.PSChildName)|$n\" } }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=25,
        ).stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [f"registry read failed ({exc})"]
    return [line.strip() for line in out.splitlines() if line.strip()]


def _devnodes() -> dict[str, str]:
    """PnP view: started MMDEVAPI devnodes, keyed by endpoint guid."""
    try:
        out = subprocess.run(
            ["pnputil", "/enum-devices", "/connected"],
            capture_output=True,
            text=True,
            timeout=25,
        ).stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"?": f"devnode read failed ({exc})"}

    found: dict[str, str] = {}
    for block in out.split("\n\n"):
        match = _DEVNODE_RE.search(block)
        if not match:
            continue
        desc = ""
        for line in block.splitlines():
            if "Device Description:" in line:
                desc = line.split(":", 1)[1].strip()
        found[match.group(2).lower()] = desc
    return found


def _core(needle: str) -> tuple[list[str], list[str]]:
    import soundcard as sc

    render = [s.name for s in sc.all_speakers() if needle.lower() in s.name.lower()]
    capture = [
        m.name
        for m in sc.all_microphones(include_loopback=True)
        if needle.lower() in m.name.lower() and "loopback" not in m.name.lower()
    ]
    return render, capture


def _try_open(needle: str, rendered: list[str], captured: list[str]) -> list[str]:
    """Open whatever just appeared, in shared mode, and say what came out."""
    import soundcard as sc

    notes: list[str] = []
    for name in rendered:
        speaker = next((s for s in sc.all_speakers() if s.name == name), None)
        if speaker is None:
            continue
        try:
            with speaker.player(samplerate=16000, channels=1, blocksize=320) as player:
                player.play(b"\x00\x00" * 1600)
            notes.append(f"render OPEN {name}")
        except Exception as exc:  # noqa: BLE001 - report the raw failure
            notes.append(f"render FAIL {name}: {type(exc).__name__}: {exc}")
    for name in captured:
        mic = next(
            (
                m
                for m in sc.all_microphones(include_loopback=True)
                if m.name == name and "loopback" not in m.name.lower()
            ),
            None,
        )
        if mic is None:
            continue
        try:
            with mic.recorder(samplerate=16000, channels=1, blocksize=320) as recorder:
                recorder.record(numframes=1600)
            notes.append(f"capture OPEN {name}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"capture FAIL {name}: {type(exc).__name__}: {exc}")
    return notes


def _report(index: int, needle: str, try_open: bool) -> bool:
    reg = _registry()
    nodes = _devnodes()
    render, capture = _core(needle)

    node_hits = [guid for guid, desc in nodes.items() if needle.lower() in desc.lower()]
    print(f"[{index:02d}] route: {_route()}", flush=True)
    print(
        f"     core: render={render or '-'} capture={capture or '-'}",
        flush=True,
    )
    for line in reg:
        flow, state, guid, name = line.split("|", 3)
        started = "devnode-started" if guid.lower() in nodes else "no devnode"
        try:
            state_text = _STATE.get(int(state), state)
        except ValueError:
            state_text = state
        print(f"     registry: {flow:<7} {state_text:<10} {started:<15} {name}", flush=True)
    if node_hits:
        for guid in node_hits:
            print(f"     devnode : STARTED {guid}  {nodes[guid]}", flush=True)
    if not reg:
        print("     registry: no iQOO entries at all", flush=True)

    appeared = bool(render or capture)
    if appeared and try_open:
        for note in _try_open(needle, render, capture):
            print(f"     open    {note}", flush=True)
    return appeared


def _route_only(index: int, needle: str) -> None:
    """Fast sampling: only what the phone is doing, plus a cheap endpoint check."""
    render, capture = _core(needle)
    print(
        f"[{index:02d}] {_route()}   core: render={render or '-'} capture={capture or '-'}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.watch_endpoint")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--match", default="iQOO")
    parser.add_argument("--no-open", action="store_true", help="only report, do not open")
    parser.add_argument(
        "--route-only",
        action="store_true",
        help="sample the phone's route often; skip the registry/devnode reads",
    )
    args = parser.parse_args(argv)

    print(f"watching for {args.seconds:.0f}s", flush=True)
    deadline = time.monotonic() + args.seconds
    index = 0
    ever = False
    while time.monotonic() < deadline:
        index += 1
        try:
            if args.route_only:
                _route_only(index, args.match)
                appeared = False
            else:
                appeared = _report(index, args.match, not args.no_open)
        except Exception as exc:  # noqa: BLE001 - never die mid-call
            print(f"[{index:02d}] sweep failed: {type(exc).__name__}: {exc}", flush=True)
            appeared = False
        if appeared:
            ever = True
        if not args.route_only:
            print(flush=True)
        time.sleep(args.interval)

    print(f"endpoint appeared during the run: {ever}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
