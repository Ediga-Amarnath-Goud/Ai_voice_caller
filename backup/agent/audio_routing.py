"""Scoped, reversible mic routing for the Phone Link software bridge.

Phone Link always takes its microphone from the Windows *console* default
capture device. During an agent call that device must be CABLE Output, so the
AI's voice reaches the caller. Outside calls the machine must go back to the
user's real microphone. This module saves the original defaults once, switches
only the capture roles for the call, and restores them afterwards.

    python -m agent.audio_routing status
    python -m agent.audio_routing enable
    python -m agent.audio_routing restore

``run_call.py --bridge`` calls enable/restore automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

SVCL = Path(__file__).resolve().parent.parent / "tools" / "svcl" / "svcl.exe"
STATE = Path(__file__).resolve().parent.parent / ".audio_routing.json"

CABLE_CAPTURE = r"VB-Audio Virtual Cable\Device\CABLE Output\Capture"
REALTEK_RENDER = r"Realtek(R) Audio\Device\Speakers\Render"

#: Muted during agent calls so the caller is not heard on the laptop speakers.
#: The WASAPI loopback still captures pre-mute audio, so Gemini still hears them.
MUTE_SPEAKERS_ON_CALL = False

PHONE_LINK_APPS = ("shell:AppsFolder\\Microsoft.YourPhone_8wekyb3d8bbwe!App",)
PHONE_LINK_PROCESSES = ("PhoneExperienceHost.exe", "YourPhoneAppProxyHost.exe")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    if not SVCL.exists():
        raise RuntimeError(f"SoundVolumeCommandLine not found at {SVCL}")
    return subprocess.run(
        [str(SVCL), *args], capture_output=True, text=True, timeout=30
    )


def _rows() -> list[dict[str, str]]:
    out = Path(os.environ.get("TEMP", ".")) / "svcl_audio_routing.csv"
    result = _run("/scomma", str(out))
    if result.returncode != 0:
        raise RuntimeError(f"svcl enumeration failed: {result.stderr.strip()}")
    with out.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.DictReader(handle) if row.get("Type") == "Device"]


def _default_id(rows: list[dict[str, str]], direction: str, column: str) -> str | None:
    for row in rows:
        if row.get("Direction") == direction and row.get(column) == direction:
            return row.get("Command-Line Friendly ID")
    return None


def status() -> dict[str, str | None]:
    rows = _rows()
    state = {
        "console_capture": _default_id(rows, "Capture", "Default"),
        "multimedia_capture": _default_id(rows, "Capture", "Default Multimedia"),
        "communications_capture": _default_id(rows, "Capture", "Default Communications"),
        "console_render": _default_id(rows, "Render", "Default"),
    }
    for key, value in state.items():
        print(f"{key:<24} {value or '<none>'}")
    return state


def enable() -> None:
    """Save the current capture defaults (once), then point Phone Link's mic at the cable."""
    current = _rows()
    if not STATE.exists():
        saved = {
            "console_capture": _default_id(current, "Capture", "Default"),
            "communications_capture": _default_id(current, "Capture", "Default Communications"),
            "console_render": _default_id(current, "Render", "Default"),
        }
        STATE.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        print(f"saved original audio defaults to {STATE}")

    for role in (0, 2):  # console and communications
        result = _run("/SetDefault", CABLE_CAPTURE, str(role))
        if result.returncode != 0:
            raise RuntimeError(f"failed to set capture role {role}: {result.stderr.strip()}")
    if MUTE_SPEAKERS_ON_CALL:
        _run("/Mute", REALTEK_RENDER, "Mute")
        print("Laptop speakers muted (headless mode — caller audio is not audible locally)")
    print("Phone Link mic switched to CABLE Output (console + communications)")


def restore() -> None:
    """Put the user's original defaults back and restart Phone Link so it re-reads them."""
    if not STATE.exists():
        print("no saved audio defaults; nothing to restore")
        return
    saved = json.loads(STATE.read_text(encoding="utf-8"))
    for key, role in (("console_capture", 0), ("communications_capture", 2)):
        device = saved.get(key)
        if device:
            result = _run("/SetDefault", device, str(role))
            if result.returncode != 0:
                print(f"warning: could not restore {key}: {result.stderr.strip()}")
    render = saved.get("console_render")
    if render:
        _run("/SetDefault", render, "all")
    if MUTE_SPEAKERS_ON_CALL:
        _run("/Mute", REALTEK_RENDER, "Unmute")
        print("Laptop speakers unmuted (normal audio restored)")
    STATE.unlink(missing_ok=True)
    print("original audio defaults restored")
    # DO NOT call restart_phone_link() here: killing PhoneExperienceHost drops
    # the Bluetooth HFP audio connection to the physical phone.


def restart_phone_link() -> None:
    for name in PHONE_LINK_PROCESSES:
        subprocess.run(
            ["taskkill", "/F", "/IM", name], capture_output=True, text=True, timeout=15
        )
    time.sleep(1)
    for app in PHONE_LINK_APPS:
        try:
            os.startfile(app)  # type: ignore[attr-defined] - Windows only
        except OSError as exc:
            print(f"warning: could not relaunch Phone Link: {exc}")
    time.sleep(3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.audio_routing")
    parser.add_argument("action", choices=("status", "enable", "restore", "restart"))
    args = parser.parse_args(argv)

    if args.action == "status":
        status()
    elif args.action == "enable":
        enable()
        status()
    elif args.action == "restart":
        restart_phone_link()
        status()
    else:
        restore()
        status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
