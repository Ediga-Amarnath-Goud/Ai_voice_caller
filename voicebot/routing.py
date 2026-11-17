"""Scoped mic routing: point Phone Link's microphone at the virtual cable.

Phone Link records from the Windows console default capture device. During a
call that must be CABLE Output so the AI's voice reaches the caller. We save the
original defaults once, switch only the capture roles, and restore on exit.

The speakers are NEVER muted: the WASAPI loopback taps pre-mute audio anyway,
and muting was proven irrelevant to capture in the audit.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

from . import config


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    if not config.SVCL.exists():
        raise RuntimeError(f"SoundVolumeCommandLine not found at {config.SVCL}")
    return subprocess.run(
        [str(config.SVCL), *args], capture_output=True, text=True, timeout=30
    )


def _rows() -> list[dict[str, str]]:
    out = Path(os.environ.get("TEMP", ".")) / "voicebot_svcl.csv"
    result = _run("/scomma", str(out))
    if result.returncode != 0:
        raise RuntimeError(f"svcl enumeration failed: {result.stderr.strip()}")
    with out.open(newline="", encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r.get("Type") == "Device"]


def _default_id(rows: list[dict[str, str]], direction: str, column: str) -> str | None:
    for row in rows:
        if row.get("Direction") == direction and row.get(column) == direction:
            return row.get("Command-Line Friendly ID")
    return None


def status(quiet: bool = False) -> dict[str, str | None]:
    rows = _rows()
    state = {
        "console_capture": _default_id(rows, "Capture", "Default"),
        "communications_capture": _default_id(rows, "Capture", "Default Communications"),
        "console_render": _default_id(rows, "Render", "Default"),
    }
    if not quiet:
        for key, value in state.items():
            print(f"{key:<24} {value or '<none>'}")
    return state


def enable() -> None:
    """Save originals once, then point console + communications mic at CABLE Output."""
    current = _rows()
    if not config.ROUTING_STATE.exists():
        saved = {
            "console_capture": _default_id(current, "Capture", "Default"),
            "communications_capture": _default_id(current, "Capture", "Default Communications"),
            "console_render": _default_id(current, "Render", "Default"),
        }
        config.ROUTING_STATE.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        print(f"saved original audio defaults to {config.ROUTING_STATE}")
    for role in (0, 2):  # console, communications
        result = _run("/SetDefault", config.CABLE_CAPTURE_ID, str(role))
        if result.returncode != 0:
            raise RuntimeError(f"failed to set capture role {role}: {result.stderr.strip()}")
    print("Phone Link mic -> CABLE Output (console + communications)")


def restore() -> None:
    if not config.ROUTING_STATE.exists():
        return
    saved = json.loads(config.ROUTING_STATE.read_text(encoding="utf-8"))
    for key, role in (("console_capture", 0), ("communications_capture", 2)):
        device = saved.get(key)
        if device:
            _run("/SetDefault", device, str(role))
    render = saved.get("console_render")
    if render:
        _run("/SetDefault", render, "all")
    config.ROUTING_STATE.unlink(missing_ok=True)
    print("original audio defaults restored")
