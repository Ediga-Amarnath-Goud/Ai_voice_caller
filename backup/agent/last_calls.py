"""Print recent phone calls with times that can be lined up against a probe run.

A probe log only shows levels; to know whether those levels were a real call we
need the phone's own record of when calls started and ended. Each row is printed
with its offset from ``--since`` (an epoch second), which is exactly what a probe
prints at start-up, so the two logs can be compared by eye.

    python -m agent.last_calls --count 5
    python -m agent.last_calls --since 1780000000

If the call log is not readable by the shell user this exits non-zero instead of
guessing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from .watch_endpoint import _adb_path, _serial

KINDS = {"1": "incoming", "2": "outgoing", "3": "missed", "4": "voicemail",
         "5": "rejected", "6": "blocked"}

QUERY = (
    "content query --uri content://call_log/calls "
    "--projection _id:date:duration:number:type "
    '--sort "date DESC" | head -10'
)


def _rows(count: int, raw: bool = False) -> list[dict[str, str]]:
    serial = _serial()
    result = subprocess.run(
        [_adb_path(), "-s", serial, "shell", QUERY],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0:
        print(f"adb failed: {result.stderr.strip()}", file=sys.stderr)
        return []

    if raw:
        print(result.stdout or "(empty stdout)")
        print("stderr:", result.stderr.strip() or "(none)")
        return []

    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("Row:"):
            continue
        # "Row: 6948 _id=21091, date=..." -> drop "Row:" and its row number,
        # otherwise the first pair's key becomes "Row: 6948 _id" and `date` is
        # lost when it happens to be the first column.
        body = line.split(" ", 2)[-1]
        fields: dict[str, str] = {}
        for pair in body.split(","):
            if "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            fields[key.strip()] = value.strip()
        rows.append(fields)
        if len(rows) >= count:
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.last_calls")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--raw", action="store_true", help="print the raw adb query")
    parser.add_argument("--since", type=int, default=0,
                        help="reference epoch second; rows get a relative offset")
    args = parser.parse_args(argv)

    rows = _rows(args.count, raw=args.raw)
    if args.raw:
        return 0
    if not rows:
        print("no rows (call log may be unreadable by the shell user)", file=sys.stderr)
        return 1

    print(f"{'#':>2}  {'kind':<9} {'number':<14} {'started':<20} "
          f"{'talk':>7}  {'offset from --since':>20}")
    for index, fields in enumerate(rows, start=1):
        try:
            epoch = int(fields.get("date", "0")) // 1000
        except ValueError:
            epoch = 0
        try:
            duration = int(fields.get("duration", "0"))
        except ValueError:
            duration = 0
        kind = KINDS.get(fields.get("type", ""), fields.get("type", "?") or "?")
        started = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S") if epoch else "?"
        offset = f"{epoch - args.since:+8d}s" if args.since and epoch else ""
        print(f"{index:>2}  {kind:<9} {fields.get('number', '?'):<14} {started:<20} "
              f"{duration:>6}s  {offset:>20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
