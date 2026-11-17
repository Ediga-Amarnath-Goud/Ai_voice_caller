"""ADB control of the handset: dial, answer, hang up, poll call state.

This is the "control link" half of the bridge. It never touches audio; it only
drives the phone and reads its state. Everything here is non-blocking so it can
run next to Pipecat's event loop without starving the audio callbacks.

Requires a headset to be attached to the phone (wired TRRS splitter or a
Bluetooth HFP device). With a headset present, ``KEYCODE_HEADSETHOOK`` answers
and ends calls without touching screen coordinates.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

STATE_IDLE = 0
STATE_RINGING = 1
STATE_OFFHOOK = 2

STATE_NAMES = {STATE_IDLE: "idle", STATE_RINGING: "ringing", STATE_OFFHOOK: "offhook"}

_CALL_STATE_RE = re.compile(r"mCallState=(\d+)")

# Wireless debugging keeps a port for as long as the toggle stays on, and mDNS
# advertising is unreliable on some ROMs. Remembering the last working address is
# the only thing that survives a Doze-induced disconnect.
_ENDPOINT_FILE = Path(__file__).resolve().parent.parent / ".adb_endpoint"


class PhoneError(RuntimeError):
    pass


class Phone:
    def __init__(
        self,
        adb: str | Path = "platform-tools/adb.exe",
        serial: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._adb_path = str(adb)
        self._serial = serial
        self._timeout = timeout

    # ------------------------------------------------------------------ low level

    @property
    def _prefix(self) -> list[str]:
        if self._serial:
            return [self._adb_path, "-s", self._serial]
        # No -d: that flag only matches USB transports and fails over wireless ADB.
        return [self._adb_path]

    async def _resolve_serial(self) -> str | None:
        """Pick a transport when the caller did not name one.

        The same handset is routinely listed twice -- once by its mDNS name and
        once by its USB/Wi-Fi address -- and every plain `adb` command then fails
        with "more than one device". Candidates are health-checked, because a
        stale transport can sit in the list as "offline" for hours.
        """
        out = await self._run(self._adb_path, "devices")
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 2 or parts[1] != "device":
                continue
            candidate = parts[0]
            if (await self._run(self._adb_path, "-s", candidate, "get-state")).strip() == "device":
                self._serial = candidate
                self._remember(candidate)
                return candidate
        return None

    @staticmethod
    def _remember(address: str) -> None:
        try:
            _ENDPOINT_FILE.write_text(address, encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _recall() -> str | None:
        try:
            text = _ENDPOINT_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return text or None

    async def reconnect(self) -> bool:
        """Reattach after Doze or a daemon restart. Safe to call repeatedly."""
        self._serial = None
        remembered = self._recall()
        if remembered:
            await self._run(self._adb_path, "connect", remembered)
            if (await self._run(self._adb_path, "-s", remembered, "get-state")).strip() == "device":
                self._serial = remembered
                return True
        return await self._mdns_connect()

    async def _run(self, *cmd: str, timeout: float | None = None) -> str:
        """Run an adb command exactly as given, with no transport prefix."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout or self._timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise PhoneError(f"ADB timed out: {' '.join(cmd)}") from None
        text = out.decode(errors="replace").strip()
        if proc.returncode != 0 and not text:
            text = err.decode(errors="replace").strip()
        return text

    async def _mdns_connect(self) -> bool:
        """Re-discover the handset and reconnect over wireless ADB.

        Wireless debugging picks a new port every time it restarts, so a stale
        address is normal rather than an error. mDNS still advertises the current
        one, which makes this the only self-healing path when the cable is absent.
        """
        out = await self._run(self._adb_path, "mdns", "services")
        for line in out.splitlines():
            if "_adb-tls-connect._tcp" not in line:
                continue
            address = line.split()[-1]
            if ":" not in address:
                continue
            await self._run(self._adb_path, "connect", address)
            if (await self._run(self._adb_path, "-s", address, "get-state")).strip() == "device":
                self._serial = address
                self._remember(address)
                return True
        return False

    async def _adb(self, *args: str, timeout: float | None = None) -> str:
        if not self._serial:
            await self._resolve_serial()
        return await self._run(*self._prefix, *args, timeout=timeout)

    # ------------------------------------------------------------------ state

    async def ensure_connected(self) -> str:
        state = await self._adb("get-state", timeout=6)
        # Compare exactly: adb's error text ("no devices found") contains "device".
        if state.strip() != "device" and await self.reconnect():
            state = await self._adb("get-state", timeout=6)
        if state.strip() != "device":
            raise PhoneError(
                f"Phone not available over ADB (state={state!r}). "
                "Check that Wireless debugging is on and the phone is on the same network."
            )
        return await self._adb("shell", "getprop", "ro.product.model")

    async def call_state(self) -> int:
        """Return 0 idle, 1 ringing, 2 offhook (dialing or answered)."""
        out = await self._adb(
            "shell", "dumpsys telephony.registry | grep mCallState", timeout=8
        )
        states = [int(value) for value in _CALL_STATE_RE.findall(out)]
        if not states:
            # grep missing or output shape changed: parse the full dump.
            out = await self._adb("shell", "dumpsys telephony.registry", timeout=25)
            states = [int(value) for value in _CALL_STATE_RE.findall(out)]
        return max(states, default=STATE_IDLE)

    async def call_states(self) -> dict[int, int]:
        """Return {phone_id: mCallState} so you can see which SIM is on the call."""
        out = await self._adb(
            "shell",
            "dumpsys telephony.registry | grep -E 'Phone Id=|mCallState='",
            timeout=8,
        )
        states: dict[int, int] = {}
        current: int | None = None
        for line in out.splitlines():
            line = line.strip()
            header = re.match(r"Phone Id=(\d+)", line)
            if header:
                current = int(header.group(1))
                continue
            value = _CALL_STATE_RE.match(line)
            if value and current is not None:
                states[current] = int(value.group(1))
        return states

    async def wait_for_state(
        self,
        target: int | set[int],
        timeout: float = 60.0,
        interval: float = 1.0,
    ) -> int:
        targets = {target} if isinstance(target, int) else set(target)
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            state = await self.call_state()
            if state in targets:
                return state
            if asyncio.get_running_loop().time() >= deadline:
                raise PhoneError(
                    f"Timed out waiting for {targets}; last state="
                    f"{STATE_NAMES.get(state, state)}"
                )
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ actions

    async def dial(self, number: str, sim_slot: int | None = None) -> None:
        args = [
            "shell",
            "am",
            "start",
            "-a",
            "android.intent.action.CALL",
            "-d",
            f"tel:{number}",
        ]
        if sim_slot is not None:
            args += ["--es", "com.android.phone.extra.slot", str(sim_slot)]
        out = await self._adb(*args)
        if "Error" in out or "Exception" in out:
            raise PhoneError(f"Dial failed: {out}")

    async def answer(self) -> None:
        await self._adb("shell", "input", "keyevent", "KEYCODE_HEADSETHOOK")
        await asyncio.sleep(1.2)
        if await self.call_state() == STATE_RINGING:
            # Some OEM dialers ignore HEADSETHOOK on incoming calls.
            await self._adb("shell", "input", "keyevent", "KEYCODE_CALL")

    async def hangup(self) -> None:
        await self._adb("shell", "input", "keyevent", "KEYCODE_ENDCALL")

    async def wake(self) -> None:
        """Optional: wake the screen without unlocking (does not end the call)."""
        await self._adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")

    async def last_call_duration(self) -> int | None:
        """Duration in seconds of the most recent call (0 means never answered)."""
        out = await self._adb(
            "shell",
            "content query --uri content://call_log/calls --projection duration "
            "--sort 'date DESC'",
            timeout=15,
        )
        match = re.search(r"duration=(\d+)", out)
        return int(match.group(1)) if match else None


def normalize_number(number: str, country_code: str = "91") -> str:
    """Turn a local number into E.164 for the dialer."""
    number = number.replace(" ", "").replace("-", "")
    if number.startswith("+"):
        return number
    if number.startswith("00"):
        return "+" + number[2:]
    if number.startswith("0"):
        number = number[1:]
    return f"+{country_code}{number}"
