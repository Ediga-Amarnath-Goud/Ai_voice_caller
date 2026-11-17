"""PyAudio device discovery for Pipecat's local transport.

Pipecat's ``LocalAudioTransport`` opens devices through PyAudio/PortAudio.
``sounddevice`` is also a PortAudio wrapper, so the indices line up today — but
they are only *positions in a list that changes* whenever any audio device
connects or disconnects. This machine enumerates 60+ endpoints, and the phone's
hands-free pair sits at 42/43 only while the current set of Bluetooth devices is
connected. Plug in a USB sound card and those numbers move.

Resolve devices by name here instead of hardcoding an index.

Usage::

    python -m agent.audio_devices
"""

from __future__ import annotations

from typing import Any

import pyaudio

_PREFERRED_HOSTS = ("MME",)
_DEPRIORITIZED_NAMES = ("sound mapper", "primary sound")


def _clean(name: str) -> str:
    return " ".join(name.split())


def list_devices() -> list[dict[str, Any]]:
    """Return every PyAudio device with its index in PyAudio's namespace."""
    pa = pyaudio.PyAudio()
    try:
        devices: list[dict[str, Any]] = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            host = pa.get_host_api_info_by_index(int(info["hostApi"]))
            devices.append(
                {
                    "index": index,
                    "name": _clean(str(info["name"])),
                    "host_api": str(host["name"]),
                    "in_channels": int(info["maxInputChannels"]),
                    "out_channels": int(info["maxOutputChannels"]),
                    "rate": int(float(info["defaultSampleRate"])),
                }
            )
        return devices
    finally:
        pa.terminate()


def format_table(devices: list[dict[str, Any]]) -> str:
    rows = [
        f"{'idx':>4}  {'in':>2} {'out':>3} {'rate':>6}  {'host api':<18} name",
    ]
    for d in devices:
        rows.append(
            f"{d['index']:>4}  {d['in_channels']:>2} {d['out_channels']:>3} "
            f"{d['rate']:>6}  {d['host_api']:<18} {d['name']}"
        )
    return "\n".join(rows)


def find_device(
    query: str,
    *,
    need_input: bool = False,
    need_output: bool = False,
    host_api: str | None = None,
) -> int:
    """Resolve a device name (or index) to a PyAudio index.

    ``query`` may be a full name, a case-insensitive substring, or a numeric
    index. When several devices match, MME is preferred because PortAudio's MME
    backend resamples arbitrary sample rates, which WDM-KS and WASAPI shared
    mode generally refuse.
    """
    devices = list_devices()

    if query.strip().isdigit():
        index = int(query)
        match = next((d for d in devices if d["index"] == index), None)
        if match is None:
            raise RuntimeError(f"No PyAudio device with index {index}.")
        _check_channels(match, need_input=need_input, need_output=need_output)
        return index

    needle = query.strip().lower()
    candidates = [
        d
        for d in devices
        if needle in d["name"].lower()
        and (not need_input or d["in_channels"] > 0)
        and (not need_output or d["out_channels"] > 0)
        and (not host_api or host_api.lower() in d["host_api"].lower())
    ]

    if not candidates:
        raise RuntimeError(
            f"No PyAudio device matching {query!r} "
            f"(input={need_input}, output={need_output}, host_api={host_api}).\n\n"
            + format_table(devices)
        )

    def score(device: dict[str, Any]) -> tuple[int, int]:
        name = device["name"].lower()
        host_bonus = 0
        for rank, h in enumerate(_PREFERRED_HOSTS):
            if h.upper() in device["host_api"].upper():
                host_bonus = 100 - (rank * 20)
                break
        name_bonus = -1 if any(junk in name for junk in _DEPRIORITIZED_NAMES) else 0
        return (host_bonus + name_bonus, -device["index"])

    best = max(candidates, key=score)
    _check_channels(best, need_input=need_input, need_output=need_output)
    return int(best["index"])


def device_info(index: int) -> dict[str, Any]:
    """Return the PyAudio device dict for an index."""
    for device in list_devices():
        if device["index"] == index:
            return device
    raise RuntimeError(f"No PyAudio device with index {index}.")


def _check_channels(device: dict[str, Any], *, need_input: bool, need_output: bool) -> None:
    if need_input and device["in_channels"] < 1:
        raise RuntimeError(f"{device['name']!r} has no input channels.")
    if need_output and device["out_channels"] < 1:
        raise RuntimeError(f"{device['name']!r} has no output channels.")


if __name__ == "__main__":
    print(format_table(list_devices()))
