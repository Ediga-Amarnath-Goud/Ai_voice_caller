"""Prove the direction of the Bluetooth hands-free link with one call.

The open question is which way round Windows exposes the pair. Two answers are
possible and they are indistinguishable while the phone sits idle:

  A) capture = caller, render = our voice   -> Bluetooth is a full-duplex bridge
  B) capture = our mic, render = caller     -> Bluetooth can only listen

The test is deliberately two-sided, because one side alone cannot tell them
apart:

  * play a distinct beep pattern into the hands-free *render* endpoint
  * record the hands-free *capture* endpoint at the same time

If the far end hears the beeps, render reaches the call. If the capture is not
silent while they talk, capture carries the call. The operator reports the first
half; the recorded level reports the second.

    python -m agent.probe_duplex --seconds 600
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import time
from pathlib import Path

from .audio_devices import device_info, find_device
from .phone import Phone
from .probe_shared import _sweep
from .watch_call import _active_route, _overall, _precise_states


def _default_adb() -> str:
    candidate = Path(__file__).resolve().parent.parent / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


def _dbfs(chunk: bytes) -> float:
    """Level of a 16-bit mono chunk, in dBFS."""
    count = len(chunk) // 2
    if not count:
        return -120.0
    samples = struct.unpack(f"<{count}h", chunk[: count * 2])
    mean_square = sum(sample * sample for sample in samples) / count
    if mean_square <= 0:
        return -120.0
    return 10.0 * math.log10(mean_square / (32768.0**2))


def _beeps(rate: int, amplitude: int = 7000) -> bytes:
    """Three short 1 kHz beeps, unmistakeable on the far end."""
    frames = bytearray()
    for _ in range(3):
        for i in range(int(rate * 0.30)):
            frames += struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 1000 * i / rate)))
        frames += b"\x00\x00" * int(rate * 0.30)
    return bytes(frames)


def _duplex(out_name: str, in_name: str, seconds: float) -> dict[str, object]:
    """Open both directions, play the beeps, record the return, report levels."""
    import pyaudio

    report: dict[str, object] = {}
    try:
        out_index = find_device(out_name, need_input=False, need_output=True)
        in_index = find_device(in_name, need_input=True, need_output=False)
    except RuntimeError as exc:
        return {"error": f"device resolve failed: {exc}"}

    out_rate = int(device_info(out_index)["rate"])
    in_rate = int(device_info(in_index)["rate"])
    report["indices"] = f"out #{out_index} @ {out_rate} Hz, in #{in_index} @ {in_rate} Hz"

    pa = pyaudio.PyAudio()
    out_stream = None
    in_stream = None
    try:
        out_stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=out_rate,
            output=True,
            output_device_index=out_index,
            frames_per_buffer=320,
        )
        report["render"] = "opened"
    except OSError as exc:
        report["render"] = f"FAILED -> {exc}"

    try:
        in_stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=in_rate,
            input=True,
            input_device_index=in_index,
            frames_per_buffer=320,
        )
        report["capture"] = "opened"
    except OSError as exc:
        report["capture"] = f"FAILED -> {exc}"

    if out_stream is not None and in_stream is not None:
        try:
            tone = _beeps(out_rate)
            WINDOW = 1600  # 100 ms at 16 kHz
            levels: list[float] = []
            out_stream.start_stream()
            in_stream.start_stream()

            played = 0
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                chunk = in_stream.read(WINDOW, exception_on_overflow=False)
                levels.append(_dbfs(chunk))
                if played < len(tone):
                    piece = tone[played : played + 640]
                    out_stream.write(piece)
                    played += len(piece)

            out_stream.stop_stream()
            in_stream.stop_stream()

            report["capture_windows"] = len(levels)
            if levels:
                report["capture_peak_dbfs"] = round(max(levels), 1)
                report["capture_mean_dbfs"] = round(sum(levels) / len(levels), 1)
                report["capture_loud_windows"] = sum(1 for value in levels if value > -50.0)
            report["tone_sent"] = played >= len(tone)
        except OSError as exc:
            report["stream_error"] = str(exc)

    for stream in (in_stream, out_stream):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    pa.terminate()

    return report


async def _main(args: argparse.Namespace) -> int:
    phone = Phone(args.adb, args.serial)
    print(f"phone  : {await phone.ensure_connected()}", flush=True)
    print(f"state  : {_overall(await _precise_states(phone))}", flush=True)
    print(f"route  : {await _active_route(phone)}", flush=True)
    print("waiting for a call to become active -- place or answer it now", flush=True)

    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            states = await _precise_states(phone)
        except Exception:  # noqa: BLE001 - keep waiting through ADB blips
            await phone.reconnect()
            await asyncio.sleep(1.0)
            continue
        if _overall(states) == "ACTIVE":
            break
        await asyncio.sleep(0.4)
    else:
        print("no active call before the timeout", flush=True)
        return 1

    print(f"\nactive -- route: {await _active_route(phone)}", flush=True)

    for delay in (0.8, 3.0, 6.0, 8.0, 10.0):
        await asyncio.sleep(delay)
        route = await _active_route(phone)
        print(f"\nattempt at +{delay:.1f}s -- route: {route}", flush=True)
        report = await asyncio.to_thread(
            _duplex, args.output_device, args.input_device, args.probe_seconds
        )
        for key, value in report.items():
            print(f"  {key:<20} {value}", flush=True)

        if report.get("render") == "opened" and report.get("capture") == "opened":
            peak = report.get("capture_peak_dbfs")
            print(
                "  verdict hint         "
                + (
                    "capture carried audio -- if the far end also heard the beeps, "
                    "Bluetooth is full duplex"
                    if isinstance(peak, float) and peak > -50.0
                    else "capture was silent -- if the far end heard nothing, the "
                    "phone never handed the call to Bluetooth"
                ),
                flush=True,
            )
            break

    print(f"\nfinal route: {await _active_route(phone)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.probe_duplex")
    parser.add_argument("--seconds", type=float, default=600.0, help="how long to wait for a call")
    parser.add_argument("--probe-seconds", type=float, default=3.0)
    parser.add_argument("--input-device", default="iQOO Z9 5G")
    parser.add_argument("--output-device", default="iQOO Z9 5G")
    parser.add_argument("--adb", default=_default_adb())
    parser.add_argument("--serial")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
