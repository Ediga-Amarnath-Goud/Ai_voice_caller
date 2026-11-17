"""Probe the phone's audio endpoints without Pipecat in the way.

Use this during a real call to prove (or disprove) that the hands-free path
carries audio. It can place the call itself, answer an incoming one, or just
attach to a call you started by hand.

    python -m agent.probe_audio --dial 9133233391 --sim-slot 2 --seconds 10 --tone
    python -m agent.probe_audio --wait --seconds 8 --tone
    python -m agent.probe_audio --seconds 6 --tone      # call already up

A PortAudio error -9999 means no call is active (or SCO never came up); -9996
means another process owns the endpoint (Phone Link, a browser, Teams).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyaudio

from .audio_devices import device_info, find_device
from .phone import STATE_OFFHOOK, STATE_RINGING, Phone, normalize_number


def _default_adb() -> str:
    candidate = Path(__file__).resolve().parent.parent / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


def _tone_chunk(rate: int, frames: int, freq: float = 440.0, amplitude: float = 0.2) -> bytes:
    t = np.arange(frames, dtype=np.float32) / rate
    wave = amplitude * np.sin(2 * math.pi * freq * t)
    return (wave * 32767).astype(np.int16).tobytes()


async def _establish(phone: Phone, args: argparse.Namespace) -> None:
    """Put the handset on an active call before any stream is opened."""
    model = await phone.ensure_connected()
    print(f"phone: {model}")

    if args.dial:
        number = normalize_number(args.dial, args.country_code)
        slot = f"SIM {args.sim_slot}" if args.sim_slot else "default SIM"
        print(f"dialing {number} from {slot} ...")
        await phone.dial(number, sim_slot=args.sim_slot)
        await phone.wait_for_state(STATE_OFFHOOK, timeout=args.call_timeout)
        print(f"line states: {await phone.call_states()}")
        print(f"offhook; waiting {args.dial_grace:.0f}s for the other side")
        await asyncio.sleep(args.dial_grace)
        return

    state = await phone.wait_for_state(
        {STATE_RINGING, STATE_OFFHOOK}, timeout=args.call_timeout
    )
    if state == STATE_OFFHOOK:
        print("already on a call; probing it")
        return
    print("ringing; answering")
    await phone.answer()
    await phone.wait_for_state(STATE_OFFHOOK, timeout=15)
    await asyncio.sleep(args.audio_delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.probe_audio")
    parser.add_argument("--input-device", default="iQOO Z9 5G")
    parser.add_argument("--output-device", default="iQOO Z9 5G")
    parser.add_argument("--rate", type=int, default=None, help="Sample rate (default: the device's rate).")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--tone", action="store_true", help="Play a 440 Hz tone into the call.")
    parser.add_argument("--dial", help="Number to call before probing.")
    parser.add_argument("--sim-slot", type=int, choices=(1, 2))
    parser.add_argument("--country-code", default="91")
    parser.add_argument("--dial-grace", type=float, default=8.0)
    parser.add_argument("--wait", action="store_true", help="Answer an incoming call, then probe.")
    parser.add_argument("--audio-delay", type=float, default=1.5)
    parser.add_argument("--call-timeout", type=float, default=60.0)
    parser.add_argument("--keep-call", action="store_true", help="Do not hang up afterwards.")
    parser.add_argument("--adb", default=_default_adb())
    parser.add_argument("--serial")
    args = parser.parse_args(argv)

    try:
        in_index = find_device(args.input_device, need_input=True)
        out_index = find_device(args.output_device, need_output=True)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    phone: Phone | None = None
    if args.dial or args.wait:
        phone = Phone(args.adb, args.serial)
        try:
            asyncio.run(_establish(phone, args))
        except Exception as exc:  # noqa: BLE001 - diagnostic tool
            print(f"call setup failed: {exc}", file=sys.stderr)
            with contextlib.suppress(Exception):
                asyncio.run(phone.hangup())
            return 1

    print(f"input : [{in_index}]")
    print(f"output: [{out_index}]")
    rate = args.rate or int(device_info(in_index)["rate"])
    print(f"opening {rate} Hz mono for {args.seconds:.1f}s ...")

    pa = pyaudio.PyAudio()
    in_stream = None
    out_stream = None
    try:
        in_stream = pa.open(
            format=pa.get_format_from_width(2),
            channels=1,
            rate=rate,
            input=True,
            input_device_index=in_index,
            frames_per_buffer=320,
        )
        print("input stream open")

        if args.tone:
            out_stream = pa.open(
                format=pa.get_format_from_width(2),
                channels=1,
                rate=rate,
                output=True,
                output_device_index=out_index,
                frames_per_buffer=320,
            )
            print("output stream open (playing tone)")

        captured = bytearray()
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            captured.extend(in_stream.read(320, exception_on_overflow=False))
            if out_stream is not None:
                out_stream.write(_tone_chunk(rate, 320))

        samples = np.frombuffer(bytes(captured), dtype=np.int16)
        if samples.size:
            peak = float(np.max(np.abs(samples))) / 32768.0
            rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
            peak_db = 20 * math.log10(peak) if peak > 0 else -math.inf
            rms_db = 20 * math.log10(rms) if rms > 0 else -math.inf
            print(f"captured {samples.size} samples")
            print(f"peak {peak_db:6.1f} dBFS   rms {rms_db:6.1f} dBFS")
            if peak_db < -60:
                print("silence: the endpoint opened but carried nothing.")
                print("  - was the other side actually off-hook?")
                print("  - is Phone Link holding the call audio?")
        else:
            print("no samples captured")
        return 0
    except OSError as exc:
        print(f"PortAudio could not open the device: {exc}", file=sys.stderr)
        if "-9996" in str(exc):
            print("  -9996 = device unavailable; another process owns it.", file=sys.stderr)
        if "-9999" in str(exc):
            print("  -9999 = no active call / SCO never came up.", file=sys.stderr)
        return 1
    finally:
        if in_stream is not None:
            in_stream.stop_stream()
            in_stream.close()
        if out_stream is not None:
            out_stream.stop_stream()
            out_stream.close()
        pa.terminate()
        if phone is not None and not args.keep_call:
            with contextlib.suppress(Exception):
                asyncio.run(phone.hangup())
            print("hung up")


if __name__ == "__main__":
    raise SystemExit(main())
