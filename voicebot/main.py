"""Entry point: python -m voicebot.main [options]

The call loop is intentionally simple:
  1. wait for debounced caller audio on the loopback
  2. greet once
  3. Gemini streams answers straight to CABLE Input as the caller speaks
  4. when the line goes quiet for CALL_GAP, reset and wait for the next call
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib

from loguru import logger

from . import audio_devices, config, routing
from .phone import Phone, STATE_IDLE, STATE_OFFHOOK, STATE_RINGING
from .pipeline import VoicePipeline

logger.add(config.LOG_FILE, rotation="5 MB", level="INFO")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m voicebot.main")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--bridge", action="store_true",
                   help="legacy flag (this build is always the software bridge)")
    p.add_argument("--output-device", default=config.CABLE_OUTPUT_DEVICE,
                   help="PyAudio output device (default: CABLE Input)")
    p.add_argument("--no-audio-switch", action="store_true",
                   help="don't switch the mic routing (call already up)")
    p.add_argument("--dial", help="number to dial on startup")
    p.add_argument("--sim-slot", type=int, choices=(1, 2))
    p.add_argument("--answer", action="store_true", help="answer the first incoming call")
    p.add_argument("--greet", action="store_true", help="agent speaks first")
    p.add_argument("--no-daemon", action="store_true", help="exit after one call")
    p.add_argument("--serial", help="ADB serial override")
    return p.parse_args(argv)


async def _establish_call(phone: Phone, args: argparse.Namespace) -> None:
    model = await phone.ensure_connected()
    logger.info(f"phone: {model}")
    if args.dial:
        number = __import__("re").sub(r"\D", "", args.dial)
        if not number.startswith("+"):
            number = "+91" + number.lstrip("0")
        logger.info(f"dialing {number}")
        await phone.dial(number, sim_slot=args.sim_slot)
        await phone.wait_for_state(STATE_OFFHOOK, timeout=60)
        await asyncio.sleep(8)  # give the callee time to answer
    else:
        logger.info("waiting for incoming call...")
        await phone.wait_for_state(STATE_RINGING, timeout=120)
        await phone.answer()
        await phone.wait_for_state(STATE_OFFHOOK, timeout=15)
        await asyncio.sleep(config.AUDIO_DELAY)
    logger.info("call is up")


async def _run(args: argparse.Namespace) -> None:
    idx = audio_devices.find_device(args.output_device, need_output=True)
    pipeline = VoicePipeline(idx, serial=args.serial)
    phone = Phone(serial=args.serial)
    controlled = bool(args.dial or args.answer)

    # 1. Start and warm up pipeline before answering/dialing
    await pipeline.prepare()
    runner_task = asyncio.create_task(pipeline.run())
    await pipeline.wait_ready()

    try:
        while True:
            if controlled:
                # Controlled call via ADB
                await _establish_call(phone, args)
                if args.greet:
                    await asyncio.sleep(config.GREET_DELAY)
                    await pipeline.greet()

                # Wait until phone reports the call ended
                quiet = 0
                while True:
                    state = await phone.call_state()
                    quiet = quiet + 1 if state == STATE_IDLE else 0
                    if quiet >= 2:
                        logger.info("call ended")
                        pipeline.input.call_active.clear()
                        break
                    await asyncio.sleep(1)
            else:
                # Daemon mode waiting for caller on Phone Link
                logger.info("waiting for caller audio on the loopback...")
                got = await pipeline.input.wait_for_call_active(timeout=config.CALL_TIMEOUT)
                if not got:
                    logger.warning(f"no caller audio within {config.CALL_TIMEOUT:.0f}s; retrying")
                    continue

                if args.greet:
                    await asyncio.sleep(config.GREET_DELAY)
                    await pipeline.greet()

                # Wait for the line to go quiet after the last speech
                while pipeline.input.last_audio_age() < config.CALL_GAP:
                    await asyncio.sleep(0.2)
                pipeline.input.call_active.clear()

            if args.no_daemon:
                logger.info("call completed")
                break
    finally:
        runner_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner_task
        pipeline.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        print(audio_devices.format_table())
        return 0
    if not args.no_audio_switch:
        routing.enable()
    try:
        asyncio.run(_run(args))
    finally:
        if not args.no_audio_switch:
            routing.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
