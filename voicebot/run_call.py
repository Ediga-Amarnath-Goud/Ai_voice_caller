"""Run the phone-in-the-loop voice agent.

Two transport modes are supported.

``--bridge`` is the working software-only path. Phone Link is the call-audio
carrier: it renders the caller into the default Windows speakers and records
its mic from the default Windows recording device. We tap the caller through a
WASAPI loopback of the speakers and feed the AI voice back through a virtual
cable set as the default mic::

    caller -> Phone Link -> Realtek speakers -> loopback -> Pipecat -> Gemini
    Gemini -> Pipecat -> CABLE Input -> CABLE Output -> default mic -> Phone Link
    phone  <------------ ADB (wireless) -> dial/answer/hangup -------------> PC

No SIP, no Android app, no Bluetooth hands-free endpoint and no extra hardware.
The legacy LocalAudioTransport path is kept for a real sound card.

Run it as a module so the relative imports work::

    python -m agent.run_call --list-devices
    python -m agent.run_call --bridge --dial 9133233391 --sim-slot 2 --greet
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path

from . import audio_routing
from .audio_devices import device_info, find_device, format_table, list_devices
from .phone import STATE_OFFHOOK, STATE_RINGING, Phone, normalize_number

try:  # loguru ships with Pipecat; keep --list-devices usable without it.
    from loguru import logger
except ImportError:  # pragma: no cover - depends on the environment
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("agent")

try:  # Pipecat is optional for --list-devices, which only needs PyAudio.
    from pipecat.frames.frames import InputTextRawFrame, LLMRunFrame, TextFrame
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import (
        PipelineParams,
        PipelineWorker,
        ProcessorUnusablePolicy,
    )
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        AssistantTurnStoppedMessage,
        LLMContextAggregatorPair,
        UserTurnMessageAddedMessage,
    )
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )
    from pipecat.workers.runner import WorkerRunner

    _PIPECAT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - depends on the environment
    _PIPECAT_ERROR = exc

DEFAULT_PROMPT = (
    "You are a friendly, efficient voice assistant on a live phone call in India. "
    "Speak in short, natural sentences and never read out lists or options. "
    "If the line is unclear, ask the caller to repeat rather than guessing. "
    "Keep the whole conversation concise: this is a phone call, not an essay."
)

MODEL = "models/gemini-3.8-live"
VOICES = ("Aoede", "Puck", "Charon", "Kore", "Fenrir")

#: The endpoint Phone Link renders the caller into. Its loopback twin is our mic.
DEFAULT_LOOPBACK = "Speakers (Realtek(R) Audio)"
#: The virtual playback device that becomes Phone Link's microphone.
DEFAULT_CABLE_OUTPUT = "CABLE Input"

#: Script ranges for allowed languages (Unicode blocks).
#: Telugu U+0C00–U+0C7F, Kannada U+0C80–U+0CFF,
#: Devanagari/Hindi U+0900–U+097F, Latin/English U+0000–U+024F.
_ALLOWED_SCRIPTS: dict[str, tuple[int, int]] = {
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "devanagari": (0x0900, 0x097F),
    "latin": (0x0000, 0x024F),
}
ALLOWED_LANGUAGES = frozenset(_ALLOWED_SCRIPTS)


def _detect_script(text: str | None) -> str | None:
    """Detect the primary Unicode script of ``text``.

    Returns one of ``telugu``, ``kannada``, ``devanagari``, ``latin`` or
    ``None`` when the text is empty or uses an unsupported script.
    """
    if not text or not text.strip():
        return None

    counts: dict[str, int] = {name: 0 for name in _ALLOWED_SCRIPTS}
    for ch in text:
        code = ord(ch)
        for name, (start, end) in _ALLOWED_SCRIPTS.items():
            if start <= code <= end:
                counts[name] += 1
                break

    total = sum(counts.values())
    if total == 0:
        return None  # characters outside all allowed scripts
    return max(counts, key=counts.get)


def _language_allowed(text: str | None) -> bool:
    """True if ``text`` uses one of the four allowed scripts."""
    script = _detect_script(text)
    return script in ALLOWED_LANGUAGES


class LanguageFilterProcessor(FrameProcessor):
    """Drop user turns that are not in Telugu, Hindi, Kannada or English.

    Valid frames pass through unchanged.  Invalid frames are replaced by a
    ``TextFrame`` asking the caller to repeat in one of the four languages,
    so the LLM speaks it back instead of hallucinating a response.
    """

    def __init__(self) -> None:
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Inspect text frames for language.
        text = getattr(frame, "text", None)
        if text and not _language_allowed(text):
            await self.push_frame(
                TextFrame(
                    "Please speak in Telugu, Hindi, Kannada, or English. "
                    "Telugu, Hindi, Kannada leda English lo matladandi."
                ),
                direction,
            )
            return
        await self.push_frame(frame, direction)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_adb() -> str:
    candidate = _repo_root() / "platform-tools" / "adb.exe"
    return str(candidate) if candidate.exists() else "adb"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agent.run_call",
        description="Pipecat voice agent bridged to a phone's SIM call over audio + ADB.",
    )
    parser.add_argument("--list-devices", action="store_true", help="Print PyAudio devices and exit.")
    parser.add_argument(
        "--bridge",
        action="store_true",
        help="Use the direct HFP bridge (Bluetooth Hands-Free mic/speaker).",
    )
    parser.add_argument(
        "--no-audio-switch",
        action="store_true",
        help="Do not switch/restart audio devices (use when the call is already up).",
    )
    parser.add_argument(
        "--loopback-device",
        default=DEFAULT_LOOPBACK,
        help="Render endpoint whose loopback carries the caller (bridge mode).",
    )
    parser.add_argument("--input-device", help="PyAudio input device name or index (phone audio in).")
    parser.add_argument("--output-device", help="PyAudio output device name or index (audio to phone).")
    parser.add_argument("--host-api", help="Restrict device matching to this host API (e.g. MME).")
    parser.add_argument("--rate-in", type=int, default=None, help="Input rate in Hz (default: the device's rate).")
    parser.add_argument("--rate-out", type=int, default=None, help="Output rate in Hz (default: the device's rate).")
    parser.add_argument("--dial", help="Number to call on startup.")
    parser.add_argument("--sim-slot", type=int, choices=(1, 2), help="Preferred SIM slot for --dial.")
    parser.add_argument("--country-code", default="91", help="Country code for numbers without '+'.")
    parser.add_argument("--answer", action="store_true", help="Answer the first incoming call.")
    parser.add_argument("--greet", action="store_true", help="Make the agent speak first.")
    parser.add_argument(
        "--dsp",
        action="store_true",
        help="Enable vocal mastering DSP (high-pass, de-box, presence boost, AGC).",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Stay running and auto-answer every Phone Link call detected on the loopback.",
    )
    parser.add_argument(
        "--greet-delay",
        type=float,
        default=0.5,
        help="Seconds after call audio is detected before the agent speaks.",
    )
    parser.add_argument(
        "--call-active-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for caller audio on the loopback before giving up.",
    )
    parser.add_argument(
        "--call-gap",
        type=float,
        default=3.0,
        help="Seconds of silence before treating a call as ended (daemon mode).",
    )
    parser.add_argument(
        "--dial-grace",
        type=float,
        default=8.0,
        help="Seconds to wait for the callee to answer an outbound call.",
    )
    parser.add_argument(
        "--audio-delay",
        type=float,
        default=1.5,
        help="Seconds between answering and opening the audio streams.",
    )
    parser.add_argument("--call-timeout", type=float, default=120.0, help="Seconds to wait for ring/answer.")
    parser.add_argument("--adb", default=_default_adb(), help="Path to adb.")
    parser.add_argument("--serial", help="ADB serial; defaults to the only attached device.")
    parser.add_argument("--api-key", default=None, help="Gemini API key (default: env).")
    parser.add_argument("--voice", default=os.environ.get("VOICE_CALLER_VOICE", "Aoede"), choices=VOICES)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--system-prompt", default=os.environ.get("VOICE_CALLER_PROMPT", DEFAULT_PROMPT))
    args = parser.parse_args(argv)

    if args.bridge and not args.output_device:
        args.output_device = DEFAULT_CABLE_OUTPUT
    return args


def _device_label(index: int | None) -> str:
    if index is None:
        return "system default"
    for device in list_devices():
        if device["index"] == index:
            return f"[{index}] {device['name']} ({device['host_api']})"
    return f"[{index}]"


async def _establish_call(phone: Phone, args: argparse.Namespace) -> None:
    """Put the handset on an active call before any audio stream is opened."""
    model_name = await phone.ensure_connected()
    logger.info(f"Phone detected: {model_name}")

    if args.dial:
        number = normalize_number(args.dial, args.country_code)
        logger.info(f"Dialing {number} ...")
        await phone.dial(number, sim_slot=args.sim_slot)
        await phone.wait_for_state(STATE_OFFHOOK, timeout=args.call_timeout)
        # Outbound answer cannot be detected without privileged APIs: give the
        # callee time to pick up, then give the audio route time to settle.
        await asyncio.sleep(args.dial_grace)
    else:
        logger.info("Waiting for an incoming call ...")
        await phone.wait_for_state(STATE_RINGING, timeout=args.call_timeout)
        logger.info("Ringing; answering.")
        await phone.answer()
        await phone.wait_for_state(STATE_OFFHOOK, timeout=15)
        await asyncio.sleep(args.audio_delay)

    logger.info("Call is up; opening audio streams.")


async def run(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.")

    phone = Phone(adb=args.adb, serial=args.serial)
    controlled = bool(args.dial or args.answer)

    if args.bridge and not args.no_audio_switch:
        # Switch the console mic to the virtual cable for this call.
        # DO NOT restart/kill Phone Link here — doing so severs the Bluetooth
        # Hands-Free (HFP) calling link to the physical phone!
        audio_routing.enable()

    output_index = find_device(
        args.output_device,
        need_output=True,
        host_api=args.host_api,
    )
    output_info = device_info(output_index)

    if args.bridge:
        from .bridge_transport import (
            BridgeTransport,
            BridgeTransportParams,
            LoopbackInputTransport,
        )

        rate_in = args.rate_in or 24000
        rate_out = args.rate_out or int(output_info["rate"])
        logger.info(f"audio in : loopback of {args.loopback_device} @ {rate_in} Hz")
        logger.info(f"audio out: {_device_label(output_index)} @ {rate_out} Hz")
        transport = BridgeTransport(
            BridgeTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                loopback_device=args.loopback_device,
                output_device_index=output_index,
            )
        )
    else:
        input_index = find_device(args.input_device, need_input=True, host_api=args.host_api)
        input_info = device_info(input_index)
        rate_in = args.rate_in or int(input_info["rate"])
        rate_out = args.rate_out or int(output_info["rate"])
        logger.info(f"audio in : {_device_label(input_index)} @ {rate_in} Hz")
        logger.info(f"audio out: {_device_label(output_index)} @ {rate_out} Hz")
        transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                input_device_index=input_index,
                output_device_index=output_index,
            )
        )

    llm = GeminiLiveLLMService(
        api_key=api_key,
        system_instruction=args.system_prompt,
        settings=GeminiLiveLLMService.Settings(
            model=args.model,
            voice=args.voice,
        ),
        # Do not run inference on startup: audio is only generated once a call
        # is actually connected.
        inference_on_context_initialization=False,
    )

    context = LLMContext([])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    language_filter = LanguageFilterProcessor()

    pipeline_processors: list[FrameProcessor] = [
        transport.input(),
        user_aggregator,
        llm,
    ]
    if getattr(args, "dsp", False):
        from .vocal_dsp import VocalMasteringProcessor

        pipeline_processors.append(VocalMasteringProcessor(sample_rate=24000))

    pipeline_processors.extend(
        [
            transport.output(),
            assistant_aggregator,
        ]
    )

    pipeline = Pipeline(pipeline_processors)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=rate_in,
            audio_out_sample_rate=rate_out,
            enable_metrics=True,
        ),
        idle_timeout_secs=None,  # a quiet call must not kill the bot
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):
        started.set()

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def _on_user(aggregator, message: UserTurnMessageAddedMessage):
        text = message.content or ""
        script = _detect_script(text)
        if script not in ALLOWED_LANGUAGES:
            logger.warning(
                f"caller: language not allowed ({script or 'unknown'}): {text[:60]!r}"
            )
        else:
            logger.info(f"caller: {text}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def _on_assistant(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"agent : {message.content}")

    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)

    # Everything above is setup. Do it *before* the call so that answering is
    # followed immediately by audio, not by several seconds of dead air.
    if controlled:
        await _establish_call(phone, args)

    async def handle_calls() -> None:
        """Loop: wait for caller audio, trigger Gemini, wait for call end, repeat."""
        await asyncio.wait_for(started.wait(), timeout=20)
        input_transport = transport.input()
        # A single physical call can make the loopback detector chatter
        # (call_active drops out for a block or two mid-sentence). Greet only
        # once per call so Gemini never receives a stack of "Hello" turns that
        # make it reply "Hello Hello Hello ..." instead of a real answer.
        greeted_this_call = False

        while True:
            if args.bridge and not controlled and hasattr(input_transport, "wait_for_call_active"):
                logger.info("Waiting for caller audio on the loopback...")
                got_call = await input_transport.wait_for_call_active(
                    timeout=args.call_active_timeout
                )
                if not got_call:
                    logger.warning(
                        "No caller audio detected within "
                        f"{args.call_active_timeout:.0f}s; trying again."
                    )
                    continue

            if args.greet and not greeted_this_call:
                await asyncio.sleep(args.greet_delay)
                logger.info("Triggering greeting.")
                await worker.queue_frames([InputTextRawFrame("Hello")])
                greeted_this_call = True

            # Wait for the call to end before listening for the next one.
            if controlled:
                quiet_polls = 0
                while True:
                    state = await phone.call_state()
                    if state == 0:
                        quiet_polls += 1
                        if quiet_polls >= 2:
                            logger.info("Call ended; hanging up and resetting.")
                            await phone.hangup()
                            # Clear session resumption handle so next call starts fresh
                            # (prevents "Hello Hello Hello" carryover from previous greetings)
                            if hasattr(llm, "_session_resumption_handle"):
                                llm._session_resumption_handle = None
                            if hasattr(input_transport, "_call_active"):
                                input_transport._call_active.clear()
                            greeted_this_call = False
                            break
                    else:
                        quiet_polls = 0
                    await asyncio.sleep(1.0)
            else:
                # Non-controlled: wait for silence on the loopback before
                # listening for the next call.
                while not input_transport.is_silent_for(args.call_gap):
                    await asyncio.sleep(0.2)
                # Clear session resumption handle so next call starts fresh
                if hasattr(llm, "_session_resumption_handle"):
                    llm._session_resumption_handle = None
                if hasattr(input_transport, "_call_active"):
                    input_transport._call_active.clear()
                greeted_this_call = False

            if not args.daemon:
                logger.info("Call completed.")
                break

    # Start the runner (blocks until cancelled) and drive calls alongside it.
    runner_task = asyncio.create_task(runner.run())
    call_task = asyncio.create_task(handle_calls())

    try:
        await asyncio.wait(
            {runner_task, call_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for task in (runner_task, call_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_devices:
        print(format_table(list_devices()))
        return 0

    if _PIPECAT_ERROR is not None:
        print(
            "Pipecat is not installed in this Python environment.\n\n"
            '    python -m venv .venv-agent\n'
            '    .venv-agent\\Scripts\\python -m pip install -r agent/requirements.txt\n'
            '    .venv-agent\\Scripts\\python -m agent.run_call --list-devices\n\n'
            f"Import error: {_PIPECAT_ERROR}",
            file=sys.stderr,
        )
        return 2

    if not args.bridge and not args.output_device:
        print("An output device is required for non-bridge mode.", file=sys.stderr)
        print("\nAvailable devices:\n" + format_table(list_devices()), file=sys.stderr)
        return 2
    if not args.bridge and not args.input_device:
        print("An input device is required for non-bridge mode.", file=sys.stderr)
        print("\nAvailable devices:\n" + format_table(list_devices()), file=sys.stderr)
        return 2

    try:
        try:
            asyncio.run(run(args))
        except OSError as exc:
            print(f"Audio device error: {exc}", file=sys.stderr)
            if "-9999" in str(exc) or "-9996" in str(exc):
                print(
                    "In bridge mode the loopback endpoint must be a live render device.\n"
                    "Check that --loopback-device matches the speakers Phone Link uses.",
                    file=sys.stderr,
                )
            return 1
    finally:
        if args.bridge and not args.no_audio_switch:
            audio_routing.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
