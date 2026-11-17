#!/usr/bin/env python3
"""
main.py — CLI Entry Point for AI Voice Calling Agent.

Launches the FastAPI web console and Gemini Live 3.8 voice engine
with a pluggable telephony driver (virtual / windows / linux).

Usage:
    python main.py                            # defaults: virtual driver, port 8000, Aoede voice
    python main.py --driver virtual --port 8000
    python main.py --driver windows --voice Aoede
    python main.py --driver linux --port 9000
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

import uvicorn
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-voice-caller",
        description="AI Voice Calling Agent — Gemini Live 3.8 powered call assistant",
    )
    parser.add_argument(
        "--driver",
        choices=["virtual", "windows", "linux"],
        default=os.environ.get("VOICE_CALLER_DRIVER", "virtual"),
        help="Telephony driver (default: virtual)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VOICE_CALLER_PORT", "8000")),
        help="Web console port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("VOICE_CALLER_HOST", "0.0.0.0"),
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("VOICE_CALLER_VOICE", "Aoede"),
        help="Gemini Live voice persona (default: Aoede)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.8-live",
        help="Gemini Live model name (default: gemini-3.8-live)",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Audio input device name or index (optional)",
    )
    parser.add_argument(
        "--output-device",
        default=None,
        help="Audio output device name or index (optional)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def build_driver(args: argparse.Namespace):
    """Instantiate the appropriate telephony driver."""
    driver_name = args.driver.lower()

    if driver_name == "virtual":
        from telephony.virtual_driver import VirtualSimulatorDriver
        return VirtualSimulatorDriver(
            input_device=args.input_device,
            output_device=args.output_device,
        )
    elif driver_name == "windows":
        try:
            from telephony.windows_driver import WindowsAudioDriver
            return WindowsAudioDriver(
                input_device=args.input_device,
                output_device=args.output_device,
            )
        except ImportError:
            logger.warning("WindowsAudioDriver not available; falling back to VirtualSimulatorDriver.")
            from telephony.virtual_driver import VirtualSimulatorDriver
            return VirtualSimulatorDriver(
                input_device=args.input_device,
                output_device=args.output_device,
            )
    elif driver_name == "linux":
        try:
            from telephony.linux_driver import LinuxDriver
            return LinuxDriver()
        except ImportError:
            logger.warning("LinuxDriver not available; falling back to VirtualSimulatorDriver.")
            from telephony.virtual_driver import VirtualSimulatorDriver
            return VirtualSimulatorDriver()
    else:
        raise ValueError(f"Unknown driver: {driver_name}")


def build_app(args: argparse.Namespace):
    """Build the FastAPI app with configured components."""
    from core.gemini_live import GeminiLiveClient
    from server.database import DatabaseManager
    from server.call_state import CallStateManager
    from server.app import create_app

    # Verify API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Set it with: set GEMINI_API_KEY=your-api-key"
        )
        sys.exit(1)

    # Build components
    driver = build_driver(args)
    db = DatabaseManager()

    gemini_client = GeminiLiveClient(
        api_key=api_key,
        voice_name=args.voice,
        model=args.model,
    )

    call_manager = CallStateManager(
        db=db,
        driver=driver,
        gemini_client=gemini_client,
    )

    app = create_app(
        db=db,
        telephony_driver=driver,
        call_manager=call_manager,
    )

    return app


def main():
    args = parse_args()

    # Apply log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Banner
    print()
    print("=" * 60)
    print("  🤖  AI Voice Calling Agent")
    print("  ─────────────────────────────────────────")
    print(f"  Model:   {args.model}")
    print(f"  Voice:   {args.voice}")
    print(f"  Driver:  {args.driver}")
    print(f"  Console: http://localhost:{args.port}")
    print("=" * 60)
    print()

    app = build_app(args)

    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
    )
    server = uvicorn.Server(config)

    # Graceful shutdown on Ctrl+C
    def handle_signal(sig, frame):
        logger.info("Received shutdown signal. Stopping gracefully...")
        server.should_exit = True

    signal.signal(signal.SIGINT, handle_signal)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_signal)

    # Run
    logger.info(f"Starting server on {args.host}:{args.port}")
    server.run()


if __name__ == "__main__":
    main()
