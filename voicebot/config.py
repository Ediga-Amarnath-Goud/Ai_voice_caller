"""All configuration for the voicebot in one place."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---------------------------------------------------------------
SVCL = ROOT / "tools" / "svcl" / "svcl.exe"
ADB = ROOT / "platform-tools" / "adb.exe"
ROUTING_STATE = ROOT / ".voicebot_routing.json"
ADB_ENDPOINT = ROOT / ".adb_endpoint"
LOG_FILE = ROOT / "voicebot.log"

# --- Audio devices --------------------------------------------------------
# Phone Link renders the caller into the ordinary speakers; we tap their loopback.
LOOPBACK_DEVICE = "Speakers (Realtek(R) Audio)"
# AI audio is written here, then CABLE Output becomes Phone Link's mic.
CABLE_OUTPUT_DEVICE = "CABLE Input"
CABLE_CAPTURE_ID = r"VB-Audio Virtual Cable\Device\CABLE Output\Capture"
REALTEK_RENDER_ID = r"Realtek(R) Audio\Device\Speakers\Render"

# 24 kHz loopback in (native, no resample on input); MME CABLE Input is 44.1 kHz
# out. Pipecat resamples Gemini's 24 kHz output up to the device rate.
RATE_IN = 24000
RATE_OUT = 44100
BLOCK_MS = 10  # capture block size

# --- Call detection (debounced) -------------------------------------------
# A physical call is "active" only while loopback peak stays above THRESHOLD.
# Onset needs ON_FRAMES consecutive loud frames; offset needs OFF_MS of quiet.
# This stops the chatter that caused ghost/duplicate greetings in the old build.
CALL_THRESHOLD_DBFS = -42.0
CALL_ON_FRAMES = 4          # ~40 ms to arm
CALL_OFF_MS = 1200          # 1.2 s of quiet to mark the call ended

# --- Behaviour ------------------------------------------------------------
GREET_DELAY = 0.5           # seconds after call active before greeting
CALL_TIMEOUT = 30.0         # seconds to wait for a caller before giving up
CALL_GAP = 5.0              # seconds of silence that ends a call
AUDIO_DELAY = 1.0           # settle time after answering

# --- Gemini ---------------------------------------------------------------
MODEL = "models/gemini-3.8-live"
VOICES = ("Aoede", "Puck", "Charon", "Kore", "Fenrir")
VOICE = os.environ.get("VOICEBOT_VOICE", "Aoede")

# Language handling: START IN ENGLISH, only switch when caller speaks a
# supported language. Do NOT cycle languages on your own.
SYSTEM_PROMPT = (
    "You are a friendly, efficient voice assistant on a live phone call in India. "
    "You may speak ONLY in Telugu, Hindi, Kannada, or English. "
    "IMPORTANT: Always START the conversation in ENGLISH. "
    "Only switch to another language AFTER the caller has spoken in that language. "
    "Do NOT change languages on your own. Match the caller's language once they "
    "have established it. If the caller speaks another language or the line is "
    "unclear, politely ask them to repeat in Telugu, Hindi, Kannada, or English. "
    "Keep the conversation concise because this is a phone call, not an essay."
)


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.")
    return key
