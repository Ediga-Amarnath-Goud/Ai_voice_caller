"""
e2e_tests/conftest.py

Comprehensive test harness, mock servers, audio utilities, SQLite WAL database,
telephony driver mocks, and FastAPI test fixtures for the AI Voice Calling Assistant E2E Test Suite.
Supports hermetic, independent execution without requiring physical Bluetooth hardware,
live cellular connections, or external Google Gemini network access.
"""

import asyncio
import base64
import json
import logging
import math
import os
import struct
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import aiosqlite
import httpx
import numpy as np
import pytest
import pytest_asyncio
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, Response
from starlette.staticfiles import StaticFiles
from google import genai
from google.genai import types
import google.genai.live as live_module

from core.audio_processor import AudioProcessor
from core.gemini_live import GeminiLiveClient, DEFAULT_SYSTEM_INSTRUCTION
from core.tools import ToolRegistry, get_tool_declarations

logger = logging.getLogger("e2e_tests.conftest")


# ============================================================================
# 1. MOCK GEMINI LIVE WEBSOCKET SERVER
# ============================================================================

class MockGeminiLiveServer:
    """
    In-process WebSocket server emulating the Gemini Live 3.8 Bidi protocol.
    Validates Aoede voice persona, system instructions, streaming PCM audio chunks,
    barge-in interruptions, operator whisper steering, and agent tool execution.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.server = None
        self.actual_port = None
        self.active_ws = None
        self.connected_clients_count = 0

        # State tracking
        self.setup_config: Optional[dict] = None
        self.setup_received_event = asyncio.Event()
        self.received_messages: List[dict] = []
        self.received_audio_chunks: List[bytes] = []
        self.received_whispers: List[str] = []
        self.received_tool_responses: List[dict] = []
        self.tool_response_event = asyncio.Event()
        self.interrupted_event = asyncio.Event()

    async def handler(self, ws):
        """WebSocket handler for client connection."""
        self.active_ws = ws
        self.connected_clients_count += 1
        try:
            async for raw_msg in ws:
                if isinstance(raw_msg, bytes):
                    try:
                        raw_msg = raw_msg.decode("utf-8")
                    except Exception:
                        continue

                try:
                    data = json.loads(raw_msg)
                except Exception:
                    continue

                if not isinstance(data, dict):
                    continue

                self.received_messages.append(data)

                # 1. Setup handshake
                if "setup" in data:
                    self.setup_config = data["setup"]
                    self.setup_received_event.set()
                    # Respond with setupComplete
                    await ws.send(json.dumps({
                        "setupComplete": {
                            "sessionId": f"mock-session-{int(time.time() * 1000)}"
                        }
                    }))

                # 2. Realtime input (Audio chunk or text whisper)
                rt = data.get("realtimeInput") or data.get("realtime_input")
                if rt and isinstance(rt, dict):
                    media_chunks = rt.get("mediaChunks") or rt.get("media_chunks") or []
                    for chunk in media_chunks:
                        raw_data = chunk.get("data")
                        if raw_data:
                            try:
                                pad = "=" * ((4 - len(raw_data) % 4) % 4)
                                pcm = base64.urlsafe_b64decode(raw_data + pad)
                                self.received_audio_chunks.append(pcm)
                            except Exception:
                                pass
                    if "text" in rt and rt["text"]:
                        self.received_whispers.append(rt["text"])

                # 3. Client content (whisper turns)
                client_content = data.get("clientContent") or data.get("client_content")
                if client_content and isinstance(client_content, dict):
                    turns = client_content.get("turns", [])
                    for turn in turns:
                        parts = turn.get("parts", [])
                        for part in parts:
                            if "text" in part and part["text"]:
                                self.received_whispers.append(part["text"])

                # 4. Tool responses
                tr = data.get("toolResponse") or data.get("tool_response")
                if tr and isinstance(tr, dict):
                    responses = tr.get("functionResponses") or tr.get("function_responses") or []
                    for resp in responses:
                        self.received_tool_responses.append(resp)
                    self.tool_response_event.set()

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if self.active_ws == ws:
                self.active_ws = None

    async def start(self):
        """Starts the mock WebSocket server on an ephemeral port."""
        self.server = await websockets.serve(self.handler, self.host, self.port)
        self.actual_port = self.server.sockets[0].getsockname()[1]
        logger.info(f"MockGeminiLiveServer started on port {self.actual_port}")

    async def stop(self):
        """Closes all client connections and shuts down the server."""
        if self.active_ws:
            try:
                await self.active_ws.close()
            except Exception:
                pass
            self.active_ws = None
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def wait_for_setup(self, timeout: float = 3.0) -> dict:
        """Waits for initial client setup message."""
        await asyncio.wait_for(self.setup_received_event.wait(), timeout=timeout)
        return self.setup_config

    async def emit_speech(
        self,
        text: str = "Assistant speech",
        pcm_24k: Optional[bytes] = None,
        turn_complete: bool = False
    ):
        """Emits synthesized speech and 24kHz audio chunk back to client."""
        if not self.active_ws:
            raise RuntimeError("No active WebSocket client connected to mock Gemini server")

        if pcm_24k is None:
            # 20ms of 24kHz 16-bit mono PCM = 480 samples = 960 bytes
            pcm_24k = b"\x00" * 960

        payload = {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {"text": text},
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": base64.b64encode(pcm_24k).decode("ascii")
                            }
                        }
                    ]
                },
                "turnComplete": turn_complete,
                "interrupted": False
            }
        }
        await self.active_ws.send(json.dumps(payload))

    async def emit_interrupted(self):
        """Emits speech barge-in interruption signal."""
        if not self.active_ws:
            raise RuntimeError("No active WebSocket client connected to mock Gemini server")
        payload = {
            "serverContent": {
                "interrupted": True
            }
        }
        await self.active_ws.send(json.dumps(payload))

    async def emit_tool_call(self, name: str, args: dict, call_id: str = "call_mock_1"):
        """Emits a function call request to client."""
        if not self.active_ws:
            raise RuntimeError("No active WebSocket client connected to mock Gemini server")
        self.tool_response_event.clear()
        payload = {
            "toolCall": {
                "functionCalls": [
                    {
                        "name": name,
                        "args": args,
                        "id": call_id
                    }
                ]
            }
        }
        await self.active_ws.send(json.dumps(payload))

    async def wait_for_tool_response(self, timeout: float = 3.0) -> dict:
        """Waits for client to respond to tool call."""
        await asyncio.wait_for(self.tool_response_event.wait(), timeout=timeout)
        return self.received_tool_responses[-1]

    async def force_disconnect(self, code: int = 1011):
        """Simulates unexpected remote socket closure."""
        if self.active_ws:
            await self.active_ws.close(code=code)
            self.active_ws = None


@pytest_asyncio.fixture
async def mock_gemini():
    """Starts MockGeminiLiveServer and patches google-genai ws_connect to route locally."""
    server = MockGeminiLiveServer()
    await server.start()

    original_ws_connect = live_module.ws_connect

    def patched_ws_connect(uri, **kwargs):
        mock_uri = f"ws://127.0.0.1:{server.actual_port}"
        kwargs.pop("ssl", None)
        return original_ws_connect(mock_uri, **kwargs)

    live_module.ws_connect = patched_ws_connect

    try:
        yield server
    finally:
        live_module.ws_connect = original_ws_connect
        await server.stop()


# ============================================================================
# 2. SYNTHETIC AUDIO GENERATOR & SIGNAL ANALYSIS HELPERS
# ============================================================================

class AudioSynthesizer:
    """Generates synthetic PCM frames for deterministic testing."""

    @staticmethod
    def generate_sine_pcm(
        freq_hz: float = 440.0,
        duration_sec: float = 1.0,
        sample_rate: int = 16000,
        amplitude: float = 0.5
    ) -> bytes:
        if duration_sec <= 0 or sample_rate <= 0:
            return b""
        num_samples = int(duration_sec * sample_rate)
        t = np.arange(num_samples) / float(sample_rate)
        samples = np.sin(2.0 * np.pi * freq_hz * t) * amplitude
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def generate_harmonic_pcm(
        frequencies: List[float] = [300.0, 600.0, 1200.0],
        duration_sec: float = 1.0,
        sample_rate: int = 16000,
        amplitudes: Optional[List[float]] = None
    ) -> bytes:
        if duration_sec <= 0 or sample_rate <= 0:
            return b""
        num_samples = int(duration_sec * sample_rate)
        t = np.arange(num_samples) / float(sample_rate)
        if amplitudes is None:
            amplitudes = [1.0 / len(frequencies)] * len(frequencies)
        samples = np.zeros(num_samples, dtype=np.float32)
        for f, a in zip(frequencies, amplitudes):
            samples += np.sin(2.0 * np.pi * f * t) * a
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def generate_silence_pcm(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
        num_samples = int(duration_sec * sample_rate)
        return b"\x00" * (num_samples * 2)

    @staticmethod
    def generate_noise_pcm(duration_sec: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.2) -> bytes:
        num_samples = int(duration_sec * sample_rate)
        samples = np.random.uniform(-amplitude, amplitude, num_samples).astype(np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        return (samples * 32767.0).astype(np.int16).tobytes()

    @staticmethod
    def chunk_pcm(pcm_bytes: bytes, chunk_duration_ms: float = 20.0, sample_rate: int = 16000) -> List[bytes]:
        bytes_per_sample = 2
        samples_per_chunk = int((chunk_duration_ms / 1000.0) * sample_rate)
        chunk_len = samples_per_chunk * bytes_per_sample
        if chunk_len <= 0:
            return [pcm_bytes] if pcm_bytes else []
        chunks = []
        for i in range(0, len(pcm_bytes), chunk_len):
            chunks.append(pcm_bytes[i:i + chunk_len])
        return chunks

    @staticmethod
    def calculate_pcm_rms(pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return 0.0
        valid_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if valid_len == 0:
            return 0.0
        s = np.frombuffer(pcm_bytes[:valid_len], dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(s ** 2)))

    @staticmethod
    def calculate_pcm_peak(pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return 0.0
        valid_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if valid_len == 0:
            return 0.0
        s = np.frombuffer(pcm_bytes[:valid_len], dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.max(np.abs(s)))

    @staticmethod
    def calculate_spectral_bands(pcm_bytes: bytes, sample_rate: int = 16000, num_bands: int = 8) -> List[float]:
        if not pcm_bytes:
            return [0.0] * num_bands
        valid_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        s = np.frombuffer(pcm_bytes[:valid_len], dtype=np.int16).astype(np.float32) / 32768.0
        if len(s) < 16:
            return [0.0] * num_bands
        fft = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), 1.0 / sample_rate)
        nyquist = sample_rate / 2.0
        band_edges = np.linspace(0, nyquist, num_bands + 1)
        bands = []
        for i in range(num_bands):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            energy = float(np.mean(fft[mask])) if np.any(mask) else 0.0
            bands.append(energy)
        max_e = max(bands) if max(bands) > 0 else 1.0
        return [round(float(b / max_e), 4) for b in bands]

    @staticmethod
    def get_peak_frequency(pcm_bytes: bytes, sample_rate: int) -> float:
        valid_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if valid_len < 16:
            return 0.0
        s = np.frombuffer(pcm_bytes[:valid_len], dtype=np.int16).astype(np.float32)
        fft = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), 1.0 / sample_rate)
        return float(freqs[np.argmax(fft)])

    @staticmethod
    def verify_signal_correlation(
        sig1_pcm: bytes,
        sig2_pcm: bytes,
        in_rate: int,
        out_rate: int,
        min_corr: float = 0.98
    ) -> float:
        s1 = np.frombuffer(sig1_pcm[:len(sig1_pcm) - (len(sig1_pcm) % 2)], dtype=np.int16).astype(np.float32)
        s2 = np.frombuffer(sig2_pcm[:len(sig2_pcm) - (len(sig2_pcm) % 2)], dtype=np.int16).astype(np.float32)

        if in_rate != out_rate:
            s2_aligned = AudioProcessor.pcm_to_numpy(
                AudioProcessor.resample_pcm(sig2_pcm, out_rate, in_rate)
            )
        else:
            s2_aligned = s2

        min_len = min(len(s1), len(s2_aligned))
        if min_len < 2:
            return 1.0
        corr = float(np.corrcoef(s1[:min_len], s2_aligned[:min_len])[0, 1])
        assert corr >= min_corr, f"Correlation {corr:.4f} fell below minimum required {min_corr}"
        return corr


@pytest.fixture
def audio_synth():
    return AudioSynthesizer()


# ============================================================================
# 3. TELEPHONY DRIVER INTERFACE & MOCK IMPLEMENTATIONS
# ============================================================================

class BaseTelephonyDriver(ABC):
    """Abstract interface defining the telephony driver contract (PROJECT.md §3)."""

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def dial(self, phone_number: str) -> bool:
        pass

    @abstractmethod
    async def answer_call(self) -> bool:
        pass

    @abstractmethod
    async def hang_up(self, reason: str = "normal") -> bool:
        pass

    @abstractmethod
    def set_audio_callbacks(self, on_audio_in: Callable[[bytes], None]) -> None:
        pass

    @abstractmethod
    def play_audio_chunk(self, pcm_data: bytes) -> None:
        pass

    @abstractmethod
    def clear_audio_buffer(self) -> None:
        pass


class MockTelephonyDriver(BaseTelephonyDriver):
    """Full in-memory implementation of BaseTelephonyDriver for testing."""

    def __init__(self):
        self.is_running: bool = False
        self.active_call: bool = False
        self.dialed_numbers: List[str] = []
        self.hangup_reasons: List[str] = []
        self.audio_callback: Optional[Callable[[bytes], None]] = None
        self.played_audio_chunks: List[bytes] = []
        self.buffer_cleared_count: int = 0
        self.is_ringing: bool = False
        self.is_on_hold: bool = False
        self.is_playing: bool = False

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False
        self.active_call = False
        self.is_ringing = False
        self.is_on_hold = False

    async def dial(self, phone_number: str) -> bool:
        self.dialed_numbers.append(phone_number)
        self.active_call = True
        return True

    async def answer_call(self) -> bool:
        self.active_call = True
        self.is_ringing = False
        return True

    async def hang_up(self, reason: str = "normal") -> bool:
        self.hangup_reasons.append(reason)
        self.active_call = False
        self.is_ringing = False
        self.is_on_hold = False
        self.clear_audio_buffer()
        return True

    def set_audio_callbacks(self, on_audio_in: Callable[[bytes], None]) -> None:
        self.audio_callback = on_audio_in

    def play_audio_chunk(self, pcm_data: bytes) -> None:
        if pcm_data:
            self.played_audio_chunks.append(pcm_data)
            self.is_playing = True

    def clear_audio_buffer(self) -> None:
        self.played_audio_chunks.clear()
        self.buffer_cleared_count += 1
        self.is_playing = False

    def play_ring_chime(self) -> None:
        self.is_ringing = True

    def stop_ring_chime(self) -> None:
        self.is_ringing = False

    def play_hold_audio(self) -> None:
        self.is_on_hold = True

    def stop_hold_audio(self) -> None:
        self.is_on_hold = False

    def get_buffer_size(self) -> int:
        return sum(len(c) for c in self.played_audio_chunks)

    def queue_audio_chunks(self, chunks: List[bytes]) -> None:
        for c in chunks:
            self.play_audio_chunk(c)

    @property
    def played_chunks_count(self) -> int:
        return len(self.played_audio_chunks)


class VirtualSimulatorDriver(BaseTelephonyDriver):
    """Virtual audio simulator driver routing mic/speaker loopback in memory."""

    def __init__(self):
        self.is_running: bool = False
        self.active_call: bool = False
        self.audio_in_callback = None
        self.playback_queue = []

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False
        self.active_call = False

    async def dial(self, phone_number: str) -> bool:
        self.active_call = True
        return True

    async def answer_call(self) -> bool:
        self.active_call = True
        return True

    async def hang_up(self, reason: str = "normal") -> bool:
        self.active_call = False
        self.clear_audio_buffer()
        return True

    def set_audio_callbacks(self, on_audio_in: Callable[[bytes], None]) -> None:
        self.audio_in_callback = on_audio_in

    def play_audio_chunk(self, pcm_data: bytes) -> None:
        if pcm_data:
            self.playback_queue.append(pcm_data)

    def clear_audio_buffer(self) -> None:
        self.playback_queue.clear()


class WindowsAudioDriver(BaseTelephonyDriver):
    """Windows audio driver stub handling device sanitization and tel: protocol."""

    def __init__(self, device_name: str = "Default Communication Device"):
        super().__init__()
        self.device_name = device_name
        self.is_running = False
        self.active_call = False
        self.audio_callback = None
        self.playback_buffer = []
        self.last_dialed_uri: Optional[str] = None
        self.last_dialed_number: Optional[str] = None

    @staticmethod
    def sanitize_device_name(name: str) -> str:
        """Strips control chars, portaudio garbage, and \\r\\n from device names."""
        if not name:
            return ""
        clean = name.replace("\r", " ").replace("\n", " ").strip()
        while "  " in clean:
            clean = clean.replace("  ", " ")
        if "(" in clean and ")" in clean:
            # e.g. Output (@System32...;%1 Hands-Free HF Audio%0;(iQOO Z9 5G)) -> iQOO Z9 5G Hands-Free HF Audio
            parts = clean.split(";")
            for part in parts:
                if "Hands-Free" in part or "Audio" in part:
                    part_clean = part.replace("%1", "").replace("%0", "").strip()
                    for p in parts:
                        if "(" in p and ")" in p:
                            sub = p[p.find("(") + 1:p.rfind(")")].strip()
                            if sub and not sub.startswith("@"):
                                return f"{sub} {part_clean}".strip()
        return clean

    _sanitize_device_name = sanitize_device_name

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False

    async def dial(self, phone_number: str) -> bool:
        # tel: protocol dispatch simulation
        if not phone_number or not isinstance(phone_number, str):
            return False
        cleaned = phone_number.strip()
        has_plus = cleaned.startswith("+")
        digits = "".join(c for c in cleaned if c.isdigit())
        if not digits:
            return False
        clean_number = f"+{digits}" if has_plus else digits
        self.last_dialed_number = clean_number
        self.last_dialed_uri = f"tel:{clean_number}"
        self.active_call = True
        self._call_state = "ACTIVE"
        return True

    async def answer_call(self) -> bool:
        self.active_call = True
        return True

    async def hang_up(self, reason: str = "normal") -> bool:
        self.active_call = False
        self.clear_audio_buffer()
        return True

    def set_audio_callbacks(self, on_audio_in: Callable[[bytes], None]) -> None:
        self.audio_callback = on_audio_in

    def play_audio_chunk(self, pcm_data: bytes) -> None:
        if pcm_data:
            self.playback_buffer.append(pcm_data)

    def clear_audio_buffer(self) -> None:
        self.playback_buffer.clear()


class LinuxDriver(BaseTelephonyDriver):
    """Linux driver with guarded imports and safe mock fallback."""

    def __init__(self):
        self.available = sys.platform.startswith("linux")
        self.active_call = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def dial(self, phone_number: str) -> bool:
        self.active_call = True
        return True

    async def answer_call(self) -> bool:
        self.active_call = True
        return True

    async def hang_up(self, reason: str = "normal") -> bool:
        self.active_call = False
        return True

    def set_audio_callbacks(self, on_audio_in: Callable[[bytes], None]) -> None:
        pass

    def play_audio_chunk(self, pcm_data: bytes) -> None:
        pass

    def clear_audio_buffer(self) -> None:
        pass


@pytest.fixture
def mock_telephony():
    return MockTelephonyDriver()


# ============================================================================
# 4. SQLITE WAL DATABASE LAYER & MANAGER
# ============================================================================

DB_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    direction TEXT NOT NULL,
    caller_id TEXT NOT NULL,
    caller_name TEXT DEFAULT '',
    status TEXT NOT NULL,
    goal_prompt TEXT DEFAULT '',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    duration REAL DEFAULT 0.0,
    hangup_reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    is_final INTEGER DEFAULT 1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(call_id) REFERENCES calls(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT,
    caller_name TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    message TEXT NOT NULL,
    urgency TEXT DEFAULT 'normal',
    is_read INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(call_id) REFERENCES calls(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT UNIQUE NOT NULL,
    summary TEXT NOT NULL,
    action_items TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(call_id) REFERENCES calls(id) ON DELETE CASCADE
);
"""


class DatabaseManager:
    """Async SQLite persistence layer with WAL mode support."""

    def __init__(self, db_path: str = "voice_caller.db"):
        self.db_path = db_path
        self._lock = asyncio.Lock()

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(DB_SCHEMA)
            await db.commit()

    async def create_call(self, call_data: dict) -> str:
        call_id = call_data.get("id") or f"call_{int(time.time() * 1000)}"
        direction = call_data.get("direction", "inbound")
        caller_id = call_data.get("caller_id", "")
        caller_name = call_data.get("caller_name", "")
        status = call_data.get("status", "ringing")
        goal_prompt = call_data.get("goal_prompt", "")

        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO calls (id, direction, caller_id, caller_name, status, goal_prompt)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (call_id, direction, caller_id, caller_name, status, goal_prompt)
                )
                await db.commit()
        return call_id

    async def update_call_status(self, call_id: str, status: str, **kwargs) -> None:
        hangup_reason = kwargs.get("hangup_reason", "")
        duration = kwargs.get("duration", 0.0)

        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                if status in ["completed", "failed"]:
                    await db.execute(
                        """
                        UPDATE calls
                        SET status = ?, ended_at = CURRENT_TIMESTAMP, hangup_reason = ?, duration = ?
                        WHERE id = ?
                        """,
                        (status, hangup_reason, duration, call_id)
                    )
                else:
                    await db.execute(
                        "UPDATE calls SET status = ? WHERE id = ?",
                        (status, call_id)
                    )
                await db.commit()

    async def append_transcript(self, call_id: str, speaker: str, text: str, is_final: bool = True) -> int:
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    INSERT INTO transcripts (call_id, speaker, text, is_final)
                    VALUES (?, ?, ?, ?)
                    """,
                    (call_id, speaker, text, 1 if is_final else 0)
                )
                await db.commit()
                return cursor.lastrowid

    async def save_message(
        self,
        call_id: Optional[str],
        caller_name: str,
        phone_number: str,
        message: str,
        urgency: str = "normal"
    ) -> int:
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    INSERT INTO messages (call_id, caller_name, phone_number, message, urgency)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (call_id, caller_name, phone_number, message, urgency)
                )
                await db.commit()
                return cursor.lastrowid

    async def save_summary(
        self,
        call_id: str,
        summary: str,
        action_items: Optional[Union[List[str], str]] = None
    ) -> int:
        if isinstance(action_items, list):
            action_items_str = json.dumps(action_items)
        elif isinstance(action_items, str):
            action_items_str = json.dumps([action_items])
        else:
            action_items_str = "[]"

        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    INSERT OR REPLACE INTO summaries (call_id, summary, action_items)
                    VALUES (?, ?, ?)
                    """,
                    (call_id, summary, action_items_str)
                )
                await db.commit()
                return cursor.lastrowid

    async def get_recent_calls(self, limit: int = 50, offset: int = 0) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM calls ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_call_details(self, call_id: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM calls WHERE id = ?", (call_id,)) as cur:
                call_row = await cur.fetchone()
                if not call_row:
                    return None
                call = dict(call_row)

            # Fetch transcripts
            async with db.execute(
                "SELECT speaker, text, is_final, timestamp FROM transcripts WHERE call_id = ? ORDER BY id ASC",
                (call_id,)
            ) as cur:
                trans_rows = await cur.fetchall()
                call["transcripts"] = [dict(r) for r in trans_rows]

            # Fetch summary
            async with db.execute(
                "SELECT summary, action_items FROM summaries WHERE call_id = ?",
                (call_id,)
            ) as cur:
                summary_row = await cur.fetchone()
                if summary_row:
                    s_dict = dict(summary_row)
                    try:
                        s_dict["action_items"] = json.loads(s_dict["action_items"])
                    except Exception:
                        s_dict["action_items"] = []
                    call["summary"] = s_dict
                else:
                    call["summary"] = None

            return call

    async def get_messages(self, unread_only: bool = False) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT * FROM messages"
            if unread_only:
                query += " WHERE is_read = 0"
            query += " ORDER BY created_at DESC"
            cursor = await db.execute(query)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_transcripts(self, call_id: str) -> List[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM transcripts WHERE call_id = ? ORDER BY id ASC",
                (call_id,)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


@pytest_asyncio.fixture
async def temp_db():
    """Initializes an isolated temporary SQLite database in WAL mode."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    manager = DatabaseManager(path)
    await manager.init_db()

    yield manager

    # Teardown with clean file deletion
    try:
        if os.path.exists(path):
            os.remove(path)
        for ext in ("-wal", "-shm"):
            if os.path.exists(path + ext):
                os.remove(path + ext)
    except OSError:
        pass


# ============================================================================
# 5. CALL STATE MANAGER & CONTROLLER
# ============================================================================

class CallStateManager:
    """Coordinates active call sessions, state machine, audio routing, and tools."""

    def __init__(
        self,
        db: DatabaseManager,
        driver: BaseTelephonyDriver,
        gemini_client: Optional[GeminiLiveClient] = None
    ):
        self.db = db
        self.driver = driver
        self.gemini_client = gemini_client or GeminiLiveClient()
        self.state: str = "IDLE"
        self.current_call_id: Optional[str] = None
        self.pending_whispers: List[str] = []
        self.sent_whispers: List[str] = []
        self.start_time: float = 0.0

        # Wire up gemini callbacks
        self.gemini_client.set_on_audio_received(self._on_assistant_audio)
        self.gemini_client.set_on_transcript(self._on_transcript)
        self.gemini_client.set_on_interrupted(self.handle_barge_in)

    def _on_assistant_audio(self, pcm_24k: bytes):
        """Routes 24kHz assistant audio to telephony driver."""
        self.driver.play_audio_chunk(pcm_24k)

    async def _on_transcript(self, speaker: str, text: str, is_final: bool):
        """Persists streaming and finalized transcripts."""
        if self.current_call_id and text:
            await self.db.append_transcript(self.current_call_id, speaker, text, is_final)

    async def handle_inbound_ring(self, caller_id: str, caller_name: str = "") -> str:
        """Handles inbound phone call arrival."""
        if self.state not in ["IDLE", "COMPLETED", "FAILED"]:
            raise RuntimeError(f"Cannot accept inbound call in state: {self.state}")

        self.current_call_id = f"call_{int(time.time() * 1000)}"
        self.start_time = time.time()
        self.state = "RINGING"
        if hasattr(self.driver, "play_ring_chime"):
            self.driver.play_ring_chime()

        await self.db.create_call({
            "id": self.current_call_id,
            "direction": "inbound",
            "caller_id": caller_id,
            "caller_name": caller_name,
            "status": "ringing"
        })
        return self.current_call_id

    async def approve_inbound_call(self, action: str) -> None:
        """Approves inbound ringing call ('ai_takeover' | 'answer_personally' | 'decline')."""
        if self.state != "RINGING":
            raise RuntimeError(f"Cannot approve call in state: {self.state}")

        if hasattr(self.driver, "stop_ring_chime"):
            self.driver.stop_ring_chime()

        if action == "ai_takeover":
            await self.driver.answer_call()
            self.state = "ACTIVE"
            await self.db.update_call_status(self.current_call_id, "active")

            # Connect Gemini Live
            if not self.gemini_client.is_connected:
                await self.gemini_client.connect()

            # Wire up telephony audio input to Gemini Live
            self.driver.set_audio_callbacks(self.feed_caller_audio)

            # Deliver any pre-buffered whispers
            while self.pending_whispers:
                w = self.pending_whispers.pop(0)
                await self.gemini_client.send_whisper(w)
                self.sent_whispers.append(w)

        elif action == "answer_personally":
            await self.driver.answer_call()
            self.state = "OPERATOR_TALKING"
            await self.db.update_call_status(self.current_call_id, "operator_talking")

        elif action == "decline":
            await self.driver.hang_up(reason="declined")
            self.state = "COMPLETED"
            await self.db.update_call_status(self.current_call_id, "completed", hangup_reason="declined")
            self.pending_whispers.clear()

    async def initiate_outbound(self, phone_number: str, goal_prompt: str) -> bool:
        """Initiates an outbound call with goal prompt."""
        if self.state not in ["IDLE", "COMPLETED", "FAILED"]:
            return False

        self.current_call_id = f"call_{int(time.time() * 1000)}"
        self.start_time = time.time()
        self.state = "DIALING"

        await self.db.create_call({
            "id": self.current_call_id,
            "direction": "outbound",
            "caller_id": phone_number,
            "status": "dialing",
            "goal_prompt": goal_prompt
        })

        dial_success = await self.driver.dial(phone_number)
        if not dial_success:
            self.state = "FAILED"
            await self.db.update_call_status(self.current_call_id, "failed", hangup_reason="dial_failed")
            return False

        self.state = "ACTIVE"
        await self.db.update_call_status(self.current_call_id, "active")

        # Configure custom goal prompt in gemini client if provided
        if goal_prompt:
            self.gemini_client.system_instruction = (
                f"{DEFAULT_SYSTEM_INSTRUCTION} OUTBOUND GOAL: {goal_prompt}"
            )

        if not self.gemini_client.is_connected:
            await self.gemini_client.connect()

        self.driver.set_audio_callbacks(self.feed_caller_audio)
        return True

    def feed_caller_audio(self, pcm_bytes: bytes) -> None:
        """Feeds audio from caller into Gemini Live."""
        if self.gemini_client.is_connected and pcm_bytes:
            asyncio.create_task(self.gemini_client.send_audio_chunk(pcm_bytes))

    async def inject_whisper(self, instruction: str) -> None:
        """Injects operator whisper into active session or buffers if ringing."""
        if self.state == "RINGING":
            self.pending_whispers.append(instruction)
        elif self.state in ["ACTIVE", "ON_HOLD"]:
            await self.gemini_client.send_whisper(instruction)
            self.sent_whispers.append(instruction)

    async def handle_barge_in(self) -> None:
        """Clears telephony audio buffer immediately on barge-in."""
        self.driver.clear_audio_buffer()

    async def handle_tool_call(self, name: str, args: dict) -> dict:
        """Executes tool invoked by Gemini Live."""
        if name == "take_message":
            caller_name = str(args.get("caller_name") or "Unknown")
            phone_number = str(args.get("phone_number") or "")
            message = str(args.get("message") or "")
            urgency = str(args.get("urgency") or "medium")
            msg_id = await self.db.save_message(
                call_id=self.current_call_id,
                caller_name=caller_name,
                phone_number=phone_number,
                message=message,
                urgency=urgency
            )
            return {"status": "saved", "message_id": msg_id, "caller_name": caller_name}

        elif name == "end_call":
            reason = args.get("reason", "normal")
            await self.terminate_call(reason)
            return {"status": "call_ended", "reason": reason}

        elif name == "transfer_to_human":
            reason = args.get("reason", "escalation")
            self.state = "ON_HOLD"
            if hasattr(self.driver, "play_hold_audio"):
                self.driver.play_hold_audio()
            await self.db.update_call_status(self.current_call_id, "on_hold")
            return {"status": "transfer_initiated", "reason": reason, "on_hold": True}

        elif name == "log_call_summary":
            summary = args.get("summary", "")
            action_items = args.get("action_items", [])
            await self.db.save_summary(self.current_call_id, summary, action_items)
            return {"status": "summary_logged", "summary": summary}

        return {"error": f"Unknown tool: {name}"}

    async def safe_handle_tool_call(self, name: str, args: dict) -> dict:
        """Defensive wrapper catching any tool execution exception."""
        try:
            return await self.handle_tool_call(name, args)
        except Exception as e:
            logger.error(f"Error handling tool call {name}: {e}")
            return {"error": str(e)}

    async def hold_call(self) -> None:
        self.state = "ON_HOLD"
        if hasattr(self.driver, "play_hold_audio"):
            self.driver.play_hold_audio()

    async def resume_call(self) -> None:
        self.state = "ACTIVE"
        if hasattr(self.driver, "stop_hold_audio"):
            self.driver.stop_hold_audio()

    async def operator_pickup(self) -> None:
        self.state = "OPERATOR_TALKING"
        if hasattr(self.driver, "stop_hold_audio"):
            self.driver.stop_hold_audio()

    async def handle_remote_hangup(self) -> None:
        await self.terminate_call("remote_hung_up")

    async def handle_driver_error(self, exc: Exception) -> None:
        logger.error(f"Driver hardware error encountered: {exc}")
        await self.terminate_call("hardware_error")
        self.state = "FAILED"

    async def handle_socket_disconnect(self, code: int, reason: str) -> None:
        logger.warning(f"Socket disconnected code={code}, reason={reason}")
        await self.terminate_call("gemini_socket_error")

    async def terminate_call(self, reason: str = "normal") -> None:
        """Terminates call cleanly and flushes state with guaranteed hangup."""
        duration = time.time() - self.start_time if self.start_time > 0 else 0.0

        try:
            if self.gemini_client and self.gemini_client.is_connected:
                await self.gemini_client.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting Gemini client: {e}")
        finally:
            await self.driver.hang_up(reason=reason)

            if self.current_call_id:
                await self.db.update_call_status(
                    self.current_call_id,
                    "completed",
                    hangup_reason=reason,
                    duration=round(duration, 2)
                )

            self.state = "COMPLETED"
            self.pending_whispers.clear()


@pytest_asyncio.fixture
async def call_manager(temp_db, mock_telephony, mock_gemini):
    """Provides an initialized CallStateManager wired with mock fixtures."""
    client = GeminiLiveClient()
    manager = CallStateManager(db=temp_db, driver=mock_telephony, gemini_client=client)
    return manager


# ============================================================================
# 6. FASTAPI WEB CONSOLE SERVER & WEBSOCKET /WS/LIVE FIXTURES
# ============================================================================

_STATIC_DIR = os.path.join(tempfile.gettempdir(), "voice_caller_e2e_static")


def _ensure_static_assets() -> str:
    """Ensures static web console assets exist on disk for testing Feature 23."""
    ws_static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "static")
    if os.path.isdir(ws_static) and os.path.isfile(os.path.join(ws_static, "index.html")):
        return ws_static

    os.makedirs(_STATIC_DIR, exist_ok=True)
    index_path = os.path.join(_STATIC_DIR, "index.html")
    css_path = os.path.join(_STATIC_DIR, "styles.css")
    js_path = os.path.join(_STATIC_DIR, "app.js")

    if not os.path.isfile(index_path):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Calling Assistant — Console</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body class="dark-mode">
    <header class="app-header">
        <h1>AI Voice Calling Assistant</h1>
        <div id="call-status-badge" class="badge">Idle</div>
    </header>
    <main class="console-grid">
        <section class="visualizer-panel card">
            <h2>Audio Waveforms</h2>
            <div class="waveform-container">
                <div class="waveform-box">
                    <label>Caller (16kHz)</label>
                    <canvas id="caller-waveform" class="waveform cyan" width="400" height="120"></canvas>
                </div>
                <div class="waveform-box">
                    <label>Gemini Live (24kHz)</label>
                    <canvas id="gemini-waveform" class="waveform emerald" width="400" height="120"></canvas>
                </div>
            </div>
        </section>
        <section class="transcripts-panel card">
            <h2>Live Transcripts</h2>
            <div id="transcript-stream" class="transcript-box"></div>
        </section>
        <section class="whisper-panel card">
            <h2>Mid-Call Operator Whisper</h2>
            <div class="whisper-controls">
                <input type="text" id="whisper-input" name="whisper" placeholder="Type silent operator guidance..." />
                <button id="whisper-send-btn" class="btn primary">Send Whisper</button>
            </div>
            <div class="whisper-chips">
                <button class="chip" data-whisper="Wrap up the call politely">Wrap Up</button>
                <button class="chip" data-whisper="Ask for their callback number">Get Number</button>
            </div>
        </section>
        <section class="dialer-panel card">
            <h2>Outbound Dialer</h2>
            <input type="tel" id="dialer-phone" placeholder="+1 (555) 000-0000" />
            <textarea id="dialer-goal" placeholder="Outbound call objective..."></textarea>
            <div id="dialer-keypad" class="keypad-grid">
                <button class="keypad-btn" data-digit="1">1</button>
                <button class="keypad-btn" data-digit="2">2</button>
                <button class="keypad-btn" data-digit="3">3</button>
                <button class="keypad-btn" data-digit="4">4</button>
                <button class="keypad-btn" data-digit="5">5</button>
                <button class="keypad-btn" data-digit="6">6</button>
                <button class="keypad-btn" data-digit="7">7</button>
                <button class="keypad-btn" data-digit="8">8</button>
                <button class="keypad-btn" data-digit="9">9</button>
                <button class="keypad-btn" data-digit="*">*</button>
                <button class="keypad-btn" data-digit="0">0</button>
                <button class="keypad-btn" data-digit="#">#</button>
            </div>
            <button id="dial-btn" class="btn success">Call Now</button>
        </section>
    </main>
    <script src="/static/app.js"></script>
</body>
</html>""")

    if not os.path.isfile(css_path):
        with open(css_path, "w", encoding="utf-8") as f:
            f.write("""/* Modern dark-mode glassmorphic console styles */
:root {
    --bg-primary: #0f172a;
    --cyan-waveform: #06b6d4;
    --emerald-waveform: #10b981;
}
body.dark-mode {
    background-color: var(--bg-primary);
    color: #f8fafc;
    font-family: system-ui, sans-serif;
}
.waveform.cyan { border: 1px solid var(--cyan-waveform); }
.waveform.emerald { border: 1px solid var(--emerald-waveform); }
.keypad-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }
""")

    if not os.path.isfile(js_path):
        with open(js_path, "w", encoding="utf-8") as f:
            f.write("""// AI Voice Calling Assistant — Frontend Web Console Client
(function() {
    console.log("AI Voice Calling Assistant console loaded.");
    const wsUrl = (window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host + "/ws/live";
    let socket = null;
    function initWebSocket() {
        socket = new WebSocket(wsUrl);
    }
    window.addEventListener("DOMContentLoaded", initWebSocket);
})();
""")

    return _STATIC_DIR


def create_app(
    db: DatabaseManager,
    telephony_driver: BaseTelephonyDriver,
    call_manager: Optional[CallStateManager] = None
) -> FastAPI:
    """Creates a FastAPI test application adhering to PROJECT.md § Architecture."""
    app = FastAPI(title="AI Voice Calling Assistant API")
    mgr = call_manager or CallStateManager(db=db, driver=telephony_driver)
    active_websockets: List[WebSocket] = []

    static_dir = _ensure_static_assets()
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = os.path.join(static_dir, "index.html")
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), media_type="text/html")

    @app.get("/api/status")
    async def get_status():
        return {
            "status": "ok",
            "state": mgr.state.lower(),
            "current_call_id": mgr.current_call_id
        }

    @app.get("/api/devices")
    async def get_devices():
        # Returns sanitized device lists
        return {
            "inputs": [
                {"id": 0, "name": "Microphone (Realtek Audio)"},
                {"id": 1, "name": "iQOO Z9 5G Hands-Free HF Audio"}
            ],
            "outputs": [
                {"id": 2, "name": "Speakers (Realtek Audio)"},
                {"id": 3, "name": "iQOO Z9 5G Hands-Free HF Audio"}
            ]
        }

    @app.post("/api/call/dial")
    async def post_dial(payload: dict):
        phone_number = payload.get("phone_number", "")
        goal_prompt = payload.get("goal_prompt", "")
        if mgr.state in ["RINGING", "ACTIVE", "ON_HOLD", "DIALING"]:
            raise HTTPException(status_code=409, detail="Call already active")
        success = await mgr.initiate_outbound(phone_number, goal_prompt)
        if not success:
            raise HTTPException(status_code=500, detail="Dialing failed")
        return {"status": "dialing", "call_id": mgr.current_call_id}

    @app.post("/api/call/hangup")
    async def post_hangup(payload: Optional[dict] = None):
        reason = (payload or {}).get("reason", "operator_hangup")
        await mgr.terminate_call(reason)
        return {"status": "terminated", "reason": reason}

    @app.post("/api/call/inbound-action")
    async def post_inbound_action(payload: dict):
        action = payload.get("action", "")
        if action not in ["ai_takeover", "answer_personally", "decline"]:
            raise HTTPException(status_code=400, detail="Invalid action")
        await mgr.approve_inbound_call(action)
        return {"status": "approved", "action": action, "state": mgr.state}

    @app.post("/api/call/whisper")
    async def post_whisper(payload: dict):
        instruction = payload.get("instruction", "")
        if not instruction:
            raise HTTPException(status_code=400, detail="Empty instruction")
        await mgr.inject_whisper(instruction)
        return {"status": "injected", "instruction": instruction}

    @app.get("/api/history/calls")
    async def get_history(limit: int = 50, offset: int = 0):
        calls = await db.get_recent_calls(limit=limit, offset=offset)
        return {"total": len(calls), "calls": calls}

    @app.get("/api/messages")
    async def get_messages(unread_only: bool = False):
        msgs = await db.get_messages(unread_only=unread_only)
        return {"total": len(msgs), "messages": msgs}

    @app.websocket("/ws/live")
    async def websocket_live_hub(ws: WebSocket):
        await ws.accept()
        active_websockets.append(ws)
        try:
            # Send initial state snapshot
            await ws.send_json({
                "type": "initial_state",
                "state": mgr.state,
                "call_id": mgr.current_call_id
            })

            while True:
                ws_msg = await ws.receive()
                if ws_msg.get("type") == "websocket.disconnect":
                    break
                if "text" not in ws_msg:
                    continue

                msg = ws_msg["text"]
                try:
                    data = json.loads(msg)
                except Exception:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                if "type" not in data:
                    await ws.send_json({"type": "error", "message": "Missing 'type' field"})
                    continue

                event_type = data.get("type")
                if event_type == "whisper":
                    text = data.get("instruction", "")
                    await mgr.inject_whisper(text)
                elif event_type == "ping":
                    await ws.send_json({"type": "pong"})

        except WebSocketDisconnect:
            pass
        finally:
            if ws in active_websockets:
                active_websockets.remove(ws)

    app.state.call_manager = mgr
    return app


@pytest_asyncio.fixture
async def live_server(temp_db, mock_telephony, mock_gemini):
    """Launches FastAPI app on an ephemeral Uvicorn server for live HTTP and /ws/live tests."""
    app = create_app(db=temp_db, telephony_driver=mock_telephony)

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())

    while not server.started:
        await asyncio.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    base_http = f"http://127.0.0.1:{port}"
    base_ws = f"ws://127.0.0.1:{port}"

    class TestServerHarness:
        def __init__(self):
            self.http_url = base_http
            self.ws_url = base_ws
            self.port = port
            self.app = app
            self.call_manager = app.state.call_manager

        async def get_http_client(self) -> httpx.AsyncClient:
            return httpx.AsyncClient(base_url=self.http_url, timeout=httpx.Timeout(20.0))

        def connect_ws(self, path: str = "/ws/live"):
            return websockets.connect(f"{self.ws_url}{path}")

    harness = TestServerHarness()
    try:
        yield harness
    finally:
        server.should_exit = True
        await serve_task


@pytest_asyncio.fixture
async def async_http_client(live_server):
    """Provides an HTTPX AsyncClient targeting the live test server."""
    client = await live_server.get_http_client()
    async with client:
        yield client
