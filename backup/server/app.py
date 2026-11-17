"""
FastAPI Server Core, REST API Endpoints, and WebSocket /ws/live Hub.
Coordinates application lifecycle, device sanitization, audio telemetry broadcasting,
and browser console interaction.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.audio_processor import AudioProcessor
from telephony.base_driver import BaseTelephonyDriver
from telephony.virtual_driver import VirtualSimulatorDriver

try:
    from telephony.windows_driver import WindowsAudioDriver
except (ImportError, Exception):
    WindowsAudioDriver = None  # type: ignore

try:
    from telephony.linux_driver import LinuxDriver
except (ImportError, Exception):
    LinuxDriver = None  # type: ignore

from server.database import DatabaseManager
from server.call_state import CallStateManager, CallState

logger = logging.getLogger("server.app")


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------

class DialRequest(BaseModel):
    phone_number: str = Field(..., description="E.164 or dialable telephone number")
    goal_prompt: Optional[str] = Field("", description="Task or objective steering Gemini Live")


class AnswerRequest(BaseModel):
    action: Optional[str] = Field("ai_takeover", description="Action: 'ai_takeover' or 'answer_personally'")


class HangupRequest(BaseModel):
    reason: Optional[str] = Field("operator_hangup", description="Reason for call termination")


class InboundActionRequest(BaseModel):
    action: str = Field(..., description="Action: 'ai_takeover', 'answer_personally', or 'decline'")


class WhisperRequest(BaseModel):
    instruction: str = Field(..., min_length=1, description="Silent guidance to inject into Gemini Live session")


class TakeoverRequest(BaseModel):
    caller_id: Optional[str] = Field("Inbound Cellular Call", description="Caller phone number or identifier")
    caller_name: Optional[str] = Field("Phone Link / Cellular", description="Caller display name")


class SimulateInboundRequest(BaseModel):
    caller_name: Optional[str] = Field("Test Caller", description="Simulated caller name")
    phone_number: Optional[str] = Field("+15550100", description="Simulated caller phone number")


class UpdateMessageRequest(BaseModel):
    is_read: Optional[bool] = Field(None, description="Read state flag")
    action_taken: Optional[str] = Field(None, description="Operator action taken notes")


# ---------------------------------------------------------------------------
# Device Sanitization Helper
# ---------------------------------------------------------------------------

def sanitize_device_name(name: str) -> str:
    """
    Strips control characters, portaudio driver tokens, and embedded \\r\\n from device names.
    Transforms driver strings like:
    'Output (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\\r\\n;(iQOO Z9 5G))'
    into clean human-readable names:
    'iQOO Z9 5G Hands-Free HF Audio'
    """
    if not name:
        return ""

    clean = re.sub(r"[\r\n]+", " ", name).strip()
    clean = re.sub(r"\s+", " ", clean)

    if "@System32" in clean or "bthhfenum" in clean or "%1" in clean:
        model_match = re.search(r";\((.*?)\)\s*\)$", clean)
        profile_match = re.search(r"%1\s*(.*?)\s*%0", clean)
        if model_match:
            model = model_match.group(1).strip()
            profile = profile_match.group(1).strip() if profile_match else "Hands-Free"
            return f"{model} {profile}"
        all_parens = re.findall(r"\((.*?)\)", clean)
        if all_parens:
            candidates = [p for p in all_parens if not p.startswith("@")]
            if candidates:
                return candidates[-1]

    return clean


# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Thread-safe and asyncio-safe connection manager for WebSocket /ws/live clients."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)

    async def send_json(self, websocket: WebSocket, message: dict) -> None:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.debug(f"Failed to send JSON to websocket: {e}")
            await self.disconnect(websocket)

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            clients = list(self.active_connections)
        if not clients:
            return
        dead = []
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self.active_connections.discard(ws)

    async def close_all(self, code: int = 1000, reason: str = "Server shutdown") -> None:
        async with self._lock:
            clients = list(self.active_connections)
            self.active_connections.clear()
        for ws in clients:
            try:
                await ws.close(code=code, reason=reason)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Static Directory Resolver
# ---------------------------------------------------------------------------

def _resolve_static_dir(custom_dir: Optional[str] = None) -> str:
    """
    Resolves static assets directory gracefully without modifying server/static/.
    If server/static/index.html exists, returns server/static/.
    Otherwise, generates and uses a fallback directory in temp storage.
    """
    if custom_dir and os.path.isdir(custom_dir):
        return custom_dir

    ws_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.isdir(ws_static) and os.path.isfile(os.path.join(ws_static, "index.html")):
        return ws_static

    fallback_dir = os.path.join(tempfile.gettempdir(), "voice_caller_fallback_static")
    os.makedirs(fallback_dir, exist_ok=True)
    index_path = os.path.join(fallback_dir, "index.html")
    css_path = os.path.join(fallback_dir, "styles.css")
    js_path = os.path.join(fallback_dir, "app.js")

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

    return fallback_dir


# ---------------------------------------------------------------------------
# FastAPI Application Factory & Endpoints
# ---------------------------------------------------------------------------

def create_app(
    db: Optional[DatabaseManager] = None,
    telephony_driver: Optional[BaseTelephonyDriver] = None,
    call_manager: Optional[CallStateManager] = None,
    static_dir: Optional[str] = None,
) -> FastAPI:
    """
    Factory creating a FastAPI application configured with lifespan management,
    CORS, REST API endpoints, WebSocket /ws/live hub, and static UI assets.
    """
    resolved_db = db or DatabaseManager()
    if telephony_driver is not None:
        resolved_driver = telephony_driver
    elif sys.platform == "win32" and WindowsAudioDriver is not None:
        try:
            resolved_driver = WindowsAudioDriver()
        except Exception:
            resolved_driver = VirtualSimulatorDriver()
    elif sys.platform.startswith("linux") and LinuxDriver is not None:
        try:
            resolved_driver = LinuxDriver()
        except Exception:
            resolved_driver = VirtualSimulatorDriver()
    else:
        resolved_driver = VirtualSimulatorDriver()

    connection_manager = ConnectionManager()

    if call_manager is not None:
        resolved_call_manager = call_manager
        resolved_call_manager.broadcast_func = connection_manager.broadcast
    else:
        resolved_call_manager = CallStateManager(
            db=resolved_db,
            driver=resolved_driver,
            broadcast_func=connection_manager.broadcast,
        )

    resolved_static = _resolve_static_dir(static_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info("Initializing voice caller server resources...")
        await resolved_db.init_db()
        if hasattr(resolved_driver, "start"):
            await resolved_driver.start()
        yield
        # Teardown
        logger.info("Shutting down voice caller server resources...")
        try:
            await resolved_call_manager.terminate_call("server_shutdown")
        except Exception as e:
            logger.debug(f"Error terminating call during shutdown: {e}")
        try:
            if hasattr(resolved_driver, "stop"):
                await resolved_driver.stop()
        except Exception as e:
            logger.debug(f"Error stopping driver during shutdown: {e}")
        try:
            await resolved_db.close()
        except Exception as e:
            logger.debug(f"Error closing db during shutdown: {e}")
        try:
            await connection_manager.close_all()
        except Exception as e:
            logger.debug(f"Error closing websockets during shutdown: {e}")

    app = FastAPI(
        title="AI Voice Calling Assistant API",
        version="1.0.0",
        description="Local voice calling assistant managing Bluetooth telephony and Gemini Live 3.8",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State injection
    app.state.db = resolved_db
    app.state.driver = resolved_driver
    app.state.call_manager = resolved_call_manager
    app.state.connection_manager = connection_manager

    # Mount static assets
    app.mount("/static", StaticFiles(directory=resolved_static, html=True), name="static")

    # -----------------------------------------------------------------------
    # Root & Static Web Console Route
    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = os.path.join(resolved_static, "index.html")
        if os.path.isfile(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), media_type="text/html")
        return HTMLResponse(
            content="<html><body><h1>AI Voice Calling Assistant</h1><canvas id='caller-waveform' class='cyan'></canvas><canvas id='gemini-waveform' class='emerald'></canvas><input id='whisper-input' /></body></html>",
            media_type="text/html"
        )

    # -----------------------------------------------------------------------
    # REST API Endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/status")
    async def get_status():
        """Returns server operational telemetry and live call state."""
        state_str = resolved_call_manager.state.value.lower() if hasattr(resolved_call_manager.state, "value") else str(resolved_call_manager.state).lower()
        driver_name = getattr(resolved_call_manager.driver, "driver_name", "virtual")
        return {
            "status": "ok",
            "state": state_str,
            "raw_state": str(resolved_call_manager.state),
            "current_call_id": resolved_call_manager.current_call_id,
            "driver": driver_name,
            "active_call": resolved_call_manager.current_call_id,
            "gemini_connected": getattr(resolved_call_manager.gemini_client, "is_connected", False),
        }

    @app.get("/api/devices")
    async def get_devices():
        """Returns sanitized audio endpoint listings and recommended Bluetooth HFP indices."""
        inputs: List[Dict[str, Any]] = []
        outputs: List[Dict[str, Any]] = []
        rec_in: Optional[int] = None
        rec_out: Optional[int] = None

        try:
            import sounddevice as sd
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                sanitized_name = sanitize_device_name(d["name"])
                max_in = d.get("max_input_channels", 0)
                max_out = d.get("max_output_channels", 0)

                if max_in > 0:
                    inputs.append({"id": idx, "name": sanitized_name, "channels": max_in})
                    if any(k in sanitized_name.lower() for k in ["audiorelay", "virtual mic"]) and rec_in is None:
                        rec_in = idx
                    elif "iqoo" in sanitized_name.lower() and rec_in is None:
                        rec_in = idx

                if max_out > 0:
                    outputs.append({"id": idx, "name": sanitized_name, "channels": max_out})
                    if any(k in sanitized_name.lower() for k in ["audiorelay", "virtual speak"]) and rec_out is None:
                        rec_out = idx
                    elif "iqoo" in sanitized_name.lower() and rec_out is None:
                        rec_out = idx

        except Exception as e:
            logger.debug(f"Audio device query fallback: {e}")
            inputs = [
                {"id": 0, "name": "Microphone (Realtek Audio)", "channels": 2},
                {"id": 1, "name": "iQOO Z9 5G Hands-Free HF Audio", "channels": 1},
            ]
            outputs = [
                {"id": 2, "name": "Speakers (Realtek Audio)", "channels": 2},
                {"id": 3, "name": "iQOO Z9 5G Hands-Free HF Audio", "channels": 1},
            ]
            rec_in = 1
            rec_out = 3

        # Default fallback to system communication devices if iQOO not matched
        try:
            import sounddevice as sd
            defaults = sd.default.device
            if rec_in is None and isinstance(defaults, (list, tuple)) and len(defaults) > 0 and defaults[0] >= 0:
                rec_in = defaults[0]
            if rec_out is None and isinstance(defaults, (list, tuple)) and len(defaults) > 1 and defaults[1] >= 0:
                rec_out = defaults[1]
        except Exception:
            pass

        if rec_in is None and inputs:
            rec_in = inputs[0]["id"]
        if rec_out is None and outputs:
            rec_out = outputs[0]["id"]

        return {
            "inputs": inputs,
            "outputs": outputs,
            "recommended_in": rec_in,
            "recommended_out": rec_out,
        }

    @app.post("/api/call/dial")
    async def post_dial(request: DialRequest):
        """Initiates an outbound phone call with goal prompt."""
        if resolved_call_manager.state in (
            CallState.RINGING,
            CallState.ACTIVE,
            CallState.ON_HOLD,
            CallState.ON_HOLD_TRANSFER,
            CallState.DIALING,
            CallState.OPERATOR_TALKING,
        ):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Call already active")

        success = await resolved_call_manager.initiate_outbound(request.phone_number, request.goal_prompt or "")
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Dialing failed")

        return {"status": "dialing", "call_id": resolved_call_manager.current_call_id}

    @app.get("/health")
    async def get_health():
        """Health check endpoint for host orchestrator and external monitoring."""
        return {"status": "ok", "state": str(resolved_call_manager.state)}

    @app.post("/api/call/answer")
    async def post_answer(request: Optional[AnswerRequest] = None):
        """Answers an incoming ringing call."""
        if resolved_call_manager.state != CallState.RINGING:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No incoming call to answer")

        action = request.action if request else "ai_takeover"
        res = await resolved_call_manager.approve_inbound_call(action)
        return {"status": "answered", "action": action, "call_id": resolved_call_manager.current_call_id}

    @app.post("/api/call/takeover")
    async def post_takeover(request: Optional[TakeoverRequest] = None):
        """
        Immediate AI takeover from ANY state (IDLE, RINGING, or ACTIVE).
        Takes over an incoming or ongoing cellular phone call using Gemini Live.
        """
        caller_id = request.caller_id if request and request.caller_id else "Inbound Cellular Call"
        caller_name = request.caller_name if request and request.caller_name else "Phone Link / Cellular"

        if resolved_call_manager.state in (CallState.IDLE, CallState.COMPLETED, CallState.DECLINED):
            await resolved_call_manager.handle_inbound_ring(caller_id=caller_id, caller_name=caller_name)

        res = await resolved_call_manager.approve_inbound_call("ai_takeover")
        return {
            "status": "takeover_started",
            "action": "ai_takeover",
            "call_id": resolved_call_manager.current_call_id,
            "state": str(resolved_call_manager.state),
        }

    @app.post("/api/call/hangup")
    async def post_hangup(request: Optional[HangupRequest] = None):
        """Terminates the active call or clears ringing state."""
        reason = request.reason if request and request.reason else "operator_hangup"
        res = await resolved_call_manager.terminate_call(reason)
        return {"status": "terminated", "reason": reason, "call_id": res.get("call_id")}

    @app.post("/api/call/inbound-action")
    async def post_inbound_action(request: InboundActionRequest):
        """Processes 3-way inbound decision: 'ai_takeover', 'answer_personally', 'decline'."""
        action = str(request.action).lower()
        if action not in ("ai_takeover", "answer_personally", "decline"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: '{request.action}'. Expected 'ai_takeover', 'answer_personally', or 'decline'."
            )

        if resolved_call_manager.state != CallState.RINGING:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No incoming call to action")

        res = await resolved_call_manager.approve_inbound_call(action)
        return {"status": "approved", "action": action, "state": str(resolved_call_manager.state)}

    @app.post("/api/call/whisper")
    async def post_whisper(request: WhisperRequest):
        """Injects a silent operator instruction into the active Gemini Live session."""
        if not request.instruction or not request.instruction.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty instruction")

        res = await resolved_call_manager.inject_whisper(request.instruction)
        return {"status": "injected", "instruction": request.instruction, "result": res}

    @app.post("/api/call/simulate-inbound")
    async def post_simulate_inbound(request: Optional[SimulateInboundRequest] = None):
        """Triggers a simulated inbound call for automated testing."""
        name = request.caller_name if request else "Test Caller"
        phone = request.phone_number if request else "+15550100"
        call_id = await resolved_call_manager.handle_inbound_ring(caller_id=phone, caller_name=name)
        return {"status": "simulating", "call_id": call_id}

    @app.get("/api/history/calls")
    async def get_history_calls(limit: int = 50, offset: int = 0, status: Optional[str] = None):
        """Retrieves paginated call history with optional status filter."""
        calls = await resolved_db.get_recent_calls(limit=limit, offset=offset, status=status)
        return {"total": len(calls), "calls": calls}

    @app.get("/api/history/calls/{call_id}")
    async def get_call_detail(call_id: str):
        """Retrieves complete call details including transcripts, messages, and summary."""
        details = await resolved_db.get_call_details(call_id)
        if not details:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found")
        # Support both top-level and nested structure
        details["call"] = dict(details)
        return details

    @app.delete("/api/history/calls/{call_id}")
    async def delete_call(call_id: str):
        """Deletes a call and cascades removal of transcripts, messages, and summary."""
        deleted = await resolved_db.delete_call(call_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Call '{call_id}' not found")
        return {"status": "deleted", "call_id": call_id}

    @app.get("/api/history/messages")
    async def get_history_messages(unread_only: bool = False, urgency: Optional[str] = None):
        """Retrieves caller messages with unread and urgency filtering."""
        msgs = await resolved_db.get_messages(unread_only=unread_only, urgency=urgency)
        return {"total": len(msgs), "messages": msgs}

    @app.get("/api/messages")
    async def get_messages_alias(unread_only: bool = False):
        """Alias route for message history queries."""
        msgs = await resolved_db.get_messages(unread_only=unread_only)
        return {"total": len(msgs), "messages": msgs}

    @app.patch("/api/history/messages/{message_id}")
    async def patch_message(message_id: int, request: UpdateMessageRequest):
        """Updates read status or action taken note on a message."""
        updated = await resolved_db.update_message(
            message_id=message_id,
            is_read=request.is_read,
            action_taken=request.action_taken,
        )
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Message {message_id} not found")
        return {"status": "updated", "message_id": message_id}

    # -----------------------------------------------------------------------
    # WebSocket /ws/live Hub Endpoint
    # -----------------------------------------------------------------------

    @app.websocket("/ws/live")
    async def websocket_live_hub(websocket: WebSocket):
        """
        WebSocket /ws/live hub. Transmits initial state upon connection,
        receives client commands (ping, whisper, inbound_decision, hangup),
        and defensively handles malformed text or binary frames.
        """
        await connection_manager.connect(websocket)

        try:
            # Transmit initial synchronization snapshot
            current_state = str(resolved_call_manager.state)
            await websocket.send_json({
                "type": "initial_state",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": current_state,
                "call_id": resolved_call_manager.current_call_id,
                "data": {
                    "state": current_state,
                    "call_id": resolved_call_manager.current_call_id,
                }
            })

            while True:
                ws_msg = await websocket.receive()
                msg_type = ws_msg.get("type")

                if msg_type == "websocket.disconnect":
                    break

                # Ignore binary payload defensively without closing socket
                if "bytes" in ws_msg and ws_msg["bytes"] is not None:
                    continue

                if "text" not in ws_msg or not ws_msg["text"]:
                    continue

                raw_text = ws_msg["text"]

                # Parse JSON
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                if not isinstance(payload, dict) or "type" not in payload:
                    await websocket.send_json({"type": "error", "message": "Missing 'type' field"})
                    continue

                event_type = payload["type"]

                if event_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif event_type == "whisper":
                    instruction = payload.get("instruction", "")
                    await resolved_call_manager.inject_whisper(instruction)

                elif event_type == "inbound_decision":
                    decision = payload.get("action") or payload.get("decision", "")
                    try:
                        await resolved_call_manager.approve_inbound_call(decision)
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": str(e)})

                elif event_type == "hangup":
                    reason = payload.get("reason", "operator_hangup")
                    await resolved_call_manager.terminate_call(reason)

                else:
                    # Unrecognized event type: ignore safely without dropping socket
                    logger.debug(f"Unknown websocket event type: {event_type}")

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WebSocket client connection terminated: {e}")
        finally:
            await connection_manager.disconnect(websocket)

    return app


# Default ASGI application instance for uvicorn server.app:app
app = create_app()
