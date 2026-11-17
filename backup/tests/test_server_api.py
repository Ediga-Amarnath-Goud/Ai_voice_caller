"""
Comprehensive Unit & Integration Test Suite for FastAPI Server, REST APIs, and WebSocket Hub.
Covers server package exports, application lifecycle, PortAudio device sanitization,
REST call controls, database history endpoints, and WebSocket /ws/live protocol.
"""

import asyncio
import json
import os
import tempfile
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import server
from server import (
    CallState,
    CallStateManager,
    ConnectionManager,
    DatabaseManager,
    create_app,
    sanitize_device_name,
)
from telephony.virtual_driver import VirtualSimulatorDriver


@pytest_asyncio.fixture
async def temp_db_path() -> AsyncGenerator[str, None]:
    """Yields a temporary database file path and cleans up afterward."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for ext in ["", "-wal", "-shm"]:
        p = path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


@pytest_asyncio.fixture
async def app_fixture(temp_db_path: str):
    """Provides an initialized FastAPI app and TestClient targeting an isolated SQLite DB."""
    db = DatabaseManager(temp_db_path)
    await db.init_db()
    driver = VirtualSimulatorDriver(headless=True)
    await driver.start()

    # Fast mock Gemini client to avoid external network timeouts in unit tests
    mock_gemini = MagicMock()
    mock_gemini.is_connected = True
    mock_gemini.connect = AsyncMock()
    mock_gemini.disconnect = AsyncMock()
    mock_gemini.send_audio_chunk = AsyncMock()
    mock_gemini.send_whisper = AsyncMock()
    mock_gemini.set_on_audio_received = MagicMock()
    mock_gemini.set_on_transcript = MagicMock()
    mock_gemini.set_on_interrupted = MagicMock()
    mock_gemini.register_tool = MagicMock()

    manager = CallStateManager(db=db, driver=driver, gemini_client=mock_gemini)
    test_app = create_app(db=db, telephony_driver=driver, call_manager=manager)

    client = TestClient(test_app)
    yield test_app, client, db, driver

    # Teardown
    await test_app.state.call_manager.terminate_call("test_teardown")
    await driver.stop()
    await db.close()


# ============================================================================
# 1. Package Exports & Lifespan Integration
# ============================================================================

class TestServerExportsAndLifespan:
    """Verifies server package exports and application lifespan management."""

    def test_server_exports(self):
        """Verifies server package exports all expected symbols."""
        assert hasattr(server, "create_app")
        assert hasattr(server, "CallStateManager")
        assert hasattr(server, "CallState")
        assert hasattr(server, "DatabaseManager")
        assert hasattr(server, "sanitize_device_name")
        assert hasattr(server, "ConnectionManager")
        for sym in server.__all__:
            assert getattr(server, sym) is not None

    @pytest.mark.asyncio
    async def test_lifespan_startup_and_shutdown(self, temp_db_path: str):
        """Verifies lifespan initializes database and driver on startup, and cleans up on shutdown."""
        db = DatabaseManager(temp_db_path)
        driver = VirtualSimulatorDriver(headless=True)

        test_app = create_app(db=db, telephony_driver=driver)

        # Enter lifespan via TestClient context manager
        with TestClient(test_app) as client:
            res = client.get("/api/status")
            assert res.status_code == 200
            assert res.json()["status"] == "ok"
            assert res.json()["state"] == "idle"


# ============================================================================
# 2. Audio Device Query & Sanitization Tests
# ============================================================================

class TestAudioDeviceSanitization:
    """Verifies audio device enumeration, CRLF stripping, and Bluetooth recommendations."""

    def test_sanitize_device_name_clean_strings(self):
        """Verifies clean device names are untouched."""
        assert sanitize_device_name("Microphone (Realtek Audio)") == "Microphone (Realtek Audio)"
        assert sanitize_device_name("Speakers (Realtek Audio)") == "Speakers (Realtek Audio)"
        assert sanitize_device_name("") == ""
        assert sanitize_device_name(None) == ""

    def test_sanitize_device_name_windows_driver_tokens(self):
        """Verifies Windows driver artifacts, CRLF, and %1 tokens are cleanly stripped."""
        raw_output = "Output (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iQOO Z9 5G))"
        clean_output = sanitize_device_name(raw_output)
        assert "\r" not in clean_output
        assert "\n" not in clean_output
        assert "iQOO Z9 5G" in clean_output
        assert "Hands-Free" in clean_output

        raw_headset = "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0\r\n;(Airdopes 141 Pro Buds))"
        clean_headset = sanitize_device_name(raw_headset)
        assert "\r" not in clean_headset
        assert "\n" not in clean_headset
        assert "Airdopes 141 Pro Buds" in clean_headset

    def test_get_devices_structure(self, app_fixture):
        """GET /api/devices returns inputs and outputs lists with integer recommendations."""
        _, client, _, _ = app_fixture
        res = client.get("/api/devices")
        assert res.status_code == 200
        data = res.json()
        assert "inputs" in data
        assert "outputs" in data
        assert isinstance(data["inputs"], list)
        assert isinstance(data["outputs"], list)
        assert len(data["inputs"]) > 0
        assert len(data["outputs"]) > 0
        assert "recommended_in" in data
        assert "recommended_out" in data

    def test_get_devices_sanitizes_crlf_mocked(self, app_fixture):
        """Verifies GET /api/devices sanitizes mocked PortAudio device listings."""
        _, client, _, _ = app_fixture
        mock_devices = [
            {"name": "Input (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iQOO Z9 5G))", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Output (@System32\\drivers\\bthhfenum.sys,#4;%1 Hands-Free HF Audio%0\r\n;(iQOO Z9 5G))", "max_input_channels": 0, "max_output_channels": 1},
        ]
        with patch("sounddevice.query_devices", return_value=mock_devices):
            res = client.get("/api/devices")
            assert res.status_code == 200
            data = res.json()
            assert len(data["inputs"]) == 1
            assert len(data["outputs"]) == 1
            assert "\r" not in data["inputs"][0]["name"]
            assert "\n" not in data["inputs"][0]["name"]
            assert data["recommended_in"] == 0
            assert data["recommended_out"] == 1


# ============================================================================
# 3. Call Control REST Endpoints Tests
# ============================================================================

class TestCallControlREST:
    """Verifies call lifecycle REST endpoints: dial, answer, hangup, whisper, inbound-action."""

    def test_get_status_idle(self, app_fixture):
        """GET /api/status returns initial IDLE status."""
        _, client, _, _ = app_fixture
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["state"] == "idle"
        assert data["current_call_id"] is None

    def test_dial_success(self, app_fixture):
        """POST /api/call/dial initiates outbound call."""
        _, client, _, _ = app_fixture
        res = client.post("/api/call/dial", json={"phone_number": "+15551234567", "goal_prompt": "Screen client inquiry"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "dialing"
        assert data["call_id"].startswith("call_")

        # Cleanup
        client.post("/api/call/hangup", json={"reason": "test_complete"})

    def test_dial_conflict_when_active(self, app_fixture):
        """POST /api/call/dial returns HTTP 409 Conflict when a call is active."""
        _, client, _, _ = app_fixture
        res1 = client.post("/api/call/dial", json={"phone_number": "+15551112222"})
        assert res1.status_code == 200

        res2 = client.post("/api/call/dial", json={"phone_number": "+15553334444"})
        assert res2.status_code == 409
        assert "Call already active" in res2.json()["detail"]

        client.post("/api/call/hangup", json={"reason": "test_complete"})

    def test_dial_validation_error(self, app_fixture):
        """POST /api/call/dial with missing required phone_number returns HTTP 422."""
        _, client, _, _ = app_fixture
        res = client.post("/api/call/dial", json={})
        assert res.status_code == 422

    def test_hangup_idle(self, app_fixture):
        """POST /api/call/hangup while idle returns terminated without error."""
        _, client, _, _ = app_fixture
        res = client.post("/api/call/hangup", json={"reason": "operator_hangup"})
        assert res.status_code == 200
        assert res.json()["status"] == "terminated"

    def test_hangup_active(self, app_fixture):
        """POST /api/call/hangup during active call cleanly terminates session."""
        _, client, _, _ = app_fixture
        dial_res = client.post("/api/call/dial", json={"phone_number": "+15559998888"})
        call_id = dial_res.json()["call_id"]

        hang_res = client.post("/api/call/hangup", json={"reason": "operator_completed"})
        assert hang_res.status_code == 200
        assert hang_res.json()["status"] == "terminated"

        status_res = client.get("/api/status")
        assert status_res.json()["state"] == "completed"

    def test_answer_ringing_call(self, app_fixture):
        """POST /api/call/answer answers a ringing call."""
        app, client, _, _ = app_fixture
        # Trigger simulated inbound
        sim_res = client.post("/api/call/simulate-inbound", json={"caller_name": "Alice", "phone_number": "+15557778888"})
        assert sim_res.status_code == 200
        assert app.state.call_manager.state == "RINGING"

        ans_res = client.post("/api/call/answer", json={"action": "ai_takeover"})
        assert ans_res.status_code == 200
        assert ans_res.json()["status"] == "answered"
        assert app.state.call_manager.state == "ACTIVE"

        client.post("/api/call/hangup", json={"reason": "test_complete"})

    def test_answer_when_not_ringing_returns_409(self, app_fixture):
        """POST /api/call/answer when not ringing returns HTTP 409 Conflict."""
        _, client, _, _ = app_fixture
        res = client.post("/api/call/answer", json={"action": "ai_takeover"})
        assert res.status_code == 409

    def test_inbound_action_ai_takeover(self, app_fixture):
        """POST /api/call/inbound-action with 'ai_takeover' transitions to ACTIVE."""
        app, client, _, _ = app_fixture
        client.post("/api/call/simulate-inbound", json={"caller_name": "Bob", "phone_number": "+15554443333"})
        assert app.state.call_manager.state == "RINGING"

        res = client.post("/api/call/inbound-action", json={"action": "ai_takeover"})
        assert res.status_code == 200
        assert res.json()["status"] == "approved"
        assert res.json()["state"] == "ACTIVE"
        assert app.state.call_manager.state == "ACTIVE"

        client.post("/api/call/hangup", json={})

    def test_inbound_action_answer_personally(self, app_fixture):
        """POST /api/call/inbound-action with 'answer_personally' transitions to OPERATOR_TALKING."""
        app, client, _, _ = app_fixture
        client.post("/api/call/simulate-inbound", json={"caller_name": "Charlie", "phone_number": "+15552221111"})
        assert app.state.call_manager.state == "RINGING"

        res = client.post("/api/call/inbound-action", json={"action": "answer_personally"})
        assert res.status_code == 200
        assert res.json()["status"] == "approved"
        assert app.state.call_manager.state == "OPERATOR_TALKING"

        client.post("/api/call/hangup", json={})

    def test_inbound_action_decline(self, app_fixture):
        """POST /api/call/inbound-action with 'decline' hangs up and completes call."""
        app, client, _, _ = app_fixture
        client.post("/api/call/simulate-inbound", json={"caller_name": "Spammer", "phone_number": "+15550000000"})
        assert app.state.call_manager.state == "RINGING"

        res = client.post("/api/call/inbound-action", json={"action": "decline"})
        assert res.status_code == 200
        assert res.json()["status"] == "approved"
        assert app.state.call_manager.state == "COMPLETED"

    def test_inbound_action_invalid_action(self, app_fixture):
        """POST /api/call/inbound-action with unrecognized action returns HTTP 400."""
        app, client, _, _ = app_fixture
        client.post("/api/call/simulate-inbound", json={"caller_name": "Dan", "phone_number": "+15553332222"})

        res = client.post("/api/call/inbound-action", json={"action": "unrecognized_action"})
        assert res.status_code == 400
        assert "Invalid action" in res.json()["detail"]

        client.post("/api/call/hangup", json={})

    def test_inbound_action_conflict_when_not_ringing(self, app_fixture):
        """POST /api/call/inbound-action when not ringing returns HTTP 409 Conflict."""
        _, client, _, _ = app_fixture
        res = client.post("/api/call/inbound-action", json={"action": "ai_takeover"})
        assert res.status_code == 409

    def test_whisper_success(self, app_fixture):
        """POST /api/call/whisper injects operator instruction."""
        _, client, _, _ = app_fixture
        client.post("/api/call/dial", json={"phone_number": "+15558881234"})

        res = client.post("/api/call/whisper", json={"instruction": "Wrap up call politely."})
        assert res.status_code == 200
        assert res.json()["status"] == "injected"
        assert res.json()["instruction"] == "Wrap up call politely."

        client.post("/api/call/hangup", json={})

    def test_whisper_empty_error(self, app_fixture):
        """POST /api/call/whisper with empty instruction returns HTTP 422 or 400."""
        _, client, _, _ = app_fixture
        res = client.post("/api/call/whisper", json={"instruction": ""})
        assert res.status_code in (400, 422)

    def test_simulate_inbound(self, app_fixture):
        """POST /api/call/simulate-inbound triggers simulated ringing state."""
        app, client, _, _ = app_fixture
        res = client.post("/api/call/simulate-inbound", json={"caller_name": "Test Inbound", "phone_number": "+15556667777"})
        assert res.status_code == 200
        assert res.json()["status"] == "simulating"
        assert app.state.call_manager.state == "RINGING"

        client.post("/api/call/hangup", json={})


# ============================================================================
# 4. Historical Database & Messaging Queries Tests
# ============================================================================

class TestHistoryAndMessagesREST:
    """Verifies call history, call details, delete cascading, and message query endpoints."""

    @pytest.mark.asyncio
    async def test_get_calls_history_pagination(self, app_fixture):
        """GET /api/history/calls returns paginated call list with total count."""
        _, client, db, _ = app_fixture
        c1 = await db.create_call({"caller_id": "1", "status": "completed"})
        c2 = await db.create_call({"caller_id": "2", "status": "completed"})

        res = client.get("/api/history/calls?limit=1&offset=0")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1
        assert len(data["calls"]) == 1

        all_res = client.get("/api/history/calls?limit=50&offset=0")
        assert all_res.json()["total"] >= 2

    @pytest.mark.asyncio
    async def test_get_calls_history_filter_status(self, app_fixture):
        """GET /api/history/calls?status=active filters by call status."""
        _, client, db, _ = app_fixture
        await db.create_call({"caller_id": "active_1", "status": "active"})
        await db.create_call({"caller_id": "completed_1", "status": "completed"})

        res = client.get("/api/history/calls?status=active")
        assert res.status_code == 200
        calls = res.json()["calls"]
        assert all(c["status"] == "active" for c in calls)

    @pytest.mark.asyncio
    async def test_get_call_details_found(self, app_fixture):
        """GET /api/history/calls/{id} returns full call details with transcripts and summary."""
        _, client, db, _ = app_fixture
        call_id = await db.create_call({"caller_id": "+15551239999", "status": "completed"})
        await db.append_transcript(call_id, "caller", "Hello world")
        await db.save_message(call_id, "Alice", "555-1234", "Call back please")
        await db.save_summary(call_id, "Demo call summary", ["Action 1"])

        res = client.get(f"/api/history/calls/{call_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == call_id
        assert len(data["transcripts"]) == 1
        assert len(data["messages"]) == 1
        assert data["summary"]["summary"] == "Demo call summary"

    def test_get_call_details_not_found(self, app_fixture):
        """GET /api/history/calls/{id} returns HTTP 404 for nonexistent call ID."""
        _, client, _, _ = app_fixture
        res = client.get("/api/history/calls/nonexistent_call_id")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_call_cascading(self, app_fixture):
        """DELETE /api/history/calls/{id} cascades deletion to transcripts, messages, summary."""
        _, client, db, _ = app_fixture
        call_id = await db.create_call({"caller_id": "+15558887777"})
        await db.append_transcript(call_id, "caller", "T1")
        await db.save_message(call_id, "Bob", "555-8887", "M1")
        await db.save_summary(call_id, "S1")

        del_res = client.delete(f"/api/history/calls/{call_id}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

        # Verify not found in subsequent GET
        get_res = client.get(f"/api/history/calls/{call_id}")
        assert get_res.status_code == 404

    def test_delete_call_not_found(self, app_fixture):
        """DELETE /api/history/calls/{id} returns HTTP 404 for nonexistent ID."""
        _, client, _, _ = app_fixture
        res = client.delete("/api/history/calls/nonexistent_id")
        assert res.status_code == 404

    @pytest.mark.asyncio
    async def test_get_messages_unread_filter(self, app_fixture):
        """GET /api/history/messages filters unread messages."""
        _, client, db, _ = app_fixture
        m1 = await db.save_message(None, "M1", "555-01", "Msg 1")
        m2 = await db.save_message(None, "M2", "555-02", "Msg 2")
        await db.update_message(m1, is_read=True)

        res = client.get("/api/history/messages?unread_only=true")
        assert res.status_code == 200
        data = res.json()
        assert any(m["id"] == m2 for m in data["messages"])
        assert not any(m["id"] == m1 for m in data["messages"])

    @pytest.mark.asyncio
    async def test_get_messages_alias_route(self, app_fixture):
        """GET /api/messages returns message listing for E2E test compatibility."""
        _, client, db, _ = app_fixture
        await db.save_message(None, "Alias Test", "555-9999", "Alias message body")
        res = client.get("/api/messages")
        assert res.status_code == 200
        assert res.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_patch_message_update(self, app_fixture):
        """PATCH /api/history/messages/{id} updates is_read and action_taken."""
        _, client, db, _ = app_fixture
        msg_id = await db.save_message(None, "Patch Caller", "555-1111", "Please review")

        res = client.patch(
            f"/api/history/messages/{msg_id}",
            json={"is_read": True, "action_taken": "Callback scheduled for 3 PM"}
        )
        assert res.status_code == 200
        assert res.json()["status"] == "updated"

        # Verify update
        msgs = await db.get_messages()
        target = [m for m in msgs if m["id"] == msg_id][0]
        assert target["is_read"] == 1
        assert target["action_taken"] == "Callback scheduled for 3 PM"

    def test_patch_message_not_found(self, app_fixture):
        """PATCH /api/history/messages/{id} returns HTTP 404 for nonexistent message ID."""
        _, client, _, _ = app_fixture
        res = client.patch("/api/history/messages/999999", json={"is_read": True})
        assert res.status_code == 404


# ============================================================================
# 5. WebSocket /ws/live Protocol Tests
# ============================================================================

class TestWebSocketLiveHub:
    """Verifies WebSocket /ws/live initial state, actions, errors, and broadcasting."""

    def test_ws_handshake_and_initial_state(self, app_fixture):
        """WebSocket handshake succeeds and receives initial_state frame immediately."""
        _, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws:
            data = ws.receive_json()
            assert data["type"] == "initial_state"
            assert "state" in data
            assert data["state"] == "IDLE"

    def test_ws_ping_pong(self, app_fixture):
        """WebSocket responds to ping with pong heartbeat."""
        _, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_whisper_client_action(self, app_fixture):
        """Sending whisper over WebSocket injects operator guidance."""
        app, client, _, _ = app_fixture
        client.post("/api/call/dial", json={"phone_number": "+15553337777"})

        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_json({"type": "whisper", "instruction": "Ask for their email address."})
            ack = ws.receive_json()
            assert ack["type"] == "operator_whisper_ack"
            assert ack["data"]["instruction"] == "Ask for their email address."
            assert "Ask for their email address." in app.state.call_manager.sent_whispers

        client.post("/api/call/hangup", json={})

    def test_ws_inbound_decision_action(self, app_fixture):
        """Sending inbound_decision over WebSocket approves ringing call."""
        app, client, _, _ = app_fixture
        client.post("/api/call/simulate-inbound", json={"caller_name": "WS Caller", "phone_number": "+15552220000"})
        assert app.state.call_manager.state == "RINGING"

        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_json({"type": "inbound_decision", "action": "ai_takeover"})
            event = ws.receive_json()
            assert event["type"] == "call_state_change"
            assert event["data"]["state"] == "active"

        assert app.state.call_manager.state == "ACTIVE"
        client.post("/api/call/hangup", json={})

    def test_ws_hangup_action(self, app_fixture):
        """Sending hangup over WebSocket terminates active call."""
        app, client, _, _ = app_fixture
        client.post("/api/call/dial", json={"phone_number": "+15551239876"})
        assert app.state.call_manager.state == "ACTIVE"

        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_json({"type": "hangup", "reason": "ws_operator_hangup"})
            event = ws.receive_json()
            assert event["type"] == "call_state_change"
            assert event["data"]["state"] == "completed"

        assert app.state.call_manager.state == "COMPLETED"

    def test_ws_malformed_json(self, app_fixture):
        """Sending unparseable text returns error response without closing socket."""
        _, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_text("{corrupt non-json string")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Invalid JSON" in err["message"]

            # Connection remains alive: verify with ping
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_missing_type(self, app_fixture):
        """Sending JSON missing 'type' field returns validation error."""
        _, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_json({"foo": "bar"})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Missing 'type' field" in err["message"]

            # Connection remains alive
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_ws_unexpected_binary_frame(self, app_fixture):
        """Sending binary frame is defensively ignored without crashing."""
        _, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_bytes(b"\x00\x01\x02\x03\x04")
            # Connection remains open
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    def test_ws_unknown_event_type(self, app_fixture):
        """Sending unknown event type does not crash server."""
        _, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state
            ws.send_json({"type": "unknown_custom_event_xyz"})
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

    @pytest.mark.asyncio
    async def test_ws_broadcast_call_state_change(self, app_fixture):
        """Broadcasting state change delivers message to connected client."""
        app, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # initial_state

            # Broadcast event from server
            await app.state.connection_manager.broadcast({
                "type": "call_state_change",
                "data": {"state": "active"}
            })

            event = ws.receive_json()
            assert event["type"] == "call_state_change"
            assert event["data"]["state"] == "active"

    @pytest.mark.asyncio
    async def test_ws_multi_client_fanout(self, app_fixture):
        """Broadcasting fanout delivers events to multiple concurrent WebSocket clients."""
        app, client, _, _ = app_fixture
        with client.websocket_connect("/ws/live") as ws1:
            ws1.receive_json()
            with client.websocket_connect("/ws/live") as ws2:
                ws2.receive_json()

                # Broadcast event to both
                await app.state.connection_manager.broadcast({
                    "type": "audio_telemetry",
                    "data": {"caller": {"rms": 0.42}}
                })

                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()
                assert msg1["type"] == "audio_telemetry"
                assert msg2["type"] == "audio_telemetry"
                assert msg1["data"]["caller"]["rms"] == 0.42
                assert msg2["data"]["caller"]["rms"] == 0.42
