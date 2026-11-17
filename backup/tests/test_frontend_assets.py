"""
tests/test_frontend_assets.py - Comprehensive Unit & Integration Tests for Web Console Frontend.
Verifies file existence on disk, FastAPI static serving, MIME types, required DOM elements,
glassmorphic CSS rules, ES6 client logic, and end-to-end REST/WebSocket interactions.
"""

from __future__ import annotations

import os
import re
import pytest
from fastapi.testclient import TestClient

from server import create_app
from server.database import DatabaseManager
from telephony.virtual_driver import VirtualSimulatorDriver


@pytest.fixture(scope="session")
def static_dir_path() -> str:
    """Returns absolute path to server/static directory."""
    ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(ws, "server", "static")


@pytest.fixture
def app_with_frontend(tmp_path):
    """Creates a configured FastAPI test app pointing to the canonical static directory."""
    db_file = str(tmp_path / "test_frontend.db")
    db = DatabaseManager(db_file)
    driver = VirtualSimulatorDriver(headless=True)
    app = create_app(db=db, telephony_driver=driver)
    client = TestClient(app)
    return app, client, db, driver


# ============================================================================
# 1. Disk Asset Existence & Integrity Tests
# ============================================================================

class TestStaticFilesOnDisk:
    """Verifies all static assets exist on disk with non-trivial genuine contents."""

    def test_static_directory_exists(self, static_dir_path: str):
        assert os.path.isdir(static_dir_path), f"Directory {static_dir_path} does not exist"

    def test_index_html_exists_on_disk(self, static_dir_path: str):
        index_file = os.path.join(static_dir_path, "index.html")
        assert os.path.isfile(index_file), "index.html missing from server/static/"
        content = open(index_file, "r", encoding="utf-8").read()
        assert len(content) > 500, "index.html content unexpectedly small"
        assert "<!DOCTYPE html>" in content or "<!doctype html>" in content.lower()

    def test_styles_css_exists_on_disk(self, static_dir_path: str):
        css_file = os.path.join(static_dir_path, "styles.css")
        assert os.path.isfile(css_file), "styles.css missing from server/static/"
        content = open(css_file, "r", encoding="utf-8").read()
        assert len(content) > 500, "styles.css content unexpectedly small"
        assert ":root" in content
        assert "#090d16" in content

    def test_app_js_exists_on_disk(self, static_dir_path: str):
        js_file = os.path.join(static_dir_path, "app.js")
        assert os.path.isfile(js_file), "app.js missing from server/static/"
        content = open(js_file, "r", encoding="utf-8").read()
        assert len(content) > 1000, "app.js content unexpectedly small"
        assert "/ws/live" in content


# ============================================================================
# 2. FastAPI HTTP Static Serving & MIME Type Tests
# ============================================================================

class TestFastAPIStaticServing:
    """Verifies that FastAPI properly serves static assets with 200 OK and correct MIME types."""

    def test_get_root_serves_index_html(self, app_with_frontend):
        _, client, _, _ = app_with_frontend
        res = client.get("/")
        assert res.status_code == 200
        ct = res.headers.get("content-type", "").lower()
        assert "text/html" in ct
        assert "AI Voice Calling Assistant" in res.text

    def test_get_static_index_html(self, app_with_frontend):
        _, client, _, _ = app_with_frontend
        res = client.get("/static/index.html")
        assert res.status_code == 200
        ct = res.headers.get("content-type", "").lower()
        assert "text/html" in ct
        assert "waveformCanvas" in res.text

    def test_get_static_styles_css(self, app_with_frontend):
        _, client, _, _ = app_with_frontend
        res = client.get("/static/styles.css")
        assert res.status_code == 200
        ct = res.headers.get("content-type", "").lower()
        assert "text/css" in ct
        assert "--bg-base" in res.text or "#090d16" in res.text

    def test_get_static_app_js(self, app_with_frontend):
        _, client, _, _ = app_with_frontend
        res = client.get("/static/app.js")
        assert res.status_code == 200
        ct = res.headers.get("content-type", "").lower()
        assert any(m in ct for m in ["javascript", "application/javascript", "text/javascript"])
        assert "initWebSocket" in res.text or "requestAnimationFrame" in res.text


# ============================================================================
# 3. DOM Structure & Required Element IDs Specification Tests
# ============================================================================

class TestIndexHtmlDOMSpecification:
    """Verifies that index.html contains all required DOM element IDs and structures."""

    @pytest.fixture(scope="class")
    def html_content(self, static_dir_path: str) -> str:
        index_file = os.path.join(static_dir_path, "index.html")
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()

    def test_topbar_elements(self, html_content: str):
        # Brand & Platform
        assert "AI Voice Calling Assistant" in html_content
        assert "Windows 11" in html_content

        # Connection Pill
        assert 'id="connectionPill"' in html_content

        # Driver Badge
        assert 'id="driverBadge"' in html_content

        # Audio Device Selectors & Refresh
        assert 'id="audioInputSelect"' in html_content
        assert 'id="audioOutputSelect"' in html_content
        assert 'id="refreshDevicesBtn"' in html_content

    def test_hero_call_monitor_elements(self, html_content: str):
        # Call State Pill
        assert 'id="callStatePill"' in html_content

        # Caller Profile Displays
        assert 'id="callerIdDisplay"' in html_content
        assert 'id="callerNameDisplay"' in html_content

        # Direction Badge
        assert 'id="directionBadge"' in html_content

        # Call Duration Stopwatch Timer
        assert 'id="callDurationTimer"' in html_content

        # Primary Call Control Action Buttons
        assert 'id="btnEndCall"' in html_content
        assert 'id="btnAnswerCall"' in html_content

    def test_dual_waveform_visualizer_elements(self, html_content: str):
        # Visualizer Container
        assert 'id="visualizerContainer"' in html_content

        # Canvas Waveform Element
        assert '<canvas id="waveformCanvas"' in html_content or 'id="waveformCanvas"' in html_content

        # Visual Channel Designations (Cyan & Emerald)
        assert "cyan" in html_content.lower()
        assert "emerald" in html_content.lower()

    def test_transcript_panel_elements(self, html_content: str):
        # Panel & Container
        assert 'id="transcriptPanel"' in html_content
        assert 'id="transcriptContainer"' in html_content

        # Auto-scroll Toggle
        assert 'id="autoscrollToggle"' in html_content

    def test_whisper_box_elements(self, html_content: str):
        # Docked Whisper Panel
        assert 'id="whisperPanel"' in html_content

        # Input & Send Trigger
        assert 'id="whisperInput"' in html_content
        assert 'id="sendWhisperBtn"' in html_content
        assert 'placeholder="Whisper silent guidance to Gemini..."' in html_content

        # Preset Chips Container & 5 Required Presets
        assert 'id="whisperPresets"' in html_content
        presets = [
            "Ask caller's name",
            "Take a message",
            "Owner in meeting",
            "Offer morning callback",
            "Wrap up call"
        ]
        for p in presets:
            assert p in html_content, f"Preset '{p}' missing from index.html"

        # Toast Container
        assert 'id="toastContainer"' in html_content

    def test_inbound_modal_elements(self, html_content: str):
        # Inbound Approval Modal
        assert 'id="inboundModal"' in html_content

        # 3 Action Buttons
        assert 'id="btnAiTakeover"' in html_content
        assert 'id="btnAnswerPersonally"' in html_content
        assert 'id="btnDeclineCall"' in html_content

    def test_outbound_dialer_elements(self, html_content: str):
        # Keypad Grid & Phone / Goal Inputs
        assert 'id="dialPhoneNumber"' in html_content
        assert 'id="dialGoalPrompt"' in html_content
        assert 'id="btnStartCall"' in html_content
        assert 'id="dialer-keypad"' in html_content or 'keypad-grid' in html_content

        # Keypad Digits
        for digit in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "*", "#"]:
            assert f'data-key="{digit}"' in html_content or f'data-digit="{digit}"' in html_content

    def test_history_and_message_center_elements(self, html_content: str):
        # History Table / List
        assert 'id="historyTable"' in html_content
        assert 'id="historyList"' in html_content

        # Message Center
        assert 'id="messageCenter"' in html_content


# ============================================================================
# 4. Stylesheet Design Tokens & CSS Rule Specification Tests
# ============================================================================

class TestStylesCssSpecification:
    """Verifies that styles.css implements the required glassmorphic dark theme tokens."""

    @pytest.fixture(scope="class")
    def css_content(self, static_dir_path: str) -> str:
        css_file = os.path.join(static_dir_path, "styles.css")
        with open(css_file, "r", encoding="utf-8") as f:
            return f.read()

    def test_color_tokens(self, css_content: str):
        # Base Dark Theme #090d16
        assert "#090d16" in css_content.lower()

        # Electric Cyan #06b6d4 & Emerald #10b981
        assert "#06b6d4" in css_content.lower()
        assert "#10b981" in css_content.lower()

        # Amber and Rose
        assert "#f59e0b" in css_content.lower()
        assert "#f43f5e" in css_content.lower()

    def test_glassmorphic_styling(self, css_content: str):
        assert "backdrop-filter" in css_content
        assert "blur" in css_content
        assert "rgba(18, 24, 38" in css_content

    def test_animations_and_keyframes(self, css_content: str):
        assert "@keyframes" in css_content
        assert "pulse-ring" in css_content
        assert "pulse" in css_content

    def test_responsive_layout_media_queries(self, css_content: str):
        assert "@media" in css_content
        assert "max-width" in css_content

    def test_dialog_modal_overscroll_containment(self, css_content: str):
        """Verifies modern web guidance scroll containment on native modal dialogs."""
        assert "dialog:modal" in css_content or "overscroll-behavior" in css_content
        assert "overscroll-behavior: contain" in css_content


# ============================================================================
# 5. Client JavaScript Logic & WebSocket Specification Tests
# ============================================================================

class TestAppJsSpecification:
    """Verifies that app.js implements the complete event handlers, visualizer, and REST actions."""

    @pytest.fixture(scope="class")
    def js_content(self, static_dir_path: str) -> str:
        js_file = os.path.join(static_dir_path, "app.js")
        with open(js_file, "r", encoding="utf-8") as f:
            return f.read()

    def test_websocket_event_handlers(self, js_content: str):
        events = [
            "initial_state",
            "call_state_change",
            "inbound_call_prompt",
            "transcript_chunk",
            "audio_telemetry",
            "tool_execution",
            "operator_whisper_ack",
            "call_summary_generated"
        ]
        for ev in events:
            assert f'"{ev}"' in js_content or f"'{ev}'" in js_content, f"Event '{ev}' missing in app.js"

    def test_60fps_canvas_rendering_loop(self, js_content: str):
        assert "requestAnimationFrame" in js_content
        assert "devicePixelRatio" in js_content
        assert "getContext" in js_content
        assert "#06b6d4" in js_content.lower()
        assert "#10b981" in js_content.lower()

    def test_operator_whisper_handling(self, js_content: str):
        assert '"whisper"' in js_content or "'whisper'" in js_content
        assert "/api/call/whisper" in js_content

    def test_inbound_decision_actions(self, js_content: str):
        assert "ai_takeover" in js_content
        assert "answer_personally" in js_content
        assert "decline" in js_content
        assert "/api/call/inbound-action" in js_content

    def test_outbound_dialing(self, js_content: str):
        assert "/api/call/dial" in js_content
        assert "phone_number" in js_content

    def test_rest_queries(self, js_content: str):
        assert "/api/status" in js_content
        assert "/api/devices" in js_content
        assert "/api/history/calls" in js_content
        assert "/api/history/messages" in js_content


# ============================================================================
# 6. End-to-End REST & WebSocket Frontend Integration Tests
# ============================================================================

class TestFrontendBackendIntegration:
    """Verifies that all API endpoints called by the frontend work flawlessly."""

    def test_status_endpoint_returns_json_for_ui(self, app_with_frontend):
        _, client, _, _ = app_with_frontend
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "state" in data
        assert "driver" in data

    def test_devices_endpoint_returns_sanitized_lists_for_ui(self, app_with_frontend):
        _, client, _, _ = app_with_frontend
        res = client.get("/api/devices")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data["inputs"], list)
        assert isinstance(data["outputs"], list)

    def test_history_calls_and_messages_endpoints(self, app_with_frontend):
        _, client, _, _ = app_with_frontend
        calls_res = client.get("/api/history/calls")
        assert calls_res.status_code == 200
        assert "calls" in calls_res.json()

        msgs_res = client.get("/api/history/messages")
        assert msgs_res.status_code == 200
        assert "messages" in msgs_res.json()

    def test_websocket_initial_state_and_whisper_flow(self, app_with_frontend):
        app, client, _, _ = app_with_frontend
        # Transition to RINGING so whisper is buffered into pending_whispers
        from server.call_state import CallState
        app.state.call_manager.state = CallState.RINGING

        with client.websocket_connect("/ws/live") as ws:
            init_frame = ws.receive_json()
            assert init_frame["type"] == "initial_state"
            assert "state" in init_frame

            # Send ping
            ws.send_json({"type": "ping", "action": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"

            # Send whisper
            ws.send_json({
                "type": "whisper",
                "instruction": "Steer call politely",
                "action": "whisper"
            })
            # Receive broadcast whisper ack to ensure server processed the whisper frame
            ack = ws.receive_json()
            assert ack["type"] == "operator_whisper_ack"
            assert ack["data"]["instruction"] == "Steer call politely"

            # Roundtrip ping/pong barrier
            ws.send_json({"type": "ping", "action": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong"

            # Verify whisper in pending buffer
            assert "Steer call politely" in app.state.call_manager.pending_whispers


# ============================================================================
# 7. Frontend Hardening & Security Remediation Tests
# ============================================================================

class TestFrontendHardeningAndRemediation:
    """Verifies all security, stability, and concurrency remediations in frontend assets."""

    @pytest.fixture(scope="class")
    def js_content(self, static_dir_path: str) -> str:
        js_file = os.path.join(static_dir_path, "app.js")
        with open(js_file, "r", encoding="utf-8") as f:
            return f.read()

    def test_hidpi_fractional_dpr_rounding(self, js_content: str):
        assert "Math.floor(displayWidth * dpr)" in js_content
        assert "Math.floor(displayHeight * dpr)" in js_content
        assert "targetWidth" in js_content
        assert "targetHeight" in js_content

    def test_canvas_telemetry_safe_float_sanitization(self, js_content: str):
        assert "function safeFloat" in js_content
        assert "Number.isFinite" in js_content
        assert "safeFloat(data.caller.rms" in js_content
        assert "safeFloat(data.gemini.rms" in js_content

    def test_html_escaping_xss_protection(self, js_content: str):
        assert "function escapeHtml" in js_content
        assert '.replace(/&/g, "&amp;")' in js_content or ".replace(/&/g, '&amp;')" in js_content
        assert '.replace(/</g, "&lt;")' in js_content or ".replace(/</g, '&lt;')" in js_content
        # Check that escapeHtml is utilized across dynamic HTML injection sites
        assert "escapeHtml(toolName)" in js_content
        assert "escapeHtml(summary)" in js_content
        assert "escapeHtml(c.id)" in js_content or "escapeHtml(c.caller_name" in js_content
        assert "escapeHtml(m.caller_name" in js_content or "escapeHtml(m.message" in js_content

    def test_whisper_dual_dispatch_fallback_guard(self, js_content: str):
        # Confirm fetch whisper is enclosed in an `if (!wsSent)` condition
        assert "if (!wsSent)" in js_content
        assert "!r.ok" in js_content or "!response.ok" in js_content

    def test_inbound_decision_double_click_guard(self, js_content: str):
        assert "isDeciding" in js_content
        assert "setInboundButtonsDisabled" in js_content
        assert "state.isDeciding = true" in js_content

    def test_outbound_dialer_button_guard(self, js_content: str):
        assert 'state.callState === "IDLE"' in js_content or "state.callState === 'IDLE'" in js_content
        assert "dom.btnStartCall.disabled = false" in js_content

    def test_inbound_call_prompt_active_guard(self, js_content: str):
        assert 'if (state.callState === "ACTIVE") return;' in js_content or "if (state.callState === 'ACTIVE') return;" in js_content

    def test_toast_stacking_limit(self, js_content: str):
        assert "toastContainer.children.length >=" in js_content
