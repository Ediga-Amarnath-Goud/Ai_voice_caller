/**
 * AI Voice Calling Assistant — Frontend Web Management Console
 * Pure Vanilla ES6 JavaScript (Zero Dependencies, Zero Build Pipeline)
 * Coordinates WebSocket /ws/live real-time streaming, 60 FPS Canvas Visualizer,
 * Live Transcripts, Operator Whisper, Outbound Dialer, and REST APIs.
 */

(function () {
    "use strict";

    // =========================================================================
    // Application State Store
    // =========================================================================
    const state = {
        ws: null,
        wsConnected: false,
        reconnectAttempts: 0,
        reconnectTimer: null,
        pingTimer: null,

        callState: "IDLE",
        currentCallId: null,
        callDirection: "IDLE",
        callStartTime: null,
        timerInterval: null,
        isDeciding: false,

        // Live Audio Telemetry for Visualizer (Interpolated at 60 FPS)
        telemetry: {
            caller: {
                rms: 0.0,
                peak: 0.0,
                spectrum: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                targetRms: 0.0,
                targetPeak: 0.0,
                targetSpectrum: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            gemini: {
                rms: 0.0,
                peak: 0.0,
                spectrum: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                targetRms: 0.0,
                targetPeak: 0.0,
                targetSpectrum: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },

        // Transcripts Turn Map
        turns: new Map(), // turn_id -> { speaker, text, element, isFinal }
        turnCount: 0,

        // Hardware Devices
        devices: {
            inputs: [],
            outputs: [],
            recommendedIn: null,
            recommendedOut: null,
        },
    };

    // =========================================================================
    // Sanitization & Security Helpers
    // =========================================================================
    function escapeHtml(str) {
        if (str === null || str === undefined) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function safeFloat(val, fallback = 0.0) {
        const n = Number(val);
        return Number.isFinite(n) ? n : fallback;
    }

    // =========================================================================
    // DOM Element References Cache
    // =========================================================================
    let dom = {};

    function initDomElements() {
        dom = {
            // Top Bar
            connectionPill: document.getElementById("connectionPill"),
            driverBadge: document.getElementById("driverBadge"),
            audioInputSelect: document.getElementById("audioInputSelect"),
            audioOutputSelect: document.getElementById("audioOutputSelect"),
            refreshDevicesBtn: document.getElementById("refreshDevicesBtn"),

            // Hero Monitor
            callStatePill: document.getElementById("callStatePill"),
            directionBadge: document.getElementById("directionBadge"),
            callDurationTimer: document.getElementById("callDurationTimer"),
            callerNameDisplay: document.getElementById("callerNameDisplay"),
            callerIdDisplay: document.getElementById("callerIdDisplay"),
            btnAnswerCall: document.getElementById("btnAnswerCall"),
            btnEndCall: document.getElementById("btnEndCall"),

            // Visualizer
            visualizerContainer: document.getElementById("visualizerContainer"),
            waveformCanvas: document.getElementById("waveformCanvas"),
            callerLevelText: document.getElementById("callerLevelText"),
            geminiLevelText: document.getElementById("geminiLevelText"),
            callerWaveformLegacy: document.getElementById("caller-waveform"),
            geminiWaveformLegacy: document.getElementById("gemini-waveform"),

            // Transcripts
            transcriptPanel: document.getElementById("transcriptPanel"),
            autoscrollToggle: document.getElementById("autoscrollToggle"),
            transcriptContainer: document.getElementById("transcriptContainer"),
            turnCounterBadge: document.getElementById("turnCounterBadge"),

            // Whisper
            whisperPanel: document.getElementById("whisperPanel"),
            whisperInput: document.getElementById("whisperInput"),
            sendWhisperBtn: document.getElementById("sendWhisperBtn"),
            whisperPresets: document.getElementById("whisperPresets"),

            // Dialer
            dialPhoneNumber: document.getElementById("dialPhoneNumber"),
            dialGoalPrompt: document.getElementById("dialGoalPrompt"),
            dialerKeypad: document.getElementById("dialer-keypad"),
            btnKeypadClear: document.getElementById("btnKeypadClear"),
            btnKeypadBackspace: document.getElementById("btnKeypadBackspace"),
            btnStartCall: document.getElementById("btnStartCall"),

            // History & Messages
            historyTable: document.getElementById("historyTable"),
            historyList: document.getElementById("historyList"),
            messageCenter: document.getElementById("messageCenter"),
            tabBtnMessages: document.getElementById("tabBtnMessages"),
            tabBtnHistory: document.getElementById("tabBtnHistory"),
            tabMessages: document.getElementById("tabMessages"),
            tabHistory: document.getElementById("tabHistory"),
            refreshHistoryBtn: document.getElementById("refreshHistoryBtn"),
            messagesCountBadge: document.getElementById("messagesCountBadge"),
            historyCountBadge: document.getElementById("historyCountBadge"),

            // Inbound Modal
            inboundModal: document.getElementById("inboundModal"),
            inboundCallerName: document.getElementById("inboundCallerName"),
            inboundCallerId: document.getElementById("inboundCallerId"),
            btnAiTakeover: document.getElementById("btnAiTakeover"),
            btnAnswerPersonally: document.getElementById("btnAnswerPersonally"),
            btnDeclineCall: document.getElementById("btnDeclineCall"),

            // Toast
            toastContainer: document.getElementById("toastContainer"),
        };
    }

    // =========================================================================
    // Toast Notification System
    // =========================================================================
    function showToast(message, type = "info", durationMs = 4000) {
        if (!dom.toastContainer) return;
        while (dom.toastContainer.children.length >= 5) {
            dom.toastContainer.removeChild(dom.toastContainer.firstChild);
        }
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;

        const iconMap = {
            info: "ℹ️",
            success: "✅",
            warning: "⚠️",
            error: "❌",
        };
        const iconSpan = document.createElement("span");
        iconSpan.textContent = iconMap[type] || "ℹ️";

        const textSpan = document.createElement("span");
        textSpan.textContent = message;

        toast.appendChild(iconSpan);
        toast.appendChild(textSpan);
        dom.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(20px)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, durationMs);
    }

    // =========================================================================
    // WebSocket /ws/live Connection & Reconnection Manager
    // =========================================================================
    function getWebSocketUrl() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.host || "localhost:8000";
        return `${protocol}//${host}/ws/live`;
    }

    function updateConnectionStatus(status) {
        if (!dom.connectionPill) return;
        dom.connectionPill.className = `status-pill ${status.toLowerCase()}`;
        dom.connectionPill.textContent =
            status === "connected"
                ? "Connected"
                : status === "reconnecting"
                ? "Reconnecting"
                : "Disconnected";
    }

    function initWebSocket() {
        if (state.ws) {
            try {
                state.ws.close();
            } catch (e) {
                // Ignore cleanup errors
            }
        }

        const url = getWebSocketUrl();
        updateConnectionStatus(state.reconnectAttempts > 0 ? "reconnecting" : "disconnected");

        try {
            state.ws = new WebSocket(url);
        } catch (err) {
            console.error("WebSocket init failed:", err);
            scheduleReconnect();
            return;
        }

        state.ws.onopen = function () {
            state.wsConnected = true;
            state.reconnectAttempts = 0;
            updateConnectionStatus("connected");
            showToast("Connected to Voice Assistant server", "success", 2500);

            // Heartbeat ping loop (every 25s)
            clearInterval(state.pingTimer);
            state.pingTimer = setInterval(() => {
                if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                    state.ws.send(JSON.stringify({ type: "ping", action: "ping" }));
                }
            }, 25000);
        };

        state.ws.onmessage = function (event) {
            try {
                const message = JSON.parse(event.data);
                handleWebSocketEvent(message);
            } catch (err) {
                console.warn("Received malformed WebSocket frame:", event.data, err);
            }
        };

        state.ws.onerror = function (err) {
            console.warn("WebSocket transport error:", err);
        };

        state.ws.onclose = function () {
            state.wsConnected = false;
            clearInterval(state.pingTimer);
            updateConnectionStatus("disconnected");
            scheduleReconnect();
        };
    }

    function scheduleReconnect() {
        if (state.reconnectTimer) clearTimeout(state.reconnectTimer);
        const delay = Math.min(1000 * Math.pow(1.5, state.reconnectAttempts), 10000);
        state.reconnectAttempts++;
        state.reconnectTimer = setTimeout(() => {
            initWebSocket();
        }, delay);
    }

    function sendWsAction(payload) {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify(payload));
            return true;
        }
        return false;
    }

    // =========================================================================
    // WebSocket Event Dispatcher
    // =========================================================================
    function handleWebSocketEvent(message) {
        const type = message.type;
        const data = message.data || {};

        switch (type) {
            case "initial_state":
                handleInitialState(message);
                break;

            case "call_state_change":
                handleCallStateChange(message);
                break;

            case "inbound_call_prompt":
                handleInboundCallPrompt(message);
                break;

            case "transcript_chunk":
                handleTranscriptChunk(data);
                break;

            case "audio_telemetry":
                handleAudioTelemetry(data);
                break;

            case "tool_execution":
                handleToolExecution(data);
                break;

            case "operator_whisper_ack":
                handleWhisperAck(data);
                break;

            case "call_summary_generated":
                handleCallSummary(data);
                break;

            case "pong":
                // Heartbeat response acknowledged
                break;

            case "error":
                showToast(`Server warning: ${message.message || "Unknown error"}`, "warning");
                break;

            default:
                console.debug("Unhandled WebSocket event:", type, message);
                break;
        }
    }

    // =========================================================================
    // State Event Handlers
    // =========================================================================
    function handleInitialState(message) {
        const data = message.data || {};
        const callState = message.state || data.state || "IDLE";
        const callId = message.call_id || data.call_id || null;
        updateCallState(callState, callId);
        loadStatus();
        loadDevices();
        loadRecentCalls();
        loadMessages();
    }

    function handleCallStateChange(message) {
        const data = message.data || {};
        const nextState = data.state || message.state || "IDLE";
        const callId = message.call_id || state.currentCallId;
        const direction = data.direction || (data.phone_number ? "OUTBOUND" : null);

        if (direction) {
            updateDirectionBadge(direction);
        }

        updateCallState(nextState, callId);

        if (nextState.toUpperCase() === "COMPLETED") {
            const reason = data.reason || "Call completed";
            showToast(`Call ended (${reason})`, "info", 3000);
            loadRecentCalls();
            loadMessages();
        }
    }

    function handleInboundCallPrompt(message) {
        if (state.callState === "ACTIVE") return;

        const data = message.data || {};
        const callerId = data.caller_id || "Unknown";
        const callerName = data.caller_name || "Unknown Caller";

        updateDirectionBadge("INBOUND");
        if (dom.callerNameDisplay) dom.callerNameDisplay.textContent = callerName;
        if (dom.callerIdDisplay) dom.callerIdDisplay.textContent = callerId;
        if (dom.inboundCallerName) dom.inboundCallerName.textContent = callerName;
        if (dom.inboundCallerId) dom.inboundCallerId.textContent = callerId;

        showInboundModal(true);
    }

    function handleTranscriptChunk(data) {
        if (!data || !data.text) return;
        const turnId = data.turn_id || "turn_default";
        const speaker = (data.speaker || "caller").toLowerCase();
        const text = data.text;
        const isFinal = Boolean(data.is_final);

        appendTranscriptChunk(turnId, speaker, text, isFinal);
    }

    function handleAudioTelemetry(data) {
        if (!data) return;

        if (data.caller) {
            state.telemetry.caller.targetRms = Math.max(0.0, safeFloat(data.caller.rms, 0.0));
            state.telemetry.caller.targetPeak = Math.max(0.0, safeFloat(data.caller.peak, 0.0));
            if (Array.isArray(data.caller.spectrum)) {
                state.telemetry.caller.targetSpectrum = data.caller.spectrum.map((v) =>
                    Math.max(0.0, safeFloat(v, 0.0))
                );
            }
        }

        if (data.gemini) {
            state.telemetry.gemini.targetRms = Math.max(0.0, safeFloat(data.gemini.rms, 0.0));
            state.telemetry.gemini.targetPeak = Math.max(0.0, safeFloat(data.gemini.peak, 0.0));
            if (Array.isArray(data.gemini.spectrum)) {
                state.telemetry.gemini.targetSpectrum = data.gemini.spectrum.map((v) =>
                    Math.max(0.0, safeFloat(v, 0.0))
                );
            }
        }

        // Update textual level indicators
        if (dom.callerLevelText) {
            dom.callerLevelText.textContent = `RMS: ${state.telemetry.caller.targetRms.toFixed(2)}`;
        }
        if (dom.geminiLevelText) {
            dom.geminiLevelText.textContent = `RMS: ${state.telemetry.gemini.targetRms.toFixed(2)}`;
        }
    }

    function handleToolExecution(data) {
        const toolName = data.tool || "unknown_tool";
        showToast(`AI executed tool: ${toolName}`, "info", 3000);
        appendToolBadge(toolName, data.result);
        if (toolName === "take_message") {
            loadMessages();
        }
    }

    function handleWhisperAck(data) {
        const instruction = data.instruction || "";
        const status = data.status || "delivered";
        showToast(`Whisper ${status}: "${instruction}"`, "success", 3000);
    }

    function handleCallSummary(data) {
        const summary = data.summary || "Summary generated";
        showToast("Post-call summary generated", "success", 4000);
        appendSummaryBlock(summary, data.action_items);
        loadRecentCalls();
    }

    // =========================================================================
    // Call State & Timer Controller
    // =========================================================================
    function updateCallState(rawState, callId = null) {
        const s = (rawState || "IDLE").toUpperCase();
        state.callState = s;
        if (callId) state.currentCallId = callId;

        // Update state pill text and CSS class
        if (dom.callStatePill) {
            dom.callStatePill.textContent = s;
            dom.callStatePill.className = `state-pill state-${s.toLowerCase()}`;
        }

        // State-dependent controls
        const isRinging = s === "RINGING";
        const isActive = s === "ACTIVE" || s === "OPERATOR_TALKING";
        const isHold = s === "ON_HOLD" || s === "ON_HOLD_TRANSFER";
        const isCompleted = s === "COMPLETED" || s === "DECLINED" || s === "IDLE";

        if (dom.btnAnswerCall) {
            if (s === "IDLE") {
                dom.btnAnswerCall.disabled = false;
                dom.btnAnswerCall.textContent = "🤖 AI Take Over Call";
                dom.btnAnswerCall.className = "btn btn-success";
            } else if (isRinging) {
                dom.btnAnswerCall.disabled = false;
                dom.btnAnswerCall.textContent = "Answer Call";
                dom.btnAnswerCall.className = "btn btn-success pulse-animation";
            } else {
                dom.btnAnswerCall.disabled = true;
                dom.btnAnswerCall.textContent = "Call In Progress";
                dom.btnAnswerCall.className = "btn btn-success";
            }
        }
        if (dom.btnEndCall) {
            dom.btnEndCall.disabled = isCompleted;
        }
        if (dom.btnStartCall) {
            dom.btnStartCall.disabled = !isCompleted;
        }

        // Duration Stopwatch Lifecycle
        if (isActive || isHold) {
            if (!state.timerInterval) {
                state.callStartTime = Date.now();
                startDurationTimer();
            }
        } else if (isCompleted) {
            stopDurationTimer();
            showInboundModal(false);
            if (s === "IDLE") {
                resetCallUI();
            }
        }

        // Inbound Modal visibility sync
        if (isRinging) {
            showInboundModal(true);
        } else {
            showInboundModal(false);
        }
    }

    function updateDirectionBadge(direction) {
        if (!dom.directionBadge) return;
        const d = (direction || "IDLE").toUpperCase();
        state.callDirection = d;
        dom.directionBadge.textContent = d;
        dom.directionBadge.className = `direction-badge direction-${d.toLowerCase()}`;
    }

    function startDurationTimer() {
        clearInterval(state.timerInterval);
        state.timerInterval = setInterval(() => {
            if (!state.callStartTime) return;
            const elapsed = Math.floor((Date.now() - state.callStartTime) / 1000);
            const hrs = String(Math.floor(elapsed / 3600)).padStart(2, "0");
            const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
            const secs = String(elapsed % 60).padStart(2, "0");
            if (dom.callDurationTimer) {
                dom.callDurationTimer.textContent = `${hrs}:${mins}:${secs}`;
            }
        }, 1000);
    }

    function stopDurationTimer() {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }

    function resetCallUI() {
        state.currentCallId = null;
        state.callStartTime = null;
        if (dom.callDurationTimer) dom.callDurationTimer.textContent = "00:00:00";
        if (dom.callerNameDisplay) dom.callerNameDisplay.textContent = "No Active Call";
        if (dom.callerIdDisplay) dom.callerIdDisplay.textContent = "—";
        updateDirectionBadge("IDLE");
    }

    // =========================================================================
    // Dual Canvas Waveform Visualizer (60 FPS requestAnimationFrame)
    // =========================================================================
    let animFrameId = null;
    let cachedDpr = 1;
    let cachedWidth = 0;
    let cachedHeight = 0;

    function updateCanvasSize(canvas) {
        if (!canvas) return;
        cachedDpr = window.devicePixelRatio || 1;
        const dpr = cachedDpr;
        const rect = canvas.getBoundingClientRect();
        const displayWidth = Math.floor(rect.width || canvas.width || 800);
        const displayHeight = Math.floor(rect.height || canvas.height || 240);
        cachedWidth = displayWidth;
        cachedHeight = displayHeight;
        const targetWidth = Math.floor(displayWidth * dpr);
        const targetHeight = Math.floor(displayHeight * dpr);
        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
        }
    }

    function initVisualizer() {
        const canvas = dom.waveformCanvas;
        if (!canvas) return;

        updateCanvasSize(canvas);
        window.addEventListener("resize", () => updateCanvasSize(canvas));

        function renderFrame() {
            renderWaveforms(canvas);
            animFrameId = requestAnimationFrame(renderFrame);
        }

        if (animFrameId) cancelAnimationFrame(animFrameId);
        animFrameId = requestAnimationFrame(renderFrame);
    }

    function renderWaveforms(canvas) {
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        if (cachedWidth === 0 || cachedHeight === 0) {
            updateCanvasSize(canvas);
        }

        ctx.save();
        ctx.scale(cachedDpr, cachedDpr);

        const w = cachedWidth;
        const h = cachedHeight;

        // Smooth 60 FPS interpolation (decay towards target telemetry values with NaN guards)
        const lerp = (curr, target, factor) => {
            const c = safeFloat(curr, 0.0);
            const t = safeFloat(target, 0.0);
            return c + (t - c) * factor;
        };
        const cal = state.telemetry.caller;
        const gem = state.telemetry.gemini;

        cal.rms = lerp(cal.rms, cal.targetRms, 0.25);
        cal.peak = lerp(cal.peak, cal.targetPeak, 0.25);
        gem.rms = lerp(gem.rms, gem.targetRms, 0.25);
        gem.peak = lerp(gem.peak, gem.targetPeak, 0.25);

        for (let i = 0; i < 8; i++) {
            cal.spectrum[i] = lerp(cal.spectrum[i] || 0, cal.targetSpectrum[i] || 0, 0.3);
            gem.spectrum[i] = lerp(gem.spectrum[i] || 0, gem.targetSpectrum[i] || 0, 0.3);
        }

        // Clear canvas
        ctx.fillStyle = "#080c14";
        ctx.fillRect(0, 0, w, h);

        // Center split line
        const midY = h / 2;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(w, midY);
        ctx.stroke();

        // Track 1: Caller (Top Half, Cyan #06b6d4)
        renderSingleTrack(
            ctx,
            0,
            0,
            w,
            midY,
            "#06b6d4",
            "rgba(6, 182, 212, 0.15)",
            cal.rms,
            cal.peak,
            cal.spectrum,
            "CALLER AUDIO (16 kHz)",
            0.0
        );

        // Track 2: Gemini Live (Bottom Half, Emerald #10b981)
        renderSingleTrack(
            ctx,
            0,
            midY,
            w,
            midY,
            "#10b981",
            "rgba(16, 185, 129, 0.15)",
            gem.rms,
            gem.peak,
            gem.spectrum,
            "GEMINI AOEDE (24 kHz)",
            Math.PI * 0.5
        );

        ctx.restore();
    }

    function renderSingleTrack(ctx, x, y, width, height, strokeColor, fillColor, rms, peak, spectrum, label, phaseShift) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.clip();

        const safeRms = Math.max(0.0, safeFloat(rms, 0.0));
        const safePeak = Math.max(0.0, safeFloat(peak, 0.0));
        const safeSpectrum = Array.isArray(spectrum) ? spectrum.map((s) => Math.max(0.0, safeFloat(s, 0.0))) : [];

        const trackMidY = y + height / 2;
        const now = performance.now() * 0.003;

        // Background Frequency Spectrum Bars (8 bands)
        const barCount = 8;
        const barWidth = Math.max(8, width / (barCount * 3));
        const barSpacing = (width - barCount * barWidth) / (barCount + 1);

        for (let i = 0; i < barCount; i++) {
            const barH = Math.min((safeSpectrum[i] || 0) * height * 0.9 + (safeRms * height * 0.2), height * 0.85);
            const bx = x + barSpacing + i * (barWidth + barSpacing);
            const by = trackMidY - barH / 2;

            ctx.fillStyle = fillColor;
            ctx.fillRect(bx, by, barWidth, barH);
        }

        // Dynamic Waveform Line (Bezier Interpolation)
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.shadowColor = strokeColor;
        ctx.shadowBlur = safeRms > 0.05 ? 12 : 4;

        ctx.beginPath();
        const points = 40;
        const step = width / points;

        for (let i = 0; i <= points; i++) {
            const px = x + i * step;
            const progress = i / points;
            // Compound oscillation modulated by live audio RMS amplitude
            const wave1 = Math.sin(progress * 12 + now + phaseShift);
            const wave2 = Math.cos(progress * 24 - now * 1.5);
            const amplitude = Math.max(2, (safeRms * 0.8 + safePeak * 0.2) * (height * 0.42));
            const py = trackMidY + (wave1 * 0.7 + wave2 * 0.3) * amplitude;

            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // Right-aligned Peak / RMS Level Meter Bar
        const meterW = 6;
        const meterX = x + width - 16;
        const meterH = height * 0.7;
        const meterY = y + (height - meterH) / 2;

        ctx.fillStyle = "rgba(255, 255, 255, 0.06)";
        ctx.fillRect(meterX, meterY, meterW, meterH);

        const fillH = Math.min(safePeak * meterH, meterH);
        ctx.fillStyle = strokeColor;
        ctx.fillRect(meterX, meterY + meterH - fillH, meterW, fillH);

        // Channel Tag Label
        ctx.shadowBlur = 0;
        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
        ctx.font = "600 10px monospace";
        ctx.fillText(label, x + 12, y + 16);

        ctx.restore();
    }

    // =========================================================================
    // Dual-Column Streaming Transcripts Manager
    // =========================================================================
    function appendTranscriptChunk(turnId, speaker, text, isFinal) {
        if (!dom.transcriptContainer) return;

        // Clear empty state message if first chunk
        const emptyState = dom.transcriptContainer.querySelector(".transcript-empty-state");
        if (emptyState) {
            dom.transcriptContainer.removeChild(emptyState);
        }

        let turnRecord = state.turns.get(turnId);

        if (!turnRecord) {
            // Create a new speech turn bubble
            const bubble = document.createElement("div");
            bubble.className = `turn-bubble turn-${speaker}`;
            bubble.id = `turn_${turnId}`;

            const header = document.createElement("div");
            header.className = "turn-header";

            const speakerSpan = document.createElement("span");
            speakerSpan.className = "turn-speaker";
            const icon = speaker === "gemini" ? "✨" : speaker === "operator" ? "👤" : "📞";
            const displayName =
                speaker === "gemini" ? "Gemini Aoede" : speaker === "operator" ? "Operator" : "Caller";
            speakerSpan.textContent = `${icon} ${displayName}`;

            const timeSpan = document.createElement("span");
            timeSpan.className = "turn-time";
            const now = new Date();
            timeSpan.textContent = now.toLocaleTimeString([], { hour12: false });

            header.appendChild(speakerSpan);
            header.appendChild(timeSpan);

            const textDiv = document.createElement("div");
            textDiv.className = "turn-text";
            textDiv.textContent = text;

            const cursor = document.createElement("span");
            cursor.className = "typing-cursor";
            if (!isFinal) textDiv.appendChild(cursor);

            bubble.appendChild(header);
            bubble.appendChild(textDiv);
            dom.transcriptContainer.appendChild(bubble);

            turnRecord = {
                speaker,
                text,
                element: bubble,
                textDiv,
                cursor,
                isFinal,
            };
            state.turns.set(turnId, turnRecord);
            state.turnCount++;

            if (dom.turnCounterBadge) {
                dom.turnCounterBadge.textContent = `${state.turnCount} turns`;
            }
        } else {
            // Update in-flight streaming chunk text
            turnRecord.text += text;
            turnRecord.textDiv.textContent = turnRecord.text;

            if (isFinal) {
                turnRecord.isFinal = true;
                if (turnRecord.cursor && turnRecord.cursor.parentNode) {
                    turnRecord.cursor.parentNode.removeChild(turnRecord.cursor);
                }
            } else {
                if (!turnRecord.textDiv.contains(turnRecord.cursor)) {
                    turnRecord.textDiv.appendChild(turnRecord.cursor);
                }
            }
        }

        // Auto-scroll snapping
        if (dom.autoscrollToggle && dom.autoscrollToggle.checked) {
            dom.transcriptContainer.scrollTop = dom.transcriptContainer.scrollHeight;
        }
    }

    function appendToolBadge(toolName, result) {
        if (!dom.transcriptContainer) return;
        const pill = document.createElement("div");
        pill.className = "tool-pill";
        const safeTool = escapeHtml(toolName);
        const safeResult = escapeHtml(JSON.stringify(result || {}));
        pill.innerHTML = `⚙️ <strong>${safeTool}</strong>: ${safeResult}`;
        dom.transcriptContainer.appendChild(pill);

        if (dom.autoscrollToggle && dom.autoscrollToggle.checked) {
            dom.transcriptContainer.scrollTop = dom.transcriptContainer.scrollHeight;
        }
    }

    function appendSummaryBlock(summary, actionItems) {
        if (!dom.transcriptContainer) return;
        const card = document.createElement("div");
        card.className = "card summary-card";
        card.style.marginTop = "0.75rem";
        card.style.background = "rgba(16, 185, 129, 0.08)";
        card.style.borderColor = "rgba(16, 185, 129, 0.3)";

        let itemsHtml = "";
        if (Array.isArray(actionItems) && actionItems.length > 0) {
            itemsHtml = `<ul style="margin-left: 1.25rem; margin-top: 0.35rem; font-size: 0.85rem;">
                ${actionItems.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>`;
        }

        card.innerHTML = `
            <h4 style="color: var(--emerald-accent); font-size: 0.9rem; margin-bottom: 0.3rem;">📋 Call Summary</h4>
            <p style="font-size: 0.85rem; color: var(--text-primary);">${escapeHtml(summary)}</p>
            ${itemsHtml}
        `;
        dom.transcriptContainer.appendChild(card);

        if (dom.autoscrollToggle && dom.autoscrollToggle.checked) {
            dom.transcriptContainer.scrollTop = dom.transcriptContainer.scrollHeight;
        }
    }

    // =========================================================================
    // Mid-Call Operator Whisper Box Controller
    // =========================================================================
    function sendWhisperInstruction(instruction) {
        if (!instruction || !instruction.trim()) return;
        const cleanText = instruction.trim();

        // 1. Send over active WebSocket
        const wsSent = sendWsAction({
            type: "whisper",
            instruction: cleanText,
            action: "whisper",
            data: { instruction: cleanText },
        });

        // 2. Dispatch via REST API fallback only if WebSocket was not sent
        if (!wsSent) {
            fetch("/api/call/whisper", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ instruction: cleanText }),
            })
                .then((r) => {
                    if (!r.ok) {
                        return r.text().then((text) => {
                            throw new Error(text || `HTTP ${r.status}`);
                        });
                    }
                    return r.json();
                })
                .then((res) => {
                    showToast(`Whisper dispatched: "${cleanText}"`, "info", 2500);
                })
                .catch((err) => {
                    showToast(`Failed to send whisper: ${err.message}`, "error");
                });
        }

        if (dom.whisperInput) {
            dom.whisperInput.value = "";
            dom.whisperInput.focus();
        }
    }

    function initWhisperHandlers() {
        if (dom.sendWhisperBtn && dom.whisperInput) {
            dom.sendWhisperBtn.addEventListener("click", () => {
                sendWhisperInstruction(dom.whisperInput.value);
            });

            dom.whisperInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendWhisperInstruction(dom.whisperInput.value);
                }
            });
        }

        // Quick Preset Chips
        if (dom.whisperPresets) {
            dom.whisperPresets.addEventListener("click", (e) => {
                const btn = e.target.closest(".chip");
                if (btn && btn.dataset.whisper) {
                    sendWhisperInstruction(btn.dataset.whisper);
                }
            });
        }
    }

    // =========================================================================
    // Inbound Call Approval Modal Controller
    // =========================================================================
    function setInboundButtonsDisabled(disabled) {
        if (dom.btnAiTakeover) dom.btnAiTakeover.disabled = disabled;
        if (dom.btnAnswerPersonally) dom.btnAnswerPersonally.disabled = disabled;
        if (dom.btnDeclineCall) dom.btnDeclineCall.disabled = disabled;
    }

    function showInboundModal(visible) {
        if (!dom.inboundModal) return;
        if (visible) {
            state.isDeciding = false;
            setInboundButtonsDisabled(false);
            if (typeof dom.inboundModal.showModal === "function") {
                if (!dom.inboundModal.open) {
                    try {
                        dom.inboundModal.showModal();
                    } catch (e) {
                        dom.inboundModal.style.display = "block";
                    }
                }
            } else {
                dom.inboundModal.style.display = "block";
            }
        } else {
            if (typeof dom.inboundModal.close === "function") {
                if (dom.inboundModal.open) {
                    dom.inboundModal.close();
                }
            }
            dom.inboundModal.style.display = "none";
        }
    }

    function handleInboundDecision(decision) {
        if (state.isDeciding) return;
        state.isDeciding = true;
        setInboundButtonsDisabled(true);

        showInboundModal(false);

        // 1. Send WebSocket decision
        sendWsAction({
            type: "inbound_decision",
            decision: decision,
            action: decision,
            data: { decision: decision },
        });

        // 2. Dispatch REST API call
        fetch("/api/call/inbound-action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: decision }),
        })
            .then((r) => {
                if (!r.ok) {
                    return r.text().then((text) => {
                        throw new Error(text || `HTTP ${r.status}`);
                    });
                }
                return r.json();
            })
            .then((res) => {
                showToast(`Inbound action '${decision}' confirmed`, "success", 2500);
            })
            .catch((err) => {
                console.warn("Inbound action REST fallback notice:", err.message);
            })
            .finally(() => {
                state.isDeciding = false;
                setInboundButtonsDisabled(false);
            });
    }

    function initInboundModalHandlers() {
        if (dom.btnAiTakeover) {
            dom.btnAiTakeover.addEventListener("click", () => handleInboundDecision("ai_takeover"));
        }
        if (dom.btnAnswerPersonally) {
            dom.btnAnswerPersonally.addEventListener("click", () => handleInboundDecision("answer_personally"));
        }
        if (dom.btnDeclineCall) {
            dom.btnDeclineCall.addEventListener("click", () => handleInboundDecision("decline"));
        }
        if (dom.inboundModal) {
            dom.inboundModal.addEventListener("close", () => {
                state.isDeciding = false;
                setInboundButtonsDisabled(false);
            });
        }
    }

    // =========================================================================
    // Outbound Dialer Controller
    // =========================================================================
    function initDialerHandlers() {
        if (!dom.dialPhoneNumber) return;

        // Keypad buttons
        if (dom.dialerKeypad) {
            dom.dialerKeypad.addEventListener("click", (e) => {
                const btn = e.target.closest(".keypad-btn");
                if (btn && btn.dataset.key) {
                    dom.dialPhoneNumber.value += btn.dataset.key;
                    dom.dialPhoneNumber.focus();
                }
            });
        }

        // Keypad Clear
        if (dom.btnKeypadClear) {
            dom.btnKeypadClear.addEventListener("click", () => {
                dom.dialPhoneNumber.value = "";
                dom.dialPhoneNumber.focus();
            });
        }

        // Keypad Backspace
        if (dom.btnKeypadBackspace) {
            dom.btnKeypadBackspace.addEventListener("click", () => {
                dom.dialPhoneNumber.value = dom.dialPhoneNumber.value.slice(0, -1);
                dom.dialPhoneNumber.focus();
            });
        }

        // Start Outbound Call Button
        if (dom.btnStartCall) {
            dom.btnStartCall.addEventListener("click", () => {
                const phoneNumber = dom.dialPhoneNumber.value.trim();
                const goalPrompt = dom.dialGoalPrompt ? dom.dialGoalPrompt.value.trim() : "";

                if (!phoneNumber) {
                    showToast("Please enter a destination phone number", "warning");
                    dom.dialPhoneNumber.focus();
                    return;
                }

                dom.btnStartCall.disabled = true;
                showToast(`Dialing ${phoneNumber}...`, "info", 3000);

                fetch("/api/call/dial", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        phone_number: phoneNumber,
                        goal_prompt: goalPrompt,
                    }),
                })
                    .then((r) => {
                        if (!r.ok) {
                            return r.json().then((err) => {
                                throw new Error(err.detail || "Dialing failed");
                            });
                        }
                        return r.json();
                    })
                    .then((res) => {
                        showToast(`Call initiated (${res.call_id})`, "success", 3000);
                        updateDirectionBadge("OUTBOUND");
                        if (dom.callerNameDisplay) dom.callerNameDisplay.textContent = "Outbound Call";
                        if (dom.callerIdDisplay) dom.callerIdDisplay.textContent = phoneNumber;
                    })
                    .catch((err) => {
                        showToast(err.message, "error", 4000);
                        if (state.callState === "IDLE") {
                            dom.btnStartCall.disabled = false;
                        }
                    });
            });
        }
    }

    // =========================================================================
    // Primary Call Controls (Answer & End Call)
    // =========================================================================
    function initCallControlHandlers() {
        if (dom.btnAnswerCall) {
            dom.btnAnswerCall.addEventListener("click", () => {
                if (state.callState === "RINGING") {
                    handleInboundDecision("ai_takeover");
                } else {
                    showToast("🤖 Initiating AI Takeover...", "info", 2000);
                    fetch("/api/call/takeover", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            caller_id: "Inbound Cellular Call",
                            caller_name: "Phone Link / Cellular",
                        }),
                    })
                        .then((r) => r.json())
                        .then((res) => {
                            showToast("AI Voice Agent connected to call!", "success", 3000);
                        })
                        .catch((err) => {
                            showToast(`Takeover error: ${err.message}`, "error", 3000);
                        });
                }
            });
        }

        if (dom.btnEndCall) {
            dom.btnEndCall.addEventListener("click", () => {
                dom.btnEndCall.disabled = true;

                // Send WebSocket hangup action
                sendWsAction({
                    type: "hangup",
                    reason: "operator_hangup",
                    action: "hangup",
                    data: { reason: "operator_hangup" },
                });

                // REST hangup fallback
                fetch("/api/call/hangup", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ reason: "operator_hangup" }),
                })
                    .then((r) => r.json())
                    .then((res) => {
                        showToast("Call ended by operator", "info", 2500);
                        updateCallState("COMPLETED");
                    })
                    .catch((err) => {
                        console.warn("Hangup REST notice:", err.message);
                    });
            });
        }
    }

    // =========================================================================
    // Tab Controller & History / Message Center
    // =========================================================================
    function initTabHandlers() {
        if (dom.tabBtnMessages && dom.tabBtnHistory) {
            dom.tabBtnMessages.addEventListener("click", () => switchTab("tabMessages"));
            dom.tabBtnHistory.addEventListener("click", () => switchTab("tabHistory"));
        }

        if (dom.refreshHistoryBtn) {
            dom.refreshHistoryBtn.addEventListener("click", () => {
                loadRecentCalls();
                loadMessages();
                showToast("Refreshed records", "info", 1500);
            });
        }

        // Message list delegation for "Call Back" action
        if (dom.messageCenter) {
            dom.messageCenter.addEventListener("click", (e) => {
                const btn = e.target.closest(".btn-callback");
                if (btn && btn.dataset.phone) {
                    if (dom.dialPhoneNumber) {
                        dom.dialPhoneNumber.value = btn.dataset.phone;
                        dom.dialPhoneNumber.scrollIntoView({ behavior: "smooth" });
                        dom.dialPhoneNumber.focus();
                        showToast(`Loaded callback number: ${btn.dataset.phone}`, "info", 2000);
                    }
                }
            });
        }
    }

    function switchTab(activeTabId) {
        const isMessages = activeTabId === "tabMessages";
        if (dom.tabBtnMessages) dom.tabBtnMessages.classList.toggle("active", isMessages);
        if (dom.tabBtnHistory) dom.tabBtnHistory.classList.toggle("active", !isMessages);

        if (dom.tabMessages) {
            dom.tabMessages.classList.toggle("active", isMessages);
            dom.tabMessages.hidden = !isMessages;
        }
        if (dom.tabHistory) {
            dom.tabHistory.classList.toggle("active", !isMessages);
            dom.tabHistory.hidden = isMessages;
        }
    }

    // =========================================================================
    // REST API Integration Functions
    // =========================================================================
    function loadStatus() {
        fetch("/api/status")
            .then((r) => r.json())
            .then((data) => {
                if (data.driver && dom.driverBadge) {
                    dom.driverBadge.textContent = `Driver: ${data.driver}`;
                }
                if (data.state) {
                    updateCallState(data.state, data.current_call_id);
                }
            })
            .catch((err) => console.debug("Status fetch:", err.message));
    }

    function loadDevices() {
        fetch("/api/devices")
            .then((r) => r.json())
            .then((data) => {
                state.devices = data;

                if (dom.audioInputSelect && Array.isArray(data.inputs)) {
                    dom.audioInputSelect.innerHTML = "";
                    let hasSelected = false;
                    data.inputs.forEach((d) => {
                        const opt = document.createElement("option");
                        opt.value = d.id;
                        opt.textContent = `${d.name} (${d.channels}ch)`;
                        if (d.id === data.recommended_in || (!hasSelected && d.name.toLowerCase().includes("iqoo"))) {
                            opt.selected = true;
                            hasSelected = true;
                        }
                        dom.audioInputSelect.appendChild(opt);
                    });
                }

                if (dom.audioOutputSelect && Array.isArray(data.outputs)) {
                    dom.audioOutputSelect.innerHTML = "";
                    let hasSelected = false;
                    data.outputs.forEach((d) => {
                        const opt = document.createElement("option");
                        opt.value = d.id;
                        opt.textContent = `${d.name} (${d.channels}ch)`;
                        if (d.id === data.recommended_out || (!hasSelected && d.name.toLowerCase().includes("iqoo"))) {
                            opt.selected = true;
                            hasSelected = true;
                        }
                        dom.audioOutputSelect.appendChild(opt);
                    });
                }
            })
            .catch((err) => console.debug("Devices fetch:", err.message));
    }

    function loadRecentCalls() {
        fetch("/api/history/calls?limit=25")
            .then((r) => r.json())
            .then((data) => {
                const calls = data.calls || [];
                if (dom.historyCountBadge) {
                    dom.historyCountBadge.textContent = String(calls.length);
                }
                if (!dom.historyList) return;

                if (calls.length === 0) {
                    dom.historyList.innerHTML = `<tr><td colspan="6" class="table-empty">No past calls found.</td></tr>`;
                    return;
                }

                dom.historyList.innerHTML = calls
                    .map((c) => {
                        const durationSec = c.duration_seconds || c.duration || 0;
                        const mins = Math.floor(durationSec / 60);
                        const secs = Math.floor(durationSec % 60);
                        const durStr = `${mins}:${String(secs).padStart(2, "0")}`;
                        const start = c.started_at ? new Date(c.started_at).toLocaleString() : "—";
                        const safeId = escapeHtml(c.id);
                        const safeDir = escapeHtml((c.direction || "inbound").toUpperCase());
                        const safeCaller = escapeHtml(c.caller_name || c.caller_id || "Unknown");
                        const safeStatus = escapeHtml((c.status || "completed").toUpperCase());
                        const safeStart = escapeHtml(start);
                        return `
                        <tr>
                            <td><code>${safeId}</code></td>
                            <td><span class="badge">${safeDir}</span></td>
                            <td><strong>${safeCaller}</strong></td>
                            <td><span class="badge">${safeStatus}</span></td>
                            <td>${durStr}</td>
                            <td>${safeStart}</td>
                        </tr>
                    `;
                    })
                    .join("");
            })
            .catch((err) => console.debug("History fetch:", err.message));
    }

    function loadMessages() {
        fetch("/api/history/messages")
            .then((r) => r.json())
            .then((data) => {
                const messages = data.messages || [];
                if (dom.messagesCountBadge) {
                    dom.messagesCountBadge.textContent = String(messages.length);
                }
                if (!dom.messageCenter) return;

                if (messages.length === 0) {
                    dom.messageCenter.innerHTML = `
                    <div class="empty-state">No messages captured yet. Messages recorded by Gemini's take_message tool will appear here.</div>
                `;
                    return;
                }

                dom.messageCenter.innerHTML = messages
                    .map((m) => {
                        const urgency = escapeHtml((m.urgency || "medium").toLowerCase());
                        const phone = m.phone_number || "";
                        const safePhone = escapeHtml(phone);
                        const safeCaller = escapeHtml(m.caller_name || "Unknown");
                        const safeMessage = escapeHtml(m.message || "");
                        return `
                        <div class="message-card">
                            <span class="urgency-badge urgency-${urgency}">${urgency}</span>
                            <div class="message-body-col">
                                <div class="message-caller-header">
                                    <span class="message-caller-name">${safeCaller}</span>
                                    ${safePhone ? `<span class="message-caller-phone">${safePhone}</span>` : ""}
                                </div>
                                <div class="message-text">${safeMessage}</div>
                            </div>
                            <div class="message-actions">
                                ${
                                    safePhone
                                        ? `<button type="button" class="btn btn-secondary btn-icon btn-callback" data-phone="${safePhone}" title="Call Back">📞 Call Back</button>`
                                        : ""
                                }
                            </div>
                        </div>
                    `;
                    })
                    .join("");
            })
            .catch((err) => console.debug("Messages fetch:", err.message));
    }

    // =========================================================================
    // Initialization Bootstrap
    // =========================================================================
    function init() {
        initDomElements();
        initVisualizer();
        initWhisperHandlers();
        initInboundModalHandlers();
        initDialerHandlers();
        initCallControlHandlers();
        initTabHandlers();

        if (dom.refreshDevicesBtn) {
            dom.refreshDevicesBtn.addEventListener("click", () => {
                loadDevices();
                showToast("Audio endpoints refreshed", "info", 1500);
            });
        }

        // Establish WebSocket connection to backend live hub
        initWebSocket();

        // Initial REST data synchronization
        loadStatus();
        loadDevices();
        loadRecentCalls();
        loadMessages();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
