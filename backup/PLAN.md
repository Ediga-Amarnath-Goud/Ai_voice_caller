# AI Voice Caller - Codebase Assessment & Delivery Plan (v2)

> **2026-11-17 update - architecture superseded.** Bluetooth HFP is a dead end on
> this PC: the hands-free endpoint is registered in the registry and its devnode is
> started, but `IMMDeviceEnumerator` never exposes it, and every WDM-KS open fails
> with `-9999`. The phone staying on `BT_SCO` throughout a call did not change that.
>
> The working transport is **Phone Link + a virtual audio cable**, with no extra
> telephony hardware:
>
> - caller -> Phone Link -> Realtek speakers -> WASAPI loopback -> Pipecat
> - Pipecat -> CABLE Input -> CABLE Output -> Phone Link's console mic -> caller
> - ADB (wireless) dials / answers / hangs up; the phone stays locked.
>
> Verified live: the caller's voice appeared on the Realtek loopback for the exact
> duration of a Phone Link call; a test tone injected into CABLE Input was heard by
> the caller; the Gemini greeting was recorded on CABLE Output at -1.9 dBFS.
> `agent/run_call.py --bridge` is the entry point. `agent/audio_routing.py` switches
> only the console mic for the duration of a call and restores it afterwards, so the
> machine's normal audio is untouched. Sections 5 and 6 below describe the abandoned
> HFP approach and are kept for history only.

**Assessment date:** 2026-11-14  
**Scope:** full read of production code, tests, UI, configuration, Windows device tree, and runtime environment.  
**Status:** analysis + plan only. No production code was changed.

> **Correction to v1:** v1 treated Twilio/SIP as the primary transport and treated
> transcription as a call blocker. Both were wrong for this project. The intended
> transport is the **phone over Bluetooth HFP**, and Gemini 3.8 Live is audio-native,
> so no separate transcription service is required. This version reflects the actual
> Bluetooth design and separates "needed for the call" from "needed for the transcript UI."

---

## 0. Bottom line

The project is a **well-built prototype of a Bluetooth call assistant**. The UI, database,
call-state machine, Gemini Live client, and test suites are all present. The intended
transport is clear from the artifacts:

- `telephony/windows_driver.py` discovers Bluetooth HFP endpoints.
- `server/app.py` recommends the iQOO Z9 5G Hands-Free endpoints.
- The Windows device tree shows `iQOO Z9 5G Hands-Free HF` with HFP hardware ID
  `0000111f` and device description **“Microsoft Bluetooth Hands-Free Profile
  Handsfree role.”**
- `qtriggerinfo` contains `BTAGService` (Bluetooth Audio Gateway Service), a Bluetooth
  diagnostic artifact.

So the Bluetooth path is real and intended. The remaining problems are not "pick a
different transport." They are:

1. **The app does not control the phone call.** It opens a `tel:` URL for outbound and
   only flips local state for answer/hangup. Inbound `RINGING` is simulated by an API
   call. The user still dials/answers on the phone (or Phone Link); the app is an audio
   bridge, not a telephony client.
2. **The HFP audio direction appears reversed for this role.** The phone is the HFP
   **Audio Gateway (AG)**; Windows is the **Hands-Free (HF)** unit. In that role, the
   “iQOO Hands-Free HF Audio” input is the **PC microphone** (your voice → caller) and
   the output is the **PC speakers** (caller → you). The app assumes input = caller and
   output = AI voice to caller. As coded, Gemini may hear the room/local user and speak
   to the local speakers instead of the remote caller. This must be proven with one real
   test call.
3. **Caller audio is dropped in production** because of an event-loop capture bug in
   `CallStateManager` (P0, independent of the transport).
4. **There is no Gemini reconnect**; a dropped socket leaves a dead call marked ACTIVE.

**Can I finish it?**

- **Bluetooth audio-bridge correctness, UI, state machine, tests, docs:** yes, I can
  implement and verify the software. Estimated 6–10 focused days after the audio
  direction is confirmed.
- **Automatic dial/answer/hangup of the phone from this app:** not with public Windows
  APIs alone. The app can drive `tel:` and observe audio, but answering a real inbound
  cellular call requires Phone Link/TAPI integration or an Android companion. I will not
  promise otherwise.
- **Guaranteed real-call quality:** only after a hardware test with the phone. The
  current tests explicitly avoid physical Bluetooth, so they cannot prove it.

---

## 1. What the project is

A Windows 11 operator console for an AI call assistant:

- **Gemini 3.8 Live** (`gemini-3.8-live`, voice `Aoede`) handles the conversation
  audio-to-audio.
- A **Bluetooth HFP audio driver** (`WindowsAudioDriver`) selects the phone's Hands-Free
  endpoints and streams PCM.
- A **FastAPI server** exposes REST + WebSocket APIs and a single-page UI.
- **SQLite** stores calls, transcripts, messages, and summaries.
- **Tool calls** let Gemini take a message, transfer to a human, end the call, and log a
  summary.

### Intended Bluetooth flow

```
Phone (cellular call, HFP Audio Gateway)
        │  Bluetooth HFP link
        ▼
Windows 11 (HFP Hands-Free unit)
        │  "iQOO Z9 5G Hands-Free HF Audio" endpoints
        ▼
WindowsAudioDriver ── input callback ──► CallStateManager.feed_caller_audio()
        │                                      │
        │                                      ▼
        │                            GeminiLiveClient (16 kHz PCM)
        │                                      │
        │                          24 kHz PCM / tools / transcripts
        ▼                                      ▼
   output callback ◄── CallStateManager._on_assistant_audio()
        │
        ▼
Phone call audio path
```

Whether the last step actually reaches the remote caller depends on the HFP role and
routing. That is the key thing to verify.

### Repository layout

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point; defaults to `--driver virtual`, not `windows`. |
| `core/gemini_live.py` | Gemini Live session, audio streaming, tool dispatch, receive loop. |
| `core/tools.py` | Tool schemas + handler registry. |
| `core/audio_processor.py` | PCM resampling, RMS/peak/spectrum levels. |
| `core/config.py` | `AgentConfig` Pydantic model (currently unused). |
| `server/app.py` | FastAPI app, REST API, `/ws/live` hub, static UI. |
| `server/call_state.py` | Call state machine and orchestration. |
| `server/database.py` | SQLite schema + async CRUD. |
| `server/static/` | Operator console (`index.html`, `app.js`, `styles.css`). |
| `telephony/` | `base_driver`, `virtual`, `windows`, `linux` drivers. |
| `tests/` | 168 unit tests. |
| `e2e_tests/` | 128 e2e tests + a standalone mock harness + custom runner. |

---

## 2. What is already done (verified)

### 2.1 Core application

- ✅ Gemini Live connect/disconnect, 16 kHz PCM input, 24 kHz PCM output.
- ✅ Tool calling (`take_message`, `transfer_to_human`, `end_call`, `log_call_summary`)
  with structured responses back to Gemini.
- ✅ Barge-in callback flushes the local playback buffer.
- ✅ Operator whisper injection (`send_realtime_input(text=...)`).
- ✅ Full call state machine: `IDLE`, `DIALING`, `RINGING`, `ACTIVE`, `ON_HOLD`,
  `ON_HOLD_TRANSFER`, `OPERATOR_TALKING`, `COMPLETED`, `FAILED`, `DECLINED`.
- ✅ Inbound approval workflow (AI takeover / answer personally / decline).
- ✅ Hold, resume, and operator-pickup methods on `CallStateManager`.
- ✅ Audio telemetry at 25 Hz (caller + Gemini RMS/peak/spectrum).
- ✅ SQLite WAL schema: `calls`, `transcripts`, `messages`, `summaries` + indexes.
- ✅ REST endpoints for status, devices, dial, answer, hangup, inbound action, whisper,
  simulated inbound, call history/detail/delete, messages, and message updates.
- ✅ WebSocket hub with initial state, ping, whisper, inbound decision, and hangup.
- ✅ Windows driver with HFP regex discovery, sanitization, 16 kHz resampling, output
  queue, and `tel:` dial dispatch.
- ✅ Virtual driver for headless/simulated testing; Linux driver with oFono hooks.

### 2.2 UI

- ✅ Windows 11 dark console with connection/status badges.
- ✅ Audio device selectors (populated from `/api/devices`).
- ✅ Active-call monitor, caller identity, duration timer, answer/end buttons.
- ✅ Dual waveform visualizer driven by live telemetry.
- ✅ Streaming transcript panel with auto-scroll and turn counter.
- ✅ Operator whisper input.
- ✅ Outbound dialer with phone number and goal prompt.
- ✅ Messages / Call History tabs, inbound approval modal, toast notifications.
- ✅ Responsive layout and keyboard-accessible controls.

### 2.3 Tests and environment

- ✅ `python -m pytest tests -q` → **168 passed** (10.92 s, 1 deprecation warning).
- ✅ `python -m e2e_tests.runner --no-color` → **128 passed** in ~133 s, but the runner
  then crashes on a Unicode table (P2) and exits non-zero.
- ✅ `google-genai 2.8.0` installed; `gemini-3.8-live` and `Aoede` are valid per current
  Google docs.
- ✅ `GEMINI_API_KEY` is visible in this session's environment (53 chars).
- ✅ Bluetooth HFP endpoints exist for the `iQOO Z9 5G` (input index 56, output 55).
- ✅ Windows device `iQOO Z9 5G Hands-Free HF` has HFP hardware ID `0000111f` and device
  description “Microsoft Bluetooth Hands-Free Profile Handsfree role.”
- ✅ `BTAGService`, `BthHFEnum`, and `bthserv` services are running.
- ❌ No `.env` file, no `voice_caller.db`, no git repository, no README.

---

## 3. Transcription: what 3.8 Live actually requires

Gemini 3.8 Live is **audio-to-audio**; the conversation does **not** need transcription.
You are correct on that point. No separate speech-to-text service is involved.

However, the Live API only emits text transcript events when the setup config asks for
them. Google's current 3.8 Live documentation says:

- “To enable transcription of the model's audio output, send `output_audio_transcription`
  in the setup config.”
- “To enable transcription of the model's audio input, send `input_audio_transcription`
  in setup config.”
- “Audio is the only supported response modality. Enable `output_audio_transcription` if
  your application requires a text transcript.”

The current `LiveConnectConfig` (`core/gemini_live.py:108-113`) sets neither. The receive
loop only reads `server_content.model_turn.parts[].text` (`:278-296`), which native audio
models generally do not emit. So:

- **For the call itself:** transcription is **optional**. Downgrade it from P0.
- **For the transcript UI and `transcripts` table:** it is required. If you want caller
  and agent text, add the two flags and handle `input_transcription` /
  `output_transcription`.
- **Agent transcript without the flag:** unreliable; do not assume it arrives.

This is a product decision, not a call blocker.

---

## 4. What is missing or broken

### P0 — blocks a working Bluetooth call

#### P0-1. Caller audio is silently dropped in production

`server/call_state.py:114-118` captures `asyncio.get_running_loop()` in `__init__`.
`main.py` builds `CallStateManager` **before** uvicorn starts, so `self._loop` becomes
`None`.

`feed_caller_audio()` (`server/call_state.py:208-233`) is called from PortAudio callback
threads. `asyncio.get_running_loop()` raises `RuntimeError` there, and the fallback checks
`self._loop` — which is `None` — so the chunk is discarded without an error. Tests call
this method from async contexts, masking the bug.

**Fix:** bind the loop in a FastAPI startup handler (or an explicit `start()` method),
never in `__init__`; use a bounded thread-safe queue; add a test that calls
`feed_caller_audio()` from a real worker thread.

#### P0-2. The app does not actually dial, answer, or hang up the phone

`telephony/windows_driver.py:380-422` opens `tel:<number>` via `os.startfile` and then
immediately sets the driver to `ACTIVE`. `answer_call()` (`:424-427`) and
`hang_up()` (`:429-433`) only change `_call_state`.

There is:
- no inbound-call event from the phone,
- no way to answer or reject a real call,
- no remote-hangup detection,
- no guarantee the `tel:` handler routes audio to this app.

`CallStateManager.initiate_outbound()` (`server/call_state.py:443-533`) connects Gemini
and marks the call `ACTIVE` as soon as the `tel:` URL is opened — **before the callee
answers**. `approve_inbound_call()` (`:353-441`) calls `driver.answer_call()`, which does
not answer the phone. Inbound `RINGING` is only produced by
`POST /api/call/simulate-inbound` (`server/app.py:539-545`).

**Reality:** the user dials/answers on the phone or Phone Link. The app is an audio
bridge. If that is the intended workflow, document it and make the state transitions
follow the actual audio, not the `tel:` dispatch. If the app must control the phone, a
Phone Link/TAPI integration or Android companion is required.

#### P0-3. HFP audio direction is likely reversed for this role

The Windows device tree shows:

- `iQOO Z9 5G Hands-Free HF` — hardware ID
  `BTHENUM\{0000111f-...}` — device description
  **“Microsoft Bluetooth Hands-Free Profile Handsfree role.”**
- A headset example (`HD116 Hands-Free AG`) uses hardware ID `0000111e` and description
  “AudioGateway role.”

In HFP terms:
- `0x111F` is the **Audio Gateway (AG)** service, advertised by the phone.
- Windows enumerates the matching **Hands-Free (HF)** role.
- The “iQOO Hands-Free HF Audio” **input** endpoint is the **PC microphone** (the HF
  microphone, sent to the phone's uplink).
- The “iQOO Hands-Free HF Audio” **output** endpoint is the **PC speakers** (the HF
  speaker, receiving the phone's downlink).

The app assumes the opposite:
- `WindowsAudioDriver._in_stream_callback()` treats input as **caller audio** and sends
  it to Gemini (`telephony/windows_driver.py:188-214`).
- `play_audio_chunk()` queues Gemini audio to the **output** endpoint
  (`:339-365`), which is the local speaker, not the phone's uplink.

As coded, Gemini hears the local room/PC microphone and speaks to the local speakers.
The remote caller hears the local microphone, not Gemini. This is the most important
hardware finding and must be verified with one real test call.

**Possible fixes (choose after the test):**
1. **Loopback + virtual cable bridge:** capture the HFP output (remote caller) with
   WASAPI loopback; feed Gemini audio into the HFP input (uplink) through a virtual
   audio cable. Feasible in software, but requires a virtual cable and careful routing.
2. **Acoustic bridge:** put the phone on speakerphone near the PC; use the PC mic and
   speakers. Simple, but echo-prone and not hands-free.
3. **Phone Link/TAPI integration:** let Windows own the call and route audio; no public
   API for arbitrary apps. Research spike only.
4. **Android companion app:** run the assistant on the phone with `Telecom` /
   `ConnectionService`; most reliable for the actual SIM, but a separate app.
5. **Twilio/SIP:** reliable and testable, but not the requested Bluetooth path.

#### P0-4. The default driver is `virtual`, not `windows`

`main.py:46-50` defaults `--driver` to `virtual`. Running `python main.py` uses the
laptop microphone/speakers, not the phone. The Bluetooth path requires
`python main.py --driver windows`. The UI badge will show “Driver: Virtual” if this is
missed. The server factory `create_app()` defaults to `WindowsAudioDriver` on win32, so
`uvicorn server.app:app` behaves differently from `python main.py`. This inconsistency
should be fixed or documented.

#### P0-5. No Gemini reconnect; a dropped socket leaves a dead call

When the receive loop exits, `_is_connected` becomes `False` and `_session` becomes
`None` (`core/gemini_live.py:303-310`). Nothing notifies `CallStateManager`.
`handle_socket_disconnect()` exists at `server/call_state.py:687-690` but is **never
called**. The call stays `ACTIVE`, the UI stays green, and the caller hears silence.

**Fix:** add an `on_disconnected` callback, use `session_resumption`, reconnect with
bounded backoff, broadcast a `reconnecting` state, and terminate the call only after the
retry budget is exhausted.

### P1 — blocks a usable product

- **Hold/resume/pickup are unreachable from the UI/API.** Methods exist
  (`server/call_state.py:633-675`) but there are no REST or WebSocket commands in
  `server/app.py`, and no buttons in `index.html`/`app.js`.
- **`operator_pickup()` does not mute or disconnect Gemini.** It sets
  `OPERATOR_TALKING` and stops hold audio, but Gemini may still be generating audio; the
  playback buffer is not cleared. The caller can hear the AI and the operator at once.
- **Transfer-to-human only emits a WebSocket event.** It sets `ON_HOLD`, plays hold
  audio, and broadcasts, but there is no operator notification/ring, no pickup endpoint,
  and no escalation timeout.
- **Telemetry stops after hold.** `_telemetry_loop()` exits when state leaves `ACTIVE`
  (`server/call_state.py:180`); `resume_call()` does not restart it.
- **Device selectors are decorative.** `loadDevices()` populates the selects
  (`server/static/app.js:1195-1223`) but no change handler applies the choice; there is
  no reconfiguration endpoint. `--input-device`/`--output-device` are only read at
  startup.
- **No call-detail / transcript viewer.** The API supports `GET /api/history/calls/{id}`,
  but the UI only renders a flat history table.
- **Message center is read-only.** `PATCH /api/history/messages/{id}` exists but the UI
  has no mark-read or action-taken controls.
- **No delete UI** for calls or messages despite the APIs.
- **`AgentConfig` is dead code.** It is defined in `core/config.py` but never imported by
  `main.py`, the server, or the drivers. `inbound_policy` (`manual_approval` /
  `auto_answer`) and `auto_answer_delay_seconds` are not implemented anywhere.
- **No authentication.** The server binds `0.0.0.0` by default (`main.py:60`), CORS is
  `allow_origins=["*"]` with `allow_credentials=True` (`server/app.py:375-379`), and
  there is no token check on REST or WebSocket.
- **Relative DB path.** `DatabaseManager()` defaults to `voice_caller.db` relative to the
  process working directory; launching from another directory silently creates a new DB.
- **No migrations.** Schema changes require manual DB deletion.
- **Gemini connect failure leaves inconsistent state.** If `connect()` raises after the
  driver dials, `initiate_outbound()` propagates the exception and the call stays
  `DIALING` with a DB row and no cleanup.
- **Remote hangup is never detected** on Windows/Linux; calls end only via operator or
  Gemini tool call.
- **Driver streams stay open across calls.** `terminate_call()` calls `driver.hang_up()`,
  which flushes buffers but does not close/reopen the PortAudio streams. After the phone
  releases the HFP link, the next call may need a restart.
- **No HFP sample-rate capability check.** The driver assumes 16 kHz input/output. If the
  phone/link only supports 8 kHz narrowband, `start()` fails and the app does not recover.
- **`server/app.py:684` creates a default app at import time**, instantiating a DB and
  driver as a side effect.

### P2 — hygiene and polish

- **E2E runner crashes on Windows cp1252** (`e2e_tests/runner.py:172-184` uses
  box-drawing characters). All tests pass, then `UnicodeEncodeError` makes the process
  exit 1.
- **`requirements.txt` is missing test dependencies**: `pytest`, `pytest-asyncio`,
  `httpx` (and `pytest-cov` if used). No `requirements-dev.txt`.
- **No `.env.example`, `.gitignore`, README, architecture doc, or license.**
- **No git repository**, so there is no history, diff review, or rollback.
- **`qtriggerinfo`** is a 13-byte Bluetooth diagnostic artifact containing `BTAGService`.
  Leave it alone unless you know why it was created; do not delete user artifacts.
- **No linting/type checking/CI** (ruff, mypy, pytest workflow).
- **Hard-coded device heuristic** `iQOO.*Hands-Free` in `server/app.py:445-450`; should be
  generic HFP detection or a user preference.
- **Dependency versions are lower bounds only**; no lockfile.
- **No structured logging or log rotation.**
- **No rate limiting** on dial/whisper endpoints.

---

## 5. What was actually verified vs. not verified

| Claim | Evidence |
|---|---|
| Unit suite passes | `python -m pytest tests -q` → 168 passed, 10.92 s |
| E2E suite passes | 128 passed in ~133 s; runner then crashes on Unicode |
| Model/voice are valid | Google Live API docs + installed SDK; `gemini-3.8-live`, `Aoede` |
| 3.8 Live is audio-native | Model docs; audio-only response modality; text needs transcription flags |
| SDK supports transcription/resumption | `LiveConnectConfig` fields confirmed in `google-genai 2.8.0` |
| API key is available | Env var present, 53 chars (value not printed) |
| iQOO HFP endpoints exist | `sounddevice` lists input 56 / output 55 |
| Phone is HFP AG, Windows is HF | Device description “Handsfree role”; hardware ID `0000111f` |
| Live Gemini conversation works | **Not tested** — would consume API quota; needs your OK |
| Bluetooth two-way call audio works | **Not tested** — requires a real call and the direction test in §6 |
| Production audio path works | **Not proven** — see P0-1 and the e2e mock caveat |
| UI renders correctly | Static inspection + frontend asset tests; not visually tested in a browser |

> Important: `e2e_tests/conftest.py` defines its own `DatabaseManager` (line 744),
> `CallStateManager` (line 949), `create_app` (line 1319), and `live_server` (line 1451).
> Its header says it runs “without requiring physical Bluetooth hardware.” The
> `live_server` / `call_manager` / `temp_db` fixtures use these local classes, not
> `server.*`. The Windows driver tests only check device-name sanitization, HFP regex
> discovery, `tel:` dispatch, state flips, and resampling. They do not test real HFP
> audio or call control.

---

## 6. The one hardware test that decides the architecture

Before writing more code, run a 2-minute call with the phone connected:

1. Start the app with `python main.py --driver windows`.
2. Call a friend/second phone. Answer on the phone (or Phone Link).
3. In the app, use the whisper input or a test tone; ask the remote person what they hear.
4. Record 5 seconds of the app's `feed_caller_audio` input; inspect whether it contains
   the remote caller's voice or the local room.
5. Play a 1 kHz tone through the app's output; ask the remote person whether they hear it
   or whether it comes out of the PC speakers.

Interpretation:

- **Remote person hears the PC speakers / app input contains the local room:** the HFP
  direction is reversed. The app needs a loopback/virtual-cable bridge or a different
  transport.
- **Remote person hears the AI / app input contains the caller:** the direction matches
  the code; proceed with the audio fixes and call-control workflow.
- **No audio at all:** the HFP link is not carrying the call, or the endpoints are held by
  Phone Link. Try Phone Link or a different pairing role.

This test is the difference between a software fix and an architecture change. It cannot
be replaced by unit tests.

---

## 7. Delivery plan

### Phase 0 — Baseline and guardrails (0.5–1 day)

- Fix `e2e_tests/runner.py` encoding; ensure exit code 0.
- Add `requirements-dev.txt` (`pytest`, `pytest-asyncio`, `httpx`, `pytest-cov`).
- Add `.env.example`, `.gitignore`, README; initialize git.
- Add `pytest.ini`/`pyproject.toml` config and a smoke CI workflow.
- **Do not delete `qtriggerinfo`**; document it as a Bluetooth diagnostic artifact.

**Acceptance:** fresh clone → install → unit + e2e pass → runner exits 0 on Windows.

### Phase 1 — Fix the production audio path (2–3 days)

- Replace `_loop` capture with an explicit `bind_loop()` called on FastAPI startup.
- Add a bounded thread-safe audio queue with drop-oldest backpressure.
- Make `--driver windows` the default on Windows (or prompt/validate the driver at
  startup) so the Bluetooth path is actually used.
- Verify/choose HFP sample rates; add an 8 kHz narrowband fallback.
- Add a diagnostic endpoint/CLI to show selected endpoints, role, sample rate, and
  whether input is receiving signal.
- Add a real-thread test for `feed_caller_audio()`.

**Acceptance:** a scripted WAV fed from a worker thread reaches a mock Gemini session;
the app reports the correct HFP endpoints and rates; the loop bug is covered by a
regression test.

### Phase 2 — Prove and implement the Bluetooth audio bridge (3–5 days, depends on §6)

- Run the §6 test; record the result.
- If the direction is reversed, implement the chosen bridge:
  - WASAPI loopback capture of the HFP render endpoint for caller audio.
  - Virtual audio cable (or equivalent) for AI audio into the HFP capture endpoint.
  - Or the acoustic speakerphone path with AEC if that is acceptable.
- If the direction matches, fix the driver mapping and level handling.
- Add remote-hangup detection (stream state / sustained silence / Phone Link event) and
  stop the Gemini session.
- Add echo/feedback protection and a barge-in test.

**Acceptance:** on a real call, the remote person hears Gemini and Gemini hears the
remote person; the app detects hangup within 2 s; no self-interruption loop.

### Phase 3 — Call-control workflow (2–3 days)

- Decide and document the workflow:
  - **Manual:** user dials/answers on the phone; app starts Gemini when audio is present.
  - **Assisted:** app opens `tel:` and waits for the user to confirm answer; no premature
    greeting.
  - **Automatic:** Phone Link/TAPI or Android companion (separate project).
- Fix outbound state: do not mark `ACTIVE` or connect Gemini until audio flows/answer is
  confirmed.
- Fix inbound state: remove/rename `simulate-inbound` as a demo; add a real event only if
  the chosen integration provides one.
- Add hold/resume/pickup REST + WS endpoints and UI buttons.
- Ensure operator pickup mutes/disconnects Gemini and clears playback.
- Restart telemetry after resume.

**Acceptance:** every state is reachable from the UI; no dead ends; no overlapping
AI/operator audio; the documented workflow matches what the app actually does.

### Phase 4 — Operator experience (1–2 days)

- Runtime device selection API + persistence.
- Call-detail drawer: transcript, summary, messages, delete.
- Message center: mark read, action taken, unread badge.
- Reload active call state and transcripts after a browser reconnect.
- Keep the UI honest: label “simulated” calls and show the driver/role.

**Acceptance:** call history and messages are usable without the API; device changes apply
without restart.

### Phase 5 — Optional transcription (0.5–1 day)

- Add `input_audio_transcription` and `output_audio_transcription` if transcripts are
  required.
- Handle `input_transcription` / `output_transcription`; persist final segments.
- If transcripts are not required, remove the transcript UI/DB promises or mark them
  optional.

**Acceptance:** caller and agent text appear live and persist; no extra STT service.

### Phase 6 — Security and docs (1–2 days)

- Default bind to `127.0.0.1`; token auth for REST/WS; restrict CORS.
- Absolute DB path; migrations.
- Validate/sanitize phone numbers and whisper/dial payloads; rate limits.
- Document Bluetooth pairing, Phone Link role, `--driver windows`, and troubleshooting.
- Add a real-call checklist and latency measurement.

**Acceptance:** unauthenticated LAN clients cannot read transcripts or place calls; a new
user can reproduce the setup from the README.

### Estimate

| Scope | Estimate |
|---|---|
| Phase 0 (hygiene) | 0.5–1 day |
| Phase 1 (audio path) | 2–3 days |
| Phase 2 (Bluetooth bridge, if needed) | 3–5 days |
| Phase 3 (call workflow/UI state) | 2–3 days |
| Phase 4 (operator UX) | 1–2 days |
| Phase 5 (optional transcription) | 0.5–1 day |
| Phase 6 (security/docs) | 1–2 days |
| **Total, Bluetooth path** | **10–17 focused days** |

The range is wide because the HFP direction test can turn Phase 2 into a small mapping
fix or a loopback/virtual-cable build.

---

## 8. Definition of done

### Application-side done

- [ ] Caller audio from the PortAudio thread reaches Gemini (regression test).
- [ ] The app starts with the Windows driver by default (or refuses to start with a clear
      message).
- [ ] Selected HFP endpoints, role, sample rate, and input signal are visible in the UI.
- [ ] Remote hangup is detected; the call ends and the DB record is complete.
- [ ] Hold, resume, pickup, and transfer work from the UI.
- [ ] Device selection applies at runtime.
- [ ] Call history detail, transcripts (if enabled), messages, and delete work.
- [ ] No unauthenticated access; localhost bind by default.
- [ ] Unit + e2e suites pass from a clean checkout; runner exits 0.

### Real Bluetooth call done

- [ ] Outbound: dial on the phone/app → answer → two-way audio → AI greeting → hangup.
- [ ] Inbound: call arrives → app shows the correct state (manual or automatic) → AI
      takeover works.
- [ ] The remote caller hears Gemini; Gemini hears the remote caller.
- [ ] 10 consecutive calls with no dead audio, no self-interruption, and no orphaned
      streams.
- [ ] End-to-end latency measured and acceptable (target < 1.5 s p50).
- [ ] The documented workflow matches the actual phone/Phone Link behavior.

---

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| HFP audio direction is reversed for the phone's AG role | High | Run §6 test first; implement loopback/virtual cable if needed |
| Windows Phone Link holds the HFP link and blocks the app | Medium | Test with Phone Link open/closed; document the required order |
| Phone only supports 8 kHz narrowband | Medium | Detect capabilities; resample to 16 kHz for Gemini; accept lower quality |
| Echo/self-interruption on speakerphone | High if acoustic | Headset/AEC; half-duplex fallback; barge-in test |
| Gemini Live latency or quota surprises | Medium | Measure early; set call duration/token budget |
| E2E mocks hide production regressions | High (current) | Rewrite server fixtures against production `create_app` |
| Windows audio device indices change between reboots | Medium | Resolve by name, persist preference, re-resolve on start |
| No public API to answer the phone | High | Manual workflow or Android companion; do not promise automation |
| Secret leakage in logs/UI | Low | Never log keys; `.env` + `.gitignore`; redact config output |

---

## 10. Decisions needed

1. **Call workflow:** do you dial/answer on the phone manually, or must the app control
   the call? If the latter, is an Android companion acceptable?
2. **Audio direction test:** can you run the §6 test call and report what the remote
   person hears?
3. **Transcripts:** required, or is audio-only acceptable? If required, enable the two
   Live API flags.
4. **UI priority:** hold/resume/pickup and call detail first, or message-center actions?
5. **Live Gemini smoke test:** may I run a small live test with the key in the
   environment? It consumes a small amount of quota.

---

## 11. Honest closing assessment

You were right about two things: 3.8 Live does not need a separate transcription service,
and the intended transport is the phone over Bluetooth. I corrected both in this plan.

The artifacts also show two things that are easy to miss:

- The app **does not make or answer the call**; it opens `tel:` and flips local state.
  The phone (or Phone Link) does the telephony. That is a valid design if documented, but
  it is not "the app controls the phone."
- The phone is the HFP **Audio Gateway** and Windows is the **Hands-Free** unit. In that
  role, the endpoints the app selects are the PC's microphone and speakers, which is the
  opposite of the app's input=caller / output=AI assumption. This is the likely reason a
  real call would not behave as intended.

I can finish the software around the Bluetooth path: fix the event-loop bug, make the
Windows driver the default, add diagnostics, implement the loopback/virtual-cable bridge if
the §6 test confirms the reversal, add hold/resume/pickup and call detail, and make the
tests exercise the production app. I cannot honestly promise automatic inbound/outbound
call control on Windows without Phone Link/TAPI integration or an Android companion, and I
will not claim the Bluetooth audio works until the test call proves the direction.
