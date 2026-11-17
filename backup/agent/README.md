# `agent/` — the missing link between the phone and Pipecat

Short version: **there is no Android-side software in this design.** Pipecat never
talks to Android. It talks to a **Windows audio device**, and ADB presses the
phone's buttons. Once you see the two legs separately, the project stops being
mysterious.

```
                 audio plane (the phone is a headset)
  phone USB-C → 3.5 mm (this phone has no jack)  ──►  Windows sound card
                                                     │
                                          pipecat LocalAudioTransport
                                                     │
                                             GeminiLiveLLMService

                 control plane (the phone is a peripheral)
  agent/phone.py  ── ADB ──►  dial / answer (key 79) / hang up (key 6) / poll state
```

No SIP. No Asterisk. No server. No Android app.

## Why people get stuck here

Pipecat's transports are chosen by **where the audio comes from**, not by what
device is on the other end:

| If the audio arrives as… | Pipecat transport | What the phone must do |
|---|---|---|
| a sound card / headset jack / Bluetooth HFP | `LocalAudioTransport` | expose a headset profile (it does) |
| a SIP call | `SIPTransport` (+ registrar) | be a SIP gateway (needs root) |
| a provider WebSocket | Twilio/Exotel serializer | not be involved at all |

For this phone, row 1 is the only row that exists. `LocalAudioTransport` is the
link — it is part of Pipecat, not something you have to build.

## The index trap

`LocalAudioTransport` enumerates devices through **PyAudio/PortAudio**, while the
rest of this project uses **sounddevice** — which is also a PortAudio wrapper, so
right now the numbers agree:

```
42 = Output (… Hands-Free HF Audio (iQOO Z9 5G))
43 = Input  (… Hands-Free HF Audio (iQOO Z9 5G))
```

They agree today, not forever. A device index is a position in a list, and this
machine enumerates 60+ endpoints. Connect one pair of earbuds and the Z9 pair
moves. Hardcoding `42`/`43` is how you end up opening somebody's AirPods, getting
a clean pipeline, and hearing silence. `agent/audio_devices.py` resolves by
**name**, so it follows the phone when the list shifts.

This is not hypothetical: on this machine the pair moved from `42/43 @ 16 kHz` to
`29/30 @ 8 kHz` across a Bluetooth reconnect and a Phone Link Calls toggle.

Two consequences of how Windows exposes this phone:

- The Z9's hands-free pair appears **only under WDM-KS**, with no MME or WASAPI
  duplicate, at **8 or 16 kHz** depending on the negotiated HFP codec (CVSD vs
  mSBC). The tools read the device's rate automatically — never hardcode it.
- WDM-KS is kernel streaming (effectively exclusive). If another app holds the
  endpoint, the open fails with `Device unavailable`. Close anything using the
  headset, or use the wired USB card instead.

## Setup

```powershell
cd C:\Users\amarn\Desktop\Ai_voice_caller
python -m venv .venv-agent
.venv-agent\Scripts\python -m pip install -r agent\requirements.txt
```

`pipecat-ai[google,local]` installs the Gemini Live service and PyAudio.

List devices **as Pipecat sees them**:

```powershell
.venv-agent\Scripts\python -m agent.run_call --list-devices
```

## Run

Incoming call, answer it and let the agent talk:

```powershell
.venv-agent\Scripts\python -m agent.run_call `
  --input-device  "iQOO Z9 5G" `
  --output-device "iQOO Z9 5G" `
  --answer --greet
```

Outbound call from SIM 1:

```powershell
.venv-agent\Scripts\python -m agent.run_call `
  --input-device  "iQOO Z9 5G" `
  --output-device "iQOO Z9 5G" `
  --dial 9133233391 --sim-slot 1 --greet
```

Audio-only bridge (you drive the phone yourself, the bot just listens and talks):

```powershell
.venv-agent\Scripts\python -m agent.run_call --input-device "Line In" --output-device "Speakers"
```

For a USB sound card, match on the card's name (for example `"USB Audio"` or
`"Line In"`) and leave the rates at `16000`/`24000` if the card is a normal
48 kHz device.

## Phone Link: disable Calls only

Phone Link's features are independent toggles. Turn off **only Calls** and
everything else keeps working — notifications, messages, photos, apps, clipboard,
instant hotspot, and the Bluetooth link itself.

**Phone Link → Settings (gear) → Features → Calls → off.**

That is the whole bypass. Phone Link stops answering, placing and routing calls,
while the Bluetooth stack still exposes the hands-free endpoints for Pipecat. No
unlinking, no uninstalling, no separating the phone from the PC.

If your build hides the Calls toggle, the phone-side equivalent is **Settings →
Connected devices → Link to Windows → calling permission off**. Do not unlink the
device; removing the link disables notifications and messages too.

Do *not* use the global kill switches (Mobile devices → *Allow this PC to access
your mobile devices*, or removing the `CrossDevice`/`YourPhone` packages) unless
you actually want the phone and PC fully separated.

Signs of the problem on this machine, for reference:

```
PhoneExperienceHost, CrossDeviceService, CrossDeviceResume   (running)
Microsoft.YourPhone 1.26092.119.0
MicrosoftWindows.CrossDevice 1.26101.77.0
```

### Order matters more than the toggle

Both hands-free endpoints fail to open with PortAudio `-9999` while no call is
active — verified on this machine. They are only usable once the phone has a live
call and has routed it to the headset.

So `run_call.py` now places/answers the call **first**, waits `--audio-delay`
(default 1.5 s) for SCO, and only then starts the pipeline and opens the streams.
Starting the pipeline before the call will fail to open Bluetooth audio. This is
also why turning the Calls toggle off is sufficient: Phone Link is not holding a
session before a call exists.

## Order of testing

1. `--list-devices` and confirm the phone's HFP entry shows `in 1 / out 1`.
2. `--answer` on a real incoming call. Watch the log for `caller:` lines. If
   they never appear, the **input** leg is wrong — try `--host-api WDM-KS`.
3. Confirm the caller hears the agent. If not, the **output** leg is wrong.
4. Only then add `--greet` and tune `--dial-grace`.

Prove the endpoint carries audio during a live call:

```powershell
python -m agent.probe_audio --seconds 6 --tone
```

The caller should hear a beep and the level line should be above -60 dBFS. A
`-9999`/`-9996` here means no call is active or something still owns the device.

## Known limits

- **Outbound answer detection** is not available to unprivileged apps. The bot
  waits `--dial-grace` seconds after dialing, then speaks. Tune it to the person
  you are calling; do not treat it as a real answered-event.
- **HFP is 8 or 16 kHz mono.** Windows exposes the Z9's hands-free endpoints at
  16 kHz, which is why `--rate-in/--rate-out 16000` is the right pair.
- **Bluetooth echo.** HFP has no echo cancellation worth the name; the phone's
  own DSP does some. The wired TRRS hybrid removes the problem entirely.
- **`KEYCODE_HEADSETHOOK` needs a headset present.** With nothing plugged in,
  the code falls back to `KEYCODE_CALL` for incoming calls; otherwise the
  dialer's own UI is the only thing that can answer.
