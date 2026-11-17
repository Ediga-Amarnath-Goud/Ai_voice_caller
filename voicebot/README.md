# voicebot (fresh build)

Clean reimplementation of the Phone Link + VB-CABLE voice agent.

## Audio path

```
caller -> Phone Link -> Realtek speakers -> WASAPI loopback -> Gemini Live
Gemini -> CABLE Input -> CABLE Output -> Windows default mic -> Phone Link
```

No DSP: the caller's voice is passed through unchanged. Language is enforced in
the system prompt (Telugu / Hindi / Kannada / English only). Call detection is
debounced (edge-triggered) so it no longer chatters into duplicate greetings.

## Run

The system `python` is used with the venv site-packages on `PYTHONPATH` (same as
the legacy build):

```powershell
$env:PYTHONPATH="C:\Users\amarn\Desktop\Ai_voice_caller\.venv-agent\Lib\site-packages"

# list audio devices
python -m voicebot.main --list-devices

# daemon: answer every call detected on the loopback
python -m voicebot.main --bridge --greet

# place a call yourself, then run with audio already routed
python -m voicebot.main --greet --no-audio-switch
```

Flags: `--greet` speaks first, `--no-daemon` exits after one call,
`--dial <number>` / `--answer` drive the phone over ADB (control only),
`--no-audio-switch` leaves the current mic routing untouched.

## Notes

- 24 kHz loopback input, MME CABLE Input (44.1 kHz) output.
- Speakers are never muted — the loopback taps pre-mute audio.
- Legacy code and all diagnostics are preserved in `../backup/`.
