# AI Voice Caller (Phone Link + Gemini Live 3.8 Bridge)

Autonomous real-time AI voice calling agent bridging **Microsoft Phone Link** on Windows 11 with **Pipecat** and **Google Gemini Live 3.8 (`models/gemini-3.8-live`, persona `Aoede`)** over cellular phone calls via Bluetooth Hands-Free (HFP).

---

## Architecture

A purely software-driven audio bridge with zero specialized telephony hardware:

```
[ Caller on Phone ]
       │ (Cellular VoLTE / SIM 2)
[ Microsoft Phone Link ]
       │ (Renders into Windows Default Speakers)
[ Speakers (Realtek(R) Audio) ]
       │ (24 kHz WASAPI Loopback)
[ voicebot.loopback ]
       │ (InputAudioRawFrame)
       ▼
[ Pipecat Pipeline (Realtime Mode) ]
       │
       ▼
[ Gemini Live 3.8 (Aoede Persona) ]
       │ (24 kHz Audio Stream)
       ▼
[ PyAudio Output Transport (MME, 44.1 kHz) ]
       │
       ▼
[ CABLE Input (VB-Audio Virtual Cable) ]
       │ (Internal loop)
[ CABLE Output (Windows Default Microphone) ]
       │
       ▼
[ Phone Link Microphone ] ──(Bluetooth HFP)──► [ Caller Handset ]
```

---

## Key Modules (`voicebot/`)

- `voicebot/pipeline.py`: Speech-to-speech pipeline managing Pipecat, Universal Aggregators, and WebSocket connection to Gemini Live 3.8.
- `voicebot/loopback.py`: Non-blocking 24 kHz WASAPI loopback capture on `Speakers (Realtek(R) Audio)` with debounced call activity detection.
- `voicebot/routing.py`: Scoped Windows audio device router pointing console & communications capture to `CABLE Output` using SoundVolumeCommandLine.
- `voicebot/main.py`: Pre-warmed daemon calling loop with ADB telephony state synchronization.
- `voicebot/watcher.py`: Real-time bug monitor tracking audio peak/RMS levels, routing drift, and live transcripts.
- `voicebot/phone.py`: Wireless ADB handset driver for call state monitoring, dialing, and answering on the physical phone.

---

## Prerequisites

1. **Windows 11** with **Microsoft Phone Link** paired to an Android phone over Bluetooth with calls enabled.
2. **VB-Audio Virtual Cable** installed.
3. **Python 3.11+**
4. **Google Gemini API Key** set in environment:
   ```powershell
   $env:GEMINI_API_KEY="your-api-key"
   ```

---

## Installation

```powershell
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Voicebot Daemon (Auto-answers & Greets)
```powershell
python -m voicebot.main --greet
```

### 2. Run the Real-Time Quality & Bug Watcher
In a second terminal window:
```powershell
python -m voicebot.watcher
```

### 3. Outbound Call via ADB
```powershell
python -m voicebot.main --dial <phone_number> --sim-slot 2 --greet
```
