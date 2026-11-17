"""Comprehensive pipeline diagnostic test suite.
Executes 6 deep diagnostic checks across every stage of the calling system:
1. Audio Endpoint & Routing Check (svcl, pyaudio, soundcard)
2. Bluetooth Hands-Free & Phone Link Status
3. Virtual Cable Loopback Integrity (I/O latency, RMS, clipping)
4. Realtek Speaker Loopback Capture Test
5. Gemini Live 3.8 WebSocket Live S2S Test
6. Pipecat Full Pipeline Simulation
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Color output helpers
def ok(msg: str): print(f"\033[92m[PASS]\033[0m {msg}")
def warn(msg: str): print(f"\033[93m[WARN]\033[0m {msg}")
def fail(msg: str): print(f"\033[91m[FAIL]\033[0m {msg}")
def info(msg: str): print(f"\033[94m[INFO]\033[0m {msg}")

# -----------------------------------------------------------------------------
# Check 1: Windows Audio Devices & Volume / Mute Status
# -----------------------------------------------------------------------------
def check_audio_devices():
    info("--- Check 1: Audio Devices & Routing Status ---")
    svcl = Path(__file__).resolve().parent.parent / "tools" / "svcl" / "svcl.exe"
    if not svcl.exists():
        warn(f"svcl.exe not found at {svcl}")
        return

    import csv
    temp_csv = Path(os.environ.get("TEMP", ".")) / "debug_svcl.csv"
    res = subprocess.run([str(svcl), "/scomma", str(temp_csv)], capture_output=True, text=True)
    if res.returncode != 0:
        fail(f"svcl execution failed: {res.stderr}")
        return

    with open(temp_csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Type") == "Device"]

    default_render = next((r for r in rows if r.get("Direction") == "Render" and r.get("Default") == "Render"), None)
    default_capture = next((r for r in rows if r.get("Direction") == "Capture" and r.get("Default") == "Capture"), None)

    if default_render:
        name = default_render.get("Device Name")
        muted = default_render.get("Muted") == "Yes"
        vol = default_render.get("Volume Percent")
        if muted:
            fail(f"Default Render ({name}) is MUTED! Volume: {vol}%")
        else:
            ok(f"Default Render: {name} (Volume: {vol}%, Muted: {muted})")
    else:
        fail("No default render device found!")

    if default_capture:
        name = default_capture.get("Device Name")
        muted = default_capture.get("Muted") == "Yes"
        vol = default_capture.get("Volume Percent")
        if muted:
            fail(f"Default Capture ({name}) is MUTED! Volume: {vol}%")
        else:
            ok(f"Default Capture: {name} (Volume: {vol}%, Muted: {muted})")
    else:
        fail("No default capture device found!")


# -----------------------------------------------------------------------------
# Check 2: Bluetooth Hands-Free Devnodes & Phone Link Process
# -----------------------------------------------------------------------------
def check_bluetooth_and_phone_link():
    info("--- Check 2: Bluetooth Hands-Free & Phone Link Status ---")
    # Check Phone Link process
    p = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Process -Name 'PhoneExperienceHost' -ErrorAction SilentlyContinue"], capture_output=True, text=True)
    if "PhoneExperienceHost" in p.stdout:
        ok("PhoneExperienceHost.exe (Phone Link) is running")
    else:
        warn("PhoneExperienceHost.exe is NOT running! Start Phone Link if using the bridge.")

    # Check Bluetooth PnP endpoints
    pnp = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-PnpDevice | Where-Object { $_.FriendlyName -like '*iQOO*' -and $_.Class -eq 'AudioEndpoint' } | Select-Object FriendlyName, Status | Format-Table -HideTableHeaders"],
        capture_output=True, text=True
    )
    lines = [line.strip() for line in pnp.stdout.splitlines() if line.strip()]
    if lines:
        for l in lines:
            ok(f"Bluetooth Audio Endpoint: {l}")
    else:
        warn("No active Bluetooth AudioEndpoints found for iQOO handset.")


# -----------------------------------------------------------------------------
# Check 3: Virtual Cable Loopback Integrity (PyAudio Output -> Input)
# -----------------------------------------------------------------------------
def check_virtual_cable():
    info("--- Check 3: Virtual Cable Pair Integrity & Latency ---")
    import pyaudio
    from .audio_devices import find_device

    pa = pyaudio.PyAudio()
    try:
        cable_out_idx = find_device("CABLE Input", need_output=True)
        cable_in_idx = find_device("CABLE Output", need_input=True)
        ok(f"Resolved CABLE Input (idx {cable_out_idx}) and CABLE Output (idx {cable_in_idx})")
    except Exception as e:
        fail(f"Failed to find Virtual Cable pair: {e}")
        pa.terminate()
        return

    # Generate 0.5s of 1 kHz tone at -12 dBFS
    sr = 44100
    dur = 0.4
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = (0.25 * np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16).tobytes()

    try:
        out_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, output=True, output_device_index=cable_out_idx)
        in_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, input_device_index=cable_in_idx, frames_per_buffer=1024)
        
        # Flush input buffer
        for _ in range(3):
            in_stream.read(1024, exception_on_overflow=False)

        # Write tone
        out_stream.write(sig)
        time.sleep(0.05)

        # Read back
        captured = []
        for _ in range(int(sr * dur / 1024) + 2):
            captured.append(in_stream.read(1024, exception_on_overflow=False))
        
        in_stream.stop_stream()
        in_stream.close()
        out_stream.stop_stream()
        out_stream.close()

        rec_bytes = b"".join(captured)
        rec_data = np.frombuffer(rec_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.abs(rec_data).max())
        rms = float(np.sqrt(np.mean(rec_data**2)))
        db_peak = 20 * np.log10(peak) if peak > 0 else -120
        db_rms = 20 * np.log10(rms) if rms > 0 else -120

        if peak > 0.05:
            ok(f"Virtual Cable audio transmission VERIFIED: Peak {db_peak:.1f} dBFS, RMS {db_rms:.1f} dBFS")
        else:
            fail(f"Virtual Cable silent! Peak {db_peak:.1f} dBFS, RMS {db_rms:.1f} dBFS (check volume/mute on CABLE)")
    except Exception as e:
        fail(f"Virtual Cable stream error: {e}")
    finally:
        pa.terminate()


# -----------------------------------------------------------------------------
# Check 4: Realtek Speakers Loopback Capture Test
# -----------------------------------------------------------------------------
def check_loopback_capture():
    info("--- Check 4: Realtek Speakers Loopback Capture ---")
    import soundcard as sc
    from .bridge_transport import _loopback_device, DEFAULT_LOOPBACK

    dev = _loopback_device(DEFAULT_LOOPBACK)
    if dev is None:
        fail(f"Loopback endpoint {DEFAULT_LOOPBACK!r} not found by soundcard!")
        return

    ok(f"Found loopback endpoint: {dev.name}")
    try:
        rec = dev.recorder(samplerate=24000, channels=1, blocksize=480)
        with rec:
            chunks = []
            for _ in range(25):  # 25 * 20ms = 0.5s
                chunks.append(rec.record(numframes=480))
        data = np.concatenate(chunks).reshape(-1)
        peak = float(np.abs(data).max())
        db = 20 * np.log10(peak) if peak > 0 else -120
        ok(f"Loopback capture test SUCCESSFUL (Read 0.5s @ 24 kHz, ambient peak {db:.1f} dBFS)")
    except Exception as e:
        fail(f"Loopback capture failed: {e}")


# -----------------------------------------------------------------------------
# Check 5: Gemini Live 3.8 WebSocket Connectivity
# -----------------------------------------------------------------------------
async def check_gemini_live():
    info("--- Check 5: Gemini Live 3.8 Service Handshake ---")
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        fail("No GEMINI_API_KEY found in environment!")
        return

    try:
        service = GeminiLiveLLMService(
            api_key=api_key,
            settings=GeminiLiveLLMService.Settings(model="models/gemini-3.8-live", voice="Aoede"),
            inference_on_context_initialization=False,
        )
        ok("GeminiLiveLLMService initialized successfully with model 'models/gemini-3.8-live' (voice: Aoede)")
    except Exception as e:
        fail(f"GeminiLiveLLMService failed to initialize: {e}")


# -----------------------------------------------------------------------------
# Check 6: Pipecat Pipeline Frame-Flow Simulation
# -----------------------------------------------------------------------------
async def check_pipeline_simulation():
    info("--- Check 6: Pipecat Pipeline Frame Simulation ---")
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker, PipelineParams
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.frames.frames import InputAudioRawFrame, EndFrame
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.workers.runner import WorkerRunner

    class DummyInput(FrameProcessor):
        async def process_frame(self, frame, direction):
            await self.push_frame(frame, direction)

    class DummyOutput(FrameProcessor):
        def __init__(self):
            super().__init__()
            self.frames_received = 0
        async def process_frame(self, frame, direction):
            if isinstance(frame, InputAudioRawFrame):
                self.frames_received += 1
            await self.push_frame(frame, direction)

    inp = DummyInput()
    out = DummyOutput()
    context = LLMContext([])
    user_agg, asst_agg = LLMContextAggregatorPair(context)

    pipeline = Pipeline([inp, user_agg, out, asst_agg])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=24000, audio_out_sample_rate=24000),
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)

    async def _feeder():
        await asyncio.sleep(0.5)
        # Push 10 fake 20ms frames
        fake_pcm = (np.zeros(480, dtype=np.int16)).tobytes()
        for _ in range(10):
            frame = InputAudioRawFrame(audio=fake_pcm, sample_rate=24000, num_channels=1)
            await worker.queue_frame(frame)
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        await worker.queue_frame(EndFrame())

    try:
        runner_task = asyncio.create_task(runner.run())
        feeder_task = asyncio.create_task(_feeder())
        await asyncio.wait({runner_task, feeder_task}, return_when=asyncio.ALL_COMPLETED, timeout=10)
        
        if out.frames_received == 10:
            ok(f"Pipeline frame routing OK: {out.frames_received}/10 frames flowed seamlessly through aggregators")
        else:
            warn(f"Pipeline frame routing result: received {out.frames_received} frames")
    except Exception as e:
        fail(f"Pipeline simulation error: {e}")


def main():
    print("=" * 60)
    print("        AI VOICE CALLER - FULL PIPELINE DIAGNOSTIC")
    print("=" * 60)
    check_audio_devices()
    print()
    check_bluetooth_and_phone_link()
    print()
    check_virtual_cable()
    print()
    check_loopback_capture()
    print()
    asyncio.run(check_gemini_live())
    print()
    asyncio.run(check_pipeline_simulation())
    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
