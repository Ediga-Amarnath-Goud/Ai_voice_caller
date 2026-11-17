import os
import sys
import time
import queue
import asyncio
import logging
import subprocess
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in environment or .env")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_call_bluetooth")

TARGET_PHONE_NUMBER = "9133233391"
ADB_PATH = "platform-tools\\adb.exe"

def run_adb(args):
    cmd = [ADB_PATH, "-d"] + args
    return subprocess.run(cmd, capture_output=True, text=True)

# Hardware indices for PC audio (which Windows routes to/from Bluetooth call)
MIC_DEVICE_IDX = 2   # AMD Microphone Array
SPK_DEVICE_IDX = 4   # Realtek Speakers

async def main():
    print("=" * 72)
    print("   AI VOICE CALLER — LIVE CELLULAR TEST VIA BLUETOOTH + BSNL SIM 2")
    print(f"   Target Number: {TARGET_PHONE_NUMBER}")
    print("   SIM Carrier  : BSNL Mobile (Subscription ID: 4)")
    print("   AI Persona   : Aoede (Google Gemini Live 3.8)")
    print("=" * 72)

    # 1. Verify USB Connection
    print("\n[1/5] Checking hardware USB-C connection...")
    dev_check = subprocess.run([ADB_PATH, "-d", "get-state"], capture_output=True, text=True)
    if "device" not in dev_check.stdout:
        print("ERROR: Phone not ready over USB.")
        sys.exit(1)
    print("      SUCCESS: Phone connected and authorized over USB-C!")

    # Wake up screen and unlock
    run_adb(["shell", "input", "keyevent", "KEYCODE_WAKEUP"])
    time.sleep(0.3)
    run_adb(["shell", "input", "swipe", "540", "1800", "540", "800"])

    # 2. Audio Hardware Setup
    devs = sd.query_devices()
    print(f"\n[2/5] Initializing Audio Hardware:")
    print(f"      Mic Input    : [{MIC_DEVICE_IDX}] {devs[MIC_DEVICE_IDX]['name']}")
    print(f"      Speaker Out  : [{SPK_DEVICE_IDX}] {devs[SPK_DEVICE_IDX]['name']}")

    audio_out_queue = queue.Queue()
    is_running = True
    ai_is_speaking = False

    def audio_out_callback(outdata, frames, time_info, status):
        nonlocal ai_is_speaking
        bytes_needed = frames * 2  # 16-bit mono
        data = bytearray()
        while len(data) < bytes_needed:
            try:
                chunk = audio_out_queue.get_nowait()
                data.extend(chunk)
            except queue.Empty:
                break
        
        ai_is_speaking = (len(data) > 0)

        if len(data) < bytes_needed:
            data.extend(b"\x00" * (bytes_needed - len(data)))
        elif len(data) > bytes_needed:
            data = data[:bytes_needed]
            
        outdata[:] = np.frombuffer(bytes(data), dtype=np.int16).reshape(-1, 1)

    out_stream = sd.OutputStream(
        device=SPK_DEVICE_IDX,
        channels=1,
        samplerate=24000,
        dtype='int16',
        blocksize=1200,
        callback=audio_out_callback
    )
    out_stream.start()

    # 3. Connect to Gemini Live 3.8
    print("\n[3/5] Connecting to Google Gemini Live 3.8 (Persona: Aoede)...")
    client = genai.Client(api_key=api_key)
    
    system_prompt = (
        "You are Aoede, a friendly, warm, and natural conversational companion on a phone call. "
        "You speak with a pleasant, cheerful, and expressive voice. "
        "When the person answers, greet them warmly: "
        "'Hello! Hi there! How are you doing today? Just calling to say hello!' "
        "Have a casual, relaxed conversation. Ask how their day is going, what they are up to, "
        "and chat naturally. Keep your answers conversational, natural, and concise as in a real phone call."
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=system_prompt)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        )
    )

    session_cm = client.aio.live.connect(model="gemini-3.8-live", config=config)
    session = await session_cm.__aenter__()
    print("      SUCCESS: Gemini Live 3.8 Connected & Ready!")

    # 4. Dial Target Number explicitly using BSNL (Subscription 4)
    print(f"\n[4/5] Dialing {TARGET_PHONE_NUMBER} explicitly on BSNL SIM 2 (subId=4)...")
    run_adb([
        "shell", "am", "start",
        "-a", "android.intent.action.CALL",
        "-d", f"tel:{TARGET_PHONE_NUMBER}",
        "--ei", "subscription", "4",
        "--ei", "com.android.phone.extra.slot", "1"
    ])

    print("\n[5/5] CALL IS DIALING OVER BSNL! Pick up your phone to talk with Aoede!")
    print("=" * 72)
    print(">>> ANSWER YOUR PHONE WHEN IT RINGS <<<")
    print("=" * 72 + "\n")

    input_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def in_callback(indata, frames, time_info, status):
        # Software echo suppression
        if not ai_is_speaking:
            data_int16 = (indata * 32767).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(input_queue.put_nowait, data_int16)

    in_stream = sd.InputStream(
        device=MIC_DEVICE_IDX,
        channels=1,
        samplerate=16000,
        dtype='float32',
        blocksize=1600,
        callback=in_callback
    )
    in_stream.start()

    # Trigger initial greeting when user picks up (after 5 seconds of ringing)
    async def initial_greeting_trigger():
        await asyncio.sleep(5.0)
        print("\n[AI initiating greeting to caller...]")
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(
                text="[The call has connected. Say hello warmly right now and start the conversation!]"
            )]
        )
        try:
            await session.send_client_content(turns=content, turn_complete=True)
        except Exception as e:
            logger.error(f"Error sending initial greeting turn: {e}")

    async def send_audio_task():
        while is_running:
            try:
                pcm_chunk = await input_queue.get()
                blob = types.Blob(data=pcm_chunk, mime_type="audio/pcm;rate=16000")
                await session.send_realtime_input(media=blob)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending mic audio to Gemini: {e}")
                break

    async def receive_gemini_task():
        nonlocal is_running
        try:
            async for message in session.receive():
                if not is_running:
                    break
                sc = getattr(message, "server_content", None)
                if sc is not None:
                    # Speech barge-in
                    if getattr(sc, "interrupted", False):
                        print("\n[User interrupted -> Clearing speech buffer]")
                        while not audio_out_queue.empty():
                            try:
                                audio_out_queue.get_nowait()
                            except queue.Empty:
                                break

                    model_turn = getattr(sc, "model_turn", None)
                    if model_turn is not None:
                        for part in getattr(model_turn, "parts", []):
                            inline_data = getattr(part, "inline_data", None)
                            if inline_data and getattr(inline_data, "data", None):
                                audio_out_queue.put(inline_data.data)

                    if getattr(sc, "turn_complete", False):
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error receiving from Gemini: {e}")

    async def monitor_call_state():
        nonlocal is_running
        # Give call 8 seconds to establish
        await asyncio.sleep(8)
        while is_running:
            await asyncio.sleep(1.5)
            res = run_adb(["shell", "dumpsys", "telephony.registry"]).stdout
            states = []
            for line in res.splitlines():
                if "mCallState=" in line:
                    val = line.split("=")[1].strip()
                    if val.isdigit():
                        states.append(int(val))
            
            # Call ends ONLY when BOTH SIM slots are idle (0)
            if states and all(s == 0 for s in states):
                print("\n[Call disconnected by remote party — both SIMs idle]")
                is_running = False
                break

    send_t = asyncio.create_task(send_audio_task())
    recv_t = asyncio.create_task(receive_gemini_task())
    mon_t = asyncio.create_task(monitor_call_state())
    greet_t = asyncio.create_task(initial_greeting_trigger())

    # Allow call up to 180 seconds (3 minutes)
    start_time = time.time()
    try:
        while is_running and (time.time() - start_time < 180):
            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[User requested call stop]")
    finally:
        is_running = False
        send_t.cancel()
        recv_t.cancel()
        mon_t.cancel()
        greet_t.cancel()

        print("\nEnding call...")
        run_adb(["shell", "input", "keyevent", "KEYCODE_ENDCALL"])
        
        in_stream.stop()
        in_stream.close()
        out_stream.stop()
        out_stream.close()

        await session_cm.__aexit__(None, None, None)
        print("Call completed cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
