import soundcard as sc, numpy as np, wave, time

print("="*60)
print("AUDIO BRIDGE DIAGNOSTIC")
print("="*60)

# Step 1: Check what devices exist
print("\n=== PLAYBACK DEVICES ===")
for s in sc.all_speakers():
    print(f"  PLAY: {s.name}")
print()
print("=== CAPTURE DEVICES ===")
for m in sc.all_microphones(include_loopback=True):
    flag = "LOOPBACK" if m.isloopback else "MIC"
    print(f"  {flag} {m.name}")

# Step 2: Test loopback capture by playing through Realtek speakers
print("\n=== LOOPBACK CAPTURE TEST ===")
print("Playing test tone through Realtek speakers...")
with wave.open('logs/caller_sim.wav','rb') as w:
    rate = w.getframerate(); ch = w.getnchannels()
    pcm = w.readframes(w.getnframes())
data = np.frombuffer(pcm, dtype='<i2').astype(np.float32)/32768.0
if ch==2: data = data.reshape(-1,2).mean(axis=1)

spk = next(s for s in sc.all_speakers() if s.name=='Speakers (Realtek(R) Audio)')
mic = next(m for m in sc.all_microphones(include_loopback=True) if m.name=='Speakers (Realtek(R) Audio)' and m.isloopback)

print(f"  Speaker: {spk.name}")
print(f"  Capture: {mic.name}")

recorder = mic.recorder(samplerate=16000, channels=1, blocksize=1600)
recorder.__enter__()

with spk.player(samplerate=rate, channels=1, blocksize=1600) as p:
    p.play(data)
    time.sleep(0.3)
    chunks=[]
    for i in range(50):
        chunks.append(np.asarray(recorder.record(numframes=1600), dtype=np.float32).reshape(-1))
    captured=np.concatenate(chunks)
recorder.__exit__(None, None, None)

peak=float(np.abs(captured).max()); db=20*np.log10(peak) if peak>0 else -240
print(f"  Loopback capture peak: {db:.1f} dBFS")
if db > -60:
    print("  RESULT: LOOPBACK WORKING - audio passes through Realtek speakers")
else:
    print("  RESULT: LOOPBACK SILENT - Phone Link may NOT route audio to Realtek speakers")

# Test 2: Try CABLE Input loopback
print()
print("=== TEST 2: CABLE Input loopback ===")
spk2 = next(s for s in sc.all_speakers() if s.name=='CABLE Input (VB-Audio Virtual Cable)')
mic2 = next(m for m in sc.all_microphones(include_loopback=True) if m.name=='CABLE Input (VB-Audio Virtual Cable)' and m.isloopback)

print(f"Playing through CABLE Input speakers...")
recorder2 = mic.recorder(samplerate=16000, channels=1, blocksize=1600)
recorder2.__enter__()

with spk.player(samplerate=rate, channels=1, blocksize=1600) as p:
    p.play(data)
    time.sleep(0.2)
    chunks=[]
    for i in range(50):
        chunks.append(np.asarray(recorder.record(numframes=1600), dtype=np.float32).reshape(-1))
    captured2=np.concatenate(chunks)
recorder.__exit__(None, None, None)

peak2=float(np.abs(captured).max()); db2=20*np.log10(peak2) if peak2>0 else -240
print(f'CABLE Input loopback capture: {db2:.1f} dBFS')
print('CABLE LOOPBACK WORKING!' if db2 > -60 else 'CABLE LOOPBACK SILENT')