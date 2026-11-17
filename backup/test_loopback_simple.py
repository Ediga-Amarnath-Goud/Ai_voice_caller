import soundcard as sc, numpy as np, wave, time

with wave.open('logs/caller_sim.wav','rb') as w:
    rate = w.getframerate(); ch = w.getnchannels()
    pcm = w.readframes(w.getnframes())
data = np.frombuffer(pcm, dtype='<i2').astype(np.float32)/32768.0
if ch==2: data = data.reshape(-1,2).mean(axis=1)

spk = next(s for s in sc.all_speakers() if s.name=='Speakers (Realtek(R) Audio)')
mic = next(m for m in sc.all_microphones(include_loopback=True) if m.name=='Speakers (Realtek(R) Audio)' and m.isloopback)

print(f'Playback: {spk.name} @ {rate} Hz')
print(f'Capture: {mic.name}')
print()

print('Playing test tone through Realtek speakers...')
print('Capturing loopback for 5 seconds...')

recorder = mic.recorder(samplerate=16000, channels=1, blocksize=1600)
recorder.__enter__()

with spk.player(samplerate=rate, channels=1, blocksize=1600) as p:
    p.play(data)
    time.sleep(0.3)
    chunks=[]
    for i in range(100):
        chunks.append(np.asarray(recorder.record(numframes=1600), dtype=np.float32).reshape(-1))
    captured=np.concatenate(chunks)
recorder.__exit__(None, None, None)

peak=float(np.abs(captured).max()); db=20*np.log10(peak) if peak>0 else -240
print(f'Loopback capture peak: {db:.1f} dBFS')
print('LOOPBACK WORKING!' if db > -60 else 'LOOPBACK SILENT')