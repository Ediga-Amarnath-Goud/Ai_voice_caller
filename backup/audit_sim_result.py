import wave
import numpy as np

def load(path):
    with wave.open(path, "rb") as w:
        r = w.getframerate(); ch = w.getnchannels()
        d = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    if ch == 2:
        d = d.reshape(-1, 2).mean(axis=1)
    return d, r

caller, cr = load("logs/caller_sim.wav")
cap, capr = load("logs/sim_cable_out.wav")
# Resample caller fixture to capture rate by linear interpolation
t = np.arange(len(caller)) / cr
t2 = np.arange(len(cap)) / capr
caller_r = np.interp(t2 % (len(caller)/cr), t, caller) if False else None
# Simpler: tile caller to capture length at same duration
reps = int(np.ceil(len(cap)/len(caller)*capr/cr))
tiled = np.tile(caller, reps)[:len(cap)]

# Cross-correlation at zero lag (and a few lags to be safe)
a = cap - cap.mean(); b = tiled - tiled.mean()
corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print(f"corr(recorded, caller_fixture) = {corr:+.3f}   (high => caller bleed-through, not AI)")

# Spectral centroid: caller speech vs AI greeting differ somewhat
def centroid(x, r):
    s = np.abs(np.fft.rfft(x[:r*5]))
    f = np.fft.rfftfreq(r*5, 1/r)
    return float((s*f).sum()/(s.sum()+1e-12))
print(f"caller centroid ~{centroid(caller, cr):.0f} Hz, recorded centroid ~{centroid(cap, capr):.0f} Hz")

# RMS over time to see if there is distinct speech activity vs constant tone
for i in range(0, 10):
    seg = cap[int(i*len(cap)/10):int((i+1)*len(cap)/10)]
    print(f"  {i*3:>2}s: {20*np.log10(np.sqrt((seg**2).mean())+1e-12):6.1f} dBFS RMS")
