"""LIVE-CALL AUDIT: find which Windows render endpoint actually carries the caller.

Run this in a terminal, THEN place a Phone Link call. For ``DURATION`` seconds it
polls every loopback endpoint and prints the peak level seen on each. The endpoint
that spikes during the call is where the caller's voice really lands.

    python audit_call_endpoints.py 40
"""
import sys, time
import numpy as np
import soundcard as sc

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
POLL = 0.25  # seconds of audio captured per endpoint per poll

print(f"Auditing all loopback endpoints for {DURATION:.0f}s -- PLACE THE CALL NOW\n")
print(f"{'time':>6}  {'device':<48} {'peak dBFS':>10}")
print("-" * 70)

peaks: dict[str, float] = {}
seen: set[str] = set()
start = time.monotonic()
last_print = 0.0

while time.monotonic() - start < DURATION:
    elapsed = time.monotonic() - start
    loopbacks = [m for m in sc.all_microphones(include_loopback=True)
                 if getattr(m, "isloopback", False)]
    for dev in loopbacks:
        try:
            name = str(dev.name)
        except Exception:
            continue
        seen.add(name)
        rate = 16000
        block = int(rate * POLL)
        try:
            rec = dev.recorder(samplerate=rate, channels=1, blocksize=block)
            rec.__enter__()
            data = np.asarray(rec.record(numframes=block), np.float32).reshape(-1)
            rec.__exit__(None, None, None)
            peak = float(np.abs(data).max()) if data.size else 0.0
            db = 20 * np.log10(peak) if peak > 0 else -240.0
            if db > peaks.get(name, -240.0):
                peaks[name] = db
            if db > -55 and elapsed - last_print >= 1.0:
                print(f"{elapsed:6.1f}  {name:<48} {db:10.1f}  <-- SIGNAL")
                last_print = elapsed
        except Exception:
            continue
    time.sleep(0.05)

print("\n" + "=" * 70)
print("SUMMARY - max peak seen on each loopback endpoint during the window:")
print("=" * 70)
for name, db in sorted(peaks.items(), key=lambda x: -x[1]):
    tag = "  <== CALLER AUDIO HERE" if db > -55 else ""
    print(f"  {db:8.1f} dBFS  {name}{tag}")

print("\nEndpoints seen (loopback twins):")
for name in sorted(seen):
    print(f"  - {name}")
