"""Query the Windows Core Audio stack directly, bypassing PortAudio.

PortAudio only exposes the Bluetooth hands-free endpoints under WDM-KS and every
open fails with -9999. That is either a PortAudio limitation or a dead endpoint;
Core Audio is the authority, because it is the API Windows itself uses.
"""

import sys

sys.path.insert(0, r"C:\Users\amarn\Desktop\Ai_voice_caller")

import soundcard as sc

NEEDLE = "iqoo"


def main() -> int:
    speakers = sc.all_speakers()
    mics = sc.all_microphones(include_loopback=True)

    print(f"core audio render endpoints: {len(speakers)}")
    for sp in speakers:
        marker = "  <-- iQOO" if NEEDLE in sp.name.lower() else ""
        print(f"  {sp.name}{marker}")

    print(f"\ncore audio capture endpoints (incl. loopback): {len(mics)}")
    for mic in mics:
        marker = "  <-- iQOO" if NEEDLE in mic.name.lower() else ""
        print(f"  {mic.name}{marker}")

    hits = [sp for sp in speakers if NEEDLE in sp.name.lower()]
    if not hits:
        print("\nRESULT: iQOO render endpoint is NOT visible to Core Audio")
    for sp in hits:
        print(f"\nopening {sp.name} @ 16000 Hz mono for 0.4 s ...")
        try:
            with sp.player(samplerate=16000, channels=1, blocksize=320) as player:
                player.play(b"\x00\x00" * 3200)
            print("  render OPENED ok")
        except Exception as exc:  # noqa: BLE001 - report the raw failure
            print(f"  render FAILED -> {type(exc).__name__}: {exc}")

    hit_mics = [m for m in mics if NEEDLE in m.name.lower() and "loopback" not in m.name.lower()]
    for mic in hit_mics:
        print(f"\nopening {mic.name} @ 16000 Hz mono for 0.4 s ...")
        try:
            with mic.recorder(samplerate=16000, channels=1, blocksize=320) as recorder:
                recorder.record(numframes=3200)
            print("  capture OPENED ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  capture FAILED -> {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
