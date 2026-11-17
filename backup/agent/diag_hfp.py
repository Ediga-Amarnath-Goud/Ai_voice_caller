"""Diagnose why the Bluetooth hands-free endpoints refuse to open.

PortAudio reports -9999 (unanticipated host error) for every open attempt, which
is the host API's catch-all: it can mean a wrong format, a wrong rate, an
exclusive-mode conflict, or a device that is present in the enumeration but has
no live pin behind it. This walks the full matrix so the cause stops being a
guess.

    python -m agent.diag_hfp
    python -m agent.diag_hfp --rates 8000 16000 44100 48000
"""

from __future__ import annotations

import argparse

import pyaudio


def _host_apis(pa: pyaudio.PyAudio) -> dict[int, str]:
    return {
        index: pa.get_host_api_info_by_index(index)["name"]
        for index in range(pa.get_host_api_count())
    }


def _matches(name: str, needle: str) -> bool:
    return needle.lower() in name.lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.diag_hfp")
    parser.add_argument("--match", default="iQOO")
    parser.add_argument("--rates", nargs="*", type=int, default=[8000, 16000, 44100, 48000])
    args = parser.parse_args(argv)

    pa = pyaudio.PyAudio()
    apis = _host_apis(pa)
    print("host APIs")
    for index, name in apis.items():
        info = pa.get_host_api_info_by_index(index)
        print(
            f"  {index}: {name:<18} devices={info['deviceCount']:>2} "
            f"default out=#{info['defaultOutputDevice']} in=#{info['defaultInputDevice']}"
        )

    print(f"\ndevices matching {args.match!r}")
    hits: list[int] = []
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if _matches(info["name"], args.match):
            hits.append(index)
            print(
                f"  #{index:<3} [{apis.get(info['hostApi'], '?')}] "
                f"in={info['maxInputChannels']} out={info['maxOutputChannels']} "
                f"rate={info['defaultSampleRate']:.0f}  {info['name']}"
            )

    if not hits:
        print("  (none)")
        pa.terminate()
        return 1

    print("\nopen attempts (shared, 1 channel, 16-bit)")
    for index in hits:
        info = pa.get_device_info_by_index(index)
        api = apis.get(info["hostApi"], "?")
        for is_input in (False, True):
            if is_input and not info["maxInputChannels"]:
                continue
            if not is_input and not info["maxOutputChannels"]:
                continue
            line = f"  #{index} {api:<12} {'in ' if is_input else 'out'} "
            outcomes = []
            for rate in args.rates:
                try:
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=is_input,
                        output=not is_input,
                        input_device_index=index if is_input else None,
                        output_device_index=None if is_input else index,
                        frames_per_buffer=320,
                    )
                    stream.close()
                    outcomes.append(f"{rate}=OK")
                except OSError as exc:
                    code = exc.errno if getattr(exc, "errno", None) else "?"
                    outcomes.append(f"{rate}={code}")
            print(line + "  ".join(outcomes))

    pa.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
