"""Autonomous watcher and bug monitor for the AI voice calling agent.

Monitors 4 critical layers simultaneously:
1. Audio Streams (Inbound loopback peak/RMS & Outbound virtual cable peak/RMS)
2. Windows Audio Device Routing (checks default capture role drift)
3. Phone Link Process & Endpoint Health (PhoneExperienceHost)
4. Daemon Transcripts & Exceptions (tails voicebot.log)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundcard as sc

# Suppress soundcard buffer discontinuity warnings
if hasattr(sc, "SoundcardRuntimeWarning"):
    warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
try:
    import soundcard.mediafoundation as sc_mf
    sc_mf.warnings.warn = lambda *args, **kwargs: None
except Exception:
    pass

from . import config, routing

# Anomaly threshold constants
THRESHOLD_CALLER_LOUD = -42.0     # dBFS
THRESHOLD_AI_SPEAKING = -45.0     # dBFS
THRESHOLD_CLIPPING = -0.5          # dBFS


def _get_phone_link_pid() -> int | None:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-Process -Name 'PhoneExperienceHost' -ErrorAction SilentlyContinue).Id"],
            text=True,
            timeout=5,
        ).strip()
        return int(out.split()[0]) if out else None
    except Exception:
        return None


class BugMonitor:
    def __init__(self, log_path: Path = config.LOG_FILE) -> None:
        self.log_path = log_path
        self.running = True
        self.bugs_detected: list[str] = []
        self._last_log_pos = 0

        # Resolve soundcard endpoints
        self.spk_loop = None
        self.cable_out = None
        for m in sc.all_microphones(include_loopback=True):
            try:
                if getattr(m, "isloopback", False) and str(m.name) == config.LOOPBACK_DEVICE:
                    self.spk_loop = m
                elif not getattr(m, "isloopback", False) and "CABLE Output" in str(m.name):
                    self.cable_out = m
            except Exception:
                continue

        if not self.spk_loop:
            print(f"[WARN] Loopback endpoint {config.LOOPBACK_DEVICE!r} not found by soundcard")
        if not self.cable_out:
            print("[WARN] CABLE Output endpoint not found by soundcard")

    def log_bug(self, category: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] [BUG: {category}] {message}"
        self.bugs_detected.append(entry)
        print(f"\033[91m{entry}\033[0m", flush=True)

    def log_info(self, category: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{category}] {message}", flush=True)

    def check_routing(self) -> None:
        """Verify Windows capture defaults are pointed to CABLE Output."""
        try:
            status = routing.status(quiet=True)
            cap = status.get("console_capture") or ""
            comm = status.get("communications_capture") or ""
            if "CABLE Output" not in cap:
                self.log_bug("ROUTING_DRIFT", f"console_capture drifted to: {cap}")
            if "CABLE Output" not in comm:
                self.log_bug("ROUTING_DRIFT", f"communications_capture drifted to: {comm}")
        except Exception as exc:
            self.log_bug("ROUTING_CHECK_FAILED", str(exc))

    def check_phone_link(self) -> None:
        """Verify PhoneExperienceHost is alive."""
        pid = _get_phone_link_pid()
        if not pid:
            self.log_bug("PHONE_LINK_OFFLINE", "PhoneExperienceHost.exe is not running")

    def tail_daemon_logs(self) -> None:
        """Read newly appended lines from voicebot.log."""
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._last_log_pos)
                lines = f.readlines()
                self._last_log_pos = f.tell()

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                if "caller:" in line_str:
                    print(f"\033[96m[CALLER] {line_str.split('caller:', 1)[1].strip()}\033[0m", flush=True)
                elif "agent :" in line_str:
                    print(f"\033[92m[AGENT]  {line_str.split('agent :', 1)[1].strip()}\033[0m", flush=True)
                elif "ERROR" in line_str or "Exception" in line_str:
                    self.log_bug("DAEMON_EXCEPTION", line_str)
                elif "WARNING" in line_str:
                    self.log_info("DAEMON_WARN", line_str)
        except Exception:
            pass

    async def run(self, interval: float = 1.0) -> None:
        self.log_info("MONITOR", "Voicebot Bug Watcher initialized and active.")
        print("-" * 65, flush=True)

        loop_rate = 24000
        cable_rate = 44100
        block_samples_loop = int(loop_rate * 0.1)  # 100ms
        block_samples_cable = int(cable_rate * 0.1)

        loop_rec = self.spk_loop.recorder(samplerate=loop_rate, channels=1, blocksize=block_samples_loop) if self.spk_loop else None
        cable_rec = self.cable_out.recorder(samplerate=cable_rate, channels=1, blocksize=block_samples_cable) if self.cable_out else None

        if loop_rec:
            loop_rec.__enter__()
        if cable_rec:
            cable_rec.__enter__()

        last_route_check = 0.0
        last_proc_check = 0.0

        try:
            while self.running:
                now = time.monotonic()

                # 1. Check Routing every 4s
                if now - last_route_check >= 4.0:
                    self.check_routing()
                    last_route_check = now

                # 2. Check Phone Link every 4s
                if now - last_proc_check >= 4.0:
                    self.check_phone_link()
                    last_proc_check = now

                # 3. Read daemon log updates
                self.tail_daemon_logs()

                # 4. Measure Loopback (Inbound caller audio)
                loop_db = -120.0
                if loop_rec:
                    try:
                        data_loop = np.asarray(loop_rec.record(numframes=block_samples_loop), dtype=np.float32).reshape(-1)
                        if data_loop.size:
                            peak = float(np.abs(data_loop).max())
                            loop_db = 20.0 * np.log10(peak) if peak > 1e-6 else -120.0
                            if loop_db > THRESHOLD_CLIPPING:
                                self.log_bug("AUDIO_CLIPPING", f"Caller audio on loopback clipped at {loop_db:.1f} dBFS")
                            elif loop_db > THRESHOLD_CALLER_LOUD:
                                self.log_info("CALLER_AUDIO", f"Incoming voice active on loopback: {loop_db:.1f} dBFS")
                    except Exception:
                        pass

                # 5. Measure CABLE Output (Outbound AI audio to Phone Link mic)
                cable_db = -120.0
                if cable_rec:
                    try:
                        data_cable = np.asarray(cable_rec.record(numframes=block_samples_cable), dtype=np.float32).reshape(-1)
                        if data_cable.size:
                            peak = float(np.abs(data_cable).max())
                            cable_db = 20.0 * np.log10(peak) if peak > 1e-6 else -120.0
                            if cable_db > THRESHOLD_CLIPPING:
                                self.log_bug("AUDIO_CLIPPING", f"AI audio on CABLE Output clipped at {cable_db:.1f} dBFS")
                            elif cable_db > THRESHOLD_AI_SPEAKING:
                                self.log_info("AI_AUDIO", f"AI voice streaming into Phone Link mic: {cable_db:.1f} dBFS")
                    except Exception:
                        pass

                await asyncio.sleep(interval)
        finally:
            if loop_rec:
                try:
                    loop_rec.__exit__(None, None, None)
                except Exception:
                    pass
            if cable_rec:
                try:
                    cable_rec.__exit__(None, None, None)
                except Exception:
                    pass
            self.log_info("MONITOR", "Voicebot Bug Watcher stopped.")


def main():
    monitor = BugMonitor()
    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        print("\nWatcher stopped by user.")


if __name__ == "__main__":
    main()
