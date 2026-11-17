"""
e2e_tests/runner.py

Standalone CLI E2E Test Runner for AI Voice Calling Assistant.
Executes 4-tier opaque-box test suite:
- Tier 1: Category-Partition Feature Coverage
- Tier 2: Boundary Value Analysis & Stress Hardening
- Tier 3: Pairwise Cross-Feature Interactions
- Tier 4: Real-World Workload Scenarios

Features:
- Colorized ASCII dashboard breakdown by tier
- Metrics: Total, Passed, Failed, Skipped, and Duration
- Command-line tier and scenario filters
- Clean exit code 0 on pass, non-zero on failure

Usage:
  python -m e2e_tests.runner
  python -m e2e_tests.runner --tier 1
  python -m e2e_tests.runner --tier 2
  python -m e2e_tests.runner --tier 3
  python -m e2e_tests.runner --tier 4
  python -m e2e_tests.runner --scenario 1
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Any
import pytest

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
GRAY = "\033[90m"


class E2ETestCollector:
    """Pytest plugin collecting execution metrics per test node."""

    def __init__(self):
        self.reports = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.failed):
            self.reports.append(report)


def parse_args():
    parser = argparse.ArgumentParser(description="AI Voice Caller E2E Test Suite Runner")
    parser.add_argument(
        "-t", "--tier",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Run a specific test tier (default: all)"
    )
    parser.add_argument(
        "-s", "--scenario",
        type=int,
        choices=range(1, 7),
        help="Run a specific Tier 4 scenario (1-6)"
    )
    parser.add_argument(
        "-k", "--filter",
        help="Expression filter for test names"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose pytest output"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output"
    )
    return parser.parse_args()


def main():
    # Enable Windows VT100 ANSI processing
    if sys.platform == "win32":
        os.system("")

    args = parse_args()

    c_cyan = "" if args.no_color else CYAN
    c_bold = "" if args.no_color else BOLD
    c_reset = "" if args.no_color else RESET
    c_green = "" if args.no_color else GREEN
    c_red = "" if args.no_color else RED
    c_yellow = "" if args.no_color else YELLOW
    c_magenta = "" if args.no_color else MAGENTA

    print(f"\n{c_cyan}{c_bold}========================================================================")
    print("        AI VOICE CALLING ASSISTANT — E2E TEST RUNNER")
    print(f"========================================================================{c_reset}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tier_files = {
        "1": os.path.join(base_dir, "test_tier1_features.py"),
        "2": os.path.join(base_dir, "test_tier2_boundaries.py"),
        "3": os.path.join(base_dir, "test_tier3_combinations.py"),
        "4": os.path.join(base_dir, "test_tier4_scenarios.py"),
    }

    if args.scenario:
        selected_files = [tier_files["4"]]
    elif args.tier == "all":
        selected_files = [tier_files["1"], tier_files["2"], tier_files["3"], tier_files["4"]]
    else:
        selected_files = [tier_files[args.tier]]

    pytest_args = ["-q"]
    if args.verbose:
        pytest_args.append("-v")

    if args.scenario:
        pytest_args.extend(["-k", f"scenario_{args.scenario}"])
    elif args.filter:
        pytest_args.extend(["-k", args.filter])

    pytest_args.extend(selected_files)

    collector = E2ETestCollector()
    t0 = time.perf_counter()
    exit_code = pytest.main(pytest_args, plugins=[collector])
    elapsed = time.perf_counter() - t0

    # Partition reports into tiers
    tier_metrics = {
        "Tier 1 (Features)": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "duration": 0.0},
        "Tier 2 (Boundaries)": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "duration": 0.0},
        "Tier 3 (Combinations)": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "duration": 0.0},
        "Tier 4 (Scenarios)": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "duration": 0.0},
    }

    for rep in collector.reports:
        tier_key = None
        if "test_tier1" in rep.nodeid:
            tier_key = "Tier 1 (Features)"
        elif "test_tier2" in rep.nodeid:
            tier_key = "Tier 2 (Boundaries)"
        elif "test_tier3" in rep.nodeid:
            tier_key = "Tier 3 (Combinations)"
        elif "test_tier4" in rep.nodeid:
            tier_key = "Tier 4 (Scenarios)"

        if tier_key and tier_key in tier_metrics:
            m = tier_metrics[tier_key]
            m["total"] += 1
            m["duration"] += getattr(rep, "duration", 0.0)
            if rep.passed:
                m["passed"] += 1
            elif rep.failed:
                m["failed"] += 1
            elif rep.skipped:
                m["skipped"] += 1

    total_tests = len(collector.reports)
    total_passed = sum(1 for r in collector.reports if r.passed)
    total_failed = sum(1 for r in collector.reports if r.failed)
    total_skipped = sum(1 for r in collector.reports if r.skipped)

    print(f"\n{c_bold}TEST EXECUTION BREAKDOWN BY TIER:{c_reset}")
    print("┌─────────────────────────────┬───────┬────────┬────────┬─────────┬──────────┐")
    print("│ Test Tier                   │ Total │ Passed │ Failed │ Skipped │ Duration │")
    print("├─────────────────────────────┼───────┼────────┼────────┼─────────┼──────────┤")
    for t_name, data in tier_metrics.items():
        if data["total"] > 0:
            p_color = c_green if data["failed"] == 0 else c_red
            print(
                f"│ {t_name:<27} │ {data['total']:>5} │ "
                f"{p_color}{data['passed']:>6}{c_reset} │ "
                f"{data['failed']:>6} │ {data['skipped']:>7} │ "
                f"{data['duration']:>7.2f}s │"
            )
    print("├─────────────────────────────┼───────┼────────┼────────┼─────────┼──────────┤")
    tot_color = c_green if total_failed == 0 else c_red
    print(
        f"│ {c_bold}{'TOTAL':<27}{c_reset} │ {c_bold}{total_tests:>5}{c_reset} │ "
        f"{tot_color}{c_bold}{total_passed:>6}{c_reset} │ "
        f"{total_failed:>6} │ {total_skipped:>7} │ {elapsed:>7.2f}s │"
    )
    print("└─────────────────────────────┴───────┴────────┴────────┴─────────┴──────────┘")

    if total_failed == 0 and total_tests > 0:
        pct = (total_passed / total_tests) * 100.0
        print(f"\n{c_green}{c_bold}✓ ALL TESTS PASSED ({pct:.1f}% SUCCESS RATE) in {elapsed:.2f}s{c_reset}\n")
        sys.exit(0)
    else:
        print(f"\n{c_red}{c_bold}✗ TEST SUITE FAILED ({total_failed} failures, {total_passed}/{total_tests} passed){c_reset}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
