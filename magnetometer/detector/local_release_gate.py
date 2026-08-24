#!/usr/bin/env python3
"""Production-readiness gate for a detector benchmark JSON report."""
from __future__ import annotations

import argparse
import json
import sys


# These are deliberately explicit operational guardrails, not claims of
# scientific ground truth. The detector must demonstrate useful event recall
# without becoming an always-on alarm generator.
MIN_SAMPLE_PRECISION = 0.80
MIN_SAMPLE_RECALL = 0.50
MAX_SAMPLE_FALSE_ALARM_RATE = 0.02
MIN_EVENT_PRECISION = 0.80
MIN_EVENT_RECALL = 0.50
MAX_FALSE_EVENTS_PER_DAY = 0.25
MAX_STATE_CHANGES_PER_DAY = 6.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate a detector benchmark report for production readiness")
    ap.add_argument("report")
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    sample = report["local"]["sample"]
    event = report["local"]["event"]
    stability = report["stability"]
    durations = report["local_detector"].get("event_durations_minutes", [])
    min_event = float(report["local_detector"]["config"]["min_event_minutes"])

    checks = [
        ("sample_precision", sample["precision"] >= MIN_SAMPLE_PRECISION, sample["precision"], f">= {MIN_SAMPLE_PRECISION:.2f}"),
        ("sample_recall", sample["recall"] >= MIN_SAMPLE_RECALL, sample["recall"], f">= {MIN_SAMPLE_RECALL:.2f}"),
        ("sample_false_alarm_rate", sample["false_alarm_rate"] <= MAX_SAMPLE_FALSE_ALARM_RATE, sample["false_alarm_rate"], f"<= {MAX_SAMPLE_FALSE_ALARM_RATE:.2f}"),
        ("event_precision", event["precision"] >= MIN_EVENT_PRECISION, event["precision"], f">= {MIN_EVENT_PRECISION:.2f}"),
        ("event_recall", event["recall"] >= MIN_EVENT_RECALL, event["recall"], f">= {MIN_EVENT_RECALL:.2f}"),
        ("false_events_per_day", event["false_events_per_day"] <= MAX_FALSE_EVENTS_PER_DAY, event["false_events_per_day"], f"<= {MAX_FALSE_EVENTS_PER_DAY:.2f}"),
        ("state_changes_per_day", stability["state_changes_per_day"] <= MAX_STATE_CHANGES_PER_DAY, stability["state_changes_per_day"], f"<= {MAX_STATE_CHANGES_PER_DAY:.1f}"),
        ("minimum_event_duration", all(d >= min_event for d in durations), min(durations) if durations else 0.0, f">= {min_event:.1f} min"),
    ]

    print("=== LOCAL DETECTOR PRODUCTION RELEASE GATE ===")
    passed = True
    for name, ok, value, rule in checks:
        status = "PASS" if ok else "FAIL"
        if isinstance(value, float):
            print(f"{status:4s} {name:28s}: {value:.4f} ({rule})")
        else:
            print(f"{status:4s} {name:28s}: {value} ({rule})")
        passed &= ok

    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
