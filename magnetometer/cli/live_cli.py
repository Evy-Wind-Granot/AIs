"""Transport-neutral JSONL adapter for the causal live detector.

Input: one JSON object per line, e.g. {"timestamp":"2026-08-17T19:00:00Z","value_nt":123.4}
Output: one JSON detection record per input sample.
"""
from __future__ import annotations

import argparse
import json
import sys

from ..core import LiveDetector, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Strict JSON/YAML pipeline config")
    parser.add_argument("--state-file", help="Optional detector state output path")
    args = parser.parse_args()

    if args.config:
        load_config(args.config)
    detector = LiveDetector.from_pipeline_defaults()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise ValueError("sample must be a JSON object")
            timestamp = sample.get("timestamp")
            if timestamp is None:
                raise ValueError("missing 'timestamp'")
            value = sample.get("value_nt", sample.get("value"))
            if value is None:
                raise ValueError("missing 'value_nt'")
            result = detector.update(timestamp, float(value))
            print(json.dumps(result, separators=(",", ":")), flush=True)
            if args.state_file:
                detector.save_state(args.state_file)
        except Exception as exc:
            error = {"status": "error", "error": str(exc)}
            print(json.dumps(error, separators=(",", ":")), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
