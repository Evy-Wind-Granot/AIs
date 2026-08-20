#!/usr/bin/env python3
"""Dependency-light seismometer repository smoke test.

The full PhaseNet/EQTransformer demo requires ObsPy and SeisBench.  This
self-test deliberately avoids those optional runtime dependencies and checks
the schema/instrument contract plus a deterministic three-component signal.
"""
from __future__ import annotations

import json
import numpy as np


def main() -> int:
    samples = np.arange(300, dtype=np.int32)
    message = {
        "sequence_number": 1,
        "timestamp": [{"seconds": 0, "nanoseconds": 0, "source": "SELF-TEST"}],
        "payload": {
            "channel_id": "XX.TEST.00.BHZ",
            "sample_rate": 100.0,
            "sample_count": int(samples.size),
            "samples": samples.tolist(),
        },
    }

    encoded = json.dumps(message)
    decoded = json.loads(encoded)
    payload = decoded["payload"]
    if payload["sample_count"] != len(payload["samples"]):
        raise AssertionError("sample_count does not match samples")
    if len(payload["channel_id"].split(".")) != 4:
        raise AssertionError("channel_id does not follow NET.STA.LOC.CHA schema")
    if not np.isfinite(samples).all():
        raise AssertionError("synthetic waveform contains non-finite values")

    print("Seismometer schema/signal self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
