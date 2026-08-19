"""Canonical strict release gate using pooled independent-event sufficiency.

Keeps all existing detection, coverage, confidence-interval, and zero-holdout
requirements. Replaces the scientifically invalid per-year quota with:
4+ usable cases per observatory/year/class, 8+ pooled cases per year/class,
and 12+ pooled cases per class per split.
"""
from __future__ import annotations

from collections import Counter

from .. import production_release_gate_v3 as _gate

MIN_STATION_CASES = 4
MIN_POOLED_CASES_PER_YEAR = 8
MIN_CASES_PER_SPLIT = 12


def _pooled_count_ok(rows, _minimum):
    if not rows:
        return False
    counts = Counter((r["observatory"], int(r["case"]["year"]), r["case"]["class_name"]) for r in rows)
    observatories = sorted({r["observatory"] for r in rows})
    years = sorted({int(r["case"]["year"]) for r in rows})
    for obs in observatories:
        for year in years:
            for cls in ("quiet", "active", "storm"):
                if counts[(obs, year, cls)] < MIN_STATION_CASES:
                    return False
    for year in years:
        for cls in ("quiet", "active", "storm"):
            if sum(counts[(obs, year, cls)] for obs in observatories) < MIN_POOLED_CASES_PER_YEAR:
                return False
    for cls in ("quiet", "active", "storm"):
        if sum(counts[(obs, year, cls)] for obs in observatories for year in years) < MIN_CASES_PER_SPLIT:
            return False
    return True


_gate.count_ok = _pooled_count_ok


def main() -> None:
    _gate.main()


if __name__ == "__main__":
    main()
