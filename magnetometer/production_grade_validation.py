#!/usr/bin/env python3
"""Production validation data preparation used by detector calibration.

This module is the stable compatibility boundary for historical-case loading.
The detector calibrator depends on it for case discovery, caching and reference
alignment.  Residual generation is explicitly pinned to the causal baseline.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import causal_baseline
from . import performance_metrics as pm
from .magnetometer_demo import (
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    parse_iaga2002_to_dataframe,
)

pm.compute_qdc_baseline = causal_baseline.compute_causal_qdc_baseline

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CACHE_DIR = HERE / "data" / "case_cache_causal_v8"

_KP_CACHE: Dict[Tuple[str, str], pd.Series] = {}
_DST_CACHE: Dict[Tuple[int, int], Optional[pd.Series]] = {}

@dataclass(frozen=True)
class Case:
    case_id: str
    center_date: str
    start_date: str
    days: int
    class_name: str
    year: int
    split: str


def _fetch_kp_cached(start: str, end: str) -> pd.Series:
    key = (start, end)
    if key not in _KP_CACHE:
        _KP_CACHE[key] = fetch_kp_gfz(start, end)
    return _KP_CACHE[key]


def _fetch_dst_cached(year: int, month: int) -> Optional[pd.Series]:
    key = (int(year), int(month))
    if key not in _DST_CACHE:
        _DST_CACHE[key] = fetch_dst_kyoto(int(year), int(month))
    return _DST_CACHE[key]


def split_years(years: Sequence[int]) -> Dict[str, List[int]]:
    values = sorted(set(int(y) for y in years))
    if len(values) < 3:
        raise ValueError("At least 3 distinct years are required.")
    n = len(values)
    cal_n = max(1, n // 2)
    val_n = max(1, (n - cal_n) // 2)
    if cal_n + val_n >= n:
        val_n = 1
        cal_n = n - 2
    return {
        "calibration": values[:cal_n],
        "validation": values[cal_n:cal_n + val_n],
        "test": values[cal_n + val_n:],
    }


def discover_cases_for_year(kp: pd.Series, year: int, class_name: str,
                            per_year: int, window_days: int, split: str) -> List[Case]:
    year_kp = kp[kp.index.year == int(year)]
    if year_kp.empty:
        return []
    daily = year_kp.resample("1D").max().dropna()
    if class_name == "quiet":
        candidates = daily[daily <= 2.0].sort_values()
    elif class_name == "active":
        candidates = daily[(daily > 2.0) & (daily < 6.0)].sort_values(ascending=False)
    else:
        candidates = daily[daily >= 6.0].sort_values(ascending=False)
    selected: List[Case] = []
    separation = pd.Timedelta(days=max(window_days + 3, 14))
    for center, _ in candidates.items():
        if any(abs(center - pd.Timestamp(c.center_date, tz="UTC")) < separation for c in selected):
            continue
        start = (center - pd.Timedelta(days=window_days // 2)).normalize()
        selected.append(Case(
            case_id=f"{split}_{class_name}_{center.strftime('%Y%m%d')}",
            center_date=center.strftime("%Y-%m-%d"),
            start_date=start.strftime("%Y-%m-%d"),
            days=window_days,
            class_name=class_name,
            year=int(year),
            split=split,
        ))
        if len(selected) >= per_year:
            break
    return selected


def discover_suite(years: Sequence[int], cases_per_class_per_year: int,
                    window_days: int) -> Tuple[Dict[str, List[int]], List[Case]]:
    splits = split_years(years)
    kp = _fetch_kp_cached(f"{min(years):04d}-01-01", f"{max(years):04d}-12-31")
    if kp.empty:
        raise RuntimeError("Kp discovery returned no data.")
    cases: List[Case] = []
    for split, split_years_list in splits.items():
        for year in split_years_list:
            for class_name in ("quiet", "active", "storm"):
                cases.extend(discover_cases_for_year(
                    kp, year, class_name, cases_per_class_per_year, window_days, split
                ))
    return splits, sorted(cases, key=lambda c: (c.split, c.year, c.class_name, c.center_date))


def _cache_path(cache_dir: Path, observatory: str, case: Case) -> Path:
    return Path(cache_dir) / f"{observatory.upper()}_{case.case_id}_{case.days}d.npz"


def _save_case_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{id(data)}.tmp")
    payload = {
        "residual": np.asarray(data["residual"], dtype=float),
        "known": np.asarray(data["refs"]["known"], dtype=bool),
        "active": np.asarray(data["refs"]["active"], dtype=bool),
        "storm": np.asarray(data["refs"]["storm"], dtype=bool),
        "kp_known": np.asarray(data["refs"]["kp_known"], dtype=bool),
        "dst_known": np.asarray(data["refs"]["dst_known"], dtype=bool),
        "cadence_s": np.asarray([data["cadence_s"]]),
        "completeness": np.asarray([data["completeness"]]),
        "kp_coverage": np.asarray([data["kp_coverage"]]),
        "dst_coverage": np.asarray([data["dst_coverage"]]),
        "reference_coverage": np.asarray([data["reference_coverage"]]),
        "series": np.asarray(data["series"].to_numpy(dtype=float)),
    }
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **payload)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_case_cache(path: Path, observatory: str, case: Case) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            residual = np.asarray(z["residual"], dtype=float)
            known = np.asarray(z["known"], dtype=bool)
            active = np.asarray(z["active"], dtype=bool)
            storm = np.asarray(z["storm"], dtype=bool)
            if not (residual.size == known.size == active.size == storm.size):
                return None
            refs = {
                "known": known,
                "active": active,
                "storm": storm,
                "kp_known": np.asarray(z["kp_known"], dtype=bool),
                "dst_known": np.asarray(z["dst_known"], dtype=bool),
            }
            return {
                "observatory": observatory,
                "case": asdict(case),
                "series": pd.Series(np.asarray(z["series"], dtype=float)),
                "residual": residual,
                "cadence_s": float(z["cadence_s"][0]),
                "completeness": float(z["completeness"][0]),
                "refs": refs,
                "kp_coverage": float(z["kp_coverage"][0]),
                "dst_coverage": float(z["dst_coverage"][0]),
                "reference_coverage": float(z["reference_coverage"][0]),
                "kp_error": None,
                "cache_hit": True,
            }
    except (OSError, ValueError, KeyError, EOFError):
        return None


def load_case(observatory: str, case: Case, cache_dir: Path | None = None) -> Dict[str, Any]:
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    path = _cache_path(cache_dir, observatory, case)
    cached = _load_case_cache(path, observatory, case)
    if cached is not None:
        return cached

    raw = fetch_intermagnet_iaga2002(
        observatory=observatory,
        start_date=case.start_date,
        duration_days=case.days,
        samples_per_day="Minute",
    )
    df = parse_iaga2002_to_dataframe(raw)
    if df.empty or "f_nt" not in df.columns:
        raise RuntimeError("No usable total-field data returned.")
    series = pd.to_numeric(df["f_nt"], errors="coerce")
    valid_count = int(series.notna().sum())
    expected = max(1, int(case.days * 24 * 60))
    completeness = valid_count / expected
    if valid_count < max(24, int(expected * 0.50)):
        raise RuntimeError(f"Too few valid samples: {valid_count}/{expected} ({completeness:.1%}).")

    index = series.index
    cadence = index.to_series().diff().dropna().dt.total_seconds()
    cadence_s = float(cadence.median()) if not cadence.empty else 60.0
    if not math.isfinite(cadence_s) or cadence_s <= 0:
        raise RuntimeError("Invalid cadence in magnetometer data.")

    _, residual = causal_baseline.compute_causal_qdc_baseline(
        series.to_numpy(dtype=float), cadence_s
    )

    try:
        kp = _fetch_kp_cached(index[0].strftime("%Y-%m-%d"), index[-1].strftime("%Y-%m-%d"))
    except Exception:
        kp = pd.Series(dtype=float)

    dst_parts = []
    for period in pd.period_range(index[0].strftime("%Y-%m"), index[-1].strftime("%Y-%m"), freq="M"):
        try:
            part = _fetch_dst_cached(int(period.year), int(period.month))
        except Exception:
            part = None
        if part is not None and not part.empty:
            dst_parts.append(part)
    dst = pd.concat(dst_parts).sort_index() if dst_parts else pd.Series(dtype=float)

    target = pd.date_range(index[0], periods=len(index), freq=pd.Timedelta(seconds=cadence_s), tz="UTC")
    tolerance = pd.Timedelta("3h")
    kp_aligned = kp.reindex(target, method="ffill", tolerance=tolerance) if not kp.empty else pd.Series(np.nan, index=target)
    dst_aligned = dst.reindex(target, method="ffill", tolerance=tolerance) if not dst.empty else pd.Series(np.nan, index=target)
    refs = pm.reference_masks(kp_aligned, dst_aligned)
    data = {
        "observatory": observatory,
        "case": asdict(case),
        "series": series.reset_index(drop=True),
        "residual": residual,
        "cadence_s": cadence_s,
        "completeness": completeness,
        "refs": refs,
        "kp_coverage": float(refs["kp_known"].mean()),
        "dst_coverage": float(refs["dst_known"].mean()),
        "reference_coverage": float(refs["known"].mean()),
        "cache_hit": False,
    }
    _save_case_cache(path, data)
    return data


def score_case(data: Dict[str, Any], active_threshold: float, storm_threshold: float) -> Dict[str, Any]:
    return pm.score_thresholds(
        data["residual"], data["refs"], data["cadence_s"], active_threshold, storm_threshold
    )
