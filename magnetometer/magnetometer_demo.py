#!/usr/bin/env python3
"""
Magnetometer Pipeline — Production-Ready Edition.

Processes magnetometer time-series using robust harmonic quiet-day curves (QDC),
residual analysis, deterministic multi-timescale activity classification, and
global index cross-validation (Kp and Dst).
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy import linalg

from detector_core import flag_activity as _production_flag_activity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("magnetometer_pipeline")

INTERMAGNET_BASE = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
KP_GFZ_URL = "https://kp.gfz-potsdam.de/app/json/"
DEFAULT_OBSERVATORY = "VIC"
DEFAULT_SAMPLES_PER_DAY = "Minute"
USER_AGENT = "MagnetometerProductionPipeline/1.2"

PROD_UNSETTLED_NT = 10.0
PROD_ACTIVE_NT = 15.0
PROD_MINOR_STORM_NT = 35.0
PROD_MAJOR_STORM_NT = 100.0
PROD_SEVERE_STORM_NT = 200.0
ANOMALY_DELTA_NT = 100.0


def create_resilient_session(retries: int = 3, backoff_factor: float = 1.0, status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504)) -> requests.Session:
    session = requests.Session(); session.headers.update({"User-Agent": USER_AGENT})
    retry_strategy = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist, raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry_strategy); session.mount("http://", adapter); session.mount("https://", adapter)
    return session


HTTP_CLIENT = create_resilient_session()


def fetch_intermagnet_iaga2002(observatory: str = DEFAULT_OBSERVATORY, start_date: Optional[str] = None, duration_days: int = 7, samples_per_day: str = DEFAULT_SAMPLES_PER_DAY) -> str:
    if start_date is None: start_date = "2024-01-01"
    if duration_days <= 0: raise ValueError("duration_days must be positive")
    params = {"Request": "GetData", "observatoryIagaCode": observatory, "samplesPerDay": samples_per_day, "dataStartDate": start_date, "dataDuration": duration_days, "format": "iaga2002", "orientation": "XYZF"}
    logger.info(f"Fetching INTERMAGNET data for {observatory} from {start_date} ({duration_days} days)...")
    resp = HTTP_CLIENT.get(INTERMAGNET_BASE, params=params, timeout=60); resp.raise_for_status(); return resp.text


def parse_iaga2002_to_dataframe(text: str) -> pd.DataFrame:
    if not text or text.strip().startswith(("<", "<!DOCTYPE", "<html")): raise ValueError("INTERMAGNET returned HTML/empty content instead of IAGA-2002 data.")
    lines = text.splitlines(); data_lines = []; col_names = None
    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("#"): continue
        if line_s.startswith("DATE"): col_names = line_s.replace("|", "").split()
        elif line_s[0].isdigit(): data_lines.append(line_s)
    if not col_names or len(col_names) < 7: raise ValueError("Could not parse IAGA-2002 headers.")
    def find_col(key: str) -> Optional[int]:
        for i, name in enumerate(col_names):
            if name.upper().endswith(key.upper()) and len(name) == 4: return i
        return None
    idx_x, idx_y, idx_z, idx_f = find_col("X"), find_col("Y"), find_col("Z"), find_col("F")
    rows = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < len(col_names): continue
        try: dt = pd.to_datetime(f"{parts[0]} {parts[1]}", utc=True)
        except (ValueError, TypeError): continue
        def parse_val(idx: Optional[int]) -> float:
            if idx is not None and idx < len(parts):
                try:
                    v = float(parts[idx]); return np.nan if abs(v) >= 99999 else v
                except ValueError: return np.nan
            return np.nan
        rows.append({"datetime": dt, "x_nt": parse_val(idx_x), "y_nt": parse_val(idx_y), "z_nt": parse_val(idx_z), "f_nt": parse_val(idx_f)})
    if not rows: raise ValueError("IAGA-2002 response contained no usable data rows.")
    return pd.DataFrame(rows).set_index("datetime").sort_index()


def fetch_kp_gfz(start_date: str, end_date: str) -> pd.Series:
    url = f"{KP_GFZ_URL}?start={start_date}T00:00:00Z&end={end_date}T23:59:59Z&index=Kp"; logger.info("Fetching Kp index from GFZ Potsdam...")
    resp = HTTP_CLIENT.get(url, timeout=30); resp.raise_for_status(); data = resp.json(); rows = [{"datetime": pd.to_datetime(ts, utc=True), "kp": float(val)} for ts, val in zip(data["datetime"], data["Kp"])]
    if not rows: raise ValueError("GFZ returned no Kp records")
    return pd.DataFrame(rows).set_index("datetime")["kp"].sort_index()


def fetch_dst_kyoto(year: int, month: int) -> Optional[pd.Series]:
    yy, mm = year % 100, month
    urls_to_try = [f"https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for", f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for", f"https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for"]
    text = None
    for url in urls_to_try:
        try:
            resp = HTTP_CLIENT.get(url, timeout=15)
            if resp.status_code == 200 and "Not Found" not in resp.text and "<html" not in resp.text.lower(): text = resp.text; break
        except requests.RequestException: continue
    if text is None:
        logger.warning(f"Dst index unavailable for {year:04d}-{mm:02d} from Kyoto WDC (server down or restricted). Skipping Dst."); return None
    rows = []
    for line in text.splitlines():
        if len(line) >= 116 and (line[:3].strip().isdigit() or line[3:5].strip().isdigit()):
            try:
                day = int(line[8:10].strip()); hourly_part = line[20:116]
                for hour in range(24):
                    val_str = hourly_part[hour * 4:(hour + 1) * 4].strip()
                    if val_str and val_str != "9999": rows.append({"datetime": datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc), "dst": int(val_str)})
            except (ValueError, IndexError): continue
    return pd.DataFrame(rows).set_index("datetime")["dst"].sort_index() if rows else None


def handle_gaps(series: pd.Series, max_gap_samples: int = 3) -> pd.Series:
    if series.empty: return series
    series = pd.to_numeric(series, errors="coerce"); deltas = series.index.to_series().diff().dropna()
    if deltas.empty: return series
    freq_s = max(1, int(deltas.median().total_seconds())); regular_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=pd.Timedelta(seconds=freq_s), tz="UTC")
    return series.reindex(regular_index).interpolate(method="linear", limit=max_gap_samples, limit_direction="both")


def build_design_matrix(t_hours: np.ndarray, t_ref_min: float, t_ref_max: float) -> np.ndarray:
    t = np.asarray(t_hours, dtype=float); t_norm = (t - t_ref_min) / (t_ref_max - t_ref_min + 1e-12); t_norm_clamped = np.clip(t_norm, -0.5, 1.5)
    return np.column_stack([np.ones_like(t), t_norm_clamped, t_norm_clamped ** 2, np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24), np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12), np.sin(2 * np.pi * t / 8), np.cos(2 * np.pi * t / 8), np.sin(2 * np.pi * t / 6), np.cos(2 * np.pi * t / 6)])


def robust_harmonic_baseline(x: np.ndarray, cadence_s: float, n_iter: int = 4, outlier_threshold_nt: float = 30.0, t_hours: Optional[np.ndarray] = None, t_ref_min: Optional[float] = None, t_ref_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float); n = len(x)
    if n == 0: return np.empty(0), np.empty(0)
    if t_hours is None: t_hours = np.arange(n) * cadence_s / 3600.0
    ref_min = t_hours.min() if t_ref_min is None else t_ref_min; ref_max = t_hours.max() if t_ref_max is None else t_ref_max; A = build_design_matrix(t_hours, ref_min, ref_max); valid = np.isfinite(x); w = np.ones(n); w[~valid] = 0.0; coeffs = np.zeros(A.shape[1])
    for _ in range(max(1, n_iter)):
        if valid.sum() < A.shape[1]: break
        Aw = A * w[:, np.newaxis]; xw = np.nan_to_num(x, nan=0.0) * w; coeffs, *_ = linalg.lstsq(Aw[valid], xw[valid])[:2]; resid = x - A @ coeffs; fr = resid[valid]
        if not fr.size: break
        mad = np.median(np.abs(fr - np.median(fr))); sigma = 1.4826 * mad + 1e-12; w = np.ones(n); w[~valid] = 0.0; w[np.abs(resid) > outlier_threshold_nt] = 0.1; w[np.abs(resid) > 3 * sigma] = 0.01
    return A @ coeffs, coeffs


def flag_activity(residual: np.ndarray, cadence_s: float = 60.0, active_threshold: Optional[float] = None, storm_threshold: Optional[float] = None) -> np.ndarray:
    """Live/demo wrapper: omitted thresholds mean use the certified detector profile."""
    kwargs = {"cadence_s": cadence_s, "unsettled_threshold": PROD_UNSETTLED_NT, "major_threshold": PROD_MAJOR_STORM_NT, "severe_threshold": PROD_SEVERE_STORM_NT}
    if active_threshold is not None: kwargs["active_threshold"] = active_threshold
    if storm_threshold is not None: kwargs["storm_threshold"] = storm_threshold
    return _production_flag_activity(residual, **kwargs)


def cross_validate_flags(local_flags: np.ndarray, dst_vals: np.ndarray, kp_vals: np.ndarray) -> np.ndarray:
    validation = np.full(len(local_flags), "ok", dtype=object)
    for i in range(len(local_flags)):
        local = local_flags[i]; dst = dst_vals[i] if np.isfinite(dst_vals[i]) else None; kp = kp_vals[i] if np.isfinite(kp_vals[i]) else None; global_main_phase = (dst is not None and dst < -50) or (kp is not None and kp >= 6); global_active = (dst is not None and dst < -30) or (kp is not None and kp >= 4)
        if local == "quiet" and global_main_phase: validation[i] = "missed_global_event"
        elif local in ("quiet", "unsettled") and global_active: validation[i] = "under_reacting"
        elif local in ("major_storm", "severe_storm") and not global_active: validation[i] = "unconfirmed_storm"
    return validation


def run_analysis(x: np.ndarray, cadence_s: float, label: str = "", start_time: Optional[datetime] = None, dst_series: Optional[pd.Series] = None, kp_series: Optional[pd.Series] = None) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float); n = len(x); logger.info(f"Running Analysis: {label} ({n} samples, {n * cadence_s / 3600:.1f} hours)"); baseline = np.zeros(n); weights = np.zeros(n); window_samples = max(1, int(24 * 3600 / max(cadence_s, 1.0))); step_samples = max(1, window_samples // 2); t_global = np.arange(n) * cadence_s / 3600.0; t_min, t_max = (t_global.min(), t_global.max()) if n else (0.0, 0.0); last_good_coeffs = None
    for start in range(0, max(1, n - step_samples + 1), step_samples):
        end = min(start + window_samples, n)
        if end - start < max(1, step_samples // 2): continue
        segment = x[start:end]; t_seg = t_global[start:end]
        if np.isfinite(segment).sum() < (end - start) * 0.5: continue
        seg_base, coeffs = robust_harmonic_baseline(segment, cadence_s, t_hours=t_seg, t_ref_min=t_min, t_ref_max=t_max); seg_res = segment - seg_base; finite_seg = seg_res[np.isfinite(seg_res)]; storm_frac = float(np.mean(np.abs(finite_seg) > 50.0)) if finite_seg.size else 0.0
        if storm_frac > 0.05 and last_good_coeffs is not None: seg_base = build_design_matrix(t_seg, t_min, t_max) @ last_good_coeffs; logger.info(f"Storm window detected [{start}:{end}]. Extrapolating quiet baseline.")
        elif storm_frac <= 0.05: last_good_coeffs = coeffs
        w_win = np.hanning(end - start); baseline[start:end] += seg_base * w_win; weights[start:end] += w_win
    mask = weights > 0; fallback = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0; baseline[mask] /= weights[mask]; baseline[~mask] = fallback; residual = x - baseline; flags = flag_activity(residual, cadence_s=cadence_s)
    if dst_series is not None or kp_series is not None:
        index = pd.date_range(start=start_time, periods=n, freq=pd.Timedelta(seconds=cadence_s), tz="UTC"); df_align = pd.DataFrame(index=index); df_align["dst"] = dst_series.reindex(index, method="ffill", tolerance=pd.Timedelta("3h")) if dst_series is not None else np.nan; df_align["kp"] = kp_series.reindex(index, method="ffill", tolerance=pd.Timedelta("3h")) if kp_series is not None else np.nan; validation = cross_validate_flags(flags, df_align["dst"].values, df_align["kp"].values)
    else: validation = np.full(n, "no_index_data", dtype=object)
    logger.info("Activity Flag Breakdown:")
    for u, c in zip(*np.unique(flags, return_counts=True)): logger.info(f"  {u:12s}: {c}")
    quiet_mask = flags == "quiet"; fr = residual[np.isfinite(residual)]
    if np.any(quiet_mask & np.isfinite(residual)): logger.info(f"Quiet Period Residual RMS: {np.std(residual[quiet_mask & np.isfinite(residual)]):.2f} nT")
    if fr.size: logger.info(f"Overall Residual RMS: {np.std(fr):.2f} nT"); logger.info(f"Residual Range: {np.min(fr):.2f} nT to {np.max(fr):.2f} nT")
    return {"baseline": baseline, "residual": residual, "flags": flags, "validation": validation}


def main():
    ap = argparse.ArgumentParser(description="Magnetometer Pipeline."); ap.add_argument("--fetch-real-data", action="store_true"); ap.add_argument("--observatory", default=DEFAULT_OBSERVATORY); ap.add_argument("--days", type=int, default=7); ap.add_argument("--start-date", default=None); ap.add_argument("--cadence-s", type=int, default=60); ap.add_argument("--column", default="x_nt"); ap.add_argument("--cross-check-indices", action="store_true"); args = ap.parse_args()
    if not args.fetch_real_data: logger.error("Must supply --fetch-real-data"); sys.exit(1)
    iaga_text = fetch_intermagnet_iaga2002(observatory=args.observatory, start_date=args.start_date, duration_days=args.days); df = parse_iaga2002_to_dataframe(iaga_text)
    if args.column not in df.columns: raise SystemExit(f"Unknown data column: {args.column}; available={list(df.columns)}")
    x = handle_gaps(df[args.column], max_gap_samples=3).values; dst_series, kp_series = None, None
    if args.cross_check_indices:
        start_dt = pd.to_datetime(df.index.min()); end_dt = pd.to_datetime(df.index.max())
        try: kp_series = fetch_kp_gfz(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        except Exception as exc: logger.warning(f"Kp cross-validation disabled: {exc}")
        try:
            months = sorted({(dt.year, dt.month) for dt in pd.date_range(start_dt, end_dt, freq="D")}); dst_parts = [fetch_dst_kyoto(y, m) for y, m in months]; valid_dst = [p for p in dst_parts if p is not None]
            if valid_dst: dst_series = pd.concat(valid_dst).sort_index()
        except Exception as exc: logger.warning(f"Dst cross-validation disabled: {exc}")
    run_analysis(x, args.cadence_s, label=f"INTERMAGNET {args.observatory}", start_time=pd.to_datetime(df.index.min()).to_pydatetime(), dst_series=dst_series, kp_series=kp_series)


if __name__ == "__main__": main()
