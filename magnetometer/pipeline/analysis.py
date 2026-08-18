"""Scientific pipeline orchestration built from the modular magnetometer components."""
from __future__ import annotations
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import requests
from ..baseline import build_design_matrix, robust_harmonic_baseline
from ..classification import cross_validate_flags, flag_activity
from ..metrics import MetricsEngine
from ..state import PipelineState, assess_health
from . import settings
LOGGER = logging.getLogger("magnetometer_pipeline")
_BASELINE_CACHE: OrderedDict[str, tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float]] = OrderedDict()

def _flag_counts(flags):
    labels, counts = np.unique(np.asarray(flags, dtype=object), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}

def _max_level(counts):
    for level in ("severe_storm", "major_storm", "minor_storm", "active", "unsettled", "quiet"):
        if counts.get(level, 0) > 0: return level
    return None

def _baseline_drift(coeffs, reference):
    if coeffs is None or reference is None or len(coeffs) == 0 or len(reference) == 0: return None
    return float(coeffs[0] - reference[0])

def _fit_baseline(x, cadence_s, kp, dst, state):
    key = None
    if settings.BASELINE_CACHE_ENABLED:
        key = str(hash((x.tobytes(), kp.tobytes(), dst.tobytes(), cadence_s)))
        cached = _BASELINE_CACHE.get(key)
        if cached is not None:
            _BASELINE_CACHE.move_to_end(key)
            baseline, uncomputed, coeffs, seed, frac = cached
            return baseline.copy(), uncomputed.copy(), coeffs, seed, frac
    n = len(x); baseline = np.zeros(n, dtype=float); weights = np.zeros(n, dtype=float)
    last_good = state.last_good_coeffs if state and state.is_fresh() else None
    seed = state.seed_coeffs if state and state.is_fresh() else None
    seed_frac = state.seed_storm_frac if state and state.is_fresh() else 1.0
    window = max(1, int(settings.BASELINE_WINDOW_HOURS * 3600 / cadence_s)); step = max(1, int(settings.BASELINE_STEP_HOURS * 3600 / cadence_s))
    t_global = np.arange(n) * cadence_s / 3600.0; design = build_design_matrix(t_global)
    for start in range(0, n, step):
        end = min(start + window, n)
        if end - start < max(10, step // 2): continue
        segment = x[start:end]
        if np.isfinite(segment).sum() < (end - start) * 0.5: continue
        kp_win, dst_win = kp[start:end], dst[start:end]
        global_storm = (np.any(np.isfinite(kp_win)) and float(np.nanmax(kp_win)) >= 5) or (np.any(np.isfinite(dst_win)) and float(np.nanmin(dst_win)) < -30)
        seg_base, coeffs = robust_harmonic_baseline(segment, cadence_s, n_iter=settings.BASELINE_N_ITER, outlier_threshold_nt=settings.BASELINE_OUTLIER_THRESHOLD_NT, t_hours=t_global[start:end], design_matrix=design[start:end])
        storm_frac = float(np.mean(np.abs(segment - seg_base) > 50))
        if storm_frac < seed_frac: seed_frac, seed = storm_frac, coeffs
        stormy = global_storm or storm_frac > settings.STORM_FRACTION_THRESHOLD
        if stormy and last_good is not None: seg_base = design[start:end] @ last_good
        elif not stormy: last_good = coeffs
        elif seed is not None: seg_base = design[start:end] @ seed
        w = np.hanning(end - start); w = np.ones(1) if len(w) == 1 else w
        baseline[start:end] += seg_base * w; weights[start:end] += w
    computed = weights > 0; baseline[computed] /= weights[computed]; missing = ~computed
    if np.any(missing):
        valid_idx = np.where(computed)[0]
        if len(valid_idx) >= 2: baseline[missing] = np.interp(np.where(missing)[0], valid_idx, baseline[valid_idx])
        elif np.any(np.isfinite(x)): baseline[missing] = np.nanmedian(x)
        else: baseline[missing] = 0.0
    if key is not None:
        _BASELINE_CACHE[key] = (baseline.copy(), missing.copy(), last_good, seed, seed_frac)
        while len(_BASELINE_CACHE) > max(1, settings.BASELINE_CACHE_SIZE): _BASELINE_CACHE.popitem(last=False)
    return baseline, missing, last_good, seed, seed_frac

def _alert(result, observatory):
    url = settings.ALERT_WEBHOOK_URL
    if not url: return
    counts = result.get("flag_counts") or {}; minimum = settings.STORM_LEVEL_ORDER.index(settings.ALERT_WEBHOOK_MIN_LEVEL)
    detected = any(counts.get(level, 0) > 0 for level in settings.STORM_LEVEL_ORDER[minimum:])
    failed = [name for name, ok in (result.get("health", {}).get("checks") or {}).items() if not ok]
    if not detected and result.get("status") != "insufficient_data" and not failed: return
    payload = {"ts": datetime.now(timezone.utc).isoformat(), "observatory": observatory, "status": result.get("status"), "storm_detected": detected, "failed_health_checks": failed}
    headers = {}; token = os.environ.get(settings.ALERT_TOKEN_ENV)
    if token: headers["Authorization"] = f"Bearer {token}"
    try: requests.post(url, json=payload, headers=headers, timeout=settings.ALERT_WEBHOOK_TIMEOUT_S).raise_for_status()
    except Exception as exc: LOGGER.warning("Alert webhook failed: %s", exc)

def run_analysis(x: np.ndarray, cadence_s: float, label: str = "", start_time: Optional[datetime] = None, analysis_start_time: Optional[datetime] = None, dst_series: Optional[pd.Series] = None, kp_series: Optional[pd.Series] = None, state: Optional[PipelineState] = None, dry_run: bool = False, live: bool = False, data_latency_min: Optional[float] = None, requested_coverage: Optional[float] = None, observatory: str = "-") -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    if x.size == 0: raise ValueError("No samples to analyse")
    input_nan_frac = float(np.mean(~np.isfinite(x))); index = None
    if start_time is not None: index = pd.date_range(start=start_time, periods=len(x), freq=pd.Timedelta(seconds=cadence_s), tz="UTC")
    kp_aligned = np.full(len(x), np.nan); dst_aligned = np.full(len(x), np.nan)
    if index is not None:
        if kp_series is not None: kp_aligned = kp_series.reindex(index, method="ffill", tolerance=pd.Timedelta(hours=3)).to_numpy()
        if dst_series is not None: dst_aligned = dst_series.reindex(index, method="ffill", tolerance=pd.Timedelta(hours=1)).to_numpy()
    if dry_run:
        return {"status":"dry_run","coverage":1.0-input_nan_frac,"median_fill_frac":0.0,"baseline":None,"residual":None,"flags":None,"validation":None,"metrics":{},"flag_counts":{},"validation_source":"none","health":assess_health(1.0-input_nan_frac,0.0,None,data_latency_min,requested_coverage,None,live)}
    baseline, uncomputed, last_good, seed, seed_frac = _fit_baseline(x,cadence_s,kp_aligned,dst_aligned,state)
    residual=x-baseline
    flags=flag_activity(residual,cadence_s,window_min=settings.FLAG_AMPLITUDE_WINDOW_MIN,mode=settings.FLAG_AMPLITUDE_MODE,centered=settings.FLAG_AMPLITUDE_CENTERED,unsettled_nt=settings.FLAG_THRESHOLD_UNSETTLED_NT,active_nt=settings.FLAG_THRESHOLD_ACTIVE_NT,minor_storm_nt=settings.FLAG_THRESHOLD_MINOR_STORM_NT,major_storm_nt=settings.FLAG_THRESHOLD_MAJOR_STORM_NT,severe_storm_nt=settings.FLAG_THRESHOLD_SEVERE_STORM_NT,anomaly_jump_nt=settings.FLAG_THRESHOLD_ANOMALY_JUMP_NT,max_plausible_nt=settings.MAX_PLAUSIBLE_RESIDUAL_NT,min_plausible_nt=settings.MIN_PLAUSIBLE_RESIDUAL_NT)
    start_idx=0
    if analysis_start_time is not None and index is not None:
        target = pd.to_datetime(analysis_start_time, utc=True)
        start_idx = int(np.searchsorted(index, target, side="left"))
    baseline_a,residual_a,flags_a,missing_a=baseline[start_idx:],residual[start_idx:],flags[start_idx:],uncomputed[start_idx:]
    kp_a,dst_a=kp_aligned[start_idx:],dst_aligned[start_idx:]
    coverage=float(np.mean(np.isfinite(residual_a))) if len(residual_a) else 0.0; fill_frac=float(np.mean(missing_a)) if len(missing_a) else 0.0
    validation=np.full(len(flags_a),"no_index_data",dtype=object); metrics={}; validation_source="none"
    if dst_series is not None or kp_series is not None:
        validation=cross_validate_flags(flags_a,dst_a,kp_a); metrics=MetricsEngine().compute(residual_a,flags_a,validation,kp_a,dst_a)
        validation_source="Kp+Dst" if dst_series is not None and kp_series is not None else "Kp-only" if kp_series is not None else "Dst-only"
    quiet=flags_a=="quiet"; quiet_rms=float(np.nanstd(residual_a[quiet])) if np.any(quiet) and np.any(np.isfinite(residual_a[quiet])) else None
    drift=_baseline_drift(last_good,state.last_good_coeffs if state else None)
    health=assess_health(coverage,fill_frac,quiet_rms,data_latency_min,requested_coverage,drift,live)
    result={"status":"insufficient_data" if coverage < settings.MIN_ANALYSIS_COVERAGE or fill_frac > settings.MAX_MEDIAN_FILL_FRACTION else "ok","coverage":coverage,"median_fill_frac":fill_frac,"baseline":baseline_a,"residual":residual_a,"flags":flags_a,"validation":validation,"metrics":metrics,"flag_counts":_flag_counts(flags_a),"max_local_level":_max_level(_flag_counts(flags_a)),"validation_source":validation_source,"health":health}
    _alert(result,observatory)
    if state is not None and settings.STATE_AUTO_SAVE:
        state.last_good_coeffs,last_good_coeffs = last_good,last_good
        state.seed_coeffs,state.seed_storm_frac = seed,seed_frac; state.observatory=observatory
    return result

def write_json_output(result,path):
    def safe(obj):
        if isinstance(obj,np.ndarray): return obj.tolist() if settings.OUTPUT_INCLUDE_ARRAYS else {"__type__":"ndarray","shape":list(obj.shape),"dtype":str(obj.dtype)}
        if isinstance(obj,(np.integer,np.floating)): return obj.item()
        if isinstance(obj,dict): return {k:safe(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)): return [safe(v) for v in obj]
        if isinstance(obj,datetime): return obj.isoformat()
        return obj
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); tmp=target.with_suffix(target.suffix+f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(safe(result),indent=2,default=str)); tmp.replace(target)
