#!/usr/bin/env python3
"""Adaptive production detector for geomagnetic activity.

The detector learns two calibrated binary models (active and storm) from
causal, station-local features derived from X/Y/Z/F magnetometer channels.
It uses only trailing information at inference time, then applies hysteresis
and persistence to reduce one-minute threshold flicker.

Training labels are Kp-derived reference states. Kp is a global reference, not
local ground truth; use the model as an operational activity detector and pair
it with station-K validation when available.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent

FEATURE_NAMES = [
    "abs_r_f",
    "abs_r_h",
    "abs_r_x",
    "abs_r_y",
    "abs_r_z",
    "dr_f",
    "dr_h",
    "dr_z",
    "std_h_15m",
    "std_h_60m",
    "range_h_60m",
    "mean_abs_r_h_15m",
    "mean_abs_dr_h_15m",
    "max_abs_dr_h_15m",
    "local_sin",
    "local_cos",
]

ACTIVE_PROB_DEFAULT = 0.45
STORM_PROB_DEFAULT = 0.40
ACTIVE_MINUTES = 10
STORM_MINUTES = 15
CLEAR_MINUTES = 10


@dataclass
class ModelArtifact:
    version: str
    feature_names: List[str]
    active_probability_threshold: float
    storm_probability_threshold: float
    active_persistence_minutes: int
    storm_persistence_minutes: int
    clear_persistence_minutes: int
    active_model: Dict[str, object]
    storm_model: Dict[str, object]


def _component_features(series: pd.Series, baseline: pd.Series, cadence_s: float) -> pd.DataFrame:
    r = series - baseline
    dt = max(float(cadence_s), 1.0)
    dr = r.diff().fillna(0.0) / dt * 60.0
    return r, dr


def build_features(df: pd.DataFrame, baselines: Dict[str, np.ndarray], cadence_s: float) -> pd.DataFrame:
    idx = df.index
    out = pd.DataFrame(index=idx)
    residuals = {}
    derivatives = {}

    for name in ("x_nt", "y_nt", "z_nt", "f_nt"):
        baseline = pd.Series(baselines[name], index=idx)
        series = pd.to_numeric(df[name], errors="coerce")
        residuals[name], derivatives[name] = _component_features(series, baseline, cadence_s)

    h = np.sqrt(residuals["x_nt"] ** 2 + residuals["y_nt"] ** 2)
    dh = np.sqrt(derivatives["x_nt"] ** 2 + derivatives["y_nt"] ** 2)

    out["abs_r_f"] = residuals["f_nt"].abs()
    out["abs_r_h"] = h.abs()
    out["abs_r_x"] = residuals["x_nt"].abs()
    out["abs_r_y"] = residuals["y_nt"].abs()
    out["abs_r_z"] = residuals["z_nt"].abs()
    out["dr_f"] = derivatives["f_nt"].abs()
    out["dr_h"] = dh
    out["dr_z"] = derivatives["z_nt"].abs()

    for minutes, label in ((15, "15m"), (60, "60m")):
        window = max(2, int(round(minutes * 60 / cadence_s)))
        out[f"std_h_{label}"] = h.rolling(window, min_periods=max(2, window // 3)).std()
    out["range_h_60m"] = h.rolling(max(2, int(round(3600 / cadence_s))), min_periods=2).max() - h.rolling(max(2, int(round(3600 / cadence_s))), min_periods=2).min()
    out["mean_abs_r_h_15m"] = h.abs().rolling(max(2, int(round(900 / cadence_s))), min_periods=2).mean()
    out["mean_abs_dr_h_15m"] = dh.rolling(max(2, int(round(900 / cadence_s))), min_periods=2).mean()
    out["max_abs_dr_h_15m"] = dh.rolling(max(2, int(round(900 / cadence_s))), min_periods=2).max()

    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0 + idx.second.to_numpy() / 3600.0
    out["local_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["local_cos"] = np.cos(2 * np.pi * hours / 24.0)

    return out[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).ffill().bfill()


def _model_payload(pipe: Pipeline) -> Dict[str, object]:
    scaler = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": clf.classes_.tolist(),
        "C": float(clf.C),
    }


def _fit_pipeline(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.75, class_weight=None, max_iter=1500, solver="lbfgs")),
    ])
    pipe.fit(x, y, clf__sample_weight=sample_weight)
    return pipe


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = np.ones_like(y, dtype=float)
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[y == cls] = len(y) / (2.0 * counts[cls])
    return weights


def train(feature_frames: Sequence[pd.DataFrame], active_labels: Sequence[np.ndarray], storm_labels: Sequence[np.ndarray], active_threshold: float = ACTIVE_PROB_DEFAULT, storm_threshold: float = STORM_PROB_DEFAULT) -> ModelArtifact:
    x = pd.concat(feature_frames, axis=0)
    active_y = np.concatenate(active_labels).astype(int)
    storm_y = np.concatenate(storm_labels).astype(int)
    valid = np.isfinite(x.to_numpy()).all(axis=1)
    x_np = x.to_numpy(dtype=float)[valid]
    active_y = active_y[valid]
    storm_y = storm_y[valid]

    active_pipe = _fit_pipeline(x_np, active_y, _balanced_weights(active_y))
    storm_pipe = _fit_pipeline(x_np, storm_y, _balanced_weights(storm_y))

    return ModelArtifact(
        version="1.0",
        feature_names=list(FEATURE_NAMES),
        active_probability_threshold=float(active_threshold),
        storm_probability_threshold=float(storm_threshold),
        active_persistence_minutes=ACTIVE_MINUTES,
        storm_persistence_minutes=STORM_MINUTES,
        clear_persistence_minutes=CLEAR_MINUTES,
        active_model=_model_payload(active_pipe),
        storm_model=_model_payload(storm_pipe),
    )


def _predict_linear(payload: Dict[str, object], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(payload["mean"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    coef = np.asarray(payload["coef"], dtype=float)[0]
    intercept = float(np.asarray(payload["intercept"], dtype=float)[0])
    z = (x - mean) / np.where(scale == 0.0, 1.0, scale)
    logits = z @ coef + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def _persistent_state(prob: np.ndarray, threshold: float, on_minutes: int, off_minutes: int, cadence_s: float) -> np.ndarray:
    n = len(prob)
    out = np.zeros(n, dtype=bool)
    on_count = 0
    off_count = 0
    on_needed = max(1, int(round(on_minutes * 60 / cadence_s)))
    off_needed = max(1, int(round(off_minutes * 60 / cadence_s)))
    state = False
    for i, p in enumerate(prob):
        if not state:
            if p >= threshold:
                on_count += 1
                if on_count >= on_needed:
                    state = True
                    off_count = 0
            else:
                on_count = 0
        else:
            if p < threshold * 0.80:
                off_count += 1
                if off_count >= off_needed:
                    state = False
                    on_count = 0
            else:
                off_count = 0
        out[i] = state
    return out


def predict(artifact: ModelArtifact, features: pd.DataFrame, cadence_s: float) -> Dict[str, np.ndarray]:
    x = features[artifact.feature_names].to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    active_prob = _predict_linear(artifact.active_model, x)
    storm_prob = _predict_linear(artifact.storm_model, x)
    active = _persistent_state(active_prob, artifact.active_probability_threshold, artifact.active_persistence_minutes, artifact.clear_persistence_minutes, cadence_s)
    storm = _persistent_state(storm_prob, artifact.storm_probability_threshold, artifact.storm_persistence_minutes, artifact.clear_persistence_minutes, cadence_s)
    # Storm always dominates active.
    active = active & ~storm
    return {
        "active_probability": active_prob,
        "storm_probability": storm_prob,
        "active": active,
        "storm": storm,
    }


def save_artifact(artifact: ModelArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(artifact), indent=2))


def load_artifact(path: Path) -> ModelArtifact:
    return ModelArtifact(**json.loads(path.read_text()))
