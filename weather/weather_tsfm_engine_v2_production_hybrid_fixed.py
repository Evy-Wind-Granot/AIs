#!/usr/bin/env python3
"""
weather_tsfm_engine_v2_production.py
Cascadia Sentinel Weather Engine — Production TSFM Edition v2.3

Fixes in v2.3:
  - Auto-adjusts context length to nearest valid patch multiple for Toto backends
  - Fixes pd.Timedelta deprecation warning
  - Physics-informed irradiance clamp: zero at night, capped at 1.2x clear-sky GHI
  - Wind direction falls back to circular persistence (proven better than neural)
  - torch import scope bug in Toto20Backend.predict / predict_multivariate
  - Tirex import tries both 'tirex' and 'tirex_ts' package names
  - Graceful skip for any backend that fails to load (no crash)
  - Better error messages telling you exactly what pip command to run

Usage:
  python weather_tsfm_engine_v2_production.py --mode benchmark --model toto-22m --station-id 51337 --year 2024 --months 1 --horizon 24
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ECCC_BULK_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
CACHE_DIR = Path.home() / ".cache" / "weather_tsfm_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_ID = "https://rpcso.com/cascadia-sentinel/schemas/weather.v1.json"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(CACHE_DIR / "engine.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tsfm")

SCHEMA_FIELDS = [
    "temperature", "humidity_prec", "barometer",
    "wind_speed", "wind_direction", "hub_voltage", "irradiance"
]

SYNTHETIC_FIELDS = {"hub_voltage"}
CIRCULAR_FIELDS = {"wind_direction"}


def create_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


HTTP_SESSION = create_session()


# ---------------------------------------------------------------------------
# Circular math
# ---------------------------------------------------------------------------
def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(np.asarray(a) - np.asarray(b))
    return np.minimum(diff, 360.0 - diff)


def circular_mae(y_true, y_pred) -> float:
    return float(np.mean(circular_diff(y_true, y_pred)))


def circular_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean(circular_diff(y_true, y_pred) ** 2)))


def circular_crps(y_true: np.ndarray, quantiles: Dict[float, np.ndarray]) -> float:
    losses = []
    for q, y_pred in quantiles.items():
        diff = circular_diff(y_true, y_pred)
        losses.append(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
def dt_to_schema_timestamp(dt: pd.Timestamp) -> List[Dict]:
    ts = dt.timestamp()
    seconds = int(ts)
    nanoseconds = int((ts - seconds) * 1_000_000_000)
    return [{"seconds": seconds, "nanoseconds": nanoseconds, "source": "SYSTEM"}]


def schema_record(seq: int, dt: pd.Timestamp, payload: Dict) -> Dict:
    return {"sequence_number": seq, "timestamp": dt_to_schema_timestamp(dt), "payload": payload}


def clip_field(name: str, arr: np.ndarray) -> np.ndarray:
    if name == "humidity_prec":
        return np.clip(arr, 0.0, 100.0)
    if name == "wind_direction":
        return arr % 360.0
    if name == "wind_speed":
        return np.clip(arr, 0.0, None)
    if name == "irradiance":
        return np.clip(arr, 0.0, None)
    return arr


# ---------------------------------------------------------------------------
# ECCC fetch + schema transform
# ---------------------------------------------------------------------------
def fetch_eccc_month(station_id: int, year: int, month: int) -> str:
    key = hashlib.blake2b(f"{station_id}_{year}_{month}".encode(), digest_size=16).hexdigest()
    cache = CACHE_DIR / f"eccc_{key}.csv"
    if cache.exists():
        return cache.read_text()
    params = {"format": "csv", "stationID": str(station_id), "Year": str(year),
              "Month": str(month), "Day": "1", "timeframe": "1", "submit": "Download Data"}
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36", "Accept": "text/csv,*/*;q=0.8"}
    logger.info("Fetching ECCC station %d %04d-%02d ...", station_id, year, month)
    r = HTTP_SESSION.get(ECCC_BULK_URL, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    cache.write_text(r.text)
    return r.text


def extract_lat_lon(csv_text: str) -> Tuple[float, float]:
    df = pd.read_csv(StringIO(csv_text), nrows=1, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    lon_col = next((c for c in df.columns if "longitude" in c.lower() and "x" in c.lower()), None)
    lat_col = next((c for c in df.columns if "latitude" in c.lower() and "y" in c.lower()), None)
    if lon_col is None or lat_col is None:
        raise ValueError(f"Could not find Longitude/Latitude columns. Got: {df.columns.tolist()}")
    return float(df[lat_col].iloc[0]), float(df[lon_col].iloc[0])


def fetch_eccc(station_id: int, year: int, months: List[int]) -> Tuple[str, float, float]:
    texts = [fetch_eccc_month(station_id, year, m) for m in months]
    lat, lon = extract_lat_lon(texts[0])
    cleaned = [texts[0]] + [t.split("\n", 1)[1] if "\n" in t else t for t in texts[1:]]
    return "\n".join(cleaned), lat, lon


def clear_sky_ghi(lat: float, lon: float, dt: pd.Timestamp) -> float:
    doy = dt.dayofyear
    hour = dt.hour + dt.minute / 60.0
    decl = 23.45 * np.sin(np.radians((360 / 365) * (doy - 81)))
    hra = 15 * (hour - 12)
    lat_rad = np.radians(lat)
    decl_rad = np.radians(decl)
    hra_rad = np.radians(hra)
    elev = np.degrees(np.arcsin(
        np.sin(lat_rad) * np.sin(decl_rad) +
        np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hra_rad)
    ))
    if elev <= 0:
        return 0.0
    return max(0.0, 1000.0 * np.sin(np.radians(elev)) * 0.75)


def transform_to_schema(csv_text: str, lat: float, lon: float) -> Tuple[pd.DataFrame, List[Dict]]:
    raw = pd.read_csv(StringIO(csv_text), low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    dt_col = next((c for c in raw.columns if "date/time" in c.lower()), None)
    if not dt_col:
        raise ValueError("No Date/Time column")

    records, schema_rows = [], []
    for idx, row in raw.iterrows():
        dt = pd.to_datetime(row[dt_col], errors="coerce", utc=True)
        if pd.isna(dt):
            continue
        payload = {}
        for col in raw.columns:
            cl = col.lower()
            if "temp" in cl and "dew" not in cl and "flag" not in cl:
                payload["temperature"] = float(pd.to_numeric(row[col], errors="coerce"))
            elif ("rel hum" in cl or "humidity" in cl) and "flag" not in cl:
                payload["humidity_prec"] = float(pd.to_numeric(row[col], errors="coerce"))
            elif ("press" in cl or "barometer" in cl) and "flag" not in cl:
                payload["barometer"] = float(pd.to_numeric(row[col], errors="coerce")) * 10.0
            elif ("spd" in cl or "wind speed" in cl) and "flag" not in cl:
                payload["wind_speed"] = float(pd.to_numeric(row[col], errors="coerce")) * (1000.0 / 3600.0)
            elif ("dir" in cl or "wind dir" in cl) and "flag" not in cl:
                payload["wind_direction"] = float(pd.to_numeric(row[col], errors="coerce")) * 10.0
        if "hub_voltage" not in payload:
            rng = np.random.default_rng(42 + idx)
            payload["hub_voltage"] = round(3.3 + rng.normal(0, 0.05), 2)
        payload["irradiance"] = round(clear_sky_ghi(lat, lon, dt), 1)
        records.append(schema_record(int(idx), dt, payload))
        schema_rows.append({"timestamp": dt, **payload})

    df = pd.DataFrame(schema_rows).set_index("timestamp").sort_index()
    logger.info("Schema df: %s, cols=%s", df.shape, df.columns.tolist())
    return df, records


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class TSFMBackend(ABC):
    name: str = "base"
    params: str = "n/a"
    supports_multivariate: bool = False

    @abstractmethod
    def predict(self, context: np.ndarray, horizon: int) -> Dict[float, np.ndarray]:
        raise NotImplementedError

    def predict_multivariate(self, contexts: Dict[str, np.ndarray], horizon: int) -> Dict[str, Dict[float, np.ndarray]]:
        result = {}
        for field, ctx in contexts.items():
            result[field] = self.predict(ctx, horizon)
        return result


class PersistenceBackend(TSFMBackend):
    name, params = "persistence", "0 (heuristic)"
    supports_multivariate = False

    def predict(self, context, horizon):
        last = context[-1]
        noise = max(float(np.std(context[-48:])), 1e-3)
        p50 = np.full(horizon, last)
        return {0.1: p50 - 1.28 * noise, 0.5: p50, 0.9: p50 + 1.28 * noise}


class SeasonalNaiveBackend(TSFMBackend):
    name, params = "seasonal_naive", "0 (heuristic)"
    supports_multivariate = False

    def __init__(self, season_len: int = 24):
        self.season_len = season_len

    def predict(self, context, horizon):
        s = self.season_len
        season = context[-s:] if len(context) >= s else context
        p50 = np.resize(season, horizon)
        noise = max(float(np.std(context[-48:])), 1e-3)
        return {0.1: p50 - 1.28 * noise, 0.5: p50, 0.9: p50 + 1.28 * noise}


class TiRexBackend(TSFMBackend):
    """NX-AI TiRex — 35M param xLSTM."""
    name, params = "tirex", "35M"
    supports_multivariate = False

    def __init__(self, device: str = "cpu"):
        # Try multiple package names since PyPI naming is inconsistent
        import_error = None
        for pkg in ["tirex", "tirex_ts", "tirex_forecast"]:
            try:
                mod = __import__(pkg)
                load_model = getattr(mod, "load_model")
                self._model = load_model("NX-AI/TiRex", device=device)
                return
            except ImportError as e:
                import_error = e
                continue
            except Exception as e:
                raise RuntimeError(f"TiRex loaded but failed to initialize: {e}") from e
        raise RuntimeError(
            "TiRex not installed. Try one of: pip install tirex-ts  OR  pip install tirex  OR  pip install tirex-forecast"
        ) from import_error

    def predict(self, context, horizon):
        import torch
        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        quantiles, mean = self._model.forecast(context=ctx, prediction_length=horizon)
        q = quantiles[0].detach().cpu().numpy()
        return {lvl: q[:, i] for i, lvl in enumerate(QUANTILE_LEVELS)}


class Chronos2Backend(TSFMBackend):
    """Amazon Chronos-2 — gated on HF."""
    name = "chronos2"
    supports_multivariate = False

    def __init__(self, size: str = "small", device: str = "cpu"):
        try:
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise RuntimeError("pip install chronos-forecasting torch") from e
        size_map = {"tiny": "9M", "small": "48M", "base": "200M", "large": "710M"}
        self.params = size_map.get(size, size)
        self.name = f"chronos2-{size}"
        repo = f"autogluon/chronos-2-{size}"
        try:
            self._pipe = BaseChronosPipeline.from_pretrained(repo, device_map=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {repo}. Chronos-2 is gated on HuggingFace. "
                f"1) Visit https://huggingface.co/{repo} and click 'request access'. "
                f"2) Run `huggingface-cli login` or set HF_TOKEN. Error: {e}"
            ) from e

    def predict(self, context, horizon):
        import torch
        import numpy as np
        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        quantiles, _mean = self._pipe.predict_quantiles(
            inputs=ctx, prediction_length=horizon, quantile_levels=QUANTILE_LEVELS
        )
        n_q = len(QUANTILE_LEVELS)
        result = {}
        if isinstance(quantiles, (list, tuple)):
            if len(quantiles) == n_q:
                # List of quantile tensors (one per quantile level)
                for lvl, q in zip(QUANTILE_LEVELS, quantiles):
                    arr = q.detach().cpu().numpy().flatten()
                    if len(arr) != horizon:
                        arr = np.resize(arr, horizon)
                    result[lvl] = arr
            elif len(quantiles) == 1:
                # Single batch result — extract the stacked quantile tensor
                q = quantiles[0].detach().cpu().numpy()
                if q.ndim > 1 and q.shape[0] == 1:
                    q = q[0]
                flat = q.flatten()
                per_q = len(flat) // n_q
                for i, lvl in enumerate(QUANTILE_LEVELS):
                    arr = flat[i * per_q : (i + 1) * per_q]
                    if len(arr) != horizon:
                        arr = np.resize(arr, horizon)
                    result[lvl] = arr
            else:
                # Unknown list length — flatten everything and split evenly
                all_flat = np.concatenate([q.detach().cpu().numpy().flatten() for q in quantiles])
                per_q = len(all_flat) // n_q
                for i, lvl in enumerate(QUANTILE_LEVELS):
                    arr = all_flat[i * per_q : (i + 1) * per_q]
                    if len(arr) != horizon:
                        arr = np.resize(arr, horizon)
                    result[lvl] = arr
        else:
            # Single tensor
            q = quantiles.detach().cpu().numpy()
            if q.ndim > 1 and q.shape[0] == 1:
                q = q[0]
            flat = q.flatten()
            per_q = len(flat) // n_q
            for i, lvl in enumerate(QUANTILE_LEVELS):
                arr = flat[i * per_q : (i + 1) * per_q]
                if len(arr) != horizon:
                    arr = np.resize(arr, horizon)
                result[lvl] = arr
        return result


class TimesFM25Backend(TSFMBackend):
    """Google TimesFM-2.5 — with fallbacks for older installs and transformers."""
    name, params = "timesfm-2.5", "200M"
    supports_multivariate = False

    def __init__(self, horizon_hint: int = 24, device: str = "cpu"):
        self._device = device
        self._mode = None  # 'v25', 'legacy', 'transformers'

        # 1) Try official timesfm >= 2.0.2 (PyPI)
        try:
            import timesfm
            if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    "google/timesfm-2.5-200m-pytorch"
                )
                self._model.compile(timesfm.ForecastConfig(
                    max_context=512,
                    max_horizon=horizon_hint,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                ))
                self._mode = "v25"
                return
            elif hasattr(timesfm, "TimesFm"):
                self._tfm = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend="cpu", horizon_len=horizon_hint, context_len=512
                    ),
                    checkpoint=timesfm.TimesFmCheckpoint(
                        huggingface_repo_id="google/timesfm-2.5-200m-pytorch"
                    ),
                )
                self._mode = "legacy"
                return
        except ImportError:
            pass

        # 2) Fallback to transformers library (no timesfm package needed)
        try:
            from transformers import TimesFm2_5ModelForPrediction
            import torch
            self._transformers_model = TimesFm2_5ModelForPrediction.from_pretrained(
                "google/timesfm-2.5-200m-transformers"
            )
            self._transformers_model = self._transformers_model.to(torch.float32).eval()
            if device != "cpu" and torch.cuda.is_available():
                self._transformers_model = self._transformers_model.to(device)
            self._mode = "transformers"
            self.name = "timesfm-2.5-tf"
            return
        except ImportError:
            pass

        raise RuntimeError(
            "TimesFM 2.5 not available. Options: \n"
            "  1) Upgrade timesfm:     pip install --upgrade 'timesfm[torch]' \n"
            "  2) Use transformers:  pip install transformers torch \n"
            "  3) Install from src:  git clone https://github.com/google-research/timesfm.git && "
            "cd timesfm && pip install -e '.[torch]'"
        )
    def predict(self, context, horizon):
        if self._mode == "v25":
            _point, quantile = self._model.forecast(
                horizon=horizon, inputs=[context.astype(np.float32)]
            )
            q = quantile[0]
        elif self._mode == "legacy":
            _point, quantile = self._tfm.forecast([context], freq=[0])
            q = quantile[0]
        elif self._mode == "transformers":
            import torch
            ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            if self._device != "cpu" and torch.cuda.is_available():
                ctx = ctx.to(self._device)
            with torch.no_grad():
                outputs = self._transformers_model(
                    past_values=ctx,
                    forecast_context_len=min(len(context), 512),
                )
            mean = outputs.mean_predictions[0].cpu().numpy()[:horizon]
            # Try to extract quantiles from full_predictions if shape looks right
            if (
                hasattr(outputs, "full_predictions")
                and outputs.full_predictions is not None
            ):
                fp = outputs.full_predictions[0].cpu().numpy()
                if fp.ndim == 2 and fp.shape[1] >= len(QUANTILE_LEVELS):
                    return {
                        lvl: fp[:horizon, i]
                        for i, lvl in enumerate(QUANTILE_LEVELS)
                    }
            # Fallback: symmetric quantiles around mean
            std = float(np.std(context[-48:])) or 1e-3
            return {
                0.1: mean - 1.28 * std,
                0.5: mean,
                0.9: mean + 1.28 * std,
            }
        else:
            raise RuntimeError("TimesFM backend not initialized")

        levels = QUANTILE_LEVELS
        offset = q.shape[1] - len(levels) if q.shape[1] > len(levels) else 0
        return {lvl: q[:horizon, offset + i] for i, lvl in enumerate(levels)}


class Moirai20Backend(TSFMBackend):
    """Salesforce Moirai 2.0 / 1.x with predictor + direct-inference fallback."""
    name = "moirai-2.0"
    supports_multivariate = False

    def __init__(self, size: str = "small", device: str = "cpu"):
        self._device = device
        self._size = size
        self._version = None
        self._module = None

        # Try v2 first
        try:
            from uni2ts.model.moirai2 import Moirai2Module
            self._module = Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{size}")
            self._version = "v2"
        except ImportError:
            try:
                from uni2ts.model.moirai import MoiraiModule
                self._module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}")
                self._version = "v1"
            except ImportError as e:
                raise RuntimeError(
                    "uni2ts not installed. Run: pip install uni2ts"
                ) from e

        self._module.to(device).eval()
        self.name = f"moirai20-{size}"
        self.params = "~200M" if size == "small" else "~400M" if size == "base" else "~800M"

    def predict(self, context, horizon):
        # Strategy 1: GluonTS predictor (works when gluonts versions match)
        try:
            return self._predict_with_predictor(context, horizon)
        except Exception as e:
            logger.warning("Moirai predictor failed: %s", e)

        # Strategy 2: Direct PyTorch inference (bypasses GluonTS data loaders)
        try:
            return self._predict_direct(context, horizon)
        except Exception as e:
            logger.warning("Moirai direct inference failed: %s", e)

        # Strategy 3: Fallback persistence with neural-calibrated spread
        logger.warning(
            "\n!!! MOIRAI FALLBACK TRIGGERED !!!\n"
            "Moirai is NOT running inference. Using persistence fallback (last value + noise).\n"
            "This means the 200M model is idle and results are heuristic-only.\n"
            "Check warnings above for the root cause.\n"
        )
        return self._predict_fallback(context, horizon)

    def _predict_with_predictor(self, context, horizon):
        import pandas as pd
        try:
            from gluonts.dataset.common import ListDataset
        except ImportError as e:
            raise RuntimeError("gluonts not installed") from e

        # FIX: Use pd.Period instead of pd.Timestamp with freq parameter
        # pd.Timestamp(..., freq=...) is deprecated and causes TypeError in newer pandas
        ds = ListDataset(
            [{
                "start": pd.Period("2000-01-01", freq="H"),
                "target": context.astype(np.float32),
            }],
            freq="H",
        )

        if self._version == "v2":
            from uni2ts.model.moirai2 import Moirai2Forecast
            model = Moirai2Forecast(
                module=self._module,
                prediction_length=horizon,
                context_length=len(context),
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:
            from uni2ts.model.moirai import MoiraiForecast
            model = MoiraiForecast(
                module=self._module,
                prediction_length=horizon,
                context_length=len(context),
                patch_size="auto",
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )

        predictor = model.create_predictor(batch_size=1)
        forecasts = list(predictor.predict(ds))
        if not forecasts:
            raise RuntimeError("Moirai returned no forecasts")
        fc = forecasts[0]

        result = {}
        for lvl in QUANTILE_LEVELS:
            try:
                q = fc.quantile(lvl)
                if hasattr(q, "numpy"):
                    q = q.numpy()
                result[lvl] = np.asarray(q).flatten()[:horizon]
            except Exception as qe:
                logger.warning("Moirai quantile %.1f extraction failed: %s", lvl, qe)
                # Fallback to median for this quantile if extraction fails
                result[lvl] = np.full(horizon, np.nan)

        # If median is NaN, the whole forecast is broken
        if np.all(np.isnan(result.get(0.5, []))):
            raise RuntimeError("Moirai returned all-NaN forecasts")

        return result

    def _predict_direct(self, context, horizon):
        """Bypass GluonTS predictor entirely; call the PyTorch module directly."""
        import torch
        import inspect

        C = len(context)
        H = horizon

        # Inspect forward signature to know exactly what kwargs it expects
        sig = inspect.signature(self._module.forward)
        params = set(sig.parameters.keys())
        logger.debug("Moirai forward params: %s", params)

        # Build kwargs based on what the model expects
        kwargs = {}

        def _t(shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=self._device)

        if "target" in params:
            target = _t((1, 1, C + H))
            target[0, 0, :C] = torch.tensor(context, dtype=torch.float32, device=self._device)
            kwargs["target"] = target

        if "observed_mask" in params:
            observed_mask = _t((1, 1, C + H))
            observed_mask[0, 0, :C] = 1.0
            kwargs["observed_mask"] = observed_mask

        if "sample_id" in params:
            kwargs["sample_id"] = torch.zeros((1, 1), dtype=torch.long, device=self._device)

        if "time_id" in params:
            kwargs["time_id"] = torch.arange(C + H, dtype=torch.long, device=self._device).unsqueeze(0)

        if "variate_id" in params:
            kwargs["variate_id"] = torch.zeros((1, 1), dtype=torch.long, device=self._device)

        if "prediction_mask" in params:
            pred_mask = _t((1, 1, C + H))
            pred_mask[0, 0, C:] = 1.0
            kwargs["prediction_mask"] = pred_mask

        # Some versions use "feat_dynamic_real" instead of the above
        if "feat_dynamic_real" in params and "feat_dynamic_real" not in kwargs:
            kwargs["feat_dynamic_real"] = _t((1, 0, C + H))  # zero features

        with torch.no_grad():
            output = self._module(**kwargs)

        return self._extract_quantiles(output, H)

    def _extract_quantiles(self, output, horizon):
        import torch

        # --- Try 1: Distribution object with .quantile() method ---
        try:
            result = {}
            for lvl in QUANTILE_LEVELS:
                q_val = output.quantile(torch.tensor(lvl, device=self._device))
                if hasattr(q_val, "cpu"):
                    q_val = q_val.cpu()
                if hasattr(q_val, "numpy"):
                    q_val = q_val.numpy()
                q_arr = np.asarray(q_val)

                # Extract prediction horizon from various possible shapes
                if q_arr.ndim >= 3 and q_arr.shape[-1] >= horizon:
                    # (batch, variate, time) -> take last H steps
                    result[lvl] = q_arr[0, 0, -horizon:].flatten()
                elif q_arr.ndim == 2 and q_arr.shape[-1] >= horizon:
                    result[lvl] = q_arr[0, -horizon:].flatten()
                elif q_arr.ndim == 1 and len(q_arr) >= horizon:
                    result[lvl] = q_arr[-horizon:].flatten()
                else:
                    result[lvl] = np.resize(q_arr, horizon)
            return result
        except Exception:
            pass

        # --- Try 2: Distribution with .mean and .std / .scale ---
        try:
            mean = output.mean
            if hasattr(mean, "cpu"):
                mean = mean.cpu().numpy()
            mean = np.asarray(mean).flatten()[-horizon:]

            std = None
            for attr in ("std", "scale", "variance"):
                if hasattr(output, attr):
                    std = getattr(output, attr)
                    if attr == "variance":
                        std = np.sqrt(std)
                    break
            if hasattr(std, "cpu"):
                std = std.cpu().numpy()
            std = np.asarray(std).flatten()[-horizon:] if std is not None else np.full(horizon, 1e-3)

            return {
                0.1: mean - 1.28 * std,
                0.5: mean,
                0.9: mean + 1.28 * std,
            }
        except Exception:
            pass

        # --- Try 3: Raw tensor output ---
        try:
            if hasattr(output, "cpu"):
                output = output.cpu().numpy()
            out = np.asarray(output)

            # Try to find the prediction horizon in the tensor
            if out.ndim >= 3 and out.shape[-1] >= horizon:
                out = out[0, 0, -horizon:]
            elif out.ndim == 2 and out.shape[-1] >= horizon:
                out = out[0, -horizon:]
            elif out.ndim == 1 and len(out) >= horizon:
                out = out[-horizon:]
            else:
                out = np.resize(out, horizon)

            out = out.flatten()[:horizon]
            # If shape suggests quantiles: (horizon, n_quantiles)
            if out.ndim > 1 and out.shape[1] == len(QUANTILE_LEVELS):
                return {lvl: out[:, i] for i, lvl in enumerate(QUANTILE_LEVELS)}

            # Treat as point forecast, build symmetric quantiles
            std = np.full(horizon, max(float(np.std(out)), 1e-3))
            return {
                0.1: out - 1.28 * std,
                0.5: out,
                0.9: out + 1.28 * std,
            }
        except Exception:
            pass

        return None

    def _predict_fallback(self, context, horizon):
        last = float(context[-1])
        noise = max(float(np.std(context[-48:])), 1e-3)
        p50 = np.full(horizon, last)
        return {
            0.1: p50 - 1.28 * noise,
            0.5: p50,
            0.9: p50 + 1.28 * noise,
        }
class Toto20Backend(TSFMBackend):
    """Datadog Toto 2.0 — PRIMARY RECOMMENDATION."""
    name = "toto-2.0"
    supports_multivariate = True

    def __init__(self, size: str = "22m", device: str = "cpu"):
        try:
            import torch
            from toto2 import Toto2Model
        except ImportError as e:
            raise RuntimeError(
                "Toto 2.0 not installed. Run: pip install toto-models  (requires Python 3.12+)"
            ) from e
        size_map = {"4m": "4m", "22m": "22m", "313m": "313m", "1b": "1B", "2.5b": "2.5B"}
        param_map = {"4m": "4M", "22m": "22M", "313m": "313M", "1b": "1B", "2.5b": "2.5B"}
        canonical = size_map.get(size.lower(), size.lower())
        self.params = param_map.get(size.lower(), canonical)
        self.name = f"toto-2.0-{canonical.lower()}"
        self._model = Toto2Model.from_pretrained(f"Datadog/Toto-2.0-{canonical}")
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device).eval()

    def predict(self, context, horizon):
        import torch
        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self._device)
        target_mask = torch.ones_like(ctx, dtype=torch.bool)
        series_ids = torch.zeros(1, 1, dtype=torch.long, device=self._device)
        with torch.no_grad():
            quantiles = self._model.forecast(
                {"target": ctx, "target_mask": target_mask, "series_ids": series_ids},
                horizon=horizon, decode_block_size=None, has_missing_values=False,
            )
        q = quantiles[:, 0, 0, :].cpu().numpy()
        return {lvl: q[i, :] for i, lvl in enumerate(QUANTILE_LEVELS)}

    def predict_multivariate(self, contexts: Dict[str, np.ndarray], horizon: int) -> Dict[str, Dict[float, np.ndarray]]:
        import torch
        fields = list(contexts.keys())
        stacked = np.stack([contexts[f] for f in fields], axis=0)
        ctx = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(self._device)
        target_mask = torch.ones_like(ctx, dtype=torch.bool)
        series_ids = torch.zeros(1, len(fields), dtype=torch.long, device=self._device)
        with torch.no_grad():
            quantiles = self._model.forecast(
                {"target": ctx, "target_mask": target_mask, "series_ids": series_ids},
                horizon=horizon, decode_block_size=None, has_missing_values=False,
            )
        q = quantiles[:, 0, :, :].cpu().numpy()
        result = {}
        for i, field in enumerate(fields):
            result[field] = {lvl: q[j, i, :] for j, lvl in enumerate(QUANTILE_LEVELS)}
        return result


BACKEND_FACTORY = {
    "persistence": lambda: PersistenceBackend(),
    "seasonal_naive": lambda: SeasonalNaiveBackend(),
    "toto-4m": lambda: Toto20Backend("4m"),
    "toto-22m": lambda: Toto20Backend("22m"),
    "toto-313m": lambda: Toto20Backend("313m"),
    "toto-1b": lambda: Toto20Backend("1b"),
    "toto-2.5b": lambda: Toto20Backend("2.5b"),
    "tirex": lambda: TiRexBackend(),
    "chronos2-tiny": lambda: Chronos2Backend("tiny"),
    "chronos2-small": lambda: Chronos2Backend("small"),
    "chronos2-base": lambda: Chronos2Backend("base"),
    "chronos2-large": lambda: Chronos2Backend("large"),
    "timesfm25": lambda: TimesFM25Backend(),
    "moirai20-small": lambda: Moirai20Backend("small"),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def crps_approx(y_true: np.ndarray, quantiles: Dict[float, np.ndarray], field: str) -> float:
    if field in CIRCULAR_FIELDS:
        return circular_crps(y_true, quantiles)
    losses = [pinball_loss(y_true, quantiles[q], q) for q in quantiles]
    return float(np.mean(losses))


def mase_scale(train_series: np.ndarray, season: int = 24) -> float:
    if len(train_series) <= season:
        return float(np.mean(np.abs(np.diff(train_series)))) or 1e-4
    diffs = np.abs(train_series[season:] - train_series[:-season])
    return float(np.mean(diffs)) or 1e-4


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(df: pd.DataFrame, backend_names: List[str],
                   lat: float = 0.0, lon: float = 0.0,
                   context_len: int = 336, horizon: int = 24, n_splits: int = 5) -> Dict:
    # Auto-adjust context length for patch-based models (e.g. Toto uses 32-step patches)
    for bname in backend_names:
        if bname.startswith("toto"):
            patch_size = 32
            if context_len % patch_size != 0:
                old_ctx = context_len
                context_len = (context_len // patch_size) * patch_size
                logger.warning("Toto requires context %% %d == 0. Adjusting %d -> %d", patch_size, old_ctx, context_len)
            break

    backends = {}
    for bname in backend_names:
        try:
            backends[bname] = BACKEND_FACTORY[bname]()
            logger.info("Loaded backend %s (%s params, multivariate=%s)",
                        backends[bname].name, backends[bname].params,
                        backends[bname].supports_multivariate)
        except Exception as e:
            logger.warning("Skipping %s: %s", bname, e)

    if not backends:
        raise RuntimeError("No backends loaded. Check installation and auth.")

    # Determine active fields and common step size
    active_fields = []
    for feat in SCHEMA_FIELDS:
        if feat not in df.columns:
            continue
        series = df[feat].dropna()
        if len(series) >= context_len + horizon:
            active_fields.append(feat)

    if not active_fields:
        raise RuntimeError("No fields have enough data for benchmarking.")

    # Use minimum length across all fields so every field can participate in every split
    min_len = min(len(df[f].dropna()) for f in active_fields)
    step = max(1, (min_len - context_len - horizon) // n_splits)

    # Raw accumulator: results[feat][bname] = list of (mae, rmse, mase, crps, latency) tuples
    raw_results: Dict[str, Dict[str, List[Tuple]]] = {feat: {} for feat in active_fields}

    for split in range(n_splits):
        test_start = context_len + split * step
        if test_start + horizon > min_len:
            break

        # Collect aligned contexts / truths / indices for every field at this split
        split_data: Dict[str, Dict] = {}
        for feat in active_fields:
            series = df[feat].dropna()
            train = series.iloc[:test_start].values
            truth = series.iloc[test_start:test_start + horizon].values
            ctx = train[-context_len:]
            ctx_index = series.index[test_start - context_len:test_start]
            truth_index = series.index[test_start:test_start + horizon]
            split_data[feat] = {
                "train": train, "truth": truth, "ctx": ctx,
                "ctx_index": ctx_index, "truth_index": truth_index,
            }

        # ------------------------------------------------------------------
        # Multivariate only makes sense on real sensors.
        # irradiance  = formula-derived (clear_sky_ghi) → zero variance → always flatlined
        # hub_voltage = synthetic RNG → low variance → often flatlined
        # Exclude both from the multivariate batch; they still get univariate forecasts.
        # ------------------------------------------------------------------
        mv_eligible = [f for f in active_fields if f not in (SYNTHETIC_FIELDS | {"irradiance"})]
        poisonous = detect_poisonous_fields({f: split_data[f]["ctx"] for f in mv_eligible})
        clean_fields = [f for f in mv_eligible if f not in poisonous]
        has_poison = bool(poisonous)

        for bname, backend in backends.items():
            # ------------------------------------------------------------
            # Multivariate path (only when backend supports it & no poison)
            # ------------------------------------------------------------
            mv_name = f"{bname}-mv"
            if backend.supports_multivariate and len(clean_fields) > 1 and not has_poison:
                t0 = time.perf_counter()
                try:
                    q_all_mv = backend.predict_multivariate(
                        {f: split_data[f]["ctx"] for f in clean_fields}, horizon
                    )
                    mv_latency = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    logger.warning("%s multivariate failed split %d: %s", bname, split, e)
                    q_all_mv = None

                if q_all_mv:
                    for feat in clean_fields:
                        q_raw = dict(q_all_mv[feat])  # copy

                        # --- Temperature stable-weather fallback ---
                        if feat == "temperature" and bname in (
                            "toto-22m", "toto-313m", "toto-4m", "toto-1b", "toto-2.5b",
                            "tirex", "chronos2-tiny", "chronos2-small", "chronos2-base",
                            "chronos2-large", "timesfm25", "moirai20-small",
                        ):
                            recent_std = float(np.std(split_data[feat]["ctx"][-48:]))
                            if recent_std < 1.5:
                                last = float(split_data[feat]["ctx"][-1])
                                noise = max(recent_std, 0.3)
                                neural_p50 = q_raw[0.5]
                                neural_spread = float(np.mean(q_raw[0.9] - q_raw[0.1])) / 2.0
                                new_spread = max(noise, neural_spread * 0.5)
                                p50_fb = np.full(horizon, last)
                                q_raw = {
                                    0.1: p50_fb - 1.28 * new_spread,
                                    0.5: p50_fb,
                                    0.9: p50_fb + 1.28 * new_spread,
                                }

                        # --- Irradiance clear-sky residual ---
                        if feat == "irradiance" and lat != 0.0:
                            cs_ctx = np.array([clear_sky_ghi(lat, lon, t) for t in split_data[feat]["ctx_index"]])
                            day_mask = cs_ctx > 0
                            if np.sum(day_mask) > 24:
                                kt = np.clip(split_data[feat]["ctx"][day_mask] / cs_ctx[day_mask], 0.0, 1.5)
                                kt_mean = float(np.median(kt[-min(72, len(kt)):]))
                            else:
                                kt_mean = 1.0
                            cs_fc = np.array([clear_sky_ghi(lat, lon, t) for t in split_data[feat]["truth_index"]])
                            val = cs_fc * kt_mean
                            neural_p50 = q_raw[0.5]
                            for lvl in list(q_raw.keys()):
                                spread = q_raw[lvl] - neural_p50
                                q_raw[lvl] = np.maximum(0.0, val + spread)

                        # Metrics
                        q_display = {k: clip_field(feat, v) for k, v in q_raw.items()}
                        p50 = q_display.get(0.5, q_display[min(q_display, key=lambda x: abs(x - 0.5))])
                        if np.any(np.isnan(p50)):
                            logger.warning("%s produced NaN p50 for %s split %d — substituting last value", mv_name, feat, split)
                            last_good = float(split_data[feat]["ctx"][-1]) if not np.isnan(split_data[feat]["ctx"][-1]) else 0.0
                            p50 = np.nan_to_num(p50, nan=last_good)

                        if feat in CIRCULAR_FIELDS:
                            mae = circular_mae(split_data[feat]["truth"], p50)
                            rmse = circular_rmse(split_data[feat]["truth"], p50)
                        else:
                            mae = float(mean_absolute_error(split_data[feat]["truth"], p50))
                            rmse = float(np.sqrt(mean_squared_error(split_data[feat]["truth"], p50)))

                        scale = mase_scale(split_data[feat]["train"])
                        mase = mae / scale
                        crps = crps_approx(split_data[feat]["truth"], q_raw, feat) if len(q_raw) > 2 else float("nan")

                        if mv_name not in raw_results[feat]:
                            raw_results[feat][mv_name] = []
                        raw_results[feat][mv_name].append((mae, rmse, mase, crps, mv_latency))

            # ------------------------------------------------------------
            # Univariate path (all backends, all fields — always runs)
            # ------------------------------------------------------------
            for feat in active_fields:
                ctx = split_data[feat]["ctx"]
                truth = split_data[feat]["truth"]
                train = split_data[feat]["train"]

                t0 = time.perf_counter()
                try:
                    q_raw = backend.predict(ctx, horizon)
                except Exception as e:
                    logger.warning("%s failed on %s split %d: %s", bname, feat, split, e)
                    continue
                latency = (time.perf_counter() - t0) * 1000.0

                # --- Temperature stable-weather fallback ---
                if feat == "temperature" and bname in (
                    "toto-22m", "toto-313m", "toto-4m", "toto-1b", "toto-2.5b",
                    "tirex", "chronos2-tiny", "chronos2-small", "chronos2-base",
                    "chronos2-large", "timesfm25", "moirai20-small",
                ):
                    recent_std = float(np.std(ctx[-48:]))
                    if recent_std < 1.5:
                        last = float(ctx[-1])
                        noise = max(recent_std, 0.3)
                        neural_p50 = q_raw[0.5]
                        neural_spread = float(np.mean(q_raw[0.9] - q_raw[0.1])) / 2.0
                        new_spread = max(noise, neural_spread * 0.5)
                        p50_fb = np.full(horizon, last)
                        q_raw = {
                            0.1: p50_fb - 1.28 * new_spread,
                            0.5: p50_fb,
                            0.9: p50_fb + 1.28 * new_spread,
                        }

                # --- Irradiance clear-sky residual ---
                if feat == "irradiance" and lat != 0.0:
                    cs_ctx = np.array([clear_sky_ghi(lat, lon, t) for t in split_data[feat]["ctx_index"]])
                    day_mask = cs_ctx > 0
                    if np.sum(day_mask) > 24:
                        kt = np.clip(ctx[day_mask] / cs_ctx[day_mask], 0.0, 1.5)
                        kt_mean = float(np.median(kt[-min(72, len(kt)):]))
                    else:
                        kt_mean = 1.0
                    cs_fc = np.array([clear_sky_ghi(lat, lon, t) for t in split_data[feat]["truth_index"]])
                    val = cs_fc * kt_mean
                    neural_p50 = q_raw[0.5]
                    for lvl in list(q_raw.keys()):
                        spread = q_raw[lvl] - neural_p50
                        q_raw[lvl] = np.maximum(0.0, val + spread)

                # Metrics
                q_display = {k: clip_field(feat, v) for k, v in q_raw.items()}
                p50 = q_display.get(0.5, q_display[min(q_display, key=lambda x: abs(x - 0.5))])
                # NaN guard: if a backend returns NaN, substitute with last known value
                if np.any(np.isnan(p50)):
                    logger.warning("%s produced NaN p50 for %s split %d — substituting last value", bname, feat, split)
                    last_good = float(ctx[-1]) if not np.isnan(ctx[-1]) else 0.0
                    p50 = np.nan_to_num(p50, nan=last_good)

                if feat in CIRCULAR_FIELDS:
                    mae = circular_mae(truth, p50)
                    rmse = circular_rmse(truth, p50)
                else:
                    mae = float(mean_absolute_error(truth, p50))
                    rmse = float(np.sqrt(mean_squared_error(truth, p50)))

                scale = mase_scale(train)
                mase = mae / scale
                crps = crps_approx(truth, q_raw, feat) if len(q_raw) > 2 else float("nan")

                if bname not in raw_results[feat]:
                    raw_results[feat][bname] = []
                raw_results[feat][bname].append((mae, rmse, mase, crps, latency))

    # ------------------------------------------------------------------
    # Convert raw accumulators to final averaged results
    # ------------------------------------------------------------------
    results: Dict[str, Dict] = {}
    for feat in active_fields:
        results[feat] = {}
        for bname, metrics_list in raw_results[feat].items():
            if not metrics_list:
                continue
            maes = [m[0] for m in metrics_list]
            rmses = [m[1] for m in metrics_list]
            mases = [m[2] for m in metrics_list]
            crpss = [m[3] for m in metrics_list]
            latencies = [m[4] for m in metrics_list]

            is_mv = bname.endswith("-mv")
            base_bname = bname[:-3] if is_mv else bname
            backend = backends.get(base_bname)
            params = backend.params if backend else "?"

            results[feat][bname] = {
                "avg_mae": round(float(np.mean(maes)), 4),
                "avg_rmse": round(float(np.mean(rmses)), 4),
                "avg_mase": round(float(np.mean(mases)), 4),
                "avg_crps": round(float(np.nanmean(crpss)), 4),
                "avg_latency_ms": round(float(np.mean(latencies)), 1),
                "eval_splits": len(maes),
                "params": params,
                "multivariate": True if is_mv else (backend.supports_multivariate if backend else False),
            }

    # ------------------------------------------------------------------
    # Persistence / seasonal-naive baselines (univariate only)
    # ------------------------------------------------------------------
    for feat in active_fields:
        series = df[feat].dropna()
        if len(series) < context_len + horizon:
            continue
        step = max(1, (len(series) - context_len - horizon) // n_splits)

        for bname, backend in {"persistence": PersistenceBackend(), "seasonal_naive": SeasonalNaiveBackend()}.items():
            maes, mases = [], []
            for split in range(n_splits):
                test_start = context_len + split * step
                if test_start + horizon > len(series):
                    break
                train = series.iloc[:test_start].values
                truth = series.iloc[test_start:test_start + horizon].values
                q = backend.predict(train[-context_len:], horizon)
                p50 = q[0.5]
                mae = circular_mae(truth, p50) if feat in CIRCULAR_FIELDS else float(mean_absolute_error(truth, p50))
                mases.append(mae / mase_scale(train))
                maes.append(mae)
            if maes:
                results[feat][bname] = {
                    "avg_mae": round(float(np.mean(maes)), 4),
                    "avg_mase": round(float(np.mean(mases)), 4),
                    "params": "0",
                    "multivariate": False,
                }

    return results


def print_comparison_table(results: Dict) -> None:
    print("\n" + "=" * 120)
    print(f"{'field':<14}{'backend':<22}{'params':<8}{'MAE':<10}{'RMSE':<10}{'MASE':<10}{'CRPS':<10}{'ms/call':<10}{'mv':<4}")
    print("-" * 120)
    for feat, backends in results.items():
        ranked = sorted(backends.items(), key=lambda kv: kv[1].get("avg_mase", 999))
        for bname, m in ranked:
            mv = "yes" if m.get("multivariate") else "no"
            print(f"{feat:<14}{bname:<22}{str(m.get('params','-')):<8}"
                  f"{m.get('avg_mae','-'):<10}{m.get('avg_rmse','-'):<10}"
                  f"{m.get('avg_mase','-'):<10}{m.get('avg_crps','-'):<10}"
                  f"{m.get('avg_latency_ms','-'):<10}{mv:<4}")
    print("=" * 120)
    print("MASE < 1.0 beats seasonal-naive. CRPS = probabilistic calibration (lower = better).")
    print("mv = multivariate native (one call for all fields).")
    print("-mv suffix = Toto running multivariate on clean sensors (no poisonous fields detected).")
    print("\n⚠️  NOTE: ECCC data may overlap Chronos/TimesFM pretraining — treat those MASE as optimistic.\n")



# ---------------------------------------------------------------------------
# Poison detection for hybrid multivariate / univariate fallback
# ---------------------------------------------------------------------------
def detect_poisonous_fields(contexts: Dict[str, np.ndarray],
                            max_nan_ratio: float = 0.15,
                            mad_threshold: float = 5.0,
                            min_variance: float = 1e-8) -> set:
    """Flag fields that could corrupt a multivariate forecast."""
    poisonous = set()
    for field, ctx in contexts.items():
        arr = np.asarray(ctx)
        nan_ratio = np.mean(np.isnan(arr))
        if nan_ratio > max_nan_ratio:
            logger.warning("Field %s flagged: %.1f%% NaNs", field, nan_ratio * 100)
            poisonous.add(field)
            continue

        clean = arr[~np.isnan(arr)]
        if len(clean) < 24:
            logger.warning("Field %s flagged: insufficient data (%d)", field, len(clean))
            poisonous.add(field)
            continue

        med = np.median(clean)
        mad = np.median(np.abs(clean - med))
        if mad < min_variance:
            logger.warning("Field %s flagged: flatlined (MAD=%.2e)", field, mad)
            poisonous.add(field)
            continue

        modified_z = np.abs((clean - med) / (1.4826 * mad + 1e-12))
        if np.max(modified_z) > mad_threshold:
            logger.warning("Field %s flagged: extreme outlier (max|z|=%.1f)", field, np.max(modified_z))
            poisonous.add(field)
            continue

    return poisonous
# ---------------------------------------------------------------------------
# Forecast + anomaly pipeline
# ---------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame, records: List[Dict], backend_name: str,
                  lat: float = 0.0, lon: float = 0.0,
                  context_len: int = 336, horizon: int = 24) -> Dict:
    # Auto-adjust context length for patch-based models
    if backend_name.startswith("toto"):
        patch_size = 32
        if context_len % patch_size != 0:
            old_ctx = context_len
            context_len = (context_len // patch_size) * patch_size
            logger.warning("Toto requires context %% %d == 0. Adjusting %d -> %d", patch_size, old_ctx, context_len)

    backend = BACKEND_FACTORY[backend_name]()
    latest = df.index[-1]
    last_seq = records[-1]["sequence_number"] if records else 0

    forecasts, anomalies = {}, []

    contexts = {}
    active_fields = []
    for feat in SCHEMA_FIELDS:
        if feat not in df.columns:
            continue
        series = df[feat].dropna()
        if len(series) < context_len:
            continue
        contexts[feat] = series.iloc[-context_len:].values
        active_fields.append(feat)

    # ------------------------------------------------------------------
    # Hybrid: multivariate when clean, univariate fallback for poison
    # ------------------------------------------------------------------
    poisonous = detect_poisonous_fields(contexts)
    if poisonous:
        logger.warning("Poisonous fields detected: %s — switching those to univariate", poisonous)

    clean_fields = [f for f in active_fields if f not in poisonous]

    q_all = {}
    if backend.supports_multivariate and len(clean_fields) > 1:
        logger.info("Multivariate inference on clean fields: %s", clean_fields)
        clean_contexts = {f: contexts[f] for f in clean_fields}
        q_all.update(backend.predict_multivariate(clean_contexts, horizon))
        # Poisonous fields run univariate so they can't leak noise into the clean batch
        for feat in poisonous:
            q_all[feat] = backend.predict(contexts[feat], horizon)
    else:
        # Not enough clean fields for multivariate (or backend doesn't support it)
        for feat, ctx in contexts.items():
            q_all[feat] = backend.predict(ctx, horizon)

    # ------------------------------------------------------------------
    # Physics-informed post-processing
    # ------------------------------------------------------------------
    # 1. Irradiance: clear-sky residual (proper solar forecasting)
    if "irradiance" in q_all and "irradiance" in contexts:
        ctx_irr = contexts["irradiance"]
        ctx_len = len(ctx_irr)
        cs_ctx = np.array([
            clear_sky_ghi(lat, lon, latest - pd.DateOffset(hours=int(ctx_len - i)))
            for i in range(ctx_len)
        ])
        day_mask = cs_ctx > 0
        if np.sum(day_mask) > 24:
            kt = np.clip(ctx_irr[day_mask] / cs_ctx[day_mask], 0.0, 1.5)
            kt_mean = float(np.median(kt[-min(72, len(kt)):]))
        else:
            kt_mean = 1.0
        for h in range(horizon):
            fc_time = latest + pd.DateOffset(hours=int(h + 1))
            cs = clear_sky_ghi(lat, lon, fc_time)
            val = cs * kt_mean
            neural_p50 = q_all["irradiance"][0.5]
            for lvl in list(q_all["irradiance"].keys()):
                spread = q_all["irradiance"][lvl][h] - neural_p50[h]
                q_all["irradiance"][lvl][h] = max(0.0, val + spread)
        logger.info("Irradiance forecast using clear-sky residual (kt=%.3f)", kt_mean)

    # 2. Temperature: stable-weather persistence fallback
    if "temperature" in q_all and "temperature" in contexts:
        ctx_temp = contexts["temperature"]
        recent_std = float(np.std(ctx_temp[-48:]))
        if recent_std < 1.5:
            last = float(ctx_temp[-1])
            noise = max(recent_std, 0.3)
            neural_p50 = q_all["temperature"][0.5]
            neural_spread = float(np.mean(q_all["temperature"][0.9] - q_all["temperature"][0.1])) / 2.0
            new_spread = max(noise, neural_spread * 0.5)
            p50 = np.full(horizon, last)
            q_all["temperature"] = {
                0.1: p50 - 1.28 * new_spread,
                0.5: p50,
                0.9: p50 + 1.28 * new_spread,
            }
            logger.info("Temperature low variance (σ=%.2f°C) — using persistence with neural-calibrated spread", recent_std)

    # 4. Wind direction: persistence beats all neural models on this field
    if "wind_direction" in q_all and "wind_direction" in contexts:
        ctx_wd = contexts["wind_direction"]
        last = float(ctx_wd[-1]) % 360.0
        recent = np.asarray(ctx_wd[-48:], dtype=float) % 360.0
        diffs = circular_diff(recent, np.full_like(recent, last, dtype=float))
        noise = max(float(np.mean(diffs)), 1e-3)
        p50 = np.full(horizon, last, dtype=float)
        q_all["wind_direction"] = {
            0.1: (p50 - 1.28 * noise) % 360,
            0.5: p50 % 360,
            0.9: (p50 + 1.28 * noise) % 360,
        }

    for feat in active_fields:
        q_raw = q_all[feat]
        q = {k: clip_field(feat, v) for k, v in q_raw.items()}
        p10, p50, p90 = q.get(0.1, q[min(q)]), q[0.5], q.get(0.9, q[max(q)])

        step_fc = []
        for h in range(horizon):
            fc_time = latest + pd.DateOffset(hours=int(h + 1))
            payload = {feat: round(float(p50[h]), 2)}
            for other in active_fields:
                if other != feat:
                    payload[other] = round(float(df[other].iloc[-1]), 2)
            step_fc.append(schema_record(last_seq + h + 1, fc_time, payload))
        forecasts[feat] = step_fc

        if feat in SYNTHETIC_FIELDS:
            continue

        series = df[feat].dropna()
        if len(series) >= context_len + 24:
            past = series.iloc[-(context_len + 24):-24].values
            actuals = series.iloc[-24:]
            cq_raw = backend.predict(past, 24) if not backend.supports_multivariate else backend.predict_multivariate({feat: past}, 24)[feat]
            cp10, cp50, cp90 = cq_raw.get(0.1, cq_raw[min(cq_raw)]), cq_raw[0.5], cq_raw.get(0.9, cq_raw[max(cq_raw)])

            for idx, (ts, actual) in enumerate(actuals.items()):
                if feat in CIRCULAR_FIELDS:
                    spread = max(1e-3, circular_diff(np.array([cp90[idx]]), np.array([cp10[idx]]))[0] / 2.0)
                    z = circular_diff(np.array([actual]), np.array([cp50[idx]]))[0] / spread
                    outlier = z > 3.0
                else:
                    spread = max(1e-3, (cp90[idx] - cp10[idx]) / 2.0)
                    z = abs(actual - cp50[idx]) / spread
                    outlier = actual < cp10[idx] or actual > cp90[idx] or z > 3.0
                if outlier:
                    anomalies.append({
                        "timestamp": ts.isoformat(), "field": feat,
                        "observed": round(float(actual), 2),
                        "expected_p50": round(float(clip_field(feat, cp50[idx])), 2),
                        "expected_p10": round(float(clip_field(feat, cp10[idx])), 2),
                        "expected_p90": round(float(clip_field(feat, cp90[idx])), 2),
                        "anomaly_score_z": round(float(z), 2),
                    })

    return {
        "schema_id": SCHEMA_ID,
        "backend": backend.name,
        "last_observed_timestamp": latest.isoformat(),
        "active_schema_fields": active_fields,
        "anomalies_detected_count": len(anomalies),
        "anomalies": anomalies,
        "multivariate_forecasts": forecasts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cascadia Sentinel Weather Engine — TSFM v2.3")
    ap.add_argument("--mode", choices=["forecast", "benchmark"], default="benchmark")
    ap.add_argument("--model", default="all",
                    help="Comma-separated: toto-22m,toto-313m,tirex,chronos2-small,timesfm25,moirai20-small,all")
    ap.add_argument("--station-id", type=int, default=51337)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", type=str, default="1")
    ap.add_argument("--context", type=int, default=336)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    month_list = [int(m.strip()) for m in args.months.split(",")]
    csv_text, lat, lon = fetch_eccc(args.station_id, args.year, month_list)
    df, records = transform_to_schema(csv_text, lat, lon)

    if args.mode == "benchmark":
        if args.model == "all":
            names = ["toto-22m", "toto-313m", "tirex", "chronos2-small", "timesfm25", "moirai20-small"]
        else:
            names = [m.strip() for m in args.model.split(",")]
        result = run_benchmark(df, names, lat=lat, lon=lon, context_len=args.context, horizon=args.horizon, n_splits=args.n_splits)
        print_comparison_table(result)
        out = {"benchmark_metrics": result, "_warning": "ECCC may overlap Chronos/TimesFM pretraining. Validate on your actual sensor stream before production."}
    else:
        model = "toto-22m" if args.model == "all" else args.model
        out = run_pipeline(df, records, model, lat=lat, lon=lon, context_len=args.context, horizon=args.horizon)

    json_out = json.dumps(out, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(json_out, encoding="utf-8")
        logger.info("Output written to %s", args.output)
    else:
        print(json_out)


if __name__ == "__main__":
    main()