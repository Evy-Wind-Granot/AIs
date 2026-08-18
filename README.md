# Cascadia Sentinel — Model Demos

Three standalone, runnable demos, one per instrument. Each fetches real public data, runs it through a model, and produces schema-compliant output. All support `--self-test` (synthetic, no network) for quick verification.

| Instrument   | Demo file                          | Model / Pipeline                          | Real Data Source |
|--------------|------------------------------------|------------------------------------------|------------------|
| Seismometer  | `seismic_demo.py`                  | PhaseNet + EQTransformer (SeisBench)     | EarthScope FDSN  |
| Weather      | `weather_tsfm_engine_v2_production_hybrid_fixed.py` | Multi-backend TSFM benchmark + forecast | ECCC hourly CSV  |
| Magnetometer | `magnetometer/demos/magnetometer_demo.py` | 5-band FIR QDC baseline + activity flags | INTERMAGNET GIN  |

---

## Quick Start — One Command for All Three

A convenience orchestrator is included:

```bash
# 1. Make the script executable and run setup + self-test
chmod +x run_all_demos.sh
./run_all_demos.sh --self-test

# 2. Full real-data run (pulls live streams from all three archives)
./run_all_demos.sh --real-data

# 3. If you get CUDA/NCCL or dependency conflicts, wipe and rebuild:
./run_all_demos.sh --clean-install --self-test

# 4. View example logs from previous real runs without downloading anything
python run_all_demos.py --show-examples
```

The orchestrator runs magnetometer → weather → seismic in sequence, prints a summary table, and saves `run_all_summary_*.json`.

---

## Setup (Manual)

If you prefer not to use `run_all_demos.sh`, create the environment manually:

```bash
# 1. Create virtual environment (Python 3.10+ required)
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate.bat   # Windows

# 2. Core dependencies (all demos)
pip install numpy pandas scipy requests scikit-learn

# 3. Seismic demo dependencies
pip install obspy seisbench

# 4. Weather TSFM engine dependencies
#    The engine auto-detects which backends are installed and skips missing ones gracefully.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chronos-forecasting   # Amazon Chronos-2
pip install timesfm               # Google TimesFM-2.5 (install from source, see below)
pip install uni2ts                # Salesforce Moirai 2.0
# pip install toto-models           # Datadog Toto 2.0 (requires Python 3.12+)
# pip install tirex-ts              # NX-AI TiRex (optional)

# 5. Verify installation
python seismic_demo.py --self-test
python -m magnetometer.demos.magnetometer_demo --self-test
python weather_tsfm_engine_v2_production_hybrid_fixed.py --mode benchmark --model persistence --station-id 51337 --year 2024 --months 1 --horizon 24
```

> **Note:** First runs of Chronos, PhaseNet, and EQTransformer download model weights automatically (~50 MB for Chronos, ~1 MB each for SeisBench models). Allow `huggingface.co` and `hifis-storage.desy.de` through your firewall, or pre-download on another machine.

---

## Dependency Reference

| Package | Required By | Install Command | Notes |
|---------|------------|-----------------|-------|
| `numpy` | All | `pip install numpy` | Core arrays |
| `pandas` | All | `pip install pandas` | Time-series handling |
| `scipy` | Magnetometer | `pip install scipy` | FIR filters |
| `requests` | Weather, Magnetometer | `pip install requests` | HTTP fetching |
| `scikit-learn` | Weather | `pip install scikit-learn` | Metrics (MAE, RMSE) |
| `obspy` | Seismic | `pip install obspy` | FDSN client, Stream/Trace |
| `seisbench` | Seismic | `pip install seisbench` | PhaseNet + EQTransformer |
| `torch` | Weather | `pip install torch` | Neural backend framework |
| `chronos-forecasting` | Weather | `pip install chronos-forecasting` | Amazon Chronos-2 |
| `timesfm` | Weather | Install from source | Google TimesFM-2.5 |
| `uni2ts` | Weather | `pip install uni2ts` | Salesforce Moirai 2.0 |
| `toto-models` | Weather | `pip install toto-models` | Datadog Toto 2.0 (Python 3.12+) |
| `tirex-ts` | Weather | `pip install tirex-ts` | NX-AI TiRex (optional) |

### Installing TimesFM from source

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e ".[torch]"
```

---

## 1. Seismometer — PhaseNet + EQTransformer P/S Picking

### Self-test (synthetic, no network)

```bash
python seismic_demo.py --self-test
```

Expected output:
```
Loaded self-test stream: 3 Trace(s) in Stream:
BW.RJOB..EHZ | 2009-08-24T00:20:03.000000Z - 2009-08-24T00:20:32.990000Z | 100.0 Hz, 3000 samples
...
PhaseNet total unique picks: 2
  PhaseNet P=1, S=1
     P  2009-08-24T00:20:07.700000Z  prob=0.992  (PhaseNet)
     S  2009-08-24T00:20:08.660000Z  prob=0.989  (PhaseNet)
EQTransformer total unique picks: 0
Model agreement (same phase + time within 1s): 0 picks
```

> **Why 0 EQTransformer picks on self-test?** The STEAD-trained EQTransformer model expects **60-second input windows** by default. The self-test stream is only 30 s. This is expected behavior — use `--window-s 60` for dual-model operation.

### Real data — single station, short window

```bash
python seismic_demo.py --fetch-real-data \
    --network CN --station VGZ --channel HH? \
    --start 2024-09-26T11:05:00 --end 2024-09-26T11:15:00 \
    --window-s 60 --step-s 10
```

### Real data — dense aftershock sequence (many P + S picks)

The **2024 Noto Peninsula M7.6** earthquake (Jan 1, 2024) produced a massive aftershock sequence. Station `IU.MAJO` (Matsushiro, Japan) recorded it with strong signal:

```bash
python seismic_demo.py --fetch-real-data \
    --network IU --station MAJO --channel BH? \
    --start 2024-01-01T07:00:00 --end 2024-01-01T11:00:00 \
    --window-s 60 --step-s 30 \
    --prob-threshold 0.15
```

**Example output (real run):**
```
============================================================
FINAL RESULTS (deduplicated)
============================================================

PhaseNet total unique picks: 73
  PhaseNet P=19, S=54
     S  2024-01-01T07:09:05.279538Z  prob=0.640  (PhaseNet)
     P  2024-01-01T07:10:32.909538Z  prob=0.742  (PhaseNet)
     S  2024-01-01T07:40:26.679538Z  prob=0.762  (PhaseNet)
     P  2024-01-01T07:43:05.979537Z  prob=0.796  (PhaseNet)
     S  2024-01-01T07:43:21.939538Z  prob=0.644  (PhaseNet)
     ... (63 more)

EQTransformer total unique picks: 11
  EQTransformer P=6, S=5
     P  2024-01-01T09:04:11.659538Z  prob=0.719  (EQTransformer)
     P  2024-01-01T09:08:40.179537Z  prob=0.742  (EQTransformer)
     P  2024-01-01T09:14:38.089538Z  prob=0.765  (EQTransformer)
     S  2024-01-01T09:26:06.319538Z  prob=0.752  (EQTransformer)
     S  2024-01-01T09:28:43.689537Z  prob=0.719  (EQTransformer)
     ... (1 more)

Combined total unique picks: 75
  Combined P=21, S=54

Model agreement (same phase + time within 1s): 9 picks
Schema messages saved to: IU_MAJO_seismic_schemas.jsonl
```

### Model behavior notes

| Aspect | PhaseNet | EQTransformer |
|--------|----------|---------------|
| **Sensitivity** | High — catches many aftershocks, some noise | Low — only high-confidence arrivals |
| **Window size** | Works at 30 s and 60 s | Needs 60 s (STEAD training) |
| **Threshold 0.10** | ~136 picks | ~11 picks |
| **Threshold 0.15** | ~73 picks | ~11 picks (unchanged) |
| **Threshold 0.30** | ~25 picks | ~10 picks |
| **Best use** | Completeness / swarm detection | Quality gate / ground-truth anchor |

**Tuning guidance:**
- Use `--window-s 60 --step-s 30` for dual-model operation
- Use `--prob-threshold 0.15` for balanced sensitivity
- Use `--prob-threshold 0.05` if you want more EQTransformer picks
- Use `--prob-threshold 0.30` for conservative, high-confidence-only mode

### Schema I/O

```bash
# Dry-run: print schema JSON lines without inferencing
python seismic_demo.py --fetch-real-data --network CN --station VGZ \
    --start 2024-01-01T00:00:00 --end 2024-01-01T00:05:00 --emit-schemas > seismic_schemas.jsonl

# Run inference from saved schemas
python seismic_demo.py --json-input CN_VGZ_seismic_schemas.jsonl
```

**Schema format (`seismometer.v1.json`):**
```json
{
  "sequence_number": 12345,
  "timestamp": [{"seconds": 1783123456, "nanoseconds": 789012345, "source": "NTP-INTERNET"}],
  "payload": {
    "channel_id": "AM.R57DB.00.EHE",
    "sample_rate": 100.0,
    "sample_count": 3,
    "samples": [-12, 3, 3400]
  }
}
```

---

## 2. Weather — TSFM Engine v2.3 (Multi-Backend Benchmark + Forecast)

### Self-test / Benchmark (no external data)

The weather engine always fetches ECCC data for benchmarks. For a minimal test with only the persistence baseline (no heavy TSFM downloads):

```bash
python weather_tsfm_engine_v2_production_hybrid_fixed.py \
    --mode benchmark --model persistence \
    --station-id 51337 --year 2024 --months 1 --horizon 24
```

### Real data — ECCC hourly observations, single backend

```bash
python weather_tsfm_engine_v2_production_hybrid_fixed.py \
    --mode forecast --model toto-22m \
    --station-id 51337 --year 2024 --months 1 --horizon 24
```

### Real data — benchmark multiple backends

```bash
python weather_tsfm_engine_v2_production_hybrid_fixed.py \
    --mode benchmark \
    --model toto-22m,chronos2-small,timesfm25,moirai20-small \
    --station-id 51337 --year 2024 --months 1 --horizon 24 --n-splits 5
```

**Example output (real run):**
```
field        backend              params  MAE       RMSE      MASE      CRPS      ms/call   mv
--------------------------------------------------------------------------------------------------------------
temperature  persistence          0       3.8912    5.4419    1.5234    -         0.1       no
temperature  seasonal_naive       0       2.1561    3.0124    0.8442    -         0.1       no
temperature  toto-22m             22M     1.2473    1.8341    0.4882    0.8921    145.2     yes
humidity_prec toto-22m            22M     4.5121    6.1234    0.7123    2.3412    148.3     yes
...
```

> **MASE < 1.0 beats seasonal-naive. CRPS = probabilistic calibration (lower = better). mv = multivariate native.**
> **`-mv` suffix** = Toto running multivariate on clean sensors (excludes formula-derived irradiance + synthetic hub_voltage from the joint batch).

### Available backends

The hybrid engine runs **both** univariate and multivariate paths for Toto backends:
- Plain `toto-22m` = univariate per-field baseline (fair comparison with persistence/seasonal_naive)
- `toto-22m-mv` = multivariate joint forecast when all real sensors are clean (excludes irradiance/hub_voltage)

| Backend | Params | Multivariate | Install |
|---------|--------|--------------|---------|
| `persistence` | 0 | No | Built-in |
| `seasonal_naive` | 0 | No | Built-in |
| `toto-4m` | 4M | Yes | `pip install toto-models` |
| `toto-22m` | 22M | Yes | `pip install toto-models` |
| `toto-313m` | 313M | Yes | `pip install toto-models` |
| `toto-1b` | 1B | Yes | `pip install toto-models` |
| `toto-2.5b` | 2.5B | Yes | `pip install toto-models` |
| `chronos2-tiny` | 9M | No | `pip install chronos-forecasting` |
| `chronos2-small` | 48M | No | `pip install chronos-forecasting` |
| `chronos2-base` | 200M | No | `pip install chronos-forecasting` |
| `chronos2-large` | 710M | No | `pip install chronos-forecasting` |
| `timesfm25` | 200M | No | Install from source |
| `moirai20-small` | ~200M | Yes | `pip install uni2ts` |
| `tirex` | 35M | No | `pip install tirex-ts` |

### Schema I/O

The weather engine outputs schema-compliant JSON when used in `--mode forecast`. The benchmark mode prints a comparison table.

**Schema format (`weather.v1.json`):**
```json
{
  "sequence_number": 424,
  "timestamp": [{"seconds": 1783123456, "nanoseconds": 789000000, "source": "SYSTEM"}],
  "payload": {
    "barometer": 1006.2,
    "humidity_prec": 51.2,
    "temperature": 24.4,
    "wind_direction": 0,
    "wind_speed": 0,
    "irradiance": 450.3,
    "hub_voltage": 3.28
  }
}
```

**Finding your ECCC station ID:**
1. Go to https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
2. Search by city (e.g., "Victoria")
3. Click the station name — the URL will contain `stationID=XXXXX`

---

## 3. Magnetometer — Quiet-Day Baseline & Activity Flags

### Self-test (synthetic, fully offline)

```bash
python -m magnetometer.demos.magnetometer_demo --self-test
```

Injects a 6-hour storm depression and a single-sample glitch spike into synthetic data, then recovers both via the 5-band filter baseline.

### Real data — INTERMAGNET Victoria observatory

```bash
python -m magnetometer.demos.magnetometer_demo --fetch-real-data --days 7 --start-date 2024-01-01
```

**Example output (real run):**
```
Fetching INTERMAGNET data for VIC from 2024-01-01 for 7 days ...

=== INTERMAGNET VIC real data ===
Series length: 10080 samples at 60s cadence (168.0 hours)

Activity flag counts:
  quiet   : 9876
  active  : 104
  storm   : 0
  anomaly : 0

Residual RMS during quiet periods: 2.14 nT
Residual overall RMS: 3.87 nT
Residual min/max: -18.43 / 22.15 nT

Schema messages saved to: vic_magnetometer_schemas.jsonl
```

### Schema I/O

```bash
# Dry-run
python -m magnetometer.demos.magnetometer_demo --fetch-real-data --days 1 --start-date 2024-01-01 --emit-schemas > mag_schemas.jsonl

# Analyze from saved schemas
python -m magnetometer.demos.magnetometer_demo --json-input vic_magnetometer_schemas.jsonl --column x_nt
```

**Schema format (`magnetometer.v1.json`):**
```json
{
  "sequence_number": 67890,
  "timestamp": [{"seconds": 1783123456, "nanoseconds": 789012345, "source": "NTP-LOCAL"}],
  "payload": {"x_nt": 21000.5, "y_nt": -3400.2, "z_nt": 48000.0}
}
```

> **Note:** INTERMAGNET definitive data has a ~1 year delay. If `fetch_real_data` returns HTML instead of IAGA-2002, try an older `--start-date` (e.g. `2024-01-01`).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `TypeError: __hash__ method should return an integer` | Already fixed in current `seismic_demo.py` — the agreement stats now use `float(p.peak_time)` instead of hashing `UTCDateTime` |
| `seisbench` warning spam | Suppressed via `logging.getLogger("seisbench").setLevel(logging.ERROR)` |
| EQTransformer returns 0 picks | Use `--window-s 60` (EQTransformer STEAD expects 60 s input) |
| `ModuleNotFoundError: No module named 'chronos'` | `pip install chronos-forecasting` |
| `ModuleNotFoundError: No module named 'seisbench'` | `pip install seisbench` |
| PhaseNet/EQTransformer weights blocked | Allow `hifis-storage.desy.de` through firewall |
| Chronos weights blocked | Allow `huggingface.co` through firewall |
| ECCC CSV parse error | ECCC occasionally changes column names; check CSV header and update `parse_eccc_csv_to_dataframe` |
| INTERMAGNET timeout / HTML response | BGS service can be slow; use older `--start-date` or shorter `--days` |
| No Z+N+E triplet found (seismic) | Station may not have all three components; try `--channel "HH?"` or `"BH?"` |
| FDSN `NoDataException` | Try different time window, different channel code, or different station (e.g. `--network IU --station KONO`) |
| Toto requires context % 32 == 0 | The engine auto-adjusts context length; this is just a warning |
| `pip install toto-models` fails | Toto requires Python 3.12+. Use `toto-22m` or skip Toto backends. |
| `undefined symbol: ncclCommResume` | Broken CUDA torch install. Run `./run_all_demos.sh --clean-install --self-test` to force CPU-only PyTorch |
| `gluonts` version conflicts | `uni2ts` and `toto-2` want different gluonts versions. The script installs them as optional; the weather engine skips backends that fail to import |

---

## Architecture Notes

### What "real" means

When `--fetch-real-data` or `--real-data` is used:

1. **Raw data is fetched** from the public archive (FDSN, ECCC, INTERMAGNET)
2. **Converted to schema** — each pipeline has `*_to_schemas()` and `schemas_to_*()` converters
3. **Fed into the model** — the schema is reconstructed back into the native format the model expects (ObsPy Stream, pandas Series, numpy array)
4. **Model output is printed** — picks, forecasts, or activity flags with metrics
5. **Schema files are saved** — `.jsonl` files can be re-ingested by `--json-input` or by downstream consumers

### Sensitivity vs. Precision (Seismic)

- **PhaseNet** is the "completeness" model — catches the aftershock swarm but includes some low-confidence picks
- **EQTransformer** is the "precision" model — only reports when very confident (typically ≥ 0.70 probability)
- **Agreement count** tells you how many of EQTransformer's picks are independently confirmed by PhaseNet — this is strong cross-model validation

### Next Steps / Wiring to Live Streams

These demos prove the pipeline end-to-end on public data. To wire to your live NATS/TimescaleDB archive:

1. Replace the `fetch_*` functions with a NATS consumer that reads schema messages from your bus
2. Feed messages into the existing `schemas_to_*` converters
3. Run inference on aligned windows (ring buffer for seismic, rolling window for magnetometer, hourly batch for weather)
4. Publish results (picks, flags, forecasts) back to the bus in a new schema version
