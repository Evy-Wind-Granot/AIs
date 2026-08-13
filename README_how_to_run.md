# Cascadia Sentinel — Schema-Compliant Model Demos (with Real Data Fetching)

These three demos are adapted from the original research brief versions. They now:

1. **Speak the project schema** — every input/output can be read/written as the JSON schema messages defined in `magnetometer.v1.json`, `seismometer.v1.json`, and `weather.v1.json`.
2. **Fetch real public data automatically** — no manual downloads required for a first run.
3. **Print metrics** — each demo reports clear, actionable numbers.

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies (all three instruments)
pip install numpy pandas scipy requests                # magnetometer + weather fetch
pip install seisbench obspy                             # seismic
pip install chronos-forecasting torch pandas numpy      # weather forecast
```

> **Note:** The first time you run Chronos or PhaseNet, model weights are downloaded automatically (~50 MB for Chronos, ~1 MB for PhaseNet). Allow `huggingface.co` and `hifis-storage.desy.de` through your firewall, or pre-download on another machine.

---

## 1. Magnetometer — Quiet-Day Baseline & Activity Flags

**Data source:** INTERMAGNET Victoria observatory (`VIC`) via the British Geological Survey web service.

### Run on real data (fetched automatically)
```bash
python magnetometer_demo.py --fetch-real-data --days 7
```

What happens:
- Downloads 7 days of 1-minute XYZF data for `VIC` in IAGA-2002 format
- Parses it into schema messages internally
- Computes the 5-band FIR baseline (>36h + 24h + 12h + 8h + 6h)
- Subtracts baseline → residual
- Flags each sample as `quiet`, `active`, `storm`, or `anomaly`
- Prints counts, residual RMS, and storm/anomaly indices
- Saves schema messages to `vic_magnetometer_schemas.jsonl`

### See the schema format (dry-run)
```bash
python magnetometer_demo.py --fetch-real-data --days 1 --emit-schemas > vic_schemas.jsonl
head -n 3 vic_schemas.jsonl
```

### Run on saved schemas
```bash
python magnetometer_demo.py --json-input vic_magnetometer_schemas.jsonl --column x_nt
```

### Self-test (synthetic, no network)
```bash
python magnetometer_demo.py --self-test
```

**Metrics reported:**
- Activity flag counts (quiet / active / storm / anomaly)
- Residual RMS during quiet periods
- Overall residual min/max
- Storm sample index range
- Anomaly sample indices

---

## 2. Weather — Chronos-2 Zero-Shot Forecasting

**Data source:** Environment and Climate Change Canada (ECCC) historical hourly observations.

### Run on real data (fetched automatically)
```bash
python weather_demo.py --fetch-real-data --station-id 51337 --year 2024 --month 1 --column temperature
```

What happens:
- Downloads January 2024 hourly CSV for station **51337** (Victoria Intl A)
- Maps ECCC columns → schema fields (`temperature`, `humidity_prec`, `barometer`, `wind_speed`, `wind_direction`)
- Converts units (kPa→hPa, km/h→m/s, tens-of-degrees→degrees)
- Runs Chronos-2 zero-shot forecast for the next `--horizon` hours (default 24)
- Compares against **persistence** and **seasonal-naive** baselines
- Prints MAE and RMSE for all three
- Saves schema messages to `weather_station_51337_schemas.jsonl`

### See the schema format (dry-run)
```bash
python weather_demo.py --fetch-real-data --station-id 51337 --year 2024 --month 1 --emit-schemas > weather_schemas.jsonl
head -n 3 weather_schemas.jsonl
```

### Run on saved schemas
```bash
python weather_demo.py --json-input weather_station_51337_schemas.jsonl --column temperature
```

### Self-test (synthetic)
```bash
python weather_demo.py --self-test
```

**Finding your ECCC station ID:**
1. Go to https://climate.weather.gc.ca/historical_data/search_historic_data_e.html
2. Search by city (e.g., "Victoria")
3. Click the station name — the URL will contain `stationID=XXXXX`

**Metrics reported:**
- MAE (Mean Absolute Error) for Chronos, persistence, seasonal-naive
- RMSE for all three
- Horizon table: truth vs. forecast vs. baselines for every step

---

## 3. Seismometer — PhaseNet P/S Picking

**Data source:** EarthScope FDSN web service (`https://service.earthscope.org`) or NRCan.

### Run on real data (fetched automatically)
```bash
python seismic_demo.py --fetch-real-data \
    --network CN --station VGZ \
    --start 2024-01-01T00:00:00 --end 2024-01-01T00:10:00
```

What happens:
- Pulls 10 minutes of broadband waveform data for `CN.VGZ`
- Converts the ObsPy Stream → schema messages (one per trace/channel)
- Reconstructs 3-component triplets (Z + N + E) by network.station.location
- Resamples to 100 Hz and aligns the three channels
- Runs PhaseNet (pretrained on STEAD) for P- and S-wave picking
- Prints pick count, phase, time, probability
- Saves schema messages to `CN_VGZ_seismic_schemas.jsonl`

### See the schema format (dry-run)
```bash
python seismic_demo.py --fetch-real-data --network CN --station VGZ \
    --start 2024-01-01T00:00:00 --end 2024-01-01T00:05:00 --emit-schemas > seismic_schemas.jsonl
```

### Run on saved schemas
```bash
python seismic_demo.py --json-input CN_VGZ_seismic_schemas.jsonl
```

### Self-test (bundled ObsPy data, no network)
```bash
python seismic_demo.py --self-test
```

**Metrics reported:**
- Total pick count (P vs. S)
- Pick probability statistics (mean/min/max)
- Per-triplet trace counts and sample lengths

---

## Schema Format Reference

### Magnetometer (`magnetometer.v1.json`)
```json
{
  "sequence_number": 67890,
  "timestamp": [{"seconds": 1783123456, "nanoseconds": 789012345, "source": "NTP-LOCAL"}],
  "payload": {"x_nt": 21000.5, "y_nt": -3400.2, "z_nt": 48000.0}
}
```

### Weather (`weather.v1.json`)
```json
{
  "sequence_number": 424,
  "timestamp": [{"seconds": 1783123456, "nanoseconds": 789000000, "source": "SYSTEM"}],
  "payload": {
    "barometer": 1006.2,
    "humidity_prec": 51.2,
    "temperature": 24.4,
    "wind_direction": 0,
    "wind_speed": 0
  }
}
```

### Seismometer (`seismometer.v1.json`)
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

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'chronos'` | `pip install chronos-forecasting` |
| `ModuleNotFoundError: No module named 'seisbench'` | `pip install seisbench` |
| PhaseNet weights download blocked | Allow `hifis-storage.desy.de` through firewall, or download on another machine and point `from_pretrained` at a local path |
| Chronos weights download blocked | Allow `huggingface.co` through firewall |
| ECCC CSV parse error | ECCC occasionally changes column names; check the CSV header and update `parse_eccc_csv_to_dataframe` |
| INTERMAGNET timeout | The BGS service can be slow; increase timeout in `fetch_intermagnet_iaga2002` or try a shorter `--days` |
| No Z+N+E triplet found (seismic) | The station may not have all three components for the requested window, or they may be named differently (e.g., `HH1` instead of `HHN`). Try `--channel "HH?"` or `"BH?"` |

---

## Next Steps / Wiring to Live Streams

These demos prove the pipeline end-to-end on public data. To wire to your live NATS/TimescaleDB archive:

1. Replace the `fetch_*` functions with a NATS consumer that reads schema messages from your bus.
2. Feed the messages into the same `schemas_to_*` converters already written here.
3. Run inference on aligned windows (ring buffer for seismic, rolling window for magnetometer, hourly batch for weather).
4. Publish results (picks, flags, forecasts) back to the bus in a new schema version.
