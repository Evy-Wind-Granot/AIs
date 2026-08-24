# Magnetometer detector benchmark

## What is ground truth?

The primary benchmark is **station-local reference**, not Kp. Kp is a global geomagnetic index and is retained only as a secondary context comparison.

The local reference is built from the observatory's own residual signal using:

- 15-minute disturbance RMS and range
- 3-hour local magnetic range
- 15-minute dB/dt RMS
- frozen high-percentile thresholds
- persistence and event merging

The reference does **not** use the adaptive detector's thresholds or state machine. This avoids directly scoring a detector against its own labels.

It is still a reference, not an absolute physical truth. A future release can add expert-reviewed event annotations for an even stronger benchmark.

## Detector design

`adaptive_detector.py` is now station-local and independent of Kp/Dst.

It uses:

- a robust quiet noise floor from the lower-amplitude residual population
- a causal rolling MAD scale over approximately 3 hours
- a separate rolling robust scale for dB/dt
- amplitude and derivative onset evidence
- hysteresis: harder to enter than to clear
- persistence requirements
- event merging
- a short derivative-only anomaly path
- adaptive severity thresholds expressed in local sigma units

The rolling scale is capped relative to the quiet floor so a prolonged storm cannot indefinitely inflate the threshold and hide itself.

## Self-test

```bash
python magnetometer/detector/local_benchmark_self_test.py
```

## One-month benchmark

```bash
python magnetometer/detector/performance_metrics.py \
  --observatory VIC \
  --start-date 2025-01-01 \
  --days 30 \
  --column f_nt \
  --chunk-days 7 \
  --output /tmp/vic_local_metrics.json
```

## Full-year benchmark

Use chunking so INTERMAGNET is not asked for a year in one HTTP request:

```bash
python magnetometer/detector/performance_metrics.py \
  --observatory VIC \
  --start-date 2025-01-01 \
  --days 365 \
  --column f_nt \
  --chunk-days 7 \
  --output /tmp/VIC_2025.json
```

Repeat for BOU and other stations.

## What to judge

For the **local** benchmark, prioritize:

1. event F1
2. event recall — missed local disturbances are costly
3. event precision / false events per day
4. median and 95th-percentile detection latency
5. flagged fraction
6. state changes per day
7. stability across stations and years

Kp precision/recall is useful as context, but it must not be used as the production detector's label or dependency.
