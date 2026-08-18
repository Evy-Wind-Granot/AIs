"""
Seismic demo: PhaseNet + EQTransformer dual-model picking with sliding windows.
Schema-compliant I/O and real data fetching from EarthScope/NRCan FDSN.

Modes:
  --self-test      : ObsPy bundled example waveform (no network)
  --fetch-real-data: pull from EarthScope/NRCan FDSN and process
  --json-input PATH: read schema-format JSON lines, aggregate Z,N,E, then pick
  --emit-schemas   : print schema JSON lines from fetched data (dry-run)

Schema format (seismometer.v1.json):
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

Usage:
    python seismic_demo.py --self-test
    python seismic_demo.py --fetch-real-data --network CN --station VGZ \
        --start 2024-09-26T11:05:00 --end 2024-09-26T11:15:00
    python seismic_demo.py --json-input seismic_schemas.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import obspy
import seisbench.models as sbm
import logging
logging.getLogger("seisbench").setLevel(logging.ERROR)
from obspy import Trace, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException


# ---------------------------------------------------------------------------
# FDSN fetching
# ---------------------------------------------------------------------------
FDSN_SERVICES = {
    "earthscope": "https://service.earthscope.org",
    "nrcan": "https://earthquakescanada.nrcan.gc.ca",
    "iris": "https://service.iris.edu",
}


def fetch_live_stream(network, station, start, end, channel="BH?",
                      service_url="https://service.earthscope.org"):
    client = Client(service_url)
    t0 = UTCDateTime(start)
    t1 = UTCDateTime(end)
    try:
        st = client.get_waveforms(network, station, "*", channel, t0, t1)
    except FDSNNoDataException as exc:
        print(f"\nERROR: No data available from {service_url} for {network}.{station} {channel} {start}--{end}")
        print("Common fixes:")
        print("  1. Try a different time window (some stations have gaps)")
        print("  2. Try a different channel code: --channel HH?  or  --channel EH?")
        print("  3. Try NRCan instead: --service nrcan")
        print("  4. Try a well-known global station: --network IU --station KONO")
        print("  5. Use --self-test to verify the pipeline works without network")
        raise SystemExit(1) from exc
    print(f"Pulled live stream: {st}")
    return st


def load_self_test_stream():
    st = obspy.read()
    print(f"Loaded self-test stream: {st}")
    return st


# ---------------------------------------------------------------------------
# Schema <-> ObsPy conversions
# ---------------------------------------------------------------------------
def stream_to_schemas(stream, sequence_start=1):
    seq = sequence_start
    for tr in stream:
        stats = tr.stats
        start_dt = stats.starttime.datetime
        ts = int(start_dt.timestamp())
        ns = int((start_dt.timestamp() - ts) * 1e9)
        samples = [int(v) for v in tr.data.tolist()]
        msg = {
            "sequence_number": seq,
            "timestamp": [{"seconds": ts, "nanoseconds": ns, "source": "FDSN-EARTHSCOPE"}],
            "payload": {
                "channel_id": f"{stats.network}.{stats.station}.{stats.location}.{stats.channel}",
                "sample_rate": float(stats.sampling_rate),
                "sample_count": len(samples),
                "samples": samples,
            }
        }
        yield msg
        seq += 1


def schemas_to_stream(schema_iterable):
    stream = Stream()
    for msg in schema_iterable:
        payload = msg.get("payload", {})
        ch_id = payload.get("channel_id", "...")
        parts = ch_id.split(".")
        if len(parts) != 4:
            continue
        net, sta, loc, cha = parts
        sr = payload.get("sample_rate", 100.0)
        samples = payload.get("samples", [])
        if not samples:
            continue
        ts_list = msg.get("timestamp", [{}])
        ts_info = ts_list[0] if ts_list else {}
        sec = ts_info.get("seconds", 0)
        ns = ts_info.get("nanoseconds", 0)
        starttime = UTCDateTime(sec + ns / 1e9)
        data = np.array(samples, dtype=np.int32)
        tr = Trace(data=data, header={
            "network": net, "station": sta, "location": loc,
            "channel": cha, "sampling_rate": sr, "starttime": starttime,
        })
        stream.append(tr)
    return stream.merge()


# ---------------------------------------------------------------------------
# Triplet aggregation
# ---------------------------------------------------------------------------
def aggregate_triplets(stream, target_sr=100.0):
    groups = defaultdict(list)
    for tr in stream:
        key = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}"
        groups[key].append(tr)

    triplets = []
    for key, traces in groups.items():
        by_chan = {}
        for tr in traces:
            comp = tr.stats.channel[-1].upper()
            by_chan[comp] = tr

        has_z = "Z" in by_chan
        if "N" in by_chan and "E" in by_chan:
            h1, h2 = "N", "E"
        elif "1" in by_chan and "2" in by_chan:
            h1, h2 = "1", "2"
        else:
            continue

        if not has_z:
            continue

        st3 = Stream()
        for comp in ["Z", h1, h2]:
            st3.append(by_chan[comp])

        st3.merge(method=1, fill_value=0)
        for tr in st3:
            if tr.stats.sampling_rate != target_sr:
                tr.resample(target_sr)

        if len(st3) < 3:
            continue

        # Rename 1/2 -> N/E for model compatibility
        for tr in st3:
            last = tr.stats.channel[-1].upper()
            if last == "1":
                tr.stats.channel = tr.stats.channel[:-1] + "N"
            elif last == "2":
                tr.stats.channel = tr.stats.channel[:-1] + "E"

        triplets.append((key, st3))
    return triplets


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
def sliding_windows(st3, window_s=30, step_s=5, min_samples=3001):
    """
    Yield overlapping windowed copies of a 3C Stream.
    Each window is padded to min_samples if needed.
    """
    if len(st3) == 0:
        return

    # Common time range
    start = max(tr.stats.starttime for tr in st3)
    end = min(tr.stats.endtime for tr in st3)
    total_s = end - start

    if total_s < window_s:
        # Only one window, pad if needed
        window = deepcopy(st3)
        for tr in window:
            n = len(tr.data)
            if n < min_samples:
                needed = min_samples - n
                new_end = tr.stats.endtime + needed / tr.stats.sampling_rate
                tr.trim(starttime=tr.stats.starttime, endtime=new_end,
                        pad=True, fill_value=0)
        yield 0, window
        return

    # Slide
    t = start
    idx = 0
    while t + window_s <= end:
        window = deepcopy(st3)
        w_start = t
        w_end = t + window_s
        for tr in window:
            tr.trim(starttime=w_start, endtime=w_end, pad=True, fill_value=0)
            # Ensure minimum length
            n = len(tr.data)
            if n < min_samples:
                needed = min_samples - n
                tr.trim(starttime=tr.stats.starttime,
                        endtime=tr.stats.endtime + needed / tr.stats.sampling_rate,
                        pad=True, fill_value=0)
        yield idx, window
        t += step_s
        idx += 1


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate_picks(picks, tolerance_s=1.0):
    """
    Remove duplicate picks (same phase, within tolerance seconds).
    Keeps the highest-probability pick.
    """
    by_phase = defaultdict(list)
    for p in picks:
        by_phase[p.phase].append(p)

    unique = []
    for phase, plist in by_phase.items():
        # Sort by time
        plist.sort(key=lambda p: p.peak_time)
        for p in plist:
            # Check if within tolerance of any already-kept pick
            matched = False
            for u in unique:
                if u.phase == phase and abs(u.peak_time - p.peak_time) <= tolerance_s:
                    matched = True
                    if p.peak_value > u.peak_value:
                        # Replace with higher probability
                        unique[unique.index(u)] = p
                    break
            if not matched:
                unique.append(p)
    return sorted(unique, key=lambda p: p.peak_time)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
def run_phasenet(stream, model):
    annotations = model.annotate(stream)
    picks = model.classify(stream).picks
    return picks, annotations


def run_eqtransformer(stream, model):
    annotations = model.annotate(stream)
    picks = model.classify(stream).picks
    return picks, annotations


def run_dual_models(window_stream, phasenet_model, eqt_model, window_idx):
    pn_picks, pn_ann = run_phasenet(window_stream, phasenet_model)
    eqt_picks, eqt_ann = run_eqtransformer(window_stream, eqt_model)

    # Tag picks with model name for reporting
    for p in pn_picks:
        p.model = "PhaseNet"
    for p in eqt_picks:
        p.model = "EQTransformer"

    return pn_picks, eqt_picks


def report_picks(picks, label=""):
    p_picks = [p for p in picks if p.phase == "P"]
    s_picks = [p for p in picks if p.phase == "S"]
    print(f"  {label} P={len(p_picks)}, S={len(s_picks)}")
    for p in picks[:10]:
        print(f"    {p.phase:>2s}  {p.peak_time}  prob={p.peak_value:.3f}  ({getattr(p, 'model', '?')})")
    if len(picks) > 10:
        print(f"    ... ({len(picks) - 10} more)")
    return p_picks, s_picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Dual-model seismic picking (PhaseNet + EQTransformer) with sliding windows")
    ap.add_argument("--self-test", action="store_true", help="Use ObsPy bundled example waveform")
    ap.add_argument("--fetch-real-data", action="store_true", help="Pull real waveform data via FDSN")
    ap.add_argument("--json-input", default=None, help="Path to JSON Lines file with schema messages")
    ap.add_argument("--emit-schemas", action="store_true", help="Print schema JSON lines and exit")
    ap.add_argument("--network", default="CN")
    ap.add_argument("--station", default="VGZ")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--channel", default="BH?")
    ap.add_argument("--service", default="earthscope",
                    help="FDSN service: earthscope, nrcan, iris, or a full URL")
    ap.add_argument("--window-s", type=int, default=30, help="Sliding window length in seconds")
    ap.add_argument("--step-s", type=int, default=5, help="Sliding window step in seconds")
    ap.add_argument("--prob-threshold", type=float, default=0.5, help="Minimum pick probability to report")#this one will not change the run demos threshold, it is just for the demo script to run with a higher threshold
    args = ap.parse_args()

    service_url = FDSN_SERVICES.get(args.service, args.service)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    if args.json_input:
        print(f"Reading schema messages from {args.json_input} ...")
        with open(args.json_input, "r") as f:
            schemas = [json.loads(line) for line in f if line.strip()]
        stream = schemas_to_stream(schemas)
        print(f"Reconstructed stream from schemas: {stream}")

    elif args.fetch_real_data:
        if not (args.start and args.end):
            sys.exit("--fetch-real-data requires --start and --end")
        stream = fetch_live_stream(args.network, args.station, args.start, args.end,
                                   channel=args.channel, service_url=service_url)
        if args.emit_schemas:
            for msg in stream_to_schemas(stream):
                print(json.dumps(msg))
            return

    else:
        stream = load_self_test_stream()

    # -----------------------------------------------------------------------
    # Aggregate to 3C
    # -----------------------------------------------------------------------
    triplets = aggregate_triplets(stream, target_sr=100.0)
    if not triplets:
        print("No complete Z+N+E triplets found.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load models once
    # -----------------------------------------------------------------------
    print("\nLoading PhaseNet (STEAD) ...")
    phasenet_model = sbm.PhaseNet.from_pretrained("stead")
    phasenet_model.eval()

    print("Loading EQTransformer (STEAD) ...")
    eqt_model = sbm.EQTransformer.from_pretrained("stead")
    eqt_model.eval()

    # -----------------------------------------------------------------------
    # Sliding window inference
    # -----------------------------------------------------------------------
    all_pn_picks = []
    all_eqt_picks = []
    all_combined = []

    for group_id, st3 in triplets:
        print(f"\n=== Station: {group_id} | Duration: {st3[0].stats.endtime - st3[0].stats.starttime:.1f}s ===")
        print(f"Sliding window: {args.window_s}s window, {args.step_s}s step")

        for idx, window in sliding_windows(st3, window_s=args.window_s, step_s=args.step_s):
            w_start = window[0].stats.starttime
            w_end = window[0].stats.endtime
            print(f"\n  Window {idx}: {w_start} -- {w_end} ({window[0].stats.npts} samples)")

            pn_picks, eqt_picks = run_dual_models(window, phasenet_model, eqt_model, idx)

            # Filter by probability threshold
            pn_picks = [p for p in pn_picks if p.peak_value >= args.prob_threshold]
            eqt_picks = [p for p in eqt_picks if p.peak_value >= args.prob_threshold]

            if pn_picks or eqt_picks:
                print(f"    PhaseNet picks: {len(pn_picks)}")
                for p in pn_picks:
                    print(f"      {p.phase:>2s}  {p.peak_time}  prob={p.peak_value:.3f}")
                print(f"    EQTransformer picks: {len(eqt_picks)}")
                for p in eqt_picks:
                    print(f"      {p.phase:>2s}  {p.peak_time}  prob={p.peak_value:.3f}")

            all_pn_picks.extend(pn_picks)
            all_eqt_picks.extend(eqt_picks)
            all_combined.extend(pn_picks + eqt_picks)

    # -----------------------------------------------------------------------
    # Deduplicate and report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULTS (deduplicated)")
    print("=" * 60)

    pn_unique = deduplicate_picks(all_pn_picks, tolerance_s=1.0)
    eqt_unique = deduplicate_picks(all_eqt_picks, tolerance_s=1.0)
    combined_unique = deduplicate_picks(all_combined, tolerance_s=1.0)

    print(f"\nPhaseNet total unique picks: {len(pn_unique)}")
    report_picks(pn_unique, "PhaseNet")

    print(f"\nEQTransformer total unique picks: {len(eqt_unique)}")
    report_picks(eqt_unique, "EQTransformer")

    print(f"\nCombined total unique picks: {len(combined_unique)}")
    report_picks(combined_unique, "Combined")

    # Agreement stats (tolerance-based, avoids hashing UTCDateTime)
    agreement = 0
    for pn in pn_unique:
        for eq in eqt_unique:
            if (pn.phase == eq.phase and
                abs(float(pn.peak_time) - float(eq.peak_time)) <= 1.0):
                agreement += 1
                break
    print(f"\nModel agreement (same phase + time within 1s): {agreement} picks")

    # Save schemas
    if args.fetch_real_data and not args.emit_schemas:
        out_file = f"{args.network}_{args.station}_seismic_schemas.jsonl"
        with open(out_file, "w") as f:
            for msg in stream_to_schemas(stream):
                f.write(json.dumps(msg) + "\n")
        print(f"\nSchema messages saved to: {out_file}")


if __name__ == "__main__":
    main()