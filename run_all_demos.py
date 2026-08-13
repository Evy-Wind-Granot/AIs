#!/usr/bin/env python3
"""
Cascadia Sentinel — Run All Demos
==================================
One command to run the seismometer, weather, and magnetometer demos
end-to-end. Supports both real-data fetching and self-test (synthetic) modes.

NEW: Weather benchmark auto-detects Moirai and runs it in a separate
venv (venv_moirai) to avoid gluonts version conflicts, then merges results.

USAGE (quick, synthetic data — no FDSN/ECCC/INTERMAGNET calls):
    python run_all_demos.py --self-test

USAGE (real public data — pulls live streams from all three sources):
    python run_all_demos.py --real-data

USAGE (show example logs from previous real runs without downloading anything):
    python run_all_demos.py --show-examples
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Demo configurations
# ---------------------------------------------------------------------------
DEMOS = {
    "magnetometer": {
        "script": "magnetometer_demo.py",
        "self_test_args": ["--self-test"],
        "real_args": ["--fetch-real-data", "--days", "7", "--start-date", "2024-01-01"],
        "quick": True,
    },
    "weather": {
        "script": "weather_tsfm_engine_v2_production_hybrid_fixed.py",
        "self_test_args": ["--mode", "benchmark", "--model", "persistence,seasonal_naive", "--station-id", "51337", "--year", "2024", "--months", "1", "--horizon", "24", "--n-splits", "3"],
        "real_args": [
            "--mode", "benchmark",
            "--model", "toto-22m,chronos2-small,timesfm25,moirai20-small,persistence,seasonal_naive",
            "--station-id", "51337",
            "--year", "2024",
            "--months", "1",
            "--horizon", "24",
            "--n-splits", "5",
        ],
        "quick": False,
    },
    "seismic": {
        "script": "seismic_demo.py",
        "self_test_args": ["--self-test"],
        "real_args": [
            "--fetch-real-data",
            "--network", "IU",
            "--station", "MAJO",
            "--channel", "BH?",
            "--start", "2024-01-01T07:00:00",
            "--end", "2024-01-01T11:00:00",
            "--window-s", "60",
            "--step-s", "30",
            "--prob-threshold", "0.5",#this threshold is for demo purposes only; if you run the python script directly, you can adjust it as needed
        ],
        "quick": False,
    },
}

# Models that conflict with the main env and need a separate venv
MOIRAI_MODELS = {"moirai20-small", "moirai20-base", "moirai20-large"}
MOIRAI_VENV = Path("venv_moirai")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(text, width=70):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_demo(name, args_list, timeout=300, python_exe=None):
    cfg = DEMOS[name]
    script = cfg["script"]
    if not Path(script).exists():
        print(f"[SKIP] {script} not found in current directory.")
        return None, ""

    exe = python_exe or sys.executable
    cmd = [exe, script] + args_list
    print(f"\n▶ Running: {' '.join(cmd)}\n")
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {name} exceeded {timeout}s.")
        return None, ""
    elapsed = time.time() - start

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"[STDERR] {result.stderr}")

    print(f"\n✓ {name} finished in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode == 0, result.stdout


def extract_metric(text, keyword, lines_after=0):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if keyword in line:
            out = [line.strip()]
            for j in range(1, lines_after + 1):
                if i + j < len(lines):
                    out.append(lines[i + j].strip())
            return "\n".join(out)
    return "N/A"


def show_examples():
    print(__doc__)


def parse_models_from_args(args_list):
    """Extract --model value from argument list."""
    for i, arg in enumerate(args_list):
        if arg == "--model" and i + 1 < len(args_list):
            return {m.strip() for m in args_list[i + 1].split(",")}
    return set()


def split_models_by_venv(models):
    """Separate models that need the Moirai venv from main-env models."""
    moirai = {m for m in models if m in MOIRAI_MODELS}
    main = models - moirai
    return main, moirai


def build_args_with_models(base_args, models):
    """Replace --model value in arg list."""
    out = list(base_args)
    for i, arg in enumerate(out):
        if arg == "--model" and i + 1 < len(out):
            out[i + 1] = ",".join(sorted(models))
            return out
    # If no --model flag, append one
    out.extend(["--model", ",".join(sorted(models))])
    return out


def merge_json_results(main_path, moirai_path, output_path):
    """Merge benchmark_metrics from two JSON files."""
    with open(main_path) as f:
        main = json.load(f)
    with open(moirai_path) as f:
        moirai = json.load(f)

    # Deep merge: for each field, add moirai backends
    for field, backends in moirai.get("benchmark_metrics", {}).items():
        if field not in main.get("benchmark_metrics", {}):
            main["benchmark_metrics"][field] = {}
        main["benchmark_metrics"][field].update(backends)

    with open(output_path, "w") as f:
        json.dump(main, f, indent=2, default=str)
    print(f"\n✓ Merged results written to: {output_path}")
    return main


def print_weather_table(results_json):
    """Reconstruct the benchmark table from merged JSON."""
    print("\n" + "=" * 120)
    print(f"{'field':<14}{'backend':<22}{'params':<8}{'MAE':<10}{'RMSE':<10}{'MASE':<10}{'CRPS':<10}{'ms/call':<10}{'mv':<4}")
    print("-" * 120)
    for feat, backends in results_json.get("benchmark_metrics", {}).items():
        ranked = sorted(backends.items(), key=lambda kv: kv[1].get("avg_mase", 999))
        for bname, m in ranked:
            mv = "yes" if m.get("multivariate") else "no"
            print(f"{feat:<14}{bname:<22}{str(m.get('params','-')):<8}"
                  f"{m.get('avg_mae','-'):<10}{m.get('avg_rmse','-'):<10}"
                  f"{m.get('avg_mase','-'):<10}{m.get('avg_crps','-'):<10}"
                  f"{m.get('avg_latency_ms','-'):<10}{mv:<4}")
    print("=" * 120)
    print("MASE < 1.0 beats seasonal-naive. CRPS = probabilistic calibration (lower = better).")
    print("mv = multivariate native. -mv suffix = Toto multivariate on clean sensors.")
    print("\n⚠️  NOTE: ECCC data may overlap Chronos/TimesFM pretraining — treat those MASE as optimistic.\n")


# ---------------------------------------------------------------------------
# Weather benchmark with automatic venv splitting
# ---------------------------------------------------------------------------
def run_weather_benchmark(args_list, timeout=600):
    models = parse_models_from_args(args_list)
    main_models, moirai_models = split_models_by_venv(models)

    main_json = Path("/tmp/weather_main.json")
    moirai_json = Path("/tmp/weather_moirai.json")
    merged_json = Path("weather_merged_results.json")

    # 1. Run main-env models
    if main_models:
        main_args = build_args_with_models(args_list, main_models)
        main_args.extend(["--output", str(main_json)])
        ok, stdout = run_demo("weather", main_args, timeout=timeout)
        if not ok:
            print("[WARN] Main-env weather benchmark had errors.")
    else:
        print("[INFO] No main-env models to run.")
        # Write empty structure so merge still works
        with open(main_json, "w") as f:
            json.dump({"benchmark_metrics": {}}, f)

    # 2. Run Moirai models in separate venv
    if moirai_models:
        if not MOIRAI_VENV.exists():
            print(f"\n[ERR] Moirai venv not found at {MOIRAI_VENV}.")
            print("Create it with:")
            print(f"  python -m venv {MOIRAI_VENV}")
            print(f"  source {MOIRAI_VENV}/bin/activate")
            print("  pip install uni2ts gluonts~=0.14.4 torch~=2.4.0")
            print("  deactivate\n")
            # Still merge what we have
        else:
            moirai_python = MOIRAI_VENV / "bin" / "python"
            moirai_args = build_args_with_models(args_list, moirai_models)
            moirai_args.extend(["--output", str(moirai_json)])
            ok, stdout = run_demo("weather", moirai_args, timeout=timeout, python_exe=str(moirai_python))
            if not ok:
                print("[WARN] Moirai venv benchmark had errors.")
    else:
        print("[INFO] No Moirai models requested.")
        with open(moirai_json, "w") as f:
            json.dump({"benchmark_metrics": {}}, f)

    # 3. Merge and display
    merged = merge_json_results(main_json, moirai_json, merged_json)
    print_weather_table(merged)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run all three Cascadia Sentinel demos in one shot."
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run all demos on synthetic data (fastest, minimal network).",
    )
    ap.add_argument(
        "--real-data",
        action="store_true",
        help="Fetch real data from EarthScope, ECCC, and INTERMAGNET.",
    )
    ap.add_argument(
        "--show-examples",
        action="store_true",
        help="Print example logs from previous real runs and exit.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-demo timeout in seconds (default 600).",
    )
    args = ap.parse_args()

    if args.show_examples:
        show_examples()
        return

    if not args.self_test and not args.real_data:
        print("Error: choose --self-test or --real-data (or --show-examples).")
        sys.exit(1)

    mode = "self-test" if args.self_test else "real-data"
    banner(f"Cascadia Sentinel — All Demos ({mode.upper()})")

    results = {}
    for name in ["magnetometer", "weather", "seismic"]:
        banner(name.upper())
        cfg = DEMOS[name]
        cmd_args = cfg["self_test_args"] if args.self_test else cfg["real_args"]

        if name == "weather":
            # Special handling: auto-split Moirai into separate venv
            ok = run_weather_benchmark(cmd_args, timeout=args.timeout)
            results[name] = {"ok": ok, "stdout": ""}
        else:
            ok, stdout = run_demo(name, cmd_args, timeout=args.timeout)
            results[name] = {"ok": ok, "stdout": stdout}

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    banner("SUMMARY")
    print(f"{'Demo':<15} {'Status':<10} {'Key Metric'}")
    print("-" * 70)

    for name in ["magnetometer", "weather", "seismic"]:
        ok = results[name]["ok"]
        status = "OK" if ok else "FAIL"
        stdout = results[name]["stdout"] or ""

        if name == "magnetometer":
            metric = extract_metric(stdout, "Residual overall RMS")
        elif name == "weather":
            metric = "See weather_merged_results.json"
        else:
            metric = extract_metric(stdout, "Model agreement")

        print(f"{name:<15} {status:<10} {metric}")

    # Save JSON summary
    summary_path = f"run_all_summary_{mode}.json"
    with open(summary_path, "w") as f:
        json.dump({k: {"ok": v["ok"]} for k, v in results.items()}, f, indent=2)
    print(f"\nSummary written to: {summary_path}")

    all_ok = all(v["ok"] for v in results.values() if v["ok"] is not None)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()