#!/usr/bin/env python3
"""Run the three Cascadia Sentinel instrument demos from one entry point."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "weather" / "results"

DEMOS = {
    "magnetometer": {
        "script": ROOT / "magnetometer" / "detector" / "magnetometer_demo.py",
        "self_test": ["--self-test"],
        "real": ["--fetch-real-data", "--days", "7", "--start-date", "2024-01-01"],
    },
    "weather": {
        "script": ROOT / "weather" / "weather_tsfm_engine_v2_production_hybrid_fixed.py",
        "self_test": [
            "--mode", "benchmark", "--model", "persistence,seasonal_naive",
            "--station-id", "51337", "--year", "2024", "--months", "1",
            "--horizon", "24", "--n-splits", "3",
        ],
        "real": [
            "--mode", "benchmark",
            "--model", "toto-22m,chronos2-small,timesfm25,moirai20-small,persistence,seasonal_naive",
            "--station-id", "51337", "--year", "2024", "--months", "1",
            "--horizon", "24", "--n-splits", "5",
        ],
    },
    "seisometer": {
        "script": ROOT / "seisometer" / "seismic_demo.py",
        "self_test": ["--self-test"],
        "real": [
            "--fetch-real-data", "--network", "IU", "--station", "MAJO",
            "--channel", "BH?", "--start", "2024-01-01T07:00:00",
            "--end", "2024-01-01T11:00:00", "--window-s", "60",
            "--step-s", "30", "--prob-threshold", "0.5",
        ],
    },
}

MOIRAI_MODELS = {"moirai20-small", "moirai20-base", "moirai20-large"}
MOIRAI_VENV = ROOT / "venv_moirai"


def run(name: str, args: list[str], *, python_exe: str | None = None, timeout: int = 600):
    script = DEMOS[name]["script"]
    if not script.exists():
        raise FileNotFoundError(f"Demo script not found: {script}")
    cmd = [python_exe or sys.executable, str(script), *args]
    print(f"\n▶ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if result.stdout:
        print(result.stdout)
    if result.returncode and result.stderr:
        print(result.stderr, file=sys.stderr)
    print(f"✓ {name}: exit={result.returncode}")
    return result


def weather_models(args: list[str]) -> set[str]:
    for i, arg in enumerate(args[:-1]):
        if arg == "--model":
            return {x.strip() for x in args[i + 1].split(",") if x.strip()}
    return set()


def replace_models(args: list[str], models: set[str]) -> list[str]:
    out = list(args)
    value = ",".join(sorted(models))
    for i, arg in enumerate(out[:-1]):
        if arg == "--model":
            out[i + 1] = value
            return out
    raise ValueError("Weather arguments are missing --model")


def run_weather(args: list[str], timeout: int):
    models = weather_models(args)
    main_models = models - MOIRAI_MODELS
    moirai_models = models & MOIRAI_MODELS
    outputs = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if main_models:
        main_args = replace_models(args, main_models) + ["--output", "/tmp/weather_main.json"]
        outputs.append(run("weather", main_args, timeout=timeout))
    else:
        Path("/tmp/weather_main.json").write_text('{"benchmark_metrics": {}}')

    if moirai_models and MOIRAI_VENV.exists():
        moirai_python = MOIRAI_VENV / "bin" / "python"
        moirai_args = replace_models(args, moirai_models) + ["--output", "/tmp/weather_moirai.json"]
        outputs.append(run("weather", moirai_args, python_exe=str(moirai_python), timeout=timeout))
    else:
        if moirai_models:
            print(f"[WARN] {MOIRAI_VENV} not found; skipping Moirai models.")
        Path("/tmp/weather_moirai.json").write_text('{"benchmark_metrics": {}}')

    main_path = Path("/tmp/weather_main.json")
    moirai_path = Path("/tmp/weather_moirai.json")
    merged = json.loads(main_path.read_text())
    extra = json.loads(moirai_path.read_text())
    for field, backends in extra.get("benchmark_metrics", {}).items():
        merged.setdefault("benchmark_metrics", {}).setdefault(field, {}).update(backends)
    (RESULTS_DIR / "weather_merged_results.json").write_text(json.dumps(merged, indent=2))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run magnetometer, weather, and seisometer demos.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--real-data", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    selected = "self_test" if args.self_test else "real"
    failures = []
    for name in ("magnetometer", "weather", "seisometer"):
        try:
            if name == "weather":
                results = run_weather(DEMOS[name][selected], args.timeout)
                if any(r.returncode for r in results):
                    failures.append(name)
            else:
                result = run(name, DEMOS[name][selected], timeout=args.timeout)
                if result.returncode:
                    failures.append(name)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)
            failures.append(name)

    print("\n=== RUN SUMMARY ===")
    for name in ("magnetometer", "weather", "seisometer"):
        print(f"{name:<14} {'FAIL' if name in failures else 'OK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
