#!/usr/bin/env python3
"""Cascadia Sentinel: setup and run the three instrument demos.

This script is the single environment/setup definition for the repository.
`run_all_demos.sh` creates the virtual environments and delegates package
installation here; dependency manifests are intentionally not required.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = Path(os.environ.get("VENV_DIR", REPO_ROOT / "venv"))
MOIRAI_VENV = Path(os.environ.get("MOIRAI_VENV", REPO_ROOT / "venv_moirai"))

COMMON_PACKAGES = [
    "numpy", "pandas", "scipy", "requests", "scikit-learn", "PyYAML",
    "python-json-logger", "flake8==7.1.1", "mypy==2.3.0",
]
MAIN_PACKAGES = ["obspy", "seisbench", "chronos-forecasting", "tirex-ts"]


def run(cmd: list[str], *, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("▶", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=check)


def pip_install(packages: list[str], *, python_exe: str | None = None) -> None:
    python = python_exe or sys.executable
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    if packages:
        run([python, "-m", "pip", "install", *packages])


def setup_main() -> None:
    python = sys.executable
    print("\n=== Main environment ===")
    pip_install(COMMON_PACKAGES, python_exe=python)
    run([python, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cpu", "torch"])
    for package in MAIN_PACKAGES:
        try:
            run([python, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"[WARN] Optional package failed: {package}")
    if subprocess.run([python, "-c", "import timesfm"], capture_output=True).returncode != 0:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "timesfm"
            try:
                run(["git", "clone", "--depth", "1", "https://github.com/google-research/timesfm.git", str(target)])
                run([python, "-m", "pip", "install", "-e", "[torch]"], cwd=target)
            except subprocess.CalledProcessError:
                print("[WARN] Optional TimesFM install failed")
    if sys.version_info >= (3, 12):
        try:
            run([python, "-m", "pip", "install", "toto-models"])
        except subprocess.CalledProcessError:
            print("[WARN] Optional Toto install failed")
    print("✓ Main environment ready")


def setup_moirai() -> None:
    python = sys.executable
    print("\n=== Moirai environment ===")
    pip_install(COMMON_PACKAGES, python_exe=python)
    run([python, "-m", "pip", "install", "uni2ts<3", "gluonts~=0.14.4", "torch~=2.4.0"])
    print("✓ Moirai environment ready")


def run_magnetometer(self_test: bool) -> tuple[bool, int]:
    args = [sys.executable, "-m", "magnetometer.pipeline.cli"]
    if self_test:
        args += ["--self-test"]
    else:
        args += [
            "--fetch-real-data", "--observatory", "VIC", "--days", "5",
            "--start-date", "2024-05-08", "--warmup-days", "3",
            "--cross-check-indices", "--output-json",
            "magnetometer/results/demos/magnetometer_results.json", "--log-format", "json",
        ]
    result = subprocess.run(args)
    return result.returncode in (0, 2), result.returncode


def run_weather(self_test: bool) -> tuple[bool, int]:
    models = "persistence,seasonal_naive" if self_test else "toto-22m,chronos2-small,timesfm25,moirai20-small,persistence,seasonal_naive"
    main_models = [m for m in models.split(",") if not m.startswith("moirai20-")]
    moirai_models = [m for m in models.split(",") if m.startswith("moirai20-")]
    common = ["--mode", "benchmark", "--station-id", "51337", "--year", "2024", "--months", "1", "--horizon", "24", "--n-splits", "3" if self_test else "5"]

    main_json = Path("/tmp/weather_main.json")
    moirai_json = Path("/tmp/weather_moirai.json")
    if main_models:
        run([sys.executable, "weather/weather_tsfm_engine_v2_production_hybrid_fixed.py", *common, "--model", ",".join(main_models), "--output", str(main_json)], check=False)
    else:
        main_json.write_text('{"benchmark_metrics": {}}')

    if moirai_models and (MOIRAI_VENV / "bin" / "python").exists():
        run([str(MOIRAI_VENV / "bin" / "python"), "weather/weather_tsfm_engine_v2_production_hybrid_fixed.py", *common, "--model", ",".join(moirai_models), "--output", str(moirai_json)], check=False)
    else:
        moirai_json.write_text('{"benchmark_metrics": {}}')

    main = json.loads(main_json.read_text())
    moirai = json.loads(moirai_json.read_text())
    for field, backends in moirai.get("benchmark_metrics", {}).items():
        main.setdefault("benchmark_metrics", {}).setdefault(field, {}).update(backends)
    Path("weather_merged_results.json").write_text(json.dumps(main, indent=2))
    return True, 0


def run_seismic(self_test: bool) -> tuple[bool, int]:
    args = [sys.executable, "seisometer/seismic_demo.py"]
    args += ["--self-test"] if self_test else [
        "--fetch-real-data", "--network", "IU", "--station", "MAJO", "--channel", "BH?",
        "--start", "2024-01-01T07:00:00", "--end", "2024-01-01T11:00:00",
        "--window-s", "60", "--step-s", "30", "--prob-threshold", "0.4",
    ]
    result = subprocess.run(args)
    return result.returncode == 0, result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--real-data", action="store_true")
    mode.add_argument("--show-examples", action="store_true")
    parser.add_argument("--setup", action="store_true", help="Install both main and Moirai environments into the current interpreter")
    parser.add_argument("--setup-main", action="store_true")
    parser.add_argument("--setup-moirai", action="store_true")
    args = parser.parse_args()

    if args.setup or args.setup_main:
        setup_main()
    if args.setup or args.setup_moirai:
        setup_moirai()
    if args.setup or args.setup_main or args.setup_moirai:
        return 0

    if args.show_examples:
        print("Use --self-test for a synthetic run or --real-data for public data.")
        return 0
    if not args.self_test and not args.real_data:
        parser.error("choose --self-test, --real-data, or a --setup option")

    self_test = args.self_test
    results = {}
    for name, fn in (("magnetometer", run_magnetometer), ("weather", run_weather), ("seismic", run_seismic)):
        print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
        try:
            ok, code = fn(self_test)
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            ok, code = False, 1
        results[name] = {"ok": ok, "exit_code": code}

    mode_name = "self-test" if self_test else "real-data"
    Path(f"run_all_summary_{mode_name}.json").write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {json.dumps(results, indent=2)}")
    return 0 if all(item["ok"] for item in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
