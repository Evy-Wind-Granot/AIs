#!/usr/bin/env python3
"""
Cascadia Sentinel — Random Demo Picker
======================================

Curated catalog of real-world scenarios across all three instruments,
each rated 1-5 on an "interestingness" scale.  Picks one at random
(per instrument or all-at-once) and runs the appropriate demo.

Great for:
  • Showcases / demos where you want a surprise every time
  • Smoke-testing the pipelines against varied data regimes
  • Building intuition about how model performance varies with
    signal complexity (quiet craton vs. aftershock swarm,
    summer doldrums vs. arctic outbreak, quiet QDC vs. G5 storm)

Usage:
  # Pick one random scenario for each instrument and run everything
  python run_random_demo.py --instrument all

  # Pick a random seismic scenario only
  python run_random_demo.py --instrument seismic

  # Only pick from the spicy stuff (interest ≥ 4)
  python run_random_demo.py --instrument all --min-interest 4

  # Weighted random — higher-interest scenarios are more likely
  python run_random_demo.py --instrument all --weighted

  # Reproducible randomness
  python run_random_demo.py --instrument seismic --seed 42

  # Just browse the catalog
  python run_random_demo.py --list

  # Dry-run: show what would be executed without running it
  python run_random_demo.py --instrument weather --dry-run
"""

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Catalog data model
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    instrument: str          # "seismic" | "weather" | "magnetometer"
    name: str                # human-readable label
    interest: int            # 1 = boring, 5 = extreme
    description: str         # what makes this scenario interesting
    args: List[str]          # CLI args to pass to the demo script
    script: str              # target demo file

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Seismic catalog
# ---------------------------------------------------------------------------
# Interest scale:
#   1 = stable continental, near-zero picks expected
#   2 = persistent microseismicity, handful of picks
#   3 = aftershock sequence, moderate pick density
#   4 = major event regional record, high pick density
#   5 = extended window on dense aftershock swarm, maximum stress test
SEISMIC_SCENARIOS: List[Scenario] = [
    Scenario(
        instrument="seismic",
        name="KONO Quiet Craton",
        interest=1,
        description=(
            "Kongsberg, Norway — stable Baltic Shield, minimal ambient seismicity. "
            "Expect 0-2 low-probability picks. Perfect baseline for false-positive rate."
        ),
        args=[
            "--fetch-real-data", "--network", "IU", "--station", "KONO", "--channel", "BH?",
            "--start", "2024-06-15T00:00:00", "--end", "2024-06-15T01:00:00",
            "--window-s", "60", "--step-s", "30", "--prob-threshold", "0.30",
        ],
        script="seismic_demo.py",
    ),
    Scenario(
        instrument="seismic",
        name="VGZ Pacific Microseisms",
        interest=2,
        description=(
            "Vancouver Island — persistent Pacific microseismicity and Cascadia "
            "subduction whisper. 5-15 picks expected, mostly S-wave coda."
        ),
        args=[
            "--fetch-real-data", "--network", "CN", "--station", "VGZ", "--channel", "HH?",
            "--start", "2024-03-10T12:00:00", "--end", "2024-03-10T13:30:00",
            "--window-s", "60", "--step-s", "30", "--prob-threshold", "0.25",
        ],
        script="seismic_demo.py",
    ),
    Scenario(
        instrument="seismic",
        name="MAJO Noto Aftershocks",
        interest=3,
        description=(
            "Matsushiro, Japan — Noto Peninsula M7.6 aftershock swarm (Jan 1, 2024). "
            "Dense pick field. PhaseNet dominates; EQTransformer acts as quality gate."
        ),
        args=[
            "--fetch-real-data", "--network", "IU", "--station", "MAJO", "--channel", "BH?",
            "--start", "2024-01-01T07:00:00", "--end", "2024-01-01T11:00:00",
            "--window-s", "60", "--step-s", "30", "--prob-threshold", "0.15",
        ],
        script="seismic_demo.py",
    ),
    Scenario(
        instrument="seismic",
        name="TATO Taiwan M7.4",
        interest=4,
        description=(
            "Yushan, Taiwan — April 2, 2024 M7.4 Hualien mainshock + immediate aftershocks. "
            "Regional strong motion at ~120 km epicentral distance. High SNR arrivals."
        ),
        args=[
            "--fetch-real-data", "--network", "IU", "--station", "TATO", "--channel", "BH?",
            "--start", "2024-04-03T00:00:00", "--end", "2024-04-03T04:00:00",
            "--window-s", "60", "--step-s", "30", "--prob-threshold", "0.12",
        ],
        script="seismic_demo.py",
    ),
    Scenario(
        instrument="seismic",
        name="MAJO Noto Marathon",
        interest=5,
        description=(
            "Matsushiro, Japan — 12-hour window during peak Noto aftershock activity. "
            "Maximum pick density stress test. Expect 100+ PhaseNet picks."
        ),
        args=[
            "--fetch-real-data", "--network", "IU", "--station", "MAJO", "--channel", "BH?",
            "--start", "2024-01-01T06:00:00", "--end", "2024-01-01T18:00:00",
            "--window-s", "60", "--step-s", "30", "--prob-threshold", "0.08",
        ],
        script="seismic_demo.py",
    ),
]


# ---------------------------------------------------------------------------
# Weather catalog
# ---------------------------------------------------------------------------
# Interest scale:
#   1 = stable maritime summer, low variance, easy forecast
#   2 = mild transition season, moderate challenge
#   3 = active winter pattern, frontal passages
#   4 = extreme temperature snap (heat dome or arctic outbreak)
#   5 = subarctic deep freeze or coastal bomb cyclone — hardest regime
WEATHER_SCENARIOS: List[Scenario] = [
    Scenario(
        instrument="weather",
        name="Victoria Summer Doldrums",
        interest=1,
        description=(
            "Victoria BC (51337), July 2024 — stable maritime summer. "
            "Low variance, easy forecast. Baseline for MASE comparison."
        ),
        args=[
            "--mode", "benchmark", "--model", "persistence,seasonal_naive,toto-22m",
            "--station-id", "51337", "--year", "2024", "--months", "7",
            "--horizon", "24", "--n-splits", "5",
        ],
        script="weather_tsfm_engine_v2_production_hybrid_fixed.py",
    ),
    Scenario(
        instrument="weather",
        name="Vancouver Spring Transition",
        interest=2,
        description=(
            "Vancouver BC (51442), April 2024 — spring transition with mixed maritime "
            "air masses. Moderate forecast challenge, occasional frontal rain."
        ),
        args=[
            "--mode", "benchmark", "--model", "persistence,seasonal_naive,toto-22m",
            "--station-id", "51442", "--year", "2024", "--months", "4",
            "--horizon", "24", "--n-splits", "5",
        ],
        script="weather_tsfm_engine_v2_production_hybrid_fixed.py",
    ),
    Scenario(
        instrument="weather",
        name="Victoria Winter Storms",
        interest=3,
        description=(
            "Victoria BC (51337), January 2024 — active winter pattern with Pacific "
            "frontal passages. Higher variance, trickier wind/temperature coupling."
        ),
        args=[
            "--mode", "benchmark", "--model", "persistence,seasonal_naive,toto-22m",
            "--station-id", "51337", "--year", "2024", "--months", "1",
            "--horizon", "24", "--n-splits", "5",
        ],
        script="weather_tsfm_engine_v2_production_hybrid_fixed.py",
    ),
    Scenario(
        instrument="weather",
        name="Edmonton Arctic Outbreak",
        interest=4,
        description=(
            "Edmonton AB (51097), January 2024 — extreme cold snap, high temperature "
            "variance. Stress test for TSFM diurnal capture and persistence breakdown."
        ),
        args=[
            "--mode", "benchmark", "--model", "persistence,seasonal_naive,toto-22m",
            "--station-id", "51097", "--year", "2024", "--months", "1",
            "--horizon", "24", "--n-splits", "5",
        ],
        script="weather_tsfm_engine_v2_production_hybrid_fixed.py",
    ),
    Scenario(
        instrument="weather",
        name="Whitehorse Deep Freeze",
        interest=5,
        description=(
            "Whitehorse YT (16108), January 2024 — subarctic winter, massive diurnal "
            "swings and radiative cooling. Hardest forecast regime in the catalog."
        ),
        args=[
            "--mode", "benchmark", "--model", "persistence,seasonal_naive,toto-22m",
            "--station-id", "16108", "--year", "2024", "--months", "1",
            "--horizon", "24", "--n-splits", "5",
        ],
        script="weather_tsfm_engine_v2_production_hybrid_fixed.py",
    ),
]


# ---------------------------------------------------------------------------
# Magnetometer catalog
# ---------------------------------------------------------------------------
# Interest scale:
#   1 = geomagnetically quiet week, residuals < 5 nT RMS
#   2 = mildly elevated, minor deviations from QDC
#   3 = G2-G3 storm, clear storm signature in residuals
#   4 = G4 severe storm, major deviations, cross-check with Kp/Dst
#   5 = G5 extreme storm, auroral-zone chaos, largest event in 20 years
MAGNETOMETER_SCENARIOS: List[Scenario] = [
    Scenario(
        instrument="magnetometer",
        name="VIC Quiet Week",
        interest=1,
        description=(
            "Victoria observatory, June 2024 — geomagnetically quiet, minimal "
            "deviation from QDC baseline. Residual RMS ~2 nT expected."
        ),
        args=[
            "--fetch-real-data", "--observatory", "VIC", "--days", "7",
            "--start-date", "2024-06-15",
        ],
        script="magnetometer/demos/magnetometer_demo.py",
    ),
    Scenario(
        instrument="magnetometer",
        name="VIC Mild Activity",
        interest=2,
        description=(
            "Victoria observatory, late March 2024 — slightly elevated activity, "
            "minor deviations. Good test of robust baseline under weak disturbance."
        ),
        args=[
            "--fetch-real-data", "--observatory", "VIC", "--days", "7",
            "--start-date", "2024-03-20",
        ],
        script="magnetometer/demos/magnetometer_demo.py",
    ),
    Scenario(
        instrument="magnetometer",
        name="VIC August Storm",
        interest=3,
        description=(
            "Victoria observatory, August 12, 2024 — G3 strong storm. "
            "Clear storm signature in residuals, flag transitions visible."
        ),
        args=[
            "--fetch-real-data", "--observatory", "VIC", "--days", "3",
            "--start-date", "2024-08-11",
        ],
        script="magnetometer/demos/magnetometer_demo.py",
    ),
    Scenario(
        instrument="magnetometer",
        name="OTT Severe Storm",
        interest=4,
        description=(
            "Ottawa observatory, March 24, 2024 — G4 severe geomagnetic storm. "
            "Major deviations expected. Good test of 5-tier classification boundary."
        ),
        args=[
            "--fetch-real-data", "--observatory", "OTT", "--days", "3",
            "--start-date", "2024-03-23",
        ],
        script="magnetometer/demos/magnetometer_demo.py",
    ),
    Scenario(
        instrument="magnetometer",
        name="BRW Extreme G5",
        interest=5,
        description=(
            "Barrow, Alaska — May 10-12, 2024 G5 extreme storm. Most intense "
            "geomagnetic event in 20 years. Auroral-zone chaos, massive residuals."
        ),
        args=[
            "--fetch-real-data", "--observatory", "BRW", "--days", "3",
            "--start-date", "2024-05-10",
        ],
        script="magnetometer/demos/magnetometer_demo.py",
    ),
]


CATALOG = {
    "seismic": SEISMIC_SCENARIOS,
    "weather": WEATHER_SCENARIOS,
    "magnetometer": MAGNETOMETER_SCENARIOS,
}

INSTRUMENT_ORDER = ["magnetometer", "weather", "seismic"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(text: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def pick_scenario(
    scenarios: List[Scenario],
    min_interest: int = 1,
    max_interest: int = 5,
    weighted: bool = False,
    rng: Optional[random.Random] = None,
) -> Scenario:
    """Pick a scenario from the list subject to interest-level filters."""
    if rng is None:
        rng = random

    filtered = [s for s in scenarios if min_interest <= s.interest <= max_interest]
    if not filtered:
        raise ValueError(
            f"No scenarios match interest filter [{min_interest}-{max_interest}]"
        )

    if weighted:
        # Weight by interest level (higher = more likely)
        weights = [s.interest for s in filtered]
        total = sum(weights)
        pick = rng.choices(filtered, weights=weights, k=1)[0]
    else:
        pick = rng.choice(filtered)

    return pick


def run_scenario(scenario: Scenario, timeout: int = 600, dry_run: bool = False) -> bool:
    """Run a single scenario. Returns True on success."""
    script_path = Path(scenario.script)
    if not script_path.exists():
        print(f"[SKIP] {scenario.script} not found in current directory.")
        return False

    if scenario.instrument == "magnetometer":
        cmd = [sys.executable, "-m", "magnetometer.demos.magnetometer_demo"] + scenario.args
    else:
        cmd = [sys.executable, scenario.script] + scenario.args

    if dry_run:
        print(f"\n[DRY-RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return True

    print(f"\n▶ Running: {' '.join(cmd)}\n")
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {scenario.name} exceeded {timeout}s.")
        return False

    elapsed = time.time() - start

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"[STDERR] {result.stderr}")

    status = "✓ OK" if result.returncode == 0 else "✗ FAIL"
    print(f"\n{status}  {scenario.name} finished in {elapsed:.1f}s")
    return result.returncode == 0


def print_catalog() -> None:
    print("\nCascadia Sentinel — Scenario Catalog")
    print("=" * 70)
    for instrument in INSTRUMENT_ORDER:
        scenarios = CATALOG[instrument]
        print(f"\n{instrument.upper()}  ({len(scenarios)} scenarios)")
        print("-" * 70)
        for s in scenarios:
            stars = "★" * s.interest + "☆" * (5 - s.interest)
            print(f"  [{s.interest}/5] {stars}  {s.name}")
            print(f"           {s.description}")
            print(f"           → python {s.script} {' '.join(s.args)}")
    print("\n" + "=" * 70)


def print_selection(scenario: Scenario) -> None:
    stars = "★" * scenario.interest + "☆" * (5 - scenario.interest)
    print(f"\n  🎲  Random pick: {scenario.name}")
    print(f"      Interest: {scenario.interest}/5 {stars}")
    print(f"      {scenario.description}")
    print(f"      Command: python {scenario.script} {' '.join(scenario.args)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Cascadia Sentinel — Random real-data scenario picker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_random_demo.py --instrument all
  python run_random_demo.py --instrument seismic --min-interest 3
  python run_random_demo.py --instrument all --weighted --seed 42
  python run_random_demo.py --list
  python run_random_demo.py --instrument weather --dry-run
        """,
    )
    ap.add_argument(
        "--instrument",
        choices=["seismic", "weather", "magnetometer", "all"],
        default="all",
        help="Which instrument to run (default: all)",
    )
    ap.add_argument(
        "--min-interest",
        type=int,
        default=1,
        help="Minimum interest level (1-5) to include in the draw (default: 1)",
    )
    ap.add_argument(
        "--max-interest",
        type=int,
        default=5,
        help="Maximum interest level (1-5) to include in the draw (default: 5)",
    )
    ap.add_argument(
        "--weighted",
        action="store_true",
        help="Weight random selection by interest level (higher = more likely)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible selection",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-demo timeout in seconds (default: 600)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print the full scenario catalog and exit",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit the selected scenario(s) as JSON and exit (no execution)",
    )
    ap.add_argument(
        "--backends",
        type=str,
        default=None,
        help=(
            "Override the TSFM backends for weather scenarios "
            '(e.g. "persistence,toto-22m,chronos2-small")'
        ),
    )
    args = ap.parse_args()

    if args.list:
        print_catalog()
        return

    rng = random.Random(args.seed)

    instruments = INSTRUMENT_ORDER if args.instrument == "all" else [args.instrument]

    selections = []
    for inst in instruments:
        scenario = pick_scenario(
            CATALOG[inst],
            min_interest=args.min_interest,
            max_interest=args.max_interest,
            weighted=args.weighted,
            rng=rng,
        )
        selections.append(scenario)

    if args.json:
        print(json.dumps([s.to_dict() for s in selections], indent=2))
        return

    banner("Cascadia Sentinel — Random Demo Picker")
    for s in selections:
        print_selection(s)

    if args.dry_run:
        print("\n[Dry-run complete — no demos were executed]")
        return

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    results = {}
    for scenario in selections:
        banner(f"RUNNING: {scenario.name}")

        # Allow backend override for weather
        if args.backends and scenario.instrument == "weather":
            # Replace the --model argument
            new_args = []
            skip_next = False
            for i, a in enumerate(scenario.args):
                if skip_next:
                    skip_next = False
                    continue
                if a == "--model":
                    new_args.extend(["--model", args.backends])
                    skip_next = True
                else:
                    new_args.append(a)
            scenario.args[:] = new_args

        ok = run_scenario(scenario, timeout=args.timeout)
        results[scenario.instrument] = {"name": scenario.name, "ok": ok}

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    banner("SUMMARY")
    print(f"{'Instrument':<15} {'Scenario':<25} {'Status'}")
    print("-" * 60)
    for inst in instruments:
        r = results[inst]
        status = "OK" if r["ok"] else "FAIL"
        print(f"{inst:<15} {r['name']:<25} {status}")

    all_ok = all(r["ok"] for r in results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
