"""Canonical detector release-gate entry point."""
from pathlib import Path

from .. import production_grade_validation as pg

pg.DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "case_cache_causal_v2"

from ..production_release_gate_v3 import main  # noqa: E402

if __name__ == "__main__":
    main()
