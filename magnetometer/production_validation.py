#!/usr/bin/env python3
"""Compatibility entry point for the production magnetometer validation gate.

Use production_grade_validation.py directly for the current release-gate
implementation. This wrapper preserves the historical command path while
preventing multiple validators with different standards from drifting apart.
"""

from __future__ import annotations

from production_grade_validation import main


if __name__ == "__main__":
    main()
