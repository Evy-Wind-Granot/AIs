"""Parsing utilities for magnetometer upstream data.

This module converts IAGA-2002 text payloads into the normalized DataFrame
schema consumed by the scientific pipeline.  It deliberately contains no
network access or scientific classification logic.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("magnetometer_pipeline")


def parse_iaga2002_to_dataframe(text: str) -> pd.DataFrame:
    """Parse an IAGA-2002 payload into a UTC-indexed DataFrame.

    The data block is parsed in bulk (single whitespace-delimited read plus
    vectorized datetime/numeric conversion) rather than row by row, which
    dominates runtime on multi-day minute-cadence fetches.

    The returned schema is stable and intentionally small:
    ``x_nt``, ``y_nt``, ``z_nt``, and ``f_nt`` with a UTC ``datetime`` index.
    Missing IAGA sentinel values are normalized to NaN.
    """
    if text.strip().startswith(("<", "<!DOCTYPE", "<html")):
        raise ValueError("INTERMAGNET returned HTML instead of IAGA-2002 data.")

    lines = text.splitlines()
    data_lines = []
    col_names = None

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue
        if line_s.startswith("DATE"):
            col_names = line_s.replace("|", "").split()
        elif line_s[0].isdigit():
            data_lines.append(line_s)

    if not col_names or len(col_names) < 7:
        raise ValueError("Could not parse IAGA-2002 headers.")

    def find_col(key: str) -> Optional[int]:
        for i, name in enumerate(col_names):
            if name.upper().endswith(key.upper()) and len(name) == 4:
                return i
        return None

    n_cols = len(col_names)
    # Truncated/short records are dropped, matching the previous parser.
    data_lines = [line for line in data_lines if line.count(" ") + 1 >= n_cols]
    empty = pd.DataFrame(
        {c: pd.Series(dtype=float) for c in ("x_nt", "y_nt", "z_nt", "f_nt")},
        index=pd.DatetimeIndex([], tz="UTC", name="datetime"),
    )
    if not data_lines:
        return empty

    raw = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        header=None,
        dtype=str,
        names=range(n_cols),
        index_col=False,
        engine="c",
    )
    raw = raw[raw[n_cols - 1].notna()]
    if raw.empty:
        return empty

    stamps = pd.to_datetime(raw[0] + " " + raw[1], utc=True, errors="coerce")
    if stamps.isna().any():
        n_bad = int(stamps.isna().sum())
        logger.warning(
            "Dropped %d IAGA-2002 records with unparseable timestamps.", n_bad
        )
        keep = stamps.notna().to_numpy()
        raw, stamps = raw[keep], stamps[keep]
        if raw.empty:
            return empty

    def column_values(idx: Optional[int]) -> np.ndarray:
        if idx is None or idx >= n_cols:
            return np.full(len(raw), np.nan)
        vals = pd.to_numeric(raw[idx], errors="coerce").to_numpy(dtype=float)
        # IAGA-2002 uses 99999.0 / 88888.0 sentinels for missing values.
        vals[np.abs(vals) >= 99999] = np.nan
        return vals

    df = pd.DataFrame(
        {
            "x_nt": column_values(find_col("X")),
            "y_nt": column_values(find_col("Y")),
            "z_nt": column_values(find_col("Z")),
            "f_nt": column_values(find_col("F")),
        },
        index=pd.DatetimeIndex(stamps.to_numpy(), tz="UTC", name="datetime"),
    )
    return df.sort_index()


__all__ = ["parse_iaga2002_to_dataframe"]
