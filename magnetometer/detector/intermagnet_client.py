#!/usr/bin/env python3
"""Robust INTERMAGNET client for long detector benchmarks.

The public INTERMAGNET endpoint is reliable for short requests but can return
HTTP 400 responses or truncated chunked HTTP bodies for large date ranges.
Production evaluation therefore downloads bounded chunks, retries transient
transport failures, validates each response, and only then combines the data.
"""
from __future__ import annotations

import time
from datetime import timedelta
from typing import List

import requests

from magnetometer_demo import (
    DEFAULT_SAMPLES_PER_DAY,
    HTTP_CLIENT,
    INTERMAGNET_BASE,
    logger,
)


DEFAULT_CHUNK_DAYS = 7
DEFAULT_RETRIES = 4


def _request_chunk(
    observatory: str,
    start_date: str,
    duration_days: int,
    samples_per_day: str,
    retries: int,
) -> str:
    params = {
        "Request": "GetData",
        "observatoryIagaCode": observatory,
        "samplesPerDay": samples_per_day,
        "dataStartDate": start_date,
        "dataDuration": duration_days,
        "format": "iaga2002",
        "orientation": "XYZF",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = HTTP_CLIENT.get(INTERMAGNET_BASE, params=params, timeout=90)
            response.raise_for_status()
            text = response.text
            if not text.strip():
                raise ValueError("INTERMAGNET returned an empty response")
            if text.lstrip().startswith("<"):
                raise ValueError("INTERMAGNET returned HTML instead of IAGA-2002 data")
            return text
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = min(30.0, 2.0 ** (attempt - 1))
            logger.warning(
                "INTERMAGNET chunk %s (%dd) attempt %d/%d failed: %s; retrying in %.1fs",
                start_date,
                duration_days,
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"INTERMAGNET failed after {retries} attempts for {observatory} "
        f"starting {start_date} ({duration_days} days): {last_error}"
    ) from last_error


def fetch_intermagnet_long_range(
    observatory: str,
    start_date: str,
    duration_days: int,
    samples_per_day: str = DEFAULT_SAMPLES_PER_DAY,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """Fetch a long INTERMAGNET interval as bounded, retryable chunks.

    The returned value is valid for the existing IAGA-2002 parser; repeated
    headers from individual chunks are harmless because the parser ignores
    comment/header lines except for the DATE header.
    """
    if duration_days < 1:
        raise ValueError("duration_days must be >= 1")
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if retries < 1:
        raise ValueError("retries must be >= 1")

    current = __import__("datetime").date.fromisoformat(start_date)
    remaining = duration_days
    chunks: List[str] = []
    chunk_number = 0
    total_chunks = (duration_days + chunk_days - 1) // chunk_days

    logger.info(
        "Fetching INTERMAGNET long range: %s %s for %d days in %d-day chunks (%d chunks)",
        observatory,
        start_date,
        duration_days,
        chunk_days,
        total_chunks,
    )

    while remaining > 0:
        days = min(chunk_days, remaining)
        chunk_number += 1
        chunk_start = current.isoformat()
        logger.info(
            "INTERMAGNET chunk %d/%d: %s (%d days)",
            chunk_number,
            total_chunks,
            chunk_start,
            days,
        )
        chunks.append(
            _request_chunk(
                observatory=observatory,
                start_date=chunk_start,
                duration_days=days,
                samples_per_day=samples_per_day,
                retries=retries,
            )
        )
        current += timedelta(days=days)
        remaining -= days

    return "\n".join(chunks)
