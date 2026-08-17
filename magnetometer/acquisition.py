"""Production upstream acquisition for the magnetometer pipeline.

This module owns network transport and historical-response caching. Scientific
processing remains elsewhere, so acquisition can be tested independently and
reused by the CLI, batch runner, and future live monitor.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cache import ResponseCache

logger = logging.getLogger("magnetometer_pipeline")

INTERMAGNET_BASE = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"
KP_GFZ_URL = "https://kp.gfz-potsdam.de/app/json/"
USER_AGENT = "MagnetometerProductionPipeline/2.0.0"
HTTP_CACHE_ENABLED = True
HTTP_CACHE_DIR = ".magnetometer_cache"
HTTP_CACHE_TTL_HOURS = 24.0


class AcquisitionError(RuntimeError):
    """Expected upstream-data failure."""


class AcquisitionClient:
    """Reusable, thread-safe acquisition client.

    A session is created once per client and uses urllib3 retries for transient
    upstream failures. Historical responses use the same cache-key and file
    layout as the previous monolithic implementation.
    """

    __slots__ = ("session", "cache", "_dst_unavailable", "_dst_lock")

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        cache_enabled: bool = HTTP_CACHE_ENABLED,
        cache_dir: str = HTTP_CACHE_DIR,
        cache_ttl_hours: float = HTTP_CACHE_TTL_HOURS,
    ) -> None:
        self.session = session or create_resilient_session()
        self.cache = (
            ResponseCache(cache_dir, cache_ttl_hours) if cache_enabled else None
        )
        self._dst_unavailable: set[Tuple[int, int]] = set()
        self._dst_lock = threading.Lock()

    def get_text(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: float = 30.0,
        cacheable: bool = True,
    ) -> Tuple[int, str]:
        key = ResponseCache.key(url, params)
        if cacheable and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                logger.debug("Cache hit for %s", url)
                return hit

        response = self.session.get(url, params=params, timeout=timeout)
        result = (response.status_code, response.text)
        if cacheable and response.status_code == 200 and self.cache is not None:
            self.cache.put(key, *result)
        return result

    def fetch_station(
        self,
        observatory: str = "VIC",
        start_date: Optional[str] = None,
        duration_days: int = 7,
        samples_per_day: str = "Minute",
    ) -> str:
        if start_date is None:
            start_date = "2024-01-01"
        params = {
            "Request": "GetData",
            "observatoryIagaCode": observatory,
            "samplesPerDay": samples_per_day,
            "dataStartDate": start_date,
            "dataDuration": duration_days,
            "format": "iaga2002",
            "orientation": "XYZF",
        }
        logger.info(
            "Fetching INTERMAGNET data for %s from %s (%s days)...",
            observatory,
            start_date,
            duration_days,
        )
        end_date = pd.to_datetime(start_date, utc=True) + pd.Timedelta(days=duration_days)
        status, text = self.get_text(
            INTERMAGNET_BASE,
            params=params,
            timeout=60,
            cacheable=_window_is_historical(end_date),
        )
        if status >= 400:
            raise requests.HTTPError(
                f"INTERMAGNET returned HTTP {status} for {observatory}"
            )
        return text

    def fetch_kp(self, start_date: str, end_date: str) -> pd.Series:
        url = (
            f"{KP_GFZ_URL}?start={start_date}T00:00:00Z"
            f"&end={end_date}T23:59:59Z&index=Kp"
        )
        cacheable = _window_is_historical(end_date)
        if not (cacheable and self.cache is not None and self.cache.contains(ResponseCache.key(url))):
            logger.info("Fetching Kp index from GFZ Potsdam...")
        status, text = self.get_text(url, timeout=30, cacheable=cacheable)
        if status >= 400:
            raise requests.HTTPError(f"Kp service returned HTTP {status}")
        data = json.loads(text)
        series = pd.Series(
            pd.array(data["Kp"], dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(data["datetime"], utc=True)),
            name="kp",
        )
        series.index.name = "datetime"
        return series.sort_index()

    def fetch_dst(self, year: int, month: int) -> Optional[pd.Series]:
        with self._dst_lock:
            if (year, month) in self._dst_unavailable:
                return None

        yy, mm = year % 100, month
        urls = [
            f"https://wdc.kugi.kyoto-u.ac.jp/dst_final/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for",
            f"https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for",
            f"https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{year:04d}{mm:02d}/dst{yy:02d}{mm:02d}.for",
        ]
        cacheable = _window_is_historical(
            pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            + pd.DateOffset(months=1)
        )

        def try_url(url: str) -> Optional[str]:
            try:
                status, body = self.get_text(url, timeout=15, cacheable=cacheable)
            except requests.RequestException:
                return None
            if status == 200 and "Not Found" not in body and "<html" not in body.lower():
                return body
            return None

        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            candidates = list(pool.map(try_url, urls))
        text = next((body for body in candidates if body is not None), None)
        if text is None:
            with self._dst_lock:
                self._dst_unavailable.add((year, month))
            logger.warning(
                "Dst index unavailable for %04d-%02d from Kyoto WDC "
                "(server down or restricted). Skipping Dst.",
                year,
                month,
            )
            return None

        rows = []
        for line in text.splitlines():
            if len(line) < 116 or not (
                line[:3].strip().isdigit() or line[3:5].strip().isdigit()
            ):
                continue
            try:
                day = int(line[8:10].strip())
                hourly_part = line[20:116]
                for hour in range(24):
                    value = hourly_part[hour * 4 : (hour + 1) * 4].strip()
                    if value and value != "9999":
                        rows.append(
                            {
                                "datetime": datetime(
                                    year, month, day, hour, tzinfo=timezone.utc
                                ),
                                "dst": int(value),
                            }
                        )
            except (ValueError, IndexError):
                continue
        if not rows:
            return None
        return pd.DataFrame(rows).set_index("datetime")["dst"].sort_index()


def create_resilient_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset({"GET", "HEAD", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _window_is_historical(end_date: Optional[Any], min_age_days: float = 2.0) -> bool:
    if end_date is None:
        return False
    try:
        end = pd.to_datetime(end_date, utc=True)
    except Exception:
        return False
    return end < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=min_age_days)


DEFAULT_ACQUISITION = AcquisitionClient()

# Compatibility exports. The runner can migrate one function at a time without
# changing its public call surface.
fetch_intermagnet_iaga2002 = DEFAULT_ACQUISITION.fetch_station
fetch_kp_gfz = DEFAULT_ACQUISITION.fetch_kp
fetch_dst_kyoto = DEFAULT_ACQUISITION.fetch_dst
http_get_text = DEFAULT_ACQUISITION.get_text

__all__ = [
    "AcquisitionClient",
    "AcquisitionError",
    "DEFAULT_ACQUISITION",
    "create_resilient_session",
    "fetch_intermagnet_iaga2002",
    "fetch_kp_gfz",
    "fetch_dst_kyoto",
    "http_get_text",
]
