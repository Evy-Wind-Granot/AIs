"""HTTP response cache used by the magnetometer acquisition layer.

The cache keeps the historical ``.magnetometer_cache/*.json`` layout so
existing cached data remains reusable across the modularization.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Tuple


class ResponseCache:
    """Thread-safe memory + disk cache for HTTP text responses."""

    __slots__ = ("directory", "ttl_seconds", "_memory", "_lock")

    def __init__(self, directory: str | Path, ttl_hours: float) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = max(0.0, float(ttl_hours) * 3600.0)
        self._memory: Dict[str, Tuple[int, str]] = {}
        self._lock = RLock()

    @staticmethod
    def key(url: str, params: Optional[Dict[str, Any]] = None) -> str:
        payload = json.dumps(
            [url, sorted((params or {}).items(), key=str)], default=str
        )
        # Keep the exact historical key format for backwards-compatible cache hits.
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Tuple[int, str]]:
        with self._lock:
            hit = self._memory.get(key)
        if hit is not None:
            return hit

        path = self.directory / f"{key}.json"
        try:
            age = time.time() - path.stat().st_mtime
            if age < 0 or age > self.ttl_seconds:
                return None
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            value = (int(data["status"]), str(data["text"]))
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            return None

        with self._lock:
            self._memory[key] = value
        return value

    def put(self, key: str, status: int, text: str) -> None:
        value = (int(status), text)
        with self._lock:
            self._memory[key] = value

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / f"{key}.json"
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=self.directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump({"status": status, "text": text, "url_key": key}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, target)
            finally:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
        except OSError:
            # Cache failures must never make scientific processing fail.
            return

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()


__all__ = ["ResponseCache"]
