"""Small, dependency-light disk cache primitives for upstream HTTP data.

The cache is deliberately independent from the scientific pipeline.  It is
safe to use for repeat historical runs and parameter sweeps while keeping the
existing acquisition behavior unchanged during the modularization.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Optional


class DiskResponseCache:
    """Content-addressed text response cache with atomic writes and TTLs."""

    def __init__(self, directory: str | Path, ttl_seconds: float) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = max(0.0, float(ttl_seconds))

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.txt"

    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        try:
            age = time.time() - path.stat().st_mtime
            if age < 0 or age > self.ttl_seconds:
                return None
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None

    def put(self, key: str, value: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(key)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def clear(self) -> None:
        if not self.directory.exists():
            return
        for path in self.directory.glob("*.txt"):
            try:
                path.unlink()
            except OSError:
                pass
