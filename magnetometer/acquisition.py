"""Upstream data acquisition boundary for the magnetometer pipeline.

The public acquisition API is intentionally narrow: station data, Kp, Dst,
and resilient HTTP transport.  During this behavior-preserving migration the
existing implementations remain in ``legacy_core``; callers should import
through this module so the implementation can be moved without changing
application code.
"""

from __future__ import annotations

from typing import Any

from .legacy_core import (
    create_resilient_session,
    fetch_dst_kyoto,
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    http_get_text,
)


class AcquisitionClient:
    """Stable acquisition facade used by higher-level pipeline code.

    Methods deliberately delegate to the validated implementation for now.
    This keeps numerical behavior and network semantics identical while giving
    the rest of the package a dependency-injection seam for the next extraction
    step and for offline/replay tests.
    """

    __slots__ = ()

    def fetch_station(self, *args: Any, **kwargs: Any) -> Any:
        return fetch_intermagnet_iaga2002(*args, **kwargs)

    def fetch_kp(self, *args: Any, **kwargs: Any) -> Any:
        return fetch_kp_gfz(*args, **kwargs)

    def fetch_dst(self, *args: Any, **kwargs: Any) -> Any:
        return fetch_dst_kyoto(*args, **kwargs)

    def session(self, *args: Any, **kwargs: Any) -> Any:
        return create_resilient_session(*args, **kwargs)

    def get_text(self, *args: Any, **kwargs: Any) -> Any:
        return http_get_text(*args, **kwargs)


DEFAULT_ACQUISITION = AcquisitionClient()

__all__ = [
    "AcquisitionClient",
    "DEFAULT_ACQUISITION",
    "fetch_intermagnet_iaga2002",
    "fetch_kp_gfz",
    "fetch_dst_kyoto",
    "create_resilient_session",
    "http_get_text",
]
