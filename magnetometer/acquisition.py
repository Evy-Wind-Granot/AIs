"""Data acquisition API for the magnetometer pipeline.

These names form the stable seam for INTERMAGNET and global-index I/O.  The
implementation currently delegates to the compatibility implementation so the
refactor is behavior-preserving; the internals can be moved here incrementally.
"""
from .legacy_core import (
    fetch_intermagnet_iaga2002,
    fetch_kp_gfz,
    fetch_dst_kyoto,
    create_resilient_session,
    http_get_text,
)

__all__ = [
    "fetch_intermagnet_iaga2002",
    "fetch_kp_gfz",
    "fetch_dst_kyoto",
    "create_resilient_session",
    "http_get_text",
]
