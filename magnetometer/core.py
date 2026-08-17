"""Public magnetometer pipeline API.

The implementation is intentionally kept behind a small, stable facade.  This
module is the compatibility boundary used by ``magnetometer_demo.py`` and
``run_magnetometer.py``.  The implementation is being migrated into focused
submodules without changing the public call surface.
"""

from .legacy_core import *  # noqa: F401,F403
from .legacy_core import main_entry, run_cli, run_loop

__all__ = [name for name in globals() if not name.startswith("_")]
