"""Public magnetometer pipeline API.

This is the stable compatibility facade used by the CLI.  Implementation
responsibilities are progressively delegated to focused submodules while the
legacy public call surface remains available.
"""

from .legacy_core import *  # noqa: F401,F403
from .legacy_core import main_entry, run_cli, run_loop

# Override the numerical baseline functions with the extracted implementation.
# The legacy module still contains its historical copies for compatibility while
# callers through the public core now exercise the standalone baseline module.
from .baseline import build_design_matrix, handle_gaps, robust_harmonic_baseline

__all__ = [name for name in globals() if not name.startswith("_")]
