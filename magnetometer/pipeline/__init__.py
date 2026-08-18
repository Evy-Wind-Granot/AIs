"""Modular magnetometer pipeline package.

Imports are intentionally lazy so low-level modules such as ``state`` can import
``pipeline.settings`` without creating package-level circular imports.
"""

from .config import load_config, validate_settings


def run_analysis(*args, **kwargs):
    from .analysis import run_analysis as impl
    return impl(*args, **kwargs)


def write_json_output(*args, **kwargs):
    from .analysis import write_json_output as impl
    return impl(*args, **kwargs)


def main(*args, **kwargs):
    from .cli import main as impl
    return impl(*args, **kwargs)


def run_cli(*args, **kwargs):
    from .cli import run_cli as impl
    return impl(*args, **kwargs)


def run_loop(*args, **kwargs):
    from .cli import run_loop as impl
    return impl(*args, **kwargs)


def main_entry(*args, **kwargs):
    from .cli import main_entry as impl
    return impl(*args, **kwargs)

__all__ = [
    "load_config",
    "validate_settings",
    "run_analysis",
    "write_json_output",
    "main",
    "run_cli",
    "run_loop",
    "main_entry",
]
