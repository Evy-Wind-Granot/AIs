"""Strict config loading for the magnetometer pipeline.

Production validation must fail closed when a config contains unrecognized
keys. The historical calibration path also wrote flat FLAG_* keys; those are
still accepted and mapped onto the same module constants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import legacy_core as _legacy


def _nested_from_flat(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate flat FLAG_* / constant names into the nested schema."""
    nested: Dict[str, Any] = {}
    for key, const_name in _legacy._CONFIG_MAPPING.items():
        if const_name in cfg and cfg[const_name] is not None:
            section, _, leaf = key.partition(".")
            nested.setdefault(section, {})[leaf] = cfg[const_name]
    return nested


def _merge_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer nested sections; fold any flat constant names into them."""
    merged: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in _legacy._SETTING_TYPES:
            continue  # handled via flat translation
        merged[k] = v
    flat_nested = _nested_from_flat(cfg)
    for section, body in flat_nested.items():
        if section in merged and isinstance(merged[section], dict):
            merged[section] = {**body, **merged[section]}
        else:
            merged[section] = body
    return merged


def _reject_unknown(cfg: Dict[str, Any]) -> None:
    known_sections: Dict[str, set] = {}
    for dotted in _legacy._CONFIG_MAPPING:
        section, _, leaf = dotted.partition(".")
        known_sections.setdefault(section, set()).add(leaf)
    known_flat = set(_legacy._SETTING_TYPES.keys())

    problems: List[str] = []
    for section, body in cfg.items():
        if section in known_flat:
            continue
        if section not in known_sections:
            problems.append(f"unknown config section '{section}'")
            continue
        if isinstance(body, dict):
            for leaf in body:
                if leaf not in known_sections[section]:
                    problems.append(f"unknown config key '{section}.{leaf}'")
    if problems:
        raise _legacy.PipelineError(
            "Invalid configuration (unrecognized keys — refusing to run with a "
            "partially recognized config):\n  - "
            + "\n  - ".join(problems),
            _legacy.EXIT_CONFIG_INVALID,
        )


def load_config(path: str) -> Dict[str, Any]:
    """Strict load: unknown keys are fatal; flat FLAG_* keys are accepted."""
    p = Path(path)
    if not p.exists():
        raise _legacy.PipelineError(f"Config file not found: {path}", _legacy.EXIT_CONFIG_INVALID)

    text = p.read_text()
    try:
        if _legacy._HAS_YAML:
            cfg = _legacy.yaml.safe_load(text)
        else:
            import json

            cfg = json.loads(text)
    except Exception as e:
        raise _legacy.PipelineError(
            f"Config file {path} is not valid: {e}", _legacy.EXIT_CONFIG_INVALID
        )

    if not isinstance(cfg, dict):
        raise _legacy.PipelineError(
            f"Config file {path} must contain a mapping at the top level",
            _legacy.EXIT_CONFIG_INVALID,
        )

    _reject_unknown(cfg)
    merged = _merge_cfg(cfg)

    # Reuse the nested flattening + validation + global application path.
    overrides = _legacy._flatten_config(merged)
    # Also apply any remaining flat keys that map directly to settings.
    for name in _legacy._SETTING_TYPES:
        if name in cfg and cfg[name] is not None:
            overrides[name] = cfg[name]

    _legacy.validate_settings({**{k: getattr(_legacy, k) for k in _legacy._SETTING_TYPES}, **overrides})
    _legacy.globals().update(overrides) if False else None
    # Apply onto the legacy module namespace (callers read these as module attrs).
    for k, v in overrides.items():
        setattr(_legacy, k, v)
    _legacy.logger.debug(f"Applied {len(overrides)} setting(s) from {path} (strict)")
    return merged


__all__ = ["load_config"]
