"""Strict pipeline configuration loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from . import settings


def _flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for dotted, setting_name in settings.CONFIG_MAPPING.items():
        value: Any = cfg
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            flat[setting_name] = value
    return flat


def validate_settings(values: Dict[str, Any]) -> None:
    errors = []
    for name, types in settings.SETTING_TYPES.items():
        if name not in values:
            continue
        value = values[name]
        if bool not in types and isinstance(value, bool):
            errors.append(f"{name} must be numeric, got boolean")
        elif not isinstance(value, types):
            errors.append(f"{name} must be {types}, got {type(value).__name__}")

    def number(name: str):
        value = values.get(name)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    tiers = [number(name) for name in (
        "FLAG_THRESHOLD_UNSETTLED_NT", "FLAG_THRESHOLD_ACTIVE_NT",
        "FLAG_THRESHOLD_MINOR_STORM_NT", "FLAG_THRESHOLD_MAJOR_STORM_NT",
        "FLAG_THRESHOLD_SEVERE_STORM_NT",
    )]
    if all(value is not None for value in tiers):
        if any(a >= b for a, b in zip(tiers, tiers[1:])):
            errors.append("activity thresholds must increase strictly")
        if tiers[0] <= 0:
            errors.append("FLAG_THRESHOLD_UNSETTLED_NT must be positive")

    mode = values.get("FLAG_AMPLITUDE_MODE")
    if isinstance(mode, str) and mode not in settings.VALID_AMPLITUDE_MODES:
        errors.append(f"FLAG_AMPLITUDE_MODE must be one of {settings.VALID_AMPLITUDE_MODES}")

    for name in ("MIN_ANALYSIS_COVERAGE", "MAX_MEDIAN_FILL_FRACTION", "MIN_REQUESTED_COVERAGE", "INPUT_NAN_WARNING_THRESHOLD"):
        value = number(name)
        if value is not None and not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be between 0 and 1")

    lower = number("MIN_PLAUSIBLE_RESIDUAL_NT")
    upper = number("MAX_PLAUSIBLE_RESIDUAL_NT")
    if lower is not None and upper is not None and lower >= upper:
        errors.append("MIN_PLAUSIBLE_RESIDUAL_NT must be below MAX_PLAUSIBLE_RESIDUAL_NT")

    level = values.get("ALERT_WEBHOOK_MIN_LEVEL")
    if isinstance(level, str) and level not in settings.STORM_LEVEL_ORDER:
        errors.append(f"ALERT_WEBHOOK_MIN_LEVEL must be one of {settings.STORM_LEVEL_ORDER}")

    if errors:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Config file not found: {path}")
    try:
        cfg = yaml.safe_load(p.read_text())
    except Exception as exc:
        raise ValueError(f"Config file {path} is not valid: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")

    known_sections: Dict[str, set[str]] = {}
    for dotted in settings.CONFIG_MAPPING:
        section, _, leaf = dotted.partition(".")
        known_sections.setdefault(section, set()).add(leaf)
    unknown = []
    for section, body in cfg.items():
        if section not in known_sections:
            unknown.append(f"unknown config section '{section}'")
        elif isinstance(body, dict):
            unknown.extend(
                f"unknown config key '{section}.{leaf}'"
                for leaf in body
                if leaf not in known_sections[section]
            )
    if unknown:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(unknown))

    values = _flatten_config(cfg)
    current = {name: getattr(settings, name) for name in settings.SETTING_TYPES}
    current.update(values)
    validate_settings(current)
    for name, value in values.items():
        setattr(settings, name, value)
    return cfg
