#!/usr/bin/env python3
"""Small, defensive runtime-state helpers for the hybrid monitor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_state(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_state(path: str | Path, state: Dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    temp_path.replace(file_path)


def merge_forecast_state(path: str | Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Merge forecast fields without deleting unrelated legacy state keys."""
    state = load_state(path)
    state["forecast"] = payload.get("forecast")
    state["hybrid"] = payload.get("hybrid")
    state["forecast_model"] = payload.get("model")
    state["last_forecast_generated_at"] = payload.get("generated_at")
    save_state(path, state)
    return state
