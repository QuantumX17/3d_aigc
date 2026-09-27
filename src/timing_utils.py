"""Small, dependency-free helpers for per-sample pipeline timing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


STAGE_FIELDS = (
    "image_generation_seconds",
    "bbox_inference_seconds",
    "mesh_generation_seconds",
    "finalize_seconds",
)


def read_timing(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def recompute_total(data: dict[str, Any]) -> dict[str, Any]:
    total = 0.0
    for field in STAGE_FIELDS:
        value = data.get(field, 0.0)
        try:
            total += float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            pass
    data["total_seconds"] = round(total, 3)
    data["total_minutes"] = round(total / 60.0, 3)
    return data


def update_timing(path: str | Path, **values: float) -> dict[str, Any]:
    """Incrementally and atomically update one sample's timing.json."""
    path = Path(path)
    data = read_timing(path)
    for key, value in values.items():
        if key not in STAGE_FIELDS:
            raise ValueError(f"Unsupported timing field: {key}")
        data[key] = round(float(value), 3)
    for field in STAGE_FIELDS:
        data.setdefault(field, 0.0)
    recompute_total(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)
    return data
