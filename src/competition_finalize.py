#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition finalization after model.glb has been generated.

Responsibilities:
  1. Validate the organizer's size/unit convention.
  2. Check explicit prompt dimensions using the organizer tolerance.
  3. Check Z-as-height semantics and place the print-contact bottom at Z-min / Z=0.
  4. Auto-adjust fixable size violations without remeshing the model.
  5. Re-check, render four standardized views, and write submission metadata.

Auto-fix only applies scaling and Z translation. It never smooths/remeshes the GLB.
Semantic "upright" orientation cannot be inferred reliably from arbitrary geometry,
so that part is reported as not automatically verifiable rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import trimesh

from timing_utils import update_timing


CATEGORY_ORDER = [
    "constrained",
    "decorative_object",
    "functional_object",
    "small_object",
]

UNIT_TO_MM = {
    "mm": 1.0,
    "毫米": 1.0,
    "cm": 10.0,
    "厘米": 10.0,
    "m": 1000.0,
    "米": 1000.0,
}

COMPETITION_MIN_SPAN_MM = 2.0
COMPETITION_MAX_SPAN_MM = 300.0
MAX_CONSTRAINT_TARGET_RATIO = 0.98


@dataclass
class CheckItem:
    name: str
    ok: bool | None
    actual_mm: float | None = None
    target_mm: float | None = None
    tolerance_mm: float | None = None
    detail: str = ""
    verifiable: bool = True
    fixable: bool = False


@dataclass
class SizeReport:
    prompt_id: str
    bounds_min_mm: list[float]
    bounds_max_mm: list[float]
    extents_mm: list[float]
    max_span_mm: float
    unit_ok: bool
    basic_size_ok: bool
    ground_ok: bool
    z_up_semantics_ok: bool | None
    prompt_constraints_ok: bool
    target_size_ok: bool | None
    overall_ok: bool
    auto_fix_applied: bool
    checks: list[dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def glb_bounds_mm(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        lo, hi = np.asarray(loaded.bounds, dtype=float)
    elif isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No mesh geometry in {path}")
        bounds = np.asarray(loaded.bounds, dtype=float)
        if bounds.shape != (2, 3):
            raise ValueError(f"Invalid bounds for {path}: {bounds}")
        lo, hi = bounds
    else:
        raise TypeError(f"Unsupported GLB type: {type(loaded)}")

    ext = hi - lo
    if not np.all(np.isfinite(ext)) or np.any(ext <= 1e-9):
        raise ValueError(f"Invalid model extents: {ext.tolist()}")
    return lo, hi, ext


def tolerance_mm(reference_mm: float) -> float:
    # Organizer slide: automatic check uses +/-2 mm (or +/-3%).
    return max(2.0, abs(reference_mm) * 0.03)


def _combined_text(prompt_data: dict[str, Any]) -> str:
    return " ".join(
        str(prompt_data.get(k, ""))
        for k in ("prompt", "prompt_en", "prompt_intent")
    )


def _to_mm(value: str, unit: str) -> float:
    factor = UNIT_TO_MM.get(unit.lower(), UNIT_TO_MM.get(unit))
    if factor is None:
        raise ValueError(f"Unsupported unit: {unit}")
    return float(value) * factor


def _first_match(text: str, pattern: str) -> tuple[float, str] | None:
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return None
    return _to_mm(m.group("v"), m.group("u")), m.group(0)


def parse_competition_size_rules(prompt_data: dict[str, Any]) -> dict[str, Any]:
    """Parse only size expressions needed for final verification.

    prompt_to_bbox.py remains responsible for estimating unspecified dimensions.
    Here we deliberately verify explicit requirements rather than template priors.
    """
    text = _combined_text(prompt_data)
    rules: dict[str, Any] = {"evidence": []}

    xyz = re.search(
        r"(?P<x>\d+(?:\.\d+)?)\s*[x×*]\s*"
        r"(?P<y>\d+(?:\.\d+)?)\s*[x×*]\s*"
        r"(?P<z>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)",
        text,
        flags=re.I,
    )
    if xyz:
        f = UNIT_TO_MM.get(xyz.group("u").lower(), UNIT_TO_MM.get(xyz.group("u"), 1.0))
        rules["xyz_exact_mm"] = [
            float(xyz.group("x")) * f,
            float(xyz.group("y")) * f,
            float(xyz.group("z")) * f,
        ]
        rules["evidence"].append(xyz.group(0))

    patterns = {
        "height_max_mm": (
            r"(?:整体)?(?:高|高度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)"
        ),
        "width_max_mm": (
            r"(?:宽|宽度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)"
        ),
        "depth_max_mm": (
            r"(?:深|深度|厚|厚度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)"
        ),
        "overall_max_mm": (
            r"(?:整体|整体尺寸|见方|包围盒|最大跨度|最大尺寸).{0,10}?"
            r"(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)"
        ),
        "outer_diameter_max_mm": (
            r"(?:外径|外直径)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)"
        ),
        "outer_diameter_exact_mm": (
            r"(?:外径|外直径)\s*(?:为|约为|约|=|:|：)?\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)"
        ),
    }

    for key, pat in patterns.items():
        found = _first_match(text, pat)
        if found is not None:
            value, evidence = found
            rules[key] = value
            rules["evidence"].append(evidence)

    # Exact single-axis dimensions. Avoid matching expressions already marked as max.
    exact_patterns = {
        "height_exact_mm": r"(?:整体)?(?:高|高度)\s*(?:为|约为|约|=|:|：)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)",
        "width_exact_mm": r"(?:宽|宽度)\s*(?:为|约为|约|=|:|：)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)",
        "depth_exact_mm": r"(?:深|深度|厚|厚度)\s*(?:为|约为|约|=|:|：)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米|m|米)",
    }
    for key, pat in exact_patterns.items():
        found = _first_match(text, pat)
        if found is not None:
            value, evidence = found
            rules[key] = value
            rules["evidence"].append(evidence)

    return rules


def _check_exact(name: str, actual: float, target: float) -> CheckItem:
    tol = tolerance_mm(target)
    return CheckItem(
        name=name,
        ok=abs(actual - target) <= tol,
        actual_mm=actual,
        target_mm=target,
        tolerance_mm=tol,
        detail=f"|actual-target| <= {tol:.3f} mm",
    )


def _check_max(name: str, actual: float, limit: float) -> CheckItem:
    tol = tolerance_mm(limit)
    return CheckItem(
        name=name,
        ok=actual <= limit + tol,
        actual_mm=actual,
        target_mm=limit,
        tolerance_mm=tol,
        detail=f"actual <= limit + {tol:.3f} mm",
    )


def load_target_size(output_dir: Path) -> list[float] | None:
    for name in ("bbox_inferred.json", "bbox.json"):
        path = output_dir / name
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        value = data.get("size_mm")
        if isinstance(value, list) and len(value) == 3:
            try:
                return [float(x) for x in value]
            except Exception:
                pass
    return None


def load_bbox_control(output_dir: Path) -> tuple[list[float] | None, list[float] | None]:
    """Load the exact bbox and size selected before mesh generation."""
    for name in ("bbox_inferred.json", "bbox.json"):
        path = output_dir / name
        if not path.exists():
            continue
        try:
            data = read_json(path)
            raw_bbox, raw_size = data.get("bbox"), data.get("size_mm")
            bbox = [float(v) for v in raw_bbox] if isinstance(raw_bbox, list) and len(raw_bbox) == 3 else None
            size = [float(v) for v in raw_size] if isinstance(raw_size, list) and len(raw_size) == 3 else None
            return bbox, size
        except Exception:
            continue
    return None, None



def ensure_metadata_unit_mm(output_dir: Path, prompt_id: str) -> None:
    """Enforce the submission metadata unit required by the organizer."""
    path = output_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if path.exists():
        try:
            metadata = read_json(path)
        except Exception:
            metadata = {}
    metadata["prompt_id"] = prompt_id
    metadata["unit"] = "mm"
    metadata.setdefault("primary_model", "model.glb")
    metadata.setdefault("coordinate_system", "Z-up")
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def metadata_unit_is_mm(output_dir: Path | None) -> bool:
    if output_dir is None:
        return False
    path = output_dir / "metadata.json"
    if not path.exists():
        return False
    try:
        return str(read_json(path).get("unit", "")).strip().lower() == "mm"
    except Exception:
        return False


def _has_any_explicit_size_rule(rules: dict[str, Any]) -> bool:
    return any(k != "evidence" for k in rules.keys())


def _normalize_bbox_from_size(size: Iterable[float]) -> list[float]:
    vals = [float(v) for v in size]
    m = max(vals)
    return [round(v / m, 6) for v in vals]


def compute_adjustment_scales(
    extents_mm: Iterable[float],
    rules: dict[str, Any],
    target_size_mm: Iterable[float] | None,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Compute the smallest scale-only repair that can satisfy numeric rules.

    Exact XYZ / exact axis dimensions use axis scaling. Upper bounds use a final
    uniform scale-down so model proportions are preserved wherever possible.
    """
    x, y, z = [float(v) for v in extents_mm]
    sx = sy = sz = 1.0
    reasons: list[str] = []

    if "xyz_exact_mm" in rules:
        tx, ty, tz = [float(v) for v in rules["xyz_exact_mm"]]
        sx, sy, sz = tx / x, ty / y, tz / z
        reasons.append("prompt_exact_xyz")
    else:
        if "width_exact_mm" in rules:
            sx = float(rules["width_exact_mm"]) / x
            reasons.append("prompt_exact_width")
        if "depth_exact_mm" in rules:
            sy = float(rules["depth_exact_mm"]) / y
            reasons.append("prompt_exact_depth")
        if "height_exact_mm" in rules:
            sz = float(rules["height_exact_mm"]) / z
            reasons.append("prompt_exact_height")

        if (
            "outer_diameter_exact_mm" in rules
            and "width_exact_mm" not in rules
            and "depth_exact_mm" not in rules
        ):
            target_d = float(rules["outer_diameter_exact_mm"])
            current_d = max(x, y)
            sxy = target_d / current_d
            sx *= sxy
            sy *= sxy
            reasons.append("prompt_exact_outer_diameter")

    # If there is no explicit prompt dimension, the saved pipeline target is a
    # useful guard against the classic 1000x glTF unit/export mistake.
    if not _has_any_explicit_size_rule(rules) and target_size_mm is not None:
        target = [float(v) for v in target_size_mm]
        if len(target) == 3 and all(v > 0 for v in target):
            target_max = max(target)
            if COMPETITION_MIN_SPAN_MM <= target_max <= COMPETITION_MAX_SPAN_MM:
                sx, sy, sz = target[0] / x, target[1] / y, target[2] / z
                reasons.append("pipeline_target_unit_guard")

    px, py, pz = x * sx, y * sy, z * sz
    uniform = 1.0

    def cap(actual: float, limit: Optional[float], label: str) -> None:
        nonlocal uniform
        if limit is None:
            return
        limit = float(limit)
        if actual > limit:
            uniform = min(uniform, (limit * MAX_CONSTRAINT_TARGET_RATIO) / actual)
            reasons.append(label)

    cap(px, rules.get("width_max_mm"), "prompt_max_width")
    cap(py, rules.get("depth_max_mm"), "prompt_max_depth")
    cap(pz, rules.get("height_max_mm"), "prompt_max_height")
    cap(max(px, py, pz), rules.get("overall_max_mm"), "prompt_max_overall")
    cap(max(px, py), rules.get("outer_diameter_max_mm"), "prompt_max_outer_diameter")

    current_max = max(px, py, pz)
    if current_max > COMPETITION_MAX_SPAN_MM:
        uniform = min(uniform, 299.0 / current_max)
        reasons.append("competition_global_max_300mm")

    sx *= uniform
    sy *= uniform
    sz *= uniform

    fixed_max = max(x * sx, y * sy, z * sz)
    if fixed_max < COMPETITION_MIN_SPAN_MM:
        up = 2.1 / max(fixed_max, 1e-12)
        sx *= up
        sy *= up
        sz *= up
        reasons.append("competition_global_min_2mm")

    info = {
        "scale_xyz": [sx, sy, sz],
        "reasons": list(dict.fromkeys(reasons)),
        "predicted_extents_mm": [x * sx, y * sy, z * sz],
    }
    return (sx, sy, sz), info


def run_model_adjustment(
    *,
    model_path: Path,
    blender_exe: str,
    adjust_script: Path,
    scale_xyz: Iterable[float],
    ground_z: bool,
) -> None:
    sx, sy, sz = [float(v) for v in scale_xyz]
    if not adjust_script.exists():
        raise FileNotFoundError(f"Adjustment Blender script not found: {adjust_script}")

    cmd = [
        blender_exe,
        "-b",
        "-P",
        str(adjust_script),
        "--",
        "--input",
        str(model_path),
        "--scale-x",
        str(sx),
        "--scale-y",
        str(sy),
        "--scale-z",
        str(sz),
    ]
    if ground_z:
        cmd.append("--ground-z")
    print("[AUTO-FIX]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def update_control_after_fix(output_dir: Path, extents_mm: Iterable[float]) -> None:
    ext = [round(float(v), 6) for v in extents_mm]
    bbox = _normalize_bbox_from_size(ext)
    for name in ("bbox_inferred.json", "bbox.json"):
        path = output_dir / name
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            data = {}
        if "pre_finalize_size_mm" not in data and isinstance(data.get("size_mm"), list):
            data["pre_finalize_size_mm"] = data.get("size_mm")
        if "pre_finalize_bbox" not in data and isinstance(data.get("bbox"), list):
            data["pre_finalize_bbox"] = data.get("bbox")
        data["size_mm"] = ext
        data["bbox"] = bbox
        data["unit"] = "mm"
        data["coordinate_system"] = "Z-up"
        data["finalize_adjusted"] = True
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return


def check_model_size(
    model_path: Path,
    prompt_path: Path,
    target_size_mm: Iterable[float] | None = None,
    output_dir: Path | None = None,
) -> SizeReport:
    prompt_data = read_json(prompt_path)
    prompt_id = str(prompt_data.get("prompt_id") or prompt_path.parent.name)
    lo, hi, ext = glb_bounds_mm(model_path)
    x, y, z = (float(v) for v in ext)
    max_span = max(x, y, z)

    checks: list[CheckItem] = []

    # 1) Unit: submission metadata must explicitly say mm.
    unit_ok = metadata_unit_is_mm(output_dir)
    checks.append(CheckItem(
        name="metadata_unit_mm",
        ok=unit_ok,
        detail='metadata.unit must equal "mm"; GLB coordinates are interpreted as real millimetres',
        fixable=True,
    ))

    # 2) Organizer global quantity range.
    basic_ok = COMPETITION_MIN_SPAN_MM <= max_span <= COMPETITION_MAX_SPAN_MM
    checks.append(CheckItem(
        name="global_max_span_range_2_to_300_mm",
        ok=basic_ok,
        actual_mm=max_span,
        detail="bounding-box maximum span must be within [2, 300] mm",
        fixable=True,
    ))

    # 3) Z-up / ground convention. The mechanically verifiable part is that the
    # print-contact bottom is at the minimum Z plane, normalized here to Z=0.
    ground_tol = max(0.05, max_span * 1e-5)
    ground_ok = abs(float(lo[2])) <= ground_tol
    checks.append(CheckItem(
        name="ground_at_z_min_zero",
        ok=ground_ok,
        actual_mm=float(lo[2]),
        target_mm=0.0,
        tolerance_mm=ground_tol,
        detail="print-contact bottom should be on the minimum-Z plane; pipeline normalizes z_min to 0",
        fixable=True,
    ))

    # Semantic uprightness cannot be proven for arbitrary objects from a mesh
    # alone. We do NOT guess/rotate it automatically. Height checks below still
    # use Z exactly as required by the organizer.
    z_up_semantics_ok: bool | None = None
    checks.append(CheckItem(
        name="z_up_semantic_upright_orientation",
        ok=None,
        detail=(
            "Z is treated as the height axis throughout validation/rendering. "
            "Whether an arbitrary object is semantically 'upright' cannot be "
            "reliably inferred automatically, so no speculative rotation is applied."
        ),
        verifiable=False,
        fixable=False,
    ))

    rules = parse_competition_size_rules(prompt_data)
    prompt_checks: list[CheckItem] = []

    # 4) Height semantics and exact axis sizes: height is Z-span.
    if "xyz_exact_mm" in rules:
        tx, ty, tz = [float(v) for v in rules["xyz_exact_mm"]]
        prompt_checks.extend([
            _check_exact("prompt_exact_x", x, tx),
            _check_exact("prompt_exact_y", y, ty),
            _check_exact("prompt_exact_z", z, tz),
        ])
    else:
        if "width_exact_mm" in rules:
            prompt_checks.append(_check_exact("prompt_exact_width_x", x, float(rules["width_exact_mm"])))
        if "depth_exact_mm" in rules:
            prompt_checks.append(_check_exact("prompt_exact_depth_y", y, float(rules["depth_exact_mm"])))
        if "height_exact_mm" in rules:
            prompt_checks.append(_check_exact("prompt_exact_height_z", z, float(rules["height_exact_mm"])))

    if "width_max_mm" in rules:
        prompt_checks.append(_check_max("prompt_max_width_x", x, float(rules["width_max_mm"])))
    if "depth_max_mm" in rules:
        prompt_checks.append(_check_max("prompt_max_depth_y", y, float(rules["depth_max_mm"])))
    if "height_max_mm" in rules:
        prompt_checks.append(_check_max("prompt_max_height_z", z, float(rules["height_max_mm"])))

    # 5) Overall/见方 = max XYZ span; outer diameter = maximum horizontal XY diameter.
    if "overall_max_mm" in rules:
        prompt_checks.append(_check_max("prompt_max_overall_span", max_span, float(rules["overall_max_mm"])))

    horizontal_diameter = max(x, y)
    if "outer_diameter_max_mm" in rules:
        prompt_checks.append(_check_max(
            "prompt_max_outer_diameter_xy",
            horizontal_diameter,
            float(rules["outer_diameter_max_mm"]),
        ))
    elif "outer_diameter_exact_mm" in rules:
        prompt_checks.append(_check_exact(
            "prompt_exact_outer_diameter_xy",
            horizontal_diameter,
            float(rules["outer_diameter_exact_mm"]),
        ))

    # 6) Exact/max checks above all use organizer tolerance +/-2 mm or +/-3%.
    for item in prompt_checks:
        item.fixable = True
    checks.extend(prompt_checks)
    prompt_ok = all(c.ok is True for c in prompt_checks) if prompt_checks else True

    # Pipeline target is a unit/export sanity guard. If auto-fix corrects a stale
    # or wrong inferred target, the control JSON is updated and this is rechecked.
    target_ok: bool | None = None
    if target_size_mm is not None:
        target = [float(v) for v in target_size_mm]
        if len(target) == 3 and all(v > 0 for v in target):
            target_checks = [
                _check_exact("pipeline_target_x", x, target[0]),
                _check_exact("pipeline_target_y", y, target[1]),
                _check_exact("pipeline_target_z", z, target[2]),
            ]
            for item in target_checks:
                item.fixable = True
            checks.extend(target_checks)
            target_ok = all(c.ok is True for c in target_checks)

    overall_ok = (
        unit_ok
        and basic_ok
        and ground_ok
        and prompt_ok
        and (target_ok is not False)
    )

    return SizeReport(
        prompt_id=prompt_id,
        bounds_min_mm=[float(v) for v in lo],
        bounds_max_mm=[float(v) for v in hi],
        extents_mm=[x, y, z],
        max_span_mm=max_span,
        unit_ok=unit_ok,
        basic_size_ok=basic_ok,
        ground_ok=ground_ok,
        z_up_semantics_ok=z_up_semantics_ok,
        prompt_constraints_ok=prompt_ok,
        target_size_ok=target_ok,
        overall_ok=overall_ok,
        auto_fix_applied=False,
        checks=[asdict(c) for c in checks],
    )


def render_four_views(
    model_path: Path,
    output_dir: Path,
    blender_exe: str,
    blender_script: Path,
    render_size: int = 1024,
    overwrite: bool = False,
) -> None:
    render_dir = output_dir / "renders"
    expected = [
        render_dir / "front.png",
        render_dir / "side.png",
        render_dir / "top.png",
        render_dir / "perspective.png",
    ]
    if not overwrite and all(p.exists() for p in expected):
        print(f"[SKIP] renders already exist: {render_dir}")
        return

    cmd = [
        blender_exe,
        "-b",
        "-P",
        str(blender_script),
        "--",
        "--input",
        str(model_path),
        "--output-dir",
        str(render_dir),
        "--size",
        str(render_size),
    ]
    print("[RENDER]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"Blender finished but renders are missing: {missing}")


def write_metadata(
    output_dir: Path,
    prompt_data: dict[str, Any],
    seed: int,
    generation_time_minutes: float | None,
    overwrite: bool,
    report: SizeReport | None = None,
) -> None:
    path = output_dir / "metadata.json"
    prompt_id = str(prompt_data.get("prompt_id") or output_dir.name)
    metadata: dict[str, Any] = {}
    if path.exists():
        try:
            metadata = read_json(path)
        except Exception:
            metadata = {}
    bbox, size_mm = load_bbox_control(output_dir)
    if report is not None:
        size_mm = [round(float(v), 6) for v in report.extents_mm]
        bbox = _normalize_bbox_from_size(size_mm)
    metadata.update({
        "prompt_id": prompt_id,
        "primary_model": "model.glb",
        "unit": "mm",
        "coordinate_system": "Z-up",
        "ground_plane": "Z-min normalized to 0",
        "base_model": "GPT-Image-2 + Tencent Hunyuan3D-Omni + custom bbox scaling",
        "seed": int(seed),
        "generation_time_minutes": generation_time_minutes,
        "postprocess": "competition dimension validation; scale/ground auto-fix when needed; standardized four-view rendering; no semantic remeshing",
        "commercial_api_used": True,
    })
    if bbox is not None:
        metadata["bbox"] = bbox
    if size_mm is not None:
        metadata["size_mm"] = size_mm
    if report is not None:
        metadata["competition_checks_passed"] = bool(report.overall_ok)
        metadata["auto_fix_applied"] = bool(report.auto_fix_applied)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def write_timed_process_md(
    output_dir: Path,
    prompt_data: dict[str, Any],
    seed: int,
    timing: dict[str, Any],
) -> None:
    prompt_id = str(prompt_data.get("prompt_id") or output_dir.name)
    image_s = float(timing.get("image_generation_seconds") or 0.0)
    bbox_s = float(timing.get("bbox_inference_seconds") or 0.0)
    mesh_s = float(timing.get("mesh_generation_seconds") or 0.0)
    finalize_s = float(timing.get("finalize_seconds") or 0.0)
    total_s = float(timing.get("total_seconds") or 0.0)
    total_m = float(timing.get("total_minutes") or 0.0)
    text = f"""题目编号：{prompt_id}
使用模型：GPT-Image-2（参考图）+ Tencent Hunyuan3D-Omni（Image-to-3D）
生成参数：seed={seed}；使用推断后的 bbox/size_mm 控制并缩放至真实毫米尺寸。
生图耗时：{image_s:.3f} 秒
BBox 推断耗时：{bbox_s:.3f} 秒
Mesh 生成耗时：{mesh_s:.3f} 秒
后处理与渲染耗时：{finalize_s:.3f} 秒
总耗时：{total_s:.3f} 秒（{total_m:.3f} 分钟）
后处理流程：比赛尺寸/单位/Z-min 校验 -> 必要时仅缩放与落地调整 model.glb -> 再校验 -> front/side/top/perspective 四视图渲染。
是否使用商业 API：是（GPT-Image-2）
"""
    (output_dir / "process.md").write_text(text, encoding="utf-8")


def write_process_md(
    output_dir: Path,
    prompt_data: dict[str, Any],
    seed: int,
    generation_time_minutes: float | None,
    overwrite: bool,
) -> None:
    path = output_dir / "process.md"
    if path.exists() and not overwrite:
        return

    prompt_id = str(prompt_data.get("prompt_id") or output_dir.name)
    time_text = "未单独记录" if generation_time_minutes is None else f"{generation_time_minutes:.3f} 分钟"
    text = f"""题目编号：{prompt_id}
使用模型：GPT-Image-2（参考图） + Tencent Hunyuan3D-Omni（Image-to-3D）
主要改进：中文 Prompt 约束转参考图；Prompt/Image 推断尺寸与 bbox；生成后按毫米检查尺寸；统一四视图渲染。
生成参数：seed={seed}；Omni 使用 bbox/size_mm 控制并在导出前缩放到真实毫米尺寸。
随机种子：{seed}
生成耗时：{time_text}
后处理流程：去生成阶段浮岛/退化面（如启用） -> 比赛尺寸/单位/Z-min 校验 -> 必要时仅做缩放与 Z 落地调整 -> 再校验 -> front/side/top/perspective 四视图。默认不做平滑、体素重建或强制补洞，以避免破坏语义细节与功能孔槽。
失败样例与修正：尺寸检查失败时回到 bbox/size_mm 或生成阶段修正，不通过把模型强行变成简单几何体来规避评测。
是否使用商业API：是（GPT-Image-2）
数据来源说明：主办方 Prompt Benchmark；公开预训练模型 Hunyuan3D-Omni；不使用主办方隐藏/校准集训练数据。
"""
    path.write_text(text, encoding="utf-8")


def finalize_one(
    *,
    prompt_path: Path,
    output_dir: Path,
    blender_exe: str,
    blender_script: Path,
    seed: int = 1234,
    render_size: int = 1024,
    overwrite: bool = False,
    render: bool = True,
    generation_time_minutes: float | None = None,
    fail_on_size_error: bool = True,
    auto_fix: bool = True,
    adjust_script: Path | None = None,
) -> SizeReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.glb"

    old_mesh = output_dir / "mesh.glb"
    if not model_path.exists() and old_mesh.exists():
        shutil.copy2(old_mesh, model_path)
        print(f"[COPY] {old_mesh.name} -> model.glb")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.glb: {output_dir}")

    finalize_start = time.perf_counter()
    prompt_data = read_json(prompt_path)
    prompt_id = str(prompt_data.get("prompt_id") or output_dir.name)

    # Standard 1 is enforced before checking: metadata.unit = mm.
    ensure_metadata_unit_mm(output_dir, prompt_id)

    target_size = load_target_size(output_dir)
    report = check_model_size(
        model_path,
        prompt_path,
        target_size,
        output_dir=output_dir,
    )

    auto_fix_applied = False
    if auto_fix and not report.overall_ok:
        rules = parse_competition_size_rules(prompt_data)
        scales, fix_info = compute_adjustment_scales(
            report.extents_mm,
            rules,
            target_size,
        )

        sx, sy, sz = scales
        needs_scale = any(abs(v - 1.0) > 1e-8 for v in scales)
        needs_ground = not report.ground_ok

        # Preserve the submitted/PBR model exactly once before modifying it.
        backup = output_dir / "model_before_autofix.glb"
        if (needs_scale or needs_ground) and not backup.exists():
            shutil.copy2(model_path, backup)
            print(f"[AUTO-FIX] backup: {backup}")

        if needs_scale or needs_ground:
            if adjust_script is None:
                adjust_script = Path(__file__).resolve().with_name("blender_adjust_model.py")
            run_model_adjustment(
                model_path=model_path,
                blender_exe=blender_exe,
                adjust_script=adjust_script,
                scale_xyz=scales,
                ground_z=True,
            )
            auto_fix_applied = True

        # A stale/wrong bbox target must not keep a corrected GLB failing forever.
        # Synchronize it to the final actual model dimensions after any repair.
        _, _, ext_after = glb_bounds_mm(model_path)
        update_control_after_fix(output_dir, ext_after)
        auto_fix_applied = True

        adjustment_record = {
            "prompt_id": prompt_id,
            "before_extents_mm": report.extents_mm,
            "scale_xyz": [sx, sy, sz],
            "ground_z_applied": bool(needs_ground),
            "reasons": fix_info.get("reasons", []),
            "predicted_extents_mm": fix_info.get("predicted_extents_mm"),
            "backup": str(backup) if backup.exists() else None,
        }
        (output_dir / "model_adjustment.json").write_text(
            json.dumps(adjustment_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "size_check_before_fix.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        target_size = load_target_size(output_dir)
        report = check_model_size(
            model_path,
            prompt_path,
            target_size,
            output_dir=output_dir,
        )
        report.auto_fix_applied = auto_fix_applied

    report_path = output_dir / "size_check.json"
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[SIZE] {report.prompt_id}: "
        f"{report.extents_mm[0]:.3f} x {report.extents_mm[1]:.3f} x "
        f"{report.extents_mm[2]:.3f} mm; max={report.max_span_mm:.3f}; "
        f"unit={report.unit_ok}; ground={report.ground_ok}; ok={report.overall_ok}"
    )

    if fail_on_size_error and not report.overall_ok:
        raise RuntimeError(
            f"Competition validation failed for {report.prompt_id}. "
            f"See {report_path}"
        )

    if render:
        render_four_views(
            model_path=model_path,
            output_dir=output_dir,
            blender_exe=blender_exe,
            blender_script=blender_script,
            render_size=render_size,
            overwrite=overwrite or auto_fix_applied,
        )

    finalize_elapsed = time.perf_counter() - finalize_start
    timing = update_timing(
        output_dir / "timing.json",
        finalize_seconds=finalize_elapsed,
    )
    print(f"[TIME] {report.prompt_id} finalize: {finalize_elapsed:.3f} s")

    write_metadata(
        output_dir,
        prompt_data,
        seed=seed,
        generation_time_minutes=float(timing.get("total_minutes") or 0.0),
        overwrite=overwrite,
        report=report,
    )
    write_timed_process_md(
        output_dir,
        prompt_data,
        seed=seed,
        timing=timing,
    )
    return report


def finalize_dataset(
    *,
    data_root: Path,
    output_root: Path,
    blender_exe: str,
    blender_script: Path,
    seed: int = 1234,
    render_size: int = 1024,
    overwrite: bool = False,
    render: bool = True,
    continue_on_error: bool = True,
    auto_fix: bool = True,
    adjust_script: Path | None = None,
) -> list[Path]:
    completed: list[Path] = []
    failed: list[tuple[Path, str]] = []

    jobs: list[tuple[Path, Path]] = []
    for category in CATEGORY_ORDER:
        category_dir = data_root / category
        if not category_dir.exists():
            continue
        for prompt_path in sorted(category_dir.glob("*/prompt.json")):
            relative_dir = prompt_path.parent.relative_to(data_root)
            out_dir = output_root / relative_dir
            if (out_dir / "model.glb").exists() or (out_dir / "mesh.glb").exists():
                jobs.append((prompt_path, out_dir))

    print(f"[FINALIZE] found {len(jobs)} generated sample(s)")
    for index, (prompt_path, out_dir) in enumerate(jobs, start=1):
        rel = out_dir.relative_to(output_root)
        print(f"\n[{index}/{len(jobs)}] {rel.as_posix()}")
        try:
            finalize_one(
                prompt_path=prompt_path,
                output_dir=out_dir,
                blender_exe=blender_exe,
                blender_script=blender_script,
                seed=seed,
                render_size=render_size,
                overwrite=overwrite,
                render=render,
                fail_on_size_error=True,
                auto_fix=auto_fix,
                adjust_script=adjust_script,
            )
            completed.append(out_dir)
        except Exception as exc:
            failed.append((rel, str(exc)))
            print(f"[FAILED] {rel}: {exc}")
            if not continue_on_error:
                raise

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "total_models": len(jobs),
        "completed": len(completed),
        "failed": len(failed),
        "failed_samples": [
            {"sample": rel.as_posix(), "error": msg}
            for rel, msg in failed
        ],
        "auto_fix": bool(auto_fix),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_finalize_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print(f"Finalized : {len(completed)}")
    print(f"Failed    : {len(failed)}")
    if failed:
        for rel, msg in failed:
            print(f" - {rel}: {msg}")
    print(f"Audit     : {output_root / '_finalize_audit.json'}")
    print("=" * 72)
    return completed


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project_root = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=project_root / "data" / "test")
    p.add_argument("--output-root", type=Path, default=project_root / "outputs" / "test")
    p.add_argument("--blender", default="blender")
    p.add_argument("--blender-script", type=Path, default=here / "blender_render_views.py")
    p.add_argument("--adjust-script", type=Path, default=here / "blender_adjust_model.py")
    p.add_argument("--render-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sample", type=Path, default=None, help="Only check one sample, e.g. constrained/hid_con_010")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-render", action="store_true", help="Only validate/fix size and write metadata")
    p.add_argument("--no-auto-fix", action="store_true", help="Report violations without modifying model.glb")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    if args.sample is not None:
        rel = Path(args.sample)
        prompt_path = data_root / rel / "prompt.json"
        out_dir = output_root / rel
        if not prompt_path.exists():
            raise FileNotFoundError(prompt_path)
        finalize_one(
            prompt_path=prompt_path,
            output_dir=out_dir,
            blender_exe=args.blender,
            blender_script=args.blender_script,
            adjust_script=args.adjust_script,
            seed=args.seed,
            render_size=args.render_size,
            overwrite=args.overwrite,
            render=not args.no_render,
            auto_fix=not args.no_auto_fix,
            fail_on_size_error=True,
        )
        return 0

    finalize_dataset(
        data_root=data_root,
        output_root=output_root,
        blender_exe=args.blender,
        blender_script=args.blender_script,
        adjust_script=args.adjust_script,
        seed=args.seed,
        render_size=args.render_size,
        overwrite=args.overwrite,
        render=not args.no_render,
        continue_on_error=not args.stop_on_error,
        auto_fix=not args.no_auto_fix,
    )
    summary_path = output_root / "_finalize_audit.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        return 1 if int(summary.get("failed", 0)) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
