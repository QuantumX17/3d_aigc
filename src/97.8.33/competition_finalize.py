#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition finalization after model.glb has been generated.

Responsibilities are intentionally narrow:
  1. Validate that GLB coordinate numbers represent plausible millimetres.
  2. Check explicit size constraints in the prompt using the organizer tolerance.
  3. Check the saved bbox/size target to catch a 1000x unit/export mistake.
  4. Render front/side/top/perspective PNGs through Blender.
  5. Write metadata.json and process.md.

This module does NOT smooth, voxelize, remesh, or otherwise alter the submitted GLB.
That avoids degrading semantic details after Image-to-3D generation.
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
from typing import Any, Iterable

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
}


@dataclass
class CheckItem:
    name: str
    ok: bool
    actual_mm: float | None = None
    target_mm: float | None = None
    tolerance_mm: float | None = None
    detail: str = ""


@dataclass
class SizeReport:
    prompt_id: str
    bounds_min_mm: list[float]
    bounds_max_mm: list[float]
    extents_mm: list[float]
    max_span_mm: float
    basic_size_ok: bool
    prompt_constraints_ok: bool
    target_size_ok: bool | None
    overall_ok: bool
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
        r"(?P<z>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
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
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)"
        ),
        "width_max_mm": (
            r"(?:宽|宽度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)"
        ),
        "depth_max_mm": (
            r"(?:深|深度|厚|厚度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)"
        ),
        "overall_max_mm": (
            r"(?:整体|整体尺寸|见方|包围盒|最大跨度|最大尺寸).{0,10}?"
            r"(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)"
        ),
        "outer_diameter_max_mm": (
            r"(?:外径|外直径)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)"
        ),
        "outer_diameter_exact_mm": (
            r"(?:外径|外直径)\s*(?:为|约为|约|=|:|：)?\s*"
            r"(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)"
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
        "height_exact_mm": r"(?:整体)?(?:高|高度)\s*(?:为|约为|约|=|:|：)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "width_exact_mm": r"(?:宽|宽度)\s*(?:为|约为|约|=|:|：)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "depth_exact_mm": r"(?:深|深度|厚|厚度)\s*(?:为|约为|约|=|:|：)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
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


def check_model_size(
    model_path: Path,
    prompt_path: Path,
    target_size_mm: Iterable[float] | None = None,
) -> SizeReport:
    prompt_data = read_json(prompt_path)
    prompt_id = str(prompt_data.get("prompt_id") or prompt_path.parent.name)
    lo, hi, ext = glb_bounds_mm(model_path)
    x, y, z = (float(v) for v in ext)
    max_span = max(x, y, z)

    checks: list[CheckItem] = []

    # Global physical range from the organizer slide.
    checks.append(CheckItem(
        name="global_max_span_range_2_to_300_mm",
        ok=2.0 <= max_span <= 300.0,
        actual_mm=max_span,
        target_mm=None,
        tolerance_mm=None,
        detail="bounding-box maximum span must be within [2, 300] mm",
    ))
    basic_ok = checks[-1].ok

    rules = parse_competition_size_rules(prompt_data)
    prompt_checks: list[CheckItem] = []

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

    checks.extend(prompt_checks)
    prompt_ok = all(c.ok for c in prompt_checks) if prompt_checks else True

    target_ok: bool | None = None
    if target_size_mm is not None:
        target = [float(v) for v in target_size_mm]
        if len(target) == 3 and all(v > 0 for v in target):
            target_checks = [
                _check_exact("pipeline_target_x", x, target[0]),
                _check_exact("pipeline_target_y", y, target[1]),
                _check_exact("pipeline_target_z", z, target[2]),
            ]
            checks.extend(target_checks)
            target_ok = all(c.ok for c in target_checks)

    overall_ok = basic_ok and prompt_ok and (target_ok is not False)

    return SizeReport(
        prompt_id=prompt_id,
        bounds_min_mm=[float(v) for v in lo],
        bounds_max_mm=[float(v) for v in hi],
        extents_mm=[x, y, z],
        max_span_mm=max_span,
        basic_size_ok=basic_ok,
        prompt_constraints_ok=prompt_ok,
        target_size_ok=target_ok,
        overall_ok=overall_ok,
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
    metadata.update({
        "prompt_id": prompt_id,
        "primary_model": "model.glb",
        "unit": "mm",
        "base_model": "GPT-Image-2 + Tencent Hunyuan3D-Omni + custom bbox scaling",
        "seed": int(seed),
        "generation_time_minutes": generation_time_minutes,
        "postprocess": "dimension validation and standardized four-view rendering; no semantic remeshing",
        "commercial_api_used": True,
    })
    if bbox is not None:
        metadata["bbox"] = bbox
    if size_mm is not None:
        metadata["size_mm"] = size_mm
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
后处理流程：GLB 尺寸校验 -> front/side/top/perspective 四视图渲染。
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
后处理流程：去生成阶段浮岛/退化面（如启用） -> GLB 尺寸校验 -> front/side/top/perspective 四视图。默认不做平滑、体素重建或强制补洞，以避免破坏语义细节与功能孔槽。
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
) -> SizeReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.glb"

    # Backward compatibility with the previous mesh.glb name.
    old_mesh = output_dir / "mesh.glb"
    if not model_path.exists() and old_mesh.exists():
        shutil.copy2(old_mesh, model_path)
        print(f"[COPY] {old_mesh.name} -> model.glb")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.glb: {output_dir}")

    finalize_start = time.perf_counter()
    prompt_data = read_json(prompt_path)
    target_size = load_target_size(output_dir)
    report = check_model_size(model_path, prompt_path, target_size)

    report_path = output_dir / "size_check.json"
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[SIZE] {report.prompt_id}: "
        f"{report.extents_mm[0]:.3f} x {report.extents_mm[1]:.3f} x "
        f"{report.extents_mm[2]:.3f} mm; max={report.max_span_mm:.3f}; "
        f"ok={report.overall_ok}"
    )

    if fail_on_size_error and not report.overall_ok:
        raise RuntimeError(
            f"Size validation failed for {report.prompt_id}. "
            f"See {report_path}"
        )

    if render:
        render_four_views(
            model_path=model_path,
            output_dir=output_dir,
            blender_exe=blender_exe,
            blender_script=blender_script,
            render_size=render_size,
            overwrite=overwrite,
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
            )
            completed.append(out_dir)
        except Exception as exc:
            failed.append((rel, str(exc)))
            print(f"[FAILED] {rel}: {exc}")
            if not continue_on_error:
                raise

    print("\n" + "=" * 72)
    print(f"Finalized : {len(completed)}")
    print(f"Failed    : {len(failed)}")
    if failed:
        for rel, msg in failed:
            print(f" - {rel}: {msg}")
    print("=" * 72)
    return completed


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project_root = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=project_root / "data")
    p.add_argument("--output-root", type=Path, default=project_root / "outputs")
    p.add_argument("--blender", default="blender")
    p.add_argument("--blender-script", type=Path, default=here / "blender_render_views.py")
    p.add_argument("--render-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-render", action="store_true", help="Only validate size/write metadata")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    finalize_dataset(
        data_root=args.data_root,
        output_root=args.output_root,
        blender_exe=args.blender,
        blender_script=args.blender_script,
        seed=args.seed,
        render_size=args.render_size,
        overwrite=args.overwrite,
        render=not args.no_render,
        continue_on_error=not args.stop_on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
