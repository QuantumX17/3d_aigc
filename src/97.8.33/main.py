#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Competition pipeline:
    GPT-Image-2
      -> Hunyuan3D geometry
      -> multi-view mesh_to_albedo
      -> DeepBump
      -> Blender PBR material
      -> size check + render views

Recommended single-sample full test:
    python main.py --single-test --overwrite --blender "D:\Program Files\Blender Foundation\Blender 5.2\blender.exe"

DeepBump uses the currently active Python environment by default.
For this project:
    conda activate hunyuan3domni

Texture-only full rerun for the first sample:
    python main.py --single-test --rerun-texture ...

Rerun every stage AFTER an existing Hunyuan3D-Omni geometry:
    python main.py --single-test --rerun-after-mesh ...

DeepBump-only rerun for the first sample:
    python main.py --single-test --rerun-deepbump

DeepBump-only rerun + rebuild final PBR model.glb:
    python main.py --single-test --rerun-deepbump --apply-pbr-after-deepbump
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from text_to_image_gpt import TextToImageRunner
from image_to_mesh_omni import ImageToMeshRunner
from competition_finalize import finalize_dataset, finalize_one
from timing_utils import read_timing


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = Path(__file__).resolve().parent
TEXTURE_ROOT = SRC_ROOT

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "test"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "inputs" / "test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "test"

DEFAULT_DEEPBUMP_DIR = PROJECT_ROOT / "DeepBump"

MESH_TO_ALBEDO_SCRIPT = TEXTURE_ROOT / "mesh_to_albedo.py"
DEEPBUMP_RUNNER_SCRIPT = TEXTURE_ROOT / "deepbump_runner.py"
BLENDER_APPLY_PBR_SCRIPT = TEXTURE_ROOT / "blender_apply_pbr.py"


def _run(cmd):
    print("[RUN]", subprocess.list2cmdline([str(x) for x in cmd]))
    subprocess.run([str(x) for x in cmd], check=True)


def _first_sample_relative(data_root: Path) -> Path:
    paths = sorted(data_root.glob("*/*/prompt.json"), key=lambda p: p.as_posix())
    if not paths:
        raise FileNotFoundError(f"No samples found under {data_root}")
    return paths[0].parent.relative_to(data_root)


def _detect_deepbump_python() -> Optional[Path]:
    """
    Prefer the Python interpreter currently running main.py.

    In this project DeepBump shares the hunyuan3domni environment, so when:
        conda activate hunyuan3domni
        python main.py ...

    DeepBump automatically uses:
        D:\ProgramData\Miniconda3\envs\hunyuan3domni\python.exe

    DEEPBUMP_PYTHON is still supported as an explicit override.
    """
    env_value = os.environ.get("DEEPBUMP_PYTHON")
    if env_value:
        p = Path(env_value)
        if p.exists():
            return p.resolve()

    current_python = Path(sys.executable)
    if current_python.exists():
        return current_python.resolve()

    return None


def _require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _update_timing_json(out_dir: Path, stage_name: str, seconds: float):
    """
    Preserve existing timing.json and append wall-clock timings for the new stages.
    This avoids depending on the exact schema used by timing_utils.
    """
    timing_path = out_dir / "timing.json"
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8-sig"))
        except Exception:
            timing = {}
    else:
        timing = {}

    stages = timing.setdefault("stage_wall_seconds", {})
    stages[stage_name] = round(float(seconds), 6)
    stages["new_texture_pipeline_total"] = round(
        float(
            stages.get("mesh_to_albedo", 0.0)
            + stages.get("deepbump", 0.0)
            + stages.get("apply_pbr", 0.0)
        ),
        6,
    )

    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        json.dumps(timing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ensure_geometry_source(out_dir: Path, overwrite: bool = False) -> Path:
    """
    Keep a clean geometry copy so that final model.glb may be overwritten by
    the textured PBR model without destroying the Hunyuan output.
    """
    source = out_dir / "model.glb"
    geometry = out_dir / "model_geometry.glb"

    _require_file(source, "Hunyuan model.glb")

    if overwrite or not geometry.exists():
        shutil.copy2(source, geometry)
        print(f"[GEOMETRY] preserved source mesh: {geometry}")

    return geometry


def run_image_stage(
    data_root,
    image_root,
    output_root,
    overwrite,
    continue_on_error,
    sample_relative=None,
):
    print("\n=== STAGE 1 / PROMPT -> GPT-IMAGE-2 ===")
    runner = TextToImageRunner(timeout=600.0)
    runner.process_dataset(
        data_root=data_root,
        output_root=image_root,
        timing_output_root=output_root,
        sample_relative=sample_relative,
        prompt_name="prompt.json",
        image_name="image.png",
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def run_mesh_stage(
    data_root,
    image_root,
    output_root,
    seed,
    inference_steps,
    octree_resolution,
    overwrite,
    continue_on_error,
    sample_relative=None,
):
    print("\n=== STAGE 2 / IMAGE + PROMPT SIZE -> HUNYUAN3D GEOMETRY ===")
    runner = ImageToMeshRunner(
        remove_background=True,
        guidance_scale=4.5,
        remove_floaters=True,
        remove_degenerate_faces=True,
    )
    try:
        runner.process_dataset(
            input_root=image_root,
            output_root=output_root,
            data_root=data_root,
            image_name="image.png",
            mesh_name="model.glb",
            prompt_json_name="prompt.json",
            auto_prompt_bbox=True,
            align_bbox_to_image=True,
            image_ambiguity_ratio=1.08,
            save_inferred_control=True,
            inferred_control_name="bbox_inferred.json",
            exact_size=True,
            seed=seed,
            inference_steps=inference_steps,
            octree_resolution=octree_resolution,
            num_chunks=8000,
            mc_level=0.0,
            overwrite=overwrite,
            continue_on_error=continue_on_error,
            require_control_file=True,
            sample_relative=sample_relative,
        )
    finally:
        runner.unload()

    # Preserve raw Hunyuan geometry for each requested sample.
    if sample_relative is not None:
        out_dir = Path(output_root) / sample_relative
        _ensure_geometry_source(out_dir, overwrite=overwrite)


def run_albedo_one(
    data_root: Path,
    image_root: Path,
    output_root: Path,
    sample_relative: Path,
    *,
    blender: str,
    texture_views,
    texture_resolution: int,
    texture_size: int,
    texture_steps: int,
    texture_strength: float,
    texture_guidance_scale: float,
    texture_controlnet_scale: float,
    seed: int,
    overwrite: bool,
):
    print("\n=== STAGE 3 / MESH + REFERENCE -> UV ALBEDO ===")

    _require_file(MESH_TO_ALBEDO_SCRIPT, "mesh_to_albedo.py")

    out_dir = output_root / sample_relative
    image_path = image_root / sample_relative / "image.png"
    prompt_path = data_root / sample_relative / "prompt.json"
    texture_dir = out_dir / "texture"
    albedo_path = texture_dir / "albedo.png"

    _require_file(image_path, "reference image")
    _require_file(prompt_path, "prompt.json")

    geometry_path = out_dir / "model_geometry.glb"
    if not geometry_path.exists():
        geometry_path = _ensure_geometry_source(out_dir, overwrite=False)

    if albedo_path.exists() and not overwrite:
        print(f"[SKIP] Albedo already exists: {albedo_path}")
        return albedo_path, 0.0

    cmd = [
        sys.executable,
        MESH_TO_ALBEDO_SCRIPT,
        "--input", geometry_path,
        "--reference", image_path,
        "--prompt-json", prompt_path,
        "--output-dir", texture_dir,
        "--blender", blender,
        "--resolution", str(texture_resolution),
        "--texture-size", str(texture_size),
        "--steps", str(texture_steps),
        "--strength", str(texture_strength),
        "--guidance-scale", str(texture_guidance_scale),
        "--controlnet-scale", str(texture_controlnet_scale),
        "--seed", str(seed),
        "--views", *texture_views,
    ]

    t0 = time.perf_counter()
    _run(cmd)
    elapsed = time.perf_counter() - t0
    _require_file(albedo_path, "generated albedo.png")
    _update_timing_json(out_dir, "mesh_to_albedo", elapsed)
    return albedo_path, elapsed


def run_deepbump_one(
    output_root: Path,
    sample_relative: Path,
    *,
    deepbump_dir: Path,
    deepbump_python: Path,
    overwrite: bool,
):
    print("\n=== STAGE 4 / ALBEDO -> DEEPBUMP NORMAL + HEIGHT + CURVATURE ===")

    _require_file(DEEPBUMP_RUNNER_SCRIPT, "deepbump_runner.py")
    _require_file(deepbump_dir / "cli.py", "DeepBump cli.py")
    _require_file(deepbump_python, "DeepBump Python")

    out_dir = output_root / sample_relative
    texture_dir = out_dir / "texture"
    albedo = texture_dir / "albedo.png"
    pbr_dir = texture_dir / "pbr"
    normal = pbr_dir / "normal.png"
    height = pbr_dir / "height.png"
    curvature = pbr_dir / "curvature.png"

    _require_file(albedo, "albedo.png")

    if normal.exists() and height.exists() and curvature.exists() and not overwrite:
        print(f"[SKIP] DeepBump outputs already exist: {pbr_dir}")
        return pbr_dir, 0.0

    cmd = [
        sys.executable,
        DEEPBUMP_RUNNER_SCRIPT,
        "--albedo", albedo,
        "--output-dir", pbr_dir,
        "--deepbump-dir", deepbump_dir,
        "--python", deepbump_python,
    ]

    t0 = time.perf_counter()
    _run(cmd)
    elapsed = time.perf_counter() - t0

    _require_file(normal, "DeepBump normal.png")
    _require_file(height, "DeepBump height.png")
    _require_file(curvature, "DeepBump curvature.png")
    _update_timing_json(out_dir, "deepbump", elapsed)
    return pbr_dir, elapsed


def run_pbr_one(
    output_root: Path,
    sample_relative: Path,
    *,
    blender: str,
    pbr_normal_strength: float,
    pbr_bump_strength: float,
    pbr_bump_distance: float,
    roughness: float,
    metallic: float,
    overwrite: bool,
):
    print("\n=== STAGE 5 / APPLY ALBEDO + DEEPBUMP MAPS -> FINAL MODEL.GLB ===")

    _require_file(BLENDER_APPLY_PBR_SCRIPT, "blender_apply_pbr.py")

    out_dir = output_root / sample_relative
    texture_dir = out_dir / "texture"
    
    albedo_mesh = texture_dir / "albedo_mesh.glb"
    uv_mesh = texture_dir / "uv_mesh.glb"
    if albedo_mesh.exists():
        source_mesh = albedo_mesh
        print(f"[PBR] source mesh: {source_mesh}")
    elif uv_mesh.exists():
        source_mesh = uv_mesh
        print(
            "[PBR] albedo_mesh.glb not found; "
            "falling back to uv_mesh.glb"
        )
    else:
        raise FileNotFoundError(
            "Neither albedo_mesh.glb nor uv_mesh.glb exists under "
            f"{texture_dir}"
    )
    albedo = texture_dir / "albedo.png"
    normal = texture_dir / "pbr" / "normal.png"
    height = texture_dir / "pbr" / "height.png"
    final_mesh = out_dir / "model.glb"

    _require_file(source_mesh, "PBR source mesh")
    _require_file(albedo, "albedo.png")
    _require_file(normal, "normal.png")
    _require_file(height, "height.png")

    # If final model.glb exists, it may simply be the geometry generated by
    # Hunyuan. Therefore only skip if a marker confirms PBR was already applied.
    marker = texture_dir / "pbr_applied.json"
    if marker.exists() and final_mesh.exists() and not overwrite:
        print(f"[SKIP] PBR model already exists: {final_mesh}")
        return final_mesh, 0.0

    cmd = [
        blender,
        "-b",
        "-P",
        BLENDER_APPLY_PBR_SCRIPT,
        "--",
        "--input", source_mesh,
        "--albedo", albedo,
        "--normal", normal,
        "--height", height,
        "--output", final_mesh,
        "--normal-strength", str(pbr_normal_strength),
        "--bump-strength", str(pbr_bump_strength),
        "--bump-distance", str(pbr_bump_distance),
        "--roughness", str(roughness),
        "--metallic", str(metallic),
    ]

    t0 = time.perf_counter()
    _run(cmd)
    elapsed = time.perf_counter() - t0
    _require_file(final_mesh, "final PBR model.glb")

    marker.write_text(
        json.dumps(
            {
                "model": str(final_mesh),
                "albedo": str(albedo),
                "normal": str(normal),
                "height": str(height),
                "normal_strength": pbr_normal_strength,
                "bump_strength": pbr_bump_strength,
                "bump_distance": pbr_bump_distance,
                "roughness": roughness,
                "metallic": metallic,
                "seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _update_timing_json(out_dir, "apply_pbr", elapsed)
    return final_mesh, elapsed


def run_texture_pipeline_one(args, rel: Path, *, force_overwrite: bool):
    """
    New Stage 3-5:
      mesh/reference -> albedo -> DeepBump -> final PBR GLB

    Important:
      Generate/preserve albedo first. DeepBump is checked only afterwards,
      so a DeepBump configuration error cannot discard the expensive albedo stage.
    """
    albedo, t_albedo = run_albedo_one(
        args.data_root,
        args.image_root,
        args.output_root,
        rel,
        blender=args.blender,
        texture_views=args.texture_views,
        texture_resolution=args.texture_resolution,
        texture_size=args.texture_size,
        texture_steps=args.texture_steps,
        texture_strength=args.texture_strength,
        texture_guidance_scale=args.texture_guidance_scale,
        texture_controlnet_scale=args.texture_controlnet_scale,
        seed=args.seed,
        overwrite=force_overwrite,
    )

    deepbump_python = args.deepbump_python or _detect_deepbump_python()
    if deepbump_python is None:
        raise FileNotFoundError(
            "Cannot locate a Python interpreter for DeepBump. "
            "Activate hunyuan3domni or pass --deepbump-python."
        )
    deepbump_python = Path(deepbump_python).resolve()
    print(f"[DEEPBUMP] Python: {deepbump_python}")
    print(f"[DEEPBUMP] Root  : {args.deepbump_dir}")

    pbr_dir, t_deepbump = run_deepbump_one(
        args.output_root,
        rel,
        deepbump_dir=args.deepbump_dir,
        deepbump_python=deepbump_python,
        overwrite=force_overwrite,
    )

    model, t_pbr = run_pbr_one(
        args.output_root,
        rel,
        blender=args.blender,
        pbr_normal_strength=args.pbr_normal_strength,
        pbr_bump_strength=args.pbr_bump_strength,
        pbr_bump_distance=args.pbr_bump_distance,
        roughness=args.roughness,
        metallic=args.metallic,
        overwrite=force_overwrite,
    )

    return {
        "albedo": str(albedo),
        "pbr_dir": str(pbr_dir),
        "model": str(model),
        "mesh_to_albedo_sec": t_albedo,
        "deepbump_sec": t_deepbump,
        "apply_pbr_sec": t_pbr,
        "texture_pipeline_sec": t_albedo + t_deepbump + t_pbr,
    }


def run_finalize_stage(
    data_root,
    output_root,
    blender,
    seed,
    render_size,
    overwrite,
    continue_on_error,
):
    print("\n=== STAGE 6 / SIZE CHECK + FOUR VIEWS ===")
    finalize_dataset(
        data_root=data_root,
        output_root=output_root,
        blender_exe=blender,
        blender_script=SRC_ROOT / "blender_render_views.py",
        seed=seed,
        render_size=render_size,
        overwrite=overwrite,
        render=True,
        continue_on_error=continue_on_error,
    )


def finalize_textured_sample(args, rel: Path, *, overwrite: bool = True):
    """
    Finalize one sample immediately after the textured/PBR model.glb exists.

    Competition output produced beside model.glb:
        model.glb
        metadata.json
        process.md
        renders/
            front.png
            side.png
            top.png
            perspective.png

    The renderer reads the FINAL model.glb, so these four views include the
    albedo/normal/height PBR material applied in Stage 5.
    """
    out_dir = args.output_root / rel
    prompt_path = args.data_root / rel / "prompt.json"
    model_path = out_dir / "model.glb"

    _require_file(prompt_path, "prompt.json")
    _require_file(model_path, "final textured model.glb")

    print("\n=== STAGE 6 / FINAL TEXTURED FOUR-VIEW RENDER + METADATA ===")
    report = finalize_one(
        prompt_path=prompt_path,
        output_dir=out_dir,
        blender_exe=args.blender,
        blender_script=SRC_ROOT / "blender_render_views.py",
        seed=args.seed,
        render_size=args.render_size,
        overwrite=overwrite,
        render=True,
    )

    render_dir = out_dir / "renders"
    expected = [
        render_dir / "front.png",
        render_dir / "side.png",
        render_dir / "top.png",
        render_dir / "perspective.png",
    ]
    for path in expected:
        _require_file(path, f"competition render {path.name}")

    _require_file(out_dir / "metadata.json", "metadata.json")
    _require_file(out_dir / "process.md", "process.md")

    print("[SUBMISSION OUTPUT]")
    print(f"  model       : {model_path}")
    print(f"  metadata    : {out_dir / 'metadata.json'}")
    print(f"  process     : {out_dir / 'process.md'}")
    print(f"  front       : {render_dir / 'front.png'}")
    print(f"  side        : {render_dir / 'side.png'}")
    print(f"  top         : {render_dir / 'top.png'}")
    print(f"  perspective : {render_dir / 'perspective.png'}")
    return report


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "GPT-Image-2 -> Hunyuan3D -> mesh_to_albedo -> "
            "DeepBump -> Blender PBR -> finalize"
        )
    )

    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument(
        "--stage",
        choices=[
            "all",
            "image",
            "mesh",
            "albedo",
            "deepbump",
            "pbr",
            "texture",
            "finalize",
        ],
        default="all",
        help=(
            "'texture' means albedo -> DeepBump -> PBR -> final textured "
            "four-view renders + metadata/process files."
        ),
    )

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inference-steps", type=int, default=20)
    p.add_argument("--octree-resolution", type=int, default=256)
    p.add_argument("--render-size", type=int, default=1024)

    # New albedo-generation settings.
    p.add_argument(
        "--texture-views",
        nargs="+",
        default=["front", "right", "back", "left", "top", "bottom"],
    )
    p.add_argument("--texture-resolution", type=int, default=512)
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--texture-steps", type=int, default=20)
    p.add_argument("--texture-strength", type=float, default=0.78)
    p.add_argument("--texture-guidance-scale", type=float, default=6.5)
    p.add_argument("--texture-controlnet-scale", type=float, default=1.05)

    # DeepBump.
    p.add_argument(
        "--deepbump-dir",
        type=Path,
        default=DEFAULT_DEEPBUMP_DIR,
    )
    p.add_argument(
        "--deepbump-python",
        type=Path,
        default=None,
        help=(
            "Python executable used to run DeepBump. "
            "If omitted, the current Python interpreter is used "
            "(normally the active hunyuan3domni environment). "
            "DEEPBUMP_PYTHON can still override it."
        ),
    )

    # Blender PBR attachment.
    p.add_argument("--pbr-normal-strength", type=float, default=0.45)
    p.add_argument("--pbr-bump-strength", type=float, default=0.10)
    p.add_argument("--pbr-bump-distance", type=float, default=0.04)
    p.add_argument("--roughness", type=float, default=0.28)
    p.add_argument("--metallic", type=float, default=0.0)

    p.add_argument("--blender", default="blender")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")

    p.add_argument(
        "--single-test",
        action="store_true",
        help="Run exactly one full sample: the first prompt under data-root.",
    )
    p.add_argument(
        "--rerun-texture",
        action="store_true",
        help=(
            "Force rerun of the complete textured submission tail: "
            "mesh_to_albedo -> DeepBump -> PBR -> final four-view renders "
            "+ metadata.json + process.md. "
            "With --single-test it affects only the first sample."
        ),
    )

    p.add_argument(
        "--rerun-after-mesh",
        action="store_true",
        help=(
            "Force rerun of every stage after an existing Hunyuan3D-Omni geometry: "
            "mesh_to_albedo -> DeepBump -> PBR -> finalize/four renders. "
            "Never reruns GPT-Image, bbox inference, or Hunyuan3D. "
            "With --single-test it affects only the first sample."
        ),
    )

    p.add_argument(
        "--rerun-deepbump",
        action="store_true",
        help=(
            "Force ONLY the DeepBump stage using an existing texture/albedo.png. "
            "Never reruns GPT-Image, Hunyuan3D, or mesh_to_albedo. "
            "With --single-test it affects only the first sample."
        ),
    )
    p.add_argument(
        "--apply-pbr-after-deepbump",
        action="store_true",
        help=(
            "When used with --rerun-deepbump, also reapply the regenerated "
            "normal/height maps to model.glb through Blender PBR."
        ),
    )

    return p.parse_args()


def run_single_test(args):
    rel = _first_sample_relative(args.data_root)
    prompt_path = args.data_root / rel / "prompt.json"
    prompt_data = json.loads(prompt_path.read_text(encoding="utf-8-sig"))
    out_dir = args.output_root / rel

    print("=" * 76)
    print("SINGLE SAMPLE FULL-PIPELINE TEST")
    print("=" * 76)
    print(f"sample   : {rel.as_posix()}")
    print(f"prompt   : {prompt_data.get('prompt') or prompt_data.get('prompt_en', '')}")
    print(f"output   : {out_dir}")
    print(f"views    : {' '.join(args.texture_views)}")
    print(f"SD steps : {args.texture_steps}")
    print("=" * 76)

    total_start = time.perf_counter()
    stage_times: Dict[str, float] = {}

    print("\n[1/6] GPT-Image-2")
    t0 = time.perf_counter()
    run_image_stage(
        args.data_root,
        args.image_root,
        args.output_root,
        args.overwrite,
        False,
        rel,
    )
    stage_times["image"] = time.perf_counter() - t0
    print(f"[TIME] image: {stage_times['image']:.2f}s")

    print("\n[2/6] BBox inference + Hunyuan3D-Omni")
    t0 = time.perf_counter()
    run_mesh_stage(
        args.data_root,
        args.image_root,
        args.output_root,
        args.seed,
        args.inference_steps,
        args.octree_resolution,
        args.overwrite,
        False,
        rel,
    )
    stage_times["mesh"] = time.perf_counter() - t0
    print(f"[TIME] mesh: {stage_times['mesh']:.2f}s")

    control_path = out_dir / "bbox_inferred.json"
    if control_path.exists():
        control = json.loads(control_path.read_text(encoding="utf-8-sig"))
        print(f"[BBOX] bbox    = {control.get('bbox')}")
        print(f"[BBOX] size_mm = {control.get('size_mm')}")

    texture_result = run_texture_pipeline_one(
        args,
        rel,
        force_overwrite=args.overwrite,
    )
    stage_times["mesh_to_albedo"] = texture_result["mesh_to_albedo_sec"]
    stage_times["deepbump"] = texture_result["deepbump_sec"]
    stage_times["apply_pbr"] = texture_result["apply_pbr_sec"]

    print("\n[6/6] Finalize / size check + four rendered views")
    t0 = time.perf_counter()
    report = finalize_one(
        prompt_path=prompt_path,
        output_dir=out_dir,
        blender_exe=args.blender,
        blender_script=SRC_ROOT / "blender_render_views.py",
        seed=args.seed,
        render_size=args.render_size,
        overwrite=args.overwrite,
        render=True,
    )
    stage_times["finalize"] = time.perf_counter() - t0

    total = time.perf_counter() - total_start
    stage_times["total"] = total

    # Store a simple, self-contained summary for debugging the one-shot test.
    summary = {
        "sample": rel.as_posix(),
        "success": True,
        "size_check": bool(report.overall_ok),
        "outputs": {
            "image": str(args.image_root / rel / "image.png"),
            "geometry": str(out_dir / "model_geometry.glb"),
            "albedo": str(out_dir / "texture" / "albedo.png"),
            "normal": str(out_dir / "texture" / "pbr" / "normal.png"),
            "height": str(out_dir / "texture" / "pbr" / "height.png"),
            "curvature": str(out_dir / "texture" / "pbr" / "curvature.png"),
            "model": str(out_dir / "model.glb"),
        },
        "seconds": stage_times,
    }
    summary_path = out_dir / "single_test_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("SINGLE TEST FINISHED")
    print("=" * 76)
    print(f"size check : {'PASS' if report.overall_ok else 'FAIL'}")
    print(f"geometry   : {out_dir / 'model_geometry.glb'}")
    print(f"albedo     : {out_dir / 'texture' / 'albedo.png'}")
    print(f"normal     : {out_dir / 'texture' / 'pbr' / 'normal.png'}")
    print(f"height     : {out_dir / 'texture' / 'pbr' / 'height.png'}")
    print(f"final GLB  : {out_dir / 'model.glb'}")
    print(f"summary    : {summary_path}")
    print(f"total      : {total:.2f}s / {total/60:.2f}min")
    print("=" * 76)


def _iter_sample_relatives(data_root: Path):
    for prompt_path in sorted(data_root.glob("*/*/prompt.json"), key=lambda p: p.as_posix()):
        yield prompt_path.parent.relative_to(data_root)


def _geometry_source_for_after_mesh_rerun(out_dir: Path) -> Path:
    """
    Resolve the raw Hunyuan geometry used as the starting point for
    --rerun-after-mesh.

    Preferred:
        model_geometry.glb

    Backward-compatible fallback:
        model.glb, but ONLY when there is no marker showing that model.glb has
        already been replaced by a PBR result. This prevents accidentally using
        a previously textured model as the new raw geometry source.
    """
    geometry = out_dir / "model_geometry.glb"
    if geometry.exists():
        return geometry

    final_model = out_dir / "model.glb"
    pbr_marker = out_dir / "texture" / "pbr_applied.json"

    if pbr_marker.exists():
        raise FileNotFoundError(
            "model_geometry.glb is missing, while texture/pbr_applied.json exists. "
            "The current model.glb may already be the textured PBR result, so it "
            "cannot safely be treated as raw Hunyuan geometry. Restore the original "
            f"Omni geometry to: {geometry}"
        )

    _require_file(final_model, "existing Hunyuan model.glb")
    shutil.copy2(final_model, geometry)
    print(f"[GEOMETRY] created raw geometry backup for rerun: {geometry}")
    return geometry


def run_after_mesh_from_args(args, continue_on_error):
    """
    Force rerun of ALL stages after Hunyuan3D-Omni geometry.

    Skipped completely:
        Stage 1  GPT-Image-2
        Stage 2  BBox inference + Hunyuan3D-Omni

    Rerun forcibly:
        Stage 3  mesh_to_albedo
        Stage 4  DeepBump
        Stage 5  Blender PBR -> model.glb
        Stage 6  size check + four rendered views

    Required existing inputs per sample:
        inputs/.../<sample>/image.png
        data/.../<sample>/prompt.json
        outputs/.../<sample>/model_geometry.glb

    For older outputs, model.glb can be promoted to model_geometry.glb only when
    no PBR marker exists yet.
    """
    rels = (
        [_first_sample_relative(args.data_root)]
        if args.single_test
        else list(_iter_sample_relatives(args.data_root))
    )

    print("=" * 76)
    print("FORCED RERUN AFTER EXISTING HUNYUAN3D-OMNI GEOMETRY")
    print(
        "sample(s): "
        + (rels[0].as_posix() if len(rels) == 1 else f"{len(rels)} samples")
    )
    print("skip     : GPT-Image / BBox inference / Hunyuan3D")
    print("rerun    : mesh_to_albedo -> DeepBump -> PBR -> finalize")
    print("=" * 76)

    total_start = time.perf_counter()
    completed = 0
    failed = []

    for rel in rels:
        try:
            print(f"\n--- {rel.as_posix()} ---")
            out_dir = args.output_root / rel
            prompt_path = args.data_root / rel / "prompt.json"
            image_path = args.image_root / rel / "image.png"

            _require_file(prompt_path, "prompt.json")
            _require_file(image_path, "reference image")
            geometry_path = _geometry_source_for_after_mesh_rerun(out_dir)
            print(f"[START] raw Omni geometry: {geometry_path}")

            # Stages 3-5: force regeneration regardless of existing texture/PBR files.
            texture_result = run_texture_pipeline_one(
                args,
                rel,
                force_overwrite=True,
            )

            # Stage 6: force size report + front/side/top/perspective re-render.
            print("\n=== STAGE 6 / SIZE CHECK + FOUR VIEWS ===")
            t0 = time.perf_counter()
            report = finalize_one(
                prompt_path=prompt_path,
                output_dir=out_dir,
                blender_exe=args.blender,
                blender_script=SRC_ROOT / "blender_render_views.py",
                seed=args.seed,
                render_size=args.render_size,
                overwrite=True,
                render=True,
            )
            finalize_sec = time.perf_counter() - t0

            completed += 1
            print("\n[AFTER-MESH DONE]")
            print(f"  sample          : {rel.as_posix()}")
            print(f"  geometry        : {geometry_path}")
            print(f"  albedo          : {texture_result['albedo']}")
            print(f"  final model     : {texture_result['model']}")
            print(f"  size check      : {'PASS' if report.overall_ok else 'FAIL'}")
            print(f"  texture pipeline: {texture_result['texture_pipeline_sec']:.2f}s")
            print(f"  finalize        : {finalize_sec:.2f}s")

        except Exception as exc:
            failed.append((rel, str(exc)))
            print(f"[ERROR] after-mesh {rel.as_posix()}: {exc}")
            if not continue_on_error:
                raise

    total = time.perf_counter() - total_start
    print("\n" + "=" * 76)
    print("RERUN AFTER MESH FINISHED")
    print(f"completed : {completed}")
    print(f"failed    : {len(failed)}")
    print(f"total     : {total:.2f}s / {total/60:.2f}min")
    if failed:
        for rel, message in failed:
            print(f" - {rel.as_posix()}: {message}")
    print("=" * 76)
    return 1 if failed else 0


def run_texture_only_from_args(args, continue_on_error):
    """
    Force rerun of the complete textured submission tail:

        Stage 3 mesh_to_albedo
        Stage 4 DeepBump
        Stage 5 PBR -> final model.glb
        Stage 6 render final textured model + metadata/process.md

    This deliberately renders AFTER Stage 5, so renders/front.png, side.png,
    top.png and perspective.png contain the texture of the final model.glb.
    """
    rels = (
        [_first_sample_relative(args.data_root)]
        if args.single_test
        else list(_iter_sample_relatives(args.data_root))
    )

    print("=" * 76)
    print("FORCED TEXTURE + FINAL FOUR-VIEW RERUN")
    print(
        "sample(s): "
        + (
            rels[0].as_posix()
            if len(rels) == 1
            else f"{len(rels)} samples"
        )
    )
    print("rerun    : mesh_to_albedo -> DeepBump -> PBR -> four renders")
    print("=" * 76)

    total_start = time.perf_counter()
    failed = []

    for rel in rels:
        try:
            print(f"--- {rel.as_posix()} ---")

            run_texture_pipeline_one(
                args,
                rel,
                force_overwrite=True,
            )

            # IMPORTANT: render only after PBR has replaced model.glb with the
            # final textured model.
            finalize_textured_sample(
                args,
                rel,
                overwrite=True,
            )

        except Exception as exc:
            failed.append((rel, str(exc)))
            print(f"[ERROR] {rel.as_posix()}: {exc}")
            if not continue_on_error:
                raise

    total = time.perf_counter() - total_start
    print(f"[TIME] texture + renders rerun total: {total:.2f}s / {total/60:.2f}min")

    if failed:
        print("[FAILED]")
        for rel, message in failed:
            print(f" - {rel.as_posix()}: {message}")

    return 1 if failed else 0


def run_deepbump_only_from_args(args, continue_on_error):
    """
    Force rerun of DeepBump only.

    Required existing input per sample:
        outputs/.../<sample>/texture/albedo.png

    Outputs regenerated:
        texture/pbr/normal.png
        texture/pbr/height.png
        texture/pbr/curvature.png

    Optional:
        --apply-pbr-after-deepbump
    also regenerates the final model.glb from the new maps.
    """
    rels = (
        [_first_sample_relative(args.data_root)]
        if args.single_test
        else list(_iter_sample_relatives(args.data_root))
    )

    deepbump_python = args.deepbump_python or _detect_deepbump_python()
    if deepbump_python is None:
        raise FileNotFoundError(
            "Cannot locate a Python interpreter for DeepBump. "
            "Activate hunyuan3domni or pass --deepbump-python."
        )
    deepbump_python = Path(deepbump_python).resolve()

    print("=" * 76)
    print("FORCED DEEPBUMP-ONLY RERUN")
    print(
        "sample(s): "
        + (rels[0].as_posix() if len(rels) == 1 else f"{len(rels)} samples")
    )
    print(f"python   : {deepbump_python}")
    print(f"deepbump : {args.deepbump_dir}")
    print(f"reapply PBR: {args.apply_pbr_after_deepbump}")
    print("=" * 76)

    total_start = time.perf_counter()

    for rel in rels:
        try:
            print(f"\\n--- {rel.as_posix()} ---")

            pbr_dir, deepbump_sec = run_deepbump_one(
                args.output_root,
                rel,
                deepbump_dir=args.deepbump_dir,
                deepbump_python=deepbump_python,
                overwrite=True,
            )

            print(
                f"[TIME] DeepBump {rel.as_posix()}: "
                f"{deepbump_sec:.2f}s / {deepbump_sec/60:.2f}min"
            )

            if args.apply_pbr_after_deepbump:
                _, pbr_sec = run_pbr_one(
                    args.output_root,
                    rel,
                    blender=args.blender,
                    pbr_normal_strength=args.pbr_normal_strength,
                    pbr_bump_strength=args.pbr_bump_strength,
                    pbr_bump_distance=args.pbr_bump_distance,
                    roughness=args.roughness,
                    metallic=args.metallic,
                    overwrite=True,
                )
                print(
                    f"[TIME] Reapply PBR {rel.as_posix()}: "
                    f"{pbr_sec:.2f}s / {pbr_sec/60:.2f}min"
                )

        except Exception as exc:
            print(f"[ERROR] DeepBump {rel.as_posix()}: {exc}")
            if not continue_on_error:
                raise

    total = time.perf_counter() - total_start
    print(
        f"[TIME] DeepBump-only rerun total: "
        f"{total:.2f}s / {total/60:.2f}min"
    )
    return 0


def main():
    args = parse_args()
    continue_on_error = not args.stop_on_error

    args.data_root = args.data_root.resolve()
    args.image_root = args.image_root.resolve()
    args.output_root = args.output_root.resolve()
    args.deepbump_dir = args.deepbump_dir.resolve()
    if args.deepbump_python is not None:
        args.deepbump_python = args.deepbump_python.resolve()

    # Highest priority: explicit rerun modes must never trigger GPT/Hunyuan again.
    if args.rerun_deepbump:
        return run_deepbump_only_from_args(args, continue_on_error)

    if args.rerun_after_mesh:
        return run_after_mesh_from_args(args, continue_on_error)

    if args.rerun_texture:
        return run_texture_only_from_args(args, continue_on_error)

    # Single test is explicitly a full one-sample end-to-end run.
    if args.single_test:
        run_single_test(args)
        return 0

    print("=" * 76)
    print("Text-to-3D competition pipeline")
    print(f"data     : {args.data_root}")
    print(f"inputs   : {args.image_root}")
    print(f"outputs  : {args.output_root}")
    print(f"stage    : {args.stage}")
    print("=" * 76)

    pipeline_start = time.perf_counter()

    if args.stage in ("all", "image"):
        t0 = time.perf_counter()
        run_image_stage(
            args.data_root,
            args.image_root,
            args.output_root,
            args.overwrite,
            continue_on_error,
        )
        e = time.perf_counter() - t0
        print(f"[TIME] Stage image: {e:.2f}s / {e/60:.2f}min")

    if args.stage in ("all", "mesh"):
        t0 = time.perf_counter()
        run_mesh_stage(
            args.data_root,
            args.image_root,
            args.output_root,
            args.seed,
            args.inference_steps,
            args.octree_resolution,
            args.overwrite,
            continue_on_error,
        )
        e = time.perf_counter() - t0
        print(f"[TIME] Stage mesh: {e:.2f}s / {e/60:.2f}min")

    # Dataset-level texture pipeline.
    #
    # --stage all:
    #   Stage 6 is still handled once by the normal finalize block below.
    #
    # --stage texture:
    #   render the FINAL textured model immediately so the output directory
    #   already satisfies the competition submission layout.
    if args.stage in ("all", "texture"):
        for rel in _iter_sample_relatives(args.data_root):
            try:
                run_texture_pipeline_one(
                    args,
                    rel,
                    force_overwrite=args.overwrite,
                )

                if args.stage == "texture":
                    finalize_textured_sample(
                        args,
                        rel,
                        overwrite=args.overwrite,
                    )

            except Exception as exc:
                print(f"[ERROR] texture {rel.as_posix()}: {exc}")
                if not continue_on_error:
                    raise

    elif args.stage == "albedo":
        for rel in _iter_sample_relatives(args.data_root):
            try:
                run_albedo_one(
                    args.data_root,
                    args.image_root,
                    args.output_root,
                    rel,
                    blender=args.blender,
                    texture_views=args.texture_views,
                    texture_resolution=args.texture_resolution,
                    texture_size=args.texture_size,
                    texture_steps=args.texture_steps,
                    texture_strength=args.texture_strength,
                    texture_guidance_scale=args.texture_guidance_scale,
                    texture_controlnet_scale=args.texture_controlnet_scale,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                print(f"[ERROR] albedo {rel.as_posix()}: {exc}")
                if not continue_on_error:
                    raise

    elif args.stage == "deepbump":
        deepbump_python = args.deepbump_python or _detect_deepbump_python()
        if deepbump_python is None:
            raise FileNotFoundError(
                "DeepBump Python not found. Activate hunyuan3domni "
                "or pass --deepbump-python."
            )
        deepbump_python = Path(deepbump_python).resolve()
        print(f"[DEEPBUMP] Python: {deepbump_python}")
        print(f"[DEEPBUMP] Root  : {args.deepbump_dir}")
        for rel in _iter_sample_relatives(args.data_root):
            try:
                run_deepbump_one(
                    args.output_root,
                    rel,
                    deepbump_dir=args.deepbump_dir,
                    deepbump_python=deepbump_python,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                print(f"[ERROR] deepbump {rel.as_posix()}: {exc}")
                if not continue_on_error:
                    raise

    elif args.stage == "pbr":
        for rel in _iter_sample_relatives(args.data_root):
            try:
                run_pbr_one(
                    args.output_root,
                    rel,
                    blender=args.blender,
                    pbr_normal_strength=args.pbr_normal_strength,
                    pbr_bump_strength=args.pbr_bump_strength,
                    pbr_bump_distance=args.pbr_bump_distance,
                    roughness=args.roughness,
                    metallic=args.metallic,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                print(f"[ERROR] pbr {rel.as_posix()}: {exc}")
                if not continue_on_error:
                    raise

    if args.stage in ("all", "finalize"):
        t0 = time.perf_counter()
        run_finalize_stage(
            args.data_root,
            args.output_root,
            args.blender,
            args.seed,
            args.render_size,
            args.overwrite,
            continue_on_error,
        )
        e = time.perf_counter() - t0
        print(f"[TIME] Stage finalize: {e:.2f}s / {e/60:.2f}min")

    total = time.perf_counter() - pipeline_start
    print(
        f"[TIME] Pipeline total: "
        f"{total:.2f}s / {total/60:.2f}min / {total/3600:.2f}h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
