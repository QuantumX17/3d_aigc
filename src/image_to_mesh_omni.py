#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from prompt_to_bbox import infer_sample_control, write_inferred_control
from timing_utils import update_timing


# Expected project layout when this file is placed in <PROJECT_ROOT>/src/:
# <PROJECT_ROOT>/
#   Hunyuan3D-Omni/
#   weights/Hunyuan3D-Omni/   # optional; HF repo id is used if absent
#   src/image_to_mesh_omni.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUNYUAN_SOURCE = PROJECT_ROOT / "Hunyuan3D-Omni"
MODEL_ROOT = PROJECT_ROOT / "weights" / "Hunyuan3D-Omni"
DEFAULT_MODEL_ID = "tencent/Hunyuan3D-Omni"

if str(HUNYUAN_SOURCE) not in sys.path:
    sys.path.insert(0, str(HUNYUAN_SOURCE))

try:
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover
except ImportError as exc:
    raise ImportError(
        "Cannot import Hunyuan3D-Omni. Clone the official repository to: "
        f"{HUNYUAN_SOURCE}"
    ) from exc

try:
    from rembg import new_session, remove as rembg_remove
except ImportError:
    new_session = None
    rembg_remove = None


DEFAULT_IMAGE_NAME = "image.png"
DEFAULT_MESH_NAME = "mesh.glb"
DEFAULT_BBOX_JSON_NAME = "bbox.json"
DEFAULT_SEED = 1234
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
# Official bbox demo uses 512. 256 is a safer default for lower-VRAM machines.
DEFAULT_OCTREE_RESOLUTION = 256
DEFAULT_NUM_CHUNKS = 8000
DEFAULT_MC_LEVEL = 0.0
DEFAULT_BBOX = (1.0, 1.0, 1.0)


BBox3 = Tuple[float, float, float]
Size3 = Tuple[float, float, float]


def _validate_positive_triplet(values: Sequence[float], name: str) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {len(values)}")
    result = tuple(float(v) for v in values)
    if any(v <= 0 for v in result):
        raise ValueError(f"{name} values must all be > 0, got {result}")
    return result  # type: ignore[return-value]


def normalize_size_to_bbox(size_xyz: Sequence[float]) -> BBox3:
    """
    Convert physical X/Y/Z dimensions to the 3-value Hunyuan3D-Omni bbox control.

    Hunyuan3D-Omni bbox control is a centered box represented by 3 relative
    dimensions in the 0..1 range. The largest requested dimension is mapped to 1.

    Example:
        80 x 60 x 120 mm -> [0.6667, 0.5, 1.0]
    """
    sx, sy, sz = _validate_positive_triplet(size_xyz, "size_xyz")
    max_dim = max(sx, sy, sz)
    return (sx / max_dim, sy / max_dim, sz / max_dim)


def validate_bbox(bbox: Sequence[float]) -> BBox3:
    """Validate Omni bbox [x_size, y_size, z_size], each in (0, 1]."""
    bx, by, bz = _validate_positive_triplet(bbox, "bbox")
    if any(v > 1.0 for v in (bx, by, bz)):
        raise ValueError(
            "Hunyuan3D-Omni bbox values are relative dimensions and should be in (0, 1]. "
            f"Got {(bx, by, bz)}"
        )
    return bx, by, bz


def exact_scale_mesh(mesh, target_size_xyz_mm: Sequence[float]):
    """Non-uniformly scale the generated mesh to exact X/Y/Z extents in mm."""
    target = np.asarray(_validate_positive_triplet(target_size_xyz_mm, "target_size_xyz_mm"), dtype=np.float64)
    current = np.asarray(mesh.extents, dtype=np.float64)
    if current.shape != (3,) or np.any(current <= 1e-12):
        raise RuntimeError(f"Invalid generated mesh extents: {current}")

    scale = target / current
    mesh.apply_scale(scale)
    return mesh


class ImageToMeshRunner:
    """
    Hunyuan3D-Omni bbox-controlled image-to-mesh runner.

    The class name is intentionally kept compatible with the previous
    Hunyuan3D-2.1 script so existing callers need minimal changes.
    """

    def __init__(
        self,
        model_root: str | Path = MODEL_ROOT,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        remove_background: bool = True,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        use_ema: bool = False,
        flashvdm: bool = False,
        remove_floaters: bool = True,
        remove_degenerate_faces: bool = True,
    ) -> None:
        self.model_root = Path(model_root)
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.remove_background = remove_background
        self.guidance_scale = guidance_scale
        self.use_ema = use_ema
        self.flashvdm = flashvdm
        self.remove_floaters = remove_floaters
        self.remove_degenerate_faces = remove_degenerate_faces

        self.pipeline: Optional[Hunyuan3DOmniSiTFlowMatchingPipeline] = None
        self.background_session = None
        self.floater_remover = FloaterRemover() if remove_floaters else None
        self.degenerate_remover = DegenerateFaceRemover() if remove_degenerate_faces else None

    @property
    def model_source(self) -> str:
        # Prefer an explicitly downloaded local model. If it is absent, Omni's
        # from_pretrained() will download the HF repository automatically.
        if self.model_root.exists():
            return str(self.model_root)
        return self.model_id

    def load(self) -> None:
        if self.pipeline is not None:
            return

        if not HUNYUAN_SOURCE.exists():
            raise FileNotFoundError(
                f"Hunyuan3D-Omni repository not found: {HUNYUAN_SOURCE}\n"
                "Clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git first."
            )

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

        print("[LOAD] Loading Hunyuan3D-Omni...")
        print(f"[LOAD] Model source: {self.model_source}")
        print(f"[LOAD] Device={self.device}, dtype={self.dtype}, EMA={self.use_ema}")

        self.pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
            self.model_source,
            variant="ema" if self.use_ema else None,
            device=self.device,
            dtype=self.dtype,
        )

        if self.remove_background:
            if new_session is None or rembg_remove is None:
                raise ImportError(
                    "remove_background=True but rembg is unavailable. Install rembg, "
                    "or run with --no-remove-bg."
                )
            self.background_session = new_session()

        print("[LOAD] Ready.")

    def unload(self) -> None:
        self.pipeline = None
        self.background_session = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as img:
            image = img.convert("RGBA")

        if self.remove_background:
            assert rembg_remove is not None
            image = rembg_remove(
                image,
                session=self.background_session,
                bgcolor=[255, 255, 255, 0],
            )
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGBA")

        return image

    def _bbox_tensor(self, bbox: Sequence[float]) -> torch.Tensor:
        assert self.pipeline is not None
        bbox3 = validate_bbox(bbox)
        # Official Omni bbox control expects shape [B, 1, 3].
        return (
            torch.tensor(bbox3, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.pipeline.device)
            .to(self.pipeline.dtype)
        )

    def generate_mesh(
        self,
        image_path: str | Path,
        output_path: str | Path,
        bbox: Sequence[float] = DEFAULT_BBOX,
        size_mm: Optional[Sequence[float]] = None,
        exact_size: bool = True,
        seed: int = DEFAULT_SEED,
        inference_steps: int = DEFAULT_INFERENCE_STEPS,
        octree_resolution: int = DEFAULT_OCTREE_RESOLUTION,
        num_chunks: int = DEFAULT_NUM_CHUNKS,
        mc_level: float = DEFAULT_MC_LEVEL,
        overwrite: bool = False,
    ) -> Path:
        image_path = Path(image_path)
        output_path = Path(output_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        if output_path.exists() and not overwrite:
            print(f"[SKIP] {output_path}")
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()
        assert self.pipeline is not None

        target_size_mm: Optional[Size3] = None
        if size_mm is not None:
            target_size_mm = _validate_positive_triplet(size_mm, "size_mm")
            bbox = normalize_size_to_bbox(target_size_mm)

        bbox3 = validate_bbox(bbox)
        bbox_tensor = self._bbox_tensor(bbox3)
        image = self._load_image(image_path)

        generator_device = self.pipeline.device.type
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        print(f"[GEN] image={image_path}")
        print(f"[GEN] bbox={bbox3}")
        if target_size_mm is not None:
            print(f"[GEN] requested exact size (mm)={target_size_mm}")

        result = self.pipeline(
            image=[image],
            bbox=bbox_tensor,
            num_inference_steps=inference_steps,
            octree_resolution=octree_resolution,
            mc_level=mc_level,
            guidance_scale=self.guidance_scale,
            generator=generator,
            num_chunks=num_chunks,
            fast_decode=self.flashvdm,
            output_type="trimesh",
        )

        try:
            mesh = result["shapes"][0][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Hunyuan3D-Omni output for {image_path}") from exc

        if mesh is None:
            raise RuntimeError(f"Hunyuan3D-Omni returned an empty mesh: {image_path}")

        if self.floater_remover is not None:
            mesh = self.floater_remover(mesh)
        if self.degenerate_remover is not None:
            mesh = self.degenerate_remover(mesh)

        if target_size_mm is not None and exact_size:
            mesh = exact_scale_mesh(mesh, target_size_mm)
            extents = np.asarray(mesh.extents, dtype=np.float64)
            print(
                "[SIZE] final extents(mm)="
                + ", ".join(f"{v:.4f}" for v in extents.tolist())
            )

        mesh.export(output_path)
        print(f"[OK] {output_path}")
        return output_path

    @staticmethod
    def iter_dev_folders(
        input_root: str | Path,
        image_name: str = DEFAULT_IMAGE_NAME,
    ) -> Iterable[Tuple[Path, Path]]:
        input_root = Path(input_root)
        if not input_root.exists():
            raise FileNotFoundError(f"Input root not found: {input_root}")

        for category_dir in sorted(input_root.iterdir()):
            if not category_dir.is_dir():
                continue
            for dev_dir in sorted(category_dir.iterdir()):
                if not dev_dir.is_dir():
                    continue
                image_path = dev_dir / image_name
                if image_path.exists():
                    yield dev_dir, image_path
                else:
                    print(f"[WARN] Missing {image_name}: {dev_dir}")

    @staticmethod
    def load_sample_control(
        dev_dir: Path,
        bbox_json_name: str = DEFAULT_BBOX_JSON_NAME,
        fallback_bbox: Sequence[float] = DEFAULT_BBOX,
        global_bbox: Optional[Sequence[float]] = None,
        global_size_mm: Optional[Sequence[float]] = None,
        require_control_file: bool = False,
    ) -> Tuple[BBox3, Optional[Size3]]:
        # CLI/global values override per-sample JSON.
        if global_size_mm is not None:
            size = _validate_positive_triplet(global_size_mm, "global_size_mm")
            return normalize_size_to_bbox(size), size
        if global_bbox is not None:
            return validate_bbox(global_bbox), None

        control_path = dev_dir / bbox_json_name
        if control_path.exists():
            with control_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if "size_mm" in data:
                size = _validate_positive_triplet(data["size_mm"], f"{control_path}: size_mm")
                return normalize_size_to_bbox(size), size
            if "bbox" in data:
                return validate_bbox(data["bbox"]), None

            raise ValueError(
                f"{control_path} must contain either 'size_mm': [x,y,z] "
                "or 'bbox': [x_ratio,y_ratio,z_ratio]."
            )

        if require_control_file:
            raise FileNotFoundError(f"Required control file not found: {control_path}")

        print(
            f"[WARN] No {bbox_json_name} in {dev_dir}; using fallback bbox "
            f"{tuple(fallback_bbox)}"
        )
        return validate_bbox(fallback_bbox), None

    def process_dataset(
        self,
        input_root: str | Path,
        output_root: str | Path,
        image_name: str = DEFAULT_IMAGE_NAME,
        mesh_name: str = DEFAULT_MESH_NAME,
        bbox_json_name: str = DEFAULT_BBOX_JSON_NAME,
        bbox: Optional[Sequence[float]] = None,
        size_mm: Optional[Sequence[float]] = None,
        exact_size: bool = True,
        seed: int = DEFAULT_SEED,
        inference_steps: int = DEFAULT_INFERENCE_STEPS,
        octree_resolution: int = DEFAULT_OCTREE_RESOLUTION,
        num_chunks: int = DEFAULT_NUM_CHUNKS,
        mc_level: float = DEFAULT_MC_LEVEL,
        overwrite: bool = False,
        continue_on_error: bool = True,
        require_control_file: bool = False,
        prompt_json_name: str = "prompt.json",
        auto_prompt_bbox: bool = False,
        align_bbox_to_image: bool = True,
        image_ambiguity_ratio: float = 1.08,
        save_inferred_control: bool = True,
        inferred_control_name: str = "bbox_inferred.json",
        data_root: str | Path | None = None,
        sample_relative: str | Path | None = None,
    ) -> List[Path]:
        input_root = Path(input_root)
        output_root = Path(output_root)
        data_root_path = Path(data_root) if data_root is not None else None

        jobs = list(self.iter_dev_folders(input_root, image_name=image_name))
        if sample_relative is not None:
            wanted = Path(sample_relative)
            jobs = [job for job in jobs if job[0].relative_to(input_root) == wanted]
        if not jobs:
            raise RuntimeError(f"No {image_name} files found under {input_root}")

        print(f"[BATCH] Found {len(jobs)} sample(s).")
        load_start = time.perf_counter()
        self.load()
        print(f"[TIME] Model loading: {time.perf_counter() - load_start:.3f} s")

        completed: List[Path] = []
        failed: List[Tuple[Path, str]] = []

        for index, (dev_dir, image_path) in enumerate(jobs, start=1):
            relative_dev = dev_dir.relative_to(input_root)
            output_path = output_root / relative_dev / mesh_name
            print(f"[{index}/{len(jobs)}] {relative_dev.as_posix()}")

            try:
                timing_path = output_path.parent / "timing.json"
                if auto_prompt_bbox:
                    if data_root_path is None:
                        raise ValueError("data_root is required when auto_prompt_bbox=True")
                    inferred_path = output_path.parent / inferred_control_name
                    if inferred_path.exists() and not overwrite:
                        sample_bbox, sample_size_mm = self.load_sample_control(
                            dev_dir=output_path.parent,
                            bbox_json_name=inferred_control_name,
                            require_control_file=True,
                        )
                        print(f"[SKIP] bbox already exists: {inferred_path}")
                    else:
                        bbox_start = time.perf_counter()
                        inferred = infer_sample_control(
                            relative_dev,
                            data_root=data_root_path,
                            image_path=image_path,
                            prompt_json_name=prompt_json_name,
                            align_to_image=align_bbox_to_image,
                            image_ambiguity_ratio=image_ambiguity_ratio,
                        )
                        sample_bbox = validate_bbox(inferred["bbox"])
                        sample_size_mm = _validate_positive_triplet(inferred["size_mm"], "inferred size_mm")
                        if save_inferred_control:
                            write_inferred_control(inferred, inferred_path)
                        bbox_elapsed = time.perf_counter() - bbox_start
                        update_timing(timing_path, bbox_inference_seconds=bbox_elapsed)
                        print(f"[TIME] {dev_dir.name} bbox: {bbox_elapsed:.3f} s")
                else:
                    sample_bbox, sample_size_mm = self.load_sample_control(
                        dev_dir=dev_dir,
                        bbox_json_name=bbox_json_name,
                        global_bbox=bbox,
                        global_size_mm=size_mm,
                        require_control_file=require_control_file,
                    )

                mesh_will_run = overwrite or not output_path.exists()
                mesh_start = time.perf_counter() if mesh_will_run else None
                result = self.generate_mesh(
                    image_path=image_path,
                    output_path=output_path,
                    bbox=sample_bbox,
                    size_mm=sample_size_mm,
                    exact_size=exact_size,
                    seed=seed,
                    inference_steps=inference_steps,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    mc_level=mc_level,
                    overwrite=overwrite,
                )
                if mesh_start is not None:
                    mesh_elapsed = time.perf_counter() - mesh_start
                    update_timing(timing_path, mesh_generation_seconds=mesh_elapsed)
                    print(f"[TIME] {dev_dir.name} mesh: {mesh_elapsed:.3f} s")
                completed.append(result)
            except Exception as exc:
                failed.append((relative_dev, str(exc)))
                print(f"[FAILED] {relative_dev}: {exc}", file=sys.stderr)
                if not continue_on_error:
                    raise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"[DONE] total={len(jobs)} completed={len(completed)} failed={len(failed)}")
        if failed:
            print("[FAILED SAMPLES]")
            for relative_dev, message in failed:
                print(f" - {relative_dev.as_posix()}: {message}")
        return completed


_DEFAULT_RUNNER: Optional[ImageToMeshRunner] = None


def get_default_runner() -> ImageToMeshRunner:
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = ImageToMeshRunner()
    return _DEFAULT_RUNNER


def generate_mesh(
    image_path: str | Path,
    output_path: str | Path,
    bbox: Sequence[float] = DEFAULT_BBOX,
    size_mm: Optional[Sequence[float]] = None,
    exact_size: bool = True,
    seed: int = DEFAULT_SEED,
    inference_steps: int = DEFAULT_INFERENCE_STEPS,
    octree_resolution: int = DEFAULT_OCTREE_RESOLUTION,
    overwrite: bool = False,
) -> Path:
    return get_default_runner().generate_mesh(
        image_path=image_path,
        output_path=output_path,
        bbox=bbox,
        size_mm=size_mm,
        exact_size=exact_size,
        seed=seed,
        inference_steps=inference_steps,
        octree_resolution=octree_resolution,
        overwrite=overwrite,
    )


def batch_generate_mesh(
    input_root: str | Path,
    output_root: str | Path,
    image_name: str = DEFAULT_IMAGE_NAME,
    mesh_name: str = DEFAULT_MESH_NAME,
    bbox_json_name: str = DEFAULT_BBOX_JSON_NAME,
    bbox: Optional[Sequence[float]] = None,
    size_mm: Optional[Sequence[float]] = None,
    exact_size: bool = True,
    seed: int = DEFAULT_SEED,
    inference_steps: int = DEFAULT_INFERENCE_STEPS,
    octree_resolution: int = DEFAULT_OCTREE_RESOLUTION,
    overwrite: bool = False,
    continue_on_error: bool = True,
) -> List[Path]:
    return get_default_runner().process_dataset(
        input_root=input_root,
        output_root=output_root,
        image_name=image_name,
        mesh_name=mesh_name,
        bbox_json_name=bbox_json_name,
        bbox=bbox,
        size_mm=size_mm,
        exact_size=exact_size,
        seed=seed,
        inference_steps=inference_steps,
        octree_resolution=octree_resolution,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hunyuan3D-Omni bbox-controlled image-to-mesh runner. "
            "Supports single image and dataset batch mode."
        )
    )
    parser.add_argument("input", help="Image path, or dataset root with --batch.")
    parser.add_argument("--output", default=None, help="Mesh path or batch output root.")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--mesh-name", default=DEFAULT_MESH_NAME)
    parser.add_argument("--bbox-json-name", default=DEFAULT_BBOX_JSON_NAME)

    control_group = parser.add_mutually_exclusive_group()
    control_group.add_argument(
        "--bbox",
        nargs=3,
        type=float,
        metavar=("X_RATIO", "Y_RATIO", "Z_RATIO"),
        help="Omni relative bbox dimensions in (0,1], e.g. --bbox 0.8 0.64 1.0",
    )
    control_group.add_argument(
        "--size-mm",
        nargs=3,
        type=float,
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help=(
            "Target physical X/Y/Z size in mm. It is converted to Omni bbox ratios, "
            "then the final mesh is scaled to exact dimensions unless --no-final-scale is used."
        ),
    )

    parser.add_argument("--no-final-scale", action="store_true")
    parser.add_argument(
        "--require-control-file",
        action="store_true",
        help="In batch mode, fail samples without bbox.json when no global --bbox/--size-mm is supplied.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--inference-steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--octree-resolution", type=int, default=DEFAULT_OCTREE_RESOLUTION)
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument("--mc-level", type=float, default=DEFAULT_MC_LEVEL)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--flashvdm", action="store_true")
    parser.add_argument("--no-remove-bg", action="store_true")
    parser.add_argument("--no-remove-floaters", action="store_true")
    parser.add_argument("--no-remove-degenerate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("HUNYUAN_SOURCE:", HUNYUAN_SOURCE)
    print("MODEL_ROOT:", args.model_root)

    runner = ImageToMeshRunner(
        model_root=args.model_root,
        model_id=args.model_id,
        device=args.device,
        remove_background=not args.no_remove_bg,
        guidance_scale=args.guidance_scale,
        use_ema=args.use_ema,
        flashvdm=args.flashvdm,
        remove_floaters=not args.no_remove_floaters,
        remove_degenerate_faces=not args.no_remove_degenerate,
    )

    input_path = Path(args.input)
    exact_size = not args.no_final_scale

    if args.batch:
        output_root = Path(args.output) if args.output else PROJECT_ROOT / "outputs_mesh_omni"
        runner.process_dataset(
            input_root=input_path,
            output_root=output_root,
            image_name=args.image_name,
            mesh_name=args.mesh_name,
            bbox_json_name=args.bbox_json_name,
            bbox=args.bbox,
            size_mm=args.size_mm,
            exact_size=exact_size,
            seed=args.seed,
            inference_steps=args.inference_steps,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
            mc_level=args.mc_level,
            overwrite=args.overwrite,
            continue_on_error=not args.stop_on_error,
            require_control_file=args.require_control_file,
        )
        return 0

    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "outputs" / f"{input_path.stem}_omni.glb"
    )

    bbox = args.bbox if args.bbox is not None else DEFAULT_BBOX
    result = runner.generate_mesh(
        image_path=input_path,
        output_path=output_path,
        bbox=bbox,
        size_mm=args.size_mm,
        exact_size=exact_size,
        seed=args.seed,
        inference_steps=args.inference_steps,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        mc_level=args.mc_level,
        overwrite=args.overwrite,
    )

    print(f"Mesh saved to: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
