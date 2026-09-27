from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)


DEFAULT_SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_CONTROLNET = "lllyasviel/control_v11f1p_sd15_depth"
DEFAULT_VIEWS = ["front", "right", "back", "left", "top", "bottom"]

GENERIC_EN_MARKERS = (
    "an fdm-printable constrained model",
    "an fdm-printable functional object",
    "an fdm-printable functional model",
    "an fdm-printable decorative object",
    "an fdm-printable decorative model",
    "an fdm-printable small object",
    "must satisfy every stated dimension",
    "must satisfy every stated",
)

HUMAN_HINTS_ZH = (
    "人物", "人像", "头像", "脸", "面部", "渔夫", "女孩", "男孩",
    "女人", "男人", "女性", "男性", "儿童", "老人", "雕像", "半身像",
    "人物雕塑", "肖像", "人形", "小人", "角色",
)

HUMAN_HINTS_EN = (
    "person", "human", "portrait", "face", "head", "bust", "man", "woman",
    "boy", "girl", "child", "fisherman", "statue", "character", "figurine",
)

ANTI_HUMAN_CATEGORIES = {
    "constrained",
    "functional_object",
    "small_object",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mesh + reference -> conservative multi-view RGB -> UV albedo"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--prompt-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--blender", required=True)
    p.add_argument("--views", nargs="+", default=DEFAULT_VIEWS)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--strength", type=float, default=0.78)
    p.add_argument("--guidance-scale", type=float, default=6.5)
    p.add_argument("--controlnet-scale", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sd-model", default=DEFAULT_SD_MODEL)
    p.add_argument("--controlnet-model", default=DEFAULT_CONTROLNET)
    p.add_argument(
        "--negative-prompt",
        default=(
            "scene, environment, extra objects, duplicate object, text, logo, watermark, "
            "dramatic lighting, strong shadow, rim light, mirror reflection, transparent material, "
            "cropped object, deformed geometry, floating parts, blurry, low quality"
        ),
    )
    p.add_argument(
        "--novel-view-init",
        choices=["palette_depth", "depth_color", "reference"],
        default="palette_depth",
        help=(
            "palette_depth is the stable default: front is the untouched reference; "
            "novel views are simple depth silhouettes with broad reference-derived materials."
        ),
    )
    p.add_argument("--save-debug-init", action="store_true")
    return p.parse_args()


def run(cmd: List[str]) -> None:
    print("[RUN]", subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)


def _combined_text(data: dict) -> str:
    return " ".join(
        str(data.get(k, "")).strip()
        for k in ("prompt", "prompt_en", "prompt_intent", "category")
    ).lower()


def is_generic_prompt_en(prompt_en: str) -> bool:
    s = str(prompt_en or "").strip().lower()
    return (not s) or any(x in s for x in GENERIC_EN_MARKERS)


def is_human_related(data: dict) -> bool:
    text = _combined_text(data)
    return (
        any(k.lower() in text for k in HUMAN_HINTS_ZH)
        or any(k.lower() in text for k in HUMAN_HINTS_EN)
    )


def load_prompt_bundle(path: Path, cli_negative_prompt: str) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    prompt_zh = str(data.get("prompt", "")).strip()
    prompt_en = str(data.get("prompt_en", "")).strip()
    intent = str(data.get("prompt_intent", "")).strip()
    category = str(data.get("category", "")).strip()
    constraints = data.get("constraints") or {}

    generic_en = is_generic_prompt_en(prompt_en)
    human_related = is_human_related(data)
    block_human = (not human_related) and category in ANTI_HUMAN_CATEGORIES

    parts: List[str] = []
    if prompt_en and not generic_en:
        parts.append(prompt_en)
    if prompt_zh:
        parts.append(f"Original Chinese object description: {prompt_zh}")
    if intent:
        parts.append(f"intended object: {intent}")
    if category:
        parts.append(f"category: {category}")

    if constraints.get("single_component"):
        parts.append("single connected object")
    if constraints.get("no_floating_parts"):
        parts.append("no floating or disconnected parts")
    if constraints.get("stable_base"):
        parts.append("stable grounded structure")
    if constraints.get("watertight_required"):
        parts.append("closed solid printable object")

    parts.extend(
        [
            "same object identity as the reference image",
            "strictly follow the supplied depth geometry",
            "clean intrinsic colors",
            "simple coherent materials",
            "no unrelated details",
        ]
    )

    if block_human:
        parts.append("strictly non-human object, never reinterpret geometry as a human face")

    neg = str(cli_negative_prompt or "").strip()
    if block_human:
        neg += (
            ", person, human, face, head, portrait, bust, eyes, nose, mouth, hair, humanoid"
        )

    return {
        "data": data,
        "base_prompt": ", ".join(parts),
        "negative_prompt": neg,
        "category": category,
        "prompt_zh": prompt_zh,
        "prompt_en": prompt_en,
        "prompt_en_generic": generic_en,
        "human_related": human_related,
        "block_human": block_human,
    }


def square_reference(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def reference_foreground_mask(reference: Image.Image) -> np.ndarray:
    """
    Near-white studio-background mask.

    Keep this deliberately simple and deterministic; the generated competition
    references use a white/light-gray background.
    """
    arr = np.asarray(reference.convert("RGB"), dtype=np.float32)
    dist = np.linalg.norm(255.0 - arr, axis=2)
    mask = dist > 30.0

    if int(mask.sum()) < 64:
        mask = np.ones(arr.shape[:2], dtype=bool)

    return mask


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def robust_reference_bbox(reference: Image.Image):
    """
    Robust foreground bbox for front-view registration.

    A plain non-white bbox can include faint contact shadows.  Here rows/columns
    must contain a minimum amount of foreground before they affect the bbox.
    This preserves the real dark base while rejecting isolated background noise.
    """
    mask = reference_foreground_mask(reference)
    h, w = mask.shape

    row_occ = mask.sum(axis=1)
    col_occ = mask.sum(axis=0)

    active_rows = np.where(row_occ >= max(3, int(0.025 * w)))[0]
    active_cols = np.where(col_occ >= max(3, int(0.025 * h)))[0]

    if len(active_rows) < 2 or len(active_cols) < 2:
        return mask_bbox(mask)

    x0 = int(active_cols[0])
    x1 = int(active_cols[-1]) + 1
    y0 = int(active_rows[0])
    y1 = int(active_rows[-1]) + 1

    return x0, y0, x1, y1


def save_front_bbox_metadata(reference: Image.Image, out_dir: Path) -> dict:
    """
    Save the untouched reference subject bbox.

    blender_bake_views.py uses this to remap the front projection so the mesh
    camera bbox maps to the reference SUBJECT bbox, not to the whole 512 canvas.
    This is the important fix for:
      - missing dark base,
      - door/window texture shifted vertically into the wall.
    """
    bbox = robust_reference_bbox(reference)

    if bbox is None:
        bbox = (0, 0, reference.width, reference.height)

    x0, y0, x1, y1 = bbox
    w, h = reference.size

    # Blender UV uses bottom-left origin, PIL uses top-left origin.
    uv_bbox = [
        x0 / w,
        1.0 - (y1 / h),
        x1 / w,
        1.0 - (y0 / h),
    ]

    data = {
        "pixel_bbox": [x0, y0, x1, y1],
        "image_size": [w, h],
        "uv_bbox": uv_bbox,
    }

    path = out_dir / "view_front_bbox.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[FRONT BBOX] pixel={data['pixel_bbox']} uv={data['uv_bbox']}")
    return data


def reference_band_color(
    reference: Image.Image,
    y_start: float,
    y_end: float,
) -> np.ndarray:
    """
    Sample one broad material color from a fraction of the OBJECT bbox,
    not from the full 512x512 canvas.
    """
    arr = np.asarray(reference.convert("RGB"), dtype=np.float32)
    mask = reference_foreground_mask(reference)
    bbox = mask_bbox(mask)

    if bbox is None:
        return np.median(arr.reshape(-1, 3), axis=0).astype(np.float32)

    x0, y0, x1, y1 = bbox
    obj_h = max(1, y1 - y0)

    by0 = max(y0, min(y1 - 1, y0 + int(obj_h * y_start)))
    by1 = max(by0 + 1, min(y1, y0 + int(obj_h * y_end)))

    region = arr[by0:by1, x0:x1]
    region_mask = mask[by0:by1, x0:x1]
    pixels = region[region_mask]

    if len(pixels) < 16:
        pixels = arr[mask]

    if len(pixels) < 16:
        return np.array([150.0, 150.0, 150.0], dtype=np.float32)

    return np.median(pixels, axis=0).astype(np.float32)


def estimate_subject_color(reference: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(reference.convert("RGB"), dtype=np.float32)
    mask = reference_foreground_mask(reference)
    c = np.median(arr[mask], axis=0)
    c = np.clip(np.rint(c), 0, 255).astype(np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def lerp_color(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def _fill_missing_profile(profile: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = profile.copy()
    ids = np.where(valid)[0]

    if len(ids) == 0:
        return out

    # Edge fill.
    out[:ids[0]] = out[ids[0]]
    out[ids[-1] + 1:] = out[ids[-1]]

    # Linear interpolation across missing rows.
    for i in range(len(ids) - 1):
        a, b = ids[i], ids[i + 1]
        if b <= a + 1:
            continue
        for y in range(a + 1, b):
            t = (y - a) / (b - a)
            out[y] = out[a] * (1.0 - t) + out[b] * t

    return out


def _smooth_profile(profile: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Small 1-D smoothing along height only.
    No 2-D blur, so the depth silhouette stays sharp.
    """
    if radius <= 0:
        return profile

    out = np.empty_like(profile)
    n = len(profile)

    for i in range(n):
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        out[i] = np.median(profile[a:b], axis=0)

    return out


def reference_edge_profile(
    reference: Image.Image,
    side: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Extract the material color along the LEFT or RIGHT OUTER EDGE of the
    reference object as a function of height.

    This is better than using global horizontal color bands:
      - windows and doors are interior features, so they do not contaminate side walls;
      - the dark bottom base naturally propagates around the side;
      - roof/wall/base boundaries stay at the same physical height.
    """
    arr = np.asarray(reference.convert("RGB"), dtype=np.float32)
    mask = reference_foreground_mask(reference)
    bbox = robust_reference_bbox(reference)

    if bbox is None:
        bbox = (0, 0, reference.width, reference.height)

    x0, y0, x1, y1 = bbox
    obj_w = max(1, x1 - x0)
    obj_h = max(1, y1 - y0)

    profile = np.zeros((obj_h, 3), dtype=np.float32)
    valid = np.zeros(obj_h, dtype=bool)

    # Sample slightly INSIDE the silhouette. This avoids the black outline/contact
    # edge dominating every side row while still preserving actual side materials.
    inner0 = max(2, int(obj_w * 0.025))
    inner1 = max(inner0 + 2, int(obj_w * 0.10))

    for oy, y in enumerate(range(y0, y1)):
        xs = np.where(mask[y, x0:x1])[0]
        if len(xs) < 2:
            continue

        left = x0 + int(xs[0])
        right = x0 + int(xs[-1])

        if side == "left":
            a = min(right, left + inner0)
            b = min(right + 1, left + inner1)
        else:
            a = max(left, right - inner1 + 1)
            b = max(left + 1, right - inner0 + 1)

        if b <= a:
            continue

        row_mask = mask[y, a:b]
        pixels = arr[y, a:b][row_mask]

        if len(pixels) >= 1:
            profile[oy] = np.median(pixels, axis=0)
            valid[oy] = True

    profile = _fill_missing_profile(profile, valid)
    profile = _smooth_profile(profile, radius=5)

    return profile, bbox


def map_profile_to_depth(
    profile: np.ndarray,
    depth_mask: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    h, w = depth_mask.shape
    out = np.full((h, w, 3), 245.0, dtype=np.float32)

    bbox = mask_bbox(depth_mask)
    if bbox is None:
        return out

    _, y0, _, y1 = bbox
    dst_h = max(1, y1 - y0)
    src_h = max(1, len(profile))

    for y in range(y0, y1):
        row = depth_mask[y]
        if not np.any(row):
            continue

        r = (y - y0) / max(dst_h - 1, 1)
        sy = min(src_h - 1, int(round(r * (src_h - 1))))
        color = profile[sy]

        # Nearly-flat albedo. Geometry is already in the mesh.
        shade = 0.985 + 0.015 * depth[y]
        out[y, row] = color[None, :] * shade[row, None]

    return out


def palette_depth_init(
    reference: Image.Image,
    control: Image.Image,
    view: str,
) -> Image.Image:
    """
    Conservative unseen-view completion.

    left:
        extend colors from the reference LEFT OUTER EDGE by height
    right:
        extend colors from the reference RIGHT OUTER EDGE by height
    back:
        average left/right edge profiles
    top:
        use dominant upper-object material
    bottom:
        use dominant lower-object material

    This specifically prevents interior door/window colors from becoming side
    wall stripes and preserves the dark base around the object's bottom.
    """
    depth = np.asarray(control.convert("L"), dtype=np.float32) / 255.0
    mask = depth > 0.025
    h, w = depth.shape

    if view in {"left", "right", "back"}:
        left_profile, _ = reference_edge_profile(reference, "left")
        right_profile, _ = reference_edge_profile(reference, "right")

        if view == "left":
            profile = left_profile
        elif view == "right":
            profile = right_profile
        else:
            n = min(len(left_profile), len(right_profile))
            profile = (left_profile[:n] + right_profile[:n]) * 0.5

        out = map_profile_to_depth(profile, mask, depth)
        return Image.fromarray(
            np.clip(out, 0, 255).astype(np.uint8),
            mode="RGB",
        )

    out = np.full((h, w, 3), 245.0, dtype=np.float32)

    # Narrower, more material-specific bands than the previous version.
    roof = reference_band_color(reference, 0.05, 0.24)
    base = reference_band_color(reference, 0.86, 0.99)

    if view == "top":
        shade = 0.985 + 0.015 * depth
        rgb = roof[None, None, :] * shade[..., None]
        out[mask] = rgb[mask]

    elif view == "bottom":
        shade = 0.985 + 0.015 * depth
        rgb = base[None, None, :] * shade[..., None]
        out[mask] = rgb[mask]

    return Image.fromarray(
        np.clip(out, 0, 255).astype(np.uint8),
        mode="RGB",
    )


def depth_colored_init(
    control: Image.Image,
    subject_color: Tuple[int, int, int],
) -> Image.Image:
    depth = np.asarray(control.convert("L"), dtype=np.float32) / 255.0
    mask = depth > 0.025
    out = np.full((depth.shape[0], depth.shape[1], 3), 245.0, dtype=np.float32)
    color = np.asarray(subject_color, dtype=np.float32)
    shade = 0.90 + 0.10 * depth
    rgb = color[None, None, :] * shade[..., None]
    out[mask] = rgb[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def load_pipeline(sd_model: str, controlnet_model: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for diffusion fallback modes.")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    return pipe


def generate_views(
    reference: Image.Image,
    views: List[str],
    depth_dir: Path,
    out_dir: Path,
    base_prompt: str,
    negative_prompt: str,
    block_human: bool,
    pipe,
    steps: int,
    strength: float,
    guidance_scale: float,
    controlnet_scale: float,
    seed: int,
    novel_view_init: str,
    save_debug_init: bool,
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    timings: Dict[str, float] = {}
    view_debug: Dict[str, dict] = {}
    subject_color = estimate_subject_color(reference)

    debug_init_dir = out_dir.parent / "debug_init"
    if save_debug_init:
        debug_init_dir.mkdir(parents=True, exist_ok=True)

    for i, view in enumerate(views):
        depth_path = depth_dir / f"depth_{view}.png"
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)

        control = Image.open(depth_path).convert("RGB").resize(
            reference.size,
            Image.Resampling.LANCZOS,
        )

        # Critical: front is now the untouched reference again.
        # No bbox stretching/alignment and no diffusion.
        if view == "front":
            output_path = out_dir / "view_front.png"

            # Keep the pixels untouched.  Registration is done in Blender by
            # remapping the camera-projected mesh bbox to this saved subject bbox.
            reference.save(output_path)
            bbox_meta = save_front_bbox_metadata(reference, out_dir)

            timings[view] = 0.0
            view_debug[view] = {
                "mode": "direct_reference_with_projection_registration",
                "depth": str(depth_path),
                "output": str(output_path),
                "front_bbox": bbox_meta,
            }

            if save_debug_init:
                reference.save(debug_init_dir / "init_front.png")

            print("[SAVE] view_front.png <- untouched reference + bbox metadata")
            continue

        if novel_view_init == "palette_depth":
            image = palette_depth_init(reference, control, view)
            output_path = out_dir / f"view_{view}.png"
            image.save(output_path)
            timings[view] = 0.0
            view_debug[view] = {
                "mode": "palette_depth_direct",
                "depth": str(depth_path),
                "output": str(output_path),
            }
            if save_debug_init:
                image.save(debug_init_dir / f"init_{view}.png")
            print(f"[SAVE] view_{view}.png <- broad palette + depth")
            continue

        # Experimental diffusion fallback.
        if pipe is None:
            raise RuntimeError(
                f"Diffusion pipeline is required for novel_view_init={novel_view_init}"
            )

        if novel_view_init == "depth_color":
            init_image = depth_colored_init(control, subject_color)
        else:
            init_image = reference

        this_seed = seed + i
        generator = torch.Generator(device="cpu").manual_seed(this_seed)

        label = {
            "right": "right side view",
            "left": "left side view",
            "back": "back view",
            "top": "top view",
            "bottom": "bottom view",
        }.get(view, f"{view} view")

        prompt = (
            f"{base_prompt}, {label}, same exact object identity, "
            "strictly follow the depth geometry, simple material continuation, "
            "do not copy front doors or windows onto unseen surfaces"
        )
        if block_human:
            prompt += ", do not reinterpret geometry as a human face"

        t0 = time.perf_counter()
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            control_image=control,
            strength=max(float(strength), 0.82),
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=max(float(controlnet_scale), 1.20),
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        timings[view] = time.perf_counter() - t0

        output_path = out_dir / f"view_{view}.png"
        result.save(output_path)
        view_debug[view] = {
            "mode": novel_view_init,
            "seed": this_seed,
            "depth": str(depth_path),
            "output": str(output_path),
        }

    return timings, view_debug


def main() -> None:
    args = parse_args()

    input_mesh = Path(args.input).resolve()
    reference_path = Path(args.reference).resolve()
    prompt_json = Path(args.prompt_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    blender = Path(args.blender)

    for p in (input_mesh, reference_path, prompt_json):
        if not p.exists():
            raise FileNotFoundError(p)

    script_dir = Path(__file__).resolve().parent
    blender_prepare = script_dir / "blender_prepare_views.py"
    blender_bake = script_dir / "blender_bake_views.py"

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth"
    view_dir = output_dir / "views"
    depth_dir.mkdir(exist_ok=True)
    view_dir.mkdir(exist_ok=True)

    uv_mesh = output_dir / "uv_mesh.glb"
    albedo = output_dir / "albedo.png"
    textured_mesh = output_dir / "albedo_mesh.glb"

    timings: Dict[str, object] = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    run([
        str(blender),
        "-b",
        "-P",
        str(blender_prepare),
        "--",
        "--input", str(input_mesh),
        "--output-mesh", str(uv_mesh),
        "--depth-dir", str(depth_dir),
        "--resolution", str(args.resolution),
        "--views", *args.views,
    ])
    timings["prepare_depth_sec"] = time.perf_counter() - t0

    reference = square_reference(reference_path, args.resolution)
    reference_square_path = output_dir / "reference_square.png"
    reference.save(reference_square_path)

    prompt_bundle = load_prompt_bundle(prompt_json, args.negative_prompt)
    base_prompt = str(prompt_bundle["base_prompt"])
    negative_prompt = str(prompt_bundle["negative_prompt"])
    block_human = bool(prompt_bundle["block_human"])

    if args.novel_view_init == "palette_depth":
        pipe = None
        print("[TEXTURE] palette_depth: diffusion is disabled")
    else:
        pipe = load_pipeline(args.sd_model, args.controlnet_model)

    view_times, view_debug = generate_views(
        reference=reference,
        views=args.views,
        depth_dir=depth_dir,
        out_dir=view_dir,
        base_prompt=base_prompt,
        negative_prompt=negative_prompt,
        block_human=block_human,
        pipe=pipe,
        steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        controlnet_scale=args.controlnet_scale,
        seed=args.seed,
        novel_view_init=args.novel_view_init,
        save_debug_init=args.save_debug_init,
    )

    timings["diffusion_sec"] = view_times
    timings["diffusion_total_sec"] = sum(view_times.values())

    if pipe is not None:
        del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    run([
        str(blender),
        "-b",
        "-P",
        str(blender_bake),
        "--",
        "--input", str(uv_mesh),
        "--view-dir", str(view_dir),
        "--output-albedo", str(albedo),
        "--output-mesh", str(textured_mesh),
        "--texture-size", str(args.texture_size),
        "--views", *args.views,
    ])
    timings["bake_sec"] = time.perf_counter() - t0
    timings["total_sec"] = time.perf_counter() - total_start

    metadata = {
        "input_mesh": str(input_mesh),
        "reference": str(reference_path),
        "reference_square": str(reference_square_path),
        "prompt_json": str(prompt_json),
        "uv_mesh": str(uv_mesh),
        "albedo": str(albedo),
        "textured_mesh": str(textured_mesh),
        "views": args.views,
        "resolution": args.resolution,
        "texture_size": args.texture_size,
        "novel_view_init": args.novel_view_init,
        "prompt_debug": {
            "category": prompt_bundle["category"],
            "prompt_zh": prompt_bundle["prompt_zh"],
            "prompt_en": prompt_bundle["prompt_en"],
            "prompt_en_generic": prompt_bundle["prompt_en_generic"],
            "human_related": prompt_bundle["human_related"],
            "block_human": prompt_bundle["block_human"],
        },
        "view_debug": view_debug,
        "timing": timings,
    }

    (output_dir / "mesh_to_albedo_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print("  UV mesh      :", uv_mesh)
    print("  Albedo       :", albedo)
    print("  Textured mesh:", textured_mesh)


if __name__ == "__main__":
    main()
