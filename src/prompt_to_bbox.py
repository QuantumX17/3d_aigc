#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from bbox_templates import TEMPLATES, CATEGORY_DEFAULT_TEMPLATE, FDM_RULES

Number3 = Tuple[float, float, float]

UNIT_TO_MM = {
    "mm": 1.0,
    "毫米": 1.0,
    "cm": 10.0,
    "厘米": 10.0,
}

AXIS_NAMES = ("width/x", "depth/y", "height/z")

# Competition size convention: real millimetres, max bbox span in [2, 300] mm.
COMPETITION_MIN_SPAN_MM = 2.0
COMPETITION_MAX_SPAN_MM = 300.0


def _combined_text(data: Dict[str, Any]) -> str:
    return " ".join([
        str(data.get("prompt", "")),
        str(data.get("prompt_en", "")),
        str(data.get("prompt_intent", "")),
    ]).lower()


def classify_template(data: Dict[str, Any]):
    text = _combined_text(data)
    category = str(data.get("category", "")).strip()
    scores = {}

    for name, spec in TEMPLATES.items():
        score = 0.0
        if category and category in spec.get("category_hint", set()):
            score += 1.0
        for kw in spec.get("keywords", []):
            if kw.lower() in text:
                score += 1.0 + min(len(kw) / 10.0, 1.0)
        scores[name] = score

    best = max(scores, key=scores.get)
    if scores[best] <= 1.0:
        best = CATEGORY_DEFAULT_TEMPLATE.get(category, "generic_decorative")
    return best, scores


def _unit_factor(unit: str) -> float:
    key = unit.lower()
    return UNIT_TO_MM[key if key in UNIT_TO_MM else unit]


def extract_explicit_dimensions(data: Dict[str, Any]) -> Dict[str, Any]:
    text = _combined_text(data)
    out = {
        "width_mm": None,
        "depth_mm": None,
        "height_mm": None,
        "width_max_mm": None,
        "depth_max_mm": None,
        "height_max_mm": None,
        "overall_max_mm": None,
        "outer_diameter_mm": None,
        "outer_diameter_max_mm": None,
        "xyz_mm": None,
        "evidence": [],
    }

    xyz_pat = re.compile(
        r"(?P<x>\d+(?:\.\d+)?)\s*[x×*]\s*"
        r"(?P<y>\d+(?:\.\d+)?)\s*[x×*]\s*"
        r"(?P<z>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>mm|毫米|cm|厘米)",
        re.I,
    )
    m = xyz_pat.search(text)
    if m:
        f = _unit_factor(m.group("unit"))
        xyz = (
            float(m.group("x")) * f,
            float(m.group("y")) * f,
            float(m.group("z")) * f,
        )
        out["xyz_mm"] = xyz
        out["width_mm"], out["depth_mm"], out["height_mm"] = xyz
        out["evidence"].append(m.group(0))

    patterns = {
        "height_max_mm": r"(?:整体)?(?:高|高度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "width_max_mm": r"(?:宽|宽度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "depth_max_mm": r"(?:深|深度|厚|厚度)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "overall_max_mm": r"(?:整体|整体尺寸|见方|包围盒|最大跨度|最大尺寸).{0,10}?(?:不超过|不得超过|≤|<=|小于等于)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "outer_diameter_max_mm": r"(?:外径|外直径)\s*(?:不超过|不得超过|≤|<=|小于等于)\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "outer_diameter_mm": r"(?:外径|外直径)\s*(?:为|约为|约|=|:|：)?\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "height_mm": r"(?:整体)?(?:高|高度)\s*(?:为|约为|约|=|:|：)?\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "width_mm": r"(?:宽|宽度)\s*(?:为|约为|约|=|:|：)?\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
        "depth_mm": r"(?:深|深度|厚|厚度)\s*(?:为|约为|约|=|:|：)?\s*(?P<v>\d+(?:\.\d+)?)\s*(?P<u>mm|毫米|cm|厘米)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m and out[key] is None:
            out[key] = float(m.group("v")) * _unit_factor(m.group("u"))
            out["evidence"].append(m.group(0))

    return out


def normalize_bbox(size_mm: Iterable[float]) -> Number3:
    vals = tuple(float(v) for v in size_mm)
    if len(vals) != 3:
        raise ValueError(f"size_mm must have 3 values, got {len(vals)}")
    m = max(vals)
    if m <= 0:
        raise ValueError("dimensions must be positive")
    return tuple(round(v / m, 6) for v in vals)  # type: ignore[return-value]


def scale_from_ratio(ratio: Number3, axis: str, value: float) -> Number3:
    rx, ry, rz = ratio
    if axis == "x":
        s = value / rx
    elif axis == "y":
        s = value / ry
    else:
        s = value / rz
    return rx * s, ry * s, rz * s


def scale_from_horizontal_diameter(ratio: Number3, value: float) -> Number3:
    """Scale template so the horizontal max diameter max(X, Y) equals value."""
    horizontal = max(float(ratio[0]), float(ratio[1]))
    if horizontal <= 0:
        raise ValueError(f"Invalid template ratio: {ratio}")
    s = float(value) / horizontal
    return tuple(float(v) * s for v in ratio)  # type: ignore[return-value]


def apply_fdm_constraints(size: Number3, data: Dict[str, Any]) -> Number3:
    w, d, h = size
    w = max(w, FDM_RULES["min_width_mm"])
    d = max(d, FDM_RULES["min_depth_mm"])
    h = max(h, FDM_RULES["min_height_mm"])

    constraints = data.get("constraints") or {}
    if constraints.get("stable_base", False):
        w = max(w, h * FDM_RULES["stable_base_min_width_ratio_to_height"])
        d = max(d, h * FDM_RULES["stable_base_min_depth_ratio_to_height"])

    return w, d, h


# ---------------------------------------------------------------------------
# Image-guided axis alignment
# ---------------------------------------------------------------------------

def _resize_for_analysis(image: Image.Image, max_side: int = 768) -> Tuple[Image.Image, float]:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = image.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def detect_object_bbox(
    image_path: str | Path,
    *,
    max_side: int = 768,
    min_bg_distance: float = 22.0,
    quantile_clip: float = 0.003,
) -> Dict[str, Any]:
    """
    Estimate the foreground object's 2D bounding box without running a heavy model.

    Strategy:
      1. If the image has useful transparency, use alpha.
      2. Otherwise estimate the background colour from the border pixels and keep
         pixels sufficiently different from that colour.
      3. Use robust coordinate quantiles so a faint shadow or isolated noise does
         not decide the object's longest side.

    This is intended for generated product-style images such as the competition
    inputs (usually simple/white background). It is not a replacement for rembg.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as src:
        original_size = src.size
        image, scale = _resize_for_analysis(src.convert("RGBA"), max_side=max_side)

    arr = np.asarray(image, dtype=np.uint8)
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]
    h, w = alpha.shape

    # Transparent PNGs: alpha is the cleanest foreground signal.
    if int(alpha.min()) < 245 and int(alpha.max()) > 10:
        mask = alpha > 20
        method = "alpha"
        threshold = 20.0
    else:
        border = max(2, min(h, w) // 40)
        border_pixels = np.concatenate(
            [
                rgb[:border, :, :].reshape(-1, 3),
                rgb[-border:, :, :].reshape(-1, 3),
                rgb[:, :border, :].reshape(-1, 3),
                rgb[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(border_pixels, axis=0)
        border_dist = np.linalg.norm(border_pixels - bg[None, :], axis=1)
        adaptive = float(np.percentile(border_dist, 99.0) + 8.0)
        threshold = max(float(min_bg_distance), adaptive)
        dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
        mask = dist > threshold
        method = "border_color"

    ys, xs = np.nonzero(mask)
    foreground_fraction = float(mask.mean())
    if len(xs) < 32 or foreground_fraction < 1e-5 or foreground_fraction > 0.95:
        raise RuntimeError(
            "Could not obtain a reliable foreground mask "
            f"(pixels={len(xs)}, fraction={foreground_fraction:.4f})."
        )

    q = min(max(float(quantile_clip), 0.0), 0.10)
    x0 = float(np.quantile(xs, q))
    x1 = float(np.quantile(xs, 1.0 - q))
    y0 = float(np.quantile(ys, q))
    y1 = float(np.quantile(ys, 1.0 - q))

    object_w = max(1.0, x1 - x0 + 1.0)
    object_h = max(1.0, y1 - y0 + 1.0)
    aspect = object_w / object_h

    inv_scale = 1.0 / scale
    bbox_original = [
        round(x0 * inv_scale, 2),
        round(y0 * inv_scale, 2),
        round(x1 * inv_scale, 2),
        round(y1 * inv_scale, 2),
    ]
    object_size_original = [
        round(object_w * inv_scale, 2),
        round(object_h * inv_scale, 2),
    ]

    return {
        "image_path": str(image_path),
        "image_size_px": list(original_size),
        "object_bbox_px": bbox_original,
        "object_size_px": object_size_original,
        "object_aspect_ratio_w_over_h": round(aspect, 6),
        "foreground_fraction": round(foreground_fraction, 6),
        "mask_method": method,
        "mask_threshold": round(float(threshold), 3),
    }


def align_size_to_image_long_axis(
    size_mm: Sequence[float],
    image_path: str | Path,
    *,
    ambiguity_ratio: float = 1.08,
    permutation_penalty: float = 0.025,
) -> Tuple[Number3, Dict[str, Any]]:
    """
    Align only the *longest* inferred 3D side with the foreground's longest 2D
    side, while changing as little of the prompt-derived axis semantics as
    possible.

    Camera assumption used here:
      image horizontal <-> X (width)
      image vertical   <-> Z (height)
      Y is depth and is not directly observable from one front-ish image.

    If the image is clearly wider than tall, the largest size is placed on X.
    If it is clearly taller than wide, the largest size is placed on Z. This is
    done with at most one axis swap. The two remaining axes are not rearranged
    just to improve the 2D aspect ratio, because doing so could corrupt useful
    prompt semantics for depth/height. Near-square images are left unchanged.

    ``permutation_penalty`` is retained for API compatibility with an earlier
    experimental implementation and is intentionally unused.
    """
    del permutation_penalty

    vals = tuple(float(v) for v in size_mm)
    if len(vals) != 3 or any(v <= 0 for v in vals):
        raise ValueError(f"size_mm must contain 3 positive values, got {vals}")
    if ambiguity_ratio <= 1.0:
        raise ValueError("ambiguity_ratio must be > 1.0")

    image_info = detect_object_bbox(image_path)
    aspect = float(image_info["object_aspect_ratio_w_over_h"])

    if aspect >= ambiguity_ratio:
        major_axis = "x"
        target_index = 0
    elif aspect <= 1.0 / ambiguity_ratio:
        major_axis = "z"
        target_index = 2
    else:
        major_axis = "ambiguous"
        target_index = None

    before = list(vals)
    after = list(vals)
    permutation = [0, 1, 2]

    if target_index is None:
        meta = {
            **image_info,
            "major_axis": major_axis,
            "aligned": False,
            "reason": "foreground aspect is too close to square",
            "before_size_mm": [round(v, 2) for v in vals],
            "after_size_mm": [round(v, 2) for v in vals],
            "projected_bbox_aspect_x_over_z": round(vals[0] / vals[2], 6),
            "permutation": [AXIS_NAMES[i] for i in permutation],
        }
        return vals, meta

    max_v = max(vals)
    eps = max(1e-8, max_v * 1e-8)

    # If the desired visible axis is already tied for the maximum, keep all
    # axes untouched. Otherwise swap it with the first current maximum axis.
    if abs(vals[target_index] - max_v) <= eps:
        changed = False
        reason = "bbox longest side already matches image major axis"
    else:
        max_index = next(i for i, v in enumerate(vals) if abs(v - max_v) <= eps)
        after[target_index], after[max_index] = after[max_index], after[target_index]
        permutation[target_index], permutation[max_index] = (
            permutation[max_index],
            permutation[target_index],
        )
        changed = True
        reason = (
            "swapped longest side onto image horizontal X axis"
            if major_axis == "x"
            else "swapped longest side onto image vertical Z axis"
        )

    aligned = tuple(round(v, 2) for v in after)
    meta = {
        **image_info,
        "major_axis": major_axis,
        "aligned": changed,
        "reason": reason,
        "before_size_mm": [round(v, 2) for v in before],
        "after_size_mm": list(aligned),
        "projected_bbox_aspect_x_over_z": round(aligned[0] / aligned[2], 6),
        "permutation": [AXIS_NAMES[i] for i in permutation],
    }
    return aligned, meta


# ---------------------------------------------------------------------------
# Prompt inference + public callable API
# ---------------------------------------------------------------------------

def infer_dimensions(
    data: Dict[str, Any],
    image_path: Optional[str | Path] = None,
    *,
    align_to_image: bool = False,
    image_ambiguity_ratio: float = 1.08,
) -> Dict[str, Any]:
    template, scores = classify_template(data)
    spec = TEMPLATES[template]
    ratio = tuple(spec["ratio"])
    explicit = extract_explicit_dimensions(data)
    source = "template_prior"

    if explicit["xyz_mm"] is not None:
        size = tuple(explicit["xyz_mm"])
        source = "explicit_xyz"
    elif explicit["height_mm"] is not None:
        size = scale_from_ratio(ratio, "z", explicit["height_mm"])
        source = "explicit_height+template_ratio"
    elif explicit["width_mm"] is not None:
        size = scale_from_ratio(ratio, "x", explicit["width_mm"])
        source = "explicit_width+template_ratio"
    elif explicit["depth_mm"] is not None:
        size = scale_from_ratio(ratio, "y", explicit["depth_mm"])
        source = "explicit_depth+template_ratio"
    elif explicit["outer_diameter_mm"] is not None:
        size = scale_from_horizontal_diameter(ratio, explicit["outer_diameter_mm"])
        source = "explicit_outer_diameter+template_ratio"
    elif explicit["height_max_mm"] is not None:
        size = scale_from_ratio(ratio, "z", explicit["height_max_mm"] * 0.94)
        source = "height_upper_bound_94pct+template_ratio"
    elif explicit["width_max_mm"] is not None:
        size = scale_from_ratio(ratio, "x", explicit["width_max_mm"] * 0.94)
        source = "width_upper_bound_94pct+template_ratio"
    elif explicit["depth_max_mm"] is not None:
        size = scale_from_ratio(ratio, "y", explicit["depth_max_mm"] * 0.94)
        source = "depth_upper_bound_94pct+template_ratio"
    elif explicit["outer_diameter_max_mm"] is not None:
        size = scale_from_horizontal_diameter(ratio, explicit["outer_diameter_max_mm"] * 0.94)
        source = "outer_diameter_upper_bound_94pct+template_ratio"
    elif explicit["overall_max_mm"] is not None:
        base = tuple(float(v) for v in spec["default_size_mm"])
        s = (float(explicit["overall_max_mm"]) * 0.94) / max(base)
        size = tuple(v * s for v in base)
        source = "overall_upper_bound_94pct+template_ratio"
    else:
        size = tuple(spec["default_size_mm"])

    size = apply_fdm_constraints(size, data)

    scales = [1.0]
    if explicit["height_max_mm"]:
        scales.append(explicit["height_max_mm"] / size[2])
    if explicit["width_max_mm"]:
        scales.append(explicit["width_max_mm"] / size[0])
    if explicit["depth_max_mm"]:
        scales.append(explicit["depth_max_mm"] / size[1])
    if explicit["overall_max_mm"]:
        scales.append(explicit["overall_max_mm"] / max(size))
    if explicit["outer_diameter_max_mm"]:
        scales.append(explicit["outer_diameter_max_mm"] / max(size[0], size[1]))
    if max(size) > COMPETITION_MAX_SPAN_MM:
        scales.append(COMPETITION_MAX_SPAN_MM / max(size))

    final_scale = min(scales)
    if final_scale < 1.0:
        size = tuple(v * final_scale for v in size)

    size = tuple(round(v, 2) for v in size)
    image_alignment = None

    axis_locked = any(explicit.get(k) is not None for k in (
        "xyz_mm", "width_mm", "depth_mm", "height_mm",
        "width_max_mm", "depth_max_mm", "height_max_mm",
        "outer_diameter_mm", "outer_diameter_max_mm",
    ))

    if align_to_image and image_path is not None and not axis_locked:
        try:
            size, image_alignment = align_size_to_image_long_axis(
                size,
                image_path,
                ambiguity_ratio=image_ambiguity_ratio,
            )
            if image_alignment.get("aligned"):
                source += "+image_axis_alignment"
        except Exception as exc:
            # Image alignment is a guardrail, not a reason to lose an otherwise
            # usable prompt-based bbox. The caller can inspect this metadata.
            image_alignment = {
                "aligned": False,
                "image_path": str(image_path),
                "error": str(exc),
            }
    elif align_to_image and image_path is not None and axis_locked:
        image_alignment = {
            "aligned": False,
            "image_path": str(image_path),
            "reason": "disabled because prompt contains explicit axis/diameter size semantics",
        }

    # Final guard AFTER image alignment: all organizer size limits must still hold.
    final_scales = [1.0]
    if explicit["height_max_mm"]:
        final_scales.append(explicit["height_max_mm"] / size[2])
    if explicit["width_max_mm"]:
        final_scales.append(explicit["width_max_mm"] / size[0])
    if explicit["depth_max_mm"]:
        final_scales.append(explicit["depth_max_mm"] / size[1])
    if explicit["overall_max_mm"]:
        final_scales.append(explicit["overall_max_mm"] / max(size))
    if explicit["outer_diameter_max_mm"]:
        final_scales.append(explicit["outer_diameter_max_mm"] / max(size[0], size[1]))
    if max(size) > COMPETITION_MAX_SPAN_MM:
        final_scales.append(COMPETITION_MAX_SPAN_MM / max(size))
    s = min(final_scales)
    if s < 1.0:
        size = tuple(round(v * s, 2) for v in size)
    if max(size) < COMPETITION_MIN_SPAN_MM:
        s = COMPETITION_MIN_SPAN_MM / max(size)
        size = tuple(round(v * s, 2) for v in size)

    bbox = normalize_bbox(size)

    return {
        "prompt_id": data.get("prompt_id"),
        "category": data.get("category"),
        "unit": "mm",
        "coordinate_system": "Z-up",
        "template": template,
        "template_ratio": list(ratio),
        "size_mm": list(size),
        "bbox": list(bbox),
        "source": source,
        "explicit_constraints": explicit,
        "template_scores": scores,
        "image_alignment": image_alignment,
    }


def infer_dimensions_from_file(
    prompt_json_path: str | Path,
    image_path: Optional[str | Path] = None,
    *,
    align_to_image: bool = True,
    image_ambiguity_ratio: float = 1.08,
) -> Dict[str, Any]:
    """Public function for image_to_mesh_omni.py or other Python callers."""
    prompt_json_path = Path(prompt_json_path)
    if not prompt_json_path.exists():
        raise FileNotFoundError(f"Prompt JSON not found: {prompt_json_path}")
    data = json.loads(prompt_json_path.read_text(encoding="utf-8"))
    result = infer_dimensions(
        data,
        image_path=image_path,
        align_to_image=align_to_image and image_path is not None,
        image_ambiguity_ratio=image_ambiguity_ratio,
    )
    result["prompt_json_path"] = str(prompt_json_path)
    return result


def infer_sample_control(
    relative_sample: str | Path,
    *,
    data_root: str | Path = "./data",
    image_path: Optional[str | Path] = None,
    prompt_json_name: str = "prompt.json",
    align_to_image: bool = True,
    image_ambiguity_ratio: float = 1.08,
) -> Dict[str, Any]:
    """
    Resolve ./data/<category>/<dev_id>/prompt.json and infer bbox/size.

    Example:
        result = infer_sample_control(
            "functional_object/dev_fun_001",
            data_root="./data",
            image_path="./inputs/functional_object/dev_fun_001/image.png",
        )
    """
    prompt_path = Path(data_root) / Path(relative_sample) / prompt_json_name
    return infer_dimensions_from_file(
        prompt_path,
        image_path=image_path,
        align_to_image=align_to_image,
        image_ambiguity_ratio=image_ambiguity_ratio,
    )


def find_prompt_json_for_image(
    image_path: str | Path,
    *,
    data_root: str | Path = "./data",
    inputs_root: str | Path = "./inputs",
    prompt_json_name: str = "prompt.json",
) -> Optional[Path]:
    """Map an image under ./inputs to its matching prompt under ./data."""
    image_path = Path(image_path).resolve()
    data_root = Path(data_root).resolve()
    inputs_root = Path(inputs_root).resolve()

    # Preferred layout: ./inputs/<category>/<dev_id>/image.png
    #                   ./data/<category>/<dev_id>/prompt.json
    try:
        rel_parent = image_path.parent.relative_to(inputs_root)
        candidate = data_root / rel_parent / prompt_json_name
        if candidate.exists():
            return candidate
    except ValueError:
        pass

    # Fallback: find by dev_* folder name. This also supports a flatter inputs dir.
    sample_id = image_path.parent.name
    matches = list(data_root.glob(f"*/{sample_id}/{prompt_json_name}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple prompt.json files match sample id '{sample_id}': "
            + ", ".join(str(p) for p in matches)
        )
    return None


def write_inferred_control(result: Dict[str, Any], output_path: str | Path) -> Path:
    """Write a compact, image-aligned bbox control JSON for inspection/reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_id": result.get("prompt_id"),
        "category": result.get("category"),
        "unit": "mm",
        "coordinate_system": "Z-up",
        "template": result.get("template"),
        "size_mm": result.get("size_mm"),
        "bbox": result.get("bbox"),
        "source": result.get("source"),
        "image_alignment": result.get("image_alignment"),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path



class PromptToBBoxRunner:
    """
    Batch adapter used by main.py.

    It keeps bbox inference completely separate from image_to_mesh_omni.py:

        data/.../prompt.json + inputs/.../image.png
                         -> inputs/.../bbox.json

    image_to_mesh_omni.py only needs to read bbox.json afterwards.
    """

    def __init__(
        self,
        *,
        align_to_image: bool = True,
        image_ambiguity_ratio: float = 1.08,
    ) -> None:
        self.align_to_image = align_to_image
        self.image_ambiguity_ratio = image_ambiguity_ratio

    def process_sample(
        self,
        prompt_json_path: str | Path,
        image_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        result = infer_dimensions_from_file(
            prompt_json_path,
            image_path=image_path,
            align_to_image=self.align_to_image,
            image_ambiguity_ratio=self.image_ambiguity_ratio,
        )
        out = write_inferred_control(result, output_path)
        print(
            f"[BBOX] {Path(prompt_json_path).parent.name}: "
            f"size_mm={result['size_mm']} bbox={result['bbox']}"
        )
        return out

    def process_dataset(
        self,
        data_root: str | Path,
        image_root: str | Path,
        *,
        prompt_name: str = "prompt.json",
        image_name: str = "image.png",
        bbox_name: str = "bbox.json",
        overwrite: bool = False,
        continue_on_error: bool = True,
    ) -> List[Path]:
        """
        Generate one bbox.json beside each generated image.

        Expected layout:
            data/<category>/<dev>/prompt.json
            inputs/<category>/<dev>/image.png

        Output:
            inputs/<category>/<dev>/bbox.json
        """
        data_root = Path(data_root)
        image_root = Path(image_root)

        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root not found: {image_root}")

        jobs = []
        for category_dir in sorted(data_root.iterdir()):
            if not category_dir.is_dir():
                continue
            for dev_dir in sorted(category_dir.iterdir()):
                if not dev_dir.is_dir():
                    continue
                prompt_path = dev_dir / prompt_name
                if not prompt_path.exists():
                    continue
                rel = dev_dir.relative_to(data_root)
                image_path = image_root / rel / image_name
                bbox_path = image_root / rel / bbox_name
                jobs.append((rel, prompt_path, image_path, bbox_path))

        if not jobs:
            raise RuntimeError(f"No {prompt_name} files found under {data_root}")

        completed: List[Path] = []
        failed = []
        print(f"[BBOX] Found {len(jobs)} sample(s).")

        for index, (rel, prompt_path, image_path, bbox_path) in enumerate(jobs, start=1):
            print(f"[BBOX {index}/{len(jobs)}] {rel.as_posix()}")
            if bbox_path.exists() and not overwrite:
                print(f"[SKIP] {bbox_path}")
                completed.append(bbox_path)
                continue
            try:
                if not image_path.exists():
                    raise FileNotFoundError(f"Generated image not found: {image_path}")
                completed.append(
                    self.process_sample(prompt_path, image_path, bbox_path)
                )
            except Exception as exc:
                failed.append((rel, str(exc)))
                print(f"[FAILED] {rel.as_posix()}: {exc}")
                if not continue_on_error:
                    raise

        print(
            f"[BBOX DONE] total={len(jobs)} "
            f"completed={len(completed)} failed={len(failed)}"
        )
        return completed


def generate_bbox(
    prompt_json_path: str | Path,
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    align_to_image: bool = True,
    image_ambiguity_ratio: float = 1.08,
) -> Dict[str, Any]:
    """Small callable API for one sample."""
    result = infer_dimensions_from_file(
        prompt_json_path,
        image_path=image_path,
        align_to_image=align_to_image,
        image_ambiguity_ratio=image_ambiguity_ratio,
    )
    if output_path is not None:
        write_inferred_control(result, output_path)
    return result

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Infer Hunyuan3D bbox/physical dimensions from prompt.json, optionally "
            "using the input image foreground to align the bbox longest axis."
        )
    )
    parser.add_argument("input_json", help="Path to one prompt.json")
    parser.add_argument("--image", default=None, help="Matching conditioning image")
    parser.add_argument("--no-image-align", action="store_true")
    parser.add_argument("--image-ambiguity-ratio", type=float, default=1.08)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--write-back", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.input_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    result = infer_dimensions(
        data,
        image_path=args.image,
        align_to_image=(args.image is not None and not args.no_image_align),
        image_ambiguity_ratio=args.image_ambiguity_ratio,
    )

    if args.write_back:
        data["inferred_dimension"] = {
            "template": result["template"],
            "size_mm": result["size_mm"],
            "source": result["source"],
            "image_alignment": result["image_alignment"],
        }
        data["bbox"] = result["bbox"]
        out = Path(args.output) if args.output else path
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote: {out}")
    elif args.output:
        out = write_inferred_control(result, args.output)
        print(f"[OK] wrote: {out}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
