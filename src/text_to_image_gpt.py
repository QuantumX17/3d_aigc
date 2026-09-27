#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
text_to_image_gpt.py

Competition JSON -> OpenAI-compatible Newcoin gateway -> clean Image-to-Mesh reference image.

No background removal / matting is included.

Default API:
    https://api.newcoin.top/v1

Default model:
    gpt-image-2

Environment:
    $env:NEWCOIN_API_KEY="YOUR_NEWCOIN_API_KEY"

Install:
    pip install requests

Usage:
    python text_to_image_gpt.py D:\3D_AIGC\inputs\test.json

    python text_to_image_gpt.py D:\3D_AIGC\inputs\test.json ^
        --output-dir D:\3D_AIGC\outputs_gpt

    python text_to_image_gpt.py D:\3D_AIGC\inputs ^
        --output-dir D:\3D_AIGC\outputs_gpt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from timing_utils import update_timing

try:
    import requests
except ImportError:
    requests = None


DEFAULT_API_URL = "https://api.newcoin.top/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "1024x1024"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_API_KEY_ENV = "NEWCOIN_API_KEY"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object.")

    prompt = str(data.get("prompt", "")).strip()
    prompt_en = str(data.get("prompt_en", "")).strip()

    if not prompt and not prompt_en:
        raise ValueError(
            f"{path}: at least one of 'prompt' or 'prompt_en' must be non-empty."
        )

    constraints = data.get("constraints", {})
    if constraints is not None and not isinstance(constraints, dict):
        raise ValueError(f"{path}: 'constraints' must be an object.")

    return data


def safe_name(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "sample"

    for ch in '<>:"/\\|?*':
        value = value.replace(ch, "_")

    value = "_".join(value.split())
    value = value.strip(" ._")
    return value[:120] or "sample"


def bool_constraint(constraints: Dict[str, Any], key: str) -> bool:
    return bool(constraints.get(key) is True)


def build_image_prompt(data: Dict[str, Any]) -> str:
    prompt_zh = str(data.get("prompt", "")).strip()
    prompt_en = str(data.get("prompt_en", "")).strip()
    category = str(data.get("category", "")).strip()
    intent = str(data.get("prompt_intent", "")).strip()
    constraints = data.get("constraints") or {}

    semantic_lines: List[str] = []

    if prompt_zh:
        semantic_lines.append(f"原始描述：{prompt_zh}")
    if prompt_en:
        semantic_lines.append(f"英文参考：{prompt_en}")
    if category:
        semantic_lines.append(f"类别：{category}")
    if intent:
        semantic_lines.append(f"用途：{intent}")

    geometry_rules: List[str] = []

    if bool_constraint(constraints, "single_component"):
        geometry_rules.append(
            "Keep the subject as one continuous, connected component."
        )

    if bool_constraint(constraints, "no_floating_parts"):
        geometry_rules.append(
            "No floating, hovering, or disconnected parts; props must be attached, held, or supported."
        )

    if bool_constraint(constraints, "stable_base"):
        geometry_rules.append(
            "Provide a clear stable ground-contact area; add only a small connected base/support if necessary."
        )

    if bool_constraint(constraints, "watertight_required"):
        geometry_rules.append(
            "The object should appear as a closed, watertight solid without open shells or fractured surfaces."
        )

    if bool_constraint(constraints, "size_constrained"):
        geometry_rules.append(
            "Strictly preserve explicit dimensions and proportion constraints from the description."
        )

    if not geometry_rules:
        geometry_rules.append(
            "Keep the geometry complete, connected, and suitable for FDM-printable 3D reconstruction."
        )

    return f"""
Generate a clean front-view reference image for single-image Image-to-Mesh / Image-to-3D.

【Subject】
{chr(10).join(semantic_lines)}

Faithfully preserve the requested subject, action, pose, shape, key props, and distinctive details.
Represent it as a real three-dimensional sculpture, figurine, maquette, or product model suitable for physical production, not a scene photo or 2D illustration.

【Composition】
- One primary 3D object only.
- Front view, centered, camera approximately at object mid-height.
- Entire object fully visible with no cropping.
- Let the object fill most of the frame with minimal empty space.
- Clear outer silhouette and clearly separated major structural regions.
- Strong sense of real volume and depth; avoid flat illustration appearance.
- Avoid extreme perspective, top-down view, low-angle view, or fisheye distortion.
- Do not add unrelated characters, animals, props, duplicated objects, or alternative designs.
- Thin structures such as wires, rods, ropes, hair, spikes, or sheets should be moderately thickened when necessary for reliable 3D reconstruction.

【Environment】
Environmental words such as ocean, beach, sky, forest, road, room, street, mountains, river, or landscape should NOT become a full scene.
Keep only the main object. If an environmental element is essential for recognizability or stability, reduce it to a small connected support/base.

【Geometry / Printability】
{chr(10).join("- " + x for x in geometry_rules)}
- Unless the description explicitly requires separate parts, keep the result as one unified object.
- Do not add contents or accessories that are not requested.
- Reveal internal structure only when the prompt explicitly asks for assembly, cutaway, exploded, transparent-shell, or visible internal components.

【Texture / Material】
- Preserve clear intrinsic base colors and distinct semantic/material regions.
- Use clean, coherent, moderately sized surface patterns suitable for UV texture projection.
- Prefer diffuse or moderately rough materials.
- Avoid strong gloss, mirror reflections, transparency, glass, holographic materials, or metallic appearance unless explicitly requested.
- Avoid tiny repetitive patterns, dense text, micro-details, random scratches, and noisy decals.
- Avoid baked shadows, strong ambient occlusion, rim lighting, colored lighting, dramatic highlights, and strong cast shadows.
- Use soft, broad, neutral studio illumination so visible colors remain close to true material colors.

【Background】
- Pure white to very light neutral-gray seamless studio background.
- No environment, ground texture, scenery, text, title, logo, watermark, border, or UI.
- Only a very subtle soft contact shadow directly beneath the object is allowed.

Final result: a clean, complete, connected, FDM-friendly single-object 3D product render suitable for Image-to-Mesh and downstream UV/PBR processing.
Output the image directly.
""".strip()


def build_raw_prompt(data: Dict[str, Any]) -> str:
    prompt = str(data.get("prompt", "")).strip()
    if prompt:
        return prompt
    return str(data.get("prompt_en", "")).strip()


def iter_json_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file must be .json: {input_path}")
        yield input_path
        return

    if input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"No .json files found in directory: {input_path}"
            )
        yield from files
        return

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def request_generation(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    output_format: str,
    watermark: bool,
    timeout: float,
) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("Missing dependency: requests. Install with: pip install requests")

    url = api_url.rstrip("/") + "/images/generations"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
    }

    # If the gateway supports these fields, uncomment them:
    # payload["output_format"] = output_format
    # payload["watermark"] = watermark

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"API request failed: HTTP {response.status_code}\n{detail}"
        )

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"API returned non-JSON response: {response.text[:1000]}"
        ) from exc


def extract_image_payload(response_json: Dict[str, Any]) -> Dict[str, str]:
    data = response_json.get("data")

    if not isinstance(data, list) or not data:
        raise RuntimeError(
            "API response does not contain a non-empty 'data' list.\n"
            f"Response: {json.dumps(response_json, ensure_ascii=False)[:2000]}"
        )

    first = data[0]
    if not isinstance(first, dict):
        raise RuntimeError("Unexpected API response item format.")

    url = first.get("url")
    b64_json = first.get("b64_json")

    if url:
        return {"type": "url", "value": str(url)}
    if b64_json:
        return {"type": "b64_json", "value": str(b64_json)}

    raise RuntimeError(
        "API response contains neither data[0].url nor data[0].b64_json.\n"
        f"Response: {json.dumps(response_json, ensure_ascii=False)[:2000]}"
    )


def download_image(url: str, output_path: Path, timeout: float) -> None:
    if requests is None:
        raise RuntimeError("Missing dependency: requests. Install with: pip install requests")

    response = requests.get(url, timeout=timeout)
    if not response.ok:
        raise RuntimeError(
            f"Failed to download generated image: HTTP {response.status_code}"
        )
    output_path.write_bytes(response.content)


def save_base64_image(b64_json: str, output_path: Path) -> None:
    try:
        output_path.write_bytes(base64.b64decode(b64_json))
    except Exception as exc:
        raise RuntimeError("Failed to decode b64_json image response.") from exc


def generate_one(
    json_path: Path,
    output_dir: Path,
    api_url: str,
    api_key: str,
    model: str,
    size: str,
    output_format: str,
    watermark: bool,
    n: int,
    raw_prompt: bool,
    overwrite: bool,
    timeout: float,
) -> List[Path]:
    data = load_json(json_path)

    prompt_id = safe_name(data.get("prompt_id") or json_path.stem)
    final_prompt = build_raw_prompt(data) if raw_prompt else build_image_prompt(data)

    sample_dir = output_dir / prompt_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    extension = "jpg" if output_format == "jpeg" else output_format

    expected_outputs = [
        sample_dir / f"{prompt_id}_{i:02d}.{extension}"
        for i in range(1, n + 1)
    ]

    if not overwrite and all(path.exists() for path in expected_outputs):
        print(f"[SKIP] {prompt_id}: {n} image(s) already exist.")
        return expected_outputs

    print(f"\n[GENERATE] {prompt_id}")
    print(f"  model      : {model}")
    print(f"  size       : {size}")
    print(f"  format     : {output_format}")
    print(f"  watermark  : {watermark}")
    print(f"  variants   : {n}")
    print("  matting    : disabled")

    saved: List[Path] = []
    response_records: List[Dict[str, Any]] = []

    for i in range(1, n + 1):
        print(f"  request    : {i}/{n}")

        result = request_generation(
            api_url=api_url,
            api_key=api_key,
            model=model,
            prompt=final_prompt,
            size=size,
            output_format=output_format,
            watermark=watermark,
            timeout=timeout,
        )
        response_records.append(result)

        payload = extract_image_payload(result)

        output_path = sample_dir / f"{prompt_id}_{i:02d}.{extension}"

        if payload["type"] == "url":
            download_image(payload["value"], output_path, timeout=timeout)
        else:
            save_base64_image(payload["value"], output_path)

        saved.append(output_path)
        print(f"[OK] {output_path}")

    prompt_path = sample_dir / "effective_prompt.txt"
    prompt_path.write_text(final_prompt, encoding="utf-8")

    metadata = {
        "source_json": str(json_path),
        "prompt_id": prompt_id,
        "provider": "newcoin",
        "api_url": api_url,
        "model": model,
        "size": size,
        "output_format": output_format,
        "watermark": watermark,
        "variants": n,
        "prompt_mode": "raw" if raw_prompt else "image_to_mesh",
        "background_removal": False,
        "effective_prompt_file": str(prompt_path),
        "outputs": [str(x) for x in saved],
        "generated_at_unix": int(time.time()),
        "input": data,
        "requests": response_records,
    }

    metadata_path = sample_dir / "generation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return saved


class TextToImageRunner:
    """
    Compatibility wrapper for main.py.

    It provides:
      - __init__(...)
      - process_dataset(...)

    main.py expects:
      runner = TextToImageRunner()
      runner.process_dataset(data_root=..., output_root=..., ...)
    """
    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        model: str = DEFAULT_MODEL,
        size: str = DEFAULT_SIZE,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        watermark: bool = False,
        timeout: float = 180.0,
    ):
        self.api_url = api_url
        self.api_key_env = api_key_env
        self.model = model
        self.size = size
        self.output_format = output_format
        self.watermark = watermark
        self.timeout = timeout

        self.api_key = os.getenv(self.api_key_env, "").strip()

    def _get_api_key(self) -> str:
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(
                f"Environment variable '{self.api_key_env}' is empty. "
                f"Set it before running, e.g.:\n"
                f'    $env:{self.api_key_env}="YOUR_NEWCOIN_API_KEY"'
            )
        return self.api_key

    def _process_single_json(
        self,
        json_path: Path,
        output_root: Path,
        overwrite: bool = False,
    ) -> Path:
        data = load_json(json_path)
        prompt_id = safe_name(data.get("prompt_id") or json_path.stem)
        final_prompt = build_image_prompt(data)

        # Keep the same directory structure as data_root
        rel_dir = json_path.parent.relative_to(json_path.parents[0]) if False else json_path.parent
        # The actual dataset root will be supplied by process_dataset and used below.
        # This placeholder is replaced by process_dataset.

        raise RuntimeError("This internal method should not be called directly.")

    def process_dataset(
        self,
        data_root: str | Path,
        output_root: str | Path,
        prompt_name: str = "prompt.json",
        image_name: str = "image.png",
        overwrite: bool = False,
        continue_on_error: bool = True,
        timing_output_root: str | Path | None = None,
        sample_relative: str | Path | None = None,
    ) -> List[Path]:
        data_root = Path(data_root).expanduser().resolve()
        output_root = Path(output_root).expanduser().resolve()
        timing_root = (
            Path(timing_output_root).expanduser().resolve()
            if timing_output_root is not None
            else None
        )
        output_root.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {data_root}")

        api_key: str | None = None

        generated_paths: List[Path] = []

        for json_path in sorted(data_root.rglob(prompt_name)):
            relative_sample = json_path.parent.relative_to(data_root)
            if sample_relative is not None and relative_sample != Path(sample_relative):
                continue
            target_dir = output_root / relative_sample
            target_dir.mkdir(parents=True, exist_ok=True)

            output_path = target_dir / image_name

            if output_path.exists() and not overwrite:
                generated_paths.append(output_path)
                continue

            try:
                generation_start = time.perf_counter()
                if api_key is None:
                    api_key = self._get_api_key()
                data = load_json(json_path)
                final_prompt = build_image_prompt(data)

                result = request_generation(
                    api_url=self.api_url,
                    api_key=api_key,
                    model=self.model,
                    prompt=final_prompt,
                    size=self.size,
                    output_format=self.output_format,
                    watermark=self.watermark,
                    timeout=self.timeout,
                )

                payload = extract_image_payload(result)

                if payload["type"] == "url":
                    download_image(payload["value"], output_path, timeout=self.timeout)
                else:
                    save_base64_image(payload["value"], output_path)

                # metadata
                metadata = {
                    "source_json": str(json_path),
                    "provider": "newcoin",
                    "api_url": self.api_url,
                    "model": self.model,
                    "size": self.size,
                    "output_format": self.output_format,
                    "watermark": self.watermark,
                    "generated_at_unix": int(time.time()),
                    "prompt": final_prompt,
                    "input": data,
                    "output_file": str(output_path),
                }

                meta_path = target_dir / "generation_metadata.json"
                meta_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                elapsed = time.perf_counter() - generation_start
                if timing_root is not None:
                    timing_path = timing_root / json_path.parent.relative_to(data_root) / "timing.json"
                    update_timing(timing_path, image_generation_seconds=elapsed)
                print(f"[TIME] {json_path.parent.name} image: {elapsed:.3f} s")

                generated_paths.append(output_path)

                print(f"[OK] {output_path}")

            except Exception as exc:
                if continue_on_error:
                    print(f"[WARN] Failed for {json_path}: {exc}", file=sys.stderr)
                    continue
                raise

        return generated_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Competition JSON -> OpenAI-compatible Newcoin "
            "-> clean Image-to-Mesh reference image."
        )
    )

    parser.add_argument(
        "input",
        help="Single JSON file or directory containing JSON files.",
    )

    parser.add_argument(
        "--output-dir",
        default="./outputs_gpt",
        help="Output root directory. Default: ./outputs_gpt",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Image model. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        help=f"Image size. Default: {DEFAULT_SIZE}",
    )

    parser.add_argument(
        "--output-format",
        choices=["png", "jpeg"],
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output image format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )

    parser.add_argument(
        "--watermark",
        action="store_true",
        help="Enable watermark. Default: disabled.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of variants per JSON. Default: 1",
    )

    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help=(
            "Send original prompt directly without Image-to-Mesh "
            "prompt enhancement."
        ),
    )

    parser.add_argument(
        "--prompt-extend",
        action="store_true",
        help=(
            "Compatibility flag. Image-to-Mesh prompt enhancement "
            "is already enabled by default."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if expected output files exist.",
    )

    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"Environment variable containing the API key. Default: {DEFAULT_API_KEY_ENV}",
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"OpenAI-compatible image generation endpoint. Default: {DEFAULT_API_URL}",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds. Default: 180",
    )

    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be >= 1.")

    if args.raw_prompt and args.prompt_extend:
        parser.error(
            "--raw-prompt and --prompt-extend cannot be used together."
        )

    return args


def print_error_hint(exc: Exception) -> None:
    text = str(exc).lower()

    print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)

    if "401" in text or "unauthorized" in text:
        print(
            f"[HINT] Check whether {DEFAULT_API_KEY_ENV} is correct.",
            file=sys.stderr,
        )
    elif "403" in text or "forbidden" in text:
        print(
            "[HINT] Check model access permission / account status.",
            file=sys.stderr,
        )
    elif "429" in text or "rate" in text:
        print(
            "[HINT] API rate limit or quota may have been reached.",
            file=sys.stderr,
        )
    elif "timeout" in text:
        print(
            "[HINT] Request timed out. Try increasing --timeout.",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()

    if requests is None:
        print(
            "[ERROR] Missing requests package.\n"
            "Install it with:\n"
            "    pip install requests",
            file=sys.stderr,
        )
        return 2

    api_key = os.getenv(args.api_key_env, "").strip()

    if not api_key:
        print(
            f"[ERROR] Environment variable {args.api_key_env} is empty.\n\n"
            "PowerShell, current terminal:\n"
            f'    $env:{args.api_key_env}="YOUR_NEWCOIN_API_KEY"\n\n'
            "Check:\n"
            f"    echo $env:{args.api_key_env}\n\n"
            "Persistent Windows user variable:\n"
            f'    setx {args.api_key_env} "YOUR_NEWCOIN_API_KEY"\n\n'
            "After setx, open a NEW PowerShell window.",
            file=sys.stderr,
        )
        return 2

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        json_files = list(iter_json_files(input_path))
    except Exception as exc:
        print_error_hint(exc)
        return 2

    failures = 0

    for json_path in json_files:
        try:
            generate_one(
                json_path=json_path,
                output_dir=output_dir,
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                size=args.size,
                output_format=args.output_format,
                watermark=args.watermark,
                n=args.n,
                raw_prompt=args.raw_prompt,
                overwrite=args.overwrite,
                timeout=args.timeout,
            )
        except Exception as exc:
            failures += 1
            print(f"\n[FAILED] {json_path}", file=sys.stderr)
            print_error_hint(exc)

    print("\n" + "=" * 68)
    print(f"Processed : {len(json_files)}")
    print(f"Failed    : {failures}")
    print(f"Output    : {output_dir.resolve()}")
    print(f"Model     : {args.model}")
    print("Matting   : disabled")
    print("=" * 68)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
