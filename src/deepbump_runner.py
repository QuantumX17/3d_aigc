from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run local DeepBump CLI on a UV-space albedo texture."
    )
    p.add_argument("--albedo", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deepbump-dir", required=True)
    p.add_argument(
        "--python",
        required=True,
        help="Python executable inside the dedicated DeepBump conda environment",
    )
    p.add_argument(
        "--seamless",
        action="store_true",
        help="Request seamless height generation where supported",
    )
    return p.parse_args()


def run(cmd):
    print("[RUN]", subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    albedo = Path(args.albedo).resolve()
    out = Path(args.output_dir).resolve()
    deepbump_dir = Path(args.deepbump_dir).resolve()
    py = Path(args.python).resolve()
    cli = deepbump_dir / "cli.py"

    if not albedo.exists():
        raise FileNotFoundError(albedo)
    if not cli.exists():
        raise FileNotFoundError(cli)
    if not py.exists():
        raise FileNotFoundError(py)

    out.mkdir(parents=True, exist_ok=True)

    normal = out / "normal.png"
    height = out / "height.png"
    curvature = out / "curvature.png"

    timing = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    run([
        str(py), str(cli),
        str(albedo), str(normal),
        "color_to_normals",
        "--verbose",
    ])
    timing["normal_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cmd = [
        str(py), str(cli),
        str(normal), str(height),
        "normals_to_height",
        "--verbose",
    ]
    if args.seamless:
        cmd += ["--normals_to_height-seamless", "TRUE"]
    run(cmd)
    timing["height_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    run([
        str(py), str(cli),
        str(normal), str(curvature),
        "normals_to_curvature",
        "--verbose",
    ])
    timing["curvature_sec"] = time.perf_counter() - t0

    timing["total_sec"] = time.perf_counter() - total_start

    metadata = {
        "albedo": str(albedo),
        "normal": str(normal),
        "height": str(height),
        "curvature": str(curvature),
        "timing": timing,
    }
    (out / "deepbump_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print(" normal   :", normal)
    print(" height   :", height)
    print(" curvature:", curvature)
    print(" metadata :", out / "deepbump_metadata.json")


if __name__ == "__main__":
    main()
