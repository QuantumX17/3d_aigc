#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hunyuan3D-Omni Windows environment checker.

Recommended placement:
    <PROJECT_ROOT>/src/check_omni_env_windows.py

Usage:
    python src/check_omni_env_windows.py

Optional model-load test (does NOT run generation):
    python src/check_omni_env_windows.py --load-model

Custom project root:
    python check_omni_env_windows.py --project-root D:\3D_AIGC
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import os
import platform
import shutil
import sys
import traceback
from pathlib import Path


RECOMMENDED_PYTHON = (3, 10)
RECOMMENDED_TORCH = "2.5.1"
RECOMMENDED_TORCHVISION = "0.20.1"
RECOMMENDED_TORCHAUDIO = "2.5.1"
RECOMMENDED_CUDA = "12.4"
OFFICIAL_VRAM_GB = 10.0

PACKAGE_CHECKS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("transformers", "transformers"),
    ("diffusers", "diffusers"),
    ("accelerate", "accelerate"),
    ("pytorch-lightning", "pytorch_lightning"),
    ("deepspeed", "deepspeed"),
    ("timm", "timm"),
    ("torchdiffeq", "torchdiffeq"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("einops", "einops"),
    ("pandas", "pandas"),
    ("opencv-python", "cv2"),
    ("imageio", "imageio"),
    ("scikit-image", "skimage"),
    ("rembg", "rembg"),
    ("trimesh", "trimesh"),
    ("pymeshlab", "pymeshlab"),
    ("pygltflib", "pygltflib"),
    ("xatlas", "xatlas"),
    ("open3d", "open3d"),
    ("omegaconf", "omegaconf"),
    ("PyYAML", "yaml"),
    ("configargparse", "configargparse"),
    ("tqdm", "tqdm"),
    ("psutil", "psutil"),
    ("cupy-cuda12x", "cupy"),
    ("onnxruntime", "onnxruntime"),
]


class Report:
    def __init__(self) -> None:
        self.failures = []
        self.warnings = []

    def ok(self, label: str, value: str = "") -> None:
        suffix = f": {value}" if value else ""
        print(f"[OK]    {label}{suffix}")

    def warn(self, label: str, value: str = "") -> None:
        suffix = f": {value}" if value else ""
        print(f"[WARN]  {label}{suffix}")
        self.warnings.append(f"{label}: {value}".rstrip(": "))

    def fail(self, label: str, value: str = "") -> None:
        suffix = f": {value}" if value else ""
        print(f"[FAIL]  {label}{suffix}")
        self.failures.append(f"{label}: {value}".rstrip(": "))


def version_of(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def try_import(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def check_basic(report: Report) -> None:
    print("\n=== 1. OS / Python ===")
    report.ok("OS", f"{platform.system()} {platform.release()} ({platform.machine()})")
    report.ok("Python executable", sys.executable)
    report.ok("Python version", platform.python_version())

    if sys.version_info[:2] == RECOMMENDED_PYTHON:
        report.ok("Python matches official tested version", "3.10")
    else:
        report.warn(
            "Python differs from official tested version",
            f"current={platform.python_version()}, recommended=3.10.x",
        )

    if platform.system() == "Windows":
        report.warn(
            "Native Windows",
            "official repository is not fully Windows-oriented; custom runner is safer than official inference.py",
        )


def check_torch(report: Report) -> None:
    print("\n=== 2. PyTorch / CUDA / GPU ===")
    torch, err = try_import("torch")
    if torch is None:
        report.fail("torch import", err or "unknown error")
        return

    report.ok("torch", torch.__version__)
    if str(torch.__version__).startswith(RECOMMENDED_TORCH):
        report.ok("torch version matches official recommendation", RECOMMENDED_TORCH)
    else:
        report.warn("torch version", f"{torch.__version__}; official={RECOMMENDED_TORCH}")

    cuda_build = getattr(torch.version, "cuda", None)
    report.ok("PyTorch CUDA build", str(cuda_build))
    if cuda_build != RECOMMENDED_CUDA:
        report.warn("CUDA build differs from official recommendation", f"current={cuda_build}, official=12.4")

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        report.fail("torch.cuda.is_available()", "False")
        return
    report.ok("torch.cuda.is_available()", "True")

    try:
        count = torch.cuda.device_count()
        report.ok("CUDA device count", str(count))
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            report.ok(f"GPU {i}", f"{props.name}; VRAM={vram_gb:.2f} GB; capability={props.major}.{props.minor}")
            if i == 0 and vram_gb < OFFICIAL_VRAM_GB:
                report.warn(
                    "VRAM below official Omni generation figure",
                    f"{vram_gb:.2f} GB available vs ~{OFFICIAL_VRAM_GB:.0f} GB stated by official repo; OOM is likely without offload/low-VRAM changes",
                )
    except Exception as exc:
        report.fail("GPU property query", f"{type(exc).__name__}: {exc}")

    try:
        x = torch.randn((512, 512), device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        del x, y
        torch.cuda.empty_cache()
        report.ok("CUDA compute smoke test", "FP16 matrix multiplication succeeded")
    except Exception as exc:
        report.fail("CUDA compute smoke test", f"{type(exc).__name__}: {exc}")


def check_packages(report: Report) -> None:
    print("\n=== 3. Python dependencies ===")
    for dist_name, module_name in PACKAGE_CHECKS:
        version = version_of(dist_name)
        if version is None:
            report.warn(dist_name, "not installed")
            continue

        module, err = try_import(module_name)
        if module is None:
            report.fail(dist_name, f"installed version={version}, but import failed: {err}")
        else:
            report.ok(dist_name, version)


def check_windows_specific(report: Report, repo_root: Path) -> None:
    print("\n=== 4. Windows-specific compatibility ===")

    if platform.system() != "Windows":
        report.ok("Platform", "not Windows; Windows-specific checks skipped")
        return

    try:
        ctypes.CDLL("libgcc_s.so.1")
        report.ok("libgcc_s.so.1", "loadable")
    except Exception as exc:
        report.warn(
            "libgcc_s.so.1",
            "not available on native Windows. Official inference.py loads it unconditionally; do not use official inference.py unchanged.",
        )

    inference_py = repo_root / "inference.py"
    if inference_py.exists():
        try:
            text = inference_py.read_text(encoding="utf-8", errors="ignore")
            if "libgcc_s.so.1" in text:
                report.warn(
                    "official inference.py",
                    "contains ctypes.CDLL('libgcc_s.so.1'); native Windows execution will fail at startup unless patched",
                )
            else:
                report.ok("official inference.py", "no libgcc_s.so.1 reference found")
        except Exception as exc:
            report.warn("inference.py scan", str(exc))

    nvcc = shutil.which("nvcc")
    if nvcc:
        report.ok("nvcc", nvcc)
    else:
        report.warn(
            "nvcc",
            "not found in PATH. PyTorch itself can still run CUDA, but packages that JIT/build CUDA extensions may fail.",
        )

    cl = shutil.which("cl")
    if cl:
        report.ok("MSVC cl.exe", cl)
    else:
        report.warn(
            "MSVC cl.exe",
            "not found in PATH. If DeepSpeed/CUDA extensions need compilation, launch a VS Developer Command Prompt or install VS2022 C++ Build Tools.",
        )


def check_project(report: Report, project_root: Path) -> tuple[Path, Path]:
    print("\n=== 5. Project / repository ===")
    repo_root = project_root / "Hunyuan3D-Omni"
    model_root = project_root / "weights" / "Hunyuan3D-Omni"

    report.ok("Project root", str(project_root))

    if not repo_root.exists():
        report.fail("Hunyuan3D-Omni repository", f"missing: {repo_root}")
    else:
        report.ok("Hunyuan3D-Omni repository", str(repo_root))

    required_repo_files = [
        repo_root / "hy3dshape",
        repo_root / "requirements.txt",
        repo_root / "inference.py",
    ]
    for path in required_repo_files:
        if path.exists():
            report.ok("repo item", str(path))
        else:
            report.fail("repo item missing", str(path))

    if model_root.exists():
        file_count = sum(1 for p in model_root.rglob("*") if p.is_file())
        total_gb = sum(p.stat().st_size for p in model_root.rglob("*") if p.is_file()) / (1024 ** 3)
        report.ok("Local Omni weights", f"{model_root}; files={file_count}; size={total_gb:.2f} GB")
    else:
        report.warn(
            "Local Omni weights",
            f"missing: {model_root}; runner will need Hugging Face download/cache",
        )

    return repo_root, model_root


def check_omni_import(report: Report, repo_root: Path) -> None:
    print("\n=== 6. Hunyuan3D-Omni import ===")
    if not repo_root.exists():
        report.fail("Omni import", "repository missing")
        return

    sys.path.insert(0, str(repo_root))
    try:
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline  # noqa: F401
        report.ok(
            "Hunyuan3DOmniSiTFlowMatchingPipeline import",
            "success",
        )
    except Exception as exc:
        report.fail(
            "Hunyuan3DOmniSiTFlowMatchingPipeline import",
            f"{type(exc).__name__}: {exc}",
        )
        print("\n--- traceback ---")
        traceback.print_exc()
        print("--- end traceback ---")


def check_runner_import(report: Report, project_root: Path) -> None:
    print("\n=== 7. Custom runner import ===")
    runner_path = project_root / "src" / "image_to_mesh_omni.py"
    if not runner_path.exists():
        report.warn("image_to_mesh_omni.py", f"missing: {runner_path}")
        return

    src_dir = runner_path.parent
    sys.path.insert(0, str(src_dir))
    try:
        module = importlib.import_module("image_to_mesh_omni")
        report.ok("image_to_mesh_omni import", str(runner_path))
        if hasattr(module, "ImageToMeshRunner"):
            report.ok("ImageToMeshRunner", "found")
        else:
            report.fail("ImageToMeshRunner", "not found in module")
    except Exception as exc:
        report.fail("image_to_mesh_omni import", f"{type(exc).__name__}: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()
        print("--- end traceback ---")


def optional_model_load(report: Report, repo_root: Path, model_root: Path) -> None:
    print("\n=== 8. Optional model-load test ===")
    if not repo_root.exists():
        report.fail("Model load", "repository missing")
        return

    sys.path.insert(0, str(repo_root))
    try:
        import torch
        from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline

        model_source = str(model_root) if model_root.exists() else "tencent/Hunyuan3D-Omni"
        print(f"[INFO]  Loading model from: {model_source}")
        pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
            model_source,
            fast_decode=False,
        )
        report.ok("Omni from_pretrained", "success")

        del pipeline
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        report.fail("Omni from_pretrained", f"{type(exc).__name__}: {exc}")
        print("\n--- traceback ---")
        traceback.print_exc()
        print("--- end traceback ---")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check native Windows environment for Hunyuan3D-Omni.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root containing Hunyuan3D-Omni/, weights/, src/. "
             "Default: parent of this script's directory.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Actually call from_pretrained(). This may download large weights and consume substantial RAM/VRAM.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else script_path.parents[1]

    report = Report()

    print("=" * 78)
    print("Hunyuan3D-Omni Windows Environment Checker")
    print("=" * 78)

    check_basic(report)
    check_torch(report)
    check_packages(report)
    repo_root, model_root = check_project(report, project_root)
    check_windows_specific(report, repo_root)
    check_omni_import(report, repo_root)
    check_runner_import(report, project_root)

    if args.load_model:
        optional_model_load(report, repo_root, model_root)
    else:
        print("\n=== 8. Optional model-load test ===")
        print("[SKIP]  Use --load-model after all import checks pass.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Failures : {len(report.failures)}")
    print(f"Warnings : {len(report.warnings)}")

    if report.failures:
        print("\nRESULT: NOT READY")
        print("Fix the [FAIL] items first.")
        return 2

    if report.warnings:
        print("\nRESULT: BASIC IMPORTS MAY WORK, BUT ENVIRONMENT IS NOT GUARANTEED FOR FULL OMNI INFERENCE.")
        print("Review [WARN] items. On an 8 GB GPU, full generation is especially likely to OOM.")
        return 1

    print("\nRESULT: BASIC ENVIRONMENT CHECK PASSED.")
    print("Next run with --load-model, then perform one low-resolution generation test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())