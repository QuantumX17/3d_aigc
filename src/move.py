#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Move every `texture/` directory under the project's `outputs/` tree into
`<project_root>/output_textrue/`, while preserving the original relative path.

Example:

    D:\3D_AIGC\outputs\test\constrained\hid_con_001\texture
        ->
    D:\3D_AIGC\output_textrue\test\constrained\hid_con_001\texture

    D:\3D_AIGC\outputs\test_dev\test\small_object\hid_sma_010\texture
        ->
    D:\3D_AIGC\output_textrue\test_dev\test\small_object\hid_sma_010\texture

By default:
- only directories whose name is exactly `texture` are moved;
- `renders/`, `model.glb`, metadata, etc. are untouched;
- existing destination `texture/` directories are NOT overwritten;
- source `texture/` is removed after a successful move.

Recommended:
    python move_all_textures.py --dry-run
    python move_all_textures.py

If this script is saved in:
    D:\3D_AIGC\src\move_all_textures.py

then defaults are:
    source = D:\3D_AIGC\outputs
    dest   = D:\3D_AIGC\output_textrue
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "Move all outputs/**/texture directories to "
            "project_root/output_textrue while preserving relative paths."
        )
    )
    p.add_argument(
        "--source",
        type=Path,
        default=project_root / "outputs",
        help="Source outputs root. Default: <project_root>/outputs",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=project_root / "output_textrue",
        help="Destination root. Default: <project_root>/output_textrue",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be moved.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If destination texture directory already exists, delete it first "
            "and then move the source texture directory."
        ),
    )
    return p.parse_args()


def find_texture_dirs(source_root: Path) -> list[Path]:
    """
    Return only top-level directories named exactly `texture`.

    Sorting deepest-first is safer if unusual nested `texture/texture` paths exist.
    """
    found = [
        p for p in source_root.rglob("texture")
        if p.is_dir() and p.name == "texture"
    ]
    return sorted(
        found,
        key=lambda p: (-len(p.parts), p.as_posix()),
    )


def move_one(
    source_dir: Path,
    source_root: Path,
    dest_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> str:
    relative = source_dir.relative_to(source_root)
    dest_dir = dest_root / relative

    print(f"[SOURCE] {source_dir}")
    print(f"[DEST]   {dest_dir}")

    if dest_dir.exists():
        if not overwrite:
            print("[SKIP] destination already exists; use --overwrite to replace it")
            return "skipped"

        if dry_run:
            print("[DRY-RUN] would remove existing destination and move source")
            return "dry-run"

        print(f"[REMOVE] existing destination: {dest_dir}")
        shutil.rmtree(dest_dir)

    if dry_run:
        print("[DRY-RUN] would move")
        return "dry-run"

    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    # shutil.move removes the source directory after successful transfer.
    shutil.move(str(source_dir), str(dest_dir))

    if source_dir.exists():
        raise RuntimeError(
            f"Move reported success but source still exists: {source_dir}"
        )
    if not dest_dir.exists():
        raise RuntimeError(
            f"Move reported success but destination is missing: {dest_dir}"
        )

    print("[MOVED]")
    return "moved"


def main() -> int:
    args = parse_args()

    source_root = args.source.resolve()
    dest_root = args.dest.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    # Prevent accidentally moving into a directory inside the source tree.
    try:
        dest_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Destination must not be inside the source outputs tree. "
            f"source={source_root}, dest={dest_root}"
        )

    texture_dirs = find_texture_dirs(source_root)

    print("=" * 78)
    print("MOVE ALL TEXTURE DIRECTORIES")
    print(f"source    : {source_root}")
    print(f"dest      : {dest_root}")
    print(f"found     : {len(texture_dirs)}")
    print(f"dry-run   : {args.dry_run}")
    print(f"overwrite : {args.overwrite}")
    print("=" * 78)

    if not texture_dirs:
        print("No texture directories found.")
        return 0

    moved = 0
    skipped = 0
    failed = []

    for index, source_dir in enumerate(texture_dirs, start=1):
        print(f"\n[{index}/{len(texture_dirs)}]")

        try:
            result = move_one(
                source_dir,
                source_root,
                dest_root,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )

            if result == "moved":
                moved += 1
            elif result == "skipped":
                skipped += 1

        except Exception as exc:
            failed.append((source_dir, str(exc)))
            print(f"[FAILED] {exc}")

    print("\n" + "=" * 78)
    if args.dry_run:
        print(f"Would process : {len(texture_dirs)}")
    else:
        print(f"Moved         : {moved}")
        print(f"Skipped       : {skipped}")
    print(f"Failed        : {len(failed)}")

    if failed:
        print("\n[FAILED ITEMS]")
        for path, message in failed:
            print(f" - {path}")
            print(f"   {message}")

    print("=" * 78)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
