#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="递归删除或移动指定目录下所有同名文件。"
    )
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help=r"要递归搜索的根目录，例如 ..\outputs\test",
    )
    p.add_argument(
        "--name",
        required=True,
        help="要处理的文件名，例如 model_adjustment.json",
    )
    p.add_argument(
        "--action",
        choices=["delete", "move"],
        required=True,
        help="delete=删除；move=移动到 trash",
    )
    p.add_argument(
        "--trash",
        type=Path,
        default=Path("../../trash"),
        help="move 模式的目标根目录，默认 ../../trash",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要进行的操作，不真正删除/移动",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="move 时如果目标文件已存在，覆盖目标文件",
    )
    return p.parse_args()


def find_targets(root: Path, filename: str) -> list[Path]:
    return sorted(
        [p for p in root.rglob(filename) if p.is_file() and p.name == filename],
        key=lambda p: p.as_posix(),
    )


def delete_file(path: Path, *, dry_run: bool):
    if dry_run:
        print(f"[DRY-RUN][DELETE] {path}")
        return
    path.unlink()
    print(f"[DELETED] {path}")


def move_file(
    path: Path,
    *,
    root: Path,
    trash: Path,
    dry_run: bool,
    overwrite: bool,
):
    relative = path.relative_to(root)
    target = trash / relative

    if target.exists():
        if not overwrite:
            print(f"[SKIP] target exists: {target}")
            return "skipped"

        if dry_run:
            print(f"[DRY-RUN][OVERWRITE] {target}")
        else:
            if target.is_file() or target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)

    if dry_run:
        print(f"[DRY-RUN][MOVE] {path}")
        print(f"               -> {target}")
        return "dry-run"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))

    print(f"[MOVED] {path}")
    print(f"        -> {target}")
    return "moved"


def main() -> int:
    args = parse_args()

    root = args.root.resolve()
    trash = args.trash.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(root)

    if args.action == "move" and trash == root:
        raise ValueError("trash 不能与 root 相同")

    targets = find_targets(root, args.name)

    print("=" * 78)
    print(f"ROOT      : {root}")
    print(f"FILE NAME : {args.name}")
    print(f"ACTION    : {args.action}")
    if args.action == "move":
        print(f"TRASH     : {trash}")
    print(f"FOUND     : {len(targets)}")
    print(f"DRY RUN   : {args.dry_run}")
    print("=" * 78)

    deleted = 0
    moved = 0
    skipped = 0
    failed = []

    for i, path in enumerate(targets, start=1):
        print(f"\n[{i}/{len(targets)}]")
        try:
            if args.action == "delete":
                delete_file(path, dry_run=args.dry_run)
                if not args.dry_run:
                    deleted += 1
            else:
                result = move_file(
                    path,
                    root=root,
                    trash=trash,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
                if result == "moved":
                    moved += 1
                elif result == "skipped":
                    skipped += 1
        except Exception as exc:
            failed.append((path, str(exc)))
            print(f"[FAILED] {path}: {exc}")

    print("\n" + "=" * 78)
    if args.dry_run:
        print(f"Would process : {len(targets)}")
    else:
        print(f"Deleted       : {deleted}")
        print(f"Moved         : {moved}")
        print(f"Skipped       : {skipped}")
    print(f"Failed        : {len(failed)}")
    print("=" * 78)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
