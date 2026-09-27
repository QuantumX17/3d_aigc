#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scale one GLB along a single Blender world axis and overwrite it in place.

Blender axis convention:
  red   = X
  green = Y
  blue  = Z

Examples:

  # Red/X width -> 70%
  blender -b -P blender_scale_axis.py -- \
    --input D:\3D_AIGC\outputs\test\constrained\hid_con_010\model.glb \
    --axis x

  # Green/Y depth -> 70%
  blender -b -P blender_scale_axis.py -- \
    --input D:\3D_AIGC\outputs\test\constrained\hid_con_010\model.glb \
    --axis y

  # Custom factor
  blender -b -P blender_scale_axis.py -- \
    --input model.glb --axis x --factor 0.8

The script:
- preserves UVs/materials;
- does not remesh;
- scales around the model bounding-box center;
- overwrites the input GLB directly;
- creates no backup file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="GLB path to modify in place")
    p.add_argument(
        "--axis",
        choices=["x", "y"],
        required=True,
        help="x = Blender red axis, y = Blender green axis",
    )
    p.add_argument(
        "--factor",
        type=float,
            default=0.71,
            help="Scale factor along the selected axis. Default: 0.71",
    )
    return p.parse_args(argv_after_dashdash())


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_and_join(path: Path):
    bpy.ops.import_scene.gltf(filepath=str(path))

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh object found in: {path}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = meshes[0]

    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = "AXIS_SCALED_MODEL"

    # Bake imported rotation/scale so X/Y below correspond to Blender world axes.
    bpy.ops.object.transform_apply(
        location=False,
        rotation=True,
        scale=True,
    )

    return obj


def world_bounds(obj):
    pts = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    lo = Vector((
        min(p.x for p in pts),
        min(p.y for p in pts),
        min(p.z for p in pts),
    ))
    hi = Vector((
        max(p.x for p in pts),
        max(p.y for p in pts),
        max(p.z for p in pts),
    ))

    return lo, hi


def scale_about_bbox_center(obj, axis: str, factor: float):
    lo, hi = world_bounds(obj)
    center = (lo + hi) * 0.5

    # Move bbox center to origin.
    obj.location -= center
    bpy.ops.object.transform_apply(
        location=True,
        rotation=False,
        scale=False,
    )

    if axis == "x":
        obj.scale = (factor, 1.0, 1.0)
    elif axis == "y":
        obj.scale = (1.0, factor, 1.0)
    else:
        raise ValueError(axis)

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True,
    )

    # Restore original bbox center.
    obj.location += center
    bpy.ops.object.transform_apply(
        location=True,
        rotation=False,
        scale=False,
    )


def export_in_place(obj, path: Path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_image_format="AUTO",
        export_extras=True,
    )


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.factor <= 0:
        raise ValueError(f"--factor must be > 0, got {args.factor}")

    clear_scene()
    obj = import_and_join(input_path)

    before_lo, before_hi = world_bounds(obj)
    before_ext = before_hi - before_lo

    scale_about_bbox_center(obj, args.axis, args.factor)

    after_lo, after_hi = world_bounds(obj)
    after_ext = after_hi - after_lo

    export_in_place(obj, input_path)

    axis_name = {
        "x": "X / red",
        "y": "Y / green",
    }[args.axis]

    print("[SCALE] axis          :", axis_name)
    print("[SCALE] factor        :", args.factor)
    print(
        "[SCALE] before extents:",
        tuple(round(float(v), 6) for v in before_ext),
    )
    print(
        "[SCALE] after extents :",
        tuple(round(float(v), 6) for v in after_ext),
    )
    print("[SCALE] overwritten   :", input_path)


if __name__ == "__main__":
    main()