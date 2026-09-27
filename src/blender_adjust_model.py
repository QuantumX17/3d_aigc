#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In-place adjust an existing GLB:
  1. Scale in competition world XYZ axes.
  2. Optionally place the model bottom at Z=0.
  3. Remove disconnected floating mesh islands conservatively.
  4. Overwrite the SAME input GLB. No backup/sidecar/output copy is created.

Run through Blender:

    blender -b -P blender_adjust_model.py -- \
      --input model.glb \
      --scale-x 0.5 --scale-y 1.0 --scale-z 1.0 \
      --ground-z

Notes:
- --output is intentionally removed. The input file is overwritten in place.
- UVs/materials are preserved for retained geometry.
- Floating cleanup only removes disconnected mesh islands. A thin plate that is
  physically connected/welded to the main mesh is NOT considered a floater.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import bpy
from mathutils import Vector


def _argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--scale-x", type=float, default=1.0)
    p.add_argument("--scale-y", type=float, default=1.0)
    p.add_argument("--scale-z", type=float, default=1.0)
    p.add_argument("--ground-z", action="store_true")

    # Floating-island cleanup is enabled by default.
    p.add_argument(
        "--no-remove-floaters",
        action="store_true",
        help="Disable disconnected floating-island cleanup.",
    )
    p.add_argument(
        "--floater-max-diag-ratio",
        type=float,
        default=0.18,
        help=(
            "Delete a disconnected component when its bbox diagonal is <= this "
            "fraction of the largest component diagonal. Default: 0.18"
        ),
    )
    p.add_argument(
        "--floater-max-vertex-ratio",
        type=float,
        default=0.08,
        help=(
            "Delete a disconnected component when its vertex count is <= this "
            "fraction of the largest component vertex count. Default: 0.08"
        ),
    )
    p.add_argument(
        "--keep-largest-only",
        action="store_true",
        help=(
            "Aggressive mode: delete every disconnected component except the "
            "largest one. Use only when the model is required to be one component."
        ),
    )
    return p.parse_args(_argv())


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_and_join(path: Path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh object found in {path}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = "COMPETITION_ADJUSTED_MODEL"

    # Make following XYZ scaling act in competition/world axes.
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
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


def _connected_vertex_components(mesh):
    """
    Return disconnected vertex-index sets using mesh edge connectivity.

    Isolated vertices are returned as one-vertex components and are normally
    removed by the conservative floater rule.
    """
    n = len(mesh.vertices)
    adjacency = [[] for _ in range(n)]

    for edge in mesh.edges:
        a, b = edge.vertices
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue

        visited[start] = True
        q = deque([start])
        comp = []

        while q:
            v = q.popleft()
            comp.append(v)
            for nb in adjacency[v]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)

        components.append(comp)

    return components


def _component_stats(mesh, indices):
    coords = [mesh.vertices[i].co for i in indices]

    lo = Vector((
        min(v.x for v in coords),
        min(v.y for v in coords),
        min(v.z for v in coords),
    ))
    hi = Vector((
        max(v.x for v in coords),
        max(v.y for v in coords),
        max(v.z for v in coords),
    ))
    ext = hi - lo
    diag = float(ext.length)

    return {
        "indices": indices,
        "vertex_count": len(indices),
        "diag": diag,
        "extents": (float(ext.x), float(ext.y), float(ext.z)),
    }


def remove_floating_islands(
    obj,
    *,
    max_diag_ratio: float,
    max_vertex_ratio: float,
    keep_largest_only: bool,
):
    mesh = obj.data

    if len(mesh.vertices) == 0:
        return 0, 0

    components = _connected_vertex_components(mesh)
    if len(components) <= 1:
        print("[FLOAT] connected components: 1; nothing to remove")
        return 0, 1

    stats = [_component_stats(mesh, c) for c in components]

    # The main component is chosen primarily by vertex count, then bbox diagonal.
    main = max(stats, key=lambda s: (s["vertex_count"], s["diag"]))
    main_vertices = max(main["vertex_count"], 1)
    main_diag = max(main["diag"], 1e-12)

    remove_indices = set()

    print(f"[FLOAT] connected components: {len(stats)}")
    print(
        "[FLOAT] main component: "
        f"vertices={main['vertex_count']} diag={main['diag']:.6f}"
    )

    for idx, stat in enumerate(stats, start=1):
        if stat is main:
            continue

        diag_ratio = stat["diag"] / main_diag
        vertex_ratio = stat["vertex_count"] / main_vertices

        if keep_largest_only:
            remove = True
            reason = "keep-largest-only"
        else:
            # Conservative rule:
            # a component is considered a floater only when it is small in BOTH
            # physical extent and mesh size relative to the main body.
            remove = (
                diag_ratio <= max_diag_ratio
                and vertex_ratio <= max_vertex_ratio
            )
            reason = (
                f"diag_ratio={diag_ratio:.4f}, "
                f"vertex_ratio={vertex_ratio:.4f}"
            )

        print(
            f"[FLOAT] component {idx}: "
            f"vertices={stat['vertex_count']} "
            f"diag={stat['diag']:.6f} "
            f"remove={remove} ({reason})"
        )

        if remove:
            remove_indices.update(stat["indices"])

    if not remove_indices:
        print("[FLOAT] no component matched floater thresholds")
        return 0, len(stats)

    # Vertex deletion also removes incident edges/faces and keeps UV/material
    # data for the remaining polygons.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

    for i in remove_indices:
        mesh.vertices[i].select = True

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    removed = len(remove_indices)
    print(f"[FLOAT] removed vertices: {removed}")
    return removed, len(stats)


def export_glb_in_place(obj, path: Path):
    """
    Export directly back to the same GLB path.

    No backup or second model file is created.
    """
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
    model_path = Path(args.input).resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    for name, value in (
        ("scale-x", args.scale_x),
        ("scale-y", args.scale_y),
        ("scale-z", args.scale_z),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")

    if not (0.0 <= args.floater_max_diag_ratio <= 1.0):
        raise ValueError("--floater-max-diag-ratio must be in [0, 1]")
    if not (0.0 <= args.floater_max_vertex_ratio <= 1.0):
        raise ValueError("--floater-max-vertex-ratio must be in [0, 1]")

    clear_scene()
    obj = import_and_join(model_path)

    before_lo, before_hi = world_bounds(obj)
    before_ext = before_hi - before_lo

    removed_vertices = 0
    component_count = 1

    if not args.no_remove_floaters:
        removed_vertices, component_count = remove_floating_islands(
            obj,
            max_diag_ratio=args.floater_max_diag_ratio,
            max_vertex_ratio=args.floater_max_vertex_ratio,
            keep_largest_only=args.keep_largest_only,
        )

    obj.scale = (args.scale_x, args.scale_y, args.scale_z)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if args.ground_z:
        lo, _ = world_bounds(obj)
        obj.location.z -= lo.z
        bpy.ops.object.transform_apply(
            location=True,
            rotation=False,
            scale=False,
        )

    after_lo, after_hi = world_bounds(obj)
    after_ext = after_hi - after_lo

    # Overwrite input GLB itself; no extra output/backup file.
    export_glb_in_place(obj, model_path)

    print("[ADJUST] before extents :", tuple(round(v, 6) for v in before_ext))
    print("[ADJUST] scale xyz      :", args.scale_x, args.scale_y, args.scale_z)
    print("[ADJUST] components     :", component_count)
    print("[ADJUST] removed verts  :", removed_vertices)
    print("[ADJUST] after extents  :", tuple(round(v, 6) for v in after_ext))
    print("[ADJUST] z_min          :", round(float(after_lo.z), 9))
    print("[ADJUST] overwritten    :", model_path)


if __name__ == "__main__":
    main()