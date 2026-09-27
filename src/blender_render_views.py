#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the four competition views of one GLB with Blender in headless mode.

Run through Blender, not normal Python:

    blender -b -P blender_render_views.py -- \
        --input model.glb --output-dir renders
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--size", type=int, default=1024)
    return p.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def import_glb(path: Path) -> list[bpy.types.Object]:
    bpy.ops.import_scene.gltf(filepath=str(path))
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"No mesh objects found in {path}")
    return objs


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    pts: list[Vector] = []
    for obj in objects:
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ Vector(corner))

    if not pts:
        raise RuntimeError("Cannot calculate model bounds")

    lo = Vector((
        min(v.x for v in pts),
        min(v.y for v in pts),
        min(v.z for v in pts),
    ))
    hi = Vector((
        max(v.x for v in pts),
        max(v.y for v in pts),
        max(v.z for v in pts),
    ))
    return lo, hi


def add_camera() -> bpy.types.Object:
    data = bpy.data.cameras.new("CompetitionCamera")
    cam = bpy.data.objects.new("CompetitionCamera", data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def point_camera(cam: bpy.types.Object, location: Vector, target: Vector) -> None:
    cam.location = location
    direction = target - location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_sun(name: str, rotation: tuple[float, float, float], energy: float) -> None:
    data = bpy.data.lights.new(name=name, type="SUN")
    data.energy = energy
    obj = bpy.data.objects.new(name, data)
    obj.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(obj)


def setup_scene(render_size: int) -> None:
    scene = bpy.context.scene

    # Fast, deterministic headless rendering.
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            pass

    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    # Neutral background. No extra floor geometry is added, so the rendered
    # views cannot disagree with the submitted model.
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.92, 0.92, 0.92, 1.0)
        bg.inputs["Strength"].default_value = 0.8

    add_sun("Key", (math.radians(35), math.radians(-25), math.radians(30)), 2.2)
    add_sun("Fill", (math.radians(65), math.radians(20), math.radians(-120)), 1.0)


def render_ortho(
    cam: bpy.types.Object,
    name: str,
    location: Vector,
    target: Vector,
    ortho_scale: float,
    output_dir: Path,
) -> None:
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(float(ortho_scale), 1e-6)
    point_camera(cam, location, target)
    bpy.context.scene.render.filepath = str(output_dir / f"{name}.png")
    bpy.ops.render.render(write_still=True)


def render_perspective(
    cam: bpy.types.Object,
    center: Vector,
    extent: Vector,
    output_dir: Path,
) -> None:
    max_dim = max(extent.x, extent.y, extent.z)
    # A mild three-quarter view with enough margin around the full asset.
    distance = max_dim * 2.8
    direction = Vector((1.25, -1.55, 1.05)).normalized()
    location = center + direction * distance

    cam.data.type = "PERSP"
    cam.data.lens = 55
    point_camera(cam, location, center)
    bpy.context.scene.render.filepath = str(output_dir / "perspective.png")
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    clear_scene()
    objects = import_glb(input_path)
    setup_scene(args.size)

    lo, hi = world_bounds(objects)
    center = (lo + hi) * 0.5
    extent = hi - lo
    max_dim = max(extent.x, extent.y, extent.z)
    if max_dim <= 1e-9:
        raise RuntimeError(f"Degenerate model bounds: {extent}")

    cam = add_camera()
    d = max_dim * 3.0
    margin = 1.20

    # Z-up convention:
    # front: look along +Y; side: look along -X; top: look along -Z.
    render_ortho(
        cam,
        "front",
        center + Vector((0.0, -d, 0.0)),
        center,
        max(extent.x, extent.z) * margin,
        output_dir,
    )
    render_ortho(
        cam,
        "side",
        center + Vector((d, 0.0, 0.0)),
        center,
        max(extent.y, extent.z) * margin,
        output_dir,
    )
    render_ortho(
        cam,
        "top",
        center + Vector((0.0, 0.0, d)),
        center,
        max(extent.x, extent.y) * margin,
        output_dir,
    )
    render_perspective(cam, center, extent, output_dir)

    print(f"[OK] four views -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
