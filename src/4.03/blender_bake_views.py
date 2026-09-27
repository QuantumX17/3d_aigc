from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


VIEW_DIRS = {
    "front": Vector((0.0, -1.0, 0.0)),
    "right": Vector((1.0, 0.0, 0.0)),
    "back": Vector((0.0, 1.0, 0.0)),
    "left": Vector((-1.0, 0.0, 0.0)),
    "top": Vector((0.0, 0.0, 1.0)),
    "bottom": Vector((0.0, 0.0, -1.0)),
}


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--view-dir", required=True)
    p.add_argument("--output-albedo", required=True)
    p.add_argument("--output-mesh", required=True)
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--views", nargs="+", required=True)
    return p.parse_args(argv_after_dashdash())


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_glb(path: str):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found in input GLB")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = "ALBEDO_TARGET"
    return obj


def world_bbox(obj):
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
    center = (lo + hi) * 0.5
    size = hi - lo
    radius = max(size.length * 0.5, 1e-3)
    return lo, hi, center, size, radius


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_camera(name: str, direction: Vector, center: Vector, size: Vector, radius: float):
    data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(cam)

    distance = radius * 3.5 + 1.0
    cam.location = center + direction.normalized() * distance
    look_at(cam, center)

    data.type = "ORTHO"
    data.ortho_scale = max(size.x, size.y, size.z) * 1.25
    data.clip_start = 0.01
    data.clip_end = distance + radius * 4.0 + 10.0
    return cam


def create_projection_uv(obj, scene, cam, layer_name: str):
    mesh = obj.data

    if layer_name in mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[layer_name])

    layer = mesh.uv_layers.new(name=layer_name)

    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vert_index = mesh.loops[loop_index].vertex_index
            world_co = obj.matrix_world @ mesh.vertices[vert_index].co
            ndc = world_to_camera_view(scene, cam, world_co)
            layer.data[loop_index].uv = (ndc.x, ndc.y)

    return layer


def polygon_world_center(obj, poly):
    if not poly.vertices:
        return obj.matrix_world @ poly.center

    acc = Vector((0.0, 0.0, 0.0))
    for vi in poly.vertices:
        acc += obj.matrix_world @ obj.data.vertices[vi].co
    return acc / len(poly.vertices)


def choose_polygon_views(obj, views):
    """
    Position-gated + front-priority assignment.

    This version specifically prevents side textures from leaking onto front
    door/window/bevel polygons.

    Side views are allowed only on polygons that are:
      (a) strongly side-facing by normal, AND
      (b) physically near the outer left/right side of the object's bbox.

    Therefore a thin bevel on the front facade cannot accidentally receive a
    side image and create a vertical streak.
    """
    dirs = {v: VIEW_DIRS[v].normalized() for v in views}
    normal_matrix = obj.matrix_world.to_3x3()

    lo, hi, center, size, _ = world_bbox(obj)

    hx = max(size.x * 0.5, 1e-9)
    hy = max(size.y * 0.5, 1e-9)
    hz = max(size.z * 0.5, 1e-9)

    choices = []

    for poly in obj.data.polygons:
        n = (normal_matrix @ poly.normal).normalized()
        p = polygon_world_center(obj, poly)

        scores = {v: n.dot(dirs[v]) for v in views}

        front_score = scores.get("front", -1.0)
        right_score = scores.get("right", -1.0)
        left_score = scores.get("left", -1.0)
        back_score = scores.get("back", -1.0)
        top_score = scores.get("top", -1.0)
        bottom_score = scores.get("bottom", -1.0)

        # Normalized position in object bbox:
        # x_pos: -1 left, +1 right
        # y_pos: -1 front, +1 back
        # z_pos: -1 bottom, +1 top
        x_pos = (p.x - center.x) / hx
        y_pos = (p.y - center.y) / hy
        z_pos = (p.z - center.z) / hz

        selected = None

        # Clear top/bottom surfaces, also require physical placement.
        if (
            "top" in views
            and top_score >= 0.80
            and z_pos >= 0.25
        ):
            selected = "top"

        elif (
            "bottom" in views
            and bottom_score >= 0.80
            and z_pos <= -0.25
        ):
            selected = "bottom"

        # Front facade and its bevels get broad priority.
        elif (
            "front" in views
            and front_score >= 0.35
        ):
            selected = "front"

        # Actual right exterior only.
        elif (
            "right" in views
            and right_score >= 0.82
            and x_pos >= 0.50
            and front_score < 0.20
        ):
            selected = "right"

        # Actual left exterior only.
        elif (
            "left" in views
            and left_score >= 0.82
            and x_pos <= -0.50
            and front_score < 0.20
        ):
            selected = "left"

        # Actual back exterior only.
        elif (
            "back" in views
            and back_score >= 0.82
            and y_pos >= 0.35
            and front_score < 0.0
        ):
            selected = "back"

        # If a polygon is in the front half of the object, prefer front rather
        # than using a conservative side texture.
        elif (
            "front" in views
            and y_pos <= 0.15
        ):
            selected = "front"

        # Remaining true outer-side polygons.
        elif (
            "right" in views
            and x_pos >= 0.65
            and right_score > max(front_score, back_score, top_score, bottom_score)
        ):
            selected = "right"

        elif (
            "left" in views
            and x_pos <= -0.65
            and left_score > max(front_score, back_score, top_score, bottom_score)
        ):
            selected = "left"

        elif (
            "back" in views
            and y_pos >= 0.50
            and back_score > front_score
        ):
            selected = "back"

        else:
            # Safe fallback:
            # use front unless the polygon is clearly behind the object's center.
            if "front" in views and y_pos < 0.30:
                selected = "front"
            else:
                selected = max(views, key=lambda v: scores[v])

        choices.append(selected)

    counts = Counter(choices)
    total = max(1, len(choices))

    print("[VIEW ASSIGNMENT]")
    for view in views:
        c = counts.get(view, 0)
        print(f"  {view:>6}: {c:8d} polygons ({c / total:.2%})")

    return choices


def make_projection_material(name: str, image_path: Path, projection_uv: str, bake_image):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path), check_existing=True)
    tex.extension = "CLIP"
    tex.interpolation = "Linear"

    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = projection_uv

    target = nodes.new("ShaderNodeTexImage")
    target.name = "BAKE_TARGET"
    target.image = bake_image

    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])

    for node in nodes:
        node.select = False
    target.select = True
    nodes.active = target

    return mat


def save_albedo_material(obj, albedo_image):
    obj.data.materials.clear()

    mat = bpy.data.materials.new("FINAL_ALBEDO")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    uv = nodes.new("ShaderNodeUVMap")

    tex.image = albedo_image
    tex.interpolation = "Linear"
    uv.uv_map = "BAKE_UV"

    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.5

    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.material_index = 0


def ensure_bake_uv_after_gltf_import(obj):
    mesh = obj.data

    if len(mesh.uv_layers) == 0:
        raise RuntimeError("Input GLB has no UV layers.")

    if "BAKE_UV" in mesh.uv_layers:
        uv = mesh.uv_layers["BAKE_UV"]
        mesh.uv_layers.active = uv
        uv.active_render = True
        return uv

    uv = mesh.uv_layers.active or mesh.uv_layers[0]
    old_name = uv.name
    try:
        uv.name = "BAKE_UV"
        print(f"[UV] Renamed '{old_name}' -> 'BAKE_UV'")
    except Exception as exc:
        print(f"[WARN] Could not rename UV layer: {exc}")

    mesh.uv_layers.active = uv
    uv.active_render = True
    return uv


def configure_cycles_for_bake(scene):
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "CPU"
    except Exception:
        pass
    print("[BLENDER] bake engine = CYCLES")


def bake_emit(scene, obj, bake_uv):
    mesh = obj.data
    mesh.uv_layers.active = bake_uv
    bake_uv.active_render = True

    for mat in mesh.materials:
        if mat is None or not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        target = nodes.get("BAKE_TARGET")
        if target is None:
            raise RuntimeError(f"Material '{mat.name}' has no BAKE_TARGET")

        for node in nodes:
            node.select = False
        target.select = True
        nodes.active = target

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    configure_cycles_for_bake(scene)
    scene.render.bake.margin = 16

    result = bpy.ops.object.bake(type="EMIT", use_clear=True)
    print("[BAKE] operator result:", result)

    if "FINISHED" not in result:
        raise RuntimeError(f"Bake failed: {result}")


def main():
    args = parse_args()

    input_mesh = Path(args.input).resolve()
    view_dir = Path(args.view_dir).resolve()
    output_albedo = Path(args.output_albedo).resolve()
    output_mesh = Path(args.output_mesh).resolve()

    clear_scene()
    obj = import_glb(str(input_mesh))
    mesh = obj.data

    bake_uv = ensure_bake_uv_after_gltf_import(obj)

    scene = bpy.context.scene
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    _, _, center, size, radius = world_bbox(obj)

    for view in args.views:
        if view not in VIEW_DIRS:
            raise ValueError(f"Unsupported view: {view}")

        cam = create_camera(
            f"CAM_{view}",
            VIEW_DIRS[view],
            center,
            size,
            radius,
        )
        create_projection_uv(obj, scene, cam, f"PROJ_{view}")

    bake_image = bpy.data.images.new(
        "ALBEDO_BAKE",
        width=args.texture_size,
        height=args.texture_size,
        alpha=False,
        float_buffer=False,
    )
    bake_image.generated_color = (0.5, 0.5, 0.5, 1.0)

    mesh.materials.clear()
    view_to_index = {}

    for view in args.views:
        image_path = view_dir / f"view_{view}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        mat = make_projection_material(
            f"MAT_{view}",
            image_path,
            f"PROJ_{view}",
            bake_image,
        )
        mesh.materials.append(mat)
        view_to_index[view] = len(mesh.materials) - 1

    choices = choose_polygon_views(obj, args.views)
    for poly, view in zip(mesh.polygons, choices):
        poly.material_index = view_to_index[view]

    bake_emit(scene, obj, bake_uv)

    output_albedo.parent.mkdir(parents=True, exist_ok=True)
    bake_image.filepath_raw = str(output_albedo)
    bake_image.file_format = "PNG"
    bake_image.save()

    if not output_albedo.exists():
        raise RuntimeError(f"Albedo was not written: {output_albedo}")

    print("[BAKE] Albedo:", output_albedo)

    save_albedo_material(obj, bake_image)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_mesh),
        export_format="GLB",
        use_selection=True,
        export_image_format="AUTO",
    )

    if not output_mesh.exists():
        raise RuntimeError(f"Textured mesh was not written: {output_mesh}")

    print("[EXPORT] Textured mesh:", output_mesh)


if __name__ == "__main__":
    main()
