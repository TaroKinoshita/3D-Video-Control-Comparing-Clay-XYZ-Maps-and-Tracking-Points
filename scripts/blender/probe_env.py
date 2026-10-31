"""
probe_env.py  rev.2  —  W2b Step 1 の前段チェック（10秒）

  & $BL --background --factory-startup --python probe_env.py

【rev.2 の修正】
初版は hasattr(bpy.types.Mesh, "validate") のように RNA の「型クラス」を見ていた。
5.2 では型クラス経由でメソッド・プロパティが見えず、
Mesh.validate / Object.pass_index / Scene.compositing_node_group の3件が
実際には動くのに False（偽陰性）になった。
→ プローブは型ではなく「インスタンス」に対して行う。
"""
import sys
import os
import math

print("=" * 64)
print("probe_env rev.2")
print("=" * 64)

try:
    import bpy
    import bmesh
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
except Exception as exc:
    print("import FAILED: %s" % exc)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def show(label, fn):
    try:
        print("%-36s %s" % (label, fn()))
    except Exception as exc:
        print("%-36s EXCEPTION %s: %s" % (label, type(exc).__name__, exc))


def probe_pass_index(o):
    o.pass_index = 7
    return "OK write/read = %s" % o.pass_index


def probe_material():
    m = bpy.data.materials.new("_probe_mat")
    nt = getattr(m, "node_tree", None)
    if nt is None:
        return "node_tree is None (use_nodes が必要かもしれない)"
    return "OK nodes = %s" % sorted({n.type for n in nt.nodes})


def probe_projection(sc):
    """matrix_world の遅延評価の有無を同時に測る。ここが Step 1 初回の失敗原因。"""
    cd = bpy.data.cameras.new("_probe_cam")
    cd.lens = 50.0
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    c = bpy.data.objects.new("_probe_cam", cd)
    sc.collection.objects.link(c)
    c.location = (0.0, -10.0, 0.0)
    c.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    sc.camera = c
    sc.render.resolution_x = 832
    sc.render.resolution_y = 480
    P = Vector((0.0, 0.0, 0.0))       # world 原点 -> depth 10.0 が正解
    before = world_to_camera_view(sc, c, P)
    bpy.context.view_layer.update()
    after = world_to_camera_view(sc, c, P)
    lazy = abs(before.z - after.z) > 1e-6
    return ("depth before=%.4f / after=%.4f / 期待=10.0000 / 遅延評価=%s"
            % (before.z, after.z, lazy))


print("blender            : %s" % bpy.app.version_string)
print("python             : %s" % sys.version.split()[0])
print("background         : %s" % bpy.app.background)
print("addons loaded      : %d  <- --factory-startup なら 0 に近いはず"
      % len(bpy.context.preferences.addons))
print("script dir         : %s" % HERE)
for f in ("scene_c_spec.py", "scene_c_expect.py", "build_scene_c.py", "verify_expect.py"):
    print("file %-19s: %s"
          % (f, "OK" if os.path.exists(os.path.join(HERE, f)) else "MISSING"))

print("-" * 64)
print("INSTANCE PROBE")
print("-" * 64)

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("%-36s OK objects=%d" % ("read_factory_settings(use_empty)",
                                   len(bpy.data.objects)))
except Exception as exc:
    print("%-36s FAILED %s" % ("read_factory_settings(use_empty)", exc))

scene = bpy.context.scene
mesh = bpy.data.meshes.new("_probe")
mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [[0, 1, 2, 3]])
obj = bpy.data.objects.new("_probe", mesh)
scene.collection.objects.link(obj)

show("mesh.from_pydata", lambda: "OK verts=%d polys=%d"
     % (len(mesh.vertices), len(mesh.polygons)))
show("mesh.validate()", lambda: "OK returned %s" % mesh.validate(verbose=False))
show("bmesh.ops.recalc_face_normals",
     lambda: "OK" if hasattr(bmesh.ops, "recalc_face_normals") else "MISSING")
show("obj.pass_index", lambda: probe_pass_index(obj))
show("scene.compositing_node_group",
     lambda: "OK type=%s" % type(scene.compositing_node_group).__name__)
show("scene.render.engine", lambda: "OK %s" % scene.render.engine)
show("material node_tree", probe_material)
show("world_to_camera_view + depsgraph", lambda: probe_projection(scene))
show("ops.wm.save_as_mainfile",
     lambda: "OK" if hasattr(bpy.ops.wm, "save_as_mainfile") else "MISSING")

print("-" * 64)
try:
    t = os.path.join(HERE, "_probe_write_test.tmp")
    with open(t, "w") as f:
        f.write("ok")
    os.remove(t)
    print("write permission   : OK")
except Exception as exc:
    print("write permission   : FAILED %s" % exc)

try:
    tb = os.path.abspath(os.path.join(HERE, "_probe_save_test.blend"))
    bpy.ops.wm.save_as_mainfile(filepath=tb)
    print("save_as_mainfile   : OK exists=%s" % os.path.exists(tb))
    if os.path.exists(tb):
        os.remove(tb)
except Exception as exc:
    print("save_as_mainfile   : FAILED %s: %s" % (type(exc).__name__, exc))
print("=" * 64)
