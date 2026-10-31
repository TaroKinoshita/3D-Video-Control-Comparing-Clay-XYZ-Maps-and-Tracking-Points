"""
probe_raw.py  —  W2b Step 3a-3

目的3つ
  1. view_transform = 'Raw' で Position / Depth の値が厳密に保存されるか
  2. Emission ID を Strength = pass_index で焼けるか（Color の 1.0 クランプ回避）
  3. CYCLES が使えるかを --factory-startup の有無で切り分ける

【2回実行してほしい】
  & $BL --background --factory-startup --python probe_raw.py "--" --out . --tag FS
  & $BL --background                   --python probe_raw.py "--" --out . --tag NORMAL

検証シーン（カメラ規約は本番と同一。depth = world_Y + 10）
  Wall  : world Y = 2.00 の面 -> depth 12.00
  Front : world Y = -2.00 の面 -> depth  8.00
  期待：Depth の一意値 = {8.0, 12.0} / Position の G の一意値 = {-2.0, 2.0}
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import bpy
import bmesh

OUT = {}
RES = 128


def sec(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def kv(k, v):
    print("  %-40s %s" % (k, v))


def guard(label, fn, store=None):
    try:
        r = fn()
        kv(label, r)
        if store:
            OUT[store] = r if isinstance(r, (str, int, float, bool, list, dict)) else str(r)
        return r
    except Exception as exc:
        m = "EXCEPTION %s: %s" % (type(exc).__name__, exc)
        kv(label, m)
        if store:
            OUT[store] = m
        return None


def args():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []
    out, tag = ".", "X"
    if "--out" in a:
        i = a.index("--out")
        if i + 1 < len(a):
            out = a[i + 1]
    if "--tag" in a:
        i = a.index("--tag")
        if i + 1 < len(a):
            tag = a[i + 1]
    return os.path.abspath(out), tag


OUTDIR, TAG = args()
OUT["tag"] = TAG


def box(name, x0, x1, y0, y1, z0, z1, pidx):
    cs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    verts = [(x, y, z0) for x, y in cs] + [(x, y, z1) for x, y in cs]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([i, j, j + 4, i + 4])
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate(verbose=False)
    me.update()
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.pass_index = pidx
    return ob


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = 1
    sc.render.resolution_x = RES
    sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.filter_size = 0.01
    sc.render.use_motion_blur = False
    sc.eevee.use_raytracing = False
    wall = box("Wall", -6, 6, 2.00, 2.05, -6, 6, 1)
    front = box("Front", -1, 1, -2.00, -1.95, -1, 1, 4)
    cd = bpy.data.cameras.new("cam")
    cd.lens, cd.sensor_fit, cd.sensor_width = 50.0, "HORIZONTAL", 36.0
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    cam.location = (0.0, -10.0, 0.0)
    cam.rotation_euler = (1.5707963267948966, 0.0, 0.0)
    sc.camera = cam
    ld = bpy.data.lights.new("L", type="POINT")
    ld.energy = 1000.0
    lo = bpy.data.objects.new("L", ld)
    sc.collection.objects.link(lo)
    lo.location = (0.5, -9.0, 1.5)
    vl = bpy.context.view_layer
    for a in ("use_pass_combined", "use_pass_z", "use_pass_position"):
        setattr(vl, a, True)
    bpy.context.view_layer.update()
    return sc, wall, front


def set_raw(sc):
    """カラーマネジメントを素通しにする。ここが Position 破損の原因だった。"""
    r = {}
    for attr, val in (("view_transform", "Raw"), ("look", "None")):
        try:
            setattr(sc.view_settings, attr, val)
            r[attr] = getattr(sc.view_settings, attr)
        except Exception as exc:
            r[attr] = "EXCEPTION %s" % exc
    for attr, val in (("exposure", 0.0), ("gamma", 1.0)):
        try:
            setattr(sc.view_settings, attr, val)
            r[attr] = getattr(sc.view_settings, attr)
        except Exception as exc:
            r[attr] = "EXCEPTION %s" % exc
    try:
        sc.sequencer_colorspace_settings.name = "Non-Color"
        r["seq_colorspace"] = sc.sequencer_colorspace_settings.name
    except Exception as exc:
        r["seq_colorspace"] = "EXCEPTION %s" % exc
    return r


def wire(sc, socket):
    ng = bpy.data.node_groups.new("C_" + socket, "CompositorNodeTree")
    sc.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = sc
    go = ng.nodes.new("NodeGroupOutput")
    src = next((s for s in rl.outputs if s.name == socket), None)
    if src is None:
        return "socket %r NOT FOUND; have %s" % (socket, [s.name for s in rl.outputs])
    ng.links.new(src, go.inputs[0])
    return "linked %s" % socket


def render_exr(sc, stem):
    ims = sc.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_mode = "RGBA"
    ims.color_depth = "32"
    ims.exr_codec = "NONE"
    path = os.path.join(OUTDIR, stem)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path + ".exr"


def readback(path, cap=12):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    px = list(img.pixels)
    w, h = img.size
    n = w * h
    res = {"size": [w, h], "depth": img.depth}
    for ci, cn in enumerate("RGB"):
        vals = [px[i * 4 + ci] for i in range(n)]
        u = sorted({round(v, 6) for v in vals})
        res[cn] = {"n_unique": len(u), "min": min(vals), "max": max(vals),
                   "unique": u if len(u) <= cap else u[:cap] + ["...(%d)" % len(u)]}
    return res


# ================================================================== 0
sec("0. 環境（--factory-startup の有無を切り分ける）  tag=%s" % TAG)
kv("blender", bpy.app.version_string)
kv("addons loaded", len(bpy.context.preferences.addons))
OUT["addons_loaded"] = len(bpy.context.preferences.addons)
sc0 = bpy.context.scene
guard("engines", lambda: [e.identifier for e in
                          sc0.render.bl_rna.properties["engine"].enum_items],
      "engines")
guard("addon_enable('cycles')",
      lambda: str(bpy.ops.preferences.addon_enable(module="cycles")), "cycles_enable")
guard("engines after enable",
      lambda: [e.identifier for e in
               sc0.render.bl_rna.properties["engine"].enum_items],
      "engines_after")
guard("bpy.types.CyclesRender exists",
      lambda: hasattr(bpy.types, "CyclesRender"), "has_CyclesRender")
guard("addon names",
      lambda: sorted(a.module for a in bpy.context.preferences.addons), "addon_names")

# ================================================================== 1
sec("1. view_transform の実測")
sc, wall, front = build()
kv("view_transform (default)", sc.view_settings.view_transform)
OUT["view_transform_default"] = sc.view_settings.view_transform
guard("view_transform enum (内省)",
      lambda: [e.identifier for e in
               sc.view_settings.bl_rna.properties["view_transform"].enum_items],
      "view_transform_enum")
OUT["set_raw"] = guard("set_raw()", lambda: set_raw(sc))
kv("display_device", sc.display_settings.display_device)

# ================================================================== 2
sec("2. Position（Raw で厳密か）  期待: G の一意値 = {-2.0, 2.0}")
kv("wire", wire(sc, "Position"))
p = guard("render", lambda: render_exr(sc, "_raw_position_%s" % TAG))
if p and os.path.exists(p):
    rb = readback(p)
    OUT["position"] = rb
    for c in "RGB":
        kv("Position %s" % c, rb[c])

# ================================================================== 3
sec("3. Depth（Raw で厳密か）  期待: 一意値 = {8.0, 12.0}")
kv("wire", wire(sc, "Depth"))
p = guard("render", lambda: render_exr(sc, "_raw_depth_%s" % TAG))
if p and os.path.exists(p):
    rb = readback(p)
    OUT["depth"] = rb
    kv("Depth R", rb["R"])

# ================================================================== 4
sec("4. Emission ID を Strength = pass_index で焼く  期待: 一意値 = {1.0, 4.0}")
sc, wall, front = build()
OUT["set_raw_2"] = set_raw(sc)
for ob in (wall, front):
    m = bpy.data.materials.new("ID_" + ob.name)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs["Strength"].default_value = float(ob.pass_index)
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
    ob.data.materials.clear()
    ob.data.materials.append(m)
    kv("%s pass_index -> Strength" % ob.name, em.inputs["Strength"].default_value)
kv("wire", wire(sc, "Image"))
p = guard("render", lambda: render_exr(sc, "_raw_idmask_%s" % TAG))
if p and os.path.exists(p):
    rb = readback(p)
    OUT["emission_id_strength"] = rb
    kv("ID R", rb["R"])

# ================================================================== 5
sec("5. Color 側に pass_index を入れた場合（クランプするか）")
for ob in (wall, front):
    m = ob.data.materials[0]
    em = next(n for n in m.node_tree.nodes if n.type == "EMISSION")
    v = float(ob.pass_index)
    em.inputs["Color"].default_value = (v, v, v, 1.0)
    em.inputs["Strength"].default_value = 1.0
    kv("%s Color set to %.1f -> stored" % (ob.name, v),
       tuple(round(c, 4) for c in em.inputs["Color"].default_value))
p = guard("render", lambda: render_exr(sc, "_raw_idcolor_%s" % TAG))
if p and os.path.exists(p):
    rb = readback(p)
    OUT["emission_id_color"] = rb
    kv("ID(color) R", rb["R"])

jp = os.path.join(OUTDIR, "probe_raw_%s.json" % TAG)
with open(jp, "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print()
print("=" * 70)
print("saved : %s" % jp)
print("=" * 70)
