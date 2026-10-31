"""
probe_masks.py  —  W2b Step 3a-2
ブロッカーA（EEVEE に IndexOB が無い）と B（--factory-startup が Cycles を消す）の
解決候補を実測で比較する。判定はしない。数値を吐く。

  & $BL --background --factory-startup --python probe_masks.py "--" --out .

検証する4案
  案1 Cycles アドオンを明示有効化 -> CYCLES が使えるか / IndexOB が出るか
  案2 EEVEE + Cryptomatte Object   -> ソケットが出るか
  案3 EEVEE + 平坦 Emission ID     -> 32bit EXR で値が厳密に保存されるか
  案4 obj.is_holdout               -> 存在するか（遮蔽を効かせた単体レンダー用）
おまけ  RLayers.Position -> Group Output の実配線で World 座標が正しく出るか
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import bpy

OUT = {}
RES = 128


def sec(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def kv(k, v):
    print("  %-42s %s" % (k, v))


def guard(label, fn, store=None):
    try:
        r = fn()
        kv(label, r)
        if store:
            OUT[store] = str(r)
        return r
    except Exception as exc:
        m = "EXCEPTION %s: %s" % (type(exc).__name__, exc)
        kv(label, m)
        if store:
            OUT[store] = m
        return None


def parse_out():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []
    if "--out" in a:
        i = a.index("--out")
        if i + 1 < len(a):
            return a[i + 1]
    return "."


OUTDIR = os.path.abspath(parse_out())


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
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.pass_index = pidx
    return ob


def build_scene():
    """カメラ規約は本番と同じ。depth = world_Y + 10。
    Wall(depth 12) の前に Front(depth 8) を置いて遮蔽関係を作る。"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.resolution_x = RES
    sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.filter_size = 0.01
    sc.render.use_motion_blur = False
    wall = box("Wall", -6, 6, 2.00, 2.05, -6, 6, 1)
    front = box("Front", -1, 1, -2.00, -1.95, -1, 1, 4)
    cd = bpy.data.cameras.new("cam")
    cd.lens = 50.0
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
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
    bpy.context.view_layer.update()
    return sc, wall, front


def wire(sc, socket_name):
    """RLayers.<socket> -> Group Output を張る（§22 の決定経路）。"""
    ng = bpy.data.node_groups.new("C_" + socket_name, "CompositorNodeTree")
    sc.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = sc
    go = ng.nodes.new("NodeGroupOutput")
    src = None
    for s in rl.outputs:
        if s.name == socket_name:
            src = s
            break
    if src is None:
        return ng, [s.name for s in rl.outputs], "socket %r not found" % socket_name
    ng.links.new(src, go.inputs[0])
    return ng, [s.name for s in rl.outputs], "linked %s -> %s" % (src.name, go.inputs[0].name)


def render_exr(sc, stem, depth="32"):
    ims = sc.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_mode = "RGBA"
    ims.color_depth = depth
    ims.exr_codec = "NONE"
    path = os.path.join(OUTDIR, stem)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path + ".exr"


def readback(path):
    """EXR を読み戻して値の分布を出す。カラースペースは Non-Color に固定。"""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    px = list(img.pixels)
    w, h = img.size
    n = w * h
    r = [px[i * 4 + 0] for i in range(n)]
    g = [px[i * 4 + 1] for i in range(n)]
    b = [px[i * 4 + 2] for i in range(n)]
    uniq = sorted({round(v, 6) for v in r})
    return {"size": (w, h), "channels": img.channels, "depth": img.depth,
            "R_unique_count": len(uniq),
            "R_unique_first20": uniq[:20],
            "R_min": min(r), "R_max": max(r),
            "G_min": min(g), "G_max": max(g),
            "B_min": min(b), "B_max": max(b)}


# ==================================================================== 案1
sec("案1. Cycles アドオンを明示有効化する")
sc = bpy.context.scene
kv("engines (before)", [e.identifier for e in
                        sc.render.bl_rna.properties["engine"].enum_items])
guard("addon_enable(module='cycles')",
      lambda: bpy.ops.preferences.addon_enable(module="cycles"), "cycles_enable")
eng_after = guard("engines (after)",
                  lambda: [e.identifier for e in
                           sc.render.bl_rna.properties["engine"].enum_items],
                  "engines_after")

sc, wall, front = build_scene()
vl = bpy.context.view_layer
for a in ("use_pass_combined", "use_pass_z", "use_pass_position",
          "use_pass_object_index", "use_pass_cryptomatte_object"):
    try:
        setattr(vl, a, True)
    except Exception as exc:
        kv("enable " + a, "EXCEPTION %s" % exc)

if eng_after and "CYCLES" in str(eng_after):
    guard("set engine CYCLES",
          lambda: (setattr(sc.render, "engine", "CYCLES"), sc.render.engine)[1])
    ng, socks, msg = wire(sc, "IndexOB")
    kv("CYCLES RLayers sockets", socks)
    OUT["cycles_rlayers_sockets"] = socks
    kv("wire IndexOB", msg)
    if "linked" in msg:
        try:
            sc.cycles.samples = 1
        except Exception:
            pass
        p = guard("render CYCLES IndexOB", lambda: render_exr(sc, "_probe_indexob"))
        if p and os.path.exists(p):
            OUT["cycles_indexob"] = guard("readback IndexOB", lambda: readback(p))
else:
    kv("CYCLES", "使用不可のまま")

# ==================================================================== 案2/3/おまけ
sec("案2. EEVEE + Cryptomatte Object のソケット")
sc, wall, front = build_scene()
vl = bpy.context.view_layer
sc.render.engine = "BLENDER_EEVEE"
sc.eevee.taa_render_samples = 1
for a in ("use_pass_combined", "use_pass_z", "use_pass_position",
          "use_pass_object_index", "use_pass_cryptomatte_object"):
    try:
        setattr(vl, a, True)
    except Exception:
        pass
ng, socks, msg = wire(sc, "Position")
kv("EEVEE RLayers sockets (crypto有効)", socks)
OUT["eevee_rlayers_sockets_crypto"] = socks

sec("おまけ. RLayers.Position -> Group Output の実配線")
kv("wire", msg)
p = guard("render Position", lambda: render_exr(sc, "_probe_position"))
if p and os.path.exists(p):
    rb = guard("readback Position", lambda: readback(p))
    OUT["eevee_position"] = rb
    kv("期待: G(=world Y) に 2.00 と -2.00 が現れる", "-> G_min/G_max を見る")

sec("案3. EEVEE + 平坦 Emission ID")
sc, wall, front = build_scene()
sc.render.engine = "BLENDER_EEVEE"
sc.eevee.taa_render_samples = 1
try:
    kv("view_transform enum",
       [e.identifier for e in
        sc.view_settings.bl_rna.properties["view_transform"].enum_items])
except Exception as exc:
    kv("view_transform enum", "EXCEPTION %s" % exc)
guard("set view_transform Raw",
      lambda: (setattr(sc.view_settings, "view_transform", "Raw"),
               sc.view_settings.view_transform)[1], "view_transform")
guard("set look None",
      lambda: (setattr(sc.view_settings, "look", "None"), sc.view_settings.look)[1])

IDVAL = {"Wall": 0.125, "Front": 0.500}
for ob in (wall, front):
    m = bpy.data.materials.new("ID_" + ob.name)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    v = IDVAL[ob.name]
    em.inputs["Color"].default_value = (v, v, v, 1.0)
    em.inputs["Strength"].default_value = 1.0
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
    ob.data.materials.clear()
    ob.data.materials.append(m)
vl = bpy.context.view_layer
vl.use_pass_combined = True
ng, socks, msg = wire(sc, "Image")
kv("wire", msg)
kv("期待する一意値", sorted(set(IDVAL.values())) + [0.0])
p = guard("render Emission ID", lambda: render_exr(sc, "_probe_idmask"))
if p and os.path.exists(p):
    OUT["eevee_emission_id"] = guard("readback Emission ID", lambda: readback(p))

# ==================================================================== 案4
sec("案4. is_holdout / hide_render")
for a in ("is_holdout", "hide_render", "visible_shadow", "visible_camera"):
    guard("Object.%s" % a,
          lambda x=a: "%r" % getattr(front, x) if hasattr(front, x) else "ATTR MISSING",
          "obj_%s" % a)

jp = os.path.join(OUTDIR, "probe_masks.json")
with open(jp, "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print()
print("=" * 70)
print("saved : %s" % jp)
print("=" * 70)
