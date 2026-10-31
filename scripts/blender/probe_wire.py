"""
probe_wire.py  —  W2b Step 3a-4

Position / Depth の書き出し経路を特定する。一意値の個数では区別がつかないので
**既知座標の画素値を直接読む**。

  & $BL --background --factory-startup --python probe_wire.py "--" --out .

検証シーン（128x128。カメラ規約は本番と同一。depth = world_Y + 10）
  Wall  : Y = 2.00 の面 -> depth 12.00 / X,Z は +-6
  Front : Y = -2.00 の面 -> depth 8.00 / X,Z は +-1
  px_per_unit(d) = 177.77778 / d      （128px 幅・sensor36mm・lens50mm）
  u = 64 + X * pxu(d)   /   v = 64 - Z * pxu(d)

試す経路
  R0 コンポジタ無し（Combined）           … 「無視されたとき」の基準値
  R1 interface.new_socket(Color)  + GroupOutput
  R2 interface.new_socket(Vector) + GroupOutput
  R3 仮想ソケットのみ（前回の wire()）      … 再現確認
  R4 OutputFile ノード + save_as_render=False
おまけ
  _cycles / cycles の import 失敗理由
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
PXU = 177.77778


def sec(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


def kv(k, v):
    print("  %-38s %s" % (k, v))


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


def outdir():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []
    if "--out" in a:
        i = a.index("--out")
        if i + 1 < len(a):
            return os.path.abspath(a[i + 1])
    return os.path.abspath(".")


OUTDIR = outdir()

# ---------------------------------------------------------------- 照合点
# (i, j) は Blender の pixels バッファ座標（j = 0 が下端）
# u = i + 0.5 / v = RES - (j + 0.5)
PROBE_PX = [
    ("A_front_centre",  64,  64),
    ("B_wall_lowleft",  10,  10),
    ("C_wall_upright", 117, 117),
    ("D_wall_below",    64,  20),
    ("E_front_left",    44,  64),
]


def expect_world(i, j):
    """その画素に見える面の world 座標を閉形式で出す。"""
    u = i + 0.5
    v = RES - (j + 0.5)
    # Front box が覆う範囲（depth 8）
    p8 = PXU / 8.0
    fu0, fu1 = 64 - 1.0 * p8, 64 + 1.0 * p8
    fv0, fv1 = 64 - 1.0 * p8, 64 + 1.0 * p8
    if fu0 <= u <= fu1 and fv0 <= v <= fv1:
        d = 8.0
        return {"obj": "Front", "pass_index": 4, "depth": d,
                "X": (u - 64) * d / PXU, "Y": -2.0, "Z": (64 - v) * d / PXU}
    d = 12.0
    return {"obj": "Wall", "pass_index": 1, "depth": d,
            "X": (u - 64) * d / PXU, "Y": 2.0, "Z": (64 - v) * d / PXU}


def box(name, x0, x1, y0, y1, z0, z1, pidx):
    cs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    verts = [(x, y, z0) for x, y in cs] + [(x, y, z1) for x, y in cs]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    for k in range(4):
        m = (k + 1) % 4
        faces.append([k, m, m + 4, k + 4])
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
    sc.eevee.use_raytracing = False
    sc.render.resolution_x = RES
    sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.filter_size = 0.01
    sc.render.use_motion_blur = False
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
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


def set_exr(sc):
    ims = sc.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_mode = "RGBA"
    ims.color_depth = "32"
    ims.exr_codec = "NONE"


def render_still(sc, stem):
    set_exr(sc)
    p = os.path.join(OUTDIR, stem)
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    return p + ".exr"


def sample(path):
    """照合画素の RGB を返す。"""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    px = list(img.pixels)
    w, h = img.size
    res = {"size": [w, h]}
    for name, i, j in PROBE_PX:
        k = (j * w + i) * 4
        res[name] = [round(px[k + 0], 5), round(px[k + 1], 5), round(px[k + 2], 5)]
    return res


def report(label, got):
    OUT[label] = got
    print("  %s" % label)
    if got is None:
        print("    (no data)")
        return
    for name, i, j in PROBE_PX:
        e = expect_world(i, j)
        print("    %-16s got RGB=%-28s | 期待 Position=(%.4f, %.4f, %.4f) depth=%.2f id=%d"
              % (name, got.get(name), e["X"], e["Y"], e["Z"], e["depth"], e["pass_index"]))


# ================================================================== 照合表
sec("照合表（閉形式。この値と突合する）")
for name, i, j in PROBE_PX:
    e = expect_world(i, j)
    kv(name, "(i=%d j=%d) %s pass_index=%d depth=%.2f  Position=(%.4f, %.4f, %.4f)"
       % (i, j, e["obj"], e["pass_index"], e["depth"], e["X"], e["Y"], e["Z"]))
    OUT.setdefault("expect", {})[name] = e

# ================================================================== R0
sec("R0. コンポジタ無し（Combined）= 無視されたときの基準値")
sc, wall, front = build()
sc.compositing_node_group = None
kv("render.use_compositing", sc.render.use_compositing)
OUT["use_compositing"] = sc.render.use_compositing
p = guard("render", lambda: render_still(sc, "_w_R0_combined"))
report("R0_combined", sample(p) if p and os.path.exists(p) else None)

# ================================================================== R1 / R2
def wire_interface(sc, socket_name, socket_type):
    ng = bpy.data.node_groups.new("W_%s_%s" % (socket_name, socket_type),
                                  "CompositorNodeTree")
    info = {}
    s = ng.interface.new_socket(name="Out", in_out="OUTPUT",
                                socket_type=socket_type)
    info["interface_socket"] = "%s / %s" % (s.identifier, s.socket_type)
    sc.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = sc
    go = ng.nodes.new("NodeGroupOutput")
    info["groupoutput_inputs"] = [(x.name, x.type) for x in go.inputs]
    src = next((x for x in rl.outputs if x.name == socket_name), None)
    if src is None:
        info["link"] = "socket %r NOT FOUND: %s" % (socket_name,
                                                    [x.name for x in rl.outputs])
        return ng, info
    tgt = go.inputs[0]
    lk = ng.links.new(src, tgt)
    info["link"] = "%s -> %s (valid=%s)" % (src.name, tgt.name,
                                            getattr(lk, "is_valid", "?"))
    info["links_count"] = len(ng.links)
    return ng, info


for tag, sockname, socktype in (("R1_position_color", "Position", "NodeSocketColor"),
                                ("R2_position_vector", "Position", "NodeSocketVector"),
                                ("R1b_depth_float", "Depth", "NodeSocketFloat")):
    sec("%s : interface.new_socket(%s) + GroupOutput" % (tag, socktype))
    sc, wall, front = build()
    ng, info = wire_interface(sc, sockname, socktype)
    for k, v in info.items():
        kv(k, v)
    OUT[tag + "_info"] = info
    p = guard("render", lambda: render_still(sc, "_w_" + tag))
    report(tag, sample(p) if p and os.path.exists(p) else None)

# ================================================================== R3
sec("R3. 仮想ソケットのみ（前回の wire()）= 再現確認")
sc, wall, front = build()
ng = bpy.data.node_groups.new("W_virtual", "CompositorNodeTree")
sc.compositing_node_group = ng
rl = ng.nodes.new("CompositorNodeRLayers")
rl.scene = sc
go = ng.nodes.new("NodeGroupOutput")
kv("groupoutput inputs", [(x.name, x.type) for x in go.inputs])
kv("interface items (before link)", [i.name for i in ng.interface.items_tree])
src = next(x for x in rl.outputs if x.name == "Position")
lk = ng.links.new(src, go.inputs[0])
kv("link valid", getattr(lk, "is_valid", "?"))
kv("interface items (after link)", [i.name for i in ng.interface.items_tree])
kv("groupoutput inputs (after)", [(x.name, x.type) for x in go.inputs])
p = guard("render", lambda: render_still(sc, "_w_R3_virtual"))
report("R3_virtual", sample(p) if p and os.path.exists(p) else None)

# ================================================================== R4
sec("R4. OutputFile ノード + save_as_render=False")
sc, wall, front = build()
ng = bpy.data.node_groups.new("W_outfile", "CompositorNodeTree")
sc.compositing_node_group = ng
rl = ng.nodes.new("CompositorNodeRLayers")
rl.scene = sc
of = ng.nodes.new("CompositorNodeOutputFile")
kv("of.directory (default)", of.directory)
guard("set of.directory", lambda: (setattr(of, "directory", OUTDIR), of.directory)[1])
guard("of.file_name", lambda: getattr(of, "file_name", "MISSING"))
guard("of.save_as_render", lambda: (setattr(of, "save_as_render", False),
                                    of.save_as_render)[1])
guard("of.format.file_format",
      lambda: (setattr(of.format, "file_format", "OPEN_EXR"),
               setattr(of.format, "color_depth", "32"),
               setattr(of.format, "color_mode", "RGBA"),
               of.format.file_format)[3])
kv("of.inputs (before)", [(x.name, x.type) for x in of.inputs])
kv("of.file_output_items (before)", len(of.file_output_items))
guard("file_output_items dir()",
      lambda: [n for n in dir(of.file_output_items) if not n.startswith("_")],
      "foi_dir")
guard("file_output_items.new('pos')",
      lambda: of.file_output_items.new("pos"), "foi_new")
kv("of.inputs (after)", [(x.name, x.type) for x in of.inputs])
kv("of.file_output_items (after)", len(of.file_output_items))
for it in of.file_output_items:
    kv("  item", [(n, getattr(it, n, "?")) for n in
                  ("name", "path", "use_node_format", "save_as_render")
                  if hasattr(it, n)])
src = next(x for x in rl.outputs if x.name == "Position")
if len(of.inputs):
    lk = ng.links.new(src, of.inputs[-1])
    kv("link Position -> of.inputs[-1]", getattr(lk, "is_valid", "?"))
guard("render (OutputFile 書き出し)",
      lambda: (bpy.ops.render.render(write_still=False), "returned")[1])
found = sorted(f for f in os.listdir(OUTDIR) if f.lower().endswith(".exr"))
kv("OUTDIR の EXR 一覧", found)
OUT["outdir_exr_after_R4"] = found
cand = [f for f in found if f.startswith("pos")]
if cand:
    report("R4_outputfile", sample(os.path.join(OUTDIR, cand[0])))
else:
    kv("R4", "pos* の EXR が見つからない")

# ================================================================== cycles
sec("おまけ. Cycles の import 失敗理由")
for mod in ("_cycles", "cycles"):
    guard("import %s" % mod, lambda m=mod: __import__(m).__file__,
          "import_%s" % mod)
try:
    import addon_utils
    guard("addon_utils.check('cycles')", lambda: addon_utils.check("cycles"))
    guard("addon_utils.enable('cycles')",
          lambda: str(addon_utils.enable("cycles", default_set=False)))
except Exception as exc:
    kv("addon_utils", "EXCEPTION %s" % exc)
guard("CyclesRender after", lambda: hasattr(bpy.types, "CyclesRender"))
try:
    import cycles as _c
    d = os.path.dirname(_c.__file__)
    kv("cycles dir", d)
    kv("cycles dir listing", sorted(os.listdir(d))[:30])
    OUT["cycles_dir_listing"] = sorted(os.listdir(d))
except Exception as exc:
    kv("cycles dir", "EXCEPTION %s" % exc)

jp = os.path.join(OUTDIR, "probe_wire.json")
with open(jp, "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print()
print("=" * 74)
print("saved : %s" % jp)
print("=" * 74)
