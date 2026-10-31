"""
probe_compositor.py  —  W2b Step 3a
コンポジタ / パス / EEVEE / EXR の API を実測してダンプする。

  & $BL --background --factory-startup --python probe_compositor.py "--" --out .

【方針】
assert しない。判定しない。**存在するものを全部吐く**だけ。
W2a で Blender API の予測的中率が 1/7 だったので、Step 3b を書く前に
ここで実物を確認する（作法#6）。プローブは型ではなくインスタンスで行う（作法#16）。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())

import bpy

OUT = {}


def sec(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def kv(k, v):
    print("  %-40s %s" % (k, v))


def guard(label, fn, store=None):
    try:
        r = fn()
        kv(label, r)
        if store is not None:
            OUT[store] = r if isinstance(r, (str, int, float, bool, list, dict)) else str(r)
        return r
    except Exception as exc:
        msg = "EXCEPTION %s: %s" % (type(exc).__name__, exc)
        kv(label, msg)
        if store is not None:
            OUT[store] = msg
        return None


def parse_out():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 < len(argv):
            return argv[i + 1]
    return "."


out_dir = parse_out()

# ------------------------------------------------------------------ 基本
sec("0. 環境")
kv("blender", bpy.app.version_string)
OUT["blender"] = bpy.app.version_string
kv("addons loaded", len(bpy.context.preferences.addons))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
vl = bpy.context.view_layer
kv("scene", scene.name)
kv("view_layer", vl.name)
kv("render.engine (default)", scene.render.engine)
OUT["engine_default"] = scene.render.engine

# ------------------------------------------------------------------ エンジン列挙
sec("1. 利用可能なレンダーエンジン")
try:
    engines = [e.identifier for e in
               scene.render.bl_rna.properties["engine"].enum_items]
    for e in engines:
        kv("engine", e)
    OUT["engines"] = engines
except Exception as exc:
    kv("engine enum", "EXCEPTION %s" % exc)

# ------------------------------------------------------------------ EEVEE
sec("2. EEVEE / render のプロパティ（幾何パスに必要なもの）")
guard("set engine BLENDER_EEVEE",
      lambda: (setattr(scene.render, "engine", "BLENDER_EEVEE"), scene.render.engine)[1],
      "engine_set")

for path, attr in (("scene.eevee", "taa_render_samples"),
                   ("scene.eevee", "taa_samples"),
                   ("scene.eevee", "use_motion_blur"),
                   ("scene.eevee", "use_raytracing"),
                   ("scene.eevee", "use_shadows"),
                   ("scene.eevee", "use_volumetric_lights"),
                   ("scene.render", "filter_size"),
                   ("scene.render", "use_motion_blur"),
                   ("scene.render", "film_transparent"),
                   ("scene.render", "use_border")):
    obj = scene.eevee if path == "scene.eevee" else scene.render
    guard("%s.%s" % (path, attr),
          lambda o=obj, a=attr: "%r  (type %s)" % (getattr(o, a), type(getattr(o, a)).__name__),
          "%s.%s" % (path, attr))

sec("2b. scene.eevee のプロパティ全列挙")
try:
    names = sorted(p.identifier for p in scene.eevee.bl_rna.properties
                   if p.identifier != "rna_type")
    OUT["eevee_props"] = names
    for i in range(0, len(names), 3):
        print("  " + "  ".join("%-30s" % n for n in names[i:i + 3]))
except Exception as exc:
    kv("eevee props", "EXCEPTION %s" % exc)

# ------------------------------------------------------------------ パス
sec("3. view_layer のパスフラグ")
pass_attrs = [p.identifier for p in vl.bl_rna.properties
              if p.identifier.startswith("use_pass")]
OUT["pass_flags"] = sorted(pass_attrs)
for a in sorted(pass_attrs):
    kv(a, getattr(vl, a, "?"))

sec("3b. 必要なパスを有効化して結果を確認")
for a in ("use_pass_combined", "use_pass_z", "use_pass_position",
          "use_pass_object_index", "use_pass_normal", "use_pass_mist"):
    guard("enable %s" % a,
          lambda x=a: (setattr(vl, x, True), getattr(vl, x))[1],
          "enabled_%s" % a)

# 一部は cycles 側にある場合がある
guard("vl.cycles exists", lambda: hasattr(vl, "cycles"))

# ------------------------------------------------------------------ コンポジタ
sec("4. コンポジタのノードグループ")
kv("scene.compositing_node_group (before)", scene.compositing_node_group)
guard("scene.use_nodes (存在するか)",
      lambda: "%r" % getattr(scene, "use_nodes", "ATTR MISSING"))

ng = guard("bpy.data.node_groups.new(type=CompositorNodeTree)",
           lambda: bpy.data.node_groups.new("CompC", "CompositorNodeTree"))
if ng is not None:
    guard("assign scene.compositing_node_group",
          lambda: (setattr(scene, "compositing_node_group", ng),
                   scene.compositing_node_group.name)[1],
          "assign_node_group")
    kv("node_group.type", getattr(ng, "type", "?"))
    kv("node_group.bl_idname", getattr(ng, "bl_idname", "?"))
    guard("node_group.interface exists", lambda: type(ng.interface).__name__)
    guard("interface.new_socket signature test",
          lambda: ng.interface.new_socket(name="Image", in_out="OUTPUT",
                                          socket_type="NodeSocketColor").identifier,
          "interface_new_socket")

sec("4b. コンポジタノードの型を列挙（Composite / OutputFile / Group Output）")
cand = sorted(n for n in dir(bpy.types)
              if n.startswith("CompositorNode") or n.startswith("NodeGroup"))
OUT["compositor_types"] = cand
for i in range(0, len(cand), 2):
    print("  " + "  ".join("%-44s" % n for n in cand[i:i + 2]))

for name in ("CompositorNodeComposite", "CompositorNodeViewer",
             "CompositorNodeRLayers", "CompositorNodeOutputFile",
             "NodeGroupOutput", "NodeGroupInput"):
    kv("bpy.types.%s" % name, hasattr(bpy.types, name))

sec("4c. ノードを実際に追加してソケットを見る")
if ng is not None:
    rl = guard("nodes.new CompositorNodeRLayers",
               lambda: ng.nodes.new("CompositorNodeRLayers"))
    if rl is not None:
        outs = [(s.name, s.type) for s in rl.outputs]
        OUT["rlayers_outputs"] = outs
        for n, t in outs:
            kv("RLayers output", "%-22s %s" % (n, t))
    go = guard("nodes.new NodeGroupOutput",
               lambda: ng.nodes.new("NodeGroupOutput"))
    if go is not None:
        kv("GroupOutput inputs", [(s.name, s.type) for s in go.inputs])
    of = guard("nodes.new CompositorNodeOutputFile",
               lambda: ng.nodes.new("CompositorNodeOutputFile"))
    if of is not None:
        for a in ("base_path", "directory", "file_slots",
                  "file_output_items", "layer_slots", "format"):
            guard("OutputFile.%s" % a,
                  lambda o=of, x=a: type(getattr(o, x)).__name__
                  if hasattr(o, x) else "ATTR MISSING",
                  "outputfile_%s" % a)
        props = sorted(p.identifier for p in of.bl_rna.properties
                       if p.identifier != "rna_type")
        OUT["outputfile_props"] = props
        print("  OutputFile props: %s" % ", ".join(props))

# ------------------------------------------------------------------ EXR
sec("5. image_settings（単層 OPEN_EXR の書き出し）")
ims = scene.render.image_settings
for a in ("file_format", "color_mode", "color_depth", "exr_codec",
          "use_zbuffer", "use_preview"):
    guard("image_settings.%s" % a,
          lambda x=a: "%r" % getattr(ims, x) if hasattr(ims, x) else "ATTR MISSING")

try:
    ff = [e.identifier for e in ims.bl_rna.properties["file_format"].enum_items]
    OUT["file_formats"] = ff
    kv("file_format enum", ", ".join(ff))
except Exception as exc:
    kv("file_format enum", "EXCEPTION %s" % exc)

guard("set OPEN_EXR",
      lambda: (setattr(ims, "file_format", "OPEN_EXR"), ims.file_format)[1])
for a in ("color_depth", "exr_codec", "color_mode"):
    try:
        vals = [e.identifier for e in ims.bl_rna.properties[a].enum_items]
        kv("%s enum (OPEN_EXR)" % a, ", ".join(vals))
        OUT["exr_%s_enum" % a] = vals
    except Exception as exc:
        kv("%s enum" % a, "EXCEPTION %s" % exc)

# ------------------------------------------------------------------ write_still
sec("6. 実レンダー（1フレーム・小サイズ）で書き出しまで通すか")
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
guard("eevee taa_render_samples = 1",
      lambda: (setattr(scene.eevee, "taa_render_samples", 1),
               scene.eevee.taa_render_samples)[1])
guard("render.filter_size = 0.01",
      lambda: (setattr(scene.render, "filter_size", 0.01),
               scene.render.filter_size)[1])

# 最小シーン：立方体 + カメラ
mesh = bpy.data.meshes.new("c")
mesh.from_pydata([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                  (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], [],
                 [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                  [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
ob = bpy.data.objects.new("c", mesh)
ob.pass_index = 3
scene.collection.objects.link(ob)
cd = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cd)
scene.collection.objects.link(cam)
cam.location = (0, -8, 0)
cam.rotation_euler = (1.5707963, 0, 0)
scene.camera = cam
bpy.context.view_layer.update()

exr = os.path.abspath(os.path.join(out_dir, "_probe_pass.exr"))
scene.render.filepath = exr[:-4]
guard("render.filepath", lambda: scene.render.filepath)
guard("ops.render.render(write_still=True)",
      lambda: (bpy.ops.render.render(write_still=True), "returned")[1])
kv("file exists (filepath + .exr)", os.path.exists(exr))
OUT["write_still_exists"] = os.path.exists(exr)
if os.path.exists(exr):
    kv("file size", os.path.getsize(exr))
    img = guard("bpy.data.images.load", lambda: bpy.data.images.load(exr))
    if img is not None:
        kv("image size / channels / depth",
           "%s / %d / %d" % (tuple(img.size), img.channels, img.depth))
        OUT["exr_channels"] = img.channels
        OUT["exr_depth"] = img.depth

# Render Result の構成
sec("6b. Render Result")
res = bpy.data.images.get("Render Result")
if res is None:
    kv("Render Result", "not found")
else:
    guard("size / channels / depth",
          lambda: "%s / %d / %d" % (tuple(res.size), res.channels, res.depth))
    guard("has_data", lambda: res.has_data)
    # レンダーパスの実体（Blender 側が何を持っているか）
    try:
        names = []
        for rl in res.render_slots:
            names.append(rl.name)
        kv("render_slots", names)
    except Exception as exc:
        kv("render_slots", "EXCEPTION %s" % exc)

# frame_path の規則（§22 の既知差分）
sec("7. filepath と frame_path の規則")
scene.render.filepath = os.path.abspath(os.path.join(out_dir, "geo_"))
guard("frame_path(frame=1)", lambda: scene.render.frame_path(frame=1))
guard("frame_path(frame=49)", lambda: scene.render.frame_path(frame=49))
kv("render.filepath", scene.render.filepath)
kv("render.use_file_extension", scene.render.use_file_extension)

# ------------------------------------------------------------------ 保存
jp = os.path.join(out_dir, "probe_compositor.json")
with open(jp, "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print()
print("=" * 70)
print("saved : %s" % jp)
print("=" * 70)
