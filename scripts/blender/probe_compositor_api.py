"""
probe_compositor_api.py -- Blender 5.2 のコンポジタ API を実測する

推測せず実測する。出力を見てから check_position_pass.py を直す。

使い方
  blender --background min_scene.blend --python probe_compositor_api.py -- --out .

出力
  probe_report.txt   端末に出るものと同じ内容
"""

import os
import sys

import bpy

BASE_NODE_PROPS = {
    "bl_description", "bl_height_default", "bl_height_max", "bl_height_min",
    "bl_icon", "bl_idname", "bl_label", "bl_static_type", "bl_width_default",
    "bl_width_max", "bl_width_min", "color", "dimensions", "height",
    "hide", "inputs", "internal_links", "label", "location",
    "location_absolute", "mute", "name", "outputs", "parent",
    "select", "show_options", "show_preview", "show_texture", "type",
    "use_custom_color", "warning_propagation", "width", "rna_type",
    "is_active_output", "bl_rna",
}

lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


def dump_props(rna_owner, title, skip_base=True):
    out("--- %s ---" % title)
    for p in rna_owner.bl_rna.properties:
        if skip_base and p.identifier in BASE_NODE_PROPS:
            continue
        try:
            val = getattr(rna_owner, p.identifier)
            if p.type in ("COLLECTION",):
                val = "<collection len=%d>" % len(val)
            elif p.type == "POINTER":
                val = "<%s>" % (val.__class__.__name__ if val else "None")
            val = str(val)
            if len(val) > 60:
                val = val[:57] + "..."
        except Exception as exc:                      # noqa: BLE001
            val = "<%s>" % exc
        out("  %-28s %-10s %s" % (p.identifier, p.type, val))
    out()


def parse_out_dir():
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if "--out" in rest:
            i = rest.index("--out") + 1
            if i < len(rest):
                return rest[i]
    return "."


def main():
    out_dir = os.path.abspath(parse_out_dir())
    scene = bpy.context.scene
    vl = bpy.context.view_layer

    out("=" * 70)
    out("Blender %s" % bpy.app.version_string)
    out("engine  : %s" % scene.render.engine)
    out("=" * 70)
    out()

    # -- 1. パスを有効化して Render Layers のソケットを見る --------------
    vl.use_pass_position = True
    vl.use_pass_z = True
    vl.use_pass_object_index = True

    tree = bpy.data.node_groups.new("Probe", "CompositorNodeTree")
    try:
        tree.interface.new_socket("Image", in_out="OUTPUT",
                                  socket_type="NodeSocketColor")
    except Exception as exc:                          # noqa: BLE001
        out("interface.new_socket 失敗: %s" % exc)
    scene.compositing_node_group = tree

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene

    out("--- Render Layers の出力ソケット ---")
    for s in rl.outputs:
        out("  %-24s enabled=%s type=%s" % (s.name, s.enabled, s.type))
    out()

    # -- 2. コンポジタのノード型を列挙 -----------------------------------
    out("--- Compositor ノード型（Output / Viewer / File を含むもの）---")
    for t in dir(bpy.types):
        if not t.startswith("CompositorNode"):
            continue
        low = t.lower()
        if any(k in low for k in ("output", "viewer", "file", "composite")):
            out("  %s" % t)
    out()

    # -- 3. File Output ノードのプロパティ実測 ---------------------------
    fo = None
    try:
        fo = tree.nodes.new("CompositorNodeOutputFile")
        dump_props(fo, "CompositorNodeOutputFile のプロパティ")
        out("--- file_slots ---")
        out("  len before new = %d" % len(fo.file_slots))
        try:
            fo.file_slots.clear()
        except Exception as exc:                      # noqa: BLE001
            out("  file_slots.clear() 失敗: %s" % exc)
        try:
            slot = fo.file_slots.new("probe_")
            out("  new() OK")
            dump_props(slot, "file_slot[0] のプロパティ", skip_base=False)
        except Exception as exc:                      # noqa: BLE001
            out("  file_slots.new() 失敗: %s" % exc)
        out("--- format ---")
        if hasattr(fo, "format"):
            dump_props(fo.format, "fo.format", skip_base=False)
        else:
            out("  fo.format が存在しない")
    except Exception as exc:                          # noqa: BLE001
        out("CompositorNodeOutputFile を作れない: %s" % exc)
    out()

    # -- 4. Viewer ノードが background で動くか --------------------------
    out("--- Viewer ノードの background 動作テスト ---")
    try:
        for n in list(tree.nodes):
            if n.bl_idname == "CompositorNodeOutputFile":
                tree.nodes.remove(n)
        viewer = tree.nodes.new("CompositorNodeViewer")
        pos_socket = rl.outputs.get("Position")
        if pos_socket is None:
            out("  Position ソケットが無いので Image を繋ぐ")
            pos_socket = rl.outputs.get("Image")
        tree.links.new(pos_socket, viewer.inputs[0])

        gout = tree.nodes.new("NodeGroupOutput")
        if len(gout.inputs):
            tree.links.new(rl.outputs["Image"], gout.inputs[0])

        scene.frame_set(1)
        bpy.ops.render.render(write_still=False)

        out("  render 後の bpy.data.images:")
        for img in bpy.data.images:
            out("    name=%-20r size=%s ch=%s has_data=%s"
                % (img.name, tuple(img.size), img.channels,
                   getattr(img, "has_data", "?")))

        vimg = bpy.data.images.get("Viewer Node")
        if vimg is None:
            out("  'Viewer Node' が存在しない -> background では使えない")
        else:
            w, h = vimg.size
            out("  Viewer Node size = %d x %d ch=%d" % (w, h, vimg.channels))
            if w == 0 or h == 0:
                out("  サイズ0 -> 中身が無い")
            else:
                px = vimg.pixels
                n = len(px)
                out("  pixels len = %d (期待 %d)" % (n, w * h * vimg.channels))
                mid = ((h // 2) * w + (w // 2)) * vimg.channels
                out("  中央画素 RGBA = %s"
                    % [round(px[mid + k], 4)
                       for k in range(min(4, vimg.channels))])
                corner = ((h - 20) * w + 20) * vimg.channels
                out("  左上寄り画素 = %s"
                    % [round(px[corner + k], 4)
                       for k in range(min(4, vimg.channels))])
    except Exception as exc:                          # noqa: BLE001
        out("  Viewer テスト失敗: %s" % exc)
    out()

    # -- 5. 参考 レンダー出力設定側の経路 --------------------------------
    out("--- scene.render.image_settings の候補 ---")
    fmts = [i.identifier for i in
            bpy.types.ImageFormatSettings.bl_rna
            .properties["file_format"].enum_items]
    out("  file_format: %s" % ", ".join(f for f in fmts if "EXR" in f))
    out("  scene.render.filepath = %r" % scene.render.filepath)
    out()

    path = os.path.join(out_dir, "probe_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("")
    print("report ->", path)


main()
