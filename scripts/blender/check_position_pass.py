"""
check_position_pass.py -- W2a-1 Position / Depth パスの読み出し検証
                          （Blender 5.2 / Group Output 経路）

目的
  1. Position パスが返すのが World 座標かを確定させる
     World なら壁の G = 2.0 / Camera なら 12.0 / Object なら 0.0
  2. taa_render_samples = 1 と filter_size = 0.01 が効いているか
     輪郭に「手前と奥の中間値」が生成されていないか
  3. Depth パスが規約A（Z-depth）であることを、視野角のあるシーンで再確認する

Blender 5.2 で実測した制約
  - Viewer ノードは --background では中身が空（256x256 のゼロ）。GUI 限定。
  - CompositorNodeOutputFile は OPEN_EXR_MULTILAYER しか受け付けない。
    多層 EXR は bpy から層を選んで読めないので使わない。
  - scene.render.image_settings は単層 OPEN_EXR を受け付ける。
    -> コンポジタの Group Output に出したいパスを繋ぎ、
       scene の出力設定で単層 EXR として書く。
  - 書き出し先は推測せず scene.render.frame_path() に問い合わせる。

使い方
  blender --background min_scene.blend --python check_position_pass.py -- --out .

出力
  position0001.exr / depth0001.exr
  w2a1_report.json
"""

import glob
import json
import math
import os
import sys

import bpy
import numpy as np

# ---------------------------------------------------------------------------
# 1. 期待値（素の式。bpy を使わない）
# ---------------------------------------------------------------------------

RES_X = 832
RES_Y = 480
FOCAL_MM = 50.0
SENSOR_MM = 36.0
CAM_Y = -10.0

WALL_Y = 2.0
OCCLUDER_Y = -3.0
TARGET_HALF = 0.5            # 立方体 1x1x1 なので表面の Y は [-0.5, +0.5]

CHECK_INDEX = 0              # index 0 = Blender frame 1
TOL_UNITS = 0.02             # 1画素 = 0.0104 units の約2倍
TOL_DEPTH = 0.02

OUT_DIR = "."


def px_per_unit(depth):
    return RES_X / (depth * SENSOR_MM / FOCAL_MM)


def depth_of(world_y):
    return world_y - CAM_Y


# ---------------------------------------------------------------------------
# 2. コンポジタと出力設定
# ---------------------------------------------------------------------------

def build_compositor(scene):
    """5.x のコンポジタは独立データブロック。scene.node_tree は削除済み。

    Composite ノードも無いので Group Output を最終出力として使う。
    """
    tree = bpy.data.node_groups.new("W2A1_Comp", "CompositorNodeTree")
    try:
        tree.interface.new_socket("Image", in_out="OUTPUT",
                                  socket_type="NodeSocketColor")
    except Exception as exc:                          # noqa: BLE001
        raise RuntimeError("Group Output 用ソケットを作れない: %s" % exc)
    scene.compositing_node_group = tree

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    rl.location = (0, 0)

    gout = tree.nodes.new("NodeGroupOutput")
    gout.location = (400, 0)
    if len(gout.inputs) == 0:
        raise RuntimeError("Group Output に入力ソケットが生成されなかった")

    return tree, rl, gout


def route(tree, rl, gout, socket_name):
    """Group Output に繋ぐパスを差し替える。"""
    for link in list(tree.links):
        if link.to_node == gout:
            tree.links.remove(link)
    if socket_name not in rl.outputs:
        return False
    tree.links.new(rl.outputs[socket_name], gout.inputs[0])
    return True


def configure_exr_output(scene, out_dir, stem):
    """単層 32bit float EXR。view transform を混入させない。"""
    st = scene.render.image_settings
    st.file_format = "OPEN_EXR"
    st.color_mode = "RGB"
    st.color_depth = "32"
    try:
        st.exr_codec = "NONE"
    except (AttributeError, TypeError):
        pass
    if hasattr(st, "color_management"):
        try:
            st.color_management = "OVERRIDE"
            st.view_settings.view_transform = "Raw"
            st.view_settings.look = "None"
            st.view_settings.exposure = 0.0
            st.view_settings.gamma = 1.0
        except (AttributeError, TypeError):
            pass
    scene.render.use_file_extension = True
    scene.render.filepath = os.path.join(out_dir, stem)


def resolve_written_path(scene, out_dir, stem, frame):
    """write_still が実際に書いた先を特定する。

    write_still=True は filepath にそのまま拡張子を足す。
    frame_path() はアニメーションレンダー用のフレーム番号付きパスを返すため
    規則が違う。両方と glob を順に当たる。
    """
    candidates = [
        os.path.join(out_dir, stem + ".exr"),
        bpy.path.abspath(scene.render.frame_path(frame=frame)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    hits = sorted(glob.glob(os.path.join(out_dir, stem + "*.exr")))
    return hits[0] if hits else None


def render_pass(scene, tree, rl, gout, out_dir, socket_name, stem, frame):
    """指定パスを単層 EXR に書き、実際の出力パスを返す。"""
    if not route(tree, rl, gout, socket_name):
        return None, "ソケット %s が存在しない" % socket_name
    configure_exr_output(scene, out_dir, stem)
    scene.frame_set(frame)

    # 古い残骸を消してから書く（前回の実行結果を拾わないため）
    for stale in glob.glob(os.path.join(out_dir, stem + "*.exr")):
        os.remove(stale)

    bpy.ops.render.render(write_still=True)

    path = resolve_written_path(scene, out_dir, stem, frame)
    if path is None:
        return None, ("書き出されなかった。探した先: %s*.exr"
                      % os.path.join(out_dir, stem))
    return path, None


# ---------------------------------------------------------------------------
# 3. EXR 読み出し
# ---------------------------------------------------------------------------

def load_exr(path):
    """(H, W, ch) float32。行0が画像の一番上。

    img.pixels は下から上に並ぶ。この反転を忘れるのが上下反転バグの発生源。
    """
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except (AttributeError, TypeError):
        pass
    w, h = img.size
    if w == 0 or h == 0:
        bpy.data.images.remove(img)
        raise RuntimeError("EXR のサイズが 0: %s" % path)
    buf = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(h, w, img.channels)[::-1].copy()
    ch = img.channels
    bpy.data.images.remove(img)
    return arr, (w, h, ch)


# ---------------------------------------------------------------------------
# 4. 検査
# ---------------------------------------------------------------------------

def histogram_of(channel, top=8):
    rounded = np.round(channel, 2)
    vals, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts)[:top]
    total = channel.size
    return [{"value": float(vals[i]), "count": int(counts[i]),
             "ratio": round(float(counts[i]) / total, 5)} for i in order]


def identify_space(hist):
    cands = {"world": WALL_Y, "camera": depth_of(WALL_Y), "object": 0.0}
    best = None
    for e in hist:
        for name, expect in cands.items():
            if abs(e["value"] - expect) < 0.05 and e["ratio"] > 0.2:
                if best is None or e["ratio"] > best["ratio"]:
                    best = {"space": name, "ratio": e["ratio"],
                            "value": e["value"]}
    return best


def check_wall_xz(pos, wall_mask):
    rows, cols = np.nonzero(wall_mask)
    if rows.size == 0:
        return None
    step = max(1, rows.size // 4000)
    rows, cols = rows[::step], cols[::step]

    s = px_per_unit(depth_of(WALL_Y))
    exp_x = ((cols + 0.5) - RES_X / 2.0) / s
    exp_z = (RES_Y / 2.0 - (rows + 0.5)) / s
    dx = np.abs(pos[rows, cols, 0] - exp_x)
    dz = np.abs(pos[rows, cols, 2] - exp_z)

    return {"n_sampled": int(rows.size),
            "max_abs_err_X": float(dx.max()),
            "max_abs_err_Z": float(dz.max()),
            "mean_abs_err_X": float(dx.mean()),
            "mean_abs_err_Z": float(dz.mean()),
            "n_over_tol": int(np.count_nonzero((dx > TOL_UNITS)
                                               | (dz > TOL_UNITS))),
            "tol": TOL_UNITS}


def check_no_averaging(pos):
    """行240のみ走査。この行は壁 / 立方体 / 遮蔽板だけでマーカーが混ざらない。"""
    row = RES_Y // 2
    g = pos[row, :, 1]
    legal = (np.abs(g - WALL_Y) < 0.05)
    legal |= (np.abs(g - OCCLUDER_Y) < 0.05)
    legal |= (g >= -TARGET_HALF - 0.05) & (g <= TARGET_HALF + 0.05)
    bad = np.nonzero(~legal)[0]
    return {"row_scanned": row,
            "legal_values": ["wall 2.0", "occluder -3.0",
                             "target [-0.5, 0.5]"],
            "n_illegal": int(bad.size),
            "illegal_cols": [int(c) for c in bad[:20]],
            "illegal_values": [round(float(g[c]), 4) for c in bad[:20]]}


def check_depth(dep, wall_mask):
    rows, cols = np.nonzero(wall_mask)
    if rows.size == 0:
        return None
    d = dep[rows, cols, 0]
    edge = (cols < 40) | (cols > RES_X - 41)
    d_edge = d[edge] if np.count_nonzero(edge) else d
    half_w_units = (RES_X / 2.0) / px_per_unit(depth_of(WALL_Y))
    theta = math.atan(half_w_units / depth_of(WALL_Y))
    return {"n_wall_px": int(rows.size),
            "median_all": float(np.median(d)),
            "median_edge": float(np.median(d_edge)),
            "expect_convention_A": depth_of(WALL_Y),
            "expect_convention_B_at_edge": depth_of(WALL_Y) / math.cos(theta),
            "edge_half_angle_deg": math.degrees(theta),
            "n_over_tol": int(np.count_nonzero(
                np.abs(d - depth_of(WALL_Y)) > TOL_DEPTH)),
            "tol": TOL_DEPTH}


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------

def parse_out_dir():
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if "--out" in rest:
            i = rest.index("--out") + 1
            if i < len(rest):
                return rest[i]
    return OUT_DIR


def main():
    out_dir = os.path.abspath(parse_out_dir())
    os.makedirs(out_dir, exist_ok=True)

    scene = bpy.context.scene
    vl = bpy.context.view_layer

    if "Wall" not in bpy.data.objects:
        raise RuntimeError(
            "Wall が見つからない。min_scene.blend を開いて実行すること:\n"
            "  blender --background min_scene.blend --python "
            "check_position_pass.py -- --out .")

    vl.use_pass_position = True
    vl.use_pass_z = True
    vl.use_pass_object_index = True

    tree, rl, gout = build_compositor(scene)
    frame = CHECK_INDEX + 1

    report = {
        "blender": bpy.app.version_string,
        "engine": scene.render.engine,
        "filter_size": scene.render.filter_size,
        "taa_render_samples": getattr(
            getattr(scene, "eevee", None), "taa_render_samples", None),
        "rlayers_sockets": [s.name for s in rl.outputs],
        "check_index": CHECK_INDEX,
        "blender_frame": frame,
        "route": "RLayers -> Group Output -> scene.render (OPEN_EXR 単層)",
    }

    print("")
    print("=" * 68)
    print("engine / filter_size / taa : %s / %s / %s"
          % (report["engine"], report["filter_size"],
             report["taa_render_samples"]))
    print("RLayers ソケット           : %s"
          % ", ".join(report["rlayers_sockets"]))
    print("-" * 68)

    paths = {}
    for socket_name, stem in (("Position", "position"), ("Depth", "depth")):
        path, err = render_pass(scene, tree, rl, gout, out_dir,
                                socket_name, stem, frame)
        paths[socket_name] = path
        if err:
            print("%-9s 失敗 : %s" % (socket_name, err))
        else:
            print("%-9s -> %s" % (socket_name, os.path.basename(path)))
    report["exr_paths"] = {k: (os.path.basename(v) if v else None)
                           for k, v in paths.items()}
    report["exr_in_dir"] = [os.path.basename(p) for p in
                            sorted(glob.glob(os.path.join(out_dir, "*.exr")))]
    print("-" * 68)

    if paths.get("Position") is None:
        print("Position が書き出せなかった。ディレクトリ内の EXR:")
        for p in report["exr_in_dir"]:
            print("   %s" % p)
        with open(os.path.join(out_dir, "w2a1_report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("report -> w2a1_report.json")
        print("=" * 68)
        return

    pos, meta = load_exr(paths["Position"])
    report["position_shape"] = list(pos.shape)
    report["position_img_meta"] = {"w": meta[0], "h": meta[1], "ch": meta[2]}

    hist = histogram_of(pos[:, :, 1])
    report["G_channel_histogram"] = hist
    verdict = identify_space(hist)
    report["space_verdict"] = verdict

    wall_mask = np.abs(pos[:, :, 1] - WALL_Y) < 0.05
    report["wall_pixel_ratio"] = round(
        float(np.count_nonzero(wall_mask)) / (RES_X * RES_Y), 5)
    report["wall_xz_check"] = check_wall_xz(pos, wall_mask)
    report["averaging_check"] = check_no_averaging(pos)

    print("G チャンネルの多数値（上位5）")
    for e in hist[:5]:
        print("   %9.2f   %8d px   %6.2f%%"
              % (e["value"], e["count"], e["ratio"] * 100))
    print("")
    if verdict:
        print("判定 : Position は %s 座標（壁で %.2f / 占有 %.1f%%）"
              % (verdict["space"], verdict["value"], verdict["ratio"] * 100))
    else:
        print("判定 : 不明。ヒストグラムを見て判断が必要")
    print("-" * 68)

    xz = report["wall_xz_check"]
    if xz:
        print("壁の X / Z 照合  標本 %d px  許容 %.3f"
              % (xz["n_sampled"], xz["tol"]))
        print("   max err X / Z : %.5f / %.5f"
              % (xz["max_abs_err_X"], xz["max_abs_err_Z"]))
        print("   許容超え      : %d px" % xz["n_over_tol"])
    print("-" * 68)

    av = report["averaging_check"]
    print("平均化の検査  行 %d を走査" % av["row_scanned"])
    print("   あり得ない値の画素 : %d" % av["n_illegal"])
    if av["n_illegal"]:
        print("   例（列 / 値）:", list(zip(av["illegal_cols"][:8],
                                            av["illegal_values"][:8])))
    print("-" * 68)

    if paths.get("Depth"):
        dep, _ = load_exr(paths["Depth"])
        report["depth_check"] = check_depth(dep, wall_mask)
        dc = report["depth_check"]
        print("Depth  壁 %d px" % dc["n_wall_px"])
        print("   中央値 全体 / 画面端 : %.4f / %.4f"
              % (dc["median_all"], dc["median_edge"]))
        print("   規約A の予想 : %.4f" % dc["expect_convention_A"])
        print("   規約B の予想 : %.4f（画面端 半画角 %.1f度）"
              % (dc["expect_convention_B_at_edge"], dc["edge_half_angle_deg"]))
        print("   許容超え     : %d px" % dc["n_over_tol"])
        print("-" * 68)

    rep_path = os.path.join(out_dir, "w2a1_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("report :", rep_path)
    print("=" * 68)


main()
