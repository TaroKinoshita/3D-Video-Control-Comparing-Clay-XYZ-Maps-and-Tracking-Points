"""
check_projection_occlusion.py -- W2a-3a 順投影と遮蔽判定の検証

目的
  A. world_to_camera_view による順投影が、素の式と一致するか
  B. 順投影した画素で Position パスを読み、元の world 座標に戻るか
     （Arm C の投影 と Arm B のラスタライズ が同じ画素を指しているか）
  C. 遮蔽判定を Depth パス比較 と ray_cast の2方式で行い、
     食い違いを数える -> protocol.md §18-17 を決着させる

Blender 5.2 で実測済みの制約
  - Viewer ノードは --background で中身が空。GUI 限定。
  - CompositorNodeOutputFile は OPEN_EXR_MULTILAYER のみ。使わない。
  - RLayers -> Group Output -> scene.render.image_settings（単層 OPEN_EXR）で書く。
  - write_still=True の出力先は filepath + 拡張子。frame_path() とは規則が違う。
  - Depth パスは規約A（depth = world_Y + 10）。最大誤差 0.00195 で確認済み。

使い方
  blender --background min_scene.blend --python check_projection_occlusion.py -- --out .

出力
  w2a3/depth_XXX.exr      49フレーム分の Depth
  w2a3/position_XXX.exr   index 0 / 24 / 48 の Position
  w2a3a_report.json
"""

import glob
import json
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

# ---------------------------------------------------------------------------
# 1. 定数
# ---------------------------------------------------------------------------

RES_X = 832
RES_Y = 480
FOCAL_MM = 50.0
SENSOR_MM = 36.0
CAM_Y = -10.0
N_FRAMES = 49

TARGET_HALF = 0.5
OCCLUDER_Y = -3.0
OCCLUDER_HALF_X = 0.35
OCCLUDER_HALF_Z = 1.5

POSITION_INDICES = (0, 24, 48)

TOL_PROJ_PX = 0.01        # 投影の一致（画素）
TOL_ROUNDTRIP_XZ = 0.02   # 往復の X / Z（world unit）。半画素 = 0.005
TOL_ROUNDTRIP_Y = 0.01    # 往復の Y。平面なので厳しくできる
EPS_DEPTH = 0.01          # 遮蔽判定の許容。実測誤差 0.002 の5倍
EPS_RAY = 1e-3            # ray_cast の自己ヒット回避

SUBDIR = "w2a3"
OUT_DIR = "."


def px_per_unit(depth):
    return RES_X / (depth * SENSOR_MM / FOCAL_MM)


def depth_of(world_y):
    return world_y - CAM_Y


def project_ref(world_xyz):
    """素の式。bpy を使わない照合基準。"""
    x, y, z = world_xyz
    d = depth_of(y)
    s = px_per_unit(d)
    return (RES_X / 2.0 + x * s, RES_Y / 2.0 - z * s, d)


def project_bpy(scene, cam, world_xyz):
    """world_to_camera_view 経由。エクスポータ本体で使う予定の経路。

    戻り値の x, y は [0, 1] で原点は左下。
    v への変換で (1 - y) を掛けるのが上下反転バグの発生点。
    """
    ndc = world_to_camera_view(scene, cam, Vector(world_xyz))
    return (ndc.x * RES_X, (1.0 - ndc.y) * RES_Y, ndc.z)


# ---------------------------------------------------------------------------
# 2. 追跡点の定義
# ---------------------------------------------------------------------------

def tracked_points(target, occluder):
    """(name, kind, object, local_coord, expect) のリスト。

    expect は可視性の幾何的な期待値。None は判定しない（シルエット上）。
    """
    pts = []
    h = TARGET_HALF

    # Target の8頂点。投影の検証専用。可視性は判定しない
    for i, (sx, sy, sz) in enumerate(
            [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]):
        pts.append({
            "name": "T_vert%d" % i, "kind": "vertex", "entity": "target",
            "obj": target, "local": (sx * h, sy * h, sz * h),
            "expect": None,
        })

    # Target の6面中心。可視性が幾何的に確定する
    faces = [
        ("T_face_-X", (-h, 0.0, 0.0), "xc_gt_+0.5"),
        ("T_face_+X", (h, 0.0, 0.0), "xc_lt_-0.5"),
        ("T_face_front", (0.0, -h, 0.0), "unless_occluder"),
        ("T_face_back", (0.0, h, 0.0), "never"),
        ("T_face_bottom", (0.0, 0.0, -h), "never"),
        ("T_face_top", (0.0, 0.0, h), "never"),
    ]
    for name, local, expect in faces:
        pts.append({"name": name, "kind": "face_center", "entity": "target",
                    "obj": target, "local": local, "expect": expect})

    # 遮蔽板の4隅。Non-target。常に可視（一番手前）
    for i, (sx, sz) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        pts.append({
            "name": "O_corner%d" % i, "kind": "vertex", "entity": "non_target",
            "obj": occluder,
            "local": (sx * OCCLUDER_HALF_X, 0.0, sz * OCCLUDER_HALF_Z),
            "expect": None,
        })
    return pts


def world_of(pt):
    """matrix_world を通した world 座標。

    min_scene.blend は scale=1.0 / rotation=0 なので平行移動のみ。
    それでも matrix_world を通す（実運用と同じ経路にするため）。
    """
    return pt["obj"].matrix_world @ Vector(pt["local"])


# ---------------------------------------------------------------------------
# 3. レンダー（W2a-1 で確立した経路）
# ---------------------------------------------------------------------------

def build_compositor(scene):
    tree = bpy.data.node_groups.new("W2A3_Comp", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT",
                              socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    gout = tree.nodes.new("NodeGroupOutput")
    gout.location = (400, 0)
    return tree, rl, gout


def route(tree, rl, gout, socket_name):
    for link in list(tree.links):
        if link.to_node == gout:
            tree.links.remove(link)
    if socket_name not in rl.outputs:
        return False
    tree.links.new(rl.outputs[socket_name], gout.inputs[0])
    return True


def configure_exr(scene, out_dir, stem):
    st = scene.render.image_settings
    st.file_format = "OPEN_EXR"
    st.color_mode = "RGB"
    st.color_depth = "32"
    try:
        st.exr_codec = "ZIP"        # 可逆。49フレーム分の容量を抑える
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


def render_one(scene, tree, rl, gout, out_dir, socket_name, stem, frame):
    if not route(tree, rl, gout, socket_name):
        raise RuntimeError("ソケット %s が無い" % socket_name)
    configure_exr(scene, out_dir, stem)
    scene.frame_set(frame)
    for stale in glob.glob(os.path.join(out_dir, stem + "*.exr")):
        os.remove(stale)
    bpy.ops.render.render(write_still=True)
    path = os.path.join(out_dir, stem + ".exr")
    if os.path.exists(path):
        return path
    hits = sorted(glob.glob(os.path.join(out_dir, stem + "*.exr")))
    if hits:
        return hits[0]
    raise RuntimeError("書き出されなかった: %s" % stem)


def load_exr(path):
    """(H, W, ch) float32。行0が画像の一番上。pixels は下から上なので反転する。"""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except (AttributeError, TypeError):
        pass
    w, h = img.size
    buf = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(h, w, img.channels)[::-1].copy()
    bpy.data.images.remove(img)
    return arr


# ---------------------------------------------------------------------------
# 4. 遮蔽判定 2方式
# ---------------------------------------------------------------------------

def sample_pixel(arr, u, v):
    """連続座標 (u, v) を含む画素を返す。範囲外は None。"""
    col = int(math.floor(u))
    row = int(math.floor(v))
    if col < 0 or col >= RES_X or row < 0 or row >= RES_Y:
        return None, (col, row)
    return arr[row, col], (col, row)


def occluded_by_depth(depth_arr, u, v, point_depth):
    """方式1 Depth パス比較。点の深度が、その画素の表面より奥なら遮蔽。"""
    val, rc = sample_pixel(depth_arr, u, v)
    if val is None:
        return None, None, rc
    scene_depth = float(val[0])
    return (point_depth > scene_depth + EPS_DEPTH), scene_depth, rc


def occluded_by_raycast(scene, depsgraph, cam_loc, world_pt):
    """方式2 ray_cast。カメラから点の少し手前までにヒットがあれば遮蔽。"""
    vec = world_pt - cam_loc
    dist = vec.length
    if dist <= EPS_RAY:
        return None
    hit, _loc, _nor, _idx, _obj, _mat = scene.ray_cast(
        depsgraph, cam_loc, vec.normalized(), distance=dist - EPS_RAY)
    return bool(hit)


# ---------------------------------------------------------------------------
# 5. 期待値の判定
# ---------------------------------------------------------------------------

def expected_visible(expect, xc, index, occ_u_range, u):
    """幾何から確定する可視性。判定できない場合は None。"""
    if expect is None:
        return None
    if expect == "never":
        return False
    hidden_by_occ = occ_u_range[0] <= u <= occ_u_range[1]
    if expect == "unless_occluder":
        return not hidden_by_occ
    if expect == "xc_lt_-0.5":
        return (xc < -0.5) and not hidden_by_occ
    if expect == "xc_gt_+0.5":
        return (xc > 0.5) and not hidden_by_occ
    return None


# ---------------------------------------------------------------------------
# 6. main
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
    base = os.path.abspath(parse_out_dir())
    out_dir = os.path.join(base, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    scene = bpy.context.scene
    vl = bpy.context.view_layer
    for name in ("Wall", "Target", "Occluder", "Cam"):
        if name not in bpy.data.objects:
            raise RuntimeError(
                "%s が無い。min_scene.blend を開いて実行すること。" % name)

    target = bpy.data.objects["Target"]
    occluder = bpy.data.objects["Occluder"]
    cam = bpy.data.objects["Cam"]

    vl.use_pass_position = True
    vl.use_pass_z = True
    tree, rl, gout = build_compositor(scene)

    pts = tracked_points(target, occluder)
    occ_s = px_per_unit(depth_of(OCCLUDER_Y))
    occ_u_range = (RES_X / 2.0 - OCCLUDER_HALF_X * occ_s,
                   RES_X / 2.0 + OCCLUDER_HALF_X * occ_s)

    report = {
        "blender": bpy.app.version_string,
        "engine": scene.render.engine,
        "filter_size": scene.render.filter_size,
        "taa_render_samples": getattr(
            getattr(scene, "eevee", None), "taa_render_samples", None),
        "n_points": len(pts),
        "occluder_u_range": list(occ_u_range),
        "tolerances": {"proj_px": TOL_PROJ_PX,
                       "roundtrip_xz": TOL_ROUNDTRIP_XZ,
                       "roundtrip_y": TOL_ROUNDTRIP_Y,
                       "eps_depth": EPS_DEPTH, "eps_ray": EPS_RAY},
    }

    # === 検査A 投影の一致 =================================================
    proj_err = []
    for index in range(N_FRAMES):
        scene.frame_set(index + 1)
        for pt in pts:
            w = world_of(pt)
            ub, vb, _ = project_bpy(scene, cam, w)
            ur, vr, _ = project_ref((w.x, w.y, w.z))
            proj_err.append((abs(ub - ur), abs(vb - vr), index, pt["name"]))

    max_u = max(proj_err, key=lambda e: e[0])
    max_v = max(proj_err, key=lambda e: e[1])
    n_over = sum(1 for e in proj_err
                 if e[0] > TOL_PROJ_PX or e[1] > TOL_PROJ_PX)
    report["check_A_projection"] = {
        "n_compared": len(proj_err),
        "max_err_u": {"value": max_u[0], "index": max_u[2], "point": max_u[3]},
        "max_err_v": {"value": max_v[0], "index": max_v[2], "point": max_v[3]},
        "n_over_tol": n_over,
    }

    # === 検査B Position パスとの往復 ======================================
    roundtrip = []
    for index in POSITION_INDICES:
        frame = index + 1
        path = render_one(scene, tree, rl, gout, out_dir,
                          "Position", "position_%03d" % index, frame)
        pos = load_exr(path)
        scene.frame_set(frame)
        for pt in pts:
            if pt["kind"] != "face_center":
                continue
            w = world_of(pt)
            ub, vb, _ = project_bpy(scene, cam, w)
            val, rc = sample_pixel(pos, ub, vb)
            if val is None:
                continue
            roundtrip.append({
                "index": index, "point": pt["name"],
                "u": ub, "v": vb, "col": rc[0], "row": rc[1],
                "world": [w.x, w.y, w.z],
                "readback": [float(val[0]), float(val[1]), float(val[2])],
                "err_x": abs(float(val[0]) - w.x),
                "err_y": abs(float(val[1]) - w.y),
                "err_z": abs(float(val[2]) - w.z),
                "expect": pt["expect"],
            })
    report["check_B_roundtrip"] = roundtrip

    # === 検査C 遮蔽判定の2方式 ============================================
    depth_paths = {}
    for index in range(N_FRAMES):
        frame = index + 1
        depth_paths[index] = render_one(
            scene, tree, rl, gout, out_dir, "Depth", "depth_%03d" % index,
            frame)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = cam.matrix_world.translation.copy()

    rows = []
    for index in range(N_FRAMES):
        frame = index + 1
        scene.frame_set(frame)
        depsgraph.update()
        dep = load_exr(depth_paths[index])
        xc = target.matrix_world.translation.x
        for pt in pts:
            w = world_of(pt)
            ub, vb, pd = project_bpy(scene, cam, w)
            occ_d, scene_d, rc = occluded_by_depth(dep, ub, vb,
                                                   depth_of(w.y))
            occ_r = occluded_by_raycast(scene, depsgraph, cam_loc, w)
            exp = expected_visible(pt["expect"], xc, index, occ_u_range, ub)
            rows.append({
                "index": index, "point": pt["name"], "kind": pt["kind"],
                "u": ub, "v": vb, "point_depth": depth_of(w.y),
                "scene_depth": scene_d,
                "vis_depth": (None if occ_d is None else (not occ_d)),
                "vis_ray": (None if occ_r is None else (not occ_r)),
                "expect_visible": exp,
            })
    report["check_C_rows"] = rows

    # 集計
    def agree(r):
        return (r["vis_depth"] is not None and r["vis_ray"] is not None
                and r["vis_depth"] == r["vis_ray"])

    comparable = [r for r in rows
                  if r["vis_depth"] is not None and r["vis_ray"] is not None]
    disagree = [r for r in comparable if not agree(r)]
    fc = [r for r in rows if r["kind"] == "face_center"
          and r["expect_visible"] is not None]
    d_wrong = [r for r in fc if r["vis_depth"] != r["expect_visible"]]
    r_wrong = [r for r in fc if r["vis_ray"] != r["expect_visible"]]

    report["check_C_summary"] = {
        "n_rows": len(rows),
        "n_comparable": len(comparable),
        "n_disagree": len(disagree),
        "disagree_sample": disagree[:20],
        "n_face_center_judged": len(fc),
        "depth_method_wrong": len(d_wrong),
        "ray_method_wrong": len(r_wrong),
        "depth_wrong_sample": d_wrong[:10],
        "ray_wrong_sample": r_wrong[:10],
    }

    # 前面中心の隠れ区間
    front = [r for r in rows if r["point"] == "T_face_front"]
    hidden_d = [r["index"] for r in front if r["vis_depth"] is False]
    hidden_r = [r["index"] for r in front if r["vis_ray"] is False]
    report["front_face_hidden"] = {
        "by_depth": [hidden_d[0], hidden_d[-1]] if hidden_d else [],
        "by_ray": [hidden_r[0], hidden_r[-1]] if hidden_r else [],
        "n_by_depth": len(hidden_d), "n_by_ray": len(hidden_r),
    }

    # === 出力 =============================================================
    with open(os.path.join(base, "w2a3a_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    a = report["check_A_projection"]
    print("")
    print("=" * 70)
    print("engine / filter_size / taa : %s / %.4f / %s"
          % (report["engine"], report["filter_size"],
             report["taa_render_samples"]))
    print("追跡点 %d / フレーム %d / 遮蔽板の u 範囲 %.2f .. %.2f"
          % (len(pts), N_FRAMES, occ_u_range[0], occ_u_range[1]))
    print("-" * 70)
    print("検査A 順投影の一致（world_to_camera_view vs 素の式）")
    print("   比較数     : %d" % a["n_compared"])
    print("   max err u  : %.6f px  (index %d / %s)"
          % (a["max_err_u"]["value"], a["max_err_u"]["index"],
             a["max_err_u"]["point"]))
    print("   max err v  : %.6f px  (index %d / %s)"
          % (a["max_err_v"]["value"], a["max_err_v"]["index"],
             a["max_err_v"]["point"]))
    print("   許容超え   : %d  (許容 %.3f px)" % (a["n_over_tol"], TOL_PROJ_PX))
    print("-" * 70)
    print("検査B Position パスとの往復（面中心のみ）")
    print("  idx point           col  row |  err X    err Y    err Z  | 期待")
    for r in roundtrip:
        print("  %3d %-15s %4d %4d | %7.5f %7.5f %7.5f | %s"
              % (r["index"], r["point"], r["col"], r["row"],
                 r["err_x"], r["err_y"], r["err_z"], r["expect"]))
    print("-" * 70)
    c = report["check_C_summary"]
    print("検査C 遮蔽判定")
    print("   判定行数           : %d" % c["n_rows"])
    print("   両方式が比較可能   : %d" % c["n_comparable"])
    print("   両方式の食い違い   : %d" % c["n_disagree"])
    print("   幾何期待値と照合可 : %d（面中心のみ）" % c["n_face_center_judged"])
    print("   Depth 方式の誤り   : %d" % c["depth_method_wrong"])
    print("   ray_cast の誤り    : %d" % c["ray_method_wrong"])
    fh = report["front_face_hidden"]
    print("")
    print("   前面中心が隠れる index")
    print("     Depth 方式 : %s (%d フレーム)" % (fh["by_depth"], fh["n_by_depth"]))
    print("     ray_cast   : %s (%d フレーム)" % (fh["by_ray"], fh["n_by_ray"]))
    if c["n_disagree"]:
        print("")
        print("   食い違いの例")
        print("   idx point           |     u      pd     sd  | depth ray")
        for r in c["disagree_sample"][:12]:
            print("   %3d %-15s | %7.2f %6.3f %6.3f |   %-5s %-5s"
                  % (r["index"], r["point"], r["u"], r["point_depth"],
                     -1 if r["scene_depth"] is None else r["scene_depth"],
                     r["vis_depth"], r["vis_ray"]))
    print("-" * 70)
    print("report : %s" % os.path.join(base, "w2a3a_report.json"))
    print("=" * 70)


main()
