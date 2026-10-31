"""
select_arm_c.py  —  W2c-3a
Arm C の候補点生成 / 可視判定 / 最小間隔選定 / 彩色 / n·v 記録。**描画はしない。**

  & $BL --background --factory-startup --python select_arm_c.py "--" ^
        --blend scene_c_original.blend --out armC --pitch 0.10

出力  armC/armc_points.json          点集合・色・per-frame の u,v / 可視 / n·v
      armC/select_arm_c.json         SELF-CHECK 結果

【新規に決めたこと（2026-08-05）】
点集合と色は **両変種で共有する。** 選定と彩色は Original / Edited の
49+49 = 98 フレームの和で行う。変種ごとに選び直すと、編集と無関係な差が
Control Video に入って B vs C の比較が壊れる（§13「同じ点＝同じ色」の系）。

軌道は .blend の fcurve ではなく scene_c_spec.KEYFRAMES から直接与える。
Arm C を fcurve API に依存させないため。値は scene_c_expect の軌道表と突合する。
"""
import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

import scene_c_spec as S
import scene_c_expect as E
import armc_palette as PAL
import armc_select as SEL

W, H = S.RENDER["resolution_x"], S.RENDER["resolution_y"]
K = S.K_PX_PER_UNIT
EPS = 1e-3                     # ray_cast の自己ヒット回避（protocol §13.4）
RADIUS = 6                     # 点の半径
ASSIGN_RADIUS = 24.0           # 分離割当半径（衝突グラフ）
# 最小スクリーン間隔の下限。
# 半径6・マスク dx^2+dy^2<=36 では、整数中心の距離が 12.0 px 以下のときだけ
# 画素を共有する。連続座標を整数に丸めると距離は最大 1.4142 px 縮むので、
# 重なりを保証で無くすには連続分離 > 13.4142 px が必要（--d-floor で上書き可）。
D_FLOOR_DEFAULT = 12.0
D_FLOOR_NO_OVERLAP = 13.4143
N_TARGET = 72                  # Target 点数（占有率 29.4% < 警戒線 30%）
N_BG = 168                     # 合計 240、Target 割合 = 0.30 ちょうど

# Entity ごとの最低点数（2026-08-05 追加）。
# 面積比だけで配ると Wall が支配し、Marker 0 点 / DeepObject 6 点になった。
# DeepObject は §12「非対象物体の位置変化」の基準、Marker は反転検出の基準
# なので、先に確保してから残りを最遠点サンプリングで配る。
QUOTA = {"DeepObject": 8, "Occluder": 8, "Marker": 2}
RESULTS = {"checks": [], "info": {}}


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-32s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def args():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []

    def opt(k, d):
        return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else d

    return (os.path.abspath(opt("--blend", "scene_c_original.blend")),
            os.path.abspath(opt("--out", "armC")),
            float(opt("--pitch", "0.10")),
            int(opt("--n-target", str(N_TARGET))),
            int(opt("--n-bg", str(N_BG))),
            float(opt("--d-floor", str(D_FLOOR_DEFAULT))))


BLEND, OUT, PITCH, NT, NB, D_FLOOR = args()


# ---------------------------------------------------------------- 軌道
def target_x(variant, idx):
    kfs = sorted(S.KEYFRAMES[variant])
    frame = idx + 1
    if frame <= kfs[0][0]:
        return kfs[0][1]
    if frame >= kfs[-1][0]:
        return kfs[-1][1]
    for (f0, x0), (f1, x1) in zip(kfs, kfs[1:]):
        if f0 <= frame <= f1:
            return x0 if f1 == f0 else x0 + (x1 - x0) * (frame - f0) / (f1 - f0)
    return kfs[-1][1]


# ---------------------------------------------------------------- 候補点
def face_grid(verts, normal, pitch):
    """凸多角形の面上に pitch 間隔の格子点を作る。verts は Vector のリスト。"""
    n = normal.normalized()
    e1 = (verts[1] - verts[0])
    if e1.length < 1e-9:
        return []
    e1 = e1.normalized()
    e2 = n.cross(e1).normalized()
    o = verts[0]
    p2 = [((v - o).dot(e1), (v - o).dot(e2)) for v in verts]
    xs = [p[0] for p in p2]
    ys = [p[1] for p in p2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    # 頂点差の誤差で ext/pitch が真値の下に振れると格子が1行少なくなる。
    # mathutils.Vector は **float32** なので誤差は 1e-6 オーダー。
    #   Marker X: 2.60-2.40 -> q = 1.99999809 -> floor 1（本来 2）
    #   Target Z: -0.10-(-1.50) -> q = 13.99999976 -> floor 13（本来 14）
    # 初版で 1e-9 を使ったが3桁小さく、候補数が1つも変わらなかった。
    # 1e-4 なら q=11.5（遮蔽板）や q=90（壁）には影響しない。
    EPS_GRID = 1e-4
    nx = max(1, int(math.floor((x1 - x0) / pitch + EPS_GRID)))
    ny = max(1, int(math.floor((y1 - y0) / pitch + EPS_GRID)))
    # bbox の中央に格子を寄せる（面ごとに決定的）
    ox = x0 + ((x1 - x0) - (nx - 1) * pitch) / 2.0
    oy = y0 + ((y1 - y0) - (ny - 1) * pitch) / 2.0
    out = []
    m = len(p2)
    for iy in range(ny):
        for ix in range(nx):
            px, py = ox + ix * pitch, oy + iy * pitch
            inside = True
            sign = 0
            for k in range(m):
                ax, ay = p2[k]
                bx, by = p2[(k + 1) % m]
                cr = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                if abs(cr) < 1e-12:
                    continue
                s = 1 if cr > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    inside = False
                    break
            if inside:
                out.append(o + e1 * px + e2 * py)
    return out


def face_extents(verts, normal, pitch):
    """診断用：face_grid と同じ基底で 2D の extent と nx / ny を返す。"""
    n = normal.normalized()
    e1 = (verts[1] - verts[0])
    if e1.length < 1e-9:
        return None
    e1 = e1.normalized()
    e2 = n.cross(e1).normalized()
    o = verts[0]
    p2 = [((v - o).dot(e1), (v - o).dot(e2)) for v in verts]
    ex = max(p[0] for p in p2) - min(p[0] for p in p2)
    ey = max(p[1] for p in p2) - min(p[1] for p in p2)
    return (ex, ey,
            max(1, int(math.floor(ex / pitch + 1e-4))),
            max(1, int(math.floor(ey / pitch + 1e-4))))


def build_candidates(objs, pitch):
    """(coords, normals, obj_names, pass_indices, face_log)。
    Target は location.x を含まない。"""
    co, nr, on, pi = [], [], [], []
    flog = {}
    for ob in objs:
        me = ob.data
        rows = []
        for k, poly in enumerate(me.polygons):
            vs = [me.vertices[i].co.copy() for i in poly.vertices]
            fe = face_extents(vs, poly.normal.copy(), pitch)
            pts = face_grid(vs, poly.normal.copy(), pitch)
            rows.append({"face": k, "n_verts": len(vs),
                         "ext": [round(fe[0], 6), round(fe[1], 6)] if fe else None,
                         "nx_ny": [fe[2], fe[3]] if fe else None,
                         "points": len(pts)})
            for p in pts:
                co.append((p.x, p.y, p.z))
                nr.append((poly.normal.x, poly.normal.y, poly.normal.z))
                on.append(ob.name)
                pi.append(ob.pass_index)
        flog[ob.name] = rows
    return (np.array(co, dtype=np.float64), np.array(nr, dtype=np.float64),
            on, np.array(pi, dtype=np.int32), flog)


# ---------------------------------------------------------------- 投影 / 可視
def project(scene, cam, p):
    ndc = world_to_camera_view(scene, cam, Vector(p))
    return ndc.x * W, (1.0 - ndc.y) * H, ndc.z


def visible(scene, dg, cam_loc, p):
    """protocol §13.4：scene.ray_cast。点の手前 EPS で止める。"""
    v = Vector(p) - cam_loc
    L = v.length
    if L <= EPS:
        return False
    hit = scene.ray_cast(dg, cam_loc, v.normalized(), distance=L - EPS)
    return not hit[0]


def main():
    print("blend  : %s" % BLEND)
    print("out    : %s" % OUT)
    print("pitch  : %.3f" % PITCH)
    print("N      : Target %d / Non-target %d = %d（Target 割合 %.4f）"
          % (NT, NB, NT + NB, NT / float(NT + NB)))
    os.makedirs(OUT, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=BLEND)
    scene = bpy.context.scene
    cam = scene.camera
    tgt = bpy.data.objects["Target"]
    objs = [o for o in bpy.data.objects if o.type == "MESH"]
    objs.sort(key=lambda o: o.pass_index)

    # 軌道は fcurve に依存させない
    tgt.animation_data_clear()
    bpy.context.view_layer.update()
    cam_loc = cam.matrix_world.translation.copy()
    rec("camera_location", (abs(cam_loc.x) < 1e-6 and abs(cam_loc.y + 10.0) < 1e-6
                            and abs(cam_loc.z) < 1e-6),
        "(%.4f, %.4f, %.4f)" % (cam_loc.x, cam_loc.y, cam_loc.z))

    # ---- 候補点 ----
    co, nr, on, pidx, flog = build_candidates(objs, PITCH)
    RESULTS["info"]["face_log"] = flog
    for name in sorted(flog):
        tot = sum(r["points"] for r in flog[name])
        if tot <= 60:
            print("  [%s] 面ごとの ext / nx x ny / 点数: %s"
                  % (name, [(r["ext"], r["nx_ny"], r["points"])
                            for r in flog[name]]))
    M0 = len(co)
    RESULTS["info"]["candidates_raw"] = M0
    print("候補点（生）: %d" % M0)

    variants = list(S.VARIANTS)
    F = len(variants) * S.FRAMES
    uv = np.zeros((M0, F), dtype=np.float64)
    uv2 = np.zeros((M0, F), dtype=np.float64)
    vis = np.zeros((M0, F), dtype=bool)
    nv = np.zeros((M0, F), dtype=np.float32)
    dep = np.zeros((M0, F), dtype=np.float64)   # 描画順（深度の遠い順）に必要

    is_tgt = (pidx == 3)
    print("Target 上の候補 %d / 静止物体 %d" % (int(is_tgt.sum()), int((~is_tgt).sum())))

    # 軌道の突合
    bad_traj = []
    for vi, var in enumerate(variants):
        table = E.TRAJ_EDITED if var == "edited" else E.TRAJ_ORIGINAL
        for idx, _f, ex, _u in table:
            # リテラルは %+.7f で丸めてあるので 1e-9 では厳しすぎる（差 3.1e-8）
            if abs(target_x(var, idx) - ex) > E.TOL_X:
                bad_traj.append((var, idx))
    rec("trajectory_matches_literals", not bad_traj, str(bad_traj[:5]))

    # ---- 可視判定 ----
    # 静止物体は u,v / n·v がフレーム不変。遮蔽が変わるのは Target が
    # その画素を覆うときだけ（statics_never_move を W2b で機械検査済み）。
    # よって「静止物体だけの遮蔽」を1回求め、毎フレームは Target の
    # シルエット bbox 内に落ちる静止点だけ再判定する。
    st = np.where(~is_tgt)[0]
    tg = np.where(is_tgt)[0]

    PARK = 1.0e5
    tgt.location = (PARK, 0.0, 0.0)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    st_u = np.zeros(len(st)); st_v = np.zeros(len(st))
    st_d = np.zeros(len(st))
    st_nv = np.zeros(len(st)); st_ok = np.zeros(len(st), dtype=bool)
    for k, m in enumerate(st):
        p = co[m]
        u, v, d = project(scene, cam, p)
        st_u[k], st_v[k], st_d[k] = u, v, d
        if not (0.0 <= u <= W and 0.0 <= v <= H) or d <= 0.0:
            continue
        view = Vector(p) - cam_loc
        nvv = -float(Vector(nr[m]).normalized().dot(view.normalized()))
        st_nv[k] = nvv
        if nvv <= 0.0:
            continue
        st_ok[k] = visible(scene, dg, cam_loc, p)
    n_static_base = int(st_ok.sum())
    print("静止物体：画面内かつ表向きかつ静止遮蔽なし = %d / %d"
          % (n_static_base, len(st)))
    RESULTS["info"]["static_visible_base"] = n_static_base

    # Target メッシュ頂点（bbox 用）
    tverts = [np.array([v.co.x, v.co.y, v.co.z]) for v in tgt.data.vertices]

    n_raycast = 0
    col = 0
    for vi, var in enumerate(variants):
        for idx in range(S.FRAMES):
            tx = target_x(var, idx)
            tgt.location = (tx, 0.0, 0.0)
            scene.frame_set(idx + 1)
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            off = np.array([tx, 0.0, 0.0])

            # Target のシルエット bbox（1px 余裕）
            tu, tv = [], []
            for vc in tverts:
                u, v, _d = project(scene, cam, vc + off)
                tu.append(u); tv.append(v)
            bu0, bu1 = min(tu) - 1.0, max(tu) + 1.0
            bv0, bv1 = min(tv) - 1.0, max(tv) + 1.0

            # 静止物体：キャッシュを流用し、bbox 内だけ再判定
            uv[st, col] = st_u
            uv2[st, col] = st_v
            dep[st, col] = st_d
            nv[st, col] = st_nv
            vis[st, col] = st_ok
            inbox = np.where(st_ok & (st_u >= bu0) & (st_u <= bu1)
                             & (st_v >= bv0) & (st_v <= bv1))[0]
            for k in inbox:
                m = st[k]
                vis[m, col] = visible(scene, dg, cam_loc, co[m])
                n_raycast += 1

            # Target 上の点：毎フレーム全部
            for m in tg:
                p = co[m] + off
                u, v, d = project(scene, cam, p)
                uv[m, col] = u
                uv2[m, col] = v
                dep[m, col] = d
                if not (0.0 <= u <= W and 0.0 <= v <= H) or d <= 0.0:
                    continue
                view = Vector(p) - cam_loc
                nvv = -float(Vector(nr[m]).normalized().dot(view.normalized()))
                nv[m, col] = nvv
                if nvv <= 0.0:
                    continue
                vis[m, col] = visible(scene, dg, cam_loc, p)
                n_raycast += 1
            col += 1
        print("  %s: %d frames done（累計 ray_cast %d）"
              % (var, S.FRAMES, n_raycast))
    RESULTS["info"]["raycast_count"] = n_raycast

    # 最適化の検証：bbox 外の静止点をフルに再判定して一致を確かめる
    rng = np.random.default_rng(0)
    mism = []
    for var_i, var in enumerate(variants):
        for idx in (0, 12, 23, 36, 41, 48):
            c = var_i * S.FRAMES + idx
            tx = target_x(var, idx)
            tgt.location = (tx, 0.0, 0.0)
            scene.frame_set(idx + 1)
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            pool = np.where(st_nv > 0.0)[0]
            if len(pool) == 0:
                continue
            for k in rng.choice(pool, size=min(60, len(pool)), replace=False):
                m = st[k]
                got = visible(scene, dg, cam_loc, co[m]) if st_nv[k] > 0 else False
                if bool(got) != bool(vis[m, c]):
                    mism.append((var, idx, int(m)))
    rec("static_cache_matches_full_raycast", not mism,
        "不一致 %d 件 %s" % (len(mism), mism[:5]))

    # depth = world_Y + 10 の規約（静止物体で確認。Target は毎フレーム変わる Y を持つ）
    dev = 0.0
    for k, m in enumerate(st):
        if st_nv[k] > 0.0:
            dev = max(dev, abs(st_d[k] - (co[m][1] + 10.0)))
    rec("depth_equals_Y_plus_10", dev <= 1e-4, "最大偏差 %.3e" % dev)

    ever = vis.any(axis=1)
    UV = np.stack([uv, uv2], axis=2)     # 診断で使うのでここで作る
    RESULTS["info"]["candidates_ever_visible"] = int(ever.sum())
    print("一度でも可視な候補: %d" % int(ever.sum()))

    # ---- 診断: オブジェクトごとの候補数 ----
    obj_names = sorted(set(on))
    diag = {}
    for name in obj_names:
        sel_m = np.array([i for i, o in enumerate(on) if o == name], dtype=int)
        diag[name] = {"raw": int(len(sel_m)),
                      "ever_visible": int(ever[sel_m].sum()),
                      "faces": len([1 for ob in objs if ob.name == name
                                    for _ in ob.data.polygons])}
    RESULTS["info"]["candidates_per_object"] = diag
    print("候補数（生 / 一度でも可視 / 面数）:")
    for name in obj_names:
        d_ = diag[name]
        print("  %-11s %6d / %6d / %3d"
              % (name, d_["raw"], d_["ever_visible"], d_["faces"]))

    # 候補が少ないオブジェクトは中身をダンプして原因を追えるようにする
    small = {}
    for name in obj_names:
        sel_m = np.array([i for i, o in enumerate(on) if o == name], dtype=int)
        vis_m = sel_m[ever[sel_m]]
        if len(vis_m) > 40 or len(vis_m) == 0:
            continue
        rows = []
        for m in vis_m:
            rows.append({"mesh_co": [round(float(x), 4) for x in co[m]],
                         "normal": [round(float(x), 3) for x in nr[m]],
                         "u": round(float(uv[m, 0]), 2),
                         "v": round(float(uv2[m, 0]), 2),
                         "nv": round(float(nv[m, 0]), 4),
                         "n_visible_frames": int(vis[m].sum())})
        dm = []
        for a_i, a in enumerate(vis_m):
            d_ = SEL.pair_min_distance(UV, vis, int(a))
            for b in vis_m[a_i + 1:]:
                dm.append(round(float(d_[int(b)]), 3))
        small[name] = {"points": rows,
                       "pairwise_min_distance": sorted(dm)[:20],
                       "n_pairs_above_floor":
                           int(sum(1 for x in dm if x >= D_FLOOR))}
        print("  [%s] 可視候補 %d 点 / ペア距離の小さい順 %s / 下限 %.2f 以上のペア %d"
              % (name, len(vis_m), sorted(dm)[:6], D_FLOOR,
                 small[name]["n_pairs_above_floor"]))
    RESULTS["info"]["small_objects"] = small

    # ---- 選定（Target 帯を先に。距離は帯をまたいで効かせる） ----
    obj_arr = np.array(on)
    pool_t = np.where(ever & is_tgt)[0]
    pool_b = np.where(ever & ~is_tgt)[0]

    sel_t = SEL.fps_select(UV, vis, NT, pool=pool_t, min_dist=D_FLOOR)
    acc = [int(x) for x in sel_t]

    quota_sel = []
    quota_got = {}
    for name in sorted(QUOTA):
        pool_o = np.where(ever & ~is_tgt & (obj_arr == name))[0]
        got = SEL.fps_select(UV, vis, QUOTA[name], pool=pool_o,
                             preselected=acc, min_dist=D_FLOOR)
        quota_got[name] = len(got)
        quota_sel += [int(x) for x in got]
        acc += [int(x) for x in got]
    rec("quota_met", all(quota_got[k] == QUOTA[k] for k in QUOTA),
        " / ".join("%s %d/%d" % (k, quota_got[k], QUOTA[k]) for k in sorted(QUOTA)))

    rest = SEL.fps_select(UV, vis, NB - len(quota_sel), pool=pool_b,
                          preselected=acc, min_dist=D_FLOOR)
    sel_b = np.array(quota_sel + [int(x) for x in rest], dtype=int)
    sel_all = np.concatenate([sel_t, sel_b])

    rec("selected_counts", len(sel_t) == NT and len(sel_b) == NB,
        "Target %d/%d / Non-target %d/%d" % (len(sel_t), NT, len(sel_b), NB))
    ratio = len(sel_t) / float(max(1, len(sel_all)))
    rec("target_ratio_0.30", abs(ratio - 0.30) < 0.005, "%.4f" % ratio)

    ach_t = SEL.min_separation(UV, vis, sel_t)
    ach_b = SEL.min_separation(UV, vis, sel_b)
    ach_all = SEL.min_separation(UV, vis, sel_all)
    RESULTS["info"]["achieved_separation"] = {
        "target": ach_t, "background": ach_b, "global": ach_all}
    rec("separation_above_floor", ach_all >= D_FLOOR,
        "global %.3f px（下限 %.1f）/ Target %.3f / Non-target %.3f"
        % (ach_all, D_FLOOR, ach_t, ach_b))

    rec("separation_no_overlap_guaranteed", ach_all > D_FLOOR_NO_OVERLAP,
        "global %.3f px（重なり無しの保証には > %.4f が必要）"
        % (ach_all, D_FLOOR_NO_OVERLAP))

    percount = {}
    for m in sel_all:
        percount[on[int(m)]] = percount.get(on[int(m)], 0) + 1
    RESULTS["info"]["points_per_object"] = percount
    print("オブジェクトごとの点数: %s" % dict(sorted(percount.items())))
    rec("all_entities_represented",
        all(percount.get(o.name, 0) > 0 for o in objs),
        str({o.name: percount.get(o.name, 0) for o in objs}))

    # ---- 彩色（帯ごとに独立） ----
    coloring = {}
    for band, sel in (("target", sel_t), ("background", sel_b)):
        pairs = SEL.collision_pairs(UV, vis, sel, ASSIGN_RADIUS)
        cmap, nc, failed = SEL.greedy_color([int(x) for x in sel], pairs, 12)
        bad = SEL.verify_coloring(cmap, pairs)
        rec("coloring_%s" % band, not failed and not bad,
            "%d ペア / %d 色使用 / failed=%s / 同色隣接=%d"
            % (len(pairs), nc, failed, len(bad)))
        deg = {}
        for a, b in pairs:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        hist = {}
        for c_ in cmap.values():
            hist[c_] = hist.get(c_, 0) + 1
        RESULTS["info"]["band_" + band] = {
            "n": len(sel), "pairs": len(pairs), "colors_used": nc,
            "max_degree": max(deg.values()) if deg else 0,
            "color_histogram": dict(sorted(hist.items()))}
        rec("color_spread_%s" % band, nc == 12 or len(sel) < 12,
            "%d 色使用 / 内訳 %s" % (nc, dict(sorted(hist.items()))))
        pal = PAL.PALETTE[band]
        # 隣接ペアの色間距離
        dmin = min((PAL.dist(pal[cmap[a]], pal[cmap[b]]) for a, b in pairs),
                   default=float("inf"))
        rec("color_distance_%s" % band, dmin >= PAL.MIN_DIST - 1e-6,
            "隣接ペアの最小色間距離 %.4f（>= %.4f）" % (dmin, PAL.MIN_DIST))
        for n_, c_ in cmap.items():
            coloring[n_] = {"band": band, "color_index": c_,
                            "rgb": list(pal[c_])}

    # ---- 占有率 ----
    area, occ = SEL.occupancy(UV, vis, sel_all, RADIUS, W, H)
    rec("disc_area_113", area == 113, area)
    rec("occupancy_below_30", float(occ.max()) < 0.30,
        "最大 %.3f%%（全画面）" % (occ.max() * 100))
    tgt_area = 27669.0
    occ_t = NT * area / tgt_area
    rec("target_local_occupancy", occ_t < 0.30,
        "Target 上 %.2f%%（面積 %d px 前提）" % (occ_t * 100, int(tgt_area)))
    RESULTS["info"]["occupancy_max_frame"] = float(occ.max())
    RESULTS["info"]["occupancy_target_local"] = occ_t

    # ---- n·v の分布 ----
    nvv = nv[np.ix_(sel_all, np.arange(F))][vis[sel_all]]
    if nvv.size:
        RESULTS["info"]["nv"] = {
            "min": float(nvv.min()), "p05": float(np.percentile(nvv, 5)),
            "median": float(np.median(nvv)), "max": float(nvv.max()),
            "below_0.2": int((nvv < 0.2).sum()), "n": int(nvv.size)}
        print("n·v（可視な点×フレーム %d 件）: min %.4f / 5%% %.4f / 中央 %.4f / max %.4f"
              % (nvv.size, nvv.min(), np.percentile(nvv, 5),
                 np.median(nvv), nvv.max()))

    # ---- 点集合を書き出す ----
    pts = []
    for rank, m in enumerate(sel_all):
        m = int(m)
        c = coloring[m]
        pts.append({
            "id": rank, "candidate_index": m, "object": on[m],
            "pass_index": int(pidx[m]), "band": c["band"],
            "color_index": c["color_index"], "rgb": c["rgb"],
            "mesh_co": [round(float(x), 6) for x in co[m]],
            "normal": [round(float(x), 6) for x in nr[m]],
            "is_target": bool(is_tgt[m]),
        })
    per_frame = {}
    for vi, var in enumerate(variants):
        rows = []
        for f in range(S.FRAMES):
            c = vi * S.FRAMES + f
            rows.append({
                "index": f,
                "u": [round(float(uv[int(m), c]), 3) for m in sel_all],
                "v": [round(float(uv2[int(m), c]), 3) for m in sel_all],
                "visible": [bool(vis[int(m), c]) for m in sel_all],
                "nv": [round(float(nv[int(m), c]), 4) for m in sel_all],
                "depth": [round(float(dep[int(m), c]), 5) for m in sel_all],
            })
        per_frame[var] = rows

    with open(os.path.join(OUT, "armc_points.json"), "w") as f:
        json.dump({"points": pts, "frames": per_frame,
                   "params": {"pitch": PITCH, "radius": RADIUS,
                              "assign_radius": ASSIGN_RADIUS,
                              "n_target": NT, "n_background": NB,
                              "d_floor": D_FLOOR,
                              "d_floor_no_overlap": D_FLOOR_NO_OVERLAP,
                              "width": W, "height": H,
                              "achieved_separation":
                                  RESULTS["info"]["achieved_separation"]}},
                  f, indent=1)

    npass = sum(1 for c in RESULTS["checks"] if c["ok"])
    nfail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("=" * 68)
    print("SELF-CHECK : PASS %d / FAIL %d" % (npass, nfail))
    print("=" * 68)
    with open(os.path.join(OUT, "select_arm_c.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("saved : %s" % os.path.join(OUT, "armc_points.json"))
    print("saved : %s" % os.path.join(OUT, "select_arm_c.json"))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
