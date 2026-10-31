"""
draw_dots_check.py -- W2a-3b Arm C の点描画とパレット割当の検証

やること
  1. 局所最大分離パレットを生成（protocol §13.5）
  2. 全フレームの和で衝突グラフを作り、24px 以内に同色が来ないよう割当
  3. ray_cast で遮蔽判定し（§18-17 で決定）、深度順に 6px 円を描く
  4. PNG を自前で書き、自前で読み戻して byte 一致を確認
  5. AA なし / 予約帯なし / 円の面積 113px / 再出現時の同色維持 を検証

PNG を Blender 経由で扱わない理由
  Blender の画像保存はカラーマネジメントを通るため 8bit の値がそのまま出る
  保証がない。さらに Blender で書いて Blender で読むと同じ変換が往復で
  打ち消され、間違っていても検出できない。zlib だけで自前実装する。

使い方
  blender --background min_scene.blend --python draw_dots_check.py -- --out .

出力
  w2a3b/armc_000.png ... armc_048.png
  w2a3b_report.json
  armc_palette.json         点と色の対応（後段で再利用する）
"""

import itertools
import json
import math
import os
import struct
import sys
import zlib

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

# ---------------------------------------------------------------------------
# 1. 定数（protocol §13）
# ---------------------------------------------------------------------------

RES_X = 832
RES_Y = 480
FOCAL_MM = 50.0
SENSOR_MM = 36.0
CAM_Y = -10.0
N_FRAMES = 49

DOT_RADIUS = 6                     # §13 確定値
DOT_AREA_EXPECT = 113              # 離散円の画素数
SEPARATION_RADIUS = 24.0           # 分離割当半径
MIN_COLOR_DIST = 34.0 * math.sqrt(2.0)   # 48.0833

BAND_NON_TARGET = (30, 64, 98)     # 13-115 の内側へ17寄せ
BAND_TARGET = (157, 191, 225)      # 140-242 の内側へ17寄せ
RESERVED = (124, 131)              # この帯の値は出てはいけない

TARGET_HALF = 0.5
OCCLUDER_HALF_X = 0.35
OCCLUDER_HALF_Z = 1.5

EPS_RAY = 1e-3
SUBDIR = "w2a3b"
OUT_DIR = "."


def px_per_unit(depth):
    return RES_X / (depth * SENSOR_MM / FOCAL_MM)


def depth_of(world_y):
    return world_y - CAM_Y


# ---------------------------------------------------------------------------
# 2. PNG 自前実装（8bit RGB / filter None / インターレースなし）
# ---------------------------------------------------------------------------

def _chunk(typ, data):
    body = typ + data
    return (struct.pack(">I", len(data)) + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))


def write_png(path, arr):
    """arr は uint8 の (H, W, 3)。値をそのまま書く。"""
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("uint8 の (H, W, 3) が必要")
    h, w = arr.shape[:2]
    raw = bytearray()
    for y in range(h):
        raw.append(0)                       # filter type 0 = None
        raw += arr[y].tobytes()
    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
           + _chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def read_png(path):
    """write_png が書いたものを読み戻す。filter が 0 以外なら例外。"""
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PNG シグネチャが違う")
    pos, idat, w, h = 8, bytearray(), None, None
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, bd, ct, comp, filt, inter = struct.unpack(">IIBBBBB", body)
            if (bd, ct, inter) != (8, 2, 0):
                raise ValueError("想定外の PNG 形式 %s" % ((bd, ct, inter),))
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
        pos += 12 + ln
    raw = zlib.decompress(bytes(idat))
    stride = w * 3
    out = np.empty((h, w, 3), np.uint8)
    for y in range(h):
        off = y * (stride + 1)
        if raw[off] != 0:
            raise ValueError("filter type %d（0 のみ対応）" % raw[off])
        out[y] = np.frombuffer(raw[off + 1:off + 1 + stride],
                               np.uint8).reshape(w, 3)
    return out


# ---------------------------------------------------------------------------
# 3. パレット
# ---------------------------------------------------------------------------

def band_palette(levels):
    """市松配置（i+j+k 偶数）14色から、全チャンネル同値の2色を除いた12色。"""
    cols = []
    for i, j, k in itertools.product(range(3), repeat=3):
        if (i + j + k) % 2:
            continue
        c = (levels[i], levels[j], levels[k])
        if c[0] == c[1] == c[2]:
            continue
        cols.append(c)
    return cols


def disc_offsets(r):
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dx * dx + dy * dy <= r * r]


# ---------------------------------------------------------------------------
# 4. 追跡点
# ---------------------------------------------------------------------------

def tracked_points(target, occluder):
    pts = []
    h = TARGET_HALF
    for i, (sx, sy, sz) in enumerate(
            [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]):
        pts.append({"name": "T_vert%d" % i, "entity": "target",
                    "obj": target, "local": (sx * h, sy * h, sz * h)})
    for name, local in (("T_face_-X", (-h, 0, 0)), ("T_face_+X", (h, 0, 0)),
                        ("T_face_front", (0, -h, 0)),
                        ("T_face_back", (0, h, 0)),
                        ("T_face_bottom", (0, 0, -h)),
                        ("T_face_top", (0, 0, h))):
        pts.append({"name": name, "entity": "target",
                    "obj": target, "local": local})
    # 遮蔽板（Non-target）。深度7で常に最前面
    # 4隅は画面外に出る（半高1.5 x 165.08px/unit = 247.6px > 240）。
    # 画面外の扱いを検証する材料として残す。
    for i, (sx, sz) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        pts.append({"name": "O_corner%d" % i, "entity": "non_target",
                    "obj": occluder,
                    "local": (sx * OCCLUDER_HALF_X, 0.0,
                              sz * OCCLUDER_HALF_Z)})
    # 画面内の点。Non-target 帯の描画経路を通すため
    # O_near は O_center から 16.5px しか離れないので衝突が強制され、
    # Non-target 帯で2色目が使われることを検証できる
    for name, local in (("O_center", (0.0, 0.0, 0.0)),
                        ("O_near", (0.0, 0.0, 0.1)),
                        ("O_tl", (-0.2, 0.0, 1.0)),
                        ("O_tr", (0.2, 0.0, 1.0)),
                        ("O_bl", (-0.2, 0.0, -1.0)),
                        ("O_br", (0.2, 0.0, -1.0))):
        pts.append({"name": name, "entity": "non_target",
                    "obj": occluder, "local": local})
    return pts


def project_bpy(scene, cam, world_xyz):
    ndc = world_to_camera_view(scene, cam, Vector(world_xyz))
    return (ndc.x * RES_X, (1.0 - ndc.y) * RES_Y)


def visible_raycast(scene, depsgraph, cam_loc, world_pt):
    vec = world_pt - cam_loc
    dist = vec.length
    if dist <= EPS_RAY:
        return False
    hit = scene.ray_cast(depsgraph, cam_loc, vec.normalized(),
                         distance=dist - EPS_RAY)[0]
    return not hit


# ---------------------------------------------------------------------------
# 5. 割当（全フレームの和で衝突グラフを作る）
# ---------------------------------------------------------------------------

def build_tracks(scene, cam, depsgraph, pts):
    """各フレームの (u, v, depth, visible) を全点について作る。"""
    cam_loc = cam.matrix_world.translation.copy()
    tracks = [[] for _ in pts]
    for index in range(N_FRAMES):
        scene.frame_set(index + 1)
        depsgraph.update()
        for pi, pt in enumerate(pts):
            w = pt["obj"].matrix_world @ Vector(pt["local"])
            u, v = project_bpy(scene, cam, w)
            vis = visible_raycast(scene, depsgraph, cam_loc, w)
            inside = (0.0 <= u < RES_X) and (0.0 <= v < RES_Y)
            tracks[pi].append({"u": u, "v": v, "depth": depth_of(w.y),
                               "visible": bool(vis and inside),
                               "in_frame": bool(inside),
                               "world": [w.x, w.y, w.z]})
    return tracks


def conflict_graph(tracks, pts):
    """一度でも 24px 以内に同時可視で来るペアを衝突とみなす。"""
    n = len(pts)
    conf = [set() for _ in range(n)]
    closest = {}
    for a, b in itertools.combinations(range(n), 2):
        if pts[a]["entity"] != pts[b]["entity"]:
            continue                      # 帯が違えば色は必ず離れている
        best = 1e9
        for i in range(N_FRAMES):
            ta, tb = tracks[a][i], tracks[b][i]
            if not (ta["visible"] and tb["visible"]):
                continue
            d = math.hypot(ta["u"] - tb["u"], ta["v"] - tb["v"])
            best = min(best, d)
        closest[(a, b)] = best
        if best < SEPARATION_RADIUS:
            conf[a].add(b)
            conf[b].add(a)
    return conf, closest


def assign_colors(pts, conf):
    """次数の大きい順に貪欲彩色。帯ごとに独立。"""
    palettes = {"target": band_palette(BAND_TARGET),
                "non_target": band_palette(BAND_NON_TARGET)}
    order = sorted(range(len(pts)), key=lambda i: -len(conf[i]))
    assigned = [None] * len(pts)
    for i in order:
        used = {assigned[j] for j in conf[i] if assigned[j] is not None}
        for c in palettes[pts[i]["entity"]]:
            if c not in used:
                assigned[i] = c
                break
        if assigned[i] is None:
            raise RuntimeError(
                "点 %s に割り当てられる色が無い（衝突 %d 件）"
                % (pts[i]["name"], len(conf[i])))
    return assigned, palettes


# ---------------------------------------------------------------------------
# 6. 描画
# ---------------------------------------------------------------------------

def render_frame(tracks, colors, index, offsets):
    """深度の遠い順に描く。近い点が上書きする（Depth sorting）。"""
    img = np.zeros((RES_Y, RES_X, 3), np.uint8)
    drawn = [(pi, tracks[pi][index]) for pi in range(len(tracks))
             if tracks[pi][index]["visible"]]
    drawn.sort(key=lambda e: -e[1]["depth"])
    for pi, t in drawn:
        c0 = int(math.floor(t["u"]))
        r0 = int(math.floor(t["v"]))
        col = colors[pi]
        for dx, dy in offsets:
            x, y = c0 + dx, r0 + dy
            if 0 <= x < RES_X and 0 <= y < RES_Y:
                img[y, x] = col
    return img, drawn


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def unit_test_depth_sort(offsets):
    """深度ソートの単体テスト。実シーンでは発動しないので合成データで検証する。

    このシーンでは遮蔽板の点と Target の点が画面上で重なるフレームが存在しない
    （遮蔽板がある場所では背後の Target が隠れるため）。
    近い点が遠い点を上書きするコードパスが一度も通らないので、
    正解が自明な人工データで確認する。
    """
    FAR = (30, 30, 98)
    NEAR = (225, 225, 157)
    cases = []

    # ケース1 完全に同じ画素。近い点（depth 5）が勝つべき
    tracks = [
        [{"u": 400.5, "v": 200.5, "depth": 9.0, "visible": True}],
        [{"u": 400.5, "v": 200.5, "depth": 5.0, "visible": True}],
    ]
    img, drawn = render_frame(tracks, [FAR, NEAR], 0, offsets)
    got = tuple(int(x) for x in img[200, 400])
    cases.append({"case": "同一画素", "expect": list(NEAR),
                  "got": list(got), "ok": got == NEAR,
                  "draw_order": [i for i, _ in drawn]})

    # ケース2 入力の順序を逆にしても結果は変わらないべき
    img2, _ = render_frame(
        [tracks[1], tracks[0]], [NEAR, FAR], 0, offsets)
    got2 = tuple(int(x) for x in img2[200, 400])
    cases.append({"case": "入力順を反転", "expect": list(NEAR),
                  "got": list(got2), "ok": got2 == NEAR})

    # ケース3 3px ずれ。重なった領域は近い点、遠い点の外周は残るべき
    tracks3 = [
        [{"u": 400.5, "v": 200.5, "depth": 9.0, "visible": True}],
        [{"u": 403.5, "v": 200.5, "depth": 5.0, "visible": True}],
    ]
    img3, _ = render_frame(tracks3, [FAR, NEAR], 0, offsets)
    n_far = int(np.count_nonzero(
        np.all(img3 == np.array(FAR, np.uint8), axis=-1)))
    n_near = int(np.count_nonzero(
        np.all(img3 == np.array(NEAR, np.uint8), axis=-1)))
    cases.append({"case": "3px ずれ", "near_area": n_near,
                  "far_area_remaining": n_far,
                  "expect_near_area": DOT_AREA_EXPECT,
                  "ok": n_near == DOT_AREA_EXPECT and 0 < n_far < DOT_AREA_EXPECT})

    # ケース4 不可視の点は描かれないべき
    tracks4 = [[{"u": 100.5, "v": 100.5, "depth": 9.0, "visible": False}]]
    img4, drawn4 = render_frame(tracks4, [FAR], 0, offsets)
    cases.append({"case": "不可視は描かない", "n_drawn": len(drawn4),
                  "n_nonzero": int(np.count_nonzero(img4)),
                  "ok": len(drawn4) == 0 and not img4.any()})

    return cases


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
    for name in ("Target", "Occluder", "Cam"):
        if name not in bpy.data.objects:
            raise RuntimeError("%s が無い。min_scene.blend を開くこと。" % name)
    target = bpy.data.objects["Target"]
    occluder = bpy.data.objects["Occluder"]
    cam = bpy.data.objects["Cam"]

    offsets = disc_offsets(DOT_RADIUS)
    unit_cases = unit_test_depth_sort(offsets)
    pts = tracked_points(target, occluder)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    tracks = build_tracks(scene, cam, depsgraph, pts)
    conf, closest = conflict_graph(tracks, pts)
    colors, palettes = assign_colors(pts, conf)

    report = {
        "blender": bpy.app.version_string,
        "dot_radius": DOT_RADIUS,
        "disc_area": len(offsets),
        "disc_area_expected": DOT_AREA_EXPECT,
        "separation_radius": SEPARATION_RADIUS,
        "min_color_dist_design": MIN_COLOR_DIST,
        "palette_sizes": {k: len(v) for k, v in palettes.items()},
        "n_points": len(pts),
        "occlusion_method": "scene.ray_cast (eps=%g)" % EPS_RAY,
    }

    # --- パレット自体の検査 ---
    pal_all = palettes["target"] + palettes["non_target"]
    intra = min(math.dist(a, b) for band in palettes.values()
                for a, b in itertools.combinations(band, 2))
    cross = min(math.dist(a, b) for a in palettes["target"]
                for b in palettes["non_target"])
    used_vals = sorted({v for c in pal_all for v in c})
    report["palette_check"] = {
        "intra_band_min_dist": intra,
        "cross_band_min_dist": cross,
        "values_used": used_vals,
        "reserved_violations": [v for v in used_vals
                                if RESERVED[0] <= v <= RESERVED[1]],
    }

    # --- 割当結果 ---
    report["assignment"] = [
        {"point": pts[i]["name"], "entity": pts[i]["entity"],
         "color": list(colors[i]), "n_conflicts": len(conf[i]),
         "n_frames_visible": sum(1 for t in tracks[i] if t["visible"])}
        for i in range(len(pts))]

    # --- 描画と検証 ---
    per_frame = []
    roundtrip_ok = True
    illegal_colors = set()
    reserved_hits = 0
    center_ok = 0
    center_shadowed = []
    isolated_area = []
    local_color_min = 1e9
    local_pair_min = None

    pal_set = {tuple(c) for c in pal_all} | {(0, 0, 0)}
    tgt_set = {tuple(c) for c in palettes["target"]}
    nt_set = {tuple(c) for c in palettes["non_target"]}
    band_seen = {"target": set(), "non_target": set()}
    frames_with_nt = 0

    for index in range(N_FRAMES):
        img, drawn = render_frame(tracks, colors, index, offsets)
        path = os.path.join(out_dir, "armc_%03d.png" % index)
        write_png(path, img)

        back = read_png(path)
        if not np.array_equal(img, back):
            roundtrip_ok = False

        flat = back.reshape(-1, 3)
        uniq = {tuple(int(x) for x in c) for c in np.unique(flat, axis=0)}
        illegal_colors |= (uniq - pal_set)
        band_seen["target"] |= (uniq & tgt_set)
        band_seen["non_target"] |= (uniq & nt_set)
        if uniq & nt_set:
            frames_with_nt += 1
        reserved_hits += int(np.count_nonzero(
            (back >= RESERVED[0]) & (back <= RESERVED[1])))

        # 中心画素が自分の色か（近い点に上書きされた場合は除外）
        for pi, t in drawn:
            c0, r0 = int(math.floor(t["u"])), int(math.floor(t["v"]))
            got = tuple(int(x) for x in back[r0, c0])
            if got == tuple(colors[pi]):
                center_ok += 1
            else:
                owner = [pts[qi]["name"] for qi, q in drawn
                         if tuple(colors[qi]) == got
                         and math.hypot(q["u"] - t["u"],
                                        q["v"] - t["v"]) <= DOT_RADIUS
                         and q["depth"] < t["depth"]]
                center_shadowed.append(
                    {"index": index, "point": pts[pi]["name"],
                     "got": list(got), "own": list(colors[pi]),
                     "nearer_owner": owner})

        # 孤立点の面積（他の可視点が 2r 以内に無い点）
        for pi, t in drawn:
            iso = all(math.hypot(q["u"] - t["u"], q["v"] - t["v"])
                      > 2 * DOT_RADIUS + 1
                      for qi, q in drawn if qi != pi)
            if not iso:
                continue
            c0, r0 = int(math.floor(t["u"])), int(math.floor(t["v"]))
            x0, x1 = max(0, c0 - DOT_RADIUS - 1), min(RES_X, c0 + DOT_RADIUS + 2)
            y0, y1 = max(0, r0 - DOT_RADIUS - 1), min(RES_Y, r0 + DOT_RADIUS + 2)
            if (x1 - x0) < 2 * DOT_RADIUS + 1 or (y1 - y0) < 2 * DOT_RADIUS + 1:
                continue                      # 画面端で切れている点は除外
            box = back[y0:y1, x0:x1]
            area = int(np.count_nonzero(
                np.all(box == np.array(colors[pi], np.uint8), axis=-1)))
            isolated_area.append({"index": index, "point": pts[pi]["name"],
                                  "area": area})

        # 局所色間距離（24px 以内の可視ペア）
        for (pi, ti), (qi, tq) in itertools.combinations(drawn, 2):
            d = math.hypot(ti["u"] - tq["u"], ti["v"] - tq["v"])
            if d > SEPARATION_RADIUS:
                continue
            cd = math.dist(colors[pi], colors[qi])
            if cd < local_color_min:
                local_color_min = cd
                local_pair_min = {"index": index,
                                  "a": pts[pi]["name"], "b": pts[qi]["name"],
                                  "screen_dist": d, "color_dist": cd}

        occ = int(np.count_nonzero(np.any(back > 0, axis=-1)))
        per_frame.append({"index": index, "n_drawn": len(drawn),
                          "occupied_px": occ,
                          "occupancy": occ / (RES_X * RES_Y)})

    report["per_frame"] = per_frame
    report["unit_test_depth_sort"] = unit_cases
    report["band_coverage"] = {
        "target_colors_seen": sorted(list(c) for c in band_seen["target"]),
        "non_target_colors_seen": sorted(
            list(c) for c in band_seen["non_target"]),
        "n_target_colors": len(band_seen["target"]),
        "n_non_target_colors": len(band_seen["non_target"]),
        "frames_with_non_target": frames_with_nt,
        "both_bands_present": bool(band_seen["target"]
                                   and band_seen["non_target"]),
    }
    report["png_roundtrip_exact"] = roundtrip_ok
    report["illegal_colors"] = [list(c) for c in sorted(illegal_colors)]
    report["reserved_band_hits"] = reserved_hits
    report["center_pixel_own_color"] = center_ok
    report["center_pixel_shadowed"] = center_shadowed
    areas = [a["area"] for a in isolated_area]
    report["isolated_area"] = {
        "n": len(areas),
        "min": min(areas) if areas else None,
        "max": max(areas) if areas else None,
        "all_equal_expected": bool(areas) and all(
            a == DOT_AREA_EXPECT for a in areas),
    }
    report["local_color_dist_min"] = (
        None if local_pair_min is None else local_pair_min)

    # --- 再出現時の同色維持 ---
    fi = [i for i, p in enumerate(pts) if p["name"] == "T_face_front"][0]
    vis_idx = [i for i in range(N_FRAMES) if tracks[fi][i]["visible"]]
    gap = [i for i in range(N_FRAMES) if not tracks[fi][i]["visible"]]
    samples = {}
    for idx in (vis_idx[0], (gap[0] - 1 if gap else None),
                (gap[-1] + 1 if gap else None), vis_idx[-1]):
        if idx is None or idx < 0 or idx >= N_FRAMES:
            continue
        img = read_png(os.path.join(out_dir, "armc_%03d.png" % idx))
        t = tracks[fi][idx]
        c0, r0 = int(math.floor(t["u"])), int(math.floor(t["v"]))
        samples[idx] = [int(x) for x in img[r0, c0]]
    report["reappearance"] = {
        "point": "T_face_front",
        "hidden_index_range": [gap[0], gap[-1]] if gap else [],
        "center_color_samples": samples,
        "all_same": len({tuple(v) for v in samples.values()}) == 1,
        "assigned": list(colors[fi]),
    }

    with open(os.path.join(base, "w2a3b_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(base, "armc_palette.json"), "w",
              encoding="utf-8") as f:
        json.dump({"assignment": report["assignment"],
                   "palette": {k: [list(c) for c in v]
                               for k, v in palettes.items()},
                   "dot_radius": DOT_RADIUS,
                   "separation_radius": SEPARATION_RADIUS}, f,
                  indent=2, ensure_ascii=False)

    # --- 表示 ---
    pc = report["palette_check"]
    print("")
    print("=" * 72)
    print("パレット  target %d色 / non_target %d色"
          % (len(palettes["target"]), len(palettes["non_target"])))
    print("   帯内 最小色間距離   : %.4f （設計 %.4f）"
          % (pc["intra_band_min_dist"], MIN_COLOR_DIST))
    print("   帯をまたぐ最小距離  : %.4f" % pc["cross_band_min_dist"])
    print("   使用値              : %s" % pc["values_used"])
    print("   予約帯 124-131 違反 : %s" % pc["reserved_violations"])
    print("-" * 72)
    print("円  半径 %d -> %d px （期待 %d）"
          % (DOT_RADIUS, report["disc_area"], DOT_AREA_EXPECT))
    print("遮蔽 %s" % report["occlusion_method"])
    print("-" * 72)
    print("割当結果")
    print("  point           entity      color             衝突  可視frame")
    for a in report["assignment"]:
        print("  %-15s %-11s %-17s %3d   %3d"
              % (a["point"], a["entity"], tuple(a["color"]),
                 a["n_conflicts"], a["n_frames_visible"]))
    print("-" * 72)
    print("検証")
    print("   PNG 往復が byte 一致    : %s" % report["png_roundtrip_exact"])
    print("   パレット外の色          : %s"
          % (report["illegal_colors"] or "なし"))
    print("   予約帯 124-131 の画素   : %d" % report["reserved_band_hits"])
    print("   中心画素が自分の色      : %d / %d"
          % (center_ok, center_ok + len(center_shadowed)))
    print("   上書きされた中心        : %d" % len(center_shadowed))
    ia = report["isolated_area"]
    print("   孤立点の面積 (%d件)     : min %s / max %s / 全部113 = %s"
          % (ia["n"], ia["min"], ia["max"], ia["all_equal_expected"]))
    lc = report["local_color_dist_min"]
    if lc:
        print("   24px 以内の最小色距離   : %.4f  (%s <-> %s, %.1fpx, idx %d)"
              % (lc["color_dist"], lc["a"], lc["b"], lc["screen_dist"],
                 lc["index"]))
    else:
        print("   24px 以内の可視ペア     : なし")
    print("-" * 72)
    print("深度ソートの単体テスト（合成データ。実シーンでは発動しない経路）")
    for u in unit_cases:
        extra = ""
        if "got" in u:
            extra = "期待 %s / 実測 %s" % (tuple(u["expect"]), tuple(u["got"]))
        elif "near_area" in u:
            extra = "近い点 %d px / 遠い点の残り %d px" % (u["near_area"],
                                                        u["far_area_remaining"])
        elif "n_drawn" in u:
            extra = "描画 %d 点 / 非ゼロ %d px" % (u["n_drawn"], u["n_nonzero"])
        print("   %-16s %-5s %s" % (u["case"], "OK" if u["ok"] else "NG", extra))
    print("-" * 72)
    bc = report["band_coverage"]
    print("帯のカバレッジ")
    print("   Target 帯     : %d色 %s"
          % (bc["n_target_colors"],
             [tuple(c) for c in bc["target_colors_seen"]]))
    print("   Non-target 帯 : %d色 %s"
          % (bc["n_non_target_colors"],
             [tuple(c) for c in bc["non_target_colors_seen"]]))
    print("   Non-target が出るフレーム : %d / %d"
          % (bc["frames_with_non_target"], N_FRAMES))
    print("   両帯が出現 : %s" % bc["both_bands_present"])
    print("-" * 72)
    rp = report["reappearance"]
    print("再出現  %s  隠れる index %s" % (rp["point"], rp["hidden_index_range"]))
    print("   割当色 %s / 標本 %s"
          % (tuple(rp["assigned"]),
             {k: tuple(v) for k, v in rp["center_color_samples"].items()}))
    print("   全部同一 : %s" % rp["all_same"])
    print("-" * 72)
    occs = [f["occupancy"] for f in per_frame]
    nd = [f["n_drawn"] for f in per_frame]
    print("描画点数 min %d / max %d   占有率 %.4f%% .. %.4f%%"
          % (min(nd), max(nd), min(occs) * 100, max(occs) * 100))
    print("PNG : %s\\armc_000.png .. armc_048.png" % out_dir)
    print("report : %s" % os.path.join(base, "w2a3b_report.json"))
    print("=" * 72)


main()
