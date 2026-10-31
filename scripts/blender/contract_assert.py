"""
contract_assert.py -- protocol.md §14 入力契約の機械検査

3層に分ける
  S  シーン検査   .blend の設定。bpy が必要。エクスポート前に走らせる
  O  出力検査     出したファイル。stdlib + numpy のみ。エクスポート後に走らせる
  M  手動確認     実行時に検出不能な項目。PASS 扱いにせず MANUAL として列挙する

M を自動 PASS にしないのが要点。「全部通った」を嘘にしないため。

使い方
  # シーン検査
  blender --background min_scene.blend --python contract_assert.py -- --scene

  # 出力検査（Blender 内でも素の python でも動く）
  blender --background --python contract_assert.py -- --armc w2a3b --palette armc_palette.json
  python contract_assert.py --armc w2a3b --palette armc_palette.json

  # 両方
  blender --background min_scene.blend --python contract_assert.py -- \
      --scene --armc w2a3b --palette armc_palette.json

  # split をまたぐ bit 一致（§14-24）
  python contract_assert.py --armc R0S0/armc --compare-armc R1S0/armc R0S1/armc

終了コード
  0 = FAIL なし / 1 = FAIL あり
"""

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import re
import struct
import sys
import zlib

import numpy as np

try:
    import bpy
    HAVE_BPY = True
except ImportError:
    HAVE_BPY = False

# ---------------------------------------------------------------------------
# 期待値（すべて protocol.md の該当節を明記する）
# ---------------------------------------------------------------------------

SPEC = {
    "res": (832, 480),                     # §5
    "fps": 16,                             # §5
    "n_frames": 49,                        # §5  4n+1 必須
    "focal_mm": 50.0,                      # §5
    "sensor_mm": 36.0,                     # §5
    "latent": (13, 60, 104),               # §5
    "engine_prefix": "BLENDER_EEVEE",      # §22.3
    "taa_render_samples": 1,               # W2a-1 で発見。EEVEE も平均化する
    "filter_size_max": 0.02,               # 同上
    "reserved_band": (124, 131),           # §4
    "band_non_target": (30, 64, 98),       # §13.5
    "band_target": (157, 191, 225),        # §13.5
    "min_color_dist": 34.0 * math.sqrt(2), # §13  48.0833
    "dot_radius": 6,                       # §13
    "dot_area": 113,                       # §16.11
    "separation_radius": 24.0,             # §13
    "v_max_design": 12.0,                  # §13.3  実測下限17 x 安全率0.7
    "vae_color_err_r6_v17": 11.30,         # §16.11  主指標の実測値
    "codebook_non_target": (13, 115),      # §4
    "codebook_target": (140, 242),         # §4
}
# §14-12 の要求: 局所色間距離 > 2 x VAE色誤差
SPEC["min_color_dist_required"] = 2.0 * SPEC["vae_color_err_r6_v17"]

MANUAL_ITEMS = [
    ("M01", "§14-19",
     "Arm C の順投影が world_to_camera_view 経由である"
     "（手書き intrinsics を組んでいない）"),
    ("M02", "§14-18",
     "Arm B が Position パス由来である（Depth からの逆投影を経由していない）"),
    ("M03", "§14-3",
     "Target / Non-target 境界を nearest で処理している"),
    ("M04", "§14-5",
     "bilinear resize を経由していない"),
    ("M05", "§14-9",
     "Control の実入力テンソルを保存している"),
    ("M06", "§14 src_mask",
     "src_mask を Visibility / ObjectID / Alpha / Target Mask / "
     "Tracking の黒背景 / Control の輝度 から生成していない"),
    ("M07", "W2a-1",
     "マスクを pass_index で定義している（値の許容範囲では隣接面が漏れる）"),
    ("M08", "§22.3",
     "幾何パスに Cycles を使っていない（使う場合はサンプル数1）"),
    ("M09", "§18-17",
     "Arm C の遮蔽判定が scene.ray_cast である"
     "（Depth パス比較は1画素内の深度勾配で誤判定する）"),
]

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


class Results:
    def __init__(self):
        self.rows = []

    def add(self, cid, ref, status, title, detail=""):
        self.rows.append({"id": cid, "ref": ref, "status": status,
                          "title": title, "detail": str(detail)})

    def check(self, cid, ref, ok, title, detail=""):
        self.add(cid, ref, PASS if ok else FAIL, title, detail)
        return ok

    def counts(self):
        c = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
        for r in self.rows:
            c[r["status"]] += 1
        return c


# ---------------------------------------------------------------------------
# PNG 自前実装（Blender のカラーマネジメントを経由しない）
# ---------------------------------------------------------------------------

def read_png(path):
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
                raise ValueError("8bit RGB 非インターレース以外: %s"
                                 % ((bd, ct, inter),))
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
            raise ValueError("filter type %d（0 以外は非対応）" % raw[off])
        out[y] = np.frombuffer(raw[off + 1:off + 1 + stride],
                               np.uint8).reshape(w, 3)
    return out


def band_palette(levels):
    return [(levels[i], levels[j], levels[k])
            for i, j, k in itertools.product(range(3), repeat=3)
            if (i + j + k) % 2 == 0 and not levels[i] == levels[j] == levels[k]]


def disc_offsets(r):
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dx * dx + dy * dy <= r * r]


def label_components(mask):
    """4近傍の連結成分。(ラベル配列, 成分数) を返す。numpy のみで実装。"""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    n = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if lab[sy, sx]:
            continue
        n += 1
        stack = [(int(sy), int(sx))]
        lab[sy, sx] = n
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                        and not lab[ny, nx]:
                    lab[ny, nx] = n
                    stack.append((ny, nx))
    return lab, n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# S  シーン検査（bpy 必須）
# ===========================================================================

def iter_fcurves(obj):
    """5.2 では Action.fcurves が消え、layers -> strips -> channelbag に移動。"""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    legacy = getattr(act, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    out, slot = [], getattr(ad, "action_slot", None)
    for layer in act.layers:
        for strip in layer.strips:
            cb = None
            if slot is not None and hasattr(strip, "channelbag"):
                try:
                    cb = strip.channelbag(slot)
                except (TypeError, RuntimeError):
                    cb = None
            if cb is None and hasattr(strip, "channelbags"):
                bags = list(strip.channelbags)
                cb = bags[0] if bags else None
            if cb is not None:
                out.extend(cb.fcurves)
    return out


def signed_volume(obj):
    """閉じたメッシュの符号付き体積。負なら法線が内向き（§17 バグ5）。"""
    me = obj.data
    if not hasattr(me, "polygons") or len(me.polygons) == 0:
        return None
    # 全エッジが2面に共有されていれば閉じているとみなす
    edge_count = {}
    for p in me.polygons:
        vs = list(p.vertices)
        for i in range(len(vs)):
            e = tuple(sorted((vs[i], vs[(i + 1) % len(vs)])))
            edge_count[e] = edge_count.get(e, 0) + 1
    if not edge_count or any(v != 2 for v in edge_count.values()):
        return None                        # 開いたメッシュは判定しない
    vol = 0.0
    mw = obj.matrix_world
    for p in me.polygons:
        vs = [mw @ me.vertices[i].co for i in p.vertices]
        for i in range(1, len(vs) - 1):
            a, b, c = vs[0], vs[i], vs[i + 1]
            vol += a.dot(b.cross(c)) / 6.0
    return vol


def project_ref(world_xyz, cam_loc_y):
    x, y, z = world_xyz
    d = y - cam_loc_y
    if d <= 1e-6:
        return None
    s = SPEC["res"][0] / (d * SPEC["sensor_mm"] / SPEC["focal_mm"])
    return (SPEC["res"][0] / 2.0 + x * s, SPEC["res"][1] / 2.0 - z * s)


def check_scene(R):
    scene = bpy.context.scene
    vl = bpy.context.view_layer
    rd = scene.render

    R.check("S01", "§5", (rd.resolution_x, rd.resolution_y) == SPEC["res"]
            and rd.resolution_percentage == 100,
            "解像度 832x480 / 100%%",
            "%dx%d @ %d%%" % (rd.resolution_x, rd.resolution_y,
                              rd.resolution_percentage))

    R.check("S02", "§5", rd.fps == SPEC["fps"], "fps 16", rd.fps)

    n = scene.frame_end - scene.frame_start + 1
    R.check("S03", "§5",
            scene.frame_start == 1 and n == SPEC["n_frames"]
            and (n - 1) % 4 == 0,
            "frame 1..49 / 枚数49 / 4n+1",
            "start %d end %d -> %d 枚"
            % (scene.frame_start, scene.frame_end, n))

    R.check("S04", "§22.3", rd.engine.startswith(SPEC["engine_prefix"]),
            "レンダラが EEVEE", rd.engine)

    taa = getattr(getattr(scene, "eevee", None), "taa_render_samples", None)
    if taa is None:
        R.add("S05", "W2a-1", SKIP, "taa_render_samples == 1", "eevee 設定なし")
    else:
        R.check("S05", "W2a-1", taa == SPEC["taa_render_samples"],
                "taa_render_samples == 1（EEVEE もサンプル平均する）", taa)

    R.check("S06", "W2a-1", rd.filter_size <= SPEC["filter_size_max"],
            "filter_size <= 0.02（画素間のぼかし）", "%.6f" % rd.filter_size)

    mb = getattr(rd, "use_motion_blur", None)
    R.check("S07", "§22.3", mb is False or mb is None,
            "モーションブラー無効", mb)

    cam = scene.camera
    if cam is None:
        R.add("S08", "§22.3", FAIL, "被写界深度 無効", "scene.camera が None")
        R.add("S10", "§5", FAIL, "焦点距離 50mm / センサー 36mm", "-")
        R.add("S11", "§17 バグ1", FAIL, "camera matrix_world が純回転", "-")
    else:
        R.check("S08", "§22.3", not cam.data.dof.use_dof,
                "被写界深度 無効", cam.data.dof.use_dof)
        R.check("S10", "§5",
                abs(cam.data.lens - SPEC["focal_mm"]) < 1e-6
                and abs(cam.data.sensor_width - SPEC["sensor_mm"]) < 1e-6
                and cam.data.sensor_fit == "HORIZONTAL",
                "焦点距離 50mm / センサー 36mm / HORIZONTAL",
                "%.3fmm / %.3fmm / %s" % (cam.data.lens,
                                          cam.data.sensor_width,
                                          cam.data.sensor_fit))
        mw = cam.matrix_world
        cols = [[mw[r][c] for r in range(3)] for c in range(3)]
        norms = [math.sqrt(sum(v * v for v in col)) for col in cols]
        dots = [sum(a * b for a, b in zip(cols[i], cols[j]))
                for i, j in ((0, 1), (0, 2), (1, 2))]
        det = mw.to_3x3().determinant()
        R.check("S11", "§17 バグ1,6",
                all(abs(nv - 1.0) < 1e-6 for nv in norms)
                and all(abs(d) < 1e-6 for d in dots)
                and abs(det - 1.0) < 1e-6,
                "camera matrix_world が純回転（列ノルム1 / 直交 / det +1）",
                "norms %s / dots %s / det %.9f"
                % ([round(v, 9) for v in norms],
                   [round(d, 9) for d in dots], det))

    cyc = getattr(scene, "cycles", None)
    if cyc is None or not rd.engine.startswith("CYCLES"):
        R.add("S09", "§22.3", SKIP, "デノイズ無効", "Cycles を使っていない")
    else:
        R.check("S09", "§22.3", not cyc.use_denoising
                and not cyc.use_adaptive_sampling
                and not cyc.use_animated_seed,
                "デノイズ / Adaptive / Animated Seed 無効",
                "denoise %s / adaptive %s / anim_seed %s"
                % (cyc.use_denoising, cyc.use_adaptive_sampling,
                   cyc.use_animated_seed))

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    bad_scale = [o.name for o in meshes
                 if any(abs(s - 1.0) > 1e-6 for s in o.scale)]
    R.check("S12", "§17 バグ1", not bad_scale,
            "全メッシュの scale == 1.0", bad_scale or "%d 個すべて 1.0"
            % len(meshes))

    R.check("S13", "§22.2", vl.use_pass_position,
            "Position パス有効", vl.use_pass_position)
    R.check("S14", "§22.4", vl.use_pass_z, "Depth パス有効", vl.use_pass_z)
    mist = getattr(vl, "use_pass_mist", False)
    R.check("S15", "§22.1", not mist,
            "Mist パス無効（GT には使えない相対値）", mist)

    neg_det = [o.name for o in meshes
               if o.matrix_world.to_3x3().determinant() <= 0]
    inward = []
    for o in meshes:
        v = signed_volume(o)
        if v is not None and v <= 0:
            inward.append("%s(vol %.4f)" % (o.name, v))
    R.check("S16", "§17 バグ5,6", not neg_det and not inward,
            "鏡像スケールなし / 閉メッシュの法線が外向き",
            {"neg_det": neg_det, "inward_normals": inward})

    anim = [o for o in bpy.data.objects if o.animation_data
            and o.animation_data.action]
    bad_interp = []
    for o in anim:
        for fc in iter_fcurves(o):
            for kp in fc.keyframe_points:
                if kp.interpolation != "LINEAR":
                    bad_interp.append("%s:%s" % (o.name, kp.interpolation))
                    break
    if not anim:
        R.add("S17", "W2a-0", SKIP, "キーフレーム補間が LINEAR",
              "アニメーションなし")
    else:
        R.check("S17", "W2a-0", not bad_interp,
                "キーフレーム補間が LINEAR（BEZIER は等速にならない）",
                bad_interp or "%d オブジェクト OK" % len(anim))

    # スクリーン空間速度。各アニメーションオブジェクトの bbox 8隅で測る
    cam_y = cam.matrix_world.translation.y if cam else None
    worst = None
    if anim and cam_y is not None:
        prev = {}
        for i in range(SPEC["n_frames"]):
            scene.frame_set(i + 1)
            for o in anim:
                mw = o.matrix_world
                for ci, corner in enumerate(o.bound_box):
                    w = mw @ type(mw.translation)(corner)
                    uv = project_ref((w.x, w.y, w.z), cam_y)
                    if uv is None:
                        continue
                    key = (o.name, ci)
                    if key in prev:
                        d = math.hypot(uv[0] - prev[key][0],
                                       uv[1] - prev[key][1])
                        if worst is None or d > worst[0]:
                            worst = (d, o.name, i)
                    prev[key] = uv
        scene.frame_set(1)
    if worst is None:
        R.add("S18", "§10 §13.3", SKIP,
              "スクリーン速度 <= 12 px/frame", "測定対象なし")
    else:
        R.check("S18", "§10 §13.3", worst[0] <= SPEC["v_max_design"],
                "スクリーン速度 <= 12 px/frame（V_max 17 x 安全率0.7）",
                "最大 %.3f px/frame (%s / index %d)"
                % (worst[0], worst[1], worst[2]))

    vs = scene.view_settings
    R.add("S19", "§7", WARN if vs.view_transform != "Standard" else PASS,
          "Color Management が固定されている",
          "view_transform %s / look %s / exposure %.3f / gamma %.3f"
          % (vs.view_transform, vs.look, vs.exposure, vs.gamma))


# ===========================================================================
# O  出力検査（stdlib + numpy のみ）
# ===========================================================================

FRAME_RE = re.compile(r"^(?P<stem>.*?)(?P<idx>\d{3,4})\.(?P<ext>[A-Za-z0-9]+)$")


def collect_frames(d):
    """ディレクトリ内の連番ファイルを (index, path) で返す。"""
    items = []
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        m = FRAME_RE.match(os.path.basename(p))
        if m:
            items.append((int(m.group("idx")), p, m.group("ext").lower()))
    items.sort()
    return items


def check_output_armc(R, d, palette_json):
    if not os.path.isdir(d):
        R.add("O01", "§14-23", FAIL, "Arm C の出力枚数 49",
              "ディレクトリが無い: %s" % d)
        return

    items = collect_frames(d)
    idxs = [i for i, _, _ in items]
    R.check("O01", "§14-23", len(items) == SPEC["n_frames"],
            "出力枚数 == 49", len(items))
    R.check("O02", "§14-10 / §17 バグ3",
            idxs == list(range(SPEC["n_frames"])),
            "index が 0..48 の連番で欠落なし",
            "先頭 %s / 末尾 %s / 重複 %d"
            % (idxs[:3], idxs[-3:], len(idxs) - len(set(idxs))))

    exts = sorted({e for _, _, e in items})
    R.check("O03", "§14-11", exts == ["png"],
            "PNG のみ（mp4 / 圧縮コーデックを経由していない）", exts)

    tgt = band_palette(SPEC["band_target"])
    nt = band_palette(SPEC["band_non_target"])
    pal_set = {tuple(c) for c in tgt + nt}

    assign = None
    if palette_json and os.path.exists(palette_json):
        with open(palette_json, encoding="utf-8") as f:
            pj = json.load(f)
        assign = pj.get("assignment")
        pj_pal = {tuple(c) for band in pj.get("palette", {}).values()
                  for c in band}
        R.check("O16", "§13.5", pj_pal == pal_set,
                "割当ファイルのパレットが §13.5 の生成結果と一致",
                "json %d色 / 期待 %d色" % (len(pj_pal), len(pal_set)))
    else:
        R.add("O16", "§13.5", SKIP,
              "割当ファイルのパレット照合", "--palette 未指定")

    lo, hi = SPEC["reserved_band"]
    offsets = disc_offsets(SPEC["dot_radius"])
    illegal, reserved_hits, shape_bad, filter_bad = set(), 0, [], []
    color_sets = []
    comp_checked = comp_skipped = 0
    comp_bad = []

    for idx, path, _ in items:
        try:
            img = read_png(path)
        except ValueError as exc:
            filter_bad.append("%s: %s" % (os.path.basename(path), exc))
            continue
        if img.shape != (SPEC["res"][1], SPEC["res"][0], 3):
            shape_bad.append("%s %s" % (os.path.basename(path), img.shape))
            continue
        flat = np.unique(img.reshape(-1, 3), axis=0)
        uniq = {tuple(int(x) for x in c) for c in flat}
        illegal |= (uniq - pal_set - {(0, 0, 0)})
        reserved_hits += int(np.count_nonzero((img >= lo) & (img <= hi)))
        color_sets.append(uniq - {(0, 0, 0)})

        # 孤立した連結成分の面積。円が欠けていないことの確認。
        # 別の色に上書きされた点、画面端で切れた点は対象外にする
        # （どちらも面積が 113 未満になるのが正常）
        nonblack = np.any(img > 0, axis=-1)
        for c in uniq - {(0, 0, 0)}:
            cmask = np.all(img == np.array(c, np.uint8), axis=-1)
            lab, ncomp = label_components(cmask)
            for k in range(1, ncomp + 1):
                ys, xs = np.nonzero(lab == k)
                area = int(ys.size)
                touches_border = bool(ys.min() == 0 or xs.min() == 0
                                      or ys.max() == img.shape[0] - 1
                                      or xs.max() == img.shape[1] - 1)
                r = SPEC["dot_radius"] + 1
                y0, y1 = max(0, ys.min() - r), min(img.shape[0], ys.max() + r + 1)
                x0, x1 = max(0, xs.min() - r), min(img.shape[1], xs.max() + r + 1)
                region = img[y0:y1, x0:x1]
                other = np.any(np.any(region > 0, axis=-1)
                               & ~np.all(region == np.array(c, np.uint8),
                                         axis=-1))
                if touches_border or other:
                    comp_skipped += 1
                    continue
                comp_checked += 1
                if area != SPEC["dot_area"]:
                    comp_bad.append({"index": idx, "color": list(c),
                                     "area": area,
                                     "at": [int(xs.min()), int(ys.min())]})

    R.check("O04", "§14-1", not shape_bad, "解像度 832x480 / 3ch",
            shape_bad or "全 %d 枚 OK" % len(items))
    R.check("O05", "§14-7 / W2a-3b", not filter_bad,
            "8bit RGB / filter 0 / 非インターレース",
            filter_bad or "全 %d 枚 OK" % len(items))
    R.check("O06", "§14-4", not illegal,
            "パレット外の色が無い（AA が入っていない）",
            [list(c) for c in sorted(illegal)] or "なし")
    R.check("O07", "§14-6 / §4", reserved_hits == 0,
            "予約帯 124-131 の画素が無い", reserved_hits)

    R.check("O08", "§13 / §16.11", not comp_bad,
            "孤立した点の面積が 113（円が欠けていない）",
            comp_bad[:8] if comp_bad
            else "検査 %d 成分 / 除外 %d 成分（画面端・他色と隣接）"
                 % (comp_checked, comp_skipped))

    # §14-12 局所色間距離 > 2 x VAE色誤差
    intra = min(math.dist(a, b) for band in (tgt, nt)
                for a, b in itertools.combinations(band, 2))
    R.check("O09", "§14-12", intra > SPEC["min_color_dist_required"],
            "帯内 最小色間距離 > 2 x VAE色誤差",
            "%.4f > %.4f（余裕 %.2f 倍）"
            % (intra, SPEC["min_color_dist_required"],
               intra / SPEC["min_color_dist_required"]))

    # 同じ色が全フレームで同じ点を指しているか（色集合の単調性は見られないが、
    # 出現した色が必ずパレット内で一貫していることは確認できる）
    all_seen = set().union(*color_sets) if color_sets else set()
    R.add("O10", "§13", WARN,
          "同一色 == 同一点 は出力だけでは検証不能",
          "出現色 %d 種。割当ファイルとの突合が必要" % len(all_seen))

    if assign:
        assigned_cols = {tuple(a["color"]) for a in assign
                         if a.get("n_frames_visible", 0) > 0}
        R.check("O11", "§13", all_seen <= assigned_cols,
                "出力に現れる色が割当ファイルの可視点の色に含まれる",
                "出力 %d 種 / 割当（可視）%d 種 / 余剰 %s"
                % (len(all_seen), len(assigned_cols),
                   [list(c) for c in sorted(all_seen - assigned_cols)]))
    else:
        R.add("O11", "§13", SKIP, "出力色と割当の突合", "--palette 未指定")


def check_src_mask(R, path):
    if not path:
        R.add("O12", "§14 src_mask", SKIP,
              "src_mask が全面 255", "--src-mask 未指定")
        return
    items = collect_frames(path) if os.path.isdir(path) else [(0, path, "png")]
    bad = []
    for idx, p, _ in items:
        img = read_png(p)
        if int(img.min()) != 255 or int(img.max()) != 255:
            bad.append("%s min %d max %d"
                       % (os.path.basename(p), img.min(), img.max()))
    R.check("O12", "§14 src_mask", not bad,
            "src_mask が全面 255（min == max == 255）",
            bad or "%d 枚すべて 255" % len(items))


def check_float_finite(R, paths):
    if not paths:
        R.add("O13", "§14-8", SKIP, "NaN / Inf が無い",
              "--npy 未指定（float テンソルの検査）")
        return
    bad = []
    for p in paths:
        a = np.load(p)
        if not np.all(np.isfinite(a)):
            bad.append("%s: NaN %d / Inf %d"
                       % (os.path.basename(p),
                          int(np.isnan(a).sum()), int(np.isinf(a).sum())))
    R.check("O13", "§14-8", not bad, "NaN / Inf が無い",
            bad or "%d ファイル OK" % len(paths))


def check_cross_split(R, base, others):
    if not others:
        R.add("O14", "§14-24 / §7", SKIP,
              "Control Video が全 split で bit 一致", "--compare-armc 未指定")
        return
    b = collect_frames(base)
    diffs = []
    for o in others:
        c = collect_frames(o)
        if len(c) != len(b):
            diffs.append("%s: 枚数 %d != %d" % (o, len(c), len(b)))
            continue
        for (i1, p1, _), (i2, p2, _) in zip(b, c):
            if sha256(p1) != sha256(p2):
                diffs.append("%s index %d" % (o, i1))
    R.check("O14", "§14-24 / §7", not diffs,
            "Control Video が全 split で bit 一致", diffs[:10] or "一致")


# ===========================================================================
# 出力
# ===========================================================================

def emit(R, out_json):
    c = R.counts()
    width = 78
    print("")
    print("=" * width)
    print("protocol.md §14 入力契約  機械検査")
    print("=" * width)
    cur = None
    for r in R.rows:
        grp = r["id"][0]
        if grp != cur:
            cur = grp
            label = {"S": "S  シーン検査", "O": "O  出力検査"}.get(grp, grp)
            print("-" * width)
            print(label)
            print("-" * width)
        print("  %-4s %-6s %-4s %s" % (r["id"], r["status"], "", r["title"]))
        if r["detail"] and (r["status"] != PASS or len(r["detail"]) < 90):
            print("            %s  [%s]" % (r["detail"][:200], r["ref"]))
    print("-" * width)
    print("M  手動確認（実行時に検出不能。個別に署名すること）")
    print("-" * width)
    for cid, ref, text in MANUAL_ITEMS:
        print("  %-4s MANUAL  [ ] %s" % (cid, text))
        print("            [%s]" % ref)
    print("=" * width)
    print("PASS %d / FAIL %d / WARN %d / SKIP %d / MANUAL %d"
          % (c[PASS], c[FAIL], c[WARN], c[SKIP], len(MANUAL_ITEMS)))
    if c[FAIL]:
        print("")
        print("FAIL:")
        for r in R.rows:
            if r["status"] == FAIL:
                print("  %s %s -> %s" % (r["id"], r["title"], r["detail"][:160]))
    print("=" * width)

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"spec": {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in SPEC.items()},
                       "rows": R.rows,
                       "manual": [{"id": a, "ref": b, "text": c2}
                                  for a, b, c2 in MANUAL_ITEMS],
                       "counts": c}, f, indent=2, ensure_ascii=False)
        print("report : %s" % out_json)
    return 1 if c[FAIL] else 0


def parse_args():
    argv = sys.argv[1:]
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--scene", action="store_true")
    ap.add_argument("--armc", default=None)
    ap.add_argument("--palette", default=None)
    ap.add_argument("--src-mask", default=None)
    ap.add_argument("--npy", nargs="*", default=None)
    ap.add_argument("--compare-armc", nargs="*", default=None)
    ap.add_argument("--report", default="contract_report.json")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    R = Results()

    if a.scene:
        if not HAVE_BPY:
            R.add("S00", "-", FAIL, "シーン検査",
                  "bpy が無い。blender --background <file> --python で実行する")
        else:
            try:
                check_scene(R)
            except Exception as exc:                  # noqa: BLE001
                import traceback
                R.add("S00", "-", FAIL,
                      "シーン検査が例外で中断した（API 不一致の可能性）",
                      "%s: %s" % (type(exc).__name__, exc))
                traceback.print_exc()
    if a.armc:
        for fn, args, cid, title in (
                (check_output_armc, (a.armc, a.palette), "O00", "Arm C 出力検査"),
                (check_src_mask, (a.src_mask,), "O00b", "src_mask 検査"),
                (check_float_finite, (a.npy,), "O00c", "NaN/Inf 検査"),
                (check_cross_split, (a.armc, a.compare_armc), "O00d",
                 "split 間 bit 一致")):
            try:
                fn(R, *args)
            except Exception as exc:                  # noqa: BLE001
                import traceback
                R.add(cid, "-", FAIL, "%s が例外で中断した" % title,
                      "%s: %s" % (type(exc).__name__, exc))
                traceback.print_exc()
    if not a.scene and not a.armc:
        R.add("X00", "-", SKIP, "検査対象が指定されていない",
              "--scene または --armc DIR を渡す")

    code = emit(R, a.report)
    if not HAVE_BPY:
        sys.exit(code)


main()
