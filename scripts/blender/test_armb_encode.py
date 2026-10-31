"""
test_armb_encode.py  —  Arm B エンコーダの単体テスト（Blender 不要）

geo_expect の照合点の world 座標から合成データを作り、
armb_expect のリテラルと一致することを確認する。
さらに壊したデータで検出が働くことも確認する（作法#11）。
"""
import sys
import numpy as np
sys.path.insert(0, ".")
import scene_c_spec as S
import geo_expect as G
import armb_expect as A
import armb_encode
import png_io

H, W = A.HEIGHT, A.WIDTH
npass = fail = 0


def ck(name, ok, detail=""):
    global npass, fail
    if ok:
        npass += 1
    else:
        fail += 1
        print("FAIL %-34s %s" % (name, detail))
    return ok


# ---- geo_expect（Blender バッファ座標）から合成入力を作る --------------
pos = np.zeros((H, W, 3), dtype=np.float32)
ids = np.zeros((H, W), dtype=np.float32)
# 全面を Wall(1) の座標で埋める（空画素を作らない）
pos[:, :] = (G.PROBE_PX[0][5], G.PROBE_PX[0][6], G.PROBE_PX[0][7])
ids[:, :] = 1.0
mapping = []
for lbl, i, j_b, d, pi, X, Y, Z, inv in G.PROBE_PX:
    j_png = H - 1 - j_b
    pos[j_png, i] = (X, Y, Z)
    ids[j_png, i] = float(pi)
    mapping.append((lbl, i, j_png, pi))

# armb_expect の j_png が geo_expect から導けることを確認（座標系の取り違え検出）
for (lbl, i, jp, pi) in mapping:
    hit = [e for e in A.PROBE_RGB if e[0] == lbl]
    ck("coord_%s" % lbl, hit and hit[0][1] == i and hit[0][2] == jp,
       "armb_expect=(%s) derived=(%d,%d)" % (hit[0][1:3] if hit else None, i, jp))

rgb, excess = armb_encode.encode(pos, ids, S.NORM_BOX)
ck("dtype_uint8", rgb.dtype == np.uint8, str(rgb.dtype))
ck("shape", rgb.shape == (H, W, 3), str(rgb.shape))
ck("q_excess", excess <= A.MAX_Q_EXCESS, "%.3e" % excess)

print("| 点 | pass_index | 期待RGB | 実測RGB |")
for lbl, i, jp, pi, R, Gc, B, inv in A.PROBE_RGB:
    got = tuple(int(x) for x in rgb[jp, i])
    ok = max(abs(got[0] - R), abs(got[1] - Gc), abs(got[2] - B)) <= A.TOL_LEVEL
    print("| %-9s | %d | (%3d,%3d,%3d) | (%3d,%3d,%3d) |%s"
          % (lbl, pi, R, Gc, B, got[0], got[1], got[2], "" if ok else "  <-- FAIL"))
    ck("rgb_%s" % lbl, ok)
    band = A.BAND_TARGET if pi == 3 else A.BAND_BACKGROUND
    ck("band_%s" % lbl, all(band[0] <= c <= band[1] for c in got),
       "%s not in %s" % (got, band))

vals = np.unique(rgb)
ck("no_reserved_band",
   not any(A.RESERVED[0] <= int(v) <= A.RESERVED[1] for v in vals),
   "hits %s" % [int(v) for v in vals if A.RESERVED[0] <= int(v) <= A.RESERVED[1]])

# ---- 空画素 --------------------------------------------------------------
pos2 = pos.copy(); ids2 = ids.copy()
ids2[5, 5] = 0.0
pos2[5, 5] = (1.234, -0.5, 0.75)          # 座標があっても空なら (0,0,0)
rgb2, _ = armb_encode.encode(pos2, ids2, S.NORM_BOX)
ck("empty_is_black", tuple(int(x) for x in rgb2[5, 5]) == A.EMPTY_RGB,
   str(tuple(int(x) for x in rgb2[5, 5])))
# world 座標が (0,0,0) でも空画素扱いにしない
pos3 = pos.copy(); ids3 = ids.copy()
pos3[6, 6] = (0.0, 0.0, 0.0); ids3[6, 6] = 1.0
rgb3, _ = armb_encode.encode(pos3, ids3, S.NORM_BOX)
ck("origin_is_not_empty", tuple(int(x) for x in rgb3[6, 6]) != A.EMPTY_RGB,
   "world 原点が空画素として潰れている")

# ---- 帯の端 --------------------------------------------------------------
lo = np.array([S.NORM_BOX[k][0] for k in "xyz"], dtype=np.float32)
hi = np.array([S.NORM_BOX[k][1] for k in "xyz"], dtype=np.float32)
pe = np.zeros((2, 2, 3), dtype=np.float32); ie = np.ones((2, 2), dtype=np.float32)
pe[0, 0] = lo; pe[0, 1] = hi; pe[1, 0] = lo; pe[1, 1] = hi
ie[1, :] = 3.0
re, _ = armb_encode.encode(pe, ie, S.NORM_BOX)
ck("band_bg_lo", tuple(int(x) for x in re[0, 0]) == (13, 13, 13), str(re[0, 0]))
ck("band_bg_hi", tuple(int(x) for x in re[0, 1]) == (115, 115, 115), str(re[0, 1]))
ck("band_tg_lo", tuple(int(x) for x in re[1, 0]) == (140, 140, 140), str(re[1, 0]))
ck("band_tg_hi", tuple(int(x) for x in re[1, 1]) == (242, 242, 242), str(re[1, 1]))

# ---- PNG 往復 ------------------------------------------------------------
png_io.write_rgb("/tmp/_armb.png", rgb, W, H)
px, w2, h2 = png_io.read_rgb("/tmp/_armb.png")
back = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)
ck("png_roundtrip", np.array_equal(back, rgb))
ck("png_size", (w2, h2) == (W, H), "%dx%d" % (w2, h2))

# ---- 破壊テスト（作法#11） ------------------------------------------------
print("--- 破壊テスト")
tests = []
# 1) floor 丸め
q = (pos.astype(np.float64) - lo) / (hi - lo)
v_floor = np.floor((0.05 + 0.40 * np.clip(q, 0, 1)) * 255.0).astype(np.uint8)
tests.append(("floor 丸め", int(v_floor[100, 100][0]) != int(rgb[100, 100][0])
              or True))
d = [(lbl, tuple(int(x) for x in v_floor[jp, i]), (R, Gc, B))
     for lbl, i, jp, pi, R, Gc, B, inv in A.PROBE_RGB if pi != 3]
diff = [x for x in d if x[1] != x[2]]
print("  floor 丸めにすると %d / %d 点がリテラルと食い違う" % (len(diff), len(d)))
ck("floor_is_detected", len(diff) > 0, "floor でも同じ値になってしまう")

# 2) チャンネル入れ替え（R=Z / B=X）
rgb_sw, _ = armb_encode.encode(pos[:, :, ::-1].copy(), ids, S.NORM_BOX)
sw_diff = [lbl for lbl, i, jp, pi, R, Gc, B, inv in A.PROBE_RGB
           if tuple(int(x) for x in rgb_sw[jp, i]) != (R, Gc, B)]
print("  RGB<->ZYX 入れ替えで %d / %d 点が食い違う" % (len(sw_diff), len(A.PROBE_RGB)))
ck("channel_swap_detected", len(sw_diff) >= 5, str(sw_diff))

# 3) 上下反転
rgb_fl, _ = armb_encode.encode(pos[::-1].copy(), ids[::-1].copy(), S.NORM_BOX)
fl_diff = [lbl for lbl, i, jp, pi, R, Gc, B, inv in A.PROBE_RGB
           if tuple(int(x) for x in rgb_fl[jp, i]) != (R, Gc, B)]
print("  上下反転で %d / %d 点が食い違う" % (len(fl_diff), len(A.PROBE_RGB)))
ck("vflip_detected", len(fl_diff) >= 5, str(fl_diff))

# 4) Target 帯を使わない（全部 background）
rgb_nb, _ = armb_encode.encode(pos, np.ones_like(ids), S.NORM_BOX)
tg = [e for e in A.PROBE_RGB if e[3] == 3][0]
ck("target_band_detected",
   tuple(int(x) for x in rgb_nb[tg[2], tg[1]]) != (tg[4], tg[5], tg[6]),
   "Target 帯を使わなくても同じ値になる")

# 5) 正規化ボックスをずらす
bad_box = {"x": (-4.50, 4.50), "y": (-3.00, 2.00), "z": (-1.60, 2.60)}
rgb_bb, _ = armb_encode.encode(pos, ids, bad_box)
bb_diff = [lbl for lbl, i, jp, pi, R, Gc, B, inv in A.PROBE_RGB
           if tuple(int(x) for x in rgb_bb[jp, i]) != (R, Gc, B)]
print("  NORM_BOX の Y 上限を 2.05->2.00 にすると %d 点が食い違う" % len(bb_diff))
ck("norm_box_detected", len(bb_diff) > 0, "ボックスを変えても値が同じ")

print()
print("PASS %d / FAIL %d" % (npass, fail))
sys.exit(1 if fail else 0)
