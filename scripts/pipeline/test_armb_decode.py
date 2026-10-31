"""
test_armb_decode.py  —  Arm B デコーダの単体テスト（Blender 不要）

閉形式で出した照合値と突合する（作法：閉形式との突合を必ず入れる）。
さらに壊したデータで検出が働くことを確認する（作法#11）。

実行（ローカル。Blender 不要）
    python test_armb_decode.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import armb_encode
import armb_decode as D

# scene_c_spec.NORM_BOX と同じ値。spec が import できるならそちらを使う
try:
    import scene_c_spec as S
    NORM_BOX = S.NORM_BOX
    SRC = "scene_c_spec.NORM_BOX"
except Exception:
    NORM_BOX = {"x": (-4.50, 4.50), "y": (-3.00, 2.05), "z": (-1.60, 2.60)}
    SRC = "テスト内リテラル（scene_c_spec が無い）"

npass = fail = 0


def ck(name, ok, detail=""):
    global npass, fail
    if ok:
        npass += 1
    else:
        fail += 1
        print("FAIL %-32s %s" % (name, detail))
    return ok


print("NORM_BOX の出所: %s" % SRC)
print("NORM_BOX = %s" % NORM_BOX)
print()

# ---- 1. 感度の閉形式 -----------------------------------------------------
sens = D.sensitivity(NORM_BOX)
EXP_SENS = np.array([9.00, 5.05, 4.20]) / 102.0
ck("span_is_102", abs(D.SPAN - 102.0) < 1e-12, str(D.SPAN))
ck("sensitivity", np.allclose(sens, EXP_SENS, atol=1e-12),
   "%s vs %s" % (sens, EXP_SENS))
print("| 軸 | ボックス幅 | units/LSB | 0.5 LSB |")
print("|---|---|---|---|")
for i, ax in enumerate("XYZ"):
    lo, hi = NORM_BOX[ax.lower()]
    print("| %s | %.2f | %.5f | %.5f |" % (ax, hi - lo, sens[i], 0.5 * sens[i]))
print()

# ---- 2. 帯の端の復号誤差はちょうど 0.25 LSB ------------------------------
print("| base | 8bit | q_hat | 真の q | 誤差(LSB) |")
print("|---|---|---|---|---|")
for base, v, tq in [(D.BASE_BG, 13, 0.0), (D.BASE_BG, 115, 1.0),
                    (D.BASE_TG, 140, 0.0), (D.BASE_TG, 242, 1.0)]:
    q = (v / 255.0 - base) / 0.40
    lsb = abs(q - tq) * D.SPAN
    print("| %.2f | %d | %+.8f | %.1f | %.4f |" % (base, v, q, tq, lsb))
    ck("edge_%.2f_%d" % (base, v), abs(lsb - 0.25) < 1e-9, "%.6f LSB" % lsb)
print()

# ---- 3. encode -> decode の往復が 0.5 LSB 以内 ---------------------------
rng = np.random.default_rng(0)
H, W = 64, 96
lo, hi = np.array([NORM_BOX[k][0] for k in "xyz"]), \
         np.array([NORM_BOX[k][1] for k in "xyz"])
pos = lo + rng.random((H, W, 3)) * (hi - lo)
ids = np.where(rng.random((H, W)) < 0.3, 3.0, 1.0)

rgb, excess = armb_encode.encode(pos, ids, NORM_BOX)
ck("encode_no_excess", excess == 0.0, "%.3e" % excess)

pos_hat, band, info = D.decode(rgb, NORM_BOX)
ck("no_reserved", info["n_reserved"] == 0, str(info["n_reserved"]))
ck("no_disagree", info["n_disagree"] == 0, str(info["n_disagree"]))
ck("band_recovered",
   np.array_equal(band, np.where(ids == 3.0, D.BAND_TG, D.BAND_BG)),
   "帯の復元が GT と一致しない")

d = np.abs(pos_hat - pos)
bound = 0.5 * sens
print("| 軸 | 実測最大誤差 | 上限 0.5LSB | 比 |")
print("|---|---|---|---|")
for i, ax in enumerate("XYZ"):
    mx = float(d[:, :, i].max())
    print("| %s | %.6f | %.6f | %.4f |" % (ax, mx, bound[i], mx / bound[i]))
    ck("roundtrip_%s" % ax, mx <= bound[i] + 1e-12,
       "%.8f > %.8f" % (mx, bound[i]))
print()

# ---- 4. 空画素 -----------------------------------------------------------
ids2 = ids.copy(); ids2[3, 4] = 0.0
rgb2, _ = armb_encode.encode(pos, ids2, NORM_BOX)
pos2, band2, info2 = D.decode(rgb2, NORM_BOX)
ck("empty_band", band2[3, 4] == D.BAND_EMPTY, str(band2[3, 4]))
ck("empty_is_nan", bool(np.isnan(pos2[3, 4]).all()), str(pos2[3, 4]))
ck("empty_count", info2["n_empty"] == 1, str(info2["n_empty"]))

# ---- 5. errors() が空画素を除外する ---------------------------------------
band_gt = np.where(ids2 == 0.0, D.BAND_EMPTY,
                   np.where(ids2 == 3.0, D.BAND_TG, D.BAND_BG)).astype(np.int8)
st = D.errors(pos2, pos, band2, band_gt)
ck("errors_excludes_empty", st["n_px"] == H * W - 1, str(st["n_px"]))
ck("errors_no_band_confusion", st["band"]["n_confused"] == 0,
   str(st["band"]["n_confused"]))
print("| 軸 | MAE(units) | p99(units) | max(units) | max(px) |")
print("|---|---|---|---|---|")
for ax in "XYZ":
    a, p = st["axis"][ax], st["px"][ax]
    print("| %s | %.6f | %.6f | %.6f | %.3f |"
          % (ax, a["mae"], a["p99"], a["max"], p["max"]))
print()

# ---- 6. scale_box -------------------------------------------------------
b2 = D.scale_box(NORM_BOX, 2.0, mode="center")
ck("scale_center_extent",
   abs((b2["x"][1] - b2["x"][0]) - 18.0) < 1e-12, str(b2["x"]))
ck("scale_center_keeps_center",
   abs((b2["x"][0] + b2["x"][1]) / 2 - 0.0) < 1e-12, str(b2["x"]))
ck("scale_sensitivity_doubles",
   np.allclose(D.sensitivity(b2), 2.0 * sens, atol=1e-12),
   str(D.sensitivity(b2)))
b3 = D.scale_box(NORM_BOX, 2.0, mode="max")
ck("scale_max_keeps_lo", b3["x"][0] == -4.50, str(b3["x"]))

# ---- 7. 破壊テスト（作法#11） --------------------------------------------
print("--- 破壊テスト")

# 7-1 チャンネル入れ替え
pos_sw, _, _ = D.decode(rgb, NORM_BOX)
pos_sw = pos_sw[:, :, ::-1]
d_sw = np.abs(pos_sw - pos).max()
print("  R<->B 入れ替えで最大誤差 %.4f units（正常時 %.6f）"
      % (d_sw, float(d.max())))
ck("swap_detected", d_sw > 10 * float(d.max()), "%.6f" % d_sw)

# 7-2 ボックスを間違える（Y 上限 2.05 -> 2.00）
bad = dict(NORM_BOX); bad["y"] = (-3.00, 2.00)
pos_bb, _, _ = D.decode(rgb, bad)
d_bb = np.abs(pos_bb[:, :, 1] - pos[:, :, 1]).max()
print("  NORM_BOX の Y 上限を 2.05->2.00 にすると Y 誤差 %.5f units" % d_bb)
ck("bad_box_detected", d_bb > bound[1] * 2, "%.8f" % d_bb)

# 7-3 帯を固定で間違える（全部 background と決めつける）
force_bg = np.full((H, W), D.BAND_BG, dtype=np.int8)
pos_fb, _, _ = D.decode(rgb, NORM_BOX, band=force_bg)
sel = (ids == 3.0)
d_fb = np.abs(pos_fb[:, :, 0][sel] - pos[:, :, 0][sel]).max()
print("  Target を background と決めつけると X 誤差 %.4f units" % d_fb)
ck("wrong_band_detected", d_fb > 5.0, "%.6f" % d_fb)

# 7-4 予約帯を混ぜる
rgb_r = rgb.copy(); rgb_r[10, 10] = (127, 127, 127)
_, _, info_r = D.decode(rgb_r, NORM_BOX)
print("  予約帯 127 を1画素混ぜると n_reserved=%d" % info_r["n_reserved"])
ck("reserved_detected", info_r["n_reserved"] == 1, str(info_r["n_reserved"]))

# 7-5 チャンネル不一致を混ぜる（帯を跨がせる）
rgb_d = rgb.copy()
bg_yx = np.argwhere(ids == 1.0)[0]
rgb_d[bg_yx[0], bg_yx[1], 0] = 200          # background 画素の R だけ Target 帯へ
_, _, info_d = D.decode(rgb_d, NORM_BOX)
print("  1チャンネルだけ帯を跨がせると n_disagree=%d" % info_d["n_disagree"])
ck("disagree_detected", info_d["n_disagree"] == 1, str(info_d["n_disagree"]))

# 7-6 帯の混同を作る（Target 画素を background 帯の値に置換）
rgb_c = rgb.copy()
tg_yx = np.argwhere(ids == 3.0)[0]
rgb_c[tg_yx[0], tg_yx[1]] = (100, 100, 100)
_, band_c, _ = D.decode(rgb_c, NORM_BOX)
bgt = np.where(ids == 3.0, D.BAND_TG, D.BAND_BG).astype(np.int8)
st_c = D.errors(pos, pos, band_c, bgt)
print("  Target 1画素を background 帯に落とすと tg_as_bg=%d"
      % st_c["band"]["tg_as_bg"])
ck("band_confusion_detected", st_c["band"]["tg_as_bg"] == 1,
   str(st_c["band"]["tg_as_bg"]))

print()
print("PASS %d / FAIL %d" % (npass, fail))
sys.exit(1 if fail else 0)
