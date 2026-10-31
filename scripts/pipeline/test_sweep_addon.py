"""
test_sweep_addon.py — armb_decode に移植した sweep / quant_worst / format_sweep の
                      追加テスト。既存の test_armb_decode.py は変更しない。
"""
import sys
import numpy as np
sys.path.insert(0, ".")
import armb_encode
import armb_decode as D

npass = fail = 0
def ck(name, ok, detail=""):
    global npass, fail
    if ok: npass += 1
    else:
        fail += 1
        print("FAIL %-40s %s" % (name, detail))
    return ok

NORM_BOX = {"x": (-4.50, 4.50), "y": (-3.00, 2.05), "z": (-1.60, 2.60)}

# quant_worst = 0.5 * sensitivity（閉形式）
qw = D.quant_worst(NORM_BOX)
for i, (a, e) in enumerate([("x", 9.00), ("y", 5.05), ("z", 4.20)]):
    ck("quant_worst_" + a, abs(qw[i] - 0.5 * e / 102.0) < 1e-12, "%.9f" % qw[i])

# k に線形か
for k in (2.0, 3.0, 4.0):
    got = D.quant_worst(D.scale_box(NORM_BOX, k))
    ck("quant_linear_k%.0f" % k, np.allclose(got, qw * k, atol=1e-12), str(got))

# 合成 GT（test_armb_decode と同じ seed で作る）
rng = np.random.default_rng(0)
H, W = 60, 104
lo = np.array([NORM_BOX[a][0] for a in "xyz"]); hi = np.array([NORM_BOX[a][1] for a in "xyz"])
pos = lo + rng.random((H, W, 3)) * (hi - lo)
ids = np.ones((H, W)); ids[10:20, 10:30] = 3.0

rows = D.sweep(pos, ids, NORM_BOX, [1.0, 2.0, 3.0])
ck("sweep_len", len(rows) == 3, str(len(rows)))
ck("sweep_k1_excess_zero", rows[0]["excess"] == 0.0, "%.3e" % rows[0]["excess"])
ck("sweep_k3_excess_zero", rows[2]["excess"] == 0.0, "%.3e" % rows[2]["excess"])
ck("sweep_no_reserved", all(r["info"]["n_reserved"] == 0 for r in rows),
   str([r["info"]["n_reserved"] for r in rows]))
ck("sweep_no_disagree", all(r["info"]["n_disagree"] == 0 for r in rows),
   str([r["info"]["n_disagree"] for r in rows]))
ck("sweep_no_band_confusion",
   all(r["err"]["band"]["n_confused"] == 0 for r in rows),
   str([r["err"]["band"]["n_confused"] for r in rows]))

mx = [r["err"]["axis"]["X"]["max"] for r in rows]
ck("sweep_units_increase", mx[0] < mx[1] < mx[2], str(mx))
# 量子化上限を超えないか（軸ごと）
for r in rows:
    for i, a in enumerate("XYZ"):
        b = r["quant_worst"][i] * (1 + 1e-9)
        ck("k%.0f_within_halflsb_%s" % (r["k"], a),
           r["err"]["axis"][a]["max"] <= b,
           "max=%.9f bound=%.9f" % (r["err"]["axis"][a]["max"], b))
# 実測 max はほぼ k 比例。厳密比例ではない：max は有限画素の丸め残差の最悪値で、
# どの画素が最悪になるかが k ごとに変わる（相対 1e-4 程度ずれる）。
# 厳密に k 比例するのは上限 quant_worst のほう（quant_linear_k* で検証）。
ck("sweep_ratio_k2", abs(mx[1] / mx[0] - 2.0) < 1e-3, "%.9f" % (mx[1] / mx[0]))
ck("sweep_ratio_k3", abs(mx[2] / mx[0] - 3.0) < 1e-3, "%.9f" % (mx[2] / mx[0]))
# 実測 max が上限に十分近いことは別に確認する（漸近の確認）
for r in rows:
    b = r["quant_worst"][0]
    ck("k%.0f_max_near_bound" % r["k"], r["err"]["axis"]["X"]["max"] > b * 0.999,
       "max=%.9f bound=%.9f" % (r["err"]["axis"]["X"]["max"], b))

# mode="max" は下限を保つ
bm = D.scale_box(NORM_BOX, 2.0, mode="max")
ck("mode_max_lo_fixed", bm["x"][0] == -4.50, str(bm["x"]))
ck("mode_max_extent", abs((bm["x"][1] - bm["x"][0]) - 18.0) < 1e-12, str(bm["x"]))

# vae callable が呼ばれるか（+1 LSB を注入）
called = {"n": 0}
def fake_vae(rgb):
    called["n"] += 1
    return np.clip(rgb.astype(np.int32) + 1, 0, 255).astype(np.uint8)
rows_v = D.sweep(pos, ids, NORM_BOX, [1.0], vae=fake_vae)
ck("vae_called", called["n"] == 1, str(called["n"]))
mae_x = rows_v[0]["err"]["axis"]["X"]["mae"]
ck("vae_plus1lsb_matches_sensitivity",
   abs(mae_x - D.sensitivity(NORM_BOX)[0]) < 2e-3,
   "mae=%.6f expect~%.6f" % (mae_x, D.sensitivity(NORM_BOX)[0]))
print("  fake VAE(+1 LSB) の X MAE = %.6f（閉形式 %.6f）"
      % (mae_x, D.sensitivity(NORM_BOX)[0]))

# 破壊：excess が立つ k（ボックスを縮める）
rows_s = D.sweep(pos, ids, NORM_BOX, [0.5])
ck("excess_fires_on_shrink", rows_s[0]["excess"] > 0.0, "%.6f" % rows_s[0]["excess"])
print("  k=0.5 に縮めると excess = %.6f" % rows_s[0]["excess"])

# format_sweep が表を返すか
t = D.format_sweep(rows)
ck("format_sweep_rows", t.count("\n") == 4, str(t.count("\n")))
ck("format_sweep_header", t.startswith("| k |"), t[:20])

print()
print("=== 量子化のみの sweep（GPU 不要。Pod 前に出せる下限） ===")
print(t)
print()
print("PASS %d / FAIL %d" % (npass, fail))
sys.exit(1 if fail else 0)
