"""
test_hist_minvariant.py  —  Blender 不要。bpy をスタブして合成データで検証する。

確認すること
  1. 正しい合成データで FAIL 0
  2. 共有 index の PNG を1枚壊すと assert 1 が発火する
  3. id 一致画素の depth をずらすと depth assert が発火する
  4. near / far の切り分けが設計どおり（AA フリンジは near、影は far）
  5. ヒストグラムの件数が閉形式と一致する
"""
import io
import os
import shutil
import sys
import types
import contextlib

import numpy as np

sys.modules["bpy"] = types.ModuleType("bpy")   # 実測しないのでスタブ可
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import png_io                                   # noqa: E402
import hist_minvariant as H                     # noqa: E402

W, HT = 16, 8
H.WIDTH, H.HEIGHT, H.NPIX = W, HT, W * HT

ROOT = "/tmp/hmtest"
EXR = {}          # path -> (h, w) float32


def ox(idx):
    return 1 + idx // 8            # 1..7


def ex(idx):
    return ox(idx) + (2 if 13 <= idx <= 35 else 0)


def build(depth_break=False):
    EXR.clear()
    for name in ("gO", "gE", "tO", "tE"):
        d = os.path.join(ROOT, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    for idx in range(49):
        frame = idx + 1
        for tag, xf in (("gO", ox), ("gE", ex)):
            tx = xf(idx)
            ida = np.full((HT, W), 2.0, dtype=np.float32)
            ida[:, tx:tx + 3] = 3.0                       # Target
            dep = np.full((HT, W), 10.0, dtype=np.float32)
            if depth_break and tag == "gE" and idx == 20:
                dep[0, 0] = 10.0 + 1e-3                   # id は一致するが depth が違う
            EXR[os.path.join(ROOT, tag, "id_%04d.exr" % frame)] = ida
            EXR[os.path.join(ROOT, tag, "depth_%04d.exr" % frame)] = dep

        for tag, xf in (("tO", ox), ("tE", ex)):
            tx = xf(idx)
            img = np.full((HT, W, 3), 100, dtype=np.uint8)
            img[:, tx:tx + 3] = 200                       # Target 本体
            if tx - 1 >= 0:
                img[:, tx - 1] = 150                      # AA フリンジ（near）
            if tx + 3 < W:
                img[:, tx + 3] = 150
            if tx + 6 < W:
                img[:, tx + 6] = 40                       # 影（far）
            png_io.write_rgb(os.path.join(ROOT, tag, "tv_%04d.png" % idx),
                             img, W, HT)


def fake_exr(path):
    return EXR[path]


def run():
    H.load_exr_ch0 = fake_exr
    argv = ["blender", "--",
            "--geo-orig", os.path.join(ROOT, "gO"),
            "--geo-edit", os.path.join(ROOT, "gE"),
            "--tv-orig", os.path.join(ROOT, "tO"),
            "--tv-edit", os.path.join(ROOT, "tE"),
            "--pngio", os.path.dirname(os.path.abspath(__file__)),
            "--out", os.path.join(ROOT, "r.json")]
    old = sys.argv
    sys.argv = argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            H.main()
    finally:
        sys.argv = old
    import json
    with open(os.path.join(ROOT, "r.json")) as f:
        rep = json.load(f)
    return buf.getvalue(), rep


npass = nfail = 0


def ck(cond, label):
    global npass, nfail
    print(("PASS  " if cond else "FAIL  ") + label)
    if cond:
        npass += 1
    else:
        nfail += 1


# ---- 1. 正しいデータ ---------------------------------------------------
build()
out, rep = run()
ck(rep["fails"] == [], "clean data -> no fails (%s)" % rep["fails"])
ck(rep["depth_violations"] == 0, "clean data -> depth_violations == 0")
ck(rep["empty_pixels"] == 0, "clean data -> empty_pixels == 0")
ck(all(rep["bit_identical"][str(i)] for i in H.SHARED_INDEX),
   "shared index bit-identical")
ck(not any(rep["bit_identical"][str(i)] for i in H.DIFF_INDEX),
   "diff index all differ")

hf = np.array(rep["hist_far"])
hn = np.array(rep["hist_near"])

# 閉形式：影の列は tx+6。orig/edit で位置が違うので delta=60 が 2 列ぶん出る
#   （orig の影列 40 vs edit の 100、edit の影列 100 vs 40）
#   ただし列が画面外に出る / near に入る場合は減る
exp_far = np.zeros(256, dtype=np.int64)
exp_near = np.zeros(256, dtype=np.int64)
for idx in H.DIFF_INDEX:
    to, te = ox(idx), ex(idx)
    near_cols = set()
    for t in (to, te):
        for c in range(t - 2, t + 5):
            if 0 <= c < W:
                near_cols.add(c)
    tgt_cols = set()
    for t in (to, te):
        tgt_cols |= {t, t + 1, t + 2}
    for c in range(W):
        if c in tgt_cols:
            continue                                  # Target で除外
        vo = 40 if c == to + 6 else (150 if c in (to - 1, to + 3) else 100)
        ve = 40 if c == te + 6 else (150 if c in (te - 1, te + 3) else 100)
        d = abs(vo - ve)
        if c in near_cols:
            exp_near[d] += HT
        else:
            exp_far[d] += HT

ck(np.array_equal(hf, exp_far),
   "far histogram matches closed form (sum %d vs %d)"
   % (int(hf.sum()), int(exp_far.sum())))
ck(np.array_equal(hn, exp_near),
   "near histogram matches closed form (sum %d vs %d)"
   % (int(hn.sum()), int(exp_near.sum())))
ck(int(hf[1:60].sum()) == 0, "far hist has an empty run 1..59 (got %d)"
   % int(hf[1:60].sum()))
ck(rep["valley_candidates"] and rep["valley_candidates"][0] == [1, 59],
   "valley candidate = [1, 59] (got %s)" % (rep["valley_candidates"][:1],))
ck(int(hn[50]) > 0, "AA fringe (delta 50) lands in near (%d)" % int(hn[50]))
ck(int(hf[50]) == 0, "AA fringe never lands in far (far[50]=%d)" % int(hf[50]))

# ---- 2. 共有 index を壊す ---------------------------------------------
build()
p = os.path.join(ROOT, "tE", "tv_0005.png")
px, w, h = png_io.read_rgb(p)
b = bytearray(px)
b[0] ^= 0xFF
png_io.write_rgb(p, bytes(b), w, h)
out, rep = run()
ck(any("bit-identical" in f for f in rep["fails"]),
   "broken shared frame fires assert 1 (%s)" % rep["fails"])
ck(rep["bit_identical"]["5"] is False, "index 5 flagged as differing")

# ---- 3. depth を壊す ---------------------------------------------------
build(depth_break=True)
out, rep = run()
ck(rep["depth_violations"] == 1,
   "depth mismatch under id agreement fires (got %d)" % rep["depth_violations"])
ck(any("depth agrees" in f for f in rep["fails"]), "depth assert in fails")

print("\nPASS %d / FAIL %d" % (npass, nfail))
sys.exit(1 if nfail else 0)
