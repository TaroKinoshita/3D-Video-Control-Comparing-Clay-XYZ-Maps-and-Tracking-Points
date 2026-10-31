#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_gt_center.py — 評価用の Target 中心 GT を ID パスから作り、JSON に凍結する。（W9, 2026-10-31）

なぜ必要か
  score_t_control.py（v1）は Scene C′ の GT 中心を
      u = Scene C の ids==3 重心
      v = Scene C の ids==3 重心 − (K/9.00)·dZ(f)
  で作っていた。だが C′ では Target が持ち上がって遮蔽板の上に出るため、
  **見えているシルエットそのものが Scene C と違う**（index 6–29 は u 全幅が可視）。
  Scene C で半分隠れている形の重心を上にずらしても、C′ で見えている形の重心にはならない。
  実測で v1 GT と C′ 本物の GT の差は median 2.38 / 最大 31.88 px（index 22–27）。
  完璧な生成（tv_R0S0_prime そのもの）を v1 GT で採点すると p95 32.98 px になる。

  → GT は C′ の ID パス（geo/prime/id_%04d.exr、Target = pass_index 3）から直接取る。

定義（v1 と同じ「マスク重心」基準。tracker も同じ座標系）
  u = mean(列 index), v = mean(行 index)   ※ +0.5 しない（tracker と揃える）
  Target 画素 0 のフレームは GT なし（完全遮蔽。評価面から除外）

出力（凍結リテラル。採点側はこれだけを読む = 経路分離）
  {"meta": {...}, "frames": [{"index": i, "npx": n, "u": .., "v": .., "u_sc": .., "v_sc": ..}, ...]}
  u_sc / v_sc は Scene C（original）の ids==3 重心。暗記判別（Scene C 軌道との距離）に使う。

依存
  OpenEXR（pip の `OpenEXR` 3.x）+ numpy。Blender 不要。
  EXR は top-down で読める（Marker 重心 v≈123 で向きを確認する）。

使い方
  python make_gt_center.py --geo data/render/geo --out data/points/gt_target_center_prime.json
"""
import argparse
import hashlib
import json
import os

import numpy as np

TARGET_ID = 3.0
MARKER_ID = 6.0
N_FRAMES = 49
# 照合値：制約4（遮蔽窓）と §12 の評価面
EXPECT_OCCLUDED = list(range(36, 41))
EXPECT_VISIBLE = 44
# Marker 中心の投影（scene_c_expect の凍結値）。向きの検査に使う
MARKER_UV = (658.7638, 123.4734)


def load_id(path):
    import OpenEXR
    ch = OpenEXR.File(path).channels()
    key = "RGBA" if "RGBA" in ch else sorted(ch)[0]
    a = ch[key].pixels
    if a.ndim == 3:
        a = a[..., 0]
    return np.asarray(a, dtype=np.float32)


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True, help="geo ディレクトリ（original/ と prime/ を含む）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for idx in range(N_FRAMES):
        frame = idx + 1  # EXR は Blender frame 番号（1..49）
        ip = load_id(os.path.join(a.geo, "prime", "id_%04d.exr" % frame))
        io = load_id(os.path.join(a.geo, "original", "id_%04d.exr" % frame))
        assert ip.shape == (480, 832), ip.shape
        if idx == 0:
            mu, mv = centroid(io == MARKER_ID)
            assert abs(mu - MARKER_UV[0]) < 2 and abs(mv - MARKER_UV[1]) < 2, \
                "EXR の向きが想定と違う（Marker 重心 %.1f, %.1f）" % (mu, mv)
            assert np.array_equal(ip, io), "index 0 は prime と original で一致するはず（制約3）"
        mp = ip == TARGET_ID
        mo = io == TARGET_ID
        r = {"index": idx, "npx": int(mp.sum())}
        if r["npx"]:
            r["u"], r["v"] = (round(x, 4) for x in centroid(mp))
        if mo.sum():
            r["u_sc"], r["v_sc"] = (round(x, 4) for x in centroid(mo))
        rows.append(r)

    occ = [r["index"] for r in rows if r["npx"] == 0]
    vis = [r["index"] for r in rows if r["npx"] > 0]
    assert occ == EXPECT_OCCLUDED, occ
    assert len(vis) == EXPECT_VISIBLE, len(vis)
    for r in rows:
        if r["index"] >= 35:  # dZ=0 区間は Scene C と同一
            assert r.get("u") == r.get("u_sc") and r.get("v") == r.get("v_sc"), r

    out = {
        "meta": {
            "made_by": "scripts/pipeline/make_gt_center.py",
            "source": "data/render/geo/{prime,original}/id_%04d.exr (Blender frame = index+1)",
            "target_pass_index": 3,
            "definition": "mask centroid in pixel index coords (no +0.5), same as tracker",
            "u_v": "Scene C' (prime) Target centroid = evaluation GT",
            "u_sc_v_sc": "Scene C (original) Target centroid = memorization reference",
            "occluded_index": occ,
            "visible_frames": len(vis),
        },
        "frames": rows,
    }
    b = (json.dumps(out, ensure_ascii=True, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(a.out, "wb") as f:
        f.write(b)
    print("wrote", a.out, len(b), "bytes  sha256(first16) =", hashlib.sha256(b).hexdigest()[:16])
    print("visible frames:", len(vis), " occluded:", occ)


if __name__ == "__main__":
    main()
