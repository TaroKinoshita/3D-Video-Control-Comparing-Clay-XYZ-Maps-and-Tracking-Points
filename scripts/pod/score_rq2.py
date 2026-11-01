# -*- coding: utf-8 -*-
"""score_rq2.py — RQ2：編集の実現度と、不変領域への漏れ（GPU 不要）。（W11, 2026-11-01）

入力：同じ arm・同じ seed で
  gen_orig = Scene C Original の control から生成した動画
  gen_edit = Scene C Edited の control から生成した動画（Target 軌道だけ違う）

代表指標（rev.16 で事前登録）
  spill_mae = M_invariant 内の |gen_edit − gen_orig| の平均（8bit、RGB 3 チャンネル平均）
              集計フレームは index 13–35（GT 上で original と edited が違う区間。§32.1）
              全フレームの M_invariant 画素をまとめて 1 つの平均にする（pooled）

報告のみ
  spill_psnr      同じ画素集合での PSNR（dB）
  spill_p95       |差| の画素ごとの max_c の p95
  outside_mae     index 0–12 / 36–48（GT が完全一致する区間）の全画面 |差| の平均。
                  時間方向の漏れ（§10 Temporal spillover、本測定は 9 月）の目安
  edit_median     gen_edit の Target 中心と Edited GT の誤差（median、GT 可視フレーム）
  orig_median     gen_orig の Target 中心と Original GT の誤差
  tracker は score_t_control_v2.track と同一

使い方
  python score_rq2.py --orig gen_armA_rq2orig_G1.mkv --edit gen_armA_rq2edit_G1.mkv \
      --mask m_invariant_R0S0.bin --gt-edit gt_target_center_edited.json \
      --gt-prime gt_target_center_prime.json [--json out.json]
  python score_rq2.py --floor --tv-orig tv_R0S0_original.mkv --tv-edit tv_R0S0_edited.mkv --mask ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_t_control_v2 import track, load_video  # noqa: E402

N, H, W = 49, 480, 832
SPILL = list(range(13, 36))
OUTSIDE = list(range(0, 13)) + list(range(36, 49))


def load_mask(path):
    bits = np.fromfile(path, dtype=np.uint8)
    m = np.unpackbits(bits)[: N * H * W].reshape(N, H, W).astype(bool)
    return m


def spill(o, e, mask):
    d = np.abs(e[SPILL].astype(np.int16) - o[SPILL].astype(np.int16))   # (23,H,W,3)
    m = mask[SPILL]
    vals = d[m]                                                         # (n,3)
    mae = float(vals.mean())
    mse = float((vals.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    p95 = float(np.percentile(vals.max(axis=1), 95))
    out = float(np.abs(e[OUTSIDE].astype(np.int16) - o[OUTSIDE].astype(np.int16)).mean())
    return {"spill_mae": mae, "spill_psnr": psnr, "spill_p95": p95, "outside_mae": out,
            "mask_px": int(m.sum())}


def center_err(gen, gt):
    """Target 中心誤差の median と、個数 ≠ 1 のフレーム（T_coherence と同じ数え方。§41.2）。"""
    e, bad, fail = [], [], []
    for f, (u, v) in sorted(gt.items()):
        det, nc = track(gen[f])
        if det is None:
            fail.append(f)
            continue
        e.append(np.hypot(det[0] - u, det[1] - v))
        if nc != 1:
            bad.append(f)
    return float(np.median(e)), len(e), bad, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True)
    ap.add_argument("--orig"); ap.add_argument("--edit")
    ap.add_argument("--gt-edit"); ap.add_argument("--gt-prime")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--tv-orig"); ap.add_argument("--tv-edit")
    ap.add_argument("--json")
    a = ap.parse_args()
    mask = load_mask(a.mask)
    if a.floor:
        r = spill(load_video(a.tv_orig), load_video(a.tv_edit), mask)
        print("=== evaluator floor（GT 同士。M_invariant は GT 差 ≤ 2 の画素なので 0〜2 の範囲）===")
        print("spill_mae %.4f  psnr %.2f  p95 %.1f  outside_mae %.4f  mask_px %d" % (
            r["spill_mae"], r["spill_psnr"], r["spill_p95"], r["outside_mae"], r["mask_px"]))
        return
    o, e = load_video(a.orig), load_video(a.edit)
    r = spill(o, e, mask)
    ge = {x["index"]: (x["u"], x["v"]) for x in json.load(open(a.gt_edit, encoding="utf-8"))["frames"] if x["npx"]}
    go = {x["index"]: (x["u_sc"], x["v_sc"]) for x in json.load(open(a.gt_prime, encoding="utf-8"))["frames"]
          if "u_sc" in x}
    r["edit_median"], r["edit_n"], r["edit_coh_bad"], r["edit_fail"] = center_err(e, ge)
    r["orig_median"], r["orig_n"], r["orig_coh_bad"], r["orig_fail"] = center_err(o, go)
    r["orig"], r["edit"] = a.orig, a.edit
    print("=== %s vs %s ===" % (os.path.basename(a.edit), os.path.basename(a.orig)))
    print("spill_mae %.3f (8bit)  psnr %.2f dB  p95 %.1f  outside_mae %.3f" % (
        r["spill_mae"], r["spill_psnr"], r["spill_p95"], r["outside_mae"]))
    print("edit realized: median %.2f px (n=%d) count!=1 %s   orig: median %.2f px (n=%d) count!=1 %s" % (
        r["edit_median"], r["edit_n"], r["edit_coh_bad"], r["orig_median"], r["orig_n"], r["orig_coh_bad"]))
    if a.json:
        json.dump(r, open(a.json, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
