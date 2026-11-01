# -*- coding: utf-8 -*-
"""score_iou.py — Target Mask IoU（§12 の Control fidelity の 1 指標、報告のみ・探索的）。（W12）

生成動画の Target 候補 = tracker と同じ色条件（orange_mask）から Marker 近傍の成分を除いた全画素。
GT = Scene C′ の ID パス（pass_index 3）。GT で Target が見える 44 フレームの IoU の median を出す。
tracker（最大成分の重心）と違い、複製や欠けも形の重なりとして数える。

使い方: python score_iou.py --mask target_mask_prime.bin --dir /workspace/runs/gen --pattern 'gen_arm{arm}_step500_G{n}.mkv' --n 20
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_t_control_v2 import orange_mask, label, MARKER, load_video  # noqa: E402

N, H, W = 49, 480, 832


def target_pixels(img):
    m = orange_mask(img)
    if not m.any():
        return m
    lab, n = label(m)
    for c in range(1, n + 1):
        ys, xs = np.where(lab == c)
        if abs(xs.mean() - MARKER[0]) < 30 and abs(ys.mean() - MARKER[1]) < 30:
            m[lab == c] = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default="gen_arm{arm}_step500_G{n}.mkv")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    gt = np.unpackbits(np.fromfile(a.mask, np.uint8))[: N * H * W].reshape(N, H, W).astype(bool)
    vis = [f for f in range(N) if gt[f].any()]
    print("arm,seed,iou_median,iou_p05")
    for arm in "ABC":
        for n in range(1, a.n + 1):
            p = os.path.join(a.dir, a.pattern.format(arm=arm, n=n))
            if not os.path.exists(p):
                print("%s,%d,MISSING," % (arm, n)); continue
            v = load_video(p)
            ious = []
            for f in vis:
                g = target_pixels(v[f])
                inter = (g & gt[f]).sum(); uni = (g | gt[f]).sum()
                ious.append(inter / uni if uni else 0.0)
            print("%s,%d,%.4f,%.4f" % (arm, n, np.median(ious), np.percentile(ious, 5)), flush=True)


if __name__ == "__main__":
    main()
