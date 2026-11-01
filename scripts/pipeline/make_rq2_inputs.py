#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_rq2_inputs.py — RQ2（編集の漏れ）の入力を作る（GPU 不要）。（W11, 2026-11-01）

作るもの
  1. arm{A,B,C}_edited.mkv / tv_R0S0_edited.mkv
       data/render/<arm>/edited/*.png → FFV1 gbrp bitexact（png_to_ffv1.py と同じフラグ）
       PNG ↔ mkv の bit 一致を確認する
  2. M_invariant（§11）
       不変であるべき画素 = 次をすべて満たす画素
         id_orig != 3 かつ id_edit != 3                 （Target でない）
         id_orig == id_edit かつ |depth_o − depth_e| < 1e-4（同じ表面・同じ深度順序）
         (id_orig==3 | id_edit==3) を city-block 2 px 膨張した領域の外（AA フリンジ除外）
         max_c |tv_orig − tv_edit| ≤ 2                  （GT 上で変化していない）
       照合値（§32.1、W3）：index 13–35 の平均 379,673 px/frame（画面の 95.070%）
       保存：np.packbits した uint8 を .bin（.npy は .gitignore 対象のため）+ meta JSON
  3. gt_target_center_edited.json
       Scene C Edited の ids==3 重心（RQ2 で「編集が実現したか」を測る GT）

使い方
  python make_rq2_inputs.py --repo .
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

H, W, N = 480, 832, 49
TGT = 3.0
SPILL_FRAMES = list(range(13, 36))       # §32.1：original と edited が違うのは index 13–35
EXPECT_MINV_MEAN = 379673                # §32.1 / §11
TOL_GT = 2                               # §11


def read_exr(path):
    import OpenEXR
    ch = OpenEXR.File(path).channels()
    a = ch["RGBA"].pixels if "RGBA" in ch else ch[sorted(ch)[0]].pixels
    return np.asarray(a[..., 0] if a.ndim == 3 else a, dtype=np.float32)


def read_png_dir(d, prefix):
    import imageio.v2 as imageio
    return np.stack([np.asarray(imageio.imread(os.path.join(d, "%s_%04d.png" % (prefix, i))))[..., :3]
                     for i in range(N)])


def read_mkv(p):
    import imageio
    rd = imageio.get_reader(p)
    a = np.stack([np.asarray(f)[..., :3] for f in rd])
    rd.close()
    return a


def write_ffv1(arr, path):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+bitexact",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", "16", "-i", "pipe:0",
           "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-g", "1",
           "-flags", "+bitexact", "-fflags", "+bitexact", path]
    subprocess.run(cmd, input=np.ascontiguousarray(arr).tobytes(), check=True)


def sha_b(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dilate_cityblock(m, r):
    out = m.copy()
    for _ in range(r):
        n = out.copy()
        n[1:] |= out[:-1]; n[:-1] |= out[1:]; n[:, 1:] |= out[:, :-1]; n[:, :-1] |= out[:, 1:]
        out = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    R = a.repo
    out_mkv = os.path.join(R, "data/inputs_rq2")
    os.makedirs(out_mkv, exist_ok=True)

    # 1. edited の mkv
    print("| file | content sha(B) | PNG↔mkv bit 一致 |")
    print("|---|---|---|")
    for name, d, pre in [("armA_edited", "data/render/armA/edited", "armA"),
                         ("armB_edited", "data/render/armB/edited", "armB"),
                         ("armC_edited", "data/render/armC/edited", "armC"),
                         ("tv_R0S0_edited", "data/render/tv/R0S0/edited", "tv")]:
        png = read_png_dir(os.path.join(R, d), pre)
        p = os.path.join(out_mkv, name + ".mkv")
        write_ffv1(png, p)
        back = read_mkv(p)
        ok = np.array_equal(png, back)
        assert ok, name
        print("| %s.mkv | `%s` | %s |" % (name, sha_b(back)[:16], ok))

    # 2. M_invariant
    tv_o = read_mkv(os.path.join(R, "data/inputs/tv_R0S0_original.mkv")).astype(np.int16)
    tv_e = read_mkv(os.path.join(out_mkv, "tv_R0S0_edited.mkv")).astype(np.int16)
    same = [i for i in range(N) if np.array_equal(tv_o[i], tv_e[i])]
    assert same == list(range(0, 13)) + list(range(36, 49)), same   # §32.1 の前提
    mask = np.zeros((N, H, W), bool)
    gt_e = {}
    for i in range(N):
        f = i + 1
        io = read_exr(os.path.join(R, "data/render/geo/original/id_%04d.exr" % f))
        ie = read_exr(os.path.join(R, "data/render/geo/edited/id_%04d.exr" % f))
        do = read_exr(os.path.join(R, "data/render/geo/original/depth_%04d.exr" % f))
        de = read_exr(os.path.join(R, "data/render/geo/edited/depth_%04d.exr" % f))
        tgt = (io == TGT) | (ie == TGT)
        geo = (~tgt) & (io == ie) & (np.abs(do - de) < 1e-4)
        # 同じ id なのに depth が違う画素は幾何パスの異常（§11）
        assert int(((io == ie) & (~tgt) & (np.abs(do - de) >= 1e-4)).sum()) == 0, i
        near = dilate_cityblock(tgt, 2)
        gtd = np.abs(tv_o[i] - tv_e[i]).max(axis=2) <= TOL_GT
        mask[i] = geo & ~near & gtd
        me = ie == TGT
        if me.sum():
            ys, xs = np.nonzero(me)
            gt_e[i] = (round(float(xs.mean()), 4), round(float(ys.mean()), 4), int(me.sum()))
    per = mask[SPILL_FRAMES].reshape(len(SPILL_FRAMES), -1).sum(1)
    mean_px = float(per.mean())
    print("\nM_invariant（index 13–35）: mean %.1f px/frame = %.3f%%  (照合値 %d)" % (
        mean_px, 100 * mean_px / (H * W), EXPECT_MINV_MEAN))
    assert abs(mean_px - EXPECT_MINV_MEAN) < 1.0, "§32.1 の照合値を再現しない"

    bits = np.packbits(mask.reshape(-1))
    mp = os.path.join(R, "data/points/m_invariant_R0S0.bin")
    bits.tofile(mp)
    meta = {"made_by": "scripts/pipeline/make_rq2_inputs.py", "shape": [N, H, W],
            "encoding": "np.packbits(mask.reshape(-1)), C order (frame, row, col)",
            "definition": "protocol §11: non-target in both, same id & depth (<1e-4), outside 2px city-block "
                          "dilation of Target (either variant), max_c |tv_orig - tv_edit| <= 2",
            "spill_frames": SPILL_FRAMES, "mean_px_spill_frames": mean_px,
            "sha256_bin": hashlib.sha256(bits.tobytes()).hexdigest()}
    json.dump(meta, open(mp.replace(".bin", ".json"), "w", encoding="utf-8"), indent=1)
    print("wrote", mp, "sha256(first16)", meta["sha256_bin"][:16])

    # 3. edited の GT 中心
    gp = os.path.join(R, "data/points/gt_target_center_edited.json")
    json.dump({"meta": {"made_by": "scripts/pipeline/make_rq2_inputs.py",
                        "source": "data/render/geo/edited/id_%04d.exr, Target = pass_index 3",
                        "definition": "mask centroid in pixel index coords (no +0.5)",
                        "occluded_index": [i for i in range(N) if i not in gt_e]},
               "frames": [{"index": i, "u": gt_e[i][0], "v": gt_e[i][1], "npx": gt_e[i][2]}
                          if i in gt_e else {"index": i, "npx": 0} for i in range(N)]},
              open(gp, "w", encoding="utf-8"), indent=1)
    print("wrote", gp, "visible frames:", len(gt_e))


if __name__ == "__main__":
    main()
