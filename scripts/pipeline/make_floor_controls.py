#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_floor_controls.py — ④ Pipeline floor 用の control を作る（GPU 不要）。（W10, 2026-11-01）

1. Micro-perturbation（§15）
   Scene C′ の control（arm{A,B,C}_prime.mkv）を **右へ 1 px** ずらした mkv を作る。
     out[:, :, 1:] = in[:, :, :-1]、左端の 1 列は元の左端列を複製（edge replicate）
   - 全 arm に同じ操作をする（arm 間で摂動の大きさを揃える）
   - 黒地の Arm C でも、座標色の Arm B でも、端に新しい値を作らない
   - 参照画像・prompt は変えない（摂動は control だけ）

2. Encoding floor（§15「意味的に同じ Control を再保存・再読込」）
   control を一度デコードして FFV1 で再保存し、content sha（方式B）が元と一致するか確かめる。
   一致すれば、DiT に入る画素は bit 一致（§36.6）なので、Encoding floor は
   Determinism floor と同じものになる（GPU で別途測る必要がない）。

出力は FFV1 gbrp bitexact（png_to_ffv1.py と同じフラグ）。

使い方
  python make_floor_controls.py --src data/inputs --out <DIR>
"""
import argparse
import hashlib
import os
import subprocess

import numpy as np

ARMS = ("A", "B", "C")


def read(path):
    import imageio
    rd = imageio.get_reader(path)
    a = np.stack([np.asarray(f)[..., :3] for f in rd])
    rd.close()
    assert a.shape == (49, 480, 832, 3), a.shape
    return a


def sha_b(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_ffv1(arr, path):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+bitexact",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "832x480", "-r", "16", "-i", "pipe:0",
           "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-g", "1",
           "-flags", "+bitexact", "-fflags", "+bitexact", path]
    subprocess.run(cmd, input=np.ascontiguousarray(arr).tobytes(), check=True)


def shift_right_1px(a):
    out = np.empty_like(a)
    out[:, :, 1:] = a[:, :, :-1]
    out[:, :, 0] = a[:, :, 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("| arm | 元 sha(B) | 再保存 sha(B) | 一致 | shift1 sha(B) | 変化画素の割合 |")
    print("|---|---|---|---|---|---|")
    for arm in ARMS:
        src = os.path.join(a.src, "arm%s_prime.mkv" % arm)
        base = read(src)
        # Encoding floor：デコード → 再保存 → 再デコード
        tmp = os.path.join(a.out, "_reencode_arm%s.mkv" % arm)
        write_ffv1(base, tmp)
        re = read(tmp)
        os.remove(tmp)
        same = sha_b(re) == sha_b(base)
        assert same, "再保存で画素が変わった（arm%s）" % arm
        # Micro-perturbation
        sh = shift_right_1px(base)
        assert not np.array_equal(sh, base)
        assert np.array_equal(sh[:, :, 1:], base[:, :, :-1])
        dst = os.path.join(a.out, "arm%s_prime_shift1.mkv" % arm)
        write_ffv1(sh, dst)
        back = read(dst)
        assert np.array_equal(back, sh), "shift1 の mkv が bit 一致しない"
        frac = float((sh != base).any(axis=3).mean())
        print("| %s | `%s` | `%s` | %s | `%s` | %.2f%% |" % (
            arm, sha_b(base)[:16], sha_b(re)[:16], same, sha_b(back)[:16], 100 * frac))


if __name__ == "__main__":
    main()
