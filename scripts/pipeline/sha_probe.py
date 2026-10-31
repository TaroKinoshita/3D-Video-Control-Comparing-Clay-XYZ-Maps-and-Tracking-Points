#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
sha_probe.py  —  W7
mkv の content sha を複数の計算方式で出し、protocol の記録値と突合して
「どの方式で記録されたか」を実測で確定する。

rev.11 変更9 の未解決点：
  §2.2/§36.6 の記録値（tv=bba8…, armC=46ab…）と
  W6 の三者一致値（tv=6190557123fdc2b3, armC=7c08f981503edb34）が違う。
  png_to_ffv1.content_sha256 は sha256(str(shape) + tobytes) だが、
  rev.11 は「imageio デコード RGB の tobytes sha」と書いており shape 前置が無い。
  どちらが記録値を再現するかは推測せず実測する。

使い方（ローカル / PowerShell、W3\.py で実行）:
  python sha_probe.py
"""
import hashlib
import os

import numpy as np
import imageio.v2 as iio

ROOT = r"C:\Bachelor Thesis\SummerWork"

# (ラベル, パス, 記録値の候補（先頭16文字。無ければ None）)
TARGETS = [
    ("tv_original",   os.path.join(ROOT, r"W3\mkv\tv_R0S0_original.mkv"), "6190557123fdc2b3"),
    ("armC_original", os.path.join(ROOT, r"W3\mkv\armC_original.mkv"),    "7c08f981503edb34"),
    ("armA_original", os.path.join(ROOT, r"W3\mkv\armA_original.mkv"),    None),
    ("armB_original", os.path.join(ROOT, r"W3\mkv\armB_original.mkv"),    None),
    ("armC_prime",    os.path.join(ROOT, r"W4\mkv\armC_prime.mkv"),       "fde80efbe48dcb51"),
    ("tv_prime",      os.path.join(ROOT, r"W3\mkv\tv_R0S0_prime.mkv"),    None),
    ("armA_prime",    os.path.join(ROOT, r"W3\mkv\armA_prime.mkv"),       None),
    ("armB_prime",    os.path.join(ROOT, r"W3\mkv\armB_prime.mkv"),       None),
]


def decode(path):
    """DiffSynth の LoadVideo と同じ経路（imageio.get_reader）で (N,H,W,3) uint8。"""
    r = iio.get_reader(path)
    n = r.count_frames()
    st = np.stack([r.get_data(i) for i in range(n)])
    r.close()
    return np.ascontiguousarray(st, dtype=np.uint8)


def methods(st, path):
    """考えられる計算方式を全部出す。"""
    out = {}

    h = hashlib.sha256()
    h.update(str(st.shape).encode("utf-8"))
    h.update(st.tobytes())
    out["A shape+tobytes"] = h.hexdigest()          # png_to_ffv1 の現行方式

    out["B tobytes"] = hashlib.sha256(st.tobytes()).hexdigest()

    # C: フレームごとに update（連結順は B と同じバイト列になるので一致するはず。
    #    一致しなければ ascontiguousarray の効き方が違うということ）
    h = hashlib.sha256()
    for i in range(st.shape[0]):
        h.update(np.ascontiguousarray(st[i]).tobytes())
    out["C per-frame"] = h.hexdigest()

    # D: BGR 順（cv2 経路で作られた可能性）
    out["D tobytes BGR"] = hashlib.sha256(
        np.ascontiguousarray(st[:, :, :, ::-1]).tobytes()).hexdigest()

    # E: 平面順 (N,3,H,W)（gbrp / torch 系の並び）
    out["E tobytes CHW"] = hashlib.sha256(
        np.ascontiguousarray(st.transpose(0, 3, 1, 2)).tobytes()).hexdigest()

    # F: ファイルそのもの
    fh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            fh.update(chunk)
    out["F file"] = fh.hexdigest()

    return out


def main():
    width = max(len(k) for k in
                ["A shape+tobytes", "B tobytes", "C per-frame",
                 "D tobytes BGR", "E tobytes CHW", "F file"])
    hits = {}
    for label, path, recorded in TARGETS:
        if not os.path.exists(path):
            print("SKIP  %-14s ファイルが無い: %s" % (label, path))
            print()
            continue
        st = decode(path)
        res = methods(st, path)
        print("== %s   shape %s" % (label, st.shape))
        if recorded:
            print("   記録値 : %s" % recorded)
        for name, digest in res.items():
            mark = ""
            if recorded and digest.startswith(recorded):
                mark = "   <<< 記録値に一致"
                hits.setdefault(name, []).append(label)
            print("   %-*s %s%s" % (width, name, digest[:32], mark))
        print()

    print("=" * 70)
    if hits:
        for name, labels in hits.items():
            print("記録値を再現した方式: %s  (%s)" % (name, ", ".join(labels)))
    else:
        print("どの方式も記録値を再現しなかった。")
        print("→ ファイルの版が違う可能性。armc_prime_points.json の2版を確認する。")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
