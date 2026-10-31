r"""
hist_minvariant.py  —  §18-7（M_invariant の許容値）を決めるための差分ヒストグラム

やること
  1. tv PNG の original / edited を全 index で突合し、bit 一致区間を assert
  2. §11 の前4条件（Non-target / 両変種可視 / 深度順序 / 同一表面）で画素をマスク
  3. 残った画素で |GT_orig - GT_edit| のヒストグラムを取る
  4. Target シルエット近傍（AA フリンジ）と遠方（影の掃引）を分けて集計

Blender 必要（EXR 読み出しに bpy.data.images.load を使う）。GPU 不要。
png_io.py と同じディレクトリに置くか、--pngio でパスを渡す。

  & $BL --background --factory-startup --python hist_minvariant.py "--" ^
      --geo-orig geo\ORIG --geo-edit geo\EDIT ^
      --tv-orig tv\R0S0\ORIG --tv-edit tv\R0S0\EDIT ^
      --out w3_minvariant_report.json
"""
import argparse
import json
import os
import sys

import numpy as np
import bpy

WIDTH = 832
HEIGHT = 480
NPIX = WIDTH * HEIGHT

TARGET_ID = 3
DEPTH_TOL = 1e-4          # float32 前提（作法#21）
NEAR_RADIUS = 2           # city-block

# §10 / §26：original と edited が一致する index 区間
SHARED_INDEX = list(range(0, 13)) + list(range(36, 49))
DIFF_INDEX = list(range(13, 36))


def load_exr_ch0(path):
    """EXR の channel 0 を (h, w) float32 で返す。行は上から下。"""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
        w, h = img.size[0], img.size[1]
        ch = img.channels
        if (w, h) != (WIDTH, HEIGHT):
            raise ValueError("size %dx%d != %dx%d" % (w, h, WIDTH, HEIGHT))
        buf = np.empty(w * h * ch, dtype=np.float32)
        img.pixels.foreach_get(buf)
        arr = buf.reshape(h, w, ch)[::-1, :, 0].copy()   # pixels は下から上
    finally:
        bpy.data.images.remove(img)
    return arr


def load_png(read_rgb, path):
    """PNG を (h, w, 3) int16 で返す。行は上から下（png_io の走査順そのまま）。"""
    px, w, h = read_rgb(path)
    if (w, h) != (WIDTH, HEIGHT):
        raise ValueError("size %dx%d != %dx%d" % (w, h, WIDTH, HEIGHT))
    return np.frombuffer(px, dtype=np.uint8).reshape(h, w, 3).astype(np.int16), px


def dilate1(m):
    """4近傍 1px 膨張。"""
    o = m.copy()
    o[1:, :] |= m[:-1, :]
    o[:-1, :] |= m[1:, :]
    o[:, 1:] |= m[:, :-1]
    o[:, :-1] |= m[:, 1:]
    return o


def dilate(m, r):
    for _ in range(r):
        m = dilate1(m)
    return m


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo-orig", required=True)
    ap.add_argument("--geo-edit", required=True)
    ap.add_argument("--tv-orig", required=True)
    ap.add_argument("--tv-edit", required=True)
    ap.add_argument("--pngio", default=None, help="png_io.py のあるディレクトリ")
    ap.add_argument("--out", default="w3_minvariant_report.json")
    a = ap.parse_args(argv)

    here = a.pngio or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from png_io import read_rgb                                    # noqa: E402

    fails = []
    def check(cond, label):
        print(("PASS  " if cond else "FAIL  ") + label)
        if not cond:
            fails.append(label)

    # ---- 1. bit 一致の assert -------------------------------------------
    print("\n=== 1. tv PNG bit identity (original vs edited) ===")
    identical = {}
    for idx in range(49):
        po = os.path.join(a.tv_orig, "tv_%04d.png" % idx)
        pe = os.path.join(a.tv_edit, "tv_%04d.png" % idx)
        _, bo = load_png(read_rgb, po)
        _, be = load_png(read_rgb, pe)
        identical[idx] = (bo == be)

    shared_ok = all(identical[i] for i in SHARED_INDEX)
    diff_ok = all(not identical[i] for i in DIFF_INDEX)
    check(shared_ok, "index 0-12 / 36-48 (%d frames) are bit-identical" % len(SHARED_INDEX))
    check(diff_ok, "index 13-35 (%d frames) all differ" % len(DIFF_INDEX))
    if not shared_ok:
        bad = [i for i in SHARED_INDEX if not identical[i]]
        print("      differing shared index: %s" % bad)
    if not diff_ok:
        bad = [i for i in DIFF_INDEX if identical[i]]
        print("      identical edit index: %s" % bad)

    # ---- 2-4. マスクとヒストグラム --------------------------------------
    print("\n=== 2. mask + histogram over index 13-35 ===")
    hist_near = np.zeros(256, dtype=np.int64)
    hist_far = np.zeros(256, dtype=np.int64)
    per_frame = []
    tot_depth_violation = 0
    tot_empty = 0

    for idx in DIFF_INDEX:
        frame = idx + 1                                  # 入力は frame 1..49
        ido = np.rint(load_exr_ch0(os.path.join(a.geo_orig, "id_%04d.exr" % frame))).astype(np.int32)
        ide = np.rint(load_exr_ch0(os.path.join(a.geo_edit, "id_%04d.exr" % frame))).astype(np.int32)
        dpo = load_exr_ch0(os.path.join(a.geo_orig, "depth_%04d.exr" % frame))
        dpe = load_exr_ch0(os.path.join(a.geo_edit, "depth_%04d.exr" % frame))

        ao, _ = load_png(read_rgb, os.path.join(a.tv_orig, "tv_%04d.png" % idx))
        ae, _ = load_png(read_rgb, os.path.join(a.tv_edit, "tv_%04d.png" % idx))

        empty = (ido == 0) | (ide == 0)
        nontarget = (ido != TARGET_ID) & (ide != TARGET_ID)
        same_id = (ido == ide)
        base = same_id & nontarget & (~empty)
        depth_ok = np.abs(dpo - dpe) < DEPTH_TOL
        mask = base & depth_ok

        tot_depth_violation += int(np.count_nonzero(base & (~depth_ok)))
        tot_empty += int(np.count_nonzero(empty))

        near = dilate((ido == TARGET_ID) | (ide == TARGET_ID), NEAR_RADIUS)

        dmax = np.abs(ao - ae).max(axis=2).astype(np.int32)
        mn = mask & near
        mf = mask & (~near)
        hist_near += np.bincount(dmax[mn], minlength=256).astype(np.int64)
        hist_far += np.bincount(dmax[mf], minlength=256).astype(np.int64)

        per_frame.append({
            "index": idx,
            "masked": int(mask.sum()),
            "near": int(mn.sum()),
            "far": int(mf.sum()),
            "far_nonzero": int((dmax[mf] > 0).sum()),
            "far_max": int(dmax[mf].max()) if mf.any() else 0,
        })

    check(tot_depth_violation == 0,
          "depth agrees wherever id agrees (violations=%d)" % tot_depth_violation)
    print("      empty pixels (id==0) total: %d" % tot_empty)

    hist_all = hist_near + hist_far
    tot = int(hist_all.sum())
    tot_near = int(hist_near.sum())
    tot_far = int(hist_far.sum())

    print("\n--- masked pixel budget (23 frames) ---")
    print("%-10s %12s %8s" % ("group", "pixels", "share"))
    print("%-10s %12d %8s" % ("all", tot, "100.00%"))
    print("%-10s %12d %7.2f%%" % ("near", tot_near, 100.0 * tot_near / max(tot, 1)))
    print("%-10s %12d %7.2f%%" % ("far", tot_far, 100.0 * tot_far / max(tot, 1)))
    print("%-10s %12d %7.2f%%" % ("(screen)", NPIX * len(DIFF_INDEX), 100.0))
    print("mask coverage of screen: %.2f%%" % (100.0 * tot / (NPIX * len(DIFF_INDEX))))

    print("\n--- histogram of |delta|max, bins 0..32 ---")
    print("%5s %12s %12s %12s" % ("d", "far", "near", "all"))
    for d in range(33):
        if hist_all[d] == 0 and d > 0:
            print("%5d %12d %12d %12d" % (d, hist_far[d], hist_near[d], hist_all[d]))
        else:
            print("%5d %12d %12d %12d" % (d, hist_far[d], hist_near[d], hist_all[d]))

    print("\n--- histogram, 33.. in 8-wide bins ---")
    print("%9s %12s %12s %12s" % ("range", "far", "near", "all"))
    for lo in range(33, 256, 8):
        hi = min(lo + 7, 255)
        f = int(hist_far[lo:hi + 1].sum())
        n = int(hist_near[lo:hi + 1].sum())
        if f + n:
            print("%9s %12d %12d %12d" % ("%d-%d" % (lo, hi), f, n, f + n))

    print("\n--- cumulative (share of group <= t) ---")
    print("%5s %10s %10s %10s" % ("t", "far", "near", "all"))
    cf = np.cumsum(hist_far)
    cn = np.cumsum(hist_near)
    ca = np.cumsum(hist_all)
    for t in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        print("%5d %9.4f%% %9.4f%% %9.4f%%" % (
            t,
            100.0 * cf[t] / max(tot_far, 1),
            100.0 * cn[t] / max(tot_near, 1),
            100.0 * ca[t] / max(tot, 1)))

    # 谷の候補：1 以上で連続する空ビンの先頭
    gaps = []
    d = 1
    while d < 200:
        if hist_far[d] == 0:
            e = d
            while e < 255 and hist_far[e] == 0:
                e += 1
            if e - d >= 2:
                gaps.append([d, e - 1])
            d = e
        else:
            d += 1
    print("\n--- empty runs in 'far' (valley candidates, width>=2) ---")
    if gaps:
        for lo, hi in gaps[:8]:
            print("      %d - %d" % (lo, hi))
    else:
        print("      none (distribution is continuous -> use the 3-value "
              "sensitivity plan instead of a single threshold)")

    print("\n--- M_invariant area at candidate tolerances ---")
    print("%5s %14s %10s %14s" % ("tol", "px/frame", "screen%", "far-only px/f"))
    nfr = len(DIFF_INDEX)
    for t in (0, 1, 2, 4, 8, 16, 24, 32):
        keep = int(ca[t])
        keepf = int(cf[t])
        print("%5d %14.1f %9.2f%% %14.1f" % (
            t, keep / nfr, 100.0 * keep / (NPIX * nfr), keepf / nfr))

    print("\n--- per frame (far group) ---")
    print("%6s %10s %10s %12s %8s" % ("index", "masked", "far", "far_nonzero", "far_max"))
    for r in per_frame:
        print("%6d %10d %10d %12d %8d" % (
            r["index"], r["masked"], r["far"], r["far_nonzero"], r["far_max"]))

    report = {
        "width": WIDTH, "height": HEIGHT,
        "depth_tol": DEPTH_TOL, "near_radius": NEAR_RADIUS,
        "shared_index": SHARED_INDEX, "diff_index": DIFF_INDEX,
        "bit_identical": {str(k): bool(v) for k, v in identical.items()},
        "depth_violations": tot_depth_violation,
        "empty_pixels": tot_empty,
        "totals": {"all": tot, "near": tot_near, "far": tot_far},
        "hist_far": hist_far.tolist(),
        "hist_near": hist_near.tolist(),
        "valley_candidates": gaps,
        "per_frame": per_frame,
        "fails": fails,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\nwrote %s" % a.out)
    print("FAILS: %d" % len(fails))


if __name__ == "__main__":
    main()
