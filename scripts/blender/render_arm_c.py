"""
render_arm_c.py  —  W2c-3b
Arm C（Object-Aware 3D Tracking）の Control Video を描画する。

  python render_arm_c.py --points armC/armc_points.json --out armC

**Blender 不要。** armc_points.json に u,v / depth / 可視 / 色が入っているので
numpy + 自前 png_io だけで完結する。

出力  armC/<variant>/armC_%04d.png   （8bit RGB。index 0..48）

protocol の規約
  点の半径 6 px / マスク dx^2+dy^2 <= r^2 / 面積 113 px
  描画順は **深度の遠い順**（近い点が上書きする）
  画面外の点は描かない（色は保持する）
  背景は空画素 = (0, 0, 0)
  PNG は自前実装（zlib / 8bit RGB / filter 0）

中心の丸め
  連続座標 (u, v) に最も近い画素中心を使う。画素 (i, j) の中心は (i+0.5, j+0.5)
  なので i = round(u - 0.5) / j = round(v - 0.5)。
  これで完全に画面内の点は必ず 113 px になる（W1 の測定条件と一致）。
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import png_io
import armc_palette as PAL

RADIUS = 6
RESULTS = {"checks": [], "frames": {}}


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-30s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def disc_offsets(r):
    off = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
           if dx * dx + dy * dy <= r * r]
    return np.array(off, dtype=np.int32)


OFF = disc_offsets(RADIUS)


def draw_frame(W, H, u, v, depth, visible, rgb):
    """1フレームを描く。返り値: (canvas, 描画点数, 重なり画素数, 面積の内訳)"""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    owner = np.full((H, W), -1, dtype=np.int32)     # どの点がその画素を持つか
    idx = [i for i in range(len(u)) if visible[i]]
    # 画面外（中心が範囲外）は描かない
    drawn = []
    for i in idx:
        ci = int(round(u[i] - 0.5))
        cj = int(round(v[i] - 0.5))
        if not (0 <= ci < W and 0 <= cj < H):
            continue
        drawn.append((depth[i], i, ci, cj))
    # 深度の遠い順（近い点が上書き）
    drawn.sort(key=lambda t: (-t[0], t[1]))
    overlap = 0
    areas = {}
    for _d, i, ci, cj in drawn:
        xs = OFF[:, 0] + ci
        ys = OFF[:, 1] + cj
        m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xs[m], ys[m]
        overlap += int((owner[ys, xs] >= 0).sum())
        canvas[ys, xs] = rgb[i]
        owner[ys, xs] = i
        areas[i] = int(m.sum())
    return canvas, len(drawn), overlap, areas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="armC/armc_points.json")
    ap.add_argument("--out", default="armC")
    a = ap.parse_args()

    with open(a.points) as f:
        D = json.load(f)
    pts = D["points"]
    prm = D["params"]
    W = int(prm.get("width", 832))
    H = int(prm.get("height", 480))
    N = len(pts)
    rgb = np.array([p["rgb"] for p in pts], dtype=np.uint8)
    is_tgt = np.array([p["is_target"] for p in pts], dtype=bool)

    print("points %d / %dx%d / pitch %s / 達成分離 %s"
          % (N, W, H, prm.get("pitch"), prm.get("achieved_separation")))

    # パレットの整合
    pal_all = set(tuple(c) for b in PAL.PALETTE for c in PAL.PALETTE[b])
    bad_col = [p["id"] for p in pts if tuple(p["rgb"]) not in pal_all]
    rec("colors_in_palette", not bad_col, "パレット外 %d 点 %s"
        % (len(bad_col), bad_col[:5]))
    rec("disc_area_113", len(OFF) == 113, len(OFF))

    if "depth" not in D["frames"][list(D["frames"])[0]][0]:
        rec("depth_present", False,
            "armc_points.json に depth が無い。select_arm_c.py を再実行して")
        return 1
    rec("depth_present", True, "ok")

    tgt_band = set(tuple(c) for c in PAL.PALETTE["target"])
    total_overlap = 0
    for variant, rows in D["frames"].items():
        vdir = os.path.join(a.out, variant)
        os.makedirs(vdir, exist_ok=True)
        occl_zero = []
        for row in rows:
            i = row["index"]
            canvas, ndrawn, overlap, areas = draw_frame(
                W, H, row["u"], row["v"], row["depth"], row["visible"], rgb)
            total_overlap += overlap

            op = os.path.join(vdir, "armC_%04d.png" % i)
            png_io.write_rgb(op, canvas, W, H)
            px, w2, h2 = png_io.read_rgb(op)
            back = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)

            problems = []
            if not np.array_equal(back, canvas):
                problems.append("PNG の読み戻しが不一致")
            # 完全に画面内の点は面積 113
            inner = [k for k, ar in areas.items()
                     if RADIUS <= round(row["u"][k] - 0.5) < W - RADIUS
                     and RADIUS <= round(row["v"][k] - 0.5) < H - RADIUS]
            wrong = [(k, areas[k]) for k in inner if areas[k] != 113]
            if wrong:
                problems.append("面積が113でない点 %s" % wrong[:3])
            # 出た色がパレット + 黒だけ
            cols = set(map(tuple, np.unique(back.reshape(-1, 3), axis=0)))
            extra = cols - pal_all - {(0, 0, 0)}
            if extra:
                problems.append("パレット外の色 %s" % list(extra)[:3])
            # Target 帯の色が出ているか（完全遮蔽区間では出ないのが正しい）
            has_tgt = bool(cols & tgt_band)
            n_tgt_vis = sum(1 for k in range(N) if row["visible"][k] and is_tgt[k])
            if has_tgt != (n_tgt_vis > 0):
                problems.append("Target 帯の有無が可視点数と矛盾 (%s / %d)"
                                % (has_tgt, n_tgt_vis))
            occl_zero.append((i, n_tgt_vis, has_tgt))

            RESULTS["frames"]["%s_%02d" % (variant, i)] = {
                "drawn": ndrawn, "overlap_px": overlap,
                "n_colors": len(cols) - (1 if (0, 0, 0) in cols else 0),
                "target_visible": n_tgt_vis,
            }
            if problems:
                rec("armC_%s_%02d" % (variant, i), False, " | ".join(problems[:3]))

        files = sorted(x for x in os.listdir(vdir)
                       if x.startswith("armC_") and x.endswith(".png"))
        rec("file_count_%s" % variant, len(files) == len(rows),
            "%d files" % len(files))
        idxs = sorted(int(x[5:9]) for x in files)
        rec("index_continuous_%s" % variant,
            idxs == list(range(len(rows))), "min=%d max=%d" % (idxs[0], idxs[-1]))
        # 完全遮蔽区間で Target 色が0であること
        occ = [(i, n) for i, n, h in occl_zero if 36 <= i <= 40 and (n or h)]
        rec("fully_occluded_%s" % variant, not occ, "残存 %s" % (occ or "なし"))

    rec("no_disc_overlap", total_overlap == 0,
        "重なり画素の総数 %d（分離 %.3f px / 保証には > %s）"
        % (total_overlap, prm["achieved_separation"]["global"],
           prm.get("d_floor_no_overlap")))
    RESULTS["total_overlap_px"] = total_overlap

    npass = sum(1 for c in RESULTS["checks"] if c["ok"])
    nfail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("=" * 66)
    print("SELF-CHECK : PASS %d / FAIL %d" % (npass, nfail))
    print("=" * 66)
    jp = os.path.join(a.out, "render_arm_c.json")
    with open(jp, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("saved : %s" % jp)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
