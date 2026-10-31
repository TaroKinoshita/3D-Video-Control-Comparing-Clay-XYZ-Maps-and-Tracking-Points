r"""
score_armc.py  —  Phase 0.0（§18-11）の採点。GPU 不要。

主指標は色誤差（§16.8）。判定基準 ≤ 24（最小色間距離 48.0833 の半分）。
Target 帯 72点 / Non-target 帯 168点を分けて集計し、悪い方で判定する。

W1 の score_video.py からの変更（合成データ前提を4つ外した）
  1. トーラス距離 → ユークリッド距離（Scene C は端で繋がらない）
  2. 近傍表を frame 0 で1回 → 毎フレーム再構築（Target だけ動くので近傍が変わる）
  3. 端 110px の採点除外を廃止（ラップしない。切ると Wall/Marker が消える）
  4. buried の判定を「インデックス順」→ render_arm_c.py と同じ「深度の遠い順」

さらに追加
  5. 描画を再現して armC_%04d.png と bit 比較する（中心画素の丸め規約の同一性を保証）
  6. 可視フラグを毎フレーム参照（index 36-40 は Target 可視点が 0）

使い方
  python score_armc.py --points armc_points.json --rt vae_armc --armc armC --out score_armc.json
"""
import argparse
import glob
import json
import os

import numpy as np

MATCH_RADIUS_PX = 20
SEARCH_RADIUS_PX = 20
THRESHOLD_COLOR_ERR = 24.0

RESULTS = {"checks": []}


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-34s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def disc_offsets(r):
    return np.array([(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                     if dx * dx + dy * dy <= r * r], dtype=np.int32)


def replay_frame(W, H, off, u, v, depth, visible, rgb):
    """render_arm_c.draw_frame と同一の規約で再描画する。

    返り値 (canvas, owner, drawn_ids)
      owner[j, i] = その画素を最後に塗った点の id（未塗装は -1）
    """
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    owner = np.full((H, W), -1, dtype=np.int32)
    drawn = []
    for i in range(len(u)):
        if not visible[i]:
            continue
        ci = int(round(u[i] - 0.5))
        cj = int(round(v[i] - 0.5))
        if not (0 <= ci < W and 0 <= cj < H):
            continue
        drawn.append((depth[i], i, ci, cj))
    drawn.sort(key=lambda t: (-t[0], t[1]))          # 遠い順。近い点が上書き
    for _d, i, ci, cj in drawn:
        xs = off[:, 0] + ci
        ys = off[:, 1] + cj
        m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xs[m], ys[m]
        canvas[ys, xs] = rgb[i]
        owner[ys, xs] = i
    return canvas, owner, [(i, ci, cj) for _d, i, ci, cj in drawn]


def band_distance(m, lo, hi):
    return np.maximum(lo - m, 0.0) + np.maximum(m - hi, 0.0)


def classify_entity(mean_val, nt, tg):
    d = np.stack([np.abs(mean_val),
                  band_distance(mean_val, nt[0], nt[1]),
                  band_distance(mean_val, tg[0], tg[1])], axis=-1)
    return np.argmin(d, axis=-1)


def score_variant(D, variant, rt, rgb, is_tgt, off, armc_dir, W, H, nt, tg):
    rows = []
    n_buried_total = 0
    png_mismatch = []
    for row in D["frames"][variant]:
        t = int(row["index"])
        u = np.asarray(row["u"], dtype=np.float64)
        v = np.asarray(row["v"], dtype=np.float64)
        dep = np.asarray(row["depth"], dtype=np.float64)
        vis = np.asarray(row["visible"], dtype=bool)

        canvas, owner, drawn = replay_frame(W, H, off, u, v, dep, vis, rgb)

        if armc_dir:
            p = os.path.join(armc_dir, variant, "armC_%04d.png" % t)
            import png_io
            px, w2, h2 = png_io.read_rgb(p)
            back = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)
            if not np.array_equal(back, canvas):
                png_mismatch.append(t)

        ids = np.array([i for i, _, _ in drawn], dtype=np.int64)
        ci = np.array([c for _, c, _ in drawn], dtype=np.int64)
        cj = np.array([c for _, _, c in drawn], dtype=np.int64)
        if len(ids) == 0:
            continue

        keep = owner[cj, ci] == ids                   # 中心を他点に奪われていない
        n_buried_total += int((~keep).sum())
        ids, ci, cj = ids[keep], ci[keep], cj[keep]
        if len(ids) == 0:
            continue

        gt = rgb[ids].astype(np.float64)
        frame = rt[t].astype(np.float64)
        s = frame[cj, ci, :]

        mae = np.abs(s - gt).mean(axis=1)
        cls_rt = classify_entity(s.mean(axis=1), nt, tg)
        cls_true = np.where(is_tgt[ids], 2, 1)

        # 近傍 20px（ユークリッド。毎フレーム作る）
        pu, pv = u[ids], v[ids]
        dm = np.sqrt((pu[:, None] - pu[None, :]) ** 2
                     + (pv[:, None] - pv[None, :]) ** 2)
        near = dm <= MATCH_RADIUS_PX
        cand_col = gt[None, :, :]
        dcol = np.linalg.norm(cand_col - s[:, None, :], axis=2)
        dcol = np.where(near, dcol, np.inf)
        ident = np.argmin(dcol, axis=1) == np.arange(len(ids))
        n_neigh = near.sum(axis=1)

        # 周辺探索（±20px。画面外は除外）
        o = np.arange(-SEARCH_RADIUS_PX, SEARCH_RADIUS_PX + 1)
        oy, ox = np.meshgrid(o, o, indexing="ij")
        oy, ox = oy.ravel(), ox.ravel()
        wd = np.sqrt(oy.astype(np.float64) ** 2 + ox.astype(np.float64) ** 2)
        wy = cj[:, None] + oy[None, :]
        wx = ci[:, None] + ox[None, :]
        inb = (wy >= 0) & (wy < H) & (wx >= 0) & (wx < W)
        win = frame[np.clip(wy, 0, H - 1), np.clip(wx, 0, W - 1), :]
        derr = np.linalg.norm(win - gt[:, None, :], axis=2)
        derr = np.where(inb, derr, np.inf)
        best = np.argmin(derr + wd[None, :] * 1e-4, axis=1)
        ar = np.arange(len(ids))

        rows.append({
            "t": t, "chunk_pos": -1 if t == 0 else (t - 1) % 4,
            "ids": ids, "is_tgt": is_tgt[ids], "mae": mae,
            "entity_ok": cls_rt == cls_true, "alive": cls_rt != 0,
            "ident": ident, "n_neigh": n_neigh.astype(np.float64),
            "best_err": derr[ar, best], "drift": wd[best],
        })
    return rows, n_buried_total, png_mismatch


def agg(rows, keys=None, mask_fn=None):
    def take(name):
        v = []
        for r in rows:
            if keys is not None and r["chunk_pos"] not in keys:
                continue
            a = r[name][mask_fn(r)] if mask_fn else r[name]
            if len(a):
                v.append(a)
        return np.concatenate(v) if v else np.array([])

    mae = take("mae")
    if len(mae) == 0:
        return None
    return {
        "n": int(len(mae)),
        "color_err": round(float(mae.mean()), 3),
        "color_err_p90": round(float(np.percentile(mae, 90)), 2),
        "color_err_max": round(float(mae.max()), 2),
        "best_color_err": round(float(np.median(take("best_err"))), 2),
        "drift_median_px": round(float(np.median(take("drift"))), 2),
        "entity_accuracy": round(float(take("entity_ok").mean()), 4),
        "survival": round(float(take("alive").mean()), 4),
        "identity": round(float(take("ident").mean()), 4),
        "mean_neighbors": round(float(take("n_neigh").mean()), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--rt", required=True, help="npy のあるディレクトリ")
    ap.add_argument("--armc", default=None,
                   help="armC ディレクトリ。渡すと PNG と bit 比較する")
    ap.add_argument("--pngio", default=None)
    ap.add_argument("--expect-bands", default="30,98,157,225",
                   help="空文字で検査を飛ばす")
    ap.add_argument("--out", default="score_armc.json")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, a.pngio or os.path.dirname(os.path.abspath(__file__)))

    with open(a.points, encoding="utf-8") as f:
        D = json.load(f)
    prm = D["params"]
    W, H = int(prm["width"]), int(prm["height"])
    R = int(prm["radius"])
    off = disc_offsets(R)
    pts = D["points"]
    rgb = np.array([p["rgb"] for p in pts], dtype=np.uint8)
    is_tgt = np.array([p["is_target"] for p in pts], dtype=bool)

    rec("disc_area", len(off) == 113, "%d px (radius %d)" % (len(off), R))
    rec("point_counts",
        int(is_tgt.sum()) == prm["n_target"]
        and int((~is_tgt).sum()) == prm["n_background"],
        "target %d / background %d" % (is_tgt.sum(), (~is_tgt).sum()))

    tg_v = rgb[is_tgt].astype(np.float64)
    nt_v = rgb[~is_tgt].astype(np.float64)
    tg = (float(tg_v.min()), float(tg_v.max()))
    nt = (float(nt_v.min()), float(nt_v.max()))
    if a.expect_bands:
        e = [float(x) for x in a.expect_bands.split(",")]
        rec("entity_bands", (nt[0], nt[1], tg[0], tg[1]) == tuple(e),
            "nt %s / tg %s（期待 %s）" % (nt, tg, e))
    else:
        print("      entity bands nt %s / tg %s" % (nt, tg))

    npys = sorted(glob.glob(os.path.join(a.rt, "*_*.npy")))
    if not npys:
        raise FileNotFoundError("復元結果が無い: %s" % a.rt)

    results = []
    for p in npys:
        stem = os.path.basename(p)[:-4]
        variant, mode = stem.rsplit("_", 1)
        if variant not in D["frames"]:
            print("      skip %s（variant が points に無い）" % stem)
            continue
        rt = np.load(p)
        n_expect = len(D["frames"][variant])
        if rt.shape != (n_expect, H, W, 3):
            rec("shape_%s" % stem, False, "%s（期待 %s）"
                % (rt.shape, (n_expect, H, W, 3)))
            continue

        rows, n_buried, mism = score_variant(
            D, variant, rt, rgb, is_tgt, off, a.armc, W, H, nt, tg)
        rec("no_buried_%s" % stem, n_buried == 0, "上書きされた中心 %d" % n_buried)
        if a.armc:
            rec("png_replay_%s" % stem, not mism,
                "不一致 index %s" % (mism[:5] if mism else "なし"))

        r = {"variant": variant, "mode": mode,
             "overall": agg(rows),
             "target": agg(rows, mask_fn=lambda x: x["is_tgt"]),
             "nontarget": agg(rows, mask_fn=lambda x: ~x["is_tgt"]),
             "frame0": agg(rows, keys=[-1]),
             "chunked": {str(k): agg(rows, keys=[k]) for k in range(4)},
             "per_frame": [{"t": x["t"], "n": int(len(x["mae"])),
                            "n_tgt": int(x["is_tgt"].sum()),
                            "color_err": round(float(x["mae"].mean()), 2),
                            "color_err_tgt": (round(float(x["mae"][x["is_tgt"]].mean()), 2)
                                              if x["is_tgt"].any() else None)}
                           for x in rows]}
        results.append(r)

    print("\n" + "=" * 78)
    print("%-9s %-10s %8s %8s %8s %8s %6s" % ("variant", "mode", "全体",
                                              "Target", "NonTgt", "最悪", "判定"))
    print("-" * 78)
    verdicts = {}
    for r in results:
        o, tt, nn = r["overall"], r["target"], r["nontarget"]
        worst = max(x["color_err"] for x in (tt, nn) if x)
        ok = worst <= THRESHOLD_COLOR_ERR
        if r["mode"] == "temporal":
            verdicts[r["variant"]] = (worst, ok)
        print("%-9s %-10s %8.3f %8s %8s %8.3f %6s"
              % (r["variant"], r["mode"], o["color_err"],
                 ("%.3f" % tt["color_err"]) if tt else "-",
                 ("%.3f" % nn["color_err"]) if nn else "-",
                 worst, "OK" if ok else "NG"))

    print("\n潜在チャンク内の位置別 色誤差（§16.13 の再現確認）")
    print("%-9s %-10s %8s %8s %8s %8s %8s" % ("variant", "mode", "frame0",
                                              "束1", "束2", "束3", "束4"))
    print("-" * 78)
    for r in results:
        def g(d):
            return ("%.3f" % d["color_err"]) if d else "-"
        print("%-9s %-10s %8s %8s %8s %8s %8s"
              % (r["variant"], r["mode"], g(r["frame0"]),
                 *[g(r["chunked"][str(k)]) for k in range(4)]))

    print("\n補助指標（Arm 間比較には使わない。§16.12）")
    print("%-9s %-10s %9s %9s %8s %8s" % ("variant", "mode", "identity",
                                          "近傍数", "Entity", "最良誤差"))
    print("-" * 78)
    for r in results:
        o = r["overall"]
        print("%-9s %-10s %8.1f%% %9.2f %7.1f%% %8.2f"
              % (r["variant"], r["mode"], 100 * o["identity"],
                 o["mean_neighbors"], 100 * o["entity_accuracy"],
                 o["best_color_err"]))

    print("\nフレーム別 色誤差（temporal / Target 帯）")
    print("%5s %6s %6s %8s %10s" % ("index", "n", "n_tgt", "全体", "Target"))
    for r in results:
        if r["mode"] != "temporal":
            continue
        print("--- %s ---" % r["variant"])
        for x in r["per_frame"]:
            print("%5d %6d %6d %8.2f %10s"
                  % (x["t"], x["n"], x["n_tgt"], x["color_err"],
                     ("%.2f" % x["color_err_tgt"]) if x["color_err_tgt"] is not None
                     else "-"))

    print("\n" + "=" * 78)
    all_ok = all(ok for _, ok in verdicts.values())
    for v, (w, ok) in verdicts.items():
        print("§18-11  %-9s worst=%.3f  基準 %.1f  →  %s"
              % (v, w, THRESHOLD_COLOR_ERR, "通過" if ok else "不通過"))
    print("判定: %s" % ("240 点を維持できる" if all_ok else "点数を減らす（240 → 200）"))
    nfail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("CHECKS FAIL: %d" % nfail)
    print("=" * 78)

    RESULTS.update({"threshold": THRESHOLD_COLOR_ERR,
                    "match_radius_px": MATCH_RADIUS_PX,
                    "search_radius_px": SEARCH_RADIUS_PX,
                    "entity_bands": {"nontarget": nt, "target": tg},
                    "results": results,
                    "verdict": {k: {"worst": v[0], "pass": v[1]}
                                for k, v in verdicts.items()}})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=1, ensure_ascii=False)
    print("保存: %s" % a.out)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
