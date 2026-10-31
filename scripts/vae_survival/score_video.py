"""
VAE往復の採点スクリプト（動画版・ラウンド3）

静止画版 measure_vae_roundtrip.py からの変更点:
  1. 全49フレーム × 全点を採点する
  2. サンプリングを2通り取り、失敗の原因を分離する
       本来の位置       → identity_at_gt（主指標）
       周辺20pxを探索   → 最良色誤差 と ズレ量(px)
     ズレが大きく最良色誤差が小さい = 位置ズレ
     最良色誤差も大きい               = 色が壊れた
     注意: ズレ量は点の「縁」までの距離。中心の実変位は およそ ズレ量 + 半径3px
  3. 潜在チャンク内の位置(0-3)ごとに分けて集計する
     frame 0 は単独で潜在1枚を持つので別枠。静止画テストと同条件
  4. 近傍探索をトーラス距離で行う（ラップアラウンド対応）
  5. 端 edge_band_px 以内の点は採点から除外（採点対象は全速度で同数）

使い方:
    python score_video.py --gt vae_test_video --rt vae_out
    python score_video.py --gt vae_test_video --rt vae_out --speeds 0
"""

import argparse
import glob
import json
import os

import numpy as np

MATCH_RADIUS_PX = 20      # 近傍照合の半径（主指標）
SEARCH_RADIUS_PX = 20     # 周辺探索の窓。分離半径24pxより小さく保つ
THRESHOLD = 0.90


def torus_dist(x0, y0, x1, y1, W, H):
    dx = np.abs(x0 - x1)
    dx = np.minimum(dx, W - dx)
    dy = np.abs(y0 - y1)
    dy = np.minimum(dy, H - dy)
    return np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)


def build_neighbors(px, py, W, H, R):
    """
    トーラス距離R以内の近傍を (n, K) の配列で返す。埋め草は -1。
    一様並進なので近傍関係は全フレームで不変。frame 0 で一度だけ作る。
    """
    n = len(px)
    cell = max(int(R), 1)
    nx, ny = W // cell + 1, H // cell + 1
    grid = {}
    for i in range(n):
        grid.setdefault((int(px[i]) // cell, int(py[i]) // cell), []).append(i)

    lists = []
    for i in range(n):
        gx, gy = int(px[i]) // cell, int(py[i]) // cell
        cand = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                cand.extend(grid.get(((gx + a) % nx, (gy + b) % ny), []))
        cand = np.unique(np.array(cand, dtype=np.int64))
        d = torus_dist(px[cand], py[cand], px[i], py[i], W, H)
        lists.append(cand[d <= R])

    K = max(len(l) for l in lists)
    out = np.full((n, K), -1, dtype=np.int64)
    for i, l in enumerate(lists):
        out[i, :len(l)] = l
    return out


def band_distance(m, lo, hi):
    return np.maximum(lo - m, 0.0) + np.maximum(m - hi, 0.0)


def classify_entity(mean_val, nt, tg):
    d = np.stack([np.abs(mean_val),
                  band_distance(mean_val, nt[0], nt[1]),
                  band_distance(mean_val, tg[0], tg[1])], axis=-1)
    return np.argmin(d, axis=-1)


def score_pass(gt, rt, nb, match_r, search_r):
    """1本ぶんの採点。フレームごとにループする"""
    px, py = gt["px"].astype(np.int64), gt["py"].astype(np.int64)
    col = gt["color"].astype(np.float64)
    is_tgt = gt["is_target"]
    W, H = int(gt["width"]), int(gt["height"])
    T = int(gt["n_frames"])
    band = int(gt["edge_band_px"])
    nt_band = gt["nontarget_band"].astype(np.float64)
    tg_band = gt["target_band"].astype(np.float64)
    cls_true = np.where(is_tgt, 2, 1)

    nb_valid = nb >= 0
    nb_safe = np.where(nb_valid, nb, 0)
    nb_col = col[nb_safe]                                   # (n, K, 3)

    # 描画の重なりで中心画素が他点に上書きされた点を特定する。
    # 描画順はインデックス順（後の点が上）。一様並進なので全フレーム不変。
    radius = int(gt["radius"])
    n_pts = len(col)
    self_i = np.arange(n_pts)[:, None]
    d_px = torus_dist(px[0][nb_safe], py[0][nb_safe],
                      px[0][:, None], py[0][:, None], W, H)
    buried = (nb_valid & (nb_safe > self_i) & (d_px <= radius)).any(axis=1)

    off = np.arange(-search_r, search_r + 1)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    oy, ox = oy.ravel(), ox.ravel()
    win_d = np.sqrt(oy.astype(np.float64) ** 2 + ox.astype(np.float64) ** 2)

    rows = []
    for t in range(T):
        x, y = px[t], py[t]
        inner = (x >= band) & (x < W - band) & (y >= band) & (y < H - band)
        idx = np.nonzero(inner & ~buried)[0]
        if len(idx) == 0:
            continue

        frame = rt[t].astype(np.float64)

        # --- サンプリングA: 本来の位置 ---
        s = frame[y[idx], x[idx], :]                        # (m, 3)
        alive = classify_entity(s.mean(axis=1), nt_band, tg_band) != 0
        cls_rt = classify_entity(s.mean(axis=1), nt_band, tg_band)
        entity_ok = cls_rt == cls_true[idx]

        d_nb = np.linalg.norm(nb_col[idx] - s[:, None, :], axis=2)
        d_nb = np.where(nb_valid[idx], d_nb, np.inf)
        ident = nb[idx][np.arange(len(idx)), np.argmin(d_nb, axis=1)] == idx

        mae = np.abs(s - col[idx]).mean(axis=1)

        # --- サンプリングB: 周辺探索 ---
        wy = y[idx][:, None] + oy[None, :]
        wx = x[idx][:, None] + ox[None, :]
        win = frame[wy, wx, :]                              # (m, P, 3)
        derr = np.linalg.norm(win - col[idx][:, None, :], axis=2)
        # 色誤差が同点のときは距離が近い方を選ぶ（点の内部は全画素が同色）
        best = np.argmin(derr + win_d[None, :] * 1e-4, axis=1)
        best_err = derr[np.arange(len(idx)), best]
        drift = win_d[best]

        chunk_pos = -1 if t == 0 else (t - 1) % 4
        rows.append({
            "t": t, "chunk_pos": chunk_pos, "idx": idx, "is_tgt": is_tgt[idx],
            "alive": alive, "entity_ok": entity_ok, "ident": ident,
            "mae": mae, "best_err": best_err, "drift": drift,
        })
    return rows, float(buried.mean())


def agg(rows, keys=None, mask_fn=None):
    def take(name):
        v = [r[name][mask_fn(r)] if mask_fn else r[name] for r in rows
             if keys is None or r["chunk_pos"] in keys]
        v = [a for a in v if len(a)]
        return np.concatenate(v) if v else np.array([])

    ident, alive = take("ident"), take("alive")
    if len(ident) == 0:
        return None
    return {
        "n": int(len(ident)),
        "identity_at_gt": round(float(ident.mean()), 4),
        "survival": round(float(alive.mean()), 4),
        "entity_accuracy": round(float(take("entity_ok").mean()), 4),
        "color_mae_at_gt": round(float(take("mae").mean()), 2),
        "best_color_err": round(float(np.median(take("best_err"))), 2),
        "drift_median_px": round(float(np.median(take("drift"))), 2),
        "drift_p90_px": round(float(np.percentile(take("drift"), 90)), 2),
        "drift_within_3px_pct": round(100 * float((take("drift") <= 3).mean()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="vae_test_video")
    ap.add_argument("--rt", default="vae_out")
    ap.add_argument("--speeds", nargs="*", type=int, default=None)
    ap.add_argument("--out", default="score_report.json")
    args = ap.parse_args()

    npys = sorted(glob.glob(os.path.join(args.rt, "v*_*.npy")))
    if not npys:
        raise FileNotFoundError(f"復元結果が無い: {args.rt}")

    results = []
    nb_cache = None
    for p in npys:
        stem = os.path.basename(p)[:-4]
        v_str, mode = stem.split("_", 1)
        V = int(v_str[1:])
        if args.speeds is not None and V not in args.speeds:
            continue

        gt = np.load(os.path.join(args.gt, f"v{V}_points.npz"), allow_pickle=True)
        rt = np.load(p)
        assert rt.shape[0] == int(gt["n_frames"]), f"{stem}: フレーム数不一致"

        if nb_cache is None:
            nb_cache = build_neighbors(gt["px"][0].astype(np.int64),
                                       gt["py"][0].astype(np.int64),
                                       int(gt["width"]), int(gt["height"]),
                                       MATCH_RADIUS_PX)
            print(f"近傍テーブル構築: 最大{nb_cache.shape[1]}近傍 "
                  f"(平均{(nb_cache>=0).sum(1).mean():.1f})")

        rows, gt_buried = score_pass(gt, rt, nb_cache, MATCH_RADIUS_PX,
                                     SEARCH_RADIUS_PX)

        rec = {"speed": V, "mode": mode,
               "buried_pct": round(100 * float(gt_buried), 1),
               "overall": agg(rows)}
        rec["frame0_only"] = agg(rows, keys=[-1])
        rec["chunked"] = {str(k): agg(rows, keys=[k]) for k in range(4)}
        rec["target"] = agg(rows, mask_fn=lambda r: r["is_tgt"])
        rec["nontarget"] = agg(rows, mask_fn=lambda r: ~r["is_tgt"])
        results.append(rec)
        print(f"  V={V:>2} {mode:<9} identity {rec['overall']['identity_at_gt']:.1%}")

    results.sort(key=lambda r: (r["mode"], r["speed"]))

    print("\n" + "=" * 92)
    print(f"{'V':>3} {'mode':<10}{'生存':>7}{'ident@GT':>10}{'Entity':>8}"
          f"{'色誤差':>8}{'最良色誤差':>11}{'ズレ中央':>9}{'ズレp90':>9}{'判定':>7}")
    print("-" * 92)
    for r in results:
        o = r["overall"]
        v = "成立" if o["identity_at_gt"] >= THRESHOLD else "不足"
        print(f"{r['speed']:>3} {r['mode']:<10}{o['survival']:>7.1%}"
              f"{o['identity_at_gt']:>10.1%}{o['entity_accuracy']:>8.1%}"
              f"{o['color_mae_at_gt']:>8}{o['best_color_err']:>11}"
              f"{o['drift_median_px']:>9}{o['drift_p90_px']:>9}{v:>7}")

    print("\n" + "=" * 72)
    print("潜在チャンク内の位置ごとの identity@GT")
    print(f"{'V':>3} {'mode':<10}{'frame0':>9}{'束1番目':>9}{'2番目':>8}"
          f"{'3番目':>8}{'4番目':>8}")
    print("-" * 72)
    for r in results:
        def g(d):
            return f"{d['identity_at_gt']:.1%}" if d else "-"
        print(f"{r['speed']:>3} {r['mode']:<10}{g(r['frame0_only']):>9}"
              + "".join(f"{g(r['chunked'][str(k)]):>8}" for k in range(4)))
    print("=" * 72)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"match_radius_px": MATCH_RADIUS_PX,
                   "search_radius_px": SEARCH_RADIUS_PX,
                   "threshold": THRESHOLD, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n保存: {args.out}  （{len(results)}件）")


if __name__ == "__main__":
    main()
