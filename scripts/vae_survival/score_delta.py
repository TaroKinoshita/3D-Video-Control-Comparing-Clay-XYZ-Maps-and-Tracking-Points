"""
時間オフセット検証（δスイープ）

仮説:
    VAEは4フレームの束を「束の代表時刻」1つとして再構成している。
    だとすると点は消えていない。時刻を取り違えているだけ。

検証:
    復元されたフレーム t を、正解フレーム t+δ の点位置で照合する。
    特定のδで復元率が跳ね上がれば、失敗の正体は情報の消失ではなく時間のズレ。

    さらに束の中の位置ごとに最良δを出す。仮説が正しければ
    束1番目→δ=+2、2番目→+1、3番目→0、4番目→-1 と1ずつずれるはず。
    この階段が出れば「代表時刻は束の3番目」が確定する。

使い方:
    python score_delta.py --gt vae_test_video --rt vae_out
    python score_delta.py --gt vae_test_video --rt vae_out --speeds 17
"""

import argparse
import glob
import os
import json

import numpy as np

from score_video import build_neighbors, torus_dist

MATCH_RADIUS_PX = 20
DELTAS = [-2, -1, 0, 1, 2]      # 端110px > 最大ずれ 2×34=68px なので巻き込みなし
POS_NAMES = {0: "束1番目", 1: "2番目", 2: "3番目", 3: "4番目"}


def burial_mask(gt, nb, W, H):
    """描画の重なりで中心が上書きされた点。一様並進なので全フレーム不変"""
    px, py = gt["px"].astype(np.int64), gt["py"].astype(np.int64)
    radius = int(gt["radius"])
    nb_valid = nb >= 0
    nb_safe = np.where(nb_valid, nb, 0)
    self_i = np.arange(len(px[0]))[:, None]
    d = torus_dist(px[0][nb_safe], py[0][nb_safe],
                   px[0][:, None], py[0][:, None], W, H)
    return (nb_valid & (nb_safe > self_i) & (d <= radius)).any(axis=1)


def sweep(gt, rt, nb):
    px, py = gt["px"].astype(np.int64), gt["py"].astype(np.int64)
    col = gt["color"].astype(np.float64)
    W, H = int(gt["width"]), int(gt["height"])
    T = int(gt["n_frames"])
    band = int(gt["edge_band_px"])

    buried = burial_mask(gt, nb, W, H)
    nb_valid = nb >= 0
    nb_safe = np.where(nb_valid, nb, 0)
    nb_col = col[nb_safe]

    lo, hi = -min(DELTAS), T - 1 - max(DELTAS)
    hits = {d: {p: [] for p in range(4)} for d in DELTAS}

    for t in range(lo, hi + 1):
        # 採点対象はフレームtの位置で決める。全δで同じ点集合になる
        inner = ((px[t] >= band) & (px[t] < W - band) &
                 (py[t] >= band) & (py[t] < H - band))
        idx = np.nonzero(inner & ~buried)[0]
        if len(idx) == 0:
            continue
        pos = (t - 1) % 4
        frame = rt[t].astype(np.float64)

        for d in DELTAS:
            s = frame[py[t + d][idx], px[t + d][idx], :]
            dd = np.linalg.norm(nb_col[idx] - s[:, None, :], axis=2)
            dd = np.where(nb_valid[idx], dd, np.inf)
            ok = nb[idx][np.arange(len(idx)), np.argmin(dd, axis=1)] == idx
            hits[d][pos].append(ok)

    out = {}
    for d in DELTAS:
        per_pos = {}
        allv = []
        for p in range(4):
            v = np.concatenate(hits[d][p]) if hits[d][p] else np.array([])
            per_pos[p] = round(float(v.mean()), 4) if len(v) else None
            allv.append(v)
        allv = np.concatenate([a for a in allv if len(a)])
        out[d] = {"overall": round(float(allv.mean()), 4), "by_pos": per_pos}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="vae_test_video")
    ap.add_argument("--rt", default="vae_out")
    ap.add_argument("--speeds", nargs="*", type=int, default=None)
    ap.add_argument("--modes", nargs="*", default=["temporal"])
    ap.add_argument("--out", default="delta_report.json")
    args = ap.parse_args()

    npys = sorted(glob.glob(os.path.join(args.rt, "v*_*.npy")))
    if not npys:
        raise FileNotFoundError(f"復元結果が無い: {args.rt}")

    nb = None
    results = []
    for p in npys:
        stem = os.path.basename(p)[:-4]
        v_str, mode = stem.split("_", 1)
        V = int(v_str[1:])
        if args.speeds is not None and V not in args.speeds:
            continue
        if mode not in args.modes:
            continue

        gt = np.load(os.path.join(args.gt, f"v{V}_points.npz"), allow_pickle=True)
        rt = np.load(p)
        if nb is None:
            nb = build_neighbors(gt["px"][0].astype(np.int64),
                                 gt["py"][0].astype(np.int64),
                                 int(gt["width"]), int(gt["height"]),
                                 MATCH_RADIUS_PX)
        results.append({"speed": V, "mode": mode, "delta": sweep(gt, rt, nb)})

    results.sort(key=lambda r: (r["mode"], r["speed"]))

    hdr = "".join(f"{'δ=' + f'{d:+d}':>9}" for d in DELTAS)
    for r in results:
        print("=" * 78)
        print(f"V={r['speed']}  {r['mode']}")
        print(f"{'':<10}{hdr}{'最良δ':>9}")
        print("-" * 78)

        row = "".join(f"{r['delta'][d]['overall']:>8.1%} " for d in DELTAS)
        best = max(DELTAS, key=lambda d: r["delta"][d]["overall"])
        print(f"{'全体':<10}{row}{best:>+9d}")

        for p in range(4):
            vals = [r["delta"][d]["by_pos"][p] for d in DELTAS]
            if all(v is None for v in vals):
                continue
            row = "".join(f"{v:>8.1%} " if v is not None else f"{'-':>9}"
                          for v in vals)
            bp = DELTAS[int(np.argmax([v if v is not None else -1
                                       for v in vals]))]
            print(f"{POS_NAMES[p]:<10}{row}{bp:>+9d}")
    print("=" * 78)

    print("\n仮説の予測: 束1番目→+2 / 2番目→+1 / 3番目→0 / 4番目→-1 の階段")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"deltas": DELTAS, "match_radius_px": MATCH_RADIUS_PX,
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
