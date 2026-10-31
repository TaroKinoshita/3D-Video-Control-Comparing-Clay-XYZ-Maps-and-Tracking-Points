"""
ラウンド4：半径 × 速度 のグリッドを作る

ラウンド3で「動くと点の色が薄まる」ことが分かった。点を大きくすれば
濃度が上がって耐えるはず、という仮説を検証するためのジェネレータ。

半径を上げると1点あたりの画素数が増えるので、占有率を揃えるために
点の数を減らす。点の数が減ると近傍照合が簡単になるので、比較は
各設定の V=0 を基準にした「落差」で行う（絶対値では比べない）。

使い方:
    python make_dots_round4.py --radius 3 --out round4_r3
    python make_dots_round4.py --radius 6 --out round4_r6
    python make_dots_round4.py --radius 9 --out round4_r9
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

W, H = 832, 480
LATENT_CELL = 8
N_FRAMES = 49
SEED = 0
TARGET_RATIO = 0.30

LEVELS_TARGET = [157, 191, 225]
LEVELS_NONTARGET = [30, 64, 98]
PALETTE_MIN_DISTANCE = 48.1
SEPARATION_PX = 24
RESERVED_BAND = np.arange(124, 132)

SPEEDS = [0, 5, 10, 17]
DIR = np.array([2.0, 1.0]) / np.sqrt(5.0)
EDGE_BAND_PX = 110

# ラウンド3の基準：半径3・密度0.3 のときのインク総量
REF_RADIUS, REF_DENSITY = 3, 0.3


def build_palette(levels):
    return np.array([[levels[i], levels[j], levels[k]]
                     for i in range(3) for j in range(3) for k in range(3)
                     if (i + j + k) % 2 == 0 and not (i == j == k)],
                    dtype=np.uint8)


def torus_dist(x0, y0, x1, y1):
    dx = np.abs(x0 - x1); dx = np.minimum(dx, W - dx)
    dy = np.abs(y0 - y1); dy = np.minimum(dy, H - dy)
    return np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)


def disc_offsets(radius):
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    m = (dx * dx + dy * dy) <= radius * radius
    oy, ox = np.nonzero(m)
    return oy - radius, ox - radius


def assign_palette(cx, cy, is_tgt, n_colors, rng):
    n = len(cx)
    cell = SEPARATION_PX
    nx, ny = W // cell + 1, H // cell + 1
    grid = {}
    for i in range(n):
        grid.setdefault((int(cx[i]) // cell, int(cy[i]) // cell), []).append(i)

    assigned = np.full(n, -1, dtype=np.int64)
    for i in rng.permutation(n):
        gx, gy = int(cx[i]) // cell, int(cy[i]) // cell
        nb = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                nb.extend(grid.get(((gx + a) % nx, (gy + b) % ny), []))
        nb = np.array([j for j in nb if j != i and assigned[j] >= 0],
                      dtype=np.int64)
        used = set()
        if len(nb):
            d = torus_dist(cx[nb], cy[nb], cx[i], cy[i])
            near = nb[(d <= SEPARATION_PX) & (is_tgt[nb] == is_tgt[i])]
            used = set(assigned[near].tolist())
        free = [c for c in range(n_colors) if c not in used]
        assigned[i] = rng.choice(free) if free else rng.integers(n_colors)
    return assigned


def min_separation(cx, cy, cols, radius_px):
    n = len(cx)
    cell = max(radius_px, 1)
    nx, ny = W // cell + 1, H // cell + 1
    grid = {}
    for i in range(n):
        grid.setdefault((int(cx[i]) // cell, int(cy[i]) // cell), []).append(i)
    c = cols.astype(np.float64)
    out = []
    for i in range(n):
        gx, gy = int(cx[i]) // cell, int(cy[i]) // cell
        nb = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                nb.extend(grid.get(((gx + a) % nx, (gy + b) % ny), []))
        nb = np.array([j for j in nb if j != i], dtype=np.int64)
        if len(nb) == 0:
            continue
        nb = nb[torus_dist(cx[nb], cy[nb], cx[i], cy[i]) <= radius_px]
        if len(nb) == 0:
            continue
        out.append(np.linalg.norm(c[nb] - c[i], axis=1).min())
    return np.array(out) if len(out) else np.array([np.inf])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=None,
                    help="点の数。既定はインク総量を半径3・密度0.3に揃える")
    args = ap.parse_args()

    radius = args.radius
    oy, ox = disc_offsets(radius)
    area = len(oy)
    ref_n = int(round((W // LATENT_CELL) * (H // LATENT_CELL) * REF_DENSITY))
    ref_area = len(disc_offsets(REF_RADIUS)[0])
    n_points = args.points or int(round(ref_n * ref_area / area))

    out_dir = os.path.join(os.getcwd(), args.out)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    pal_t, pal_n = build_palette(LEVELS_TARGET), build_palette(LEVELS_NONTARGET)

    cx0 = rng.integers(0, W, size=n_points)
    cy0 = rng.integers(0, H, size=n_points)
    is_tgt = rng.random(n_points) < TARGET_RATIO
    pi = assign_palette(cx0, cy0, is_tgt, len(pal_t), rng)
    colors = np.where(is_tgt[:, None], pal_t[pi], pal_n[pi]).astype(np.uint8)

    sep = min_separation(cx0, cy0, colors, 20)
    print(f"半径 {radius}px  1点{area}画素  点{n_points}個 "
          f"(Target {int(is_tgt.sum())})")
    print(f"  想定インク量 {n_points*area} 画素 "
          f"(半径3基準 {ref_n*ref_area})")
    print(f"  半径20px以内の最小色間距離 {sep.min():.1f}")
    assert sep.min() >= PALETTE_MIN_DISTANCE - 0.5, "近傍で色が重複している"

    report = []
    for V in SPEEDS:
        d = os.path.join(out_dir, f"v{V}")
        os.makedirs(d, exist_ok=True)
        t = np.arange(N_FRAMES)
        px = np.mod(np.round(cx0[None, :] + V * DIR[0] * t[:, None]), W).astype(np.int64)
        py = np.mod(np.round(cy0[None, :] + V * DIR[1] * t[:, None]), H).astype(np.int64)

        cov = 0
        for f in range(N_FRAMES):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            rr = (py[f][:, None] + oy[None, :]) % H
            cc = (px[f][:, None] + ox[None, :]) % W
            img[rr.ravel(), cc.ravel()] = np.repeat(colors, area, axis=0)
            assert not np.isin(img, RESERVED_BAND).any(), "予約帯が混入"
            if f == 0:
                cov = 100 * (img.sum(axis=2) > 0).mean()
            Image.fromarray(img).save(os.path.join(d, f"frame_{f:04d}.png"))

        np.savez_compressed(
            os.path.join(out_dir, f"v{V}_points.npz"),
            px=px.astype(np.int32), py=py.astype(np.int32),
            color=colors, is_target=is_tgt, palette_index=pi.astype(np.int32),
            speed=np.float32(V), direction=DIR.astype(np.float32),
            radius=np.int32(radius), density=np.float32(n_points / 6240),
            width=np.int32(W), height=np.int32(H), n_frames=np.int32(N_FRAMES),
            latent_cell=np.int32(LATENT_CELL), edge_band_px=np.int32(EDGE_BAND_PX),
            target_band=np.array([140, 242], dtype=np.int32),
            nontarget_band=np.array([13, 115], dtype=np.int32),
            wraparound=np.bool_(True), color_mode=np.str_("palette12"),
        )

        inner = ((px >= EDGE_BAND_PX) & (px < W - EDGE_BAND_PX) &
                 (py >= EDGE_BAND_PX) & (py < H - EDGE_BAND_PX))
        conc = np.pi * radius / (np.pi * radius + 6 * V) if V else 1.0
        report.append({
            "speed_px_per_frame": V, "radius_px": radius,
            "n_points": n_points, "coverage_pct": round(float(cov), 2),
            "footprint_px": 3 * V + radius,
            "predicted_concentration": round(float(conc), 3),
            "scored_points_per_frame_mean": round(float(inner.sum(axis=1).mean()), 1),
        })
        print(f"  V={V:>2}  占有{cov:>5.1f}%  足跡{3*V+radius:>4}px  "
              f"濃さ{conc:>5.1%}  採点対象 平均{report[-1]['scored_points_per_frame_mean']}点")

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"resolution": [W, H], "n_frames": N_FRAMES, "seed": SEED,
                   "radius_px": radius, "n_points": n_points,
                   "disc_area_px": area, "color_mode": "palette12",
                   "motion": {"model": "wraparound", "direction_2to1": DIR.tolist()},
                   "edge_band_px": EDGE_BAND_PX,
                   "min_separation_at_20px": round(float(sep.min()), 1),
                   "speeds": report}, f, indent=2, ensure_ascii=False)
    print(f"出力: {out_dir}")


if __name__ == "__main__":
    main()
