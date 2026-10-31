"""
動画版VAE生存テスト用の点シーケンスを作る（ラウンド3：速度スイープ）

ラウンド1-2で確定した仕様（半径3px / 12色パレット / 密度0.3）を固定し、
点の速度だけを変える。移動はトーラス（ラップアラウンド）。

使い方:
    python make_dots_video.py

出力:
    vae_test_video/v{速度}/frame_0000.png ... frame_0048.png
    vae_test_video/v{速度}_points.npz      ← 採点用の正解データ
    vae_test_video/manifest.json
"""

import numpy as np
from PIL import Image
import os
import json

# ============================================================
# 固定パラメータ（ラウンド1-2で確定。触らない）
# ============================================================

W, H = 832, 480
LATENT_CELL = 8
N_FRAMES = 49                 # 4n+1 必須
RADIUS = 3
DENSITY = 0.3                 # 点 / 空間潜在セル
TARGET_RATIO = 0.30
SEED = 0

# パレット（vae_test_manifest.json から復元）
LEVELS_TARGET = [157, 191, 225]      # 帯140-242 を内側17寄せ、刻み34
LEVELS_NONTARGET = [30, 64, 98]      # 帯 13-115 を内側17寄せ、刻み34
PALETTE_MIN_DISTANCE = 48.1          # 34 * sqrt(2)
SEPARATION_PX = 24                   # この距離以内の同帯の点には別色を割り当てる

RESERVED_BAND = np.arange(124, 132)

# ============================================================
# 今回スイープする軸
# ============================================================

SPEEDS = [0, 5, 10, 17, 25, 34]      # px / frame

# 移動方向：2:1の斜め。軸平行でも完全な45度でもない
DIR = np.array([2.0, 1.0])
DIR = DIR / np.linalg.norm(DIR)      # (0.894427, 0.447214)

# 端の除外帯。最悪ケース(V=34)の足跡 3*34+3 = 105px を包含する固定値。
# 全速度で同じ値にすることで、採点対象の点数を速度間で揃える。
EDGE_BAND_PX = 110

# ============================================================

CELLS = (W // LATENT_CELL) * (H // LATENT_CELL)   # 6240
N_POINTS = int(round(CELLS * DENSITY))            # 1872
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vae_test_video")
os.makedirs(OUT_DIR, exist_ok=True)


def build_palette(levels):
    """3段階の市松配置から12色を作る"""
    cols = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i + j + k) % 2 != 0:
                    continue
                if i == j == k:          # 全同値の両端2色を除外
                    continue
                cols.append([levels[i], levels[j], levels[k]])
    return np.array(cols, dtype=np.uint8)


def torus_delta(a, b, size):
    """トーラス上の差分。画面をまたぐ距離を正しく扱う"""
    d = np.abs(a - b)
    return np.minimum(d, size - d)


def torus_dist(x0, y0, x1, y1):
    dx = torus_delta(x0, x1, W)
    dy = torus_delta(y0, y1, H)
    return np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)


def disc_offsets(radius):
    """アンチエイリアスなしの円マスク"""
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    m = (dx * dx + dy * dy) <= radius * radius
    oy, ox = np.nonzero(m)
    return oy - radius, ox - radius


def assign_palette(cx, cy, is_tgt, palette, rng):
    """
    近傍SEPARATION_PX以内の同帯の点と色が重ならないよう貪欲に割り当てる。
    距離はトーラス上で測る（ラップアラウンド後に隣接する点を見逃さないため）。
    """
    n = len(cx)
    cell = SEPARATION_PX
    grid = {}
    for i in range(n):
        key = (int(cx[i]) // cell, int(cy[i]) // cell)
        grid.setdefault(key, []).append(i)

    nx, ny = W // cell + 1, H // cell + 1
    assigned = np.full(n, -1, dtype=np.int64)
    order = rng.permutation(n)

    for i in order:
        gx, gy = int(cx[i]) // cell, int(cy[i]) // cell
        nb = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                nb.extend(grid.get(((gx + a) % nx, (gy + b) % ny), []))
        nb = np.array([j for j in nb if j != i and assigned[j] >= 0], dtype=np.int64)

        used = set()
        if len(nb):
            d = torus_dist(cx[nb], cy[nb], cx[i], cy[i])
            near = nb[(d <= SEPARATION_PX) & (is_tgt[nb] == is_tgt[i])]
            used = set(assigned[near].tolist())

        free = [c for c in range(len(palette)) if c not in used]
        assigned[i] = rng.choice(free) if free else rng.integers(len(palette))

    return assigned


def check_separation(cx, cy, cols, radius_px):
    """指定半径以内の最小色間距離。パレット割当が効いているかの検証"""
    n = len(cx)
    cell = max(radius_px, 1)
    grid = {}
    for i in range(n):
        grid.setdefault((int(cx[i]) // cell, int(cy[i]) // cell), []).append(i)
    nx, ny = W // cell + 1, H // cell + 1

    out = []
    c = cols.astype(np.float64)
    for i in range(n):
        gx, gy = int(cx[i]) // cell, int(cy[i]) // cell
        nb = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                nb.extend(grid.get(((gx + a) % nx, (gy + b) % ny), []))
        nb = np.array([j for j in nb if j != i], dtype=np.int64)
        if len(nb) == 0:
            continue
        d = torus_dist(cx[nb], cy[nb], cx[i], cy[i])
        nb = nb[d <= radius_px]
        if len(nb) == 0:
            continue
        out.append(np.linalg.norm(c[nb] - c[i], axis=1).min())
    return np.array(out) if out else np.array([np.inf])


# ============================================================
# 生成
# ============================================================

rng = np.random.default_rng(SEED)

pal_tgt = build_palette(LEVELS_TARGET)
pal_non = build_palette(LEVELS_NONTARGET)
assert len(pal_tgt) == 12 and len(pal_non) == 12, "パレットが12色になっていない"

# 点の初期配置とEntityは全速度で共通にする（速度以外の変数を消す）
cx0 = rng.integers(0, W, size=N_POINTS)
cy0 = rng.integers(0, H, size=N_POINTS)
is_tgt = rng.random(N_POINTS) < TARGET_RATIO

pal_idx = assign_palette(cx0, cy0, is_tgt, pal_tgt, rng)
colors = np.where(is_tgt[:, None], pal_tgt[pal_idx], pal_non[pal_idx]).astype(np.uint8)

sep20 = check_separation(cx0, cy0, colors, 20)
print(f"点 {N_POINTS}個 (Target {int(is_tgt.sum())})")
print(f"半径20px以内の最小色間距離: 中央値 {np.median(sep20):.1f} / "
      f"下位5% {np.percentile(sep20, 5):.1f} / 最小 {sep20.min():.1f}")
assert sep20.min() >= PALETTE_MIN_DISTANCE - 0.5, "近傍で色が重複している"

oy, ox = disc_offsets(RADIUS)
n_disc = len(oy)
report = []

for V in SPEEDS:
    d = os.path.join(OUT_DIR, f"v{V}")
    os.makedirs(d, exist_ok=True)

    t = np.arange(N_FRAMES)
    fx = cx0[None, :] + V * DIR[0] * t[:, None]
    fy = cy0[None, :] + V * DIR[1] * t[:, None]
    px = np.mod(np.round(fx), W).astype(np.int64)      # (49, n)
    py = np.mod(np.round(fy), H).astype(np.int64)

    bad_total = 0
    for f in range(N_FRAMES):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        rr = (py[f][:, None] + oy[None, :]) % H
        cc = (px[f][:, None] + ox[None, :]) % W
        img[rr.ravel(), cc.ravel()] = np.repeat(colors, n_disc, axis=0)
        bad_total += int(np.isin(img, RESERVED_BAND).sum())
        Image.fromarray(img).save(os.path.join(d, f"frame_{f:04d}.png"))
    assert bad_total == 0, f"V={V} で予約帯124-131が混入"

    np.savez_compressed(
        os.path.join(OUT_DIR, f"v{V}_points.npz"),
        px=px.astype(np.int32), py=py.astype(np.int32),
        color=colors, is_target=is_tgt, palette_index=pal_idx.astype(np.int32),
        speed=np.float32(V), direction=DIR.astype(np.float32),
        radius=np.int32(RADIUS), density=np.float32(DENSITY),
        width=np.int32(W), height=np.int32(H), n_frames=np.int32(N_FRAMES),
        latent_cell=np.int32(LATENT_CELL), edge_band_px=np.int32(EDGE_BAND_PX),
        target_band=np.array([140, 242], dtype=np.int32),
        nontarget_band=np.array([13, 115], dtype=np.int32),
        wraparound=np.bool_(True), color_mode=np.str_("palette12"),
    )

    inner = ((px >= EDGE_BAND_PX) & (px < W - EDGE_BAND_PX) &
             (py >= EDGE_BAND_PX) & (py < H - EDGE_BAND_PX))
    drift = V * (N_FRAMES - 1)
    rec = {
        "speed_px_per_frame": V,
        "footprint_px": round(3 * V + RADIUS, 1),
        "total_drift_px": round(drift, 1),
        "wraps_horizontally": bool(drift * DIR[0] >= W),
        "scored_points_per_frame_mean": round(float(inner.sum(axis=1).mean()), 1),
        "reserved_band_count": bad_total,
    }
    report.append(rec)
    print(f"  V={V:>2}  足跡{rec['footprint_px']:>5}px  "
          f"採点対象 平均{rec['scored_points_per_frame_mean']}点")

manifest = {
    "resolution": [W, H], "n_frames": N_FRAMES, "seed": SEED,
    "radius_px": RADIUS, "density_per_latent_cell": DENSITY,
    "color_mode": "palette12", "palette_min_distance": PALETTE_MIN_DISTANCE,
    "separation_px": SEPARATION_PX, "antialiasing": False,
    "motion": {"model": "wraparound", "direction_2to1": DIR.tolist()},
    "edge_band_px": EDGE_BAND_PX,
    "measured_separation_at_20px": {
        "median": round(float(np.median(sep20)), 1),
        "p5": round(float(np.percentile(sep20, 5)), 1),
        "min": round(float(sep20.min()), 1),
    },
    "speeds": report,
}
with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\n出力: {OUT_DIR}")
print(f"{len(SPEEDS)}速度 × {N_FRAMES}フレーム = {len(SPEEDS)*N_FRAMES}枚")
