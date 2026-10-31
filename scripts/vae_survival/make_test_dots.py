"""
VAE生存テスト用の点画像を作るスクリプト

使い方:
    python make_test_dots.py

出力先: このスクリプトと同じ場所に vae_test_dots フォルダができる
"""

import numpy as np
from PIL import Image
import os
import json

# ============================================================
# ここだけ触れば設定が変わる
# ============================================================

W, H = 832, 480          # 解像度（本番と同じにする）
LATENT_CELL = 8          # VAEが何ピクセルを1セルにまとめるか

RADII = [1, 2, 3]        # 点の半径（px）
DENSITIES = [0.9]        # 1セルあたりの点の数
                         # ↑ [0.3, 0.9, 1.5] にすれば9枚できる

TARGET_RATIO = 0.30      # Target点の割合（残りが背景・非対象）
SEED = 0                 # 同じ数字なら毎回同じ配置になる

# ============================================================
# Entity codebook（研究設計書の定義そのまま）
# ============================================================
# Target        : 0.55 + 0.40q  → 8bit  140-242（明るい）
# Non-target    : 0.05 + 0.40q  → 8bit   13-115（暗い）
# 空画素        : (0,0,0)                       （真っ黒）
# 予約帯        : 124-131 は絶対に使わない

RESERVED_BAND = np.arange(124, 132)

# ============================================================

CELLS = (W // LATENT_CELL) * (H // LATENT_CELL)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vae_test_dots")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


def to_codebook(q, is_target):
    """正規化座標 q(0-1) を Entity ごとの色域へ写像する"""
    if is_target:
        v = 0.55 + 0.40 * q
    else:
        v = 0.05 + 0.40 * q
    return np.clip(np.round(v * 255), 0, 255).astype(np.uint8)


def disc_offsets(radius):
    """アンチエイリアスなしの円マスク。整数ピクセルのみ"""
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    mask = (dx * dx + dy * dy) <= radius * radius
    oy, ox = np.nonzero(mask)
    return oy - radius, ox - radius


def make_image(density, radius):
    n = int(round(CELLS * density))

    # 点の中心（整数座標・画面端は避ける）
    cx = rng.integers(radius, W - radius, size=n)
    cy = rng.integers(radius, H - radius, size=n)

    # Target / Non-target を割り振る
    is_tgt = rng.random(n) < TARGET_RATIO
    q = rng.random((n, 3))

    cols = np.zeros((n, 3), dtype=np.uint8)
    cols[is_tgt] = to_codebook(q[is_tgt], True)
    cols[~is_tgt] = to_codebook(q[~is_tgt], False)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    oy, ox = disc_offsets(radius)
    for i in range(n):
        img[cy[i] + oy, cx[i] + ox] = cols[i]

    return img, n, int(is_tgt.sum())


report = []

for density in DENSITIES:
    for radius in RADII:
        img, n, n_tgt = make_image(density, radius)

        # 検査：予約帯が混入していないこと
        bad = int(np.isin(img, RESERVED_BAND).sum())
        assert bad == 0, f"予約帯124-131が{bad}個混入している"

        name = f"dots_d{density}_r{radius}.png"
        Image.fromarray(img).save(os.path.join(OUT_DIR, name))

        nonblack = int((img.sum(axis=2) > 0).sum())
        rec = {
            "file": name,
            "radius_px": radius,
            "density_per_latent_cell": density,
            "total_points": n,
            "target_points": n_tgt,
            "nontarget_points": n - n_tgt,
            "nonblack_pixels": nonblack,
            "coverage_pct": round(100 * nonblack / (W * H), 2),
            "reserved_band_count": bad,
        }
        report.append(rec)
        print(f"{name}  点{n}個  占有率{rec['coverage_pct']}%  予約帯{bad}個")

manifest = {
    "resolution": [W, H],
    "latent_cell_px": LATENT_CELL,
    "spatial_latent_cells": CELLS,
    "seed": SEED,
    "antialiasing": False,
    "codebook": {
        "target_8bit": [140, 242],
        "nontarget_8bit": [13, 115],
        "empty": [0, 0, 0],
        "reserved_band": [124, 131],
    },
    "images": report,
}

with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\n出力先: {OUT_DIR}")
print(f"{len(report)}枚できた")
