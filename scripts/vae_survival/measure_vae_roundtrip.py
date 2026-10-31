"""
VAE往復の採点スクリプト（修正版 v2）

v1からの変更点:
  1. identity復元率を「局所」に変更（主指標）
     パレット方式は色を共有するので、全画面照合では必ず失敗する。
     点は動くので、フレーム間の移動距離以内で一意なら対応が取れる。
  2. 全画面照合は「色が全点で一意な方式」のみ有効として併記
  3. 複数ペアを1回で処理する（JSONを上書きしない）
  4. 判定条件を明示（色誤差 < 局所色間距離 ÷ 2）

使い方:
    python measure_vae_roundtrip.py 元1.png 往復1.png [元2.png 往復2.png ...]
    python measure_vae_roundtrip.py --pairs pairs.json

必要ファイル:
    元画像.png と同じ場所に 元画像_points.npz があること
"""

import numpy as np
from PIL import Image
from collections import defaultdict
import os
import json
import argparse


# 局所照合の半径。16fpsで画面を約3秒で横断する物体は1フレーム約17px動く
LOCAL_RADII_PX = [10, 20, 40]
PRIMARY_RADIUS_PX = 20


def load_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.float64)


def band_distance(m, lo, hi):
    return np.maximum(lo - m, 0.0) + np.maximum(m - hi, 0.0)


def classify_entity(mean_val, nt_band, t_band):
    """0=空画素, 1=Non-target, 2=Target。最も近い帯を選ぶ"""
    d = np.stack([
        np.abs(mean_val),
        band_distance(mean_val, nt_band[0], nt_band[1]),
        band_distance(mean_val, t_band[0], t_band[1]),
    ], axis=1)
    return np.argmin(d, axis=1)


def spatial_grid(cx, cy, R):
    gx = (cx // R).astype(int)
    gy = (cy // R).astype(int)
    g = defaultdict(list)
    for i in range(len(cx)):
        g[(gx[i], gy[i])].append(i)
    return g, gx, gy


def neighbors_within(i, cx, cy, g, gx, gy, R):
    nb = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            nb.extend(g.get((gx[i] + a, gy[i] + b), []))
    nb = np.array(nb, dtype=np.int64)
    d2 = (cx[nb] - cx[i]) ** 2 + (cy[nb] - cy[i]) ** 2
    return nb[d2 <= R * R]


def local_identity(cx, cy, col_gt, samp_rt, alive, R):
    """
    往復後の色を、近傍R以内の点の「元の色」と照合する。
    最も近いのが自分自身なら復元成功。
    """
    g, gx, gy = spatial_grid(cx, cy, R)
    ok = np.zeros(len(cx), dtype=bool)
    tested = np.zeros(len(cx), dtype=bool)
    cand_n = np.zeros(len(cx), dtype=np.int64)
    for i in range(len(cx)):
        if not alive[i]:
            continue
        nb = neighbors_within(i, cx, cy, g, gx, gy, R)
        cand_n[i] = len(nb)
        if len(nb) == 0:
            continue
        d = np.linalg.norm(col_gt[nb] - samp_rt[i], axis=1)
        ok[i] = nb[np.argmin(d)] == i
        tested[i] = True
    return ok, tested, cand_n


def local_separation_source(cx, cy, col_gt, R):
    """元データ側の局所色間距離。この指標の床にあたる"""
    g, gx, gy = spatial_grid(cx, cy, R)
    out = []
    for i in range(len(cx)):
        nb = neighbors_within(i, cx, cy, g, gx, gy, R)
        nb = nb[nb != i]
        if len(nb) == 0:
            continue
        out.append(np.linalg.norm(col_gt[nb] - col_gt[i], axis=1).min())
    return np.array(out) if out else np.array([np.inf])


def global_identity(col_gt, samp_rt, alive, chunk=512):
    """全画面照合。色が全点で一意な方式でのみ意味を持つ"""
    q = samp_rt[alive]
    idx = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):
        e = min(s + chunk, len(q))
        d = np.linalg.norm(q[s:e, None, :] - col_gt[None, :, :], axis=2)
        idx[s:e] = np.argmin(d, axis=1)
    return (idx == np.nonzero(alive)[0])


def measure(orig_path, rt_path):
    npz_path = os.path.splitext(orig_path)[0] + "_points.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"正解データが無い: {npz_path}")

    gt = np.load(npz_path, allow_pickle=True)
    cx, cy = gt["cx"].astype(np.int64), gt["cy"].astype(np.int64)
    col_gt = gt["color"].astype(np.float64)
    is_tgt = gt["is_target"]
    radius = int(gt["radius"])
    density = float(gt["density"])
    cell_px = int(gt["latent_cell"])
    W, H = int(gt["width"]), int(gt["height"])
    nt_band = gt["nontarget_band"].astype(np.float64)
    t_band = gt["target_band"].astype(np.float64)
    mode = str(gt["color_mode"]) if "color_mode" in gt else "unknown"
    unique_global = bool(gt["unique_global_colors"]) if "unique_global_colors" in gt else True

    orig = load_rgb(orig_path)
    rt = load_rgb(rt_path)
    assert orig.shape == (H, W, 3), f"元画像のサイズ不一致: {orig.shape}"
    assert rt.shape == orig.shape, (
        f"往復後のサイズ不一致: {rt.shape}。リサイズノードが挟まっていないか確認")

    n = len(cx)
    samp_rt = rt[cy, cx, :]
    samp_or = orig[cy, cx, :]
    mean_rt = samp_rt.mean(axis=1)
    mean_or = samp_or.mean(axis=1)

    # --- Entity 三値分類 ---
    cls_true = np.where(is_tgt, 2, 1)
    cls_rt = classify_entity(mean_rt, nt_band, t_band)
    cls_or = classify_entity(mean_or, nt_band, t_band)

    alive = cls_rt != 0
    valid = cls_or == cls_true          # 描画の重なりで潰れた点を母数から除く
    entity_ok = cls_rt == cls_true

    def safe(m):
        return float(entity_ok[m].mean()) if m.any() else float("nan")

    conf = np.zeros((3, 3), dtype=np.int64)
    for tr in (1, 2):
        for pr in (0, 1, 2):
            conf[tr, pr] = int(((cls_true == tr) & (cls_rt == pr) & valid).sum())

    # --- 色誤差 ---
    err = np.abs(samp_rt - col_gt)
    mae = float(err[alive].mean()) if alive.any() else float("nan")
    bias = float((samp_rt - col_gt)[alive].mean()) if alive.any() else float("nan")

    # --- 局所 identity（主指標）---
    loc = {}
    for R in LOCAL_RADII_PX:
        ok, tested, cand = local_identity(cx, cy, col_gt, samp_rt, alive, R)
        sep = local_separation_source(cx, cy, col_gt, R)
        m_t, m_n = tested & is_tgt, tested & (~is_tgt)
        loc[R] = {
            "identity_rate": round(float(ok[tested].mean()), 4) if tested.any() else None,
            "identity_rate_target": round(float(ok[m_t].mean()), 4) if m_t.any() else None,
            "identity_rate_nontarget": round(float(ok[m_n].mean()), 4) if m_n.any() else None,
            "candidates_mean": round(float(cand[tested].mean()), 1) if tested.any() else None,
            "source_separation_median": round(float(np.median(sep)), 1),
            "source_separation_p5": round(float(np.percentile(sep, 5)), 1),
            "margin_ratio": round(float(np.percentile(sep, 5) / (2 * mae)), 2) if mae > 0 else None,
        }

    # --- 全画面 identity（一意方式のみ有効）---
    g_ok = global_identity(col_gt, samp_rt, alive)
    g_rate = round(float(g_ok.mean()), 4) if len(g_ok) else None

    # --- 潜在セル ---
    ncols = W // cell_px
    cell = (cy // cell_px) * ncols + (cx // cell_px)
    uniq, counts = np.unique(cell, return_counts=True)
    shared = int(counts[counts > 1].sum())

    return {
        "file_orig": os.path.basename(orig_path),
        "file_roundtrip": os.path.basename(rt_path),
        "color_mode": mode,
        "radius_px": radius,
        "density_per_latent_cell": round(density, 2),
        "total_points": n,
        "points_valid_in_source": int(valid.sum()),

        "survival_rate": round(float(alive.mean()), 4),
        "survival_target": round(float(alive[is_tgt].mean()), 4),
        "survival_nontarget": round(float(alive[~is_tgt].mean()), 4),

        "entity_accuracy": round(safe(valid), 4),
        "entity_accuracy_target": round(safe(valid & is_tgt), 4),
        "entity_accuracy_nontarget": round(safe(valid & ~is_tgt), 4),
        "confusion_matrix_rows_true_cols_pred": conf.tolist(),

        "color_mae_8bit": round(mae, 2),
        "color_bias_8bit": round(bias, 2),

        "local_identity": {str(k): v for k, v in loc.items()},
        "primary_radius_px": PRIMARY_RADIUS_PX,
        "primary_identity_rate": loc[PRIMARY_RADIUS_PX]["identity_rate"],

        "global_identity_rate": g_rate,
        "global_identity_valid": unique_global,

        "latent_coverage_pct": round(100 * len(uniq) / (ncols * (H // cell_px)), 2),
        "collision_rate_pct": round(100 * shared / n, 2),
    }


def print_report(r):
    print("=" * 68)
    print(f"{r['file_orig']}  →  {r['file_roundtrip']}")
    print(f"色の付け方 {r['color_mode']} / 半径 {r['radius_px']}px / "
          f"密度 {r['density_per_latent_cell']} / 点 {r['total_points']}")
    print("-" * 68)
    print(f"  点の生存率      全体 {r['survival_rate']:.1%}  "
          f"Target {r['survival_target']:.1%}  Non-target {r['survival_nontarget']:.1%}")
    print(f"  Entity分類精度  全体 {r['entity_accuracy']:.1%}  "
          f"Target {r['entity_accuracy_target']:.1%}  "
          f"Non-target {r['entity_accuracy_nontarget']:.1%}")
    print(f"  色誤差          平均 {r['color_mae_8bit']}  "
          f"偏り {r['color_bias_8bit']:+}  (8bit)")
    print("-" * 68)
    print("  identity復元率（近傍照合。これが主指標）")
    print("      半径   復元率   Target  Non-tgt  候補数   元の色間距離(下位5%)  余裕")
    for R in LOCAL_RADII_PX:
        d = r["local_identity"][str(R)]
        star = " ←主" if R == r["primary_radius_px"] else ""
        def pc(x):
            return f"{x:6.1%}" if x is not None else "     -"
        print(f"     {R:>3}px  {pc(d['identity_rate'])}  {pc(d['identity_rate_target'])}  "
              f"{pc(d['identity_rate_nontarget'])}  {d['candidates_mean']:>6}  "
              f"{d['source_separation_p5']:>16}  {d['margin_ratio']:>5}{star}")
    print()
    if r["global_identity_valid"]:
        print(f"  全画面照合      {r['global_identity_rate']:.1%}")
    else:
        print(f"  全画面照合      {r['global_identity_rate']:.1%}  "
              f"※色を共有する方式なので参考値。判定に使わない")
    print(f"  潜在セル        占有 {r['latent_coverage_pct']}%  "
          f"衝突 {r['collision_rate_pct']}%")

    conf = np.array(r["confusion_matrix_rows_true_cols_pred"])
    names = ["空", "Non-tgt", "Target"]
    print("-" * 68)
    print("  混同行列（行=正解 / 列=推定）")
    print("            " + "".join(f"{x:>10}" for x in names))
    for i in (1, 2):
        print(f"    {names[i]:>8}" + "".join(f"{conf[i, j]:>10}" for j in range(3)))

    d = r["local_identity"][str(r["primary_radius_px"])]
    verdict = "成立" if (d["identity_rate"] or 0) >= 0.90 else "不足"
    print("-" * 68)
    print(f"  判定（半径{r['primary_radius_px']}px, 90%以上で成立）: "
          f"{d['identity_rate']:.1%} → {verdict}")
    print(f"  余裕 = 元の色間距離(下位5%) ÷ (2×色誤差) = {d['margin_ratio']}  "
          f"（1.0以上が必要）")
    print("=" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="元1.png 往復1.png 元2.png 往復2.png ...")
    ap.add_argument("--pairs", help="ペアを書いたJSON")
    ap.add_argument("--out", default="vae_roundtrip_report.json")
    args = ap.parse_args()

    pairs = []
    if args.pairs:
        with open(args.pairs, encoding="utf-8") as f:
            pairs = [(p["orig"], p["roundtrip"]) for p in json.load(f)]
    elif args.files:
        if len(args.files) % 2 != 0:
            ap.error("ファイルは 元/往復 のペアで偶数個指定する")
        pairs = list(zip(args.files[0::2], args.files[1::2]))
    else:
        ap.error("ファイルか --pairs を指定する")

    results = []
    for o, rt in pairs:
        r = measure(o, rt)
        print_report(r)
        results.append(r)

    if len(results) > 1:
        print("\n" + "=" * 68)
        print("まとめ（半径20px）")
        print("-" * 68)
        print(f"  {'ファイル':<28}{'方式':<10}{'復元率':>8}{'Entity':>8}{'色誤差':>8}{'余裕':>7}")
        for r in results:
            d = r["local_identity"][str(PRIMARY_RADIUS_PX)]
            print(f"  {r['file_orig']:<28}{r['color_mode']:<10}"
                  f"{d['identity_rate']:>8.1%}{r['entity_accuracy']:>8.1%}"
                  f"{r['color_mae_8bit']:>8}{d['margin_ratio']:>7}")
        print("=" * 68)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {args.out}  （{len(results)}件）")


if __name__ == "__main__":
    main()
