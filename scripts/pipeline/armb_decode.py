"""
armb_decode.py  —  Arm B の codebook 逆写像と誤差指標（bpy 非依存）

armb_encode.encode() の逆。VAE 往復後の PNG から world 座標を復元し、
GT との誤差を出す。§18-4（正規化方式 Dataset-fixed vs Scene-bounded）の
判定に使う。

codebook（protocol §4。変更しない）
    Non-target / Background : 0.05 + 0.40q -> 8bit  13-115
    Target Entity           : 0.55 + 0.40q -> 8bit 140-242
    空画素                   : (0, 0, 0)
    実効境界                 : 空/Non-target = 6.5   Non-target/Target = 127.5
    予約帯                   : 124-131（出現してはいけない）

閉形式の性質（test_armb_decode.py で検証）
    8bit スパン        = 0.40 * 255 = 102
    1 LSB あたり Δq    = 1/102 = 0.00980392
    量子化誤差の上限   = 0.5 LSB → 軸ごと 0.5 * extent / 102
    帯の端の復号誤差   = ちょうど 0.25 LSB（丸めオフセット 12.75->13 由来）
"""
import numpy as np

SPAN = 0.40 * 255.0          # 102.0
BASE_BG = 0.05
BASE_TG = 0.55
BOUND_EMPTY = 6.5            # 空 / Non-target
BOUND_TARGET = 127.5         # Non-target / Target
RESERVED = (124, 131)
BAND_EMPTY, BAND_BG, BAND_TG = 0, 1, 3   # pass_index 規約に合わせる


def _lohi(norm_box):
    lo = np.array([norm_box["x"][0], norm_box["y"][0], norm_box["z"][0]],
                  dtype=np.float64)
    hi = np.array([norm_box["x"][1], norm_box["y"][1], norm_box["z"][1]],
                  dtype=np.float64)
    return lo, hi


def sensitivity(norm_box):
    """1 LSB が world 座標で何単位に相当するか（軸ごと）。"""
    lo, hi = _lohi(norm_box)
    return (hi - lo) / SPAN


def scale_box(norm_box, k, mode="center"):
    """
    ボックスを k 倍する。§18-4 の k スイープ用。
    mode="center" : 中心を保って各軸を k 倍
    mode="max"    : 下限を保って上へ伸ばす
    """
    out = {}
    for ax in "xyz":
        lo, hi = norm_box[ax]
        if mode == "center":
            c = 0.5 * (lo + hi)
            h = 0.5 * (hi - lo) * k
            out[ax] = (c - h, c + h)
        elif mode == "max":
            out[ax] = (lo, lo + (hi - lo) * k)
        else:
            raise ValueError("mode must be 'center' or 'max'")
    return out


def classify(rgb):
    """
    画素ごとに帯を判定する。

    返り値
        band       : (H, W) int8   0=空 / 1=Non-target / 3=Target
        disagree   : (H, W) bool   3チャンネルの判定が一致しなかった画素
        reserved   : (H, W) bool   予約帯 124-131 に入ったチャンネルがある画素

    クリーンなデータでは 3 チャンネルの判定は必ず一致する
    （Non-target は 13-115、Target は 140-242 で境界 127.5 を跨がない）。
    VAE 往復後に一致しなくなった画素数そのものが帯の破れの指標になる。
    """
    v = np.asarray(rgb).astype(np.float64)
    per = np.where(v <= BOUND_EMPTY, BAND_EMPTY,
                   np.where(v >= BOUND_TARGET, BAND_TG, BAND_BG)).astype(np.int8)
    # 多数決（3チャンネルなので、2つ以上一致する側を採る）
    band = np.zeros(per.shape[:2], dtype=np.int8)
    for b in (BAND_EMPTY, BAND_BG, BAND_TG):
        band = np.where((per == b).sum(axis=2) >= 2, np.int8(b), band)
    # 3すべて異なる（0/1/3 が1つずつ）画素は Non-target に寄せる
    tie = (per == BAND_EMPTY).sum(2) + (per == BAND_BG).sum(2) \
        + (per == BAND_TG).sum(2)
    band = np.where(((per == BAND_EMPTY).sum(2) == 1)
                    & ((per == BAND_BG).sum(2) == 1)
                    & ((per == BAND_TG).sum(2) == 1), np.int8(BAND_BG), band)
    assert tie.min() == 3, "per-channel 判定が3つ揃っていない"
    disagree = (per != band[:, :, None]).any(axis=2)
    iv = np.asarray(rgb).astype(np.int32)
    reserved = ((iv >= RESERVED[0]) & (iv <= RESERVED[1])).any(axis=2)
    return band, disagree, reserved


def decode(rgb, norm_box, band=None):
    """
    rgb      : (H, W, 3) uint8
    norm_box : encode に渡したものと同じ辞書
    band     : 省略時は classify() で推定。GT を渡せば帯判定を固定できる

    返り値
        pos_hat : (H, W, 3) float64  world 座標。空画素は NaN
        band    : (H, W) int8
        info    : dict（disagree / reserved の画素数）

    q はクリップしない。帯からはみ出した量をそのまま見せる。
    """
    lo, hi = _lohi(norm_box)
    v = np.asarray(rgb).astype(np.float64) / 255.0
    b_est, disagree, reserved = classify(rgb)
    if band is None:
        band = b_est
    band = np.asarray(band).astype(np.int8)

    base = np.where((band == BAND_TG)[:, :, None], BASE_TG, BASE_BG)
    q = (v - base) / 0.40
    pos = lo + q * (hi - lo)
    pos = np.where((band == BAND_EMPTY)[:, :, None], np.nan, pos)
    info = {
        "n_disagree": int(disagree.sum()),
        "n_reserved": int(reserved.sum()),
        "n_empty": int((band == BAND_EMPTY).sum()),
        "n_background": int((band == BAND_BG).sum()),
        "n_target": int((band == BAND_TG).sum()),
    }
    return pos, band, info


def px_per_unit(depth):
    """§26 のカメラ規約。depth = world_Y + 10。"""
    return 1155.5556 / np.asarray(depth, dtype=np.float64)


def errors(pos_hat, pos_gt, band_hat=None, band_gt=None, quantiles=(50, 95, 99)):
    """
    軸ごとの world 座標誤差と、画面上の px 換算誤差を返す。
    空画素（NaN）は除外する。

    平均だけを見ない（§18-6 の教訓：平均は保証ラインの破れを見落とす）。
    """
    a = np.asarray(pos_hat, dtype=np.float64)
    g = np.asarray(pos_gt, dtype=np.float64)
    m = np.isfinite(a).all(axis=2) & np.isfinite(g).all(axis=2)
    if band_gt is not None:
        m &= (np.asarray(band_gt) != BAND_EMPTY)
    d = np.abs(a - g)

    out = {"n_px": int(m.sum()), "axis": {}}
    for i, ax in enumerate(("X", "Y", "Z")):
        e = d[:, :, i][m]
        st = {"mae": float(e.mean()), "max": float(e.max())}
        for q in quantiles:
            st["p%d" % q] = float(np.percentile(e, q))
        out["axis"][ax] = st

    # 画面上の px 換算（GT の depth を使う）
    ppu = px_per_unit(g[:, :, 1] + 10.0)
    out["px"] = {}
    for i, ax in enumerate(("X", "Y", "Z")):
        e = (d[:, :, i] * ppu)[m]
        st = {"mae": float(e.mean()), "max": float(e.max())}
        for q in quantiles:
            st["p%d" % q] = float(np.percentile(e, q))
        out["px"][ax] = st

    if band_hat is not None and band_gt is not None:
        bh = np.asarray(band_hat)
        bg = np.asarray(band_gt)
        out["band"] = {
            "n_confused": int((bh != bg).sum()),
            "tg_as_bg": int(((bg == BAND_TG) & (bh == BAND_BG)).sum()),
            "bg_as_tg": int(((bg == BAND_BG) & (bh == BAND_TG)).sum()),
            "any_as_empty": int(((bg != BAND_EMPTY) & (bh == BAND_EMPTY)).sum()),
        }
    return out


def quant_worst(norm_box):
    """量子化のみの最悪誤差（rint は最近傍なので 0.5 LSB）。軸ごと。"""
    return sensitivity(norm_box) * 0.5


def sweep(pos_gt, ids_gt, norm_box, ks, mode="center", vae=None,
          quantiles=(50, 95, 99)):
    """
    §18-4 の判定本体。正規化ボックスを k 倍しながら誤差を出す。

    pos_gt   : (H, W, 3) float  GT world 座標（EXR 由来 or 合成）
    ids_gt   : (H, W)    float  pass_index（0 = 空、3 = Target）
    ks       : 倍率のリスト。1.0 が Scene-bounded 相当
    mode     : scale_box に渡す（"center" / "max"）
    vae      : None なら量子化のみ（GPU 不要）。
               callable(rgb_uint8) -> rgb_uint8 を渡すと VAE 往復後を評価する

    返り値 : [dict, ...]  各 dict は
        k / box / excess / quant_worst / err / info

    帯は GT で固定して decode する（座標誤差と band confusion を混ぜない。
    protocol §11 と同じ理由）。band confusion は info と err["band"] に出る。
    """
    import armb_encode

    band_gt = np.where(np.asarray(ids_gt) == 0.0, BAND_EMPTY,
                       np.where(np.asarray(ids_gt) == 3.0, BAND_TG,
                                BAND_BG)).astype(np.int8)
    rows = []
    for k in ks:
        box = scale_box(norm_box, k, mode=mode)
        rgb, excess = armb_encode.encode(pos_gt, ids_gt, box)
        target = rgb if vae is None else vae(rgb)
        pos_hat, band_est, info = decode(target, box, band=band_gt)
        err = errors(pos_hat, pos_gt, band_hat=band_est, band_gt=band_gt,
                     quantiles=quantiles)
        rows.append({"k": float(k), "box": box, "excess": float(excess),
                     "quant_worst": quant_worst(box).tolist(),
                     "err": err, "info": info})
    return rows


def format_sweep(rows, depth_ref=8.60):
    """照合用の表。コードブロックではなく表で出すため。"""
    out = []
    out.append("| k | X幅 | units/LSB (X) | 量子化最悪 X (units) | 同 px @d=%.2f "
               "| 実測 max X (units) | 実測 max X (px) | 予約帯 | 帯不一致 |"
               % depth_ref)
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        box = r["box"]
        ex = box["x"][1] - box["x"][0]
        upl = sensitivity(box)[0]
        qw = r["quant_worst"][0]
        qpx = qw * float(px_per_unit(depth_ref))
        ax = r["err"]["axis"]["X"]
        px = r["err"]["px"]["X"]
        nb = r["err"].get("band", {}).get("n_confused", 0)
        out.append("| %.2f | %.2f | %.5f | %.6f | %.2f | %.6f | %.3f | %d | %d |"
                   % (r["k"], ex, upl, qw, qpx, ax["max"], px["max"],
                      r["info"]["n_reserved"], nb))
    return "\n".join(out)
