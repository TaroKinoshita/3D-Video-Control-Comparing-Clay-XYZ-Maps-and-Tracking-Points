# -*- coding: utf-8 -*-
"""score_t_control_v2.py — T_control / T_coherence / 暗記判別（GPU 不要）。（W9, 2026-10-31）

v1 からの変更は 2 点だけ。tracker（色セグメンテーション）は v1 と同一で凍結のまま。

  1. GT 中心を C′ の ID パスから取る（data/points/gt_target_center_prime.json）。
     v1 は Scene C の重心を (K/9.00)·dZ だけ上にずらしていたが、C′ では見えている
     シルエット自体が違うため最大 31.9 px ずれていた（protocol rev.14 変更1）。
  2. T_coherence を「本数」で判定する：個数 ≠ 1 のフレームが ≤ 2 本なら PASS。
     tracker failure（検出 0）は分母にも分子にも入れず、別カウントで報告する。
     v1 は 本数 ÷ 採点フレーム数 ≤ 2/44 だったため、tracker failure が出るほど
     基準が厳しくなっていた（2/42 > 2/44 で FAIL）。2/44 の根拠は「幾何的に判定困難な
     2 フレーム分だけ予算を認める」であり本数の予算なので、本数で判定する（rev.14 変更2）。

暗記判別（v1 では memorization_probe が main から呼ばれていなかった）
  生成の検出中心と、(a) C′ GT、(b) Scene C の重心 の距離の median を両方出す。
  (a) < (b) なら control 追従、逆なら Scene C 軌道（学習動画）の再生寄り。
  dZ=0 の区間（index 0, 35–48）は (a)=(b) なので、差は index 1–34 で決まる。

使い方
  python score_t_control_v2.py --gen gen_armA.mkv --gt gt_target_center_prime.json [--json out.json]
  python score_t_control_v2.py --selftest --tv-prime tv_R0S0_prime.mkv --tv-orig tv_R0S0_original.mkv \
                               --gt gt_target_center_prime.json
"""
import argparse
import json
import sys

import numpy as np

MARKER = (658.76, 123.47)     # Marker 重心（固定物）。±30 px の成分は Target 候補から外す
GATE_CTRL_PX = 47.0           # §37.5
GATE_COH_FRAMES = 2           # rev.14 変更2（本数）
EXPECT_SCORED = 44            # §37.4：GT 可視フレーム数


# ---------------------------------------------------------------- tracker（v1 と同一）
def _label_numpy(mask):
    """4 連結ラベリング（scipy.ndimage.label の既定構造と同じ）。scipy が無い環境用。"""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        st = [(y0, x0)]
        while st:
            y, x = st.pop()
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    st.append((yy, xx))
    return lab, n


try:
    from scipy import ndimage as _nd

    def label(mask):
        return _nd.label(mask)
    LABELER = "scipy.ndimage.label"
except ImportError:  # pragma: no cover
    label = _label_numpy
    LABELER = "numpy fallback (4-connected)"


def orange_mask(img):
    R, G, B = (img[..., i].astype(int) for i in range(3))
    return (R > G) & (G >= B) & ((R - B) > 40) & (R > 80)


def track(img, labeler=None):
    """戻り値 (center or None, Target 候補成分の個数)。v1 の track() と同一の規則。"""
    labeler = labeler or label
    mask = orange_mask(img)
    if mask.sum() == 0:
        return None, 0
    lab, n = labeler(mask)
    best, bestsz, ncomp = None, 0, 0
    for c in range(1, n + 1):
        ys, xs = np.where(lab == c)
        cu, cv = xs.mean(), ys.mean()
        if abs(cu - MARKER[0]) < 30 and abs(cv - MARKER[1]) < 30:
            continue
        ncomp += 1
        if len(xs) > bestsz:
            bestsz, best = len(xs), (cu, cv)
    return best, ncomp


# ---------------------------------------------------------------- I/O
def load_video(path):
    import imageio.v2 as imageio
    rd = imageio.get_reader(path)
    arr = np.stack([np.asarray(f)[..., :3] for f in rd])
    rd.close()
    if arr.shape != (49, 480, 832, 3):
        raise SystemExit("[FATAL] 想定外の shape %s (%s)" % (arr.shape, path))
    return arr


def load_gt(path):
    d = json.load(open(path, encoding="utf-8"))
    gt, sc = {}, {}
    for r in d["frames"]:
        if r["npx"] > 0:
            gt[r["index"]] = (r["u"], r["v"])
            sc[r["index"]] = (r["u_sc"], r["v_sc"])
    if len(gt) != EXPECT_SCORED:
        raise SystemExit("[FATAL] GT の可視フレーム数 %d != %d" % (len(gt), EXPECT_SCORED))
    return gt, sc


# ---------------------------------------------------------------- 採点
def score(gen, gt, sc):
    rows = []
    for f in range(len(gen)):
        if f not in gt:
            continue  # 完全遮蔽 36–40：評価面の外
        det, nc = track(gen[f])
        if det is None:
            rows.append({"index": f, "status": "tracker_fail"})
            continue
        rows.append({"index": f, "status": "ok", "ncomp": nc,
                     "err": float(np.hypot(det[0] - gt[f][0], det[1] - gt[f][1])),
                     "err_sc": float(np.hypot(det[0] - sc[f][0], det[1] - sc[f][1])),
                     "det": [float(det[0]), float(det[1])]})
    ok = [r for r in rows if r["status"] == "ok"]
    e = np.array([r["err"] for r in ok])
    esc = np.array([r["err_sc"] for r in ok])
    bad = [r["index"] for r in ok if r["ncomp"] != 1]
    fail = [r["index"] for r in rows if r["status"] == "tracker_fail"]
    res = {
        "labeler": LABELER,
        "scored": len(ok), "tracker_fail": fail,
        "ctrl_median": float(np.median(e)), "ctrl_p95": float(np.percentile(e, 95)),
        "ctrl_max": float(e.max()),
        "coh_bad_frames": bad, "coh_bad_count": len(bad),
        "mem_median_vs_Cprime": float(np.median(e)), "mem_median_vs_SceneC": float(np.median(esc)),
        "pass_ctrl": bool(np.median(e) <= GATE_CTRL_PX),
        "pass_coh": bool(len(bad) <= GATE_COH_FRAMES),
        "follows_control": bool(np.median(e) < np.median(esc)),
        "rows": rows,
    }
    return res


def report(r, title):
    print("=== %s ===" % title)
    print("labeler        :", r["labeler"])
    print("scored frames  : %d / %d   tracker_fail: %s" % (r["scored"], EXPECT_SCORED, r["tracker_fail"]))
    print("T_control      : median %.2f px  p95 %.2f  max %.2f  -> %s (gate <= %.0f)" % (
        r["ctrl_median"], r["ctrl_p95"], r["ctrl_max"], "PASS" if r["pass_ctrl"] else "FAIL", GATE_CTRL_PX))
    print("T_coherence    : count!=1 frames %s = %d  -> %s (gate <= %d frames)" % (
        r["coh_bad_frames"], r["coh_bad_count"], "PASS" if r["pass_coh"] else "FAIL", GATE_COH_FRAMES))
    print("memorization   : median vs C' %.2f / vs Scene C %.2f  -> %s" % (
        r["mem_median_vs_Cprime"], r["mem_median_vs_SceneC"],
        "control 追従" if r["follows_control"] else "Scene C 軌道寄り（暗記の疑い）"))


# ---------------------------------------------------------------- selftest
def selftest(a):
    """陽性対照 2 本 + labeler の一致確認。"""
    gt, sc = load_gt(a.gt)
    ok = True
    # 1) 完璧な生成 = C′ の正解動画そのもの → Evaluator floor（C′ 面）
    rp = score(load_video(a.tv_prime), gt, sc)
    report(rp, "陽性対照1: tv_R0S0_prime（完璧な制御）")
    ok &= rp["pass_ctrl"] and rp["follows_control"] and rp["ctrl_p95"] < GATE_CTRL_PX
    # 2) 暗記 = Scene C の正解動画を出した場合
    ro = score(load_video(a.tv_orig), gt, sc)
    report(ro, "陽性対照2: tv_R0S0_original（暗記の模擬）")
    ok &= not ro["follows_control"]
    # 3) scipy があれば numpy fallback と結果が一致するか
    if LABELER.startswith("scipy"):
        v = load_video(a.tv_prime)
        same = all(track(v[f]) == track(v[f], _label_numpy) for f in range(0, 49, 6))
        print("labeler 一致（scipy vs numpy、9 フレーム）:", same)
        ok &= same
    print("\n[selftest]", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tv-prime")
    ap.add_argument("--tv-orig")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a) else 1)
    if not a.gen:
        ap.error("--gen が必要（または --selftest）")
    gt, sc = load_gt(a.gt)
    r = score(load_video(a.gen), gt, sc)
    report(r, a.gen)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
