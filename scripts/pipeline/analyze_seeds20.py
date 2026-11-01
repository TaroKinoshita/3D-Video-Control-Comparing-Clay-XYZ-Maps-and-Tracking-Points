#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_seeds20.py — 追加実験（protocol rev.17）の判定（GPU 不要、numpy と標準ライブラリのみ）。（W12）

入力：score_t_control_v2.py の JSON（arm{X}_step500_G{n}.json、X = A/B/C、n = 1..N）
判定式は rev.17 で凍結。ここを変えるときは protocol を先に変える。

1. 失敗（大失敗）：T_control の median > 47 px（§37.5 の関門を落ちた生成）
2. 精度：同じ seed で 2 arm とも失敗していない seed だけを使い、
         d_s = median_X,s − median_Y,s の平均と、Bonferroni 補正した 98.33% のブートストラップ区間
         （10,000 回、percentile 法、乱数 seed 0）
         主張する条件：使える seed ≥ 10 かつ 区間が 0 を含まない かつ |平均 d| > max(F_det, F_micro, F_eval) = 1.81 px
3. 安定性：失敗の本数を Fisher の正確検定（両側）で比べる。p < 0.05 / 3 なら主張する
"""
import argparse
import json
import math
import os

import numpy as np

GATE = 47.0
F_FLOOR = max(0.0, 0.10, 1.81)        # F_det（§42.5）, F_micro（W10）, F_eval（§41.4）
ALPHA = 0.05 / 3                       # Bonferroni（3 ペア）
N_BOOT = 10000
MIN_SEEDS = 10                         # 使える seed がこれ未満なら判定不能（少数だと区間が狭く出すぎる）
PAIRS = [("B", "C"), ("A", "B"), ("A", "C")]   # Primary が先（§2）


def fisher_two_sided(a, b, c, d):
    """2x2 表 [[a,b],[c,d]] の Fisher 正確検定（両側、確率が観測以下の表の和）。"""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p_obs = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * (1 + 1e-9)))


def boot_ci(d, alpha, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pattern", default="arm{arm}_step500_G{n}.json")
    a = ap.parse_args()

    med, coh = {}, {}
    for arm in "ABC":
        for n in range(1, a.n + 1):
            r = json.load(open(os.path.join(a.dir, a.pattern.format(arm=arm, n=n)), encoding="utf-8"))
            med[arm, n] = r["ctrl_median"]
            coh[arm, n] = r["coh_bad_count"]

    print("## 1. 各 arm（N = %d seed）\n" % a.n)
    print("| arm | 失敗（median > 47） | 失敗した seed | 失敗以外の median の中央値 | T_coherence 不合格（> 2 本） |")
    print("|---|---|---|---|---|")
    fail = {}
    for arm in "ABC":
        f = [n for n in range(1, a.n + 1) if med[arm, n] > GATE]
        fail[arm] = f
        ok = [med[arm, n] for n in range(1, a.n + 1) if n not in f]
        nc = sum(coh[arm, n] > 2 for n in range(1, a.n + 1))
        print("| %s | %d / %d | %s | %.2f | %d / %d |" % (arm, len(f), a.n, f or "—", np.median(ok), nc, a.n))

    print("\n## 2. 精度（同じ seed どうし、両方とも失敗していない seed のみ）\n")
    print("床 max(F_det, F_micro, F_eval) = %.2f px、区間は %.2f%%（Bonferroni）\n" % (F_FLOOR, 100 * (1 - ALPHA)))
    print("| ペア | 使った seed 数 | 平均 d (X−Y) | 区間 | 0 を含まない | \\|d\\| > 床 | 判定 |")
    print("|---|---|---|---|---|---|---|")
    for x, y in PAIRS:
        ss = [n for n in range(1, a.n + 1) if n not in fail[x] and n not in fail[y]]
        d = np.array([med[x, n] - med[y, n] for n in ss])
        lo, hi = boot_ci(d, ALPHA, N_BOOT)
        ex0 = lo > 0 or hi < 0
        big = abs(d.mean()) > F_FLOOR
        verdict = "判定不能（seed < %d）" % MIN_SEEDS if len(ss) < MIN_SEEDS else (
            "主張可" if ex0 and big else "主張不可")
        print("| %s vs %s | %d | %+.2f | [%+.2f, %+.2f] | %s | %s | **%s** |" % (
            x, y, len(ss), d.mean(), lo, hi, ex0, big, verdict))

    print("\n## 3. 安定性（失敗の本数、Fisher 正確検定・両側）\n")
    print("| ペア | 失敗 X | 失敗 Y | p | 判定（p < %.4f） |" % ALPHA)
    print("|---|---|---|---|---|")
    for x, y in PAIRS:
        fx, fy = len(fail[x]), len(fail[y])
        p = fisher_two_sided(fx, a.n - fx, fy, a.n - fy)
        print("| %s vs %s | %d / %d | %d / %d | %.4f | **%s** |" % (
            x, y, fx, a.n, fy, a.n, p, "主張可" if p < ALPHA else "主張不可"))


if __name__ == "__main__":
    main()
