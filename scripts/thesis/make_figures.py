#!/usr/bin/env python3
"""make_figures.py — thesis figures from the frozen score files (no GPU). (W13)

Usage: python scripts/thesis/make_figures.py   (run from the repository root)
Writes docs/thesis/figures/fig{1..4}_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts/pipeline")
from analyze_seeds20 import boot_ci, ALPHA, N_BOOT, F_FLOOR, GATE  # noqa: E402

OUT = "docs/thesis/figures"
S20 = "runs/seeds20/gen/seeds20/arm{a}_step500_G{n}.json"
CUR = "runs/curve/gen/curve/arm{a}_{s}_G{n}.json"
RQ2 = "runs/rq2/gen/rq2_arm{a}_step500_G{n}.json"
COL = {"A": "#7a6f62", "B": "#2b6cb0", "C": "#c05621"}
NAME = {"A": "A: Clay Previs", "B": "B: World-XYZ", "C": "C: 3D Tracking"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def med(path):
    return json.load(open(path, encoding="utf-8"))["ctrl_median"]


def fig1():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rng = np.random.default_rng(0)
    for i, a in enumerate("ABC"):
        v = np.array([med(S20.format(a=a, n=n)) for n in range(1, 21)])
        ok = v <= GATE
        x = i + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(x[ok], v[ok], s=18, color=COL[a], alpha=0.8)
        ax.hlines(np.median(v[ok]), i - 0.25, i + 0.25, color="k", lw=1.5)
        for n in np.where(~ok)[0]:
            ax.annotate("G%d: %.1f px\n(Target duplicated)" % (n + 1, v[n]), xy=(i, 11.6), ha="center",
                        va="top", fontsize=8, color=COL[a],
                        arrowprops=dict(arrowstyle="->", color=COL[a]), xytext=(i + 0.05, 10.6))
    ax.set_xticks(range(3), [NAME[a] for a in "ABC"])
    ax.set_ylim(0, 12)
    ax.set_ylabel("Target centre error, median over frames (px)")
    ax.set_title("Control accuracy on held-out Scene C′ (20 seeds, step 500)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_accuracy_20seeds.png"), dpi=200)


def fig2():
    m = {(a, n): med(S20.format(a=a, n=n)) for a in "ABC" for n in range(1, 21)}
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    pairs = [("B", "C"), ("A", "C"), ("A", "B")]
    for i, (x, y) in enumerate(pairs):
        ss = [n for n in range(1, 21) if m[x, n] <= GATE and m[y, n] <= GATE]
        d = np.array([m[x, n] - m[y, n] for n in ss])
        lo, hi = boot_ci(d, ALPHA, N_BOOT)
        claim = (hi < 0 or lo > 0) and abs(d.mean()) > F_FLOOR
        ax.errorbar(d.mean(), i, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o",
                    color="k" if claim else "0.55", capsize=4)
        ax.text(0.45, i, "claim" if claim else "no claim", va="center", fontsize=9,
                color="k" if claim else "0.45")
    ax.axvspan(-F_FLOOR, F_FLOOR, color="0.9", zorder=0)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(3), ["%s − %s" % p for p in pairs])
    ax.set_xlim(-5, 1.6)
    ax.set_xlabel("Paired difference in error (px); grey band = evaluator floor ±1.81 px")
    ax.set_title("Follow-up: paired difference, 98.33% bootstrap CI", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_paired_differences.png"), dpi=200)


def fig3():
    steps = [0, 100, 200, 300, 400, 500]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for a in "ABC":
        ys = []
        for s in steps:
            p = (S20 if s == 500 else CUR).format(a=a, s="step%d" % s, n="{n}")
            v = [med(p.format(n=n)) for n in range(1, 5)]
            ax.scatter([s] * 4, v, s=8, color=COL[a], alpha=0.35)
            ys.append(np.median(v))
        ax.plot(steps, ys, "-o", color=COL[a], label=NAME[a], ms=4)
    ax.axhline(GATE, color="k", ls="--", lw=0.8); ax.text(505, GATE * 1.08, "gate 47 px", fontsize=8, ha="right")
    ax.set_yscale("log"); ax.set_ylim(1, 300)
    ax.set_xlabel("LoRA optimizer steps (≈ 7 s per step)"); ax.set_ylabel("Target centre error (px, log)")
    ax.set_title("Learning curve on Scene C′ (median of 4 seeds; dots = seeds)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_learning_curve.png"), dpi=200)


def fig4():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, a in enumerate("ABC"):
        v = [json.load(open(RQ2.format(a=a, n=n), encoding="utf-8"))["spill_mae"] for n in range(1, 5)]
        ax.scatter([i] * 4, v, color=COL[a], s=22)
        ax.hlines(np.mean(v), i - 0.25, i + 0.25, color="k", lw=1.5)
    ax.set_xticks(range(3), [NAME[a] for a in "ABC"])
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Spill MAE inside M_invariant (8-bit)")
    ax.set_title("RQ2: change leaked into invariant region (4 seeds; bar = mean)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_rq2_spill.png"), dpi=200)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3(); fig4()
    print("ok")
