"""
armc_palette.py  —  Arm C の局所最大分離パレット（stdlib のみ）

protocol §13.5
    Non-target帯 13-115 -> 内側 30-98    3段階: 30 / 64 / 98   （1段階 34）
    Target帯    140-242 -> 内側 157-225  3段階: 157 / 191 / 225
    市松配置（i+j+k が偶数）で14色 -> 全同値の両端2色を除外 -> 12色
    最小色間距離 = 34 x sqrt(2) = 48.1
"""
import itertools
import math

LEVELS = {
    "background": (30, 64, 98),
    "target": (157, 191, 225),
}
STEP = 34
MIN_DIST = STEP * math.sqrt(2.0)      # 48.0833...


def build(band):
    """12色を返す。順序は決定的（インデックスの辞書順）。"""
    lv = LEVELS[band]
    cols = []
    for i, j, k in itertools.product(range(3), repeat=3):
        if (i + j + k) % 2 != 0:
            continue                       # 市松配置
        if i == j == k and i in (0, 2):
            continue                       # 全同値の両端2色を除外
        cols.append((lv[i], lv[j], lv[k]))
    return cols


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def min_pair_distance(cols):
    return min(dist(a, b) for a, b in itertools.combinations(cols, 2))


PALETTE = {b: build(b) for b in LEVELS}
