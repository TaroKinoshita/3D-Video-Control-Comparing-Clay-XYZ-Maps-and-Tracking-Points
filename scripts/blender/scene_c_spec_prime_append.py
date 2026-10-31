# ================================================================
# 以下を scene_c_spec.py の末尾に追記する（W7 / Scene C' = variant "prime"）
# ================================================================

import math as _math

# ---------------------------------------------------------------- Scene C' (prime)
# Scene C' は Scene C Original の X 軌道をそのまま継承し、Z に raised-cosine の
# 縦揺れを足したもの。目的は「LoRA が control を読んでいるか」を測る C' 面を作ること。
#
#   X(f) = -0.90 + 0.0375*f            （original と bit 単位で同一。触らない）
#   Y(f) = 0                           （面積不変 → 制約2 が自動で満たされる）
#   Z(f) = DZ_A * (1 - cos(2*pi*f/DZ_PERIOD)) / 2   for f <= DZ_CLAMP_INDEX
#        = 0                                        for f >  DZ_CLAMP_INDEX
#
# ここで Z(f) は **obj.location.z に入れる差分** であり、Target の幾何中心 Z
# （TARGET_Z の中点 = -0.80）は含まない。中点はメッシュ頂点に焼かれており
# obj.location.z は Scene C では 0.0 のままである。
# したがって `location.z = Z0 + dZ` としてはいけない（-0.80 を二重に足して床を貫通する）。
# scene_c_prime_traj.json の meta にある Z0_note / Z0_placeholder は
# 点群重心から depth 8.60 で逆算した別物であり、**使わない**。

DZ_A = 0.85                # raised-cosine の振幅（連続ピーク値）
DZ_PERIOD = 35.0           # index 0 -> 35 で 1 周
DZ_CLAMP_INDEX = 35        # index 36 以降は 0（制約4：遮蔽窓 36-40 を Scene C と揃える）


def dz_at(index):
    """index（0..48）における location.z の差分を閉形式で返す。

    stdlib のみ。Blender からも RunPod からも同じ値が出ること。
    照合値は scene_c_expect.TRAJ_PRIME（JSON 由来の凍結リテラル・別経路）。
    """
    if index > DZ_CLAMP_INDEX:
        return 0.0
    return DZ_A * (1.0 - _math.cos(2.0 * _math.pi * float(index) / DZ_PERIOD)) / 2.0


# X は original と完全に同一。別リテラルを置くと片方だけ直す事故が起きるため参照で持つ。
KEYFRAMES["prime"] = list(KEYFRAMES["original"])

VARIANTS = ("original", "edited", "prime")

# Target 幾何中心の Z（TARGET_Z の中点）。location.z ではない。
TARGET_CENTRE_Z = (TARGET_Z[0] + TARGET_Z[1]) / 2.0     # = -0.80

# 制約1（速度）の余裕は 4.99% しかない。DZ_A / DZ_PERIOD は凍結扱い。
#   最大 2D スクリーン速度 11.4014 px/frame @ depth 8.60（index 8->9 と 26->27）
#   VELOCITY["design_limit"] = 12.0
DZ_MAX_SCREEN_VEL = 11.4014
