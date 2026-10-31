"""
armb_encode.py  —  Arm B の codebook 適用（bpy 非依存）

Blender の外でも走るので単体テストできる。encode_arm_b.py（Blender 側）と
contract_assert.py（RunPod 側）の両方から import する。
"""
import numpy as np


def encode(pos, ids, norm_box):
    """
    pos      : (H, W, 3) float  world 座標（R=X / G=Y / B=Z）
    ids      : (H, W)    float  pass_index（0 = 空画素、3 = Target Entity）
    norm_box : {"x": (lo, hi), "y": (lo, hi), "z": (lo, hi)}

    返り値 : ((H, W, 3) uint8, q がボックスから外れた最大量)

    codebook（protocol §4。変更しない）
        Non-target / Background : 0.05 + 0.40q -> 8bit  13-115
        Target Entity           : 0.55 + 0.40q -> 8bit 140-242
        空画素                   : (0, 0, 0)
    丸めは round（floor だと 0.05->12 / 0.45->114 になり protocol と食い違う）
    """
    lo = np.array([norm_box["x"][0], norm_box["y"][0], norm_box["z"][0]],
                  dtype=np.float64)
    hi = np.array([norm_box["x"][1], norm_box["y"][1], norm_box["z"][1]],
                  dtype=np.float64)
    q = (np.asarray(pos, dtype=np.float64) - lo) / (hi - lo)
    excess = max(float(np.max(q) - 1.0), float(-np.min(q)), 0.0)
    q = np.clip(q, 0.0, 1.0)

    ids = np.asarray(ids)
    is_target = (ids == 3.0)[:, :, None]
    is_empty = (ids == 0.0)[:, :, None]
    base = np.where(is_target, 0.55, 0.05)
    v = np.rint((base + 0.40 * q) * 255.0).astype(np.int32)
    v = np.where(is_empty, 0, v)
    return v.astype(np.uint8), excess
