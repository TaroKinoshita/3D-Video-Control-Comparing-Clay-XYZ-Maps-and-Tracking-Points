"""
Scene C 寸法定義（protocol rev.6 §3 相当）

stdlib のみ。Blender からも RunPod からも import できる。
ここは「作る側」が読む。照合値は scene_c_expect.py（別ファイル・別経路）。
"""

# ---------------------------------------------------------------- カメラ / 出力
CAMERA = {
    "location": (0.0, -10.0, 0.0),
    "rotation_euler_deg": (90.0, 0.0, 0.0),
    "lens_mm": 50.0,
    "sensor_width_mm": 36.0,
    "sensor_fit": "HORIZONTAL",
}
RENDER = {"resolution_x": 832, "resolution_y": 480, "resolution_percentage": 100}

# depth = world_Y + 10.0 / px_per_unit(d) = K / d
K_PX_PER_UNIT = 1155.5556

# ---------------------------------------------------------------- 光源
# 位置・回転・本数・種類は全 split で凍結（§7）。energy / color は Style set 側。
LIGHT = {
    "type": "POINT",
    "location": (0.50, -9.00, 1.50),
    "energy": 1000.0,
    "color": (1.0, 1.0, 1.0),
    "shadow_soft_size": 0.05,
}

# ---------------------------------------------------------------- 断面プリズム
# 全オブジェクトが「X-Y 断面を Z 方向に押し出したプリズム」。
# cs = 断面の (x, y) 列。z0 / z1 = 押し出しの下端 / 上端。
# location は Target のみ X を動かす（Step 2）。他は全て (0,0,0)。

TARGET_CS = [
    (-0.55, -1.20),   # ベベル②の外側端（offset 0.20）
    (-0.35, -1.40),   # ベベル②の内側端
    ( 0.25, -1.40),   # ベベル①の内側端
    ( 0.55, -1.10),   # ベベル①の外側端（offset 0.30）
    ( 0.55, -0.60),
    (-0.55, -0.60),
]
TARGET_Z = (-1.50, -0.10)          # 床に接地
TARGET_X_START = -0.90             # index 0
TARGET_X_END = 0.90                # index 48
TARGET_STEP_X = 0.10               # 候補点格子のピッチ

OBJECTS = [
    # name,        pass_index, cross-section (x,y),                                    z0,    z1,    base_color(setA),        roughness
    ("Wall",        1, [(-4.50, 2.00), (4.50, 2.00), (4.50, 2.05), (-4.50, 2.05)],   -1.60,  2.60, (0.62, 0.60, 0.57), 0.85),
    ("Floor",       2, [(-4.50,-3.00), (4.50,-3.00), (4.50, 2.00), (-4.50, 2.00)],   -1.55, -1.50, (0.42, 0.44, 0.46), 0.70),
    ("Target",      3, TARGET_CS,                                                    -1.50, -0.10, (0.78, 0.32, 0.18), 0.45),
    ("Occluder",    4, [(-0.10,-2.00), (1.05,-2.00), (1.05,-1.95), (-0.10,-1.95)],   -1.50,  0.05, (0.30, 0.34, 0.52), 0.55),
    ("DeepObject",  5, [(-2.90, 0.25), (-2.10, 0.25), (-2.10, 0.75), (-2.90, 0.75)], -1.50, -0.40, (0.50, 0.52, 0.28), 0.60),
    ("Marker",      6, [( 2.40, 1.80), ( 2.60, 1.80), ( 2.60, 2.00), ( 2.40, 2.00)],  1.10,  1.30, (0.90, 0.86, 0.20), 0.35),
]

# ---------------------------------------------------------------- 正規化ボックス
# Arm B の World-XYZ 正規化はジオメトリの bbox ではなく、この固定ボックスを使う。
# 壁を大きくしただけで量子化が粗くなる事故を防ぐ（§18-4）。
# y の上限は 2.05（壁の裏面）。2.00 だと壁の裏面がボックス外に出て
# q が [0,1] を外れ、codebook がクリップして黙って壊れる。
NORM_BOX = {"x": (-4.50, 4.50), "y": (-3.00, 2.05), "z": (-1.60, 2.60)}

# ---------------------------------------------------------------- 編集区間
EDIT = {"hold": (12, 23), "catchup": (24, 35), "t_sync": 36}

# 軌道（Step 2。2026-08-04 確定「index 36 で一致」）
#   通常   index  0 -> 12 : -0.90 -> -0.45   0.037500 unit/frame
#   停止   index 12 -> 23 : -0.45 固定
#   catch  index 23 -> 36 : -0.45 -> +0.45   0.0692308 unit/frame（13ステップ）
#   通常   index 36 -> 48 : +0.45 -> +0.90   0.037500 unit/frame
# blender frame = index + 1。補間は LINEAR。終点は frame 49（一方向運動）。
KEYFRAMES = {
    "original": [(1, -0.90), (49, 0.90)],
    "edited":   [(1, -0.90), (13, -0.45), (24, -0.45), (37, 0.45), (49, 0.90)],
}
VARIANTS = ("original", "edited")
# 完全遮蔽区間（2026-08-04 再設計）。
# 旧設計 (37,43) は幅 1.30 で、Target が最終フレームでも 16.4px しか再出現せず
# 再出現位置の測定が不可能だった。速度上限の下で「完全遮蔽」と「完全再出現」は
# 両立しないため、予算 約56px を 完全遮蔽5f / 再出現8f に分割した。
OCCLUSION_INDEX = (36, 40)
REAPPEAR_INDEX = (41, 48)
MIN_FINAL_VISIBLE_PX = 30.0
FRAMES = 49


# ---------------------------------------------------------------- Style set
# §7：Style set が変えてよいのは材質パラメータのみ。光源の位置・回転・本数・種類は
# 全 split で凍結。**metallic は 0 固定**（use_raytracing=False では鏡面反射の
# 光源が存在せず、metallic 面がほぼ黒に落ちて外観シフトではなく情報消失になる）。
# set A は OBJECTS に埋め込まれている値。set B は色相を大きく回し、
# 壁と床の roughness の関係を逆転させる。
STYLE_A = {name: {"base_color": col, "roughness": rough, "metallic": 0.0}
           for name, _p, _cs, _z0, _z1, col, rough in OBJECTS}

STYLE_B = {
    "Wall":       {"base_color": (0.30, 0.35, 0.42), "roughness": 0.45, "metallic": 0.0},
    "Floor":      {"base_color": (0.55, 0.50, 0.38), "roughness": 0.85, "metallic": 0.0},
    "Target":     {"base_color": (0.20, 0.55, 0.70), "roughness": 0.25, "metallic": 0.0},
    "Occluder":   {"base_color": (0.72, 0.55, 0.22), "roughness": 0.75, "metallic": 0.0},
    "DeepObject": {"base_color": (0.45, 0.28, 0.52), "roughness": 0.40, "metallic": 0.0},
    "Marker":     {"base_color": (0.15, 0.60, 0.35), "roughness": 0.65, "metallic": 0.0},
}

STYLE = {"S0": STYLE_A, "S1": STYLE_B}

# Target video のレンダラ設定（§7。R0 = EEVEE / R1 = Cycles）
RENDERER = {
    "R0": {"engine": "BLENDER_EEVEE", "samples": 32},
    "R1": {"engine": "CYCLES", "samples": 512},
}
SPLITS = ("R0S0", "R1S0", "R0S1")
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
