"""
build_min_scene.py -- W2a 極小検証シーン生成

目的
  エクスポータの正解が手計算できるシーンを作る。
  全オブジェクトは scale = 1.0 / rotation = 0 で、matrix_world は平行移動のみ。
  これで既知バグ#1（matrix_world への Scale 焼き込み）を構造的に排除する。

使い方
  blender --background --python build_min_scene.py -- --out C:/thesis/w2a

  GUI のテキストエディタから実行する場合は下の OUT_DIR を書き換えてから実行。
  print の出力は Info エディタではなくシステムコンソールに出る
  （Windows: Window -> Toggle System Console）。

出力
  min_scene.blend   シーン本体
  scene_spec.json   正解値。bpy を使わない素の式で計算している

Blender 5.2 前提。
"""

import json
import math
import os
import sys

import bpy

# ---------------------------------------------------------------------------
# 1. 定数（protocol.md §5 と一致させる）
# ---------------------------------------------------------------------------

RES_X = 832
RES_Y = 480
FPS = 16
N_FRAMES = 49                        # index 0..48 / Blender frame 1..49
FOCAL_MM = 50.0
SENSOR_MM = 36.0                     # 水平方向

CAM_LOC = (0.0, -10.0, 0.0)
CAM_ROT = (math.pi / 2, 0.0, 0.0)    # 局所 -Z が world +Y を向く

TARGET_X_START = -1.8
TARGET_X_END = 1.8
TARGET_SIZE = 1.0                    # 1 x 1 x 1

OCCLUDER_Y = -3.0
OCCLUDER_HALF_X = 0.35
OCCLUDER_HALF_Z = 1.5

WALL_Y = 2.0
WALL_HALF = 6.0

MARKER_LOC = (2.5, 1.9, 1.2)         # 非対称マーカー。上下左右の反転検出用
MARKER_SIZE = 0.2

# pass_index（Object ID パス用）
PASS_TARGET = 1
PASS_OCCLUDER = 2
PASS_WALL = 3
PASS_MARKER = 4

OUT_DIR = "."


# ---------------------------------------------------------------------------
# 2. 素の投影式（bpy を一切使わない。すべての照合の基準）
# ---------------------------------------------------------------------------

def px_per_unit(depth):
    """深度 depth における 1 world unit あたりの画素数。"""
    return RES_X / (depth * SENSOR_MM / FOCAL_MM)


def depth_of(world_y):
    """camera 空間の Z-depth（規約A）。カメラが +Y を向いているので足し算だけ。"""
    return world_y - CAM_LOC[1]


def project(world_xyz):
    """world 座標 -> (u, v, depth)。v は上下反転済み（画像の行は上から下）。"""
    x, y, z = world_xyz
    d = depth_of(y)
    s = px_per_unit(d)
    u = RES_X / 2.0 + x * s
    v = RES_Y / 2.0 - z * s
    return (u, v, d)


def target_x_at(index):
    """index 0..48 における Target の world X。等速・一方向。

    区間数は 48（= N_FRAMES - 1）。終点キーフレームは Blender frame 49。
    ループ運動の場合のみ frame 50 に打つが、このシーンは一方向なので 49。
    """
    span = N_FRAMES - 1
    return TARGET_X_START + (TARGET_X_END - TARGET_X_START) * (index / span)


# ---------------------------------------------------------------------------
# 3. シーン構築
# ---------------------------------------------------------------------------

def wipe():
    """bpy.ops を使わずデータレベルで消す（background でも文脈エラーにならない）。"""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials,
                 bpy.data.cameras, bpy.data.lights):
        for item in list(coll):
            coll.remove(item)


def make_material(name, rgb):
    """Principled BSDF のみ。レンダラ間で解釈が変わるノードを使わない。"""
    mat = bpy.data.materials.new(name)
    # 5.x では材質は常にノードベース。use_nodes は 6.0 で削除予定なので、
    # ノードツリーが未生成の古い挙動のときだけ触る。
    if mat.node_tree is None:
        mat.use_nodes = True
    bsdf = None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf = node
            break
    if bsdf is None:
        raise RuntimeError("Principled BSDF ノードが見つからない: " + name)
    bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat


def make_quad(name, location, half_x, half_z, mat, pass_index):
    """XZ 平面上の四角形。法線は -Y（カメラ側）を向く。scale は 1.0 のまま。

    巻き方向の確認:
      v0=(-hx,0,-hz) v1=(hx,0,-hz) v2=(hx,0,hz)
      (v1-v0) x (v2-v0) = (0, -4*hx*hz, 0)  ->  -Y。カメラ側で正しい。
    """
    verts = [
        (-half_x, 0.0, -half_z),
        (half_x, 0.0, -half_z),
        (half_x, 0.0, half_z),
        (-half_x, 0.0, half_z),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.data.materials.append(mat)
    obj.pass_index = pass_index
    bpy.context.collection.objects.link(obj)
    return obj


def make_cube(name, location, size, mat, pass_index):
    """primitive_cube_add に任せる。法線は Blender が外向きに作る。

    size はメッシュ頂点に焼かれ、object.scale は 1.0 のまま維持される。
    """
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name + "_mesh"
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.pass_index = pass_index
    return obj


def iter_fcurves(obj):
    """オブジェクトの F-Curve を新旧どちらの Action 構造でも辿る。

    Blender 4.4 で slotted actions が入り、5.2 では Action.fcurves が消えた。
    F-Curve は action.layers -> strips -> channelbag(slot) の下に移動している。
    """
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action

    # 旧構造（4.3 以前）
    legacy = getattr(act, "fcurves", None)
    if legacy is not None:
        return list(legacy)

    # 新構造（4.4 以降）
    out = []
    slot = getattr(ad, "action_slot", None)
    for layer in act.layers:
        for strip in layer.strips:
            cb = None
            if slot is not None and hasattr(strip, "channelbag"):
                try:
                    cb = strip.channelbag(slot)
                except (TypeError, RuntimeError):
                    cb = None
            if cb is None and hasattr(strip, "channelbags"):
                bags = list(strip.channelbags)
                cb = bags[0] if bags else None
            if cb is not None:
                out.extend(cb.fcurves)
    return out


def set_default_interpolation(kind="LINEAR"):
    """新規キーフレームの既定補間。既定の BEZIER だと等速にならない。"""
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = kind
        return True
    except (AttributeError, TypeError):
        return False


def pick_eevee():
    """5.2 は BLENDER_EEVEE。念のため実在するエンジン名を選ぶ。"""
    available = [
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ]
    for name in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if name in available:
            return name
    raise RuntimeError("EEVEE が見つからない。利用可能: %s" % available)


def build_scene():
    scene = bpy.context.scene

    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = N_FRAMES

    # --- カメラ ---
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = FOCAL_MM
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = SENSOR_MM
    cam_data.dof.use_dof = False
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = CAM_LOC
    cam.rotation_euler = CAM_ROT
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    # --- ライト（単一 Sun。レンダラ間で解釈が揺れにくい） ---
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("Sun", light_data)
    light.location = (4.0, -6.0, 6.0)
    light.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    bpy.context.collection.objects.link(light)

    # --- 材質 ---
    m_target = make_material("M_Target", (0.80, 0.25, 0.15))
    m_occ = make_material("M_Occluder", (0.20, 0.45, 0.70))
    m_wall = make_material("M_Wall", (0.55, 0.55, 0.55))
    m_mark = make_material("M_Marker", (0.90, 0.85, 0.10))

    # --- オブジェクト ---
    wall = make_quad("Wall", (0.0, WALL_Y, 0.0),
                     WALL_HALF, WALL_HALF, m_wall, PASS_WALL)
    occ = make_quad("Occluder", (0.0, OCCLUDER_Y, 0.0),
                    OCCLUDER_HALF_X, OCCLUDER_HALF_Z, m_occ, PASS_OCCLUDER)
    target = make_cube("Target", (TARGET_X_START, 0.0, 0.0),
                       TARGET_SIZE, m_target, PASS_TARGET)
    marker = make_cube("Marker", MARKER_LOC,
                       MARKER_SIZE, m_mark, PASS_MARKER)

    # --- アニメーション ---
    # LINEAR 必須。既定の BEZIER は ease-in/out がかかり等速にならず、
    # 手計算値と合わず V_max の assert も意味を失う。
    set_default_interpolation("LINEAR")
    target.location = (TARGET_X_START, 0.0, 0.0)
    target.keyframe_insert(data_path="location", frame=1)
    target.location = (TARGET_X_END, 0.0, 0.0)
    target.keyframe_insert(data_path="location", frame=N_FRAMES)

    fcurves = iter_fcurves(target)
    if not fcurves:
        raise RuntimeError(
            "F-Curve が取得できない。Action 構造が想定外なので "
            "iter_fcurves() を確認する必要がある。")
    for fc in fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.update()

    return {"wall": wall, "occluder": occ, "target": target,
            "marker": marker, "camera": cam, "light": light}


# ---------------------------------------------------------------------------
# 4. レンダー設定
# ---------------------------------------------------------------------------

def configure_passes(view_layer):
    """出力するパスを .blend に焼く（protocol.md §22.2 / §22.4）。

    シーンファイルが「何を出力するか」を自己記述している状態にする。
    実行時に有効化する設計だと、.blend を見ても何が出るか分からない。
    エクスポータ側でも冪等に設定するが、正本はここ。
    """
    view_layer.use_pass_position = True      # Arm B / §22.2
    view_layer.use_pass_z = True             # VGGT 評価の GT / §22.4
    view_layer.use_pass_object_index = True  # マスクは pass_index で定義（W2a-1）
    view_layer.use_pass_combined = True      # Arm A クレイ / Target video

    # Mist は 0-1 に正規化された相対値。GT には使えない（§22.1）
    if hasattr(view_layer, "use_pass_mist"):
        view_layer.use_pass_mist = False


def configure_common(scene):
    """全用途で無効化するもの（protocol.md §22.3）。"""
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def configure_geometry(scene):
    """幾何パス用。空間平均を全部切る。

    EEVEE も taa_render_samples 個のサンプルを平均する。
    Position / Depth を平均すると輪郭に実在しない座標が生まれるため、
    Cycles と同じ理由で 1 に落とす必要がある。
    """
    scene.render.engine = pick_eevee()
    configure_common(scene)
    scene.render.filter_size = 0.01
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 1
        if hasattr(scene.eevee, "use_motion_blur"):
            scene.eevee.use_motion_blur = False


def configure_appearance(scene):
    """Arm A クレイ / R0 の Target video 用。AA は色にしか効かないので許容。"""
    scene.render.engine = pick_eevee()
    configure_common(scene)
    scene.render.filter_size = 1.5
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 64


def configure_cycles_appearance(scene):
    """R1（Cycles）で Target video を焼くとき用。フリッカー源を全部切る。"""
    scene.render.engine = "CYCLES"
    configure_common(scene)
    scene.render.filter_size = 1.5
    if hasattr(scene, "cycles"):
        scene.cycles.use_denoising = False
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_animated_seed = False
        scene.cycles.seed = 0
        scene.cycles.samples = 512


# ---------------------------------------------------------------------------
# 5. 正解値（scene_spec.json）
# ---------------------------------------------------------------------------

def build_spec():
    spec = {
        "index_base": 0,
        "n_frames": N_FRAMES,
        "blender_frame_of_index_0": 1,
        "resolution": [RES_X, RES_Y],
        "fps": FPS,
        "camera": {
            "location": list(CAM_LOC),
            "rotation_euler_rad": list(CAM_ROT),
            "focal_mm": FOCAL_MM,
            "sensor_mm": SENSOR_MM,
            "sensor_fit": "HORIZONTAL",
            "looks_along": "+Y",
            "depth_formula": "depth = world_y + 10",
            "u_formula": "u = 416 + X * 1155.5556 / depth",
            "v_formula": "v = 240 - Z * 1155.5556 / depth",
        },
        "pass_index": {
            "Target": PASS_TARGET,
            "Occluder": PASS_OCCLUDER,
            "Wall": PASS_WALL,
            "Marker": PASS_MARKER,
        },
        "entity_class": {
            "Target": "target",
            "Occluder": "non_target_foreground",
            "Wall": "background",
            "Marker": "background",
        },
        "static_depth": {
            "Wall": depth_of(WALL_Y),
            "Occluder": depth_of(OCCLUDER_Y),
            "Target_center": depth_of(0.0),
        },
        "px_per_unit": {
            "depth_7": px_per_unit(7.0),
            "depth_10": px_per_unit(10.0),
            "depth_11_9": px_per_unit(11.9),
            "depth_12": px_per_unit(12.0),
        },
        "marker": {
            "world": list(MARKER_LOC),
            "expected_uvd": list(project(MARKER_LOC)),
            "note": "左右反転すると u が 173 付近、上下反転すると v が 356 付近になる",
        },
        "frame_edge_world_x_at_depth_10": [-3.6, 3.6],
        "render_config": {
            "geometry": {"engine": "EEVEE", "taa_render_samples": 1,
                         "filter_size": 0.01},
            "appearance_R0": {"engine": "EEVEE", "taa_render_samples": 64,
                              "filter_size": 1.5},
            "appearance_R1": {"engine": "CYCLES", "samples": 512,
                              "denoising": False, "adaptive_sampling": False,
                              "animated_seed": False},
            "always_off": ["motion_blur", "dof", "denoise"],
        },
        "target_track": [],
        "target_velocity_px_per_frame": None,
    }

    for i in range(N_FRAMES):
        x = target_x_at(i)
        u, v, d = project((x, 0.0, 0.0))
        spec["target_track"].append({
            "index": i,
            "blender_frame": i + 1,
            "world_x": x,
            "u": u,
            "v": v,
            "depth": d,
        })

    spec["target_velocity_px_per_frame"] = (
        spec["target_track"][1]["u"] - spec["target_track"][0]["u"])

    # Target 立方体の8頂点（index 0 と 48）。Arm C の照合に使う。
    h = TARGET_SIZE / 2.0
    local = [(sx * h, sy * h, sz * h)
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    spec["target_vertices"] = {}
    for i in (0, N_FRAMES - 1):
        x = target_x_at(i)
        spec["target_vertices"][str(i)] = [
            {
                "local": list(lv),
                "world": [x + lv[0], lv[1], lv[2]],
                "uvd": list(project((x + lv[0], lv[1], lv[2]))),
            }
            for lv in local
        ]

    # 遮蔽区間（Target 中心が遮蔽板の裏に入る index の範囲）
    occ_s = px_per_unit(depth_of(OCCLUDER_Y))
    occ_u_min = RES_X / 2.0 - OCCLUDER_HALF_X * occ_s
    occ_u_max = RES_X / 2.0 + OCCLUDER_HALF_X * occ_s
    hidden = [e["index"] for e in spec["target_track"]
              if occ_u_min <= e["u"] <= occ_u_max]
    spec["occluder_screen_u_range"] = [occ_u_min, occ_u_max]
    spec["target_center_hidden_index"] = (
        [hidden[0], hidden[-1]] if hidden else [])

    return spec


# ---------------------------------------------------------------------------
# 6. self-check（bpy の結果と素の式を突き合わせる）
# ---------------------------------------------------------------------------

def self_check(objs):
    """ここで落ちたらシーンの作り方が意図と違う。先に直す。"""
    problems = []

    for name in ("Target", "Occluder", "Wall", "Marker", "Cam"):
        s = tuple(round(v, 9) for v in bpy.data.objects[name].scale)
        if s != (1.0, 1.0, 1.0):
            problems.append("%s の scale が 1.0 でない: %s" % (name, s))

    for name in ("Target", "Occluder", "Wall", "Marker"):
        r = tuple(round(v, 9) for v in bpy.data.objects[name].rotation_euler)
        if r != (0.0, 0.0, 0.0):
            problems.append("%s の rotation が 0 でない: %s" % (name, r))

    # 既知バグ#1 の検査。カメラ matrix_world が純回転になっているか
    mw = objs["camera"].matrix_world
    for col in range(3):
        n = math.sqrt(sum(mw[row][col] ** 2 for row in range(3)))
        if abs(n - 1.0) > 1e-6:
            problems.append("camera matrix_world 列%d のノルムが 1.0 でない: %f"
                            % (col, n))

    # 法線が -Y を向いているか（既知バグ#5）
    for name in ("Wall", "Occluder"):
        obj = bpy.data.objects[name]
        for poly in obj.data.polygons:
            ny = poly.normal.y
            if ny > -0.99:
                problems.append("%s の法線が -Y を向いていない: %s"
                                % (name, tuple(round(c, 4) for c in poly.normal)))
                break

    scene = bpy.context.scene
    n_render = scene.frame_end - scene.frame_start + 1
    if n_render != N_FRAMES:
        problems.append("レンダー枚数が %d（49 でない）" % n_render)

    if scene.render.engine.startswith("BLENDER_EEVEE"):
        if hasattr(scene, "eevee") and scene.eevee.taa_render_samples != 1:
            problems.append("幾何パス設定なのに taa_render_samples が %d"
                            % scene.eevee.taa_render_samples)
    if scene.render.filter_size > 0.02:
        problems.append("filter_size が %f（幾何パスでは 0.01）"
                        % scene.render.filter_size)

    # パスが .blend に焼かれているか（contract_assert の S13-S15 と同じ検査）
    vl = bpy.context.view_layer
    for attr, label in (("use_pass_position", "Position"),
                        ("use_pass_z", "Depth"),
                        ("use_pass_object_index", "Object Index")):
        if not getattr(vl, attr, False):
            problems.append("%s パスが有効化されていない" % label)
    if getattr(vl, "use_pass_mist", False):
        problems.append("Mist パスが有効になっている（GT には使えない相対値）")

    # 補間が LINEAR か。BEZIER だと ease-in/out で等速にならない。
    fcurves = iter_fcurves(objs["target"])
    if not fcurves:
        problems.append("Target の F-Curve が取得できない")
    for fc in fcurves:
        for kp in fc.keyframe_points:
            if kp.interpolation != "LINEAR":
                problems.append("Target の補間が %s（LINEAR でない）"
                                % kp.interpolation)
                break

    # 素の式と bpy 側の投影を突き合わせる。
    # index 24 は区間の中点なので BEZIER でも一致してしまう。
    # 12 と 36 を必ず含めて非等速を検出する。
    from bpy_extras.object_utils import world_to_camera_view
    for i in (0, 12, 24, 36, N_FRAMES - 1):
        scene.frame_set(i + 1)
        wpos = objs["target"].matrix_world.translation
        ndc = world_to_camera_view(scene, objs["camera"], wpos)
        u_bpy = ndc.x * RES_X
        v_bpy = (1.0 - ndc.y) * RES_Y      # ここが上下反転の要注意点
        u_ref, v_ref, _ = project((target_x_at(i), 0.0, 0.0))
        if abs(u_bpy - u_ref) > 0.5 or abs(v_bpy - v_ref) > 0.5:
            problems.append(
                "index %d の投影が不一致 bpy=(%.2f, %.2f) 式=(%.2f, %.2f)"
                % (i, u_bpy, v_bpy, u_ref, v_ref))

    # 速度が §13 の設計上限 12 px/frame 以内か
    spec_v = px_per_unit(10.0) * (TARGET_X_END - TARGET_X_START) / (N_FRAMES - 1)
    if spec_v > 12.0:
        problems.append("Target 速度 %.2f px/frame が上限 12 を超えている" % spec_v)

    scene.frame_set(1)
    return problems


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def parse_out_dir():
    if "--" in sys.argv:
        rest = sys.argv[sys.argv.index("--") + 1:]
        if "--out" in rest:
            idx = rest.index("--out") + 1
            if idx < len(rest):
                return rest[idx]
    return OUT_DIR


def main():
    out_dir = parse_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    wipe()
    objs = build_scene()
    configure_geometry(bpy.context.scene)
    configure_passes(bpy.context.view_layer)

    problems = self_check(objs)

    spec = build_spec()
    spec_path = os.path.join(out_dir, "scene_spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    blend_path = os.path.join(out_dir, "min_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    tt = spec["target_track"]
    print("")
    print("=" * 62)
    print("scene_spec.json :", spec_path)
    print("min_scene.blend :", blend_path)
    print("-" * 62)
    print("engine                     : %s" % bpy.context.scene.render.engine)
    print("Target velocity            : %.4f px/frame  (limit 12)"
          % spec["target_velocity_px_per_frame"])
    print("Target u  idx 0 / 24 / 48  : %.2f / %.2f / %.2f"
          % (tt[0]["u"], tt[24]["u"], tt[48]["u"]))
    print("Target v  (all frames)     : %.2f" % tt[0]["v"])
    print("Marker    u / v / depth    : %.2f / %.2f / %.2f"
          % tuple(spec["marker"]["expected_uvd"]))
    print("Wall depth / Occluder      : %.2f / %.2f"
          % (spec["static_depth"]["Wall"], spec["static_depth"]["Occluder"]))
    print("Occluder u range           : %.2f .. %.2f"
          % tuple(spec["occluder_screen_u_range"]))
    print("Target center hidden idx   : %s" % spec["target_center_hidden_index"])
    print("-" * 62)
    vl = bpy.context.view_layer
    print("Passes  Position / Depth / ObjIdx / Mist : %s / %s / %s / %s"
          % (vl.use_pass_position, vl.use_pass_z,
             vl.use_pass_object_index, getattr(vl, "use_pass_mist", None)))
    print("-" * 62)
    if problems:
        print("SELF-CHECK FAILED  %d 件" % len(problems))
        for p in problems:
            print("   -", p)
    else:
        print("SELF-CHECK  全項目 OK")
    print("=" * 62)


main()
