"""
build_scene_c.py  —  W2b Step 1
Scene C 本番シーンの構築：形状 / カメラ / 光源 / pass_index / 材質 + SELF-CHECK

アニメーション（Step 2）とレンダー設定・コンポジタ（Step 3）は含まない。

使い方
  $BL = "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe"
  & $BL --background --python build_scene_c.py "--" --out .

方針（作法に対応）
  #2  生成 / GPU / 採点 を分ける → これは生成のみ。レンダーしない
  #6  API を推測しない → 冒頭で PROBE を出す。落ちた箇所が特定できる
  #7  例外を捕まえて残りを続行 → 各検査は独立。最初の FAIL で止まらない
  #12 自前実装で書いて自前実装で読む → 照合値は scene_c_expect.py（別経路）
  #13 scale = 1.0 / rotation = 0 → 全オブジェクト。サイズは頂点に焼く
  #14 サイズは scale ではなくメッシュ頂点に焼く → プリズムの断面座標が絶対座標
  #15 法線は Blender に作らせる → recalc_face_normals に決めさせる
"""
import sys
import os
import json
import math

# Blender の --python はスクリプトのディレクトリを sys.path に入れない。
# 入れないと import scene_c_spec が ModuleNotFoundError で落ちる。
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() \
    else os.path.abspath(os.getcwd())
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bpy
import bmesh
from mathutils import Vector

import scene_c_spec as S
import scene_c_expect as E

RESULTS = {"probe": {}, "checks": [], "topology": {}, "volumes": {}}


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-34s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def guarded(name, fn):
    """作法#7：例外で全体を止めない。"""
    try:
        return fn()
    except Exception as exc:
        rec(name, False, "EXCEPTION %s: %s" % (type(exc).__name__, exc))
        return None


# ---------------------------------------------------------------- 引数
def parse_out():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    out = "."
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 < len(argv):
            out = argv[i + 1]
    variant = "original"
    if "--variant" in argv:
        i = argv.index("--variant")
        if i + 1 < len(argv):
            variant = argv[i + 1]
    if variant not in S.VARIANTS:
        raise SystemExit("--variant は %s のいずれか（指定: %r）" % (list(S.VARIANTS), variant))
    return out, variant


# ---------------------------------------------------------------- API プローブ
def probe():
    """作法#6：使う API の存在を先に実測して出す。推測しない。"""
    p = RESULTS["probe"]
    p["blender_version"] = bpy.app.version_string
    p["has_read_factory_settings"] = hasattr(bpy.ops.wm, "read_factory_settings")
    p["bmesh_has_recalc"] = hasattr(bmesh.ops, "recalc_face_normals")
    # 型クラス基準の hasattr は 5.2 で偽陰性を出す（Mesh.validate /
    # Object.pass_index / Scene.compositing_node_group が実際は動くのに False）。
    # ここではインスタンス基準で確認する。
    try:
        _m = bpy.data.meshes.new("_probe_tmp")
        _m.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [], [[0, 1, 2]])
        p["mesh_from_pydata"] = "OK verts=%d" % len(_m.vertices)
        p["mesh_validate"] = "OK returned %s" % _m.validate(verbose=False)
        _o = bpy.data.objects.new("_probe_tmp", _m)
        _o.pass_index = 7
        p["object_pass_index"] = "OK %d" % _o.pass_index
        bpy.data.objects.remove(_o, do_unlink=True)
        bpy.data.meshes.remove(_m)
    except Exception as exc:
        p["instance_probe"] = "EXCEPTION %s: %s" % (type(exc).__name__, exc)
    try:
        p["scene_compositing_node_group"] = type(
            bpy.context.scene.compositing_node_group).__name__
    except Exception as exc:
        p["scene_compositing_node_group"] = "EXCEPTION %s" % exc
    try:
        from bpy_extras.object_utils import world_to_camera_view  # noqa
        p["has_world_to_camera_view"] = True
    except Exception as exc:
        p["has_world_to_camera_view"] = "IMPORT FAILED: %s" % exc
    print("-" * 68)
    print("PROBE")
    for k, v in p.items():
        print("  %-38s %s" % (k, v))
    print("-" * 68)


# ---------------------------------------------------------------- シーン初期化
def reset_scene():
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return "read_factory_settings(use_empty=True)"
    except Exception as exc:
        # フォールバック：手で消す
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        return "manual removal (read_factory_settings failed: %s)" % exc


# ---------------------------------------------------------------- プリズム生成
def signed_volume(mesh):
    """四面体分割による符号付き体積。外向き法線なら正。"""
    total = 0.0
    for poly in mesh.polygons:
        idx = list(poly.vertices)
        v0 = mesh.vertices[idx[0]].co
        for a, b in zip(idx[1:-1], idx[2:]):
            v1 = mesh.vertices[a].co
            v2 = mesh.vertices[b].co
            total += v0.dot(v1.cross(v2)) / 6.0
    return total


def make_prism(name, cs, z0, z1, pass_index, color, roughness):
    n = len(cs)
    verts = [(x, y, z0) for x, y in cs] + [(x, y, z1) for x, y in cs]
    faces = [list(range(n)), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n, i + n])

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    bad = mesh.validate(verbose=False)
    mesh.update()

    # 作法#15：向きは Blender に決めさせる
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # 作法#13：location 以外は触らない。scale=1 / rotation=0 のまま
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    obj.pass_index = pass_index

    mat = bpy.data.materials.new("M_" + name)
    # use_nodes は Blender 6.0 で削除予定（5.2 で DeprecationWarning）。
    # 5.2 の新規マテリアルは既に node_tree を持つので触らない。
    bsdf = None
    if getattr(mat, "node_tree", None) is not None:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    obj.data.materials.append(mat)

    RESULTS["topology"][name] = [len(mesh.vertices), len(mesh.polygons)]
    RESULTS["volumes"][name] = signed_volume(mesh)
    return obj, bad, bsdf is not None


# ---------------------------------------------------------------- カメラ / 光源
def make_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = S.CAMERA["lens_mm"]
    cam_data.sensor_fit = S.CAMERA["sensor_fit"]
    cam_data.sensor_width = S.CAMERA["sensor_width_mm"]
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = S.CAMERA["location"]
    cam.rotation_euler = tuple(math.radians(a)
                               for a in S.CAMERA["rotation_euler_deg"])
    cam.scale = (1.0, 1.0, 1.0)
    bpy.context.scene.camera = cam
    return cam


def make_light():
    ld = bpy.data.lights.new("KeyLight", type=S.LIGHT["type"])
    ld.energy = S.LIGHT["energy"]
    ld.color = S.LIGHT["color"]
    if hasattr(ld, "shadow_soft_size"):
        ld.shadow_soft_size = S.LIGHT["shadow_soft_size"]
    lo = bpy.data.objects.new("KeyLight", ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = S.LIGHT["location"]
    lo.rotation_euler = (0.0, 0.0, 0.0)
    lo.scale = (1.0, 1.0, 1.0)
    return lo



# ---------------------------------------------------------------- アニメーション
def get_fcurves(obj):
    """5.2 は Action.fcurves を layers/strips/channelbag に移した（§22）。
    どの経路が生きているか推測せず、順に試して見つかったものを返す。"""
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return [], "no action"
    act = ad.action
    try:
        fcs = list(act.fcurves)
        if fcs:
            return fcs, "Action.fcurves (旧API)"
    except Exception:
        pass
    out = []
    try:
        slot = getattr(ad, "action_slot", None)
        for layer in act.layers:
            for strip in layer.strips:
                cb = None
                if slot is not None and hasattr(strip, "channelbag"):
                    try:
                        cb = strip.channelbag(slot)
                    except Exception:
                        cb = None
                if cb is not None:
                    out.extend(cb.fcurves)
                elif hasattr(strip, "channelbags"):
                    for cb2 in strip.channelbags:
                        out.extend(cb2.fcurves)
        if out:
            return out, "layers/strips/channelbag (5.2 API)"
    except Exception as exc:
        return [], "layered API EXCEPTION %s" % exc
    return [], "not found"


def animate_target(obj, variant):
    """location.x にキーフレームを打つ。補間は LINEAR。

    variant "prime"（Scene C′）のみ location.z にも 49 本打つ（W7）。
    z に入れるのは **差分 dz_at(index)** であり、Target 幾何中心の -0.80 は含まない。
    -0.80 はメッシュ頂点に焼かれており、Scene C では location.z は 0.0 のまま。
    `location.z = Z0 + dZ` にすると -0.80 を二重に足して床を貫通する。
    """
    # 既定補間を LINEAR にしておく（fcurve に触れなくても済むように）
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
        pref_ok = "LINEAR"
    except Exception as exc:
        pref_ok = "FAILED %s" % exc
    print("keyframe_new_interpolation_type : %s" % pref_ok)

    obj.animation_data_clear()
    for frame, x in S.KEYFRAMES[variant]:
        obj.location = (x, 0.0, 0.0)
        obj.keyframe_insert(data_path="location", index=0, frame=frame)

    # W7：Scene C′ の縦揺れ。index ごとに1本打つ（点が密なので LINEAR で足りる）。
    n_z = 0
    if variant == "prime":
        for index in range(S.FRAMES):
            obj.location.z = S.dz_at(index)
            obj.keyframe_insert(data_path="location", index=2, frame=index + 1)
            n_z += 1
    print("Z keyframes inserted : %d (variant %s)" % (n_z, variant))

    fcs, how = get_fcurves(obj)
    print("fcurve access : %s (%d curve(s))" % (how, len(fcs)))
    n_forced = 0
    for fc in fcs:
        # W7：index 2（Z）も対象。ここを 0 だけにしていると Z が BEZIER のまま通り、
        # 補間オーバーシュートで遮蔽窓 36-40 の dZ が 0 でなくなる（制約4 が黙って壊れる）。
        if fc.data_path != "location" or fc.array_index not in (0, 2):
            continue
        for kp in fc.keyframe_points:
            if kp.interpolation != "LINEAR":
                kp.interpolation = "LINEAR"
                n_forced += 1
        try:
            fc.update()
        except Exception:
            pass
    print("interpolation forced to LINEAR : %d keyframe(s)" % n_forced)
    return len(fcs), how


def target_centre_world_z(obj):
    """Target 幾何中心の world Z を頂点から求める（W7・案2）。

    spec の定数 -0.80 を信じず、matrix_world @ 頂点 で読む。
    これで location.z の誤りとメッシュ頂点の誤りの両方が捕まる。
    """
    zs = [(obj.matrix_world @ vt.co).z for vt in obj.data.vertices]
    return (min(zs) + max(zs)) / 2.0


def check_trajectory_prime(scene, obj):
    """Scene C′：X / location.z / u / v / depth を全フレーム突合（W7）。

    照合先は E.TRAJ_PRIME（scene_c_prime_traj.json 由来の凍結リテラル）。
    書き込みは S.dz_at() の閉形式なので経路が分かれている（作法#12）。
    """
    nbad = 0
    worst = {"x": 0.0, "z": 0.0, "u": 0.0, "v": 0.0, "d": 0.0}
    for idx, frame, ex, ez, eu, ev in E.TRAJ_PRIME:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        t = obj.matrix_world.translation
        zc = target_centre_world_z(obj)
        u, v, d = project(scene, cam_ref[0], (t.x, -1.00, zc))
        dd = {"x": abs(t.x - ex), "z": abs(t.z - ez),
              "u": abs(u - eu), "v": abs(v - ev), "d": abs(d - 9.00)}
        for k in worst:
            worst[k] = max(worst[k], dd[k])
        if (dd["x"] > E.TOL_X or dd["z"] > E.TOL_Z
                or dd["u"] > E.TOL_PX or dd["v"] > E.TOL_V
                or dd["d"] > E.TOL_DEPTH):
            nbad += 1
            if nbad <= 8:
                print("FAIL  prime index %2d frame %2d: got X=%+.7f z=%+.7f "
                      "u=%.4f v=%.4f d=%.5f | want X=%+.7f z=%+.7f u=%.4f v=%.4f d=9.00000"
                      % (idx, frame, t.x, t.z, u, v, d, ex, ez, eu, ev))
    return nbad, worst


def check_prime_constraints(scene, obj):
    """§33.4 の制約1-4 を frame_set + matrix_world で実測する（W7）。

    制約1 速度  : 最大 2D スクリーン速度 <= VELOCITY["design_limit"]（depth 8.60）
    制約2 面積  : Y が動かない = depth 9.00 一定（check_trajectory_prime の d で担保）
    制約3 始点  : index 0 で X=-0.90 かつ location.z=0
    制約4 遮蔽窓: index 36-40 で location.z が厳密に 0
    """
    pxu = E.SCALARS["px_per_unit_at_8.60"]
    samples = []
    for index in range(S.FRAMES):
        scene.frame_set(index + 1)
        bpy.context.view_layer.update()
        t = obj.matrix_world.translation
        samples.append((index, t.x, t.z))

    vel = []
    for (i0, x0, z0), (i1, x1, z1) in zip(samples, samples[1:]):
        du = (x1 - x0) * pxu
        dv = -(z1 - z0) * pxu
        vel.append(math.sqrt(du * du + dv * dv))
    vmax = max(vel)
    rec("prime_c1_velocity", vmax <= E.VELOCITY["design_limit"],
        "max %.4f px/frame at index %d->%d (limit %.1f / expect %.4f)"
        % (vmax, vel.index(vmax), vel.index(vmax) + 1,
           E.VELOCITY["design_limit"], E.PRIME_SCALARS["max_screen_vel_at_8.60"]))

    i0x, i0z = samples[0][1], samples[0][2]
    rec("prime_c3_start", abs(i0x - S.TARGET_X_START) <= E.TOL_X and i0z == 0.0,
        "index 0  X=%+.7f  location.z=%+.9f" % (i0x, i0z))

    lo, hi = S.OCCLUSION_INDEX
    occ = [(i, z) for i, _x, z in samples if lo <= i <= hi and z != 0.0]
    rec("prime_c4_occlusion_window", not occ,
        "index %d-%d の location.z 非ゼロ: %s" % (lo, hi, occ or "なし"))

    # ピーク値と NORM_BOX（Arm B の量子化がクリップしないこと）
    zmax = max(z for _i, _x, z in samples)
    ipk = [i for i, _x, z in samples if z == zmax][0]
    rec("prime_dz_peak", abs(zmax - E.PRIME_SCALARS["dz_peak"]) <= E.TOL_Z,
        "peak %.7f at index %d (expect %.7f at %d)"
        % (zmax, ipk, E.PRIME_SCALARS["dz_peak"], E.PRIME_SCALARS["dz_peak_index"]))

    vz = (S.TARGET_Z[0] + zmax, S.TARGET_Z[1] + zmax)
    rec("prime_vertex_z_in_norm_box",
        S.NORM_BOX["z"][0] <= vz[0] and vz[1] <= S.NORM_BOX["z"][1],
        "peak 頂点Z [%.7f, %.7f] / NORM_BOX z %s" % (vz[0], vz[1], S.NORM_BOX["z"]))

    scene.frame_set(1)
    return vmax


def check_trajectory(scene, obj, variant):
    """fcurve を覗かず、frame_set して matrix_world を読む。
    作法#12：書き込み経路と読み出し経路を分ける。"""
    if variant == "prime":
        nbad, worst = check_trajectory_prime(scene, obj)
        print("prime worst: dX=%.3e dz=%.3e du=%.4f dv=%.4f dd=%.3e"
              % (worst["x"], worst["z"], worst["u"], worst["v"], worst["d"]))
        return nbad, max(worst["u"], worst["v"])
    table = E.TRAJ_EDITED if variant == "edited" else E.TRAJ_ORIGINAL
    nbad = 0
    worst = 0.0
    for idx, frame, ex, eu in table:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        x = obj.matrix_world.translation.x
        u, _v, _d = project(scene, cam_ref[0], (x, -1.00, 0.0))
        dx, du = abs(x - ex), abs(u - eu)
        worst = max(worst, du)
        if dx > E.TOL_X or du > E.TOL_U:
            nbad += 1
            if nbad <= 8:
                print("FAIL  traj index %2d frame %2d: got X=%+.7f u=%.4f | "
                      "want X=%+.7f u=%.4f" % (idx, frame, x, u, ex, eu))
    return nbad, worst


cam_ref = [None]

# ---------------------------------------------------------------- SELF-CHECK
def project(scene, cam, world_xyz):
    """protocol §13.4：world_to_camera_view / v = (1 - ndc.y) * H"""
    from bpy_extras.object_utils import world_to_camera_view
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    ndc = world_to_camera_view(scene, cam, Vector(world_xyz))
    return ndc.x * W, (1.0 - ndc.y) * H, ndc.z


def self_check(scene, cam, objs):
    ok_all = True

    ok_all &= rec("resolution",
                  scene.render.resolution_x == E.SCALARS["resolution_x"]
                  and scene.render.resolution_y == E.SCALARS["resolution_y"]
                  and scene.render.resolution_percentage == 100,
                  "%dx%d @%d%%" % (scene.render.resolution_x,
                                   scene.render.resolution_y,
                                   scene.render.resolution_percentage))

    ok_all &= rec("camera_params",
                  abs(cam.data.lens - 50.0) < 1e-6
                  and cam.data.sensor_fit == "HORIZONTAL"
                  and abs(cam.data.sensor_width - 36.0) < 1e-6,
                  "%.1fmm / %s / %.1fmm" % (cam.data.lens, cam.data.sensor_fit,
                                            cam.data.sensor_width))

    m = cam.matrix_world
    col_norms = [m.col[i].to_3d().length for i in range(3)]
    ok_all &= rec("camera_matrix_no_scale",
                  all(abs(c - 1.0) < 1e-6 for c in col_norms),
                  "col norms " + " ".join("%.6f" % c for c in col_norms))

    # 既知バグ#1：全オブジェクトの scale / rotation
    bad = [o.name for o in objs
           if any(abs(s - 1.0) > 1e-9 for s in o.scale)
           or any(abs(r) > 1e-9 for r in o.rotation_euler)]
    ok_all &= rec("all_scale1_rot0", not bad, "violations: %s" % (bad or "none"))

    # pass_index の一意性と値
    pidx = {o.name: o.pass_index for o in objs}
    ok_all &= rec("pass_index_values", pidx == E.PASS_INDEX, str(pidx))
    ok_all &= rec("pass_index_unique",
                  len(set(pidx.values())) == len(pidx), str(sorted(pidx.values())))

    # トポロジ（プリズム：頂点 2n / 面 n+2）
    ok_all &= rec("topology", RESULTS["topology"] ==
                  {k: list(v) for k, v in E.EXPECT_TOPOLOGY.items()},
                  str(RESULTS["topology"]))

    # 符号付き体積が正（法線が外向き）
    neg = {k: round(v, 6) for k, v in RESULTS["volumes"].items() if v <= 0.0}
    ok_all &= rec("signed_volume_positive", not neg,
                  "negative: %s" % (neg or "none"))

    # 順投影の突合（本題）
    n_ok = n_bad = 0
    for label, x, y, z, eu, ev, ed in E.POINTS:
        try:
            u, v, d = project(scene, cam, (x, y, z))
        except Exception as exc:
            n_bad += 1
            print("FAIL  %-16s EXCEPTION %s" % (label, exc))
            continue
        du, dv, dd = abs(u - eu), abs(v - ev), abs(d - ed)
        if du <= E.TOL_PX and dv <= E.TOL_PX and dd <= E.TOL_DEPTH:
            n_ok += 1
        else:
            n_bad += 1
            print("FAIL  %-16s got u=%.4f v=%.4f d=%.5f | want u=%.4f v=%.4f d=%.5f"
                  % (label, u, v, d, eu, ev, ed))
    ok_all &= rec("projection_points", n_bad == 0,
                  "%d ok / %d bad (tol %.3f px)" % (n_ok, n_bad, E.TOL_PX))

    # depth = world_Y + 10 の規約
    dev = []
    for label, x, y, z, _u, _v, _d in E.POINTS:
        try:
            _, _, d = project(scene, cam, (x, y, z))
            dev.append(abs(d - (y + 10.0)))
        except Exception:
            dev.append(float("inf"))
    ok_all &= rec("depth_equals_Y_plus_10", max(dev) <= 1e-4,
                  "max deviation %.3e" % max(dev))

    # 材質は Principled BSDF のみ
    bad_mat = []
    for o in objs:
        for slot in o.material_slots:
            if slot.material is None or getattr(slot.material, "node_tree", None) is None:
                bad_mat.append(o.name)
                continue
            types = {n.type for n in slot.material.node_tree.nodes}
            if "BSDF_PRINCIPLED" not in types:
                bad_mat.append(o.name)
            if types - {"BSDF_PRINCIPLED", "OUTPUT_MATERIAL"}:
                bad_mat.append(o.name + ":extra_nodes")
    ok_all &= rec("principled_bsdf_only", not bad_mat,
                  "violations: %s" % (bad_mat or "none"))

    # 光源
    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    lok = (len(lights) == 1
           and lights[0].data.type == S.LIGHT["type"]
           and all(abs(a - b) < 1e-9
                   for a, b in zip(lights[0].location, S.LIGHT["location"])))
    ok_all &= rec("light_frozen_placement", lok,
                  "%d light(s) %s" % (len(lights),
                                      tuple(round(c, 4) for c in lights[0].location)
                                      if lights else "-"))

    # 正規化ボックスが全ジオメトリを含むこと（Arm B）
    # W7：frame 1 だけ見ていると Target が動いた先での逸脱を取り逃がす。全フレーム見る。
    xs, ys, zs = [], [], []
    _frame_back = scene.frame_current
    for _f in range(1, S.FRAMES + 1):
        scene.frame_set(_f)
        bpy.context.view_layer.update()
        for o in objs:
            for vt in o.data.vertices:
                w = o.matrix_world @ vt.co
                xs.append(w.x); ys.append(w.y); zs.append(w.z)
    scene.frame_set(_frame_back)
    bpy.context.view_layer.update()
    # メッシュ頂点は float32 で保存される。-1.60 は -1.6000000238 になるため
    # float64 リテラルとの比較に 1e-9 は細かすぎる（この桁で分解能は約 1e-7）。
    TOL_F32 = 1e-5
    inside = (S.NORM_BOX["x"][0] - TOL_F32 <= min(xs) and max(xs) <= S.NORM_BOX["x"][1] + TOL_F32
              and S.NORM_BOX["y"][0] - TOL_F32 <= min(ys) and max(ys) <= S.NORM_BOX["y"][1] + TOL_F32
              and S.NORM_BOX["z"][0] - TOL_F32 <= min(zs) and max(zs) <= S.NORM_BOX["z"][1] + TOL_F32)
    ok_all &= rec("geometry_inside_norm_box", inside,
                  "X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]"
                  % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))

    return ok_all


# ---------------------------------------------------------------- main
def main():
    out, variant = parse_out()
    print("variant : %s" % variant)
    probe()

    how = reset_scene()
    print("scene reset : %s" % how)

    scene = bpy.context.scene
    scene.render.resolution_x = S.RENDER["resolution_x"]
    scene.render.resolution_y = S.RENDER["resolution_y"]
    scene.render.resolution_percentage = S.RENDER["resolution_percentage"]
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.frame_start = 1
    scene.frame_end = S.FRAMES

    objs = []
    for name, pidx, cs, z0, z1, color, rough in S.OBJECTS:
        res = guarded("build_" + name,
                      lambda n=name, p=pidx, c=cs, a=z0, b=z1, col=color, r=rough:
                      make_prism(n, c, a, b, p, col, r))
        if res is None:
            continue
        obj, bad, had_bsdf = res
        objs.append(obj)
        rec("mesh_validate_" + name, not bad, "validate removed geometry" if bad else "clean")
        rec("bsdf_found_" + name, had_bsdf, "Principled BSDF")

    # Target にアニメーションを付ける（Step 2）
    tgt = bpy.data.objects.get("Target")
    if tgt is not None:
        guarded("animate_Target", lambda: animate_target(tgt, variant))
        scene.frame_set(1)
        print("Target keyframed (%s) : frame1 location = (%.4f, %.4f, %.4f)"
              % (variant, tgt.location.x, tgt.location.y, tgt.location.z))

    cam = guarded("build_camera", make_camera)
    guarded("build_light", make_light)

    # ---- 重要 ----
    # Blender は matrix_world を遅延評価する。location / rotation_euler を
    # 代入した直後に matrix_world を読むと更新前（単位行列）が返る。
    # world_to_camera_view は内部で matrix_world.inverted() を使うため、
    # ここを飛ばすと投影が全部ずれる（W2b Step 1 初回の 24 FAIL の原因）。
    bpy.context.view_layer.update()
    print("view_layer.update() done (matrix_world 確定)")

    if cam is not None:
        m = cam.matrix_world
        print("camera matrix_world translation : (%.4f, %.4f, %.4f)"
              % (m.translation.x, m.translation.y, m.translation.z))
        fwd = (m.to_3x3() @ Vector((0.0, 0.0, -1.0)))
        print("camera forward (-Z local)       : (%.4f, %.4f, %.4f)"
              % (fwd.x, fwd.y, fwd.z))

    cam_ref[0] = cam

    print("=" * 68)
    print("SELF-CHECK")
    print("=" * 68)
    ok = False
    if cam is not None:
        ok = bool(guarded("self_check", lambda: self_check(scene, cam, objs)))

    # 軌道の検査（frame_set + matrix_world。fcurve は覗かない）
    if cam is not None and tgt is not None:
        res = guarded("check_trajectory",
                      lambda: check_trajectory(scene, tgt, variant))
        if res is not None:
            nbad, worst = res
            rec("trajectory_%s" % variant, nbad == 0,
                "%d/49 bad, worst du=%.4f px (tol %.3f)"
                % (nbad, worst, E.TOL_U))
        # W7：Scene C′ は制約1-4 を別経路で実測する
        if variant == "prime":
            guarded("prime_constraints",
                    lambda: check_prime_constraints(scene, tgt))
        # 静止物体が動いていないこと
        statics = [o for o in objs if o.name != "Target"]
        moved = []
        for frame in (1, 13, 24, 37, 49):
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            for o in statics:
                t = o.matrix_world.translation
                if max(abs(t.x), abs(t.y), abs(t.z)) > 1e-6:
                    moved.append((o.name, frame))
        rec("statics_never_move", not moved, "moved: %s" % (moved or "none"))
        scene.frame_set(1)

    n_pass = sum(1 for c in RESULTS["checks"] if c["ok"])
    n_fail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("=" * 68)
    print("SELF-CHECK : PASS %d / FAIL %d" % (n_pass, n_fail))
    print("=" * 68)

    os.makedirs(out, exist_ok=True)
    RESULTS["variant"] = variant
    jpath = os.path.join(out, "scene_c_selfcheck_%s.json" % variant)
    with open(jpath, "w") as f:
        json.dump(RESULTS, f, indent=2)
    blend = os.path.join(out, "scene_c_%s.blend" % variant)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend))
    print("saved : %s" % os.path.abspath(blend))
    print("saved : %s" % jpath)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
