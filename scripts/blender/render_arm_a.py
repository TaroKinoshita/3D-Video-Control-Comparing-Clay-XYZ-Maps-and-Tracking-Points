"""
render_arm_a.py  —  W2c-2
Arm A（Clay Previs）の Control Video を生成する。

  & $BL --background --factory-startup --python render_arm_a.py "--" ^
        --blend scene_c_original.blend --geo geo --variant original ^
        --out armA --frames smoke

出力  armA/<variant>/armA_%04d.png   （8bit RGB。index 0..48）

【2026-08-05 に決めたこと】
1. §18-21：**Arm A に Entity ラベルを入れない。** 全オブジェクトに単一のクレイ材質。
   Target を色で区別しない。§2 の「Targetラベルなし」をそのまま実装する。
   マスク GT は Emission ID の別レンダーで作るので pass_index は参照しない。
2. 照明は **位置・種類は §7 の凍結値を流用、強度・色は Arm A 専用の固定値。**
   位置を共有すると影の幾何が target video と一致して情報量が増える。
   強度を切り離すことで Control の全 split bit 一致を保つ。
3. `view_transform` は **AgX**（幾何パスの Raw とは違う）。Raw だと線形値が
   そのまま 8bit に落ちて真っ暗になりプレビズとして機能しない。Style からは外す。
4. 出力経路は **EXR -> 自前で round(v*255) -> 自前 PNG。**
   Blender の 8bit 保存は経由しない（§14）。AgX は EXR にも適用される
   （W2b で Position が [0,0.475] に圧縮された件で実証済み）。

【検査】
   同一フレーム2回で bit 一致（再現性が無ければ split 間 bit 一致は成立しない）
   クリッピング / 平均輝度
   **ラベル漏れ**：Target 領域と非 Target 領域が単一閾値で分離できないこと。
   分離できてしまうと Arm A が暗黙の Entity ラベルを持つことになり設計が崩れる。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import bpy
import numpy as np

import scene_c_spec as S
import scene_c_expect as E
import png_io

W, H = S.RENDER["resolution_x"], S.RENDER["resolution_y"]
RESULTS = {"checks": [], "frames": {}, "params": {}}

# Arm A 専用の凍結値（Style set からは独立）
ARMA_BASE_COLOR = (0.50, 0.50, 0.50)
ARMA_ROUGHNESS = 0.60
ARMA_METALLIC = 0.0
ARMA_SAMPLES = 32
ARMA_LABEL_SEP_MAX = 0.99      # これ以上分離できたら実質ラベル


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-32s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def args():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []

    def opt(k, d):
        return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else d

    fr = opt("--frames", "smoke")
    if fr == "all":
        fl = list(range(1, S.FRAMES + 1))
    elif fr == "smoke":
        fl = [1, 13, 24, 37, 49]
    else:
        fl = [int(x) for x in fr.split(",") if x.strip()]
    return dict(blend=os.path.abspath(opt("--blend", "scene_c_original.blend")),
                geo=os.path.abspath(opt("--geo", "geo")),
                variant=opt("--variant", "original"),
                out=os.path.abspath(opt("--out", "armA")),
                frames=fl,
                energy=float(opt("--energy", "1000")),
                world=float(opt("--world", "0.10")),
                samples=int(opt("--samples", str(ARMA_SAMPLES))),
                sweep=("--sweep" in a),
                vt=opt("--view-transform", "Standard"),
                srgb=(opt("--encode", "srgb") == "srgb"),
                raytracing=(opt("--raytracing", "0") == "1"))


A = args()


def target_z(variant, idx):
    """location.z に入れる値（W7）。

    このスクリプトは .blend を開いた直後に animation_data_clear() でキーフレームを
    捨てて location を直接代入する設計なので、build_scene_c.py の Z キーフレームは
    ここには伝わらない。同じ閉形式を S.dz_at() から取り直す
    （render_target_video.py と同一方式）。variant "prime" 以外は 0.0。
    """
    return S.dz_at(idx) if variant == "prime" else 0.0


def project_centre(x, z_loc):
    """Target 幾何中心の (u, v)。scene_c_expect と同じ閉形式。depth は 9.00 固定。"""
    zc = (S.TARGET_Z[0] + S.TARGET_Z[1]) / 2.0 + z_loc
    k = S.K_PX_PER_UNIT / 9.00
    return 416.0 + x * k, 240.0 - zc * k


def check_position(tgt, variant, idx):
    """レンダ直前の Target 位置を照合値と突合する（W7）。

    Arm A は全オブジェクトが単一のクレイ材質なので、Target の位置が間違っていても
    「それらしい絵」が出る。mean / frac_0 / frac_255 / roughness の検査は
    Z が反映されていなくても通ってしまうため、位置は別途突合する。
    """
    t = tgt.matrix_world.translation
    if variant == "prime":
        _i, _f, ex, ez, eu, ev = next(r for r in E.TRAJ_PRIME if r[0] == idx)
        u, v = project_centre(t.x, t.z)
        ok = (abs(t.x - ex) <= E.TOL_X and abs(t.z - ez) <= E.TOL_Z
              and abs(u - eu) <= E.TOL_PX and abs(v - ev) <= E.TOL_V)
        return rec("pos_idx%02d" % idx, ok,
                   "X=%+.7f z=%+.7f u=%.4f v=%.4f (want %+.7f %+.7f %.4f %.4f)"
                   % (t.x, t.z, u, v, ex, ez, eu, ev))
    table = E.TRAJ_EDITED if variant == "edited" else E.TRAJ_ORIGINAL
    _i, _f, ex, eu = next(r for r in table if r[0] == idx)
    u, _v = project_centre(t.x, t.z)
    ok = abs(t.x - ex) <= E.TOL_X and abs(u - eu) <= E.TOL_PX and t.z == 0.0
    return rec("pos_idx%02d" % idx, ok,
               "X=%+.7f z=%+.7f u=%.4f (want %+.7f 0 %.4f)"
               % (t.x, t.z, u, ex, eu))


def target_x(variant, idx):
    kfs = sorted(S.KEYFRAMES[variant])
    frame = idx + 1
    if frame <= kfs[0][0]:
        return kfs[0][1]
    if frame >= kfs[-1][0]:
        return kfs[-1][1]
    for (f0, x0), (f1, x1) in zip(kfs, kfs[1:]):
        if f0 <= frame <= f1:
            return x0 if f1 == f0 else x0 + (x1 - x0) * (frame - f0) / (f1 - f0)
    return kfs[-1][1]


def set_view_transform(scene, name):
    """enum 内省は 2 回連続で嘘をついた（['NONE'] しか返さない）ので直接代入する。"""
    scene.view_settings.view_transform = name
    return scene.view_settings.view_transform


def probe_view_transforms(scene):
    """代入を試して通るものを列挙する。"""
    ok = []
    cur = scene.view_settings.view_transform
    for nm in ("Standard", "Khronos PBR Neutral", "AgX", "Filmic",
               "AgX Log", "False Color", "Raw"):
        try:
            scene.view_settings.view_transform = nm
            if scene.view_settings.view_transform == nm:
                ok.append(nm)
        except Exception:
            pass
    scene.view_settings.view_transform = cur
    return ok


def make_clay():
    """単一のクレイ材質。全オブジェクトに同じものを割り当てる（ラベルなし）。"""
    m = bpy.data.materials.new("ArmA_Clay")
    nt = getattr(m, "node_tree", None)
    if nt is None:
        raise RuntimeError("material has no node_tree")
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        raise RuntimeError("Principled BSDF not found")
    c = ARMA_BASE_COLOR
    bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1.0)
    for k, v in (("Roughness", ARMA_ROUGHNESS), ("Metallic", ARMA_METALLIC)):
        if k in bsdf.inputs:
            bsdf.inputs[k].default_value = v
    return m


def setup(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = A["samples"]
    scene.eevee.use_raytracing = A["raytracing"]
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = False
    scene.compositing_node_group = None
    # 外観レンダーは AgX（幾何パスの Raw とは違う）
    set_view_transform(scene, A["vt"])
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    ims = scene.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_mode = "RGBA"
    ims.color_depth = "32"
    ims.exr_codec = "ZIP"

    clay = make_clay()
    n = 0
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        ob.data.materials.clear()
        ob.data.materials.append(clay)
        n += 1

    lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    for lo in lights:
        lo.data.energy = A["energy"]          # 強度は Arm A 専用
        lo.data.color = (1.0, 1.0, 1.0)
    wd = scene.world
    if wd is not None and getattr(wd, "node_tree", None) is not None:
        for nd in wd.node_tree.nodes:
            if nd.type == "BACKGROUND":
                nd.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
                nd.inputs["Strength"].default_value = A["world"]
    return n, len(lights)


TMP = None


def render_frame(scene, stem):
    scene.render.filepath = stem
    bpy.ops.render.render(write_still=True)
    return stem + ".exr"


def read_exr(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1, :, :3].copy()   # 上から下へ


def srgb_encode(v):
    """線形 -> sRGB。IEC 61966-2-1 の定義そのまま。自前で持つ（§14）。"""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(np.maximum(v, 1e-12), 1.0 / 2.4) - 0.055)


def quantize(rgb, srgb=True):
    """EXR の値は**線形**（実測：view_transform は EXR 出力に効かない）。
    動画モデルは sRGB 符号化された映像で学習されているので、
    線形のまま 8bit にすると domain shift になる。よって自前で符号化する。"""
    v = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    if srgb:
        v = srgb_encode(v)
    return np.rint(v * 255.0).astype(np.uint8)


def best_threshold_accuracy(vals_in, vals_out):
    """単一閾値での balanced accuracy の最大値。1.0 に近いほど分離している。"""
    hi, _ = np.histogram(vals_in, bins=256, range=(0, 256))
    ho, _ = np.histogram(vals_out, bins=256, range=(0, 256))
    ci, co = np.cumsum(hi), np.cumsum(ho)
    ni, no = ci[-1], co[-1]
    if ni == 0 or no == 0:
        return None
    best = 0.0
    for t in range(256):
        # t 以下を Target とみなす場合
        acc1 = 0.5 * (ci[t] / ni + (no - co[t]) / no)
        # t 以下を非 Target とみなす場合
        acc2 = 0.5 * ((ni - ci[t]) / ni + co[t] / no)
        best = max(best, acc1, acc2)
    return float(best)


def main():
    print("blend   : %s" % A["blend"])
    print("variant : %s" % A["variant"])
    print("frames  : %s" % A["frames"])
    print("energy  : %.1f / world %.3f / samples %d"
          % (A["energy"], A["world"], A["samples"]))
    RESULTS["params"] = {k: A[k] for k in
                         ("variant", "energy", "world", "samples")}
    RESULTS["params"].update(
        {"base_color": list(ARMA_BASE_COLOR), "roughness": ARMA_ROUGHNESS,
         "view_transform": A["vt"], "entity_label": "none",
         "encode": "srgb" if A["srgb"] else "linear",
         "use_raytracing": A["raytracing"]})

    bpy.ops.wm.open_mainfile(filepath=A["blend"])
    scene = bpy.context.scene
    tgt = bpy.data.objects["Target"]
    tgt.animation_data_clear()
    nmat, nlight = setup(scene)
    bpy.context.view_layer.update()
    rec("single_clay_material", len(bpy.data.materials) >= 1,
        "%d mesh に単一材質を割当 / 光源 %d" % (nmat, nlight))
    mats = {m.name for ob in bpy.data.objects if ob.type == "MESH"
            for m in ob.data.materials}
    rec("no_entity_label", mats == {"ArmA_Clay"},
        "使われている材質 %s" % sorted(mats))
    rec("view_transform", scene.view_settings.view_transform == A["vt"],
        "%s（要求 %s）" % (scene.view_settings.view_transform, A["vt"]))

    vdir = os.path.join(A["out"], A["variant"])
    os.makedirs(vdir, exist_ok=True)
    tmp = os.path.join(A["out"], "_tmp_arma")

    if A["sweep"]:
        avail = probe_view_transforms(scene)
        print("代入できた view_transform: %s" % avail)
        RESULTS["available_view_transforms"] = avail
        idp = os.path.join(A["geo"], A["variant"], "id_0001.exr")
        ids = read_exr(idp)[:, :, 0] if os.path.exists(idp) else None
        m_t = (ids == 3.0) if ids is not None else None
        tgt.location = (target_x(A["variant"], 0), 0.0,
                        target_z(A["variant"], 0))
        scene.frame_set(1)
        rows = []
        print()
        print("| RT | energy | world | 符号化 | mean | min | max | 0の割合 | 255の割合 | sep |")
        for rt in (False, True):
            scene.eevee.use_raytracing = rt
            for en in (1000.0, 1500.0, 2000.0, 3000.0):
                for wo in (0.10, 0.50):
                    for lo in bpy.data.objects:
                        if lo.type == "LIGHT":
                            lo.data.energy = en
                    wd = scene.world
                    if wd is not None and getattr(wd, "node_tree", None):
                        for nd in wd.node_tree.nodes:
                            if nd.type == "BACKGROUND":
                                nd.inputs["Strength"].default_value = wo
                    bpy.context.view_layer.update()
                    lin = read_exr(render_frame(scene, tmp))
                    for enc in (True,):
                        g8 = quantize(lin, enc)
                        g = g8[:, :, 0].astype(np.int32)
                        sep = (best_threshold_accuracy(g[m_t], g[~m_t])
                               if m_t is not None and m_t.sum() else None)
                        row = {"rt": rt, "energy": en, "world": wo,
                               "encode": "srgb" if enc else "linear",
                               "mean": round(float(g.mean()) / 255.0, 4),
                               "min": int(g.min()), "max": int(g.max()),
                               "frac_0": round(float((g == 0).mean()), 5),
                               "frac_255": round(float((g == 255).mean()), 5),
                               "lin_max": round(float(lin.max()), 4),
                               "sep": round(sep, 4) if sep is not None else None}
                        rows.append(row)
                        print("| %s | %5.0f | %.2f | %-6s | %.4f | %3d | %3d | %.4f | %.4f | %s |"
                              % ("on " if rt else "off", en, wo, row["encode"],
                                 row["mean"], row["min"], row["max"],
                                 row["frac_0"], row["frac_255"],
                                 "%.4f" % sep if sep is not None else "-"))
        # raytracing on/off それぞれで再現性を確認
        for rt in (False, True):
            scene.eevee.use_raytracing = rt
            for lo in bpy.data.objects:
                if lo.type == "LIGHT":
                    lo.data.energy = 1500.0
            bpy.context.view_layer.update()
            b1 = quantize(read_exr(render_frame(scene, tmp)), True)
            b2 = quantize(read_exr(render_frame(scene, tmp)), True)
            rec("reproducible_rt_%s" % ("on" if rt else "off"),
                np.array_equal(b1, b2),
                "差分画素 %d" % int((b1 != b2).any(axis=2).sum()))
        RESULTS["sweep"] = rows
        if os.path.exists(tmp + ".exr"):
            os.remove(tmp + ".exr")
        jp = os.path.join(A["out"], "render_arm_a_sweep.json")
        with open(jp, "w") as fh:
            json.dump(RESULTS, fh, indent=2, default=str)
        print()
        print("saved : %s" % jp)
        print("判定：mean 0.15-0.75 / 0の割合 < 3%% / 255の割合 < 1%% / sep < 0.99 / 再現性 差分0")
        return 0

    # ---- 再現性：frame 1 を2回焼いて bit 一致を見る ----
    tgt.location = (target_x(A["variant"], 0), 0.0,
                    target_z(A["variant"], 0))
    scene.frame_set(1)
    bpy.context.view_layer.update()
    check_position(tgt, A["variant"], 0)
    a1 = quantize(read_exr(render_frame(scene, tmp)), A["srgb"])
    a2 = quantize(read_exr(render_frame(scene, tmp)), A["srgb"])
    rec("reproducible", np.array_equal(a1, a2),
        "同一フレーム2回で bit 一致 / 差分画素 %d"
        % int((a1 != a2).any(axis=2).sum()))

    stats = []
    for f in A["frames"]:
        idx = f - 1
        tgt.location = (target_x(A["variant"], idx), 0.0,
                        target_z(A["variant"], idx))
        scene.frame_set(f)
        bpy.context.view_layer.update()
        check_position(tgt, A["variant"], idx)
        rgb = quantize(read_exr(render_frame(scene, tmp)), A["srgb"])

        op = os.path.join(vdir, "armA_%04d.png" % idx)
        png_io.write_rgb(op, rgb, W, H)
        px, w2, h2 = png_io.read_rgb(op)
        back = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)

        g = back[:, :, 0].astype(np.int32)
        fr = {"mean": float(g.mean()) / 255.0,
              "min": int(g.min()), "max": int(g.max()),
              "frac_0": float((g == 0).mean()),
              "frac_255": float((g == 255).mean()),
              "roundtrip": bool(np.array_equal(back, rgb))}

        # ---- ラベル漏れの検査 ----
        idp = os.path.join(A["geo"], A["variant"], "id_%04d.exr" % f)
        if os.path.exists(idp):
            ids = read_exr(idp)[:, :, 0]
            m_t = (ids == 3.0)
            if m_t.sum() > 0:
                sep = best_threshold_accuracy(g[m_t], g[~m_t])
                fr["target_px"] = int(m_t.sum())
                fr["label_separability"] = sep
                fr["mean_target"] = float(g[m_t].mean()) / 255.0
                fr["mean_other"] = float(g[~m_t].mean()) / 255.0
            else:
                fr["target_px"] = 0
        stats.append((f, fr))
        RESULTS["frames"][idx] = fr

        problems = []
        if not fr["roundtrip"]:
            problems.append("PNG の読み戻しが不一致")
        if fr["frac_0"] > 0.03:
            problems.append("値0の画素が %.2f%%" % (fr["frac_0"] * 100))
        if fr["frac_255"] > 0.01:
            problems.append("値255の画素が %.2f%%" % (fr["frac_255"] * 100))
        if not (0.15 <= fr["mean"] <= 0.75):
            problems.append("平均輝度 %.3f が [0.15,0.75] の外" % fr["mean"])
        s = fr.get("label_separability")
        if s is not None and s > ARMA_LABEL_SEP_MAX:
            problems.append("Target が単一閾値で分離できる（%.4f）" % s)
        rec("armA_f%02d(idx%02d)" % (f, idx), not problems,
            ("ok mean=%.3f min=%d max=%d sep=%s"
             % (fr["mean"], fr["min"], fr["max"],
                "%.4f" % s if s is not None else "-"))
            if not problems else " | ".join(problems[:3]))

    if os.path.exists(tmp + ".exr"):
        os.remove(tmp + ".exr")

    files = sorted(x for x in os.listdir(vdir)
                   if x.startswith("armA_") and x.endswith(".png"))
    if A["frames"] == list(range(1, S.FRAMES + 1)):
        rec("file_count_49", len(files) == 49, "%d files" % len(files))
        idxs = sorted(int(x[5:9]) for x in files)
        rec("index_0_to_48", idxs == list(range(49)),
            "min=%d max=%d" % (idxs[0], idxs[-1]))
    else:
        rec("file_count_smoke", len(files) == len(A["frames"]),
            "%d files" % len(files))

    npass = sum(1 for c in RESULTS["checks"] if c["ok"])
    nfail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("=" * 68)
    print("SELF-CHECK : PASS %d / FAIL %d" % (npass, nfail))
    print("=" * 68)
    print("輝度の要約（調整の判断用）")
    for f, fr in stats:
        print("  f%02d mean %.3f / min %3d / max %3d / 0が%.3f%% / 255が%.3f%% / sep %s"
              % (f, fr["mean"], fr["min"], fr["max"], fr["frac_0"] * 100,
                 fr["frac_255"] * 100,
                 ("%.4f" % fr["label_separability"])
                 if fr.get("label_separability") is not None else "-"))
    jp = os.path.join(A["out"], "render_arm_a_%s.json" % A["variant"])
    with open(jp, "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print("saved : %s" % jp)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
