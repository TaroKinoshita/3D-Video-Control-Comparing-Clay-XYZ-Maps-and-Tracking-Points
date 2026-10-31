"""
render_target_video.py  —  W2c-4
Target video（学習ターゲット）を split ごとに焼く。

  # 輝度を決める（1回の起動で set A / set B x energy を総当たり）
  & $BL --background --factory-startup --python render_target_video.py "--" ^
        --blend scene_c_original.blend --out tv --sweep

  # 本番
  & $BL --background --factory-startup --python render_target_video.py "--" ^
        --blend scene_c_original.blend --variant original --split R0S0 ^
        --out tv --energy 2000 --frames all

  # split 間の差の大きさを測る（Blender 不要）
  python render_target_video.py --compare --out tv --variant original

出力  tv/<split>/<variant>/tv_%04d.png   （8bit RGB。index 0..48）

【規約】
  R0 = EEVEE（taa 32 / use_raytracing False）  R1 = Cycles（512 sample / seed 固定）
  S0 = STYLE_A（.blend に埋め込み済み）        S1 = STYLE_B（材質のみ差し替え）
  光源の位置・回転・本数・種類は全 split で凍結。**強度も全 split で同一**にして
  シフトを純粋に材質由来にする。
  EXR は線形で出る（view_transform は EXR に効かない。Arm A で実測）ので
  自前で sRGB 符号化してから 8bit 量子化する。PNG は自前実装（§14）。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import numpy as np

import scene_c_spec as S
import scene_c_expect as E
import png_io

W, H = S.RENDER["resolution_x"], S.RENDER["resolution_y"]
RESULTS = {"checks": [], "frames": {}, "params": {}}


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-30s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def args():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else a[1:]

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
                variant=opt("--variant", "original"),
                split=opt("--split", "R0S0"),
                out=os.path.abspath(opt("--out", "tv")),
                frames=fl,
                energy=float(opt("--energy", "2000")),
                samples=int(opt("--samples", "0")),
                sweep=("--sweep" in a),
                compare=("--compare" in a))


A = args()


def srgb_encode(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(np.maximum(v, 1e-12), 1.0 / 2.4) - 0.055)


def quantize(rgb):
    return np.rint(srgb_encode(
        np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)) * 255.0
    ).astype(np.uint8)


# ================================================================= compare
def do_compare():
    """生成済み PNG を読んで split 間の差の大きさを測る。Blender 不要。"""
    base = os.path.join(A["out"], "R0S0", A["variant"])
    if not os.path.isdir(base):
        print("R0S0 が無い: %s" % base)
        return 1
    idxs = sorted(int(x[3:7]) for x in os.listdir(base)
                  if x.startswith("tv_") and x.endswith(".png"))
    out = {}
    print("| split | 共通フレーム | MAE(8bit) | 最大差 | mean差 |")
    for sp in S.SPLITS:
        d = os.path.join(A["out"], sp, A["variant"])
        if sp == "R0S0" or not os.path.isdir(d):
            continue
        maes, mx, dm = [], 0, []
        n = 0
        for i in idxs:
            pa = os.path.join(base, "tv_%04d.png" % i)
            pb = os.path.join(d, "tv_%04d.png" % i)
            if not os.path.exists(pb):
                continue
            xa, wa, ha = png_io.read_rgb(pa)
            xb, wb, hb = png_io.read_rgb(pb)
            a_ = np.frombuffer(xa, dtype=np.uint8).astype(np.int32)
            b_ = np.frombuffer(xb, dtype=np.uint8).astype(np.int32)
            maes.append(float(np.abs(a_ - b_).mean()))
            mx = max(mx, int(np.abs(a_ - b_).max()))
            dm.append(float(b_.mean() - a_.mean()))
            n += 1
        if n:
            out[sp] = {"frames": n, "mae": round(float(np.mean(maes)), 3),
                       "max_diff": mx, "mean_shift": round(float(np.mean(dm)), 3)}
            print("| %s | %d | %.3f | %d | %+.3f |"
                  % (sp, n, out[sp]["mae"], mx, out[sp]["mean_shift"]))
    jp = os.path.join(A["out"], "compare_%s.json" % A["variant"])
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print("saved : %s" % jp)
    print("判定：MAE が 8bit で 10 以上なら split 間の差は十分（軸が意味を持つ）")
    return 0


if A["compare"]:
    sys.exit(do_compare())

# ================================================================= Blender
import bpy


def target_z(variant, idx):
    """location.z に入れる値（W7）。

    このスクリプトは .blend を開いた直後に animation_data_clear() でキーフレームを
    捨て、location を直接代入する設計（fcurve を信用しない作法）。よって
    build_scene_c.py の Z キーフレームはここには伝わらない。同じ閉形式を
    S.dz_at() から取り直す。真実の出処が2箇所になるため、レンダ直前に
    check_position() で E.TRAJ_PRIME と突合して重複経路を検証する。

    variant "prime" 以外は 0.0（Scene C の挙動と bit 単位で不変）。
    """
    return S.dz_at(idx) if variant == "prime" else 0.0


def project_centre(scene, x, z_loc):
    """Target 幾何中心の (u, v, depth)。scene_c_expect と同じ閉形式。"""
    zc = (S.TARGET_Z[0] + S.TARGET_Z[1]) / 2.0 + z_loc
    d = -1.00 + 10.0
    k = S.K_PX_PER_UNIT / d
    return 416.0 + x * k, 240.0 - zc * k, d


def check_position(scene, tgt, variant, idx):
    """レンダ直前の Target 位置を照合値と突合する（W7）。

    元の self-check は mean / frac_255 / roundtrip しか見ておらず、
    Z が反映されていなくても全て PASS してしまう（W7 で発見）。
    ここで matrix_world を読んで E.TRAJ_PRIME / E.TRAJ_ORIGINAL と突合する。
    """
    t = tgt.matrix_world.translation
    if variant == "prime":
        row = next(r for r in E.TRAJ_PRIME if r[0] == idx)
        _i, _f, ex, ez, eu, ev = row
        u, v, _d = project_centre(scene, t.x, t.z)
        dd = (abs(t.x - ex), abs(t.z - ez), abs(u - eu), abs(v - ev))
        ok = (dd[0] <= E.TOL_X and dd[1] <= E.TOL_Z
              and dd[2] <= E.TOL_PX and dd[3] <= E.TOL_V)
        return rec("pos_idx%02d" % idx, ok,
                   "X=%+.7f z=%+.7f u=%.4f v=%.4f (want %+.7f %+.7f %.4f %.4f)"
                   % (t.x, t.z, u, v, ex, ez, eu, ev))
    table = E.TRAJ_EDITED if variant == "edited" else E.TRAJ_ORIGINAL
    row = next(r for r in table if r[0] == idx)
    _i, _f, ex, eu = row
    u, _v, _d = project_centre(scene, t.x, t.z)
    ok = abs(t.x - ex) <= E.TOL_X and abs(u - eu) <= E.TOL_PX and t.z == 0.0
    return rec("pos_idx%02d" % idx, ok,
               "X=%+.7f z=%+.7f u=%.4f (want %+.7f 0 %.4f)"
               % (t.x, t.z, u, eu, ex))


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


def apply_style(sk):
    """材質パラメータだけを差し替える。ジオメトリ・光源には触らない。"""
    st = S.STYLE[sk]
    done = {}
    for ob in bpy.data.objects:
        if ob.type != "MESH" or ob.name not in st:
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is None or getattr(m, "node_tree", None) is None:
                continue
            b = next((n for n in m.node_tree.nodes
                      if n.type == "BSDF_PRINCIPLED"), None)
            if b is None:
                continue
            c = st[ob.name]["base_color"]
            b.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1.0)
            for k, v in (("Roughness", st[ob.name]["roughness"]),
                         ("Metallic", st[ob.name]["metallic"])):
                if k in b.inputs:
                    b.inputs[k].default_value = v
            done[ob.name] = [round(x, 3) for x in c] + [st[ob.name]["roughness"]]
    return done


def setup(scene, split, energy, samples_override=0):
    rk, sk = split[:2], split[2:]
    cfg = S.RENDERER[rk]
    eng = cfg["engine"]
    scene.render.engine = eng          # enum 内省は信用しない。直接代入する
    n_samples = samples_override or cfg["samples"]
    if eng == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = n_samples
        scene.eevee.use_raytracing = False
    else:
        cy = scene.cycles
        cy.samples = n_samples
        cy.seed = 0
        for k, v in (("use_animated_seed", False),
                     ("use_adaptive_sampling", False),
                     ("use_denoising", False)):
            if hasattr(cy, k):
                setattr(cy, k, v)
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = False
    scene.compositing_node_group = None
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    ims = scene.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_mode = "RGBA"
    ims.color_depth = "32"
    ims.exr_codec = "ZIP"
    for lo in bpy.data.objects:
        if lo.type == "LIGHT":
            lo.data.energy = energy          # 全 split で同一
            lo.data.color = (1.0, 1.0, 1.0)
    st = apply_style(sk)
    return eng, n_samples, st


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
    return buf.reshape(h, w, 4)[::-1, :, :3].copy()


def enable_cycles():
    try:
        import addon_utils
        addon_utils.enable("cycles", default_set=False)
    except Exception:
        pass


def main():
    print("blend   : %s" % A["blend"])
    print("variant : %s / split %s" % (A["variant"], A["split"]))
    print("energy  : %.1f" % A["energy"])
    bpy.ops.wm.open_mainfile(filepath=A["blend"])
    enable_cycles()
    scene = bpy.context.scene
    tgt = bpy.data.objects["Target"]
    tgt.animation_data_clear()
    tmp = os.path.join(A["out"], "_tmp_tv")
    os.makedirs(A["out"], exist_ok=True)

    # ---- sweep：R0 で set A / set B x energy を総当たり ----
    if A["sweep"]:
        tgt.location = (target_x(A["variant"], 0), 0.0,
                        target_z(A["variant"], 0))
        scene.frame_set(1)
        rows = []
        print()
        print("| split | energy | mean | min | max | 0の割合 | 255の割合 | 線形max |")
        for sp in ("R0S0", "R0S1"):
            for en in (1000.0, 1500.0, 2000.0, 3000.0):
                setup(scene, sp, en)
                bpy.context.view_layer.update()
                lin = read_exr(render_frame(scene, tmp))
                g8 = quantize(lin)
                g = g8[:, :, 0].astype(np.int32)
                row = {"split": sp, "energy": en,
                       "mean": round(float(g8.mean()) / 255.0, 4),
                       "min": int(g8.min()), "max": int(g8.max()),
                       "frac_0": round(float((g8 == 0).mean()), 5),
                       "frac_255": round(float((g8 == 255).mean()), 5),
                       "lin_max": round(float(lin.max()), 4)}
                rows.append(row)
                print("| %s | %5.0f | %.4f | %3d | %3d | %.4f | %.4f | %.4f |"
                      % (sp, en, row["mean"], row["min"], row["max"],
                         row["frac_0"], row["frac_255"], row["lin_max"]))
        RESULTS["sweep"] = rows
        if os.path.exists(tmp + ".exr"):
            os.remove(tmp + ".exr")
        jp = os.path.join(A["out"], "sweep.json")
        with open(jp, "w") as f:
            json.dump(RESULTS, f, indent=2, default=str)
        print()
        print("saved : %s" % jp)
        print("判定：mean 0.30-0.60 / 255の割合 < 0.5%% / 線形max < 0.90（余裕を残す）")
        return 0

    eng, ns, st = setup(scene, A["split"], A["energy"], A["samples"])
    bpy.context.view_layer.update()
    rec("engine", scene.render.engine == eng,
        "%s（要求 %s）/ samples %d" % (scene.render.engine, eng, ns))
    rec("style_applied", len(st) == 6, "%d オブジェクトに材質を適用" % len(st))
    RESULTS["params"] = {"split": A["split"], "variant": A["variant"],
                         "energy": A["energy"], "engine": eng, "samples": ns,
                         "style": st, "encode": "srgb"}
    print("材質: %s" % st)

    vdir = os.path.join(A["out"], A["split"], A["variant"])
    os.makedirs(vdir, exist_ok=True)

    # 再現性（R1 は時間がかかるので frame 1 のみ）
    tgt.location = (target_x(A["variant"], 0), 0.0,
                    target_z(A["variant"], 0))
    scene.frame_set(1)
    bpy.context.view_layer.update()
    check_position(scene, tgt, A["variant"], 0)
    b1 = quantize(read_exr(render_frame(scene, tmp)))
    b2 = quantize(read_exr(render_frame(scene, tmp)))
    rec("reproducible", np.array_equal(b1, b2),
        "差分画素 %d" % int((b1 != b2).any(axis=2).sum()))

    for f in A["frames"]:
        idx = f - 1
        tgt.location = (target_x(A["variant"], idx), 0.0,
                        target_z(A["variant"], idx))
        scene.frame_set(f)
        bpy.context.view_layer.update()
        check_position(scene, tgt, A["variant"], idx)
        rgb = quantize(read_exr(render_frame(scene, tmp)))
        op = os.path.join(vdir, "tv_%04d.png" % idx)
        png_io.write_rgb(op, rgb, W, H)
        px, w2, h2 = png_io.read_rgb(op)
        back = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)
        fr = {"mean": round(float(back.mean()) / 255.0, 4),
              "min": int(back.min()), "max": int(back.max()),
              "frac_0": round(float((back == 0).mean()), 5),
              "frac_255": round(float((back == 255).mean()), 5),
              "roundtrip": bool(np.array_equal(back, rgb))}
        RESULTS["frames"][idx] = fr
        problems = []
        if not fr["roundtrip"]:
            problems.append("PNG の読み戻しが不一致")
        if fr["frac_255"] > 0.005:
            problems.append("値255が %.3f%%" % (fr["frac_255"] * 100))
        if not (0.20 <= fr["mean"] <= 0.70):
            problems.append("mean %.3f が [0.20,0.70] の外" % fr["mean"])
        rec("tv_f%02d(idx%02d)" % (f, idx), not problems,
            "ok mean=%.3f min=%d max=%d" % (fr["mean"], fr["min"], fr["max"])
            if not problems else " | ".join(problems[:2]))

    if os.path.exists(tmp + ".exr"):
        os.remove(tmp + ".exr")
    files = sorted(x for x in os.listdir(vdir)
                   if x.startswith("tv_") and x.endswith(".png"))
    if A["frames"] == list(range(1, S.FRAMES + 1)):
        rec("file_count_49", len(files) == 49, "%d files" % len(files))
        idxs = sorted(int(x[3:7]) for x in files)
        rec("index_0_to_48", idxs == list(range(49)),
            "min=%d max=%d" % (idxs[0], idxs[-1]))
    else:
        rec("file_count_smoke", len(files) == len(A["frames"]),
            "%d files" % len(files))

    npass = sum(1 for c in RESULTS["checks"] if c["ok"])
    nfail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("=" * 66)
    print("SELF-CHECK : PASS %d / FAIL %d" % (npass, nfail))
    print("=" * 66)
    jp = os.path.join(A["out"], "tv_%s_%s.json" % (A["split"], A["variant"]))
    with open(jp, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("saved : %s" % jp)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
