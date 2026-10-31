"""
export_geometry.py  —  W2b Step 3b
Scene C の幾何パス（Position / Depth / ID マスク）を書き出す。

  & $BL --background --factory-startup --python export_geometry.py "--" ^
        --blend scene_c_original.blend --out geo --variant original --frames smoke

引数
  --blend   入力 .blend
  --variant original | edited （出力先とSELF-CHECKの照合表に使う）
  --out     出力ルート（既定 geo）
  --frames  all | smoke（1,13,24,37,49 のみ）| カンマ区切り

【実測で確定した規約。推測で変えるとサイレントに壊れる】
  1. Group Output のインターフェースソケットは **必ず NodeSocketColor**。
     Position(VECTOR) に NodeSocketVector、Depth(VALUE) に NodeSocketFloat を
     使うと links.new は is_valid=True を返すが **コンポジタが無視され
     Combined がそのまま書かれる**。エラーも警告も出ない。
  2. interface.new_socket を呼ばずに仮想ソケットへリンクすると is_valid=False。
     やはり Combined が書かれる。
  3. view_settings.view_transform = 'Raw' が必須。既定の AgX を通ると
     値が [0, 0.475] に圧縮される。32bit EXR でも起きる。
  4. この故障は一意値の個数・値域・ファイルサイズでは検出できない。
     **既知座標の画素値を手計算と突合する以外に検出手段がない。**
"""
import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import bpy

import scene_c_spec as S
import geo_expect as G

try:
    import numpy as np
except Exception:
    np = None

RESULTS = {"checks": [], "cycles": {}, "frames": {}}
K = S.K_PX_PER_UNIT
W, H = S.RENDER["resolution_x"], S.RENDER["resolution_y"]


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-34s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def args():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []

    def opt(k, d):
        return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else d

    blend = opt("--blend", "scene_c_original.blend")
    variant = opt("--variant", "original")
    out = opt("--out", "geo")
    frames = opt("--frames", "smoke")
    if frames == "all":
        fl = list(range(1, S.FRAMES + 1))
    elif frames == "smoke":
        fl = [1, 13, 24, 37, 49]
    else:
        fl = [int(x) for x in frames.split(",") if x.strip()]
    return os.path.abspath(blend), variant, os.path.abspath(out), fl


BLEND, VARIANT, OUTROOT, FRAMES = args()


# ---------------------------------------------------------------- 配線
def wire(sc, socket_name):
    """RLayers.<socket> -> GroupOutput。出力ソケット型は必ず NodeSocketColor。"""
    ng = bpy.data.node_groups.new("GEO_" + socket_name, "CompositorNodeTree")
    ng.interface.new_socket(name="Out", in_out="OUTPUT",
                            socket_type="NodeSocketColor")
    sc.compositing_node_group = ng
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = sc
    go = ng.nodes.new("NodeGroupOutput")
    src = next((s for s in rl.outputs if s.name == socket_name), None)
    if src is None:
        raise RuntimeError("socket %r not found; have %s"
                           % (socket_name, [s.name for s in rl.outputs]))
    lk = ng.links.new(src, go.inputs[0])
    if not getattr(lk, "is_valid", False):
        raise RuntimeError("link %s -> Out is not valid" % socket_name)
    return ng


def setup_common(sc):
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = 1
    sc.eevee.use_raytracing = False
    sc.render.filter_size = 0.01
    sc.render.use_motion_blur = False
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.resolution_percentage = 100
    sc.render.use_compositing = True
    # カラーマネジメントを素通しにする（規約3）
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    ims = sc.render.image_settings
    ims.file_format = "OPEN_EXR"
    ims.color_mode = "RGBA"
    ims.color_depth = "32"
    ims.exr_codec = "ZIP"          # 可逆。SELF-CHECK で厳密性を確認する
    vl = bpy.context.view_layer
    for a in ("use_pass_combined", "use_pass_z", "use_pass_position"):
        setattr(vl, a, True)


def to_emission_ids(sc):
    """全オブジェクトを Emission (Strength = pass_index) に差し替える。
    Color は白固定。値は 32bit EXR に厳密に保存されることを実測済み。"""
    swapped = {}
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        m = bpy.data.materials.new("EMID_" + ob.name)
        nt = m.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        em.inputs["Strength"].default_value = float(ob.pass_index)
        o = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
        ob.data.materials.clear()
        ob.data.materials.append(m)
        swapped[ob.name] = ob.pass_index
    # 背景を厳密に 0 にする
    sc.render.film_transparent = True
    try:
        wd = sc.world
        if wd is not None and wd.node_tree is not None:
            for n in wd.node_tree.nodes:
                if n.type == "BACKGROUND":
                    n.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                    n.inputs["Strength"].default_value = 0.0
    except Exception:
        pass
    return swapped


def render_to(sc, path_stem):
    sc.render.filepath = path_stem
    bpy.ops.render.render(write_still=True)
    return path_stem + ".exr"


# ---------------------------------------------------------------- 読み戻し
def load_buf(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    w, h = img.size
    if np is not None:
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
    else:
        buf = list(img.pixels)
    return buf, w, h, img


def px(buf, w, i, j):
    k = (j * w + i) * 4
    return float(buf[k]), float(buf[k + 1]), float(buf[k + 2])


def _occluder_u():
    """遮蔽板の u 範囲（閉形式）。spec の断面から出す。"""
    for name, _p, cs, _z0, _z1, _c, _r in S.OBJECTS:
        if name == "Occluder":
            us = [416.0 + x * K / (y + 10.0) for x, y in cs]
            return min(us), max(us)
    raise RuntimeError("Occluder not in spec")


OCC_L, OCC_R = _occluder_u()


def target_x(idx, variant):
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


def silhouette_u(idx, variant="original"):
    """裸のシルエット u 範囲（遮蔽を考慮しない）。"""
    xc = target_x(idx, variant)
    us = [416.0 + (xc + x) * K / (y + 10.0) for x, y in S.TARGET_CS]
    return min(us), max(us)


def prime_band_px(idx):
    """Scene C' で Target が遮蔽板の上端より上に出ている帯の高さ（px。W7）。

    <= 0 なら板の下（従来の 1D 判定が正解）。
    > 0 なら Target の一部が板の上に出ており、Target は X-Y 断面を Z に押し出した
    プリズムなので **どの Z スライスでも u 範囲は同一** → 可視シルエットは u 全幅。
    照合値は G.PRIME_TARGET_V_TOP（scene_c_prime_traj.json 由来の凍結リテラル）。
    """
    _i, _v_top, band = G.PRIME_TARGET_V_TOP[idx]
    return band


def visible_u(idx, variant="original"):
    """遮蔽板を差し引いた**可視**シルエットの u 範囲。
    None なら完全遮蔽。ラスタライズ結果と突合するのはこちら。

    W7：variant "prime" は Target が浮いて遮蔽板の上端を越えるため、u だけの
    1次元判定では 24 フレーム（index 6-29）で誤った期待値を出す。板の上に出て
    いる場合は全幅を返す。帯がサブピクセルの境界フレームは呼び出し側で両方を許容。
    """
    a, b = silhouette_u(idx, variant)
    if variant == "prime" and prime_band_px(idx) > 0.0:
        return (a, b)
    if a >= OCC_L - 1e-9 and b <= OCC_R + 1e-9:
        return None
    if b <= OCC_L or a >= OCC_R:
        return (a, b)
    if a < OCC_L:
        return (a, min(b, OCC_L))
    return (max(a, OCC_R), b)


def visible_u_1d(idx, variant="original"):
    """板の上下を考慮しない従来の 1D 判定。境界フレームの代替解として使う（W7）。"""
    a, b = silhouette_u(idx, variant)
    if a >= OCC_L - 1e-9 and b <= OCC_R + 1e-9:
        return None
    if b <= OCC_L or a >= OCC_R:
        return (a, b)
    if a < OCC_L:
        return (a, min(b, OCC_L))
    return (max(a, OCC_R), b)


# ---------------------------------------------------------------- Cycles
def cycles_check():
    c = RESULTS["cycles"]
    c["engines_enum"] = [e.identifier for e in
                         bpy.context.scene.render.bl_rna
                         .properties["engine"].enum_items]
    try:
        import addon_utils
        addon_utils.enable("cycles", default_set=False)
        c["addon_utils_enable"] = "ok"
    except Exception as exc:
        c["addon_utils_enable"] = "EXCEPTION %s" % exc
    # enum 内省は信用できない（view_transform で偽 ['NONE'] を返した実績）。直接代入する。
    try:
        bpy.context.scene.render.engine = "CYCLES"
        c["direct_assign"] = "OK -> %s" % bpy.context.scene.render.engine
    except Exception as exc:
        c["direct_assign"] = "EXCEPTION %s: %s" % (type(exc).__name__, exc)
    c["has_CyclesRender"] = hasattr(bpy.types, "CyclesRender")
    try:
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        pass
    print("-" * 70)
    print("CYCLES 切り分け")
    for k, v in c.items():
        print("  %-24s %s" % (k, v))
    print("-" * 70)


# ---------------------------------------------------------------- main
def main():
    print("blend   : %s" % BLEND)
    print("variant : %s" % VARIANT)
    print("out     : %s" % OUTROOT)
    print("frames  : %s" % FRAMES)
    print("numpy   : %s" % ("yes" if np is not None else "NO (遅い経路)"))

    if not os.path.exists(BLEND):
        raise SystemExit("blend が無い: %s" % BLEND)
    bpy.ops.wm.open_mainfile(filepath=BLEND)
    sc = bpy.context.scene
    setup_common(sc)
    bpy.context.view_layer.update()

    cycles_check()
    setup_common(sc)   # cycles_check でエンジンを触ったので再適用

    vdir = os.path.join(OUTROOT, VARIANT)
    os.makedirs(vdir, exist_ok=True)

    # ---- パス1: Position, パス2: Depth（同じ配線規約） ----
    for pass_name, socket in (("position", "Position"), ("depth", "Depth")):
        try:
            wire(sc, socket)
        except Exception as exc:
            rec("wire_" + pass_name, False, str(exc))
            continue
        rec("wire_" + pass_name, True, "%s -> NodeSocketColor" % socket)
        for f in FRAMES:
            sc.frame_set(f)
            bpy.context.view_layer.update()
            stem = os.path.join(vdir, "%s_%04d" % (pass_name, f))
            render_to(sc, stem)

    # ---- パス3: ID マスク（Emission。材質を破壊するので最後） ----
    swapped = to_emission_ids(sc)
    RESULTS["swapped_pass_index"] = swapped
    try:
        wire(sc, "Image")
        rec("wire_id", True, "Image -> NodeSocketColor / %d obj" % len(swapped))
        for f in FRAMES:
            sc.frame_set(f)
            bpy.context.view_layer.update()
            render_to(sc, os.path.join(vdir, "id_%04d" % f))
    except Exception as exc:
        rec("wire_id", False, str(exc))

    # ================================================== SELF-CHECK
    print("=" * 70)
    print("SELF-CHECK")
    print("=" * 70)

    for f in FRAMES:
        idx = f - 1
        fr = {}
        pos_p = os.path.join(vdir, "position_%04d.exr" % f)
        dep_p = os.path.join(vdir, "depth_%04d.exr" % f)
        id_p = os.path.join(vdir, "id_%04d.exr" % f)

        # Position
        if os.path.exists(pos_p):
            buf, w, h, _ = load_buf(pos_p)
            bad = []
            for lbl, i, j, d, pi, X, Y, Z, inv in G.PROBE_PX:
                if not inv and idx != 0:
                    continue
                r, g, b = px(buf, w, i, j)
                e = max(abs(r - X), abs(g - Y), abs(b - Z))
                if e > G.TOL_POSITION:
                    bad.append("%s got(%.5f,%.5f,%.5f) want(%.5f,%.5f,%.5f) err=%.5f"
                               % (lbl, r, g, b, X, Y, Z, e))
            fr["position_bad"] = bad
            rec("position_f%02d" % f, not bad,
                "ok" if not bad else " | ".join(bad[:3]))
        else:
            rec("position_f%02d" % f, False, "file missing")

        # Depth
        if os.path.exists(dep_p):
            buf, w, h, _ = load_buf(dep_p)
            bad = []
            for lbl, i, j, d, pi, X, Y, Z, inv in G.PROBE_PX:
                if not inv and idx != 0:
                    continue
                r, g, b = px(buf, w, i, j)
                if abs(r - d) > G.TOL_DEPTH:
                    bad.append("%s got=%.5f want=%.5f" % (lbl, r, d))
            fr["depth_bad"] = bad
            rec("depth_f%02d" % f, not bad,
                "ok" if not bad else " | ".join(bad[:3]))
        else:
            rec("depth_f%02d" % f, False, "file missing")

        # ID マスク
        if os.path.exists(id_p):
            buf, w, h, _ = load_buf(id_p)
            bad = []
            for lbl, i, j, d, pi, X, Y, Z, inv in G.PROBE_PX:
                if not inv and idx != 0:
                    continue
                r, _g, _b = px(buf, w, i, j)
                if abs(r - float(pi)) > G.TOL_ID:
                    bad.append("%s got=%.5f want=%d" % (lbl, r, pi))
            # 値集合が 0-6 に収まっているか
            if np is not None:
                arr = buf.reshape(-1, 4)[:, 0]
                uniq = sorted(set(round(float(v), 6) for v in np.unique(arr)))
                cols = np.where(arr.reshape(h, w) == 3.0)[1]
                tmin = float(cols.min()) + 0.5 if cols.size else None
                tmax = float(cols.max()) + 0.5 if cols.size else None
                npix = int(cols.size)
            else:
                vals = [buf[k] for k in range(0, len(buf), 4)]
                uniq = sorted({round(v, 6) for v in vals})
                cells = [(k // 4) for k, v in enumerate(vals) if v == 3.0]
                cols = [c % w for c in cells]
                tmin = min(cols) + 0.5 if cols else None
                tmax = max(cols) + 0.5 if cols else None
                npix = len(cols)
            fr["id_unique"] = uniq
            fr["target_px"] = npix
            fr["target_u_range"] = [tmin, tmax]
            extra = [v for v in uniq if v not in G.VALID_IDS]
            if extra:
                bad.append("ID 以外の値 %s" % extra[:6])
            # 可視シルエット（遮蔽板を差し引いたもの）と突合
            exp = visible_u(idx, VARIANT)
            fr["target_u_expect"] = None if exp is None else [round(exp[0], 3),
                                                              round(exp[1], 3)]
            fr["fully_occluded_expected"] = exp is None
            # W7：prime の境界フレーム（板上の帯がサブピクセル）は 1D 判定も許容する
            alts = [exp]
            if VARIANT == "prime":
                band = prime_band_px(idx)
                fr["prime_band_px"] = round(band, 4)
                if 0.0 < band < G.PRIME_BAND_SUBPIXEL_PX:
                    alt = visible_u_1d(idx, VARIANT)
                    fr["prime_boundary_frame"] = True
                    if alt != exp:
                        alts.append(alt)

            def _match(e):
                if e is None:
                    return tmin is None
                if tmin is None:
                    return False
                return max(abs(tmin - e[0]), abs(tmax - e[1])) <= G.TOL_SILHOUETTE_PX

            if not any(_match(e) for e in alts):
                if exp is None:
                    bad.append("完全遮蔽のはずが Target %d px 見えている" % npix)
                elif tmin is None:
                    bad.append("Target 画素が0（可視のはず u[%.1f,%.1f]）" % exp)
                else:
                    d0, d1 = abs(tmin - exp[0]), abs(tmax - exp[1])
                    fr["target_u_err"] = [round(d0, 3), round(d1, 3)]
                    bad.append("visible u got[%.1f,%.1f] want[%.1f,%.1f]%s"
                               % (tmin, tmax, exp[0], exp[1],
                                  " (境界の代替解 %s も不一致)" % (alts[1],)
                                  if len(alts) > 1 else ""))
            elif exp is not None and tmin is not None:
                fr["target_u_err"] = [round(abs(tmin - exp[0]), 3),
                                      round(abs(tmax - exp[1]), 3)]

            # W7：Z が geo に載っているかの唯一の検査。
            # ID マスクの最上行 -> v に変換して凍結リテラルと突合する。
            # Target 上端は他のオブジェクトに隠されないため遮蔽状態に無依存。
            if VARIANT == "prime" and np is not None:
                rowsj = np.where(arr.reshape(h, w) == 3.0)[0]
                if rowsj.size:
                    # pixels バッファ座標：v = 480 - (j + 0.5)。j 最大が画面最上端
                    v_meas = float(H) - (float(rowsj.max()) + 0.5)
                    _i, v_exp, _band = G.PRIME_TARGET_V_TOP[idx]
                    dv = abs(v_meas - v_exp)
                    fr["target_v_top"] = round(v_meas, 3)
                    fr["target_v_top_expect"] = v_exp
                    fr["target_v_top_err"] = round(dv, 3)
                    if dv > G.TOL_SILHOUETTE_PX:
                        bad.append("Target 上端 v got=%.2f want=%.2f err=%.2f"
                                   % (v_meas, v_exp, dv))
                else:
                    fr["target_v_top"] = None
            if not bad:
                if exp is None:
                    detail = "ok 完全遮蔽 uniq=%s" % uniq
                else:
                    detail = ("ok uniq=%s target=%dpx u[%.1f,%.1f] err=%.2f/%.2f"
                              % (uniq, npix, tmin, tmax,
                                 abs(tmin - exp[0]), abs(tmax - exp[1])))
            else:
                detail = " | ".join(bad[:3])
            rec("id_f%02d" % f, not bad, detail)
        else:
            rec("id_f%02d" % f, False, "file missing")

        RESULTS["frames"][f] = fr

    npass = sum(1 for c in RESULTS["checks"] if c["ok"])
    nfail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("=" * 70)
    print("SELF-CHECK : PASS %d / FAIL %d" % (npass, nfail))
    print("=" * 70)

    jp = os.path.join(OUTROOT, "export_geometry_%s.json" % VARIANT)
    RESULTS["variant"] = VARIANT
    RESULTS["frame_list"] = FRAMES
    with open(jp, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("saved : %s" % jp)
    print("files : %s" % vdir)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
