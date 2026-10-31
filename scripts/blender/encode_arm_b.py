"""
encode_arm_b.py  —  W2c-1
Arm B（Object-Aware Dense World-XYZ）の Control Video を生成する。

  & $BL --background --factory-startup --python encode_arm_b.py "--" ^
        --geo geo --variant original --out armB --frames smoke

入力  geo/<variant>/position_%04d.exr   （world 座標。frame 1..49）
      geo/<variant>/id_%04d.exr         （pass_index。Emission ID）
出力  armB/<variant>/armB_%04d.png      （8bit RGB。**index 0..48**）

Blender 内で走らせる理由は EXR を読むためだけ。エンコードと PNG 書き出しは
numpy + 自前 png_io で行う（protocol §14：Blender の画像保存を経由しない）。

frame（1..49）と index（0..48）を取り違えるのが既知バグ#3 の系統なので、
入力は frame 番号、出力は index 番号で名前を付け、対応を毎回 assert する。
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else os.getcwd())
import bpy
import numpy as np

import scene_c_spec as S
import armb_expect as A
import armb_encode
import png_io

W, H = S.RENDER["resolution_x"], S.RENDER["resolution_y"]
NB = S.NORM_BOX
RESULTS = {"checks": [], "frames": {}}


def rec(name, ok, detail=""):
    RESULTS["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print("%-5s %-30s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def args():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []

    def opt(k, d):
        return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else d

    geo = os.path.abspath(opt("--geo", "geo"))
    variant = opt("--variant", "original")
    out = os.path.abspath(opt("--out", "armB"))
    fr = opt("--frames", "smoke")
    if fr == "all":
        fl = list(range(1, A.N_FRAMES + 1))
    elif fr == "smoke":
        fl = [1, 13, 24, 37, 49]
    else:
        fl = [int(x) for x in fr.split(",") if x.strip()]
    return geo, variant, out, fl


GEO, VARIANT, OUT, FRAMES = args()


def load_rgb(path):
    """EXR の RGB を (H, W, 3) float32 で返す。行は PNG 走査順（上が先）。"""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        img.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    w, h = img.size
    if (w, h) != (W, H):
        raise RuntimeError("%s size %dx%d != %dx%d" % (path, w, h, W, H))
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    # Blender の pixels は下から上。PNG は上から下なので反転する
    return buf.reshape(h, w, 4)[::-1, :, :3].copy()


def main():
    print("geo     : %s" % GEO)
    print("variant : %s" % VARIANT)
    print("out     : %s" % OUT)
    print("frames  : %s" % FRAMES)
    print("norm box: X%s Y%s Z%s" % (NB["x"], NB["y"], NB["z"]))

    vdir = os.path.join(OUT, VARIANT)
    os.makedirs(vdir, exist_ok=True)
    gdir = os.path.join(GEO, VARIANT)

    max_excess = 0.0
    written = []
    for f in FRAMES:
        idx = f - 1
        pp = os.path.join(gdir, "position_%04d.exr" % f)
        ip = os.path.join(gdir, "id_%04d.exr" % f)
        if not (os.path.exists(pp) and os.path.exists(ip)):
            rec("input_f%02d" % f, False, "missing %s or %s"
                % (os.path.basename(pp), os.path.basename(ip)))
            continue
        pos = load_rgb(pp)
        ids = load_rgb(ip)[:, :, 0]

        fr = {}
        # ID パスが整数値のみであること（AA が入っていない証拠）
        uniq_id = np.unique(ids)
        non_int = [float(u) for u in uniq_id if abs(u - round(float(u))) > 1e-6]
        fr["id_unique"] = [float(u) for u in uniq_id]
        if non_int:
            rec("id_integral_f%02d" % f, False, "非整数の ID %s" % non_int[:5])
        # NaN / Inf
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(ids))):
            rec("finite_f%02d" % f, False, "NaN / Inf が存在する")

        rgb, excess = armb_encode.encode(pos, ids, NB)
        max_excess = max(max_excess, excess)
        fr["q_excess"] = excess

        op = os.path.join(vdir, "armB_%04d.png" % idx)
        png_io.write_rgb(op, rgb, W, H)
        written.append(op)

        # 自前で書いて自前で読む（filter 0 なので往復に変換が無い）
        px, w2, h2 = png_io.read_rgb(op)
        back = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)
        fr["roundtrip_exact"] = bool(np.array_equal(back, rgb))

        # 帯と予約帯
        vals = np.unique(back)
        res_hit = [int(x) for x in vals if A.RESERVED[0] <= x <= A.RESERVED[1]]
        fr["reserved_hits"] = res_hit
        fr["value_min"] = int(vals.min())
        fr["value_max"] = int(vals.max())

        # 照合画素
        bad = []
        for lbl, i, jp, pi, R, G, B, inv in A.PROBE_RGB:
            if not inv and idx != 0:
                continue
            got = tuple(int(x) for x in back[jp, i])
            if max(abs(got[0] - R), abs(got[1] - G), abs(got[2] - B)) > A.TOL_LEVEL:
                bad.append("%s got%s want(%d,%d,%d)" % (lbl, got, R, G, B))
            # 帯の所属も検査
            band = A.BAND_TARGET if pi == 3 else A.BAND_BACKGROUND
            if not all(band[0] <= c <= band[1] for c in got):
                bad.append("%s %s が帯 %s の外" % (lbl, got, band))
        fr["probe_bad"] = bad

        # W7：Scene C' の Z 検証。
        # 既存の Target 照合（PROBE_RGB の target_f1）は index 0 限定であり、
        # dZ>0 のフレームで B チャンネルが一度も照合されない。ID パスで Target
        # 画素を選び、その B の最大値（= world Z の最大 = Target 上端）を突合する。
        if VARIANT == "prime":
            m_t = (ids == 3.0)
            n_t = int(m_t.sum())
            fr["target_px"] = n_t
            _i, b_exp = A.PRIME_TARGET_B_MAX[idx]
            if n_t > 0:
                b_max = int(back[:, :, 2][m_t].max())
                fr["target_b_max"] = b_max
                fr["target_b_max_expect"] = b_exp
                fr["target_b_max_err"] = abs(b_max - b_exp)
                if abs(b_max - b_exp) > A.PRIME_TOL_ZLEVEL:
                    bad.append("Target の B 最大 got=%d want=%d err=%d"
                               % (b_max, b_exp, abs(b_max - b_exp)))
                lo, hi = A.BAND_TARGET
                b_min = int(back[:, :, 2][m_t].min())
                fr["target_b_min"] = b_min
                if not (lo <= b_min and b_max <= hi):
                    bad.append("Target の B [%d,%d] が帯 %s の外"
                               % (b_min, b_max, A.BAND_TARGET))
            elif idx not in A.PRIME_B_SCALARS["occluded_indices"]:
                bad.append("Target 画素が0（完全遮蔽区間 %s 以外）"
                           % A.PRIME_B_SCALARS["occluded_indices"])

        problems = list(bad)
        if res_hit:
            problems.append("予約帯に %s が存在" % res_hit)
        if not fr["roundtrip_exact"]:
            problems.append("PNG の読み戻しが不一致")
        if non_int:
            problems.append("ID が非整数")
        ok = not problems
        detail = ("ok min=%d max=%d q_excess=%.2e"
                  % (vals.min(), vals.max(), excess)) if ok \
            else " | ".join(problems[:3])
        rec("armB_f%02d(idx%02d)" % (f, idx), ok, detail)
        RESULTS["frames"][idx] = fr

    # ---- 全体の検査 ----
    rec("q_excess_within_box", max_excess <= A.MAX_Q_EXCESS,
        "max excess %.3e (許容 %.1e)" % (max_excess, A.MAX_Q_EXCESS))

    files = sorted(x for x in os.listdir(vdir)
                   if x.startswith("armB_") and x.endswith(".png"))
    RESULTS["file_count"] = len(files)
    if FRAMES == list(range(1, A.N_FRAMES + 1)):
        rec("file_count_49", len(files) == 49, "%d files" % len(files))
        idxs = sorted(int(x[5:9]) for x in files)
        rec("index_0_to_48", idxs == list(range(49)),
            "min=%d max=%d n=%d" % (min(idxs), max(idxs), len(idxs)))
    else:
        rec("file_count_smoke", len(files) == len(FRAMES),
            "%d files (smoke)" % len(files))

    npass = sum(1 for c in RESULTS["checks"] if c["ok"])
    nfail = sum(1 for c in RESULTS["checks"] if not c["ok"])
    print("=" * 66)
    print("SELF-CHECK : PASS %d / FAIL %d" % (npass, nfail))
    print("=" * 66)

    RESULTS["variant"] = VARIANT
    RESULTS["frame_list"] = FRAMES
    jp = os.path.join(OUT, "encode_arm_b_%s.json" % VARIANT)
    with open(jp, "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print("saved : %s" % jp)
    print("files : %s" % vdir)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
