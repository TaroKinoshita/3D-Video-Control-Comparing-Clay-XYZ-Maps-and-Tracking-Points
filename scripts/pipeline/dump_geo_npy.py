"""
dump_geo_npy.py  —  §18-4 用。幾何パス EXR を .npy に吸い出す

Blender 内で走らせる理由は EXR を読むためだけ。load_rgb() は
encode_arm_b.py のものをそのまま写した（反転・チャンネル順を変えない）。

  & $BL --background --factory-startup --python dump_geo_npy.py "--" ^
        --pypath <W2\\.py> --geo <geo dir> --variant original ^
        --out <out dir> --frames all

  Blender の埋め込み Python は PYTHONPATH を無視するので --pypath が必須。

入力  <geo>/<variant>/position_%04d.exr   （world 座標。frame 1..49）
      <geo>/<variant>/id_%04d.exr         （pass_index。Emission ID）
出力  <out>/pos_<variant>.npy   (N, H, W, 3) float32
      <out>/ids_<variant>.npy   (N, H, W)    float32
      <out>/dump_geo_<variant>.json

frame（1..49）と index（0..48）の対応は json に残す（既知バグ#3 の系統）。
"""
import sys
import os
import json


def _setup_syspath():
    """依存モジュール（scene_c_spec / armb_encode / png_io など）の場所を通す。

    Blender の埋め込み Python は PYTHONPATH を無視するため、
    --pypath で明示的に渡せるようにする。環境変数からも拾う。
    優先順は --pypath > PYTHONPATH > このスクリプトの位置。
    """
    import sys as _s, os as _o
    cand = []
    a = _s.argv
    a = a[a.index("--") + 1:] if "--" in a else a
    if "--pypath" in a:
        i = a.index("--pypath")
        if i + 1 < len(a):
            cand += [p for p in a[i + 1].split(_o.pathsep) if p.strip()]
    cand += [p for p in _o.environ.get("PYTHONPATH", "").split(_o.pathsep)
             if p.strip()]
    here = _o.path.dirname(_o.path.abspath(__file__)) \
        if "__file__" in globals() else _o.getcwd()
    cand.append(here)
    added = []
    for p in cand:
        p = _o.path.abspath(p)
        if _o.path.isdir(p) and p not in _s.path:
            _s.path.insert(0, p)
            added.append(p)
    return added


_SYSPATH_ADDED = _setup_syspath()

import bpy
import numpy as np

import scene_c_spec as S
import armb_expect as A

W, H = S.RENDER["resolution_x"], S.RENDER["resolution_y"]


def args():
    a = sys.argv
    a = a[a.index("--") + 1:] if "--" in a else []

    def opt(k, d):
        return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else d

    opt("--pypath", None)          # _setup_syspath で処理済み
    geo = opt("--geo", None)
    out = opt("--out", None)
    if geo is None or out is None:
        raise SystemExit("--geo と --out は必須（既定パスに頼らない）")
    variant = opt("--variant", "original")
    fr = opt("--frames", "smoke")
    if fr == "all":
        fl = list(range(1, A.N_FRAMES + 1))
    elif fr == "smoke":
        fl = [1, 13, 24, 37, 49]
    else:
        fl = [int(x) for x in fr.split(",") if x.strip()]
    return os.path.abspath(geo), variant, os.path.abspath(out), fl


GEO, VARIANT, OUT, FRAMES = args()


def load_rgb(path):
    """EXR の RGB を (H, W, 3) float32 で返す。行は PNG 走査順（上が先）。

    encode_arm_b.py の load_rgb と同一。1文字も変えない。
    """
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
    print("res     : %dx%d" % (W, H))
    print("syspath : %s" % (_SYSPATH_ADDED[:3],))

    os.makedirs(OUT, exist_ok=True)
    gdir = os.path.join(GEO, VARIANT)

    n = len(FRAMES)
    pos = np.empty((n, H, W, 3), dtype=np.float32)
    ids = np.empty((n, H, W), dtype=np.float32)
    rows = []
    nfail = 0

    for s, f in enumerate(FRAMES):
        idx = f - 1
        pp = os.path.join(gdir, "position_%04d.exr" % f)
        ip = os.path.join(gdir, "id_%04d.exr" % f)
        for p in (pp, ip):
            if not os.path.exists(p):
                raise SystemExit("missing: %s" % p)
        p_arr = load_rgb(pp)
        i_arr = load_rgb(ip)[:, :, 0]

        finite = bool(np.all(np.isfinite(p_arr)) and np.all(np.isfinite(i_arr)))
        uniq = np.unique(i_arr)
        non_int = [float(u) for u in uniq if abs(u - round(float(u))) > 1e-6]

        pos[s] = p_arr
        ids[s] = i_arr

        ok = finite and not non_int
        nfail += 0 if ok else 1
        rows.append({"slot": s, "frame": f, "index": idx,
                     "finite": finite,
                     "id_unique": [float(u) for u in uniq],
                     "non_integral_ids": non_int[:5]})
        print("%-4s slot=%02d frame=%02d index=%02d ids=%s"
              % ("PASS" if ok else "FAIL", s, f, idx,
                 [int(u) for u in uniq]))

    pp_out = os.path.join(OUT, "pos_%s.npy" % VARIANT)
    ip_out = os.path.join(OUT, "ids_%s.npy" % VARIANT)
    np.save(pp_out, pos)
    np.save(ip_out, ids)

    meta = {"variant": VARIANT, "frames": FRAMES,
            "indices": [f - 1 for f in FRAMES],
            "shape_pos": list(pos.shape), "shape_ids": list(ids.shape),
            "dtype": "float32", "width": W, "height": H,
            "row_order": "PNG scan order (top first)",
            "channels": "R=X G=Y B=Z",
            "load_rgb_source": "encode_arm_b.py (verbatim)",
            "per_frame": rows}
    jp = os.path.join(OUT, "dump_geo_%s.json" % VARIANT)
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)

    mb = (pos.nbytes + ids.nbytes) / 1024.0 / 1024.0
    print("=" * 66)
    print("PASS %d / FAIL %d" % (len(FRAMES) - nfail, nfail))
    print("saved : %s" % pp_out)
    print("saved : %s" % ip_out)
    print("saved : %s" % jp)
    print("size  : %.1f MB" % mb)
    print("=" * 66)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
