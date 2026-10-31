"""
run_boxsweep.py  —  §18-4 の判定本体（ローカル。Blender / GPU 不要）

dump_geo_npy.py が吸い出した .npy で、正規化ボックスの倍率 k を振って
Arm B の world 座標誤差を測る。

  python run_boxsweep.py --npy <dir> --variant original --armb <armB dir> ^
                         --ks 1.0,1.5,2.0,3.0 --mode center

必ず最初に **k=1 の再エンコードを既存 armB_%04d.png と bit 一致で突合する。**
ここが通らなければ .npy の吸い出しが間違っている（上下反転・チャンネル順）。
Arm B は画素の 99.92% が隣接差分 1 以下なので、誤差の大きさでは反転を
検出できない。bit 一致だけが確実な検出手段。

複数フレームは (N*H, W, 3) に縦積みして 1 枚として扱う。
armb_decode の全処理が画素ごとで、空間近傍を使わないため安全。
"""
import sys
import os
import json
import argparse


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

import numpy as np

import scene_c_spec as S
import armb_encode
import armb_decode as D
import png_io

npass = nfail = 0


def ck(name, ok, detail=""):
    global npass, nfail
    if ok:
        npass += 1
    else:
        nfail += 1
    print("%-4s %-34s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pypath", default=None,
                    help="依存モジュールの dir（_setup_syspath で処理済み）")
    ap.add_argument("--npy", required=True, help="dump_geo_npy.py の出力 dir")
    ap.add_argument("--variant", default="original")
    ap.add_argument("--armb", required=True, help="既存 armB の dir（照合用）")
    ap.add_argument("--ks", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--mode", default="center", choices=["center", "max"])
    ap.add_argument("--out", default=None, help="結果 json の保存先 dir")
    ap.add_argument("--emit", default=None,
                    help="k 別に <emit>/k{k}/armB_%04d.png を書き出す（Pod へ上げる用）")
    a = ap.parse_args()

    ks = [float(x) for x in a.ks.split(",") if x.strip()]
    pos = np.load(os.path.join(a.npy, "pos_%s.npy" % a.variant))
    ids = np.load(os.path.join(a.npy, "ids_%s.npy" % a.variant))
    with open(os.path.join(a.npy, "dump_geo_%s.json" % a.variant),
              encoding="utf-8") as fh:
        meta = json.load(fh)

    N, H, W = ids.shape
    print("syspath : %s" % (_SYSPATH_ADDED[:3],))
    print("npy     : %s" % a.npy)
    print("variant : %s" % a.variant)
    print("frames  : %s" % meta["frames"])
    print("shape   : pos%s ids%s" % (list(pos.shape), list(ids.shape)))
    print("norm box: X%s Y%s Z%s" % (S.NORM_BOX["x"], S.NORM_BOX["y"],
                                     S.NORM_BOX["z"]))
    print("ks      : %s  mode=%s" % (ks, a.mode))
    print()

    # ---- 1. k=1 の再エンコードを既存 PNG と bit 一致で突合 ------------------
    print("--- 既存 armB との bit 照合（k=1）")
    vdir = os.path.join(a.armb, a.variant)
    for s, idx in enumerate(meta["indices"]):
        p = os.path.join(vdir, "armB_%04d.png" % idx)
        if not os.path.exists(p):
            ck("bit_idx%02d" % idx, False, "missing %s" % p)
            continue
        rgb, excess = armb_encode.encode(pos[s], ids[s], S.NORM_BOX)
        px, w2, h2 = png_io.read_rgb(p)
        ref = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)
        same = bool(np.array_equal(rgb, ref))
        if same:
            ck("bit_idx%02d" % idx, True, "excess=%.2e" % excess)
        else:
            d = np.abs(rgb.astype(np.int32) - ref.astype(np.int32))
            flip = np.array_equal(rgb[::-1], ref)
            swap = np.array_equal(rgb[:, :, ::-1], ref)
            hint = " / 上下反転で一致" if flip else \
                   (" / RGB→BGR で一致" if swap else "")
            ck("bit_idx%02d" % idx, False,
               "差分 %d px, max %d%s" % (int((d > 0).any(2).sum()),
                                         int(d.max()), hint))

    if nfail:
        print()
        print("!! bit 照合が通っていない。.npy の吸い出しが正しくない。")
        print("   sweep は実行しない（間違った値がもっともらしく出るため）。")
        return 1

    # ---- 2. --emit : k 別に PNG を書き出す ---------------------------------
    if a.emit:
        print()
        print("--- emit（k 別に PNG を書き出す）")
        if N != 49:
            print("!! フレーム数が %d。VAE の temporal は 49 枚（4n+1）を要求する。" % N)
            print("   dump_geo_npy.py を --frames all で撮り直す。")
            return 1
        idxs = meta["indices"]
        if idxs != list(range(49)):
            print("!! index が 0..48 になっていない: min=%s max=%s n=%d"
                  % (min(idxs), max(idxs), len(idxs)))
            return 1
        man = {"variant": a.variant, "ks": ks, "mode": a.mode,
               "norm_box": {k: list(v) for k, v in S.NORM_BOX.items()},
               "prefix": "armB_", "dirs": {}}
        for k in ks:
            box = D.scale_box(S.NORM_BOX, k, mode=a.mode)
            kd = os.path.join(a.emit, "k%g" % k)
            os.makedirs(kd, exist_ok=True)
            mx_ex = 0.0
            files = []
            for s_i in range(N):
                rgb, ex = armb_encode.encode(pos[s_i], ids[s_i], box)
                mx_ex = max(mx_ex, ex)
                op = os.path.join(kd, "armB_%04d.png" % s_i)
                png_io.write_rgb(op, rgb, W, H)
                # 自前で書いて自前で読む（encode_arm_b.py と同じ検査）
                px, w2, h2 = png_io.read_rgb(op)
                back = np.frombuffer(px, dtype=np.uint8).reshape(h2, w2, 3)
                if not np.array_equal(back, rgb):
                    ck("emit_roundtrip_k%g_i%02d" % (k, s_i), False, "PNG 往復不一致")
                    return 1
                files.append(os.path.basename(op))
            man["dirs"]["k%g" % k] = {
                "box": {kk: list(vv) for kk, vv in box.items()},
                "units_per_lsb": D.sensitivity(box).tolist(),
                "quant_worst_units": D.quant_worst(box).tolist(),
                "max_q_excess": float(mx_ex), "n_files": len(files)}
            ck("emit_k%g" % k, mx_ex == 0.0,
               "%d files, X幅 %.2f, max_excess %.2e"
               % (len(files), box["x"][1] - box["x"][0], mx_ex))
        mp = os.path.join(a.emit, "emit_manifest_%s.json" % a.variant)
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2, ensure_ascii=False, default=str)
        print("saved : %s" % mp)
        print()
        print("Pod へ上げるのは %s 配下。GT の .npy は上げない。" % a.emit)

    # ---- 3. 縦積みして sweep ------------------------------------------------
    print()
    print("--- sweep（量子化のみ。vae=None）")
    pos_s = pos.reshape(N * H, W, 3).astype(np.float64)
    ids_s = ids.reshape(N * H, W)
    rows = D.sweep(pos_s, ids_s, S.NORM_BOX, ks, mode=a.mode)

    print()
    print(D.format_sweep(rows))
    print()
    print("| k | X p50 | X p95 | X p99 | X max | (すべて px) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        px = r["err"]["px"]["X"]
        print("| %.2f | %.3f | %.3f | %.3f | %.3f | |"
              % (r["k"], px["p50"], px["p95"], px["p99"], px["max"]))
    print()
    print("| k | 空 | 背景 | Target | 予約帯 | 帯不一致 | ch食い違い |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        i = r["info"]
        nb = r["err"].get("band", {}).get("n_confused", 0)
        print("| %.2f | %d | %d | %d | %d | %d | %d |"
              % (r["k"], i["n_empty"], i["n_background"], i["n_target"],
                 i["n_reserved"], nb, i["n_disagree"]))

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        jp = os.path.join(a.out, "boxsweep_%s.json" % a.variant)
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump({"variant": a.variant, "frames": meta["frames"],
                       "ks": ks, "mode": a.mode,
                       "norm_box": {k: list(v) for k, v in S.NORM_BOX.items()},
                       "rows": rows}, fh, indent=2, ensure_ascii=False,
                      default=str)
        print()
        print("saved : %s" % jp)

    print()
    print("=" * 66)
    print("bit 照合 PASS %d / FAIL %d" % (npass, nfail))
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
