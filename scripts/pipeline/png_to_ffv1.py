#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG 連番 -> FFV1 可逆動画 -> bit 一致検証

DiffSynth 2.0.18 の `vace_video` / `video` は動画ファイルでなければならない。
PNG 1枚を渡すと 1 フレームの list になり、エラーも警告も出さずに
49 フレームのうち 1 フレームだけで学習が回る（operators.py:47-60）。

検証は DiffSynth が実際に使う経路で行う。
  diffsynth/core/data/operators.py の LoadVideo:
      reader = imageio.get_reader(path)
      raw_frame_rate = reader.get_meta_data()['fps']
      total_raw_frames = reader.count_frames()
      frame = reader.get_data(frame_id)      # fix_frame_rate=False なので frame_id は 1:1
別の経路（cv2 / PyAV 直読み）で一致を確認しても意味がない。

使い方（ローカル / PowerShell）:
  python png_to_ffv1.py encode  --src <PNG_DIR> --prefix armB_ --out <OUT.mkv>
  python png_to_ffv1.py verify  --src <PNG_DIR> --prefix armB_ --mkv <OUT.mkv>
  python png_to_ffv1.py roundtrip --src <PNG_DIR> --prefix armB_ --out <OUT.mkv>
  python png_to_ffv1.py selftest --workdir <TMP_DIR>
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# 定数（protocol §5）
# --------------------------------------------------------------------------
EXPECT_W = 832
EXPECT_H = 480
EXPECT_N = 49
FPS = 16  # LoadVideo が get_meta_data()['fps'] を読むため、値は必要。1:1 読みなので内容には影響しない


# --------------------------------------------------------------------------
# 入出力
# --------------------------------------------------------------------------
def list_pngs(src, prefix):
    """prefix で始まる PNG を名前順に返す。"""
    if not os.path.isdir(src):
        raise SystemExit(f"[FATAL] src が存在しない: {src}")
    names = sorted(f for f in os.listdir(src)
                   if f.startswith(prefix) and f.lower().endswith(".png"))
    if not names:
        raise SystemExit(f"[FATAL] {src} に {prefix}*.png が無い")
    return [os.path.join(src, n) for n in names]


def load_png_stack(paths):
    """PNG を uint8 の (N,H,W,3) で返す。RGBA / パレットは RGB に変換する。"""
    frames = []
    for p in paths:
        with Image.open(p) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            frames.append(np.asarray(im, dtype=np.uint8))
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise SystemExit(f"[FATAL] フレーム間で shape が違う: {shapes}")
    return np.stack(frames, axis=0)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(stack):
    """デコード後の画素そのもののハッシュ。

    ファイルのハッシュは ffmpeg のバージョンやコンテナのメタデータで変わるため、
    環境をまたいだ照合には使えない。画素の中身だけを見るこの値を使う。
    ローカルと Pod で同じ mkv がこの値で一致すれば、デコード経路の差は無い。
    """
    a = np.ascontiguousarray(stack, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit("[FATAL] ffmpeg が見つからない")


# --------------------------------------------------------------------------
# encode
# --------------------------------------------------------------------------
def encode(src, prefix, out, fps=FPS, strict=True):
    paths = list_pngs(src, prefix)
    stack = load_png_stack(paths)
    n, h, w, _ = stack.shape
    print(f"[encode] {len(paths)} 枚  {w}x{h}")
    if strict:
        if (w, h) != (EXPECT_W, EXPECT_H):
            raise SystemExit(f"[FATAL] 解像度が §5 と違う: {w}x{h} != {EXPECT_W}x{EXPECT_H}")
        if n != EXPECT_N:
            raise SystemExit(f"[FATAL] フレーム数が §5 と違う: {n} != {EXPECT_N}")
    if n % 4 != 1:
        raise SystemExit(f"[FATAL] フレーム数が 4 の倍数+1 でない: {n}")

    # PNG のファイル名は連番でない場合があるので concat ではなく画像パイプで渡す
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cmd = [
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-fflags", "+bitexact",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        # FFV1: RGB を平面のまま可逆保存する。gbrp なので YUV 変換を通らない
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp",
        "-g", "1",           # 全フレーム intra。フレーム単体取得（get_data）が確実になる
        "-flags", "+bitexact",   # encoder 文字列を書かない
        "-fflags", "+bitexact",  # SegmentUID / DateUTC を書かない（同一入力で同一ファイル）
        out,
    ]
    proc = subprocess.run(cmd, input=stack.tobytes(), capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"[FATAL] ffmpeg が失敗: rc={proc.returncode}")

    png_bytes = sum(os.path.getsize(p) for p in paths)
    mkv_bytes = os.path.getsize(out)
    print(f"[encode] 出力      : {out}")
    print(f"[encode] PNG 合計  : {png_bytes:,} bytes")
    print(f"[encode] FFV1      : {mkv_bytes:,} bytes  ({mkv_bytes / png_bytes:.3f} 倍)")
    print(f"[encode] file sha256    : {sha256_file(out)}   (ffmpeg のバージョン依存)")
    print(f"[encode] content sha256 : {content_sha256(stack)}   (PNG の画素。環境非依存)")
    return out


# --------------------------------------------------------------------------
# verify（DiffSynth と同じ経路）
# --------------------------------------------------------------------------
def check_backend(quiet=False):
    """imageio が動画を読めるバックエンドを持っているか先に確認する。

    DiffSynth の LoadVideo も imageio.get_reader を呼ぶ（operators.py:146）ので、
    ここで読めない環境は学習時にも同じ ValueError で落ちる。
    ローカルと Pod で別のバックエンドが選ばれると、同じ mkv を別経路で
    デコードすることになるため、どちらが選ばれたかを必ず表示する。
    """
    found = []
    try:
        import imageio_ffmpeg  # noqa: F401
        found.append("FFMPEG (imageio-ffmpeg)")
    except Exception:
        pass
    try:
        import av  # noqa: F401
        found.append("pyav")
    except Exception:
        pass
    if not quiet:
        print(f"[backend] 利用可能: {found if found else 'なし'}")
    if not found:
        raise SystemExit(
            "[FATAL] imageio に動画を読むバックエンドが無い。\n"
            "        ローカル / PowerShell:  pip install \"imageio[ffmpeg]\"\n"
            "        Pod / bash:            pip install \"imageio[ffmpeg]\"\n"
            "        DiffSynth の LoadVideo も同じ経路を使うため、Pod 側にも必要。")
    return found


def read_via_diffsynth_path(mkv, expect_n=None):
    """LoadVideo と同じ手順で読む。"""
    import imageio
    check_backend()
    reader = imageio.get_reader(mkv)
    print(f"[backend] 実際に選ばれた reader: {type(reader).__module__}.{type(reader).__name__}")
    meta = reader.get_meta_data()
    if "fps" not in meta:
        reader.close()
        raise SystemExit("[FATAL] get_meta_data() に 'fps' が無い。LoadVideo が KeyError で落ちる")
    fps = meta["fps"]
    total = reader.count_frames()
    n = expect_n if expect_n is not None else total
    frames = []
    for i in range(n):
        frames.append(np.asarray(reader.get_data(i), dtype=np.uint8))  # fix_frame_rate=False → 1:1
    reader.close()
    return np.stack(frames, axis=0), fps, total


def verify(src, prefix, mkv):
    paths = list_pngs(src, prefix)
    ref = load_png_stack(paths)
    got, fps, total = read_via_diffsynth_path(mkv, expect_n=ref.shape[0])

    print(f"[verify] imageio fps        : {fps}")
    print(f"[verify] count_frames()     : {total}   (PNG {ref.shape[0]} 枚)")
    print(f"[verify] shape  PNG / mkv   : {ref.shape} / {got.shape}")

    ok_shape = ref.shape == got.shape
    ok_count = total == ref.shape[0]
    if not ok_shape:
        print("[verify] FAIL: shape 不一致")
        return False

    diff = (ref.astype(np.int16) - got.astype(np.int16))
    bad = np.abs(diff) > 0
    n_bad_frames = int(np.count_nonzero(bad.any(axis=(1, 2, 3))))
    n_bad_px = int(np.count_nonzero(bad.any(axis=-1)))
    max_abs = int(np.abs(diff).max())

    print()
    print("| 項目 | 値 | 期待 |")
    print("|---|---|---|")
    print(f"| フレーム数一致 | {ok_count} | True |")
    print(f"| 不一致フレーム数 | {n_bad_frames} / {ref.shape[0]} | 0 |")
    print(f"| 不一致画素数 | {n_bad_px:,} | 0 |")
    print(f"| 最大絶対差 | {max_abs} | 0 |")
    for ci, cn in enumerate("RGB"):
        print(f"| max diff {cn} | {int(np.abs(diff[..., ci]).max())} | 0 |")

    print(f"| content sha256 PNG | `{content_sha256(ref)[:32]}…` | — |")
    print(f"| content sha256 mkv | `{content_sha256(got)[:32]}…` | 左に一致 |")

    ok = ok_count and n_bad_px == 0 and max_abs == 0
    print()
    print(f"[verify] {'PASS  bit 一致' if ok else 'FAIL  bit 一致していない'}")
    if not ok and n_bad_px:
        fi, yi, xi = np.argwhere(bad.any(axis=-1))[0]
        print(f"[verify] 最初の不一致 frame={fi} (x={xi}, y={yi}) "
              f"PNG={ref[fi, yi, xi].tolist()} mkv={got[fi, yi, xi].tolist()}")
    return ok


# --------------------------------------------------------------------------
# selftest（assert が壊したデータで発火するか）
# --------------------------------------------------------------------------
def _synth_frames(n=EXPECT_N, w=EXPECT_W, h=EXPECT_H, seed=0):
    """Arm B / Arm C の危険な性質を模した合成フレーム。

    - 帯の端の値（13 / 15 / 124 / 127 / 128 / 131）を必ず含める
    - チャンネルごとに違う勾配を持たせる（クロマ間引きで壊れる形）
    - 半径6の円板を置く（Arm C の点。面積が小さく色が混ざりやすい）
    """
    rng = np.random.default_rng(seed)
    edge = np.array([13, 15, 124, 127, 128, 131, 196, 50], dtype=np.uint8)
    out = np.zeros((n, h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for f in range(n):
        out[f, :, :, 0] = (xx * 102 // w + 13).astype(np.uint8)     # X 相当の連続勾配
        out[f, :, :, 1] = (yy * 102 // h + 13).astype(np.uint8)     # Y 相当
        out[f, :, :, 2] = np.uint8(13 + (f * 2) % 102)              # Z 相当（時間変化）
        out[f, 0, :edge.size * 4, 0] = np.repeat(edge, 4)           # 帯の端を1行に埋める
        out[f, 1, :edge.size * 4, 1] = np.repeat(edge, 4)
        out[f, 2, :edge.size * 4, 2] = np.repeat(edge, 4)
        for k in range(24):                                          # 半径6の円板
            cx = int(rng.integers(10, w - 10)); cy = int(rng.integers(10, h - 10))
            m = (xx - cx) ** 2 + (yy - cy) ** 2 <= 36
            out[f][m] = np.array([196, 30, 98], dtype=np.uint8)
    return out


def selftest(workdir):
    check_backend()
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, "png")
    os.makedirs(src, exist_ok=True)
    stack = _synth_frames()
    for i in range(stack.shape[0]):
        Image.fromarray(stack[i]).save(os.path.join(src, f"synth_{i:04d}.png"))
    print(f"[selftest] 合成 PNG {stack.shape[0]} 枚を {src} に作成")

    print("\n===== ケース1: FFV1（可逆）=====")
    mkv_ok = os.path.join(workdir, "ok_ffv1.mkv")
    encode(src, "synth_", mkv_ok, strict=False)
    r1 = verify(src, "synth_", mkv_ok)

    print("\n===== ケース2: H.264 既定（非可逆。assert が発火するべき）=====")
    mkv_bad = os.path.join(workdir, "bad_h264.mkv")
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{EXPECT_W}x{EXPECT_H}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", mkv_bad]
    subprocess.run(cmd, input=stack.tobytes(), capture_output=True, check=True)
    r2 = verify(src, "synth_", mkv_bad)

    print("\n===== ケース3: H.264 crf0 yuv444p（数学的可逆だが RGB 変換が非可逆）=====")
    mkv_444 = os.path.join(workdir, "crf0_yuv444.mkv")
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{EXPECT_W}x{EXPECT_H}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", mkv_444]
    subprocess.run(cmd, input=stack.tobytes(), capture_output=True, check=True)
    r3 = verify(src, "synth_", mkv_444)

    print("\n===== ケース4: 1枚だけの PNG を渡した場合の DiffSynth 挙動 =====")
    print("  operators.py:49 の png 経路は LoadImage >> resize >> ToList() で 1 要素の list を返す。")
    print("  エラーは出ない。verify では shape 不一致として検出される。")

    print("\n===== selftest まとめ =====")
    print("| ケース | bit 一致 | 期待 |")
    print("|---|---|---|")
    print(f"| 1 FFV1 gbrp | {r1} | True |")
    print(f"| 2 H.264 yuv420p | {r2} | **False**（発火するべき） |")
    print(f"| 3 H.264 crf0 yuv444p | {r3} | False または True（要観察） |")
    ok = (r1 is True) and (r2 is False)
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}  "
          f"（可逆で一致し、非可逆で不一致を検出できたか）")
    return ok


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("encode")
    p.add_argument("--src", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--no-strict", action="store_true")

    p = sub.add_parser("verify")
    p.add_argument("--src", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--mkv", required=True)

    p = sub.add_parser("roundtrip")
    p.add_argument("--src", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--no-strict", action="store_true")

    p = sub.add_parser("selftest")
    p.add_argument("--workdir", required=True)

    a = ap.parse_args()
    if a.cmd == "encode":
        encode(a.src, a.prefix, a.out, a.fps, strict=not a.no_strict)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.src, a.prefix, a.mkv) else 1)
    elif a.cmd == "roundtrip":
        check_backend()   # 1.8 MB 書いてから落ちるのを防ぐため先に確認する
        encode(a.src, a.prefix, a.out, a.fps, strict=not a.no_strict)
        print()
        sys.exit(0 if verify(a.src, a.prefix, a.out) else 1)
    elif a.cmd == "selftest":
        sys.exit(0 if selftest(a.workdir) else 1)


if __name__ == "__main__":
    main()
