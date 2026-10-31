# -*- coding: utf-8 -*-
"""
DiffSynth の default_video_operator を通して bit 一致を確認する（地雷3 の検証）。

なぜ operator をそのまま呼ぶのか
  ImageCropAndResize は scale = 1.0 でも torchvision の resize を必ず通る
  （operators.py:127-129）。自分で resize を再実装して確認しても、
  DiffSynth が実際に走らせる経路とは別物になる。
  train.py:141 が UnifiedDataset.default_video_operator を使っているので、
  同じ関数を同じ引数で呼ぶ。

照合値
  EXPECT は PNG 側から計算した content sha256（デコード後の画素のハッシュ）。
  ファイルのハッシュは ffmpeg のバージョンで変わるため使わない。

使い方（Pod / bash）:
  python check_operator.py
  python check_operator.py --base /workspace/w4/mkv
"""
import argparse
import hashlib
import sys

import numpy as np

DEFAULT_BASE = "/workspace/w4/mkv"

# ローカルで png_to_ffv1.py が出した content sha256
EXPECT = {
    "armA_original.mkv": "246ed61a47ad39cdf30a5388b137ce4df13024458cffb3495ea7750d240028c7",
    "armB_original.mkv": "ab5eb47f98a1bd745509292f1124397928fdbc43b7569fd30880458ca6fbb85a",
    "armC_original.mkv": "46abbcded632a61bd84589daec3249b2b0320669b54c0c146b35b0213763b8bf",
    "tv_R0S0_original.mkv": "bba8f4b9e280b9e3b599e01f523d17bb540d6031a883f7fd0eb6c24458f6269d",
}

EXPECT_W = 832
EXPECT_H = 480
EXPECT_N = 49


def content_sha256(stack):
    a = np.ascontiguousarray(stack, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def to_array(frames):
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames])


def run(base, height, width, max_pixels, label):
    from diffsynth.core import UnifiedDataset
    op = UnifiedDataset.default_video_operator(
        base_path=base,
        max_pixels=max_pixels, height=height, width=width,
        height_division_factor=16, width_division_factor=16,
        num_frames=EXPECT_N, time_division_factor=4, time_division_remainder=1,
    )
    print()
    print(f"===== {label} =====")
    print(f"  height={height} width={width} max_pixels={max_pixels}")
    print()
    print("| file | frames | W | H | bit 一致 |")
    print("|---|---|---|---|---|")
    rows = []
    for name in sorted(EXPECT):
        try:
            frames = op(name)
        except Exception as e:
            print(f"| {name} | — | — | — | **例外 {type(e).__name__}** |")
            rows.append(False)
            continue
        arr = to_array(frames)
        got = content_sha256(arr)
        ok = (got == EXPECT[name])
        rows.append(ok and arr.shape[0] == EXPECT_N
                    and arr.shape[1] == EXPECT_H and arr.shape[2] == EXPECT_W)
        print(f"| {name} | {arr.shape[0]} | {arr.shape[2]} | {arr.shape[1]} | {ok} |")
        if not ok:
            print(f"|   期待 | `{EXPECT[name][:24]}…` | | | |")
            print(f"|   実測 | `{got[:24]}…` | | | |")
    return all(rows)


def run_reference_image(base):
    """参照画像（PNG 1枚）が 1 要素の list になるか。train.py:71 が [0] を取る。"""
    from diffsynth.core import UnifiedDataset
    op = UnifiedDataset.default_video_operator(
        base_path=base, max_pixels=None, height=EXPECT_H, width=EXPECT_W,
        height_division_factor=16, width_division_factor=16,
        num_frames=EXPECT_N, time_division_factor=4, time_division_remainder=1,
    )
    print()
    print("===== 参照画像（PNG 1枚）=====")
    frames = op("ref_R0S0_original.png")
    first = np.asarray(frames[0].convert("RGB"), dtype=np.uint8)
    print("| 項目 | 値 | 期待 |")
    print("|---|---|---|")
    print(f"| list の長さ | {len(frames)} | 1 |")
    print(f"| W x H | {first.shape[1]} x {first.shape[0]} | {EXPECT_W} x {EXPECT_H} |")
    print(f"| content sha256 | `{content_sha256(first[None])[:24]}…` | — |")
    return len(frames) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    a = ap.parse_args()

    import imageio
    print(f"[env] imageio {imageio.__version__}")
    try:
        import imageio_ffmpeg
        print(f"[env] imageio_ffmpeg {imageio_ffmpeg.__version__}")
    except Exception:
        print("[env] imageio_ffmpeg なし")
    try:
        import av  # noqa: F401
        print("[env] pyav あり（ローカルは無かった。デコード経路が違う可能性）")
    except Exception:
        print("[env] pyav なし（ローカルと同じ）")
    import torch, torchvision
    print(f"[env] torch {torch.__version__} / torchvision {torchvision.__version__}")

    ok_explicit = run(a.base, EXPECT_H, EXPECT_W, None, "height/width 明示")
    ok_dynamic = run(a.base, None, None, 1024 * 1024,
                     "動的解像度（--height/--width 未指定）")
    ok_ref = run_reference_image(a.base)

    print()
    print("| 条件 | 結果 |")
    print("|---|---|")
    print(f"| height/width 明示 | {ok_explicit} |")
    print(f"| 動的解像度 | {ok_dynamic} |")
    print(f"| 参照画像が 1 要素 | {ok_ref} |")
    print()
    if ok_explicit and ok_ref:
        print("[result] PASS  地雷3 は無い。control は無改変で DiT に届く")
    else:
        print("[result] FAIL  resize で画素が変わっている。ここを解決するまで学習しない")
    sys.exit(0 if (ok_explicit and ok_ref) else 1)


if __name__ == "__main__":
    main()
