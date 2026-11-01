# -*- coding: utf-8 -*-
"""infer_seeds.py — 1 arm につきモデルを 1 回だけ読み込み、seed を連続で生成する。（W12, 2026-11-01）

追加実験（protocol rev.17）用。生成の設定は infer_arm.py と同一（rev.11 変更4）。
  cfg 5.0 / steps 50 / sigma_shift 5.0 / negative "" / num_frames 49 / tiled False
  LoRA = /workspace/runs/g1_arm{X}/step-500.safetensors（S1）、alpha=1、pipe.vace
  control = Scene C′（arm{X}_prime.mkv）

infer_arm.py との違い
  - モデルと LoRA の読み込みを 1 回にする（LoRA は fuse すると外せないので arm ごとに 1 プロセス）
  - **最初に seed 0（G1）を作り直し、Phase 0.8 の content sha と一致するか確かめる。**
    一致しなければ「1 プロセスで続けて生成すると結果が変わる」ことになるので、その場で止める
  - PNG を経由せず、生成フレームを直接 FFV1 に書く（mkv を読み戻して bit 一致を確認）

出力名は infer_arm.py と同じ規則：gen_arm{X}_step500_G{seed+1}.mkv（上書きしない）

使い方（Pod）
  python infer_seeds.py --arm A --seeds 4-19
"""
import argparse
import contextlib
import hashlib
import io
import os
import subprocess
import sys

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_arm import DIT, T5, VAE, TOK, MKV, REF, RUNS, PROMPT, FUSE_MSG  # noqa: E402

# Phase 0.8（W9）で記録した G1 の content sha(B)。バッチ生成の同一性検査に使う
G1_SHA = {"A": "6005c7087f6722f5", "B": "15eb5a35f508dae2", "C": "9b0ba42677cc0792"}
OUT = RUNS + "/gen"


def parse_seeds(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seeds", required=True, help="例: 4-19（= G5〜G20）")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    ctrl_path = "%s/arm%s_prime.mkv" % (MKV, a.arm)
    lora = "%s/g1_arm%s/step-500.safetensors" % (RUNS, a.arm)
    for p in [DIT, T5, VAE, TOK, ctrl_path, REF, lora]:
        if not os.path.exists(p):
            raise SystemExit("[FATAL] 見つからない: " + p)
    for s in seeds:
        p = os.path.join(OUT, "gen_arm%s_step500_G%d.mkv" % (a.arm, s + 1))
        if os.path.exists(p):
            raise SystemExit("[FATAL] 既に存在する（上書き防止）: " + p)

    import numpy as np
    import torch
    import imageio
    from PIL import Image
    from diffsynth.utils.data import VideoData
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=DIT), ModelConfig(path=T5), ModelConfig(path=VAE)],
        tokenizer_config=ModelConfig(path=TOK),
    )
    buf_o, buf_e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_o), contextlib.redirect_stderr(buf_e):
        pipe.load_lora(pipe.vace, lora, alpha=1)
    log = buf_o.getvalue() + buf_e.getvalue()
    sys.stdout.write(log)
    if FUSE_MSG not in log:
        raise SystemExit("[FATAL] '%s' が出ていない（LoRA が乗っていない）" % FUSE_MSG)
    print("[check] LoRA fused OK  arm", a.arm, flush=True)

    vd = VideoData(ctrl_path, height=480, width=832)
    ctrl = [vd[i] for i in range(49)]
    ref = Image.open(REF).convert("RGB").resize((832, 480))

    def generate(seed):
        video = pipe(prompt=PROMPT, negative_prompt="", vace_video=ctrl, vace_reference_image=ref,
                     num_frames=49, cfg_scale=5.0, num_inference_steps=50, sigma_shift=5.0,
                     seed=seed, tiled=False)
        arr = np.stack([np.array(fr)[..., :3] for fr in video]).astype(np.uint8)
        if arr.shape != (49, 480, 832, 3):
            raise SystemExit("[FATAL] 想定外の shape %s" % (arr.shape,))
        return arr

    def write_mkv(arr, path):
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+bitexact",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "832x480", "-r", "16", "-i", "pipe:0",
                        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-g", "1",
                        "-flags", "+bitexact", "-fflags", "+bitexact", path],
                       input=np.ascontiguousarray(arr).tobytes(), check=True)
        rd = imageio.get_reader(path)
        back = np.stack([np.asarray(f)[..., :3] for f in rd])
        rd.close()
        if not np.array_equal(back, arr):
            raise SystemExit("[FATAL] mkv が生成フレームと bit 一致しない: " + path)
        return hashlib.sha256(back.tobytes()).hexdigest()[:16]

    # 1) バッチ生成の同一性検査：G1 を作り直して Phase 0.8 と比べる（ファイルは残さない）
    sha = hashlib.sha256(generate(0).tobytes()).hexdigest()[:16]
    if sha != G1_SHA[a.arm]:
        raise SystemExit("[FATAL] G1 の再生成が Phase 0.8 と一致しない: %s != %s" % (sha, G1_SHA[a.arm]))
    print("[check] G1 regenerated in batch mode: sha %s == Phase 0.8" % sha, flush=True)

    # 2) 追加の seed
    for s in seeds:
        path = os.path.join(OUT, "gen_arm%s_step500_G%d.mkv" % (a.arm, s + 1))
        sha = write_mkv(generate(s), path)
        print("wrote %s content sha(B): %s" % (path, sha), flush=True)


if __name__ == "__main__":
    main()
