# -*- coding: utf-8 -*-
"""infer_arm.py — Scene C′ の control から動画を生成する（Phase 0.8 / Pipeline floor 共通）。（W9）

infer_g1_armC_v2.py を arm 切替式にしたもの。設定は protocol rev.11 変更4 のまま変えない。
  cfg 5.0 / steps 50 / sigma_shift 5.0 / negative "" / num_frames 49 / tiled False / seed = G
  LoRA は pipe.vace に alpha=1 で load（学習の --remove_prefix_in_ckpt pipe.vace. と対応）

v2 から足したもの
  - --arm A/B/C で LoRA と control を切り替える（パスは下の既定値。上書き可）
  - DIFFSYNTH_SKIP_DOWNLOAD=True と tokenizer の実パス直指定（W6 ハマりどころ #4 / #5）
  - 「150 tensors are fused by LoRA」が出なければ止める（静かな失敗：ベースモデルで推論）
  - --no-lora で Step 0（LoRA なし）を生成
  - 出力を FFV1 gbrp bitexact の mkv で保存し、content sha（方式B）を表示
    （mp4 は目視用。採点は mkv で行う）

⚠️ GPU 環境で未実行。DiffSynth 2.0.18 の ModelConfig(path=...) / tokenizer_config は
   W6 の worklog の記述に基づく。最初の 1 本で動作を確認すること。

使い方（Pod）
  python infer_arm.py --arm A                 # G1, step-500
  python infer_arm.py --arm B --seed 1        # G2
  python infer_arm.py --arm C --no-lora       # Step 0
"""
import argparse
import contextlib
import hashlib
import io
import os
import subprocess
import sys

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

MODELS = "/workspace/DiffSynth-Studio/models"
DIT = MODELS + "/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"
T5 = MODELS + "/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE = MODELS + "/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
TOK = MODELS + "/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"   # 末尾スラッシュ無し

MKV = "/workspace/w4/mkv"
REF = MKV + "/ref_R0S0_original.png"
RUNS = "/workspace/runs"
# 学習 run の出力先。Arm C は W6 の偵察 run（g1_armC）を S1 として流用する
# 出力先は /workspace/runs/{run}_arm{X}。run = g1/g2/g3 が training seed S1/S2/S3 のラベル（§9）

# metadata_arm*.jsonl と 1 バイトも違わないこと（make_metadata_v2.PROMPT）
PROMPT = ("static camera, simple matte geometric solids on a flat floor, "
          "plain even lighting, an orange hexagonal prism")
FUSE_MSG = "150 tensors are fused by LoRA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=0, help="G1=0 G2=1 G3=2 G4=3（§9）")
    ap.add_argument("--step", type=int, default=500)
    ap.add_argument("--run", default="g1", choices=["g1", "g2", "g3"], help="学習 run（S1/S2/S3）")
    ap.add_argument("--suffix", default="", help="出力名の末尾（例: _rep / _shift1）。同名の上書きを防ぐ")
    ap.add_argument("--lora", help="既定: /workspace/runs/{run}_arm{X}/step-{step}.safetensors")
    ap.add_argument("--ctrl", help="既定: /workspace/w4/mkv/arm{X}_prime.mkv")
    ap.add_argument("--no-lora", action="store_true", help="Step 0（LoRA なし）")
    ap.add_argument("--out-dir", default=RUNS + "/gen")
    a = ap.parse_args()

    ctrl_path = a.ctrl or "%s/arm%s_prime.mkv" % (MKV, a.arm)
    lora = None if a.no_lora else (a.lora or "%s/%s_arm%s/step-%d.safetensors" % (RUNS, a.run, a.arm, a.step))
    run_tag = "" if a.run == "g1" else a.run + "_"
    tag = "arm%s_%s%s_G%d%s" % (a.arm, run_tag, "step0" if a.no_lora else "step%d" % a.step,
                               a.seed + 1, a.suffix)
    if os.path.exists(os.path.join(a.out_dir, "gen_%s.mkv" % tag)):
        raise SystemExit("[FATAL] 既に存在する（上書き防止）: gen_%s.mkv。--suffix で名前を変える" % tag)
    for p in [DIT, T5, VAE, TOK, ctrl_path, REF] + ([lora] if lora else []):
        if not os.path.exists(p):
            raise SystemExit("[FATAL] 見つからない: " + p)
    print("tag :", tag)
    print("ctrl:", ctrl_path)
    print("lora:", lora)

    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    from diffsynth.utils.data import save_video, VideoData
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=DIT), ModelConfig(path=T5), ModelConfig(path=VAE)],
        tokenizer_config=ModelConfig(path=TOK),
    )
    if lora:
        buf_o, buf_e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_o), contextlib.redirect_stderr(buf_e):
            pipe.load_lora(pipe.vace, lora, alpha=1)
        log = buf_o.getvalue() + buf_e.getvalue()
        sys.stdout.write(log)
        if FUSE_MSG not in log:
            raise SystemExit("[FATAL] '%s' が出ていない。LoRA が乗っていない可能性"
                             "（ベースモデルで推論する静かな失敗）。" % FUSE_MSG)
        print("[check] LoRA fused OK")

    vd = VideoData(ctrl_path, height=480, width=832)
    ctrl = [vd[i] for i in range(49)]
    ref = Image.open(REF).convert("RGB").resize((832, 480))

    video = pipe(
        prompt=PROMPT, negative_prompt="",
        vace_video=ctrl, vace_reference_image=ref,
        num_frames=49, cfg_scale=5.0, num_inference_steps=50, sigma_shift=5.0,
        seed=a.seed, tiled=False,
    )
    if len(video) != 49:
        raise SystemExit("[FATAL] 生成フレーム数 %d != 49" % len(video))

    png_dir = os.path.join(a.out_dir, tag + "_png")
    os.makedirs(png_dir, exist_ok=True)
    for i, fr in enumerate(video):
        imageio.imwrite(os.path.join(png_dir, "gen_%04d.png" % i), np.array(fr))
    mkv = os.path.join(a.out_dir, "gen_%s.mkv" % tag)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "16",
                    "-i", os.path.join(png_dir, "gen_%04d.png"),
                    "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", "-g", "1",
                    "-flags", "+bitexact", "-fflags", "+bitexact", mkv], check=True)
    rd = imageio.get_reader(mkv)
    arr = np.stack([np.asarray(f)[..., :3] for f in rd])
    rd.close()
    png = np.stack([np.array(fr)[..., :3] for fr in video])
    if not np.array_equal(arr, png):
        raise SystemExit("[FATAL] mkv と生成フレームが bit 一致しない")
    print("wrote", mkv, arr.shape, "content sha(B):", hashlib.sha256(arr.tobytes()).hexdigest()[:16])
    save_video(video, os.path.join(a.out_dir, "gen_%s.mp4" % tag), fps=16, quality=9)
    print("next: python score_t_control_v2.py --gen %s --gt gt_target_center_prime.json" % mkv)


if __name__ == "__main__":
    main()
