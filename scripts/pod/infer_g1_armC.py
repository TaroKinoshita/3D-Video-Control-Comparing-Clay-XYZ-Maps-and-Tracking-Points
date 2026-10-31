# -*- coding: utf-8 -*-
"""偵察run 推論: 学習した g1_armC LoRA で armC_prime(C' control)から動画生成。
validate_lora/Wan2.1-VACE-1.3B.py がベース。机上設定を適用。"""
import os, torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

LORA = os.environ.get('LORA', '/workspace/runs/g1_armC/step-500.safetensors')
CTRL = os.environ.get('CTRL', '/workspace/w4/mkv/armC_prime.mkv')
REF  = os.environ.get('REF',  '/workspace/w4/mkv/ref_R0S0_original.png')
OUT  = os.environ.get('OUT',  '/workspace/runs/gen_g1_armC.mp4')
PROMPT = "static camera, simple matte geometric solids on a flat floor, plain even lighting, an orange hexagonal prism"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
# 学習LoRAを vace に load（学習の remove_prefix_in_ckpt pipe.vace. と対応、alpha=1）
pipe.load_lora(pipe.vace, LORA, alpha=1)
print("LoRA loaded:", LORA)

# control を49フレームのリストに、ref は1枚（PNGを直接）
from PIL import Image
ctrl = VideoData(CTRL, height=480, width=832)
ctrl = [ctrl[i] for i in range(49)]
ref  = Image.open(REF).convert("RGB").resize((832, 480))

# 机上で確定した設定: neg="", cfg 5.0, steps 50, num_frames 49, tiled False, seed 0(G1)
video = pipe(
    prompt=PROMPT,
    negative_prompt="",
    vace_video=ctrl,
    vace_reference_image=ref,
    num_frames=49,
    cfg_scale=5.0,
    num_inference_steps=50,
    sigma_shift=5.0,
    seed=0,
    tiled=False,
)
# FFV1 lossless で保存したいが save_video は mp4。判定は画素なので lossless が要る
# -> フレームを PNG 連番で出し、後段でFFV1化 or 直接判定
import numpy as np, imageio.v2 as imageio
os.makedirs('/workspace/runs/gen_png', exist_ok=True)
for i,fr in enumerate(video):
    imageio.imwrite('/workspace/runs/gen_png/gen_%04d.png'%i, np.array(fr))
print("wrote", len(video), "frames to /workspace/runs/gen_png/")
save_video(video, OUT, fps=15, quality=9)   # 目視用mp4も
print("wrote preview", OUT)
