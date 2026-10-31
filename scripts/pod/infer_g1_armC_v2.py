# -*- coding: utf-8 -*-
"""偵察run 推論 v2: モデルを実パス直指定（再DL回避、quota対策）。
ModelConfig の引数名が path でなければ ここを直す（下の PATHS を使う）。"""
import os, torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from PIL import Image
import numpy as np, imageio.v2 as imageio

DIT = "/workspace/DiffSynth-Studio/models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"
T5  = "/workspace/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE = "/workspace/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
LORA = "/workspace/runs/g1_armC/step-500.safetensors"
CTRL = "/workspace/w4/mkv/armC_prime.mkv"
REF  = "/workspace/w4/mkv/ref_R0S0_original.png"
OUT  = "/workspace/runs/gen_g1_armC.mp4"
PROMPT = "static camera, simple matte geometric solids on a flat floor, plain even lighting, an orange hexagonal prism"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(path=DIT),
        ModelConfig(path=T5),
        ModelConfig(path=VAE),
    ],
)
pipe.load_lora(pipe.vace, LORA, alpha=1)
print("LoRA loaded:", LORA)

ctrl = VideoData(CTRL, height=480, width=832)
ctrl = [ctrl[i] for i in range(49)]
ref  = Image.open(REF).convert("RGB").resize((832, 480))

video = pipe(
    prompt=PROMPT, negative_prompt="",
    vace_video=ctrl, vace_reference_image=ref,
    num_frames=49, cfg_scale=5.0, num_inference_steps=50, sigma_shift=5.0,
    seed=0, tiled=False,
)
os.makedirs('/workspace/runs/gen_png', exist_ok=True)
for i,fr in enumerate(video):
    imageio.imwrite('/workspace/runs/gen_png/gen_%04d.png'%i, np.array(fr))
print("wrote", len(video), "frames to /workspace/runs/gen_png/")
save_video(video, OUT, fps=15, quality=9)
print("wrote preview", OUT)
