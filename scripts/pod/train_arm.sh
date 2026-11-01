#!/usr/bin/env bash
# train_arm.sh — Phase 0.8 の学習 1 本（W9）
#   使い方:  bash train_arm.sh A          （1 本目 = S1。出力 /workspace/runs/g1_armA）
#            bash train_arm.sh C g2       （④ Training 変動の S2。出力 /workspace/runs/g2_armC）
# フラグは runs/g1_armC/training_args.json（W6 偵察 run）と同一。
# 違うのは metadata と output_path だけ（protocol §36.14 / §8）。
# 長い処理なので末尾で Pod を止める（&& ではなく ;）。止めたくなければ STOP=0。
set -u
ARM="${1:?usage: bash train_arm.sh A|B|C [g1|g2|g3]}"
RUN="${2:-g1}"
case "$ARM" in A|B|C) ;; *) echo "ARM must be A/B/C"; exit 1;; esac
case "$RUN" in g1|g2|g3) ;; *) echo "RUN must be g1/g2/g3 (= S1/S2/S3)"; exit 1;; esac
OUT=/workspace/runs/${RUN}_arm${ARM}
META=/workspace/w4/meta/metadata_arm${ARM}.jsonl

[ -f "$META" ] || { echo "missing $META"; exit 1; }
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "output dir not empty: $OUT（上書き防止で停止）"; exit 1
fi
mkdir -p "$OUT"

cd /workspace/DiffSynth-Studio || exit 1
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$OUT/vram.log"; sleep 2; done ) &
VRAM_PID=$!

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /workspace/w4/mkv \
  --dataset_metadata_path "$META" \
  --data_file_keys video,vace_video,vace_reference_image \
  --extra_inputs vace_video,vace_reference_image \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 500 \
  --save_steps 100 \
  --remove_prefix_in_ckpt pipe.vace. \
  --output_path "$OUT" \
  --lora_base_model vace \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  2>&1 | tee "$OUT/train.log"

kill $VRAM_PID 2>/dev/null
echo "=== PEAK VRAM ==="; sort -n "$OUT/vram.log" | tail -1
echo "=== OUTPUT ==="; ls -la "$OUT"
[ "${STOP:-1}" = "1" ] && runpodctl stop pod "${RUNPOD_POD_ID:-}"
