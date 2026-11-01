#!/usr/bin/env bash
# run_curve.sh — 学習曲線と zero-shot（RQ1 の「適応速度」「zero-shot 互換性」）（W12）
#   使い方: cd /workspace/w4/py && nohup bash run_curve.sh > /workspace/runs/curve.log 2>&1 &
# 計画・予測: docs/protocol/protocol_rev18_changes.md（実行前に凍結）。学習はしない。
set -u
cd /workspace/w4/py || exit 1
G=/workspace/runs/gen
echo "=== curve start $(date -u) ==="
for A in A B C; do
  python infer_seeds.py --arm $A --seeds 0-3 --no-lora
  for S in 100 200 300 400; do python infer_seeds.py --arm $A --seeds 0-3 --step $S; done
done
mkdir -p $G/curve
for A in A B C; do for S in step0 step100 step200 step300 step400; do for N in 1 2 3 4; do
  T=arm${A}_${S}_G${N}
  [ -f $G/gen_$T.mkv ] || { echo "MISSING $T"; continue; }
  python score_t_control_v2.py --gen $G/gen_$T.mkv --gt gt_target_center_prime.json --json $G/curve/$T.json \
    | grep -E "T_control" | sed "s/^/$T  /"
done; done; done
ls $G/curve/*.json | wc -l
echo "=== curve end $(date -u) ==="
[ "${STOP:-1}" = "1" ] && runpodctl stop pod "${RUNPOD_POD_ID:-}"
