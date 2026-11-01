#!/usr/bin/env bash
# run_seeds20.sh — 追加実験：Scene C′ で各 arm の seed を G5〜G20 まで追加（W12）
#   使い方: cd /workspace/w4/py && nohup bash run_seeds20.sh > /workspace/runs/seeds20.log 2>&1 &
# 計画・判定式・予測: docs/protocol/protocol_rev17_changes.md（実行前に凍結）
# 学習はしない。各 arm で最初に G1 を作り直し、Phase 0.8 と sha が一致しなければその arm は止まる。
set -u
cd /workspace/w4/py || exit 1
G=/workspace/runs/gen
echo "=== seeds20 start $(date -u) ==="
for A in A B C; do python infer_seeds.py --arm $A --seeds 4-19; done

# G1〜G20 を全部同じ採点器で採点し直す（G1〜G4 も含めて揃える）
mkdir -p $G/seeds20
for A in A B C; do for N in $(seq 1 20); do
  T=arm${A}_step500_G${N}
  [ -f $G/gen_$T.mkv ] || { echo "MISSING $T"; continue; }
  python score_t_control_v2.py --gen $G/gen_$T.mkv --gt gt_target_center_prime.json \
    --json $G/seeds20/$T.json | grep -E "T_control" | sed "s/^/$T  /"
done; done
ls $G/seeds20/*.json | wc -l
echo "=== seeds20 end $(date -u) ==="
[ "${STOP:-1}" = "1" ] && runpodctl stop pod "${RUNPOD_POD_ID:-}"
