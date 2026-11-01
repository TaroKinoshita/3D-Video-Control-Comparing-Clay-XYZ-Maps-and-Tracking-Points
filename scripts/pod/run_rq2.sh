#!/usr/bin/env bash
# run_rq2.sh — ⑤ RQ2（編集の漏れ）の生成 24 本 + 採点（W11）
#   使い方: cd /workspace/w4/py && nohup bash run_rq2.sh > /workspace/runs/rq2.log 2>&1 &
# 前提: /workspace/w4/mkv_rq2/arm{A,B,C}_edited.mkv、
#       /workspace/w4/py/{score_rq2.py, m_invariant_R0S0.bin, gt_target_center_edited.json}
# 計画と予測: docs/protocol/protocol_rev16_changes.md（実行前に凍結）
# 学習はしない（S1 の LoRA = g1_arm{A,B,C} を使う）。止めたくなければ STOP=0。
set -u
cd /workspace/w4/py || exit 1
O=/workspace/w4/mkv
E=/workspace/w4/mkv_rq2
G=/workspace/runs/gen
echo "=== rq2 start $(date -u) ==="

for S in 0 1 2 3; do
  for A in A B C; do
    python infer_arm.py --arm $A --seed $S --ctrl $O/arm${A}_original.mkv --suffix _rq2orig
    python infer_arm.py --arm $A --seed $S --ctrl $E/arm${A}_edited.mkv   --suffix _rq2edit
  done
done

for S in 1 2 3 4; do
  for A in A B C; do
    T=arm${A}_step500_G${S}
    python score_rq2.py --orig $G/gen_${T}_rq2orig.mkv --edit $G/gen_${T}_rq2edit.mkv \
      --mask m_invariant_R0S0.bin --gt-edit gt_target_center_edited.json \
      --gt-prime gt_target_center_prime.json --json $G/rq2_${T}.json
  done
done
ls $G/gen_*_rq2*.mkv | wc -l
echo "=== rq2 end $(date -u) ==="
[ "${STOP:-1}" = "1" ] && runpodctl stop pod "${RUNPOD_POD_ID:-}"
