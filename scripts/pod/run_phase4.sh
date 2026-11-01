#!/usr/bin/env bash
# run_phase4.sh — ④ 床の測定を 1 本で回す（W10）
#   使い方: cd /workspace/w4/py && nohup bash run_phase4.sh > /workspace/runs/phase4.log 2>&1 &
# 前提: /workspace/w4/mkv_floor/arm{A,B,C}_prime_shift1.mkv を置いてあること
# 各段は ; でつなぐ（途中が失敗しても最後の Pod 停止は必ず実行される）。止めたくなければ STOP=0。
# 計画と予測: docs/protocol/protocol_rev15_changes.md（実行前に凍結）
set -u
cd /workspace/w4/py || exit 1
F=/workspace/w4/mkv_floor
echo "=== phase4 start $(date -u) ==="

# 1. Determinism floor：Phase 0.8 と同じ条件で再生成（Arm C は W9 の試運転で済み）
for A in A B; do python infer_arm.py --arm $A --suffix _rep; done

# 2. Micro-perturbation：control を右へ 1 px（G1）
for A in A B C; do python infer_arm.py --arm $A --ctrl $F/arm${A}_prime_shift1.mkv --suffix _shift1; done

# 3. Seed variability：G2 / G3 / G4（G1 は Phase 0.8 で済み）
for S in 1 2 3; do for A in A B C; do python infer_arm.py --arm $A --seed $S; done; done

# 4. Training-run 変動：代表 Arm C を S2 / S3 で学習（学習 run 予算の残り 2 本）→ G1 で推論
STOP=0 bash train_arm.sh C g2
STOP=0 bash train_arm.sh C g3
for R in g2 g3; do python infer_arm.py --arm C --run $R; done

# 5. 全部採点
for M in /workspace/runs/gen/gen_arm*.mkv; do
  python score_t_control_v2.py --gen "$M" --gt gt_target_center_prime.json --json "${M%.mkv}.score.json" | grep -E "^===|T_control|T_coherence|tracker_fail"
done
ls /workspace/runs/gen/*.mkv | wc -l
echo "=== phase4 end $(date -u) ==="
[ "${STOP:-1}" = "1" ] && runpodctl stop pod "${RUNPOD_POD_ID:-}"
