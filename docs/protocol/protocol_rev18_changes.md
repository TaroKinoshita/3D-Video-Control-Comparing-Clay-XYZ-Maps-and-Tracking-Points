# protocol rev.17 → rev.18 更新差分（W12、2026-11-02）— 学習曲線・zero-shot・Mask IoU（事前登録）

**実行前に凍結する。** 学習はしない（既存の checkpoint と既存の生成動画だけを使う）。

---

## 変更1：学習曲線と zero-shot（RQ1 の「適応速度」「zero-shot 互換性」）

RQ1（§1）は「zero-shot 互換性、相対的な適応速度、共通実用閾値へ到達する GPU コスト」を問うが、
これまで測ったのは step-500 の精度だけだった。保存済みの checkpoint で残りを測る。

| 項目 | 決定 |
|---|---|
| LoRA | 各 arm の S1。step 0（LoRA なし）/ 100 / 200 / 300 / 400。step 500 は既存の G1〜G4 を使う |
| seed | G1〜G4 |
| 面・採点 | Scene C′、`score_t_control_v2.py`（rev.14） |
| 生成 | 3 arm × 5 step × 4 seed = **60 本**（`run_curve.sh`、約 1.75 時間） |

**代表値**：各 (arm, step) の T_control median を 4 seed で取り、その**中央値**。

**到達 step**：代表値が 47 px 以下になる最小の step（0 / 100 / 200 / 300 / 400 / 500）。
GPU コストへの換算：1 step ≈ 7 秒（W9 実測）。

**報告のみ。** 4 seed しかないので、arm 間の差を Claim Gate にはかけない（§44 の「seed ≥ 10」に届かない）。
学習曲線の形と到達 step を記述として報告する。

### 予測

| # | 予測 |
|---|---|
| C1 | step 0（zero-shot）：**A だけが 47 px 以下**。B と C は超える（XYZ 色と点は VACE の学習時の control に似ていない。粘土は形と陰影があるので近い） |
| C2 | 到達 step：A 0〜100、B 100〜200、C 200〜300 |
| C3 | step 100 以降は、各 step で誤差が A ≤ B < C の順に並ぶ |

---

## 変更2：Target Mask IoU（探索的、報告のみ）

既存の G1〜G20（step-500、60 本）に、§12 の Control fidelity の 1 指標 **Target Mask IoU** を当てる。
これらの動画の T_control はすでに見ているので、**探索的な分析**として扱い、判定には使わない。

- 生成側：tracker と同じ色条件の全画素から Marker 近傍の成分を除いたもの（複製や欠けも形として数える）
- GT：Scene C′ の ID パス（`data/points/target_mask_prime.bin`）
- 各生成で、GT 可視 44 フレームの IoU の median
- 採点器の上限：正解動画（tv_R0S0_prime）を採点すると **0.968**

### 予測

| # | 予測 |
|---|---|
| I1 | IoU の 20 seed の中央値：A ≥ B > C。A と B は 0.85 以上、C は 0.75〜0.85 |
