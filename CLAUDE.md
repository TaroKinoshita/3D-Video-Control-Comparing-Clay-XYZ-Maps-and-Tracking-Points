# CLAUDE.md

卒論プロジェクト「3D control 表現（Clay / World-XYZ / Tracking points）を Wan2.1-VACE-1.3B に LoRA で教えたときの比較」のリポジトリ。

## 最初に読むもの

1. **`HANDOFF.md`**：現状・環境・手順・罠・次の作業・ユーザーとのやり取り方が全部ある
2. `docs/thesis/thesis.md`：論文 v1.0
3. `docs/protocol/protocol.md`（rev.18）：凍結した全決定事項の正本

## 守ること

- **事前登録**：新しい実験の判定式・閾値・予測は、データを生成する前に `docs/protocol/protocol_revNN_changes.md` に書き、`protocol.md` に節として追記し、commit する。結果を見てからルールを変えない
- 結果は `docs/worklog/worklog_WNN_日付.md` に記録する。外れた予測は理由つきで書く
- 数値を推測しない。スクリプトや API の挙動は実物で確かめる
- 採点は `scripts/pod/score_t_control_v2.py`（v1 は使わない）
- control の動画は FFV1 gbrp bitexact のみ。照合は content sha 方式B
- GPU は RTX PRO 6000 Blackwell に固定（RunPod、EU-CZ-1、Volume `test_pod_volume`）。学習 run 予算は 5 / 5 使用済み
- Pod 用の実行スクリプトは `nohup` で動かし、最後に `runpodctl stop pod` を入れる。Pod では最初に `source /workspace/env.sh`

## ユーザーへの応答

日本語・カジュアル・結論から。素人前提で表や図を使い、1 ステップずつ。コマンドは短く区切りプレースホルダを入れない。詳細は `HANDOFF.md` §10。
