# セッションノート：W0 完了（2026-08-01）

## 結論

**W0（Day 1-3）が閉じた。今月の最大リスクが消えた。**

`--lora_base_model vace` で LoRA が 1 step 回り、VRAM 実測値も取れた。
未決定事項が 2.5 個クローズ。

---

## 1. 到達点

| 項目 | 結果 |
|---|---|
| LoRA 1 step | **成功** |
| Peak VRAM | **30,109 MiB（29.4 GB）** |
| 出力 LoRA | `epoch-0.safetensors` / **43.8 MB** |
| 節約オプション | **不要**（gradient checkpointing なしで通った） |
| かかった時間 | 約 3 時間（うち大半がダウンロードと貼り付け事故） |
| かかった費用 | 概算 $6〜8 |

### LoRA 43.8MB が意味すること

rank 32 / 対象 `q,k,v,o,ffn.0,ffn.2` で 43.8MB は、
**VACE ブランチのみにアダプタが載っている**ことの裏付け。
本体 DiT（1.3B）に載っていれば桁が違う。

→ **RQ1 の前提（適応コストを VACE 側だけで測る）が実機で成立した。**

---

## 2. 確定した環境（protocol.md §8 に転記する）

```
GPU          : NVIDIA RTX PRO 6000 Blackwell 96GB
Provider     : RunPod / EU-CZ-1
Driver       : 595.71.05
CUDA         : 13.2 (driver)
PyTorch      : 2.8.0+cu128
Compute Cap  : sm_120 (12, 0)
Python       : 3.12.3
DiffSynth    : 2.0.18
transformers : 5.14.1
numpy        : 2.1.2
librosa      : 追加インストール済み（依存漏れ対応）
Peak VRAM    : 30,109 MiB（49f / 832×480 / rank32 / 節約なし）
確認日       : 2026-08-01
```

**Template 名を控えること。** 次回同じテンプレートを選ばないと
torch 2.8.0+cu128 以外を引く可能性がある。

§15 の「再現性の床」を測る際、この情報がないと比較できない。

---

## 3. 未決定事項の更新（protocol.md §18）

| # | 項目 | 状態 |
|---|---|---|
| **1** | LoRA学習が1 step回るか | **CLOSED. 回る** |
| **2** | クラウドGPU未確保 | **CLOSED. RunPod / EU-CZ-1** |
| **3** | 49フレームでメモリが足りるか | **CLOSED. 29.4GB / 96GB** |
| 4 | Arm C の時間圧縮耐性 | **未。次の最優先** |
| 5 | Depth規約 | 未。Blender前 |
| **6** | lora_alpha / batch size / grad accum | **一部CLOSED**（下記） |
| 7 | T_control / T_coherence | 未。Main Test前 |
| 8 | 検出・追跡器の選定 | 未。Day 25-31前 |

### #6 の内訳

- `--lora_alpha` … **オプションが存在しない。** DiffSynth 内部で固定。
  決定事項ではなく「フレームワークの仕様」として記録する。
  論文には「DiffSynth-Studio の既定に従う」と書く。
- `--batch_size` … **存在しない。** デバイスあたり 1 本固定。
- `gradient_accumulation_steps` … 既定 1 のまま。
  実効バッチ 1。3 arm で揃えれば公平性は保たれる。

→ **残るのは実質ゼロ。#6 はクローズ扱いでよい。**

---

## 4. 予測が外れた点（worklog.md §6 に記録）

| 予測 | 実際 | 何を間違えたか |
|---|---|---|
| 公式サンプルは 17〜21 フレーム | **1.3B は 49。17 は 14B 用** | 14B のスクリプトを 1.3B のものと混同していた |
| 49f でメモリが足りるか不明 | **29.4GB。96GB の 3 割** | offload 前提で見積もっていた |
| 必要 GPU 時間 100〜500h（別案） | **30〜50h** | 81 フレーム・1,000 step の旧計画に基づく試算だった |
| RunPod 最低入金 $50 | **未確認だった** | 出典なしで数字を書いた |
| protocol.md への書き込みが未完 | **前回セッションで完了済み** | 引継ぎ文の状態表が古かった |

**外れは全部「思ったより楽」の方向。** 逆方向の外れがなかったのは良い兆候。

---

## 5. DiffSynth の仕様で判明したこと

### 5.1 step 数の作り方

`--max_steps` は**存在しない。** 掛け算で作る。

```
step数 = 動画本数 × dataset_repeat × num_epochs ÷ gradient_accumulation_steps
```

| 設定 | step数 |
|---|---|
| `repeat 1` × `epochs 1` | 1（今回のテスト） |
| `repeat 100` × `epochs 2` | 200（公式サンプル） |
| `repeat 500` × `epochs 1` | **500（protocol.md の設定）** |

### 5.2 データセットの形式

`metadata.csv` の列（BOM 付き UTF-8）：

```
video, prompt, vace_video, vace_reference_image
```

| 列 | 中身 | 3-arm で何になるか |
|---|---|---|
| `video` | 正解の動画 | Blender の本レンダー |
| `prompt` | テキスト | シーン説明 |
| `vace_video` | Control Video | **Arm A/B/C で差し替わる箇所** |
| `vace_reference_image` | 参照画像 | **扱いを決める必要あり（下記）** |

**Day 11-17 の Blender エクスポータは、最終的にこの 4 列を吐けばよい。**

自前 CSV を書くときは **BOM なし UTF-8** で保存する（Excel で作ると BOM が付く）。

### 5.3 1.3B 用の LoRA スクリプトは存在しない

```
lora/Wan2.1-VACE-14B.sh   … ある
lora/Wan2.1-VACE-1.3B.sh  … ない
full/Wan2.1-VACE-1.3B.sh  … ある
```

→ **`full/1.3B.sh`（モデルID・49f）＋ `lora/14B.sh`（LoRA指定）を合成して作った。**
protocol.md §5 の値が、この 2 ファイルの組み合わせと完全に一致していた。

### 5.4 学習率が 2 種類ある

```
full/1.3B    5e-5   # コメント「原論文と揃えた」
lora/14B     1e-4
```

protocol.md は 1e-4。**LoRA サンプル側と一致しているので問題なし。**
ただし論文では「full finetune は 5e-5、LoRA は 1e-4」と区別して書く必要がある。

### 5.5 メモリ節約の階段（今回は全部不要だった）

弱い順：

| オプション | 効果 | 速度 |
|---|---|---|
| `--use_gradient_checkpointing` | 中間データを捨てて再計算 | 20〜30% 遅い |
| `--use_gradient_checkpointing_offload` | それを CPU に退避 | もっと遅い |
| `--enable_model_cpu_offload` | 重みを 1 層ずつ GPU に載せる | かなり遅い |
| `--fp8_models` | 指定モデルを fp8 に | 速い。精度低下 |

**OOM が出たら 1 段ずつ足す。まとめて足さない**（どれが効いたか分からなくなる）。

---

## 6. 新たに気づいた要検討事項

### 6.1 `vace_reference_image` の扱い（未決定・要 protocol 追記）

3 arm で参照画像をどう扱うか未定。選択肢：

- 3 arm で同一の参照画像を使う
- そもそも使わない（`--extra_inputs` から外す）

**RQ1 の交絡要因になりうる。** 参照画像が強い手がかりを与えると、
Control Video の表現差が薄まる可能性がある。

### 6.2 検証スクリプトが `tiled=True` で生成している

`validate_lora/Wan2.1-VACE-1.3B.py` は生成時の VAE デコードで tiling を使っている。
**Arm C の点が生成後にどう潰れるかを見る際に関係する。**

---

## 7. GPU の判断

### 7.1 なぜ RTX PRO 6000 を選んだか

当初 A100 80GB（$1.39）を推していたが、実際の在庫を見て変更。

| GPU | VRAM | 価格 | 在庫（EU） |
|---|---|---|---|
| A100 PCIe/SXM | 80GB | $1.39/$1.49 | **北米のみ** |
| **RTX PRO 6000** | **96GB** | **$1.99** | **EU にあり（Med）** |
| H100 SXM | 80GB | $2.99 | Low |
| H200 SXM | 141GB | $4.39 | Low |

- A100 との差額は今月 **$18〜30 程度**。北米に行く価値はない
- PRO 6000 は Blackwell（2025年）なので世代リスクがあったが、**$0.3 の事前チェックで問題なしを確認**
- step 単価は H100 とほぼ同じ（$0.0018 vs $0.0017）。**デバッグ主体の期間は安い方が得**

### 7.2 48GB に降りられるか

**降りられる。ただし今日ではない。**

```
Peak 29.4GB → A6000 / A40 48GB（$0.44〜0.53）に入る
              RTX 4090 24GB には入らない
```

降りない理由：

1. 29.4GB は **1 step の値**。500 step では optimizer 状態や断片化で数 GB 増える
2. 降りると**壁時計が 2 倍**になる。デバッグ期間は速い方が得

→ **Phase 0.8 ゲート（Day 25-31）の直前に、500 step の実測を見てから判断する。**

### 7.3 ハードウェア固定のルール

| 工程 | 変更可否 |
|---|---|
| W0（環境確認） | 自由 |
| 動画版 VAE テスト | 自由（VAE は決定的） |
| Day 18-24（Phase 0.0 本番） | 変更可。ただし記録する |
| **Day 25-31（Phase 0.8 ゲート）** | **固定必須。ここが Arm 比較の本体** |

固定すべき項目：GPU 型番 / provider / region / Docker image / CUDA / PyTorch / CPU・RAM / ストレージ方式。

---

## 8. 費用の実績と見通し

### 8.1 単価

```
step 時間      3〜4 秒（推定）
1 step         $0.0015
500 step run   $0.81
生成 1 clip    $0.035
```

**Phase 0.8 ゲート（3 arm）+ 変動測定 = 約 $4。**

### 8.2 今月の見込み

| 項目 | 費用 |
|---|---|
| 環境構築（済） | $6〜8 |
| 動画版 VAE テスト | $3 |
| デバッグ | $11〜28 |
| 学習 5 run + 検証生成 | $8 |
| Claim Gate 床測定 | $3 |
| Network Volume 50GB | $3.50/月 |
| **合計** | **$40〜60** |

### 8.3 卒論完走まで

9月以降の Main Training / Scene 追加 / 外部 Depth 参照を含めて
**通算 $150〜200 のオーダー。** 自費で成立する規模。

---

## 9. RunPod の運用ルール

### 9.1 課金の構造

**プリペイド。入金額が上限。** 残高 $0 で全 Pod が自動停止する。

| 操作 | GPU | Container Disk |
|---|---|---|
| 稼働中 | $1.99/hr | $0.10/GB/月 |
| **Stop** | **$0** | **$0.20/GB/月（単価が上がる）** |
| **Terminate** | **$0** | **$0** |

Network Volume（50GB / $3.50 月）は Pod と無関係に常時課金。

### 9.2 設定

```
[x] 入金 $30（それ以上入れない）
[x] auto-payment には触らない（オフのまま）
[ ] 低残高通知しきい値 $15
[ ] 2段階認証
```

**auto-payment がプリペイド上限を破る唯一の機能。** 有効化しない。

しきい値 $15 は「残高が減った＝止め忘れ」の検知器として使う
（通常 3 週間で到達するので、2 日で到達したら異常）。

### 9.3 止め忘れ対策

```bash
# 長い処理の末尾に必ず付ける。&& ではなく ; にする
python なにか.py ; runpodctl stop pod $RUNPOD_POD_ID
```

`&&` だと OOM で落ちたときに停止コマンドが実行されない。

**放置 1 晩（12h）= $17。実験本体（$4）の 4 倍。** これが最大の出費源。

### 9.4 データ配置の原則

```
/workspace（Network Volume）  永住。Pod を消しても残る
Container Disk                使い捨て。作業終了時は Terminate
```

Volume を付けた Pod は、残高切れでも **停止されるだけでデータは保全される。**
Volume 無しの Pod は terminate されてデータが消える。

### 9.5 API キーの扱い

**protocol.md は OneDrive 同期されるので、キーを書くと同期先まで広がる。**

```
NG : ノートブックのセルに直書き（git に commit される）
NG : 設定ファイルや protocol.md に記載
OK : 環境変数 / Pod の secret 機能
```

---

## 10. 環境の復帰手順（次回 Pod を立てるとき）

### Pod 設定

```
Network volume : test_pod_volume（EU-CZ-1 / 50GB）
GPU            : RTX PRO 6000
Template       : 前回と同じ（要メモ）
Container Disk : 20GB
Mount Path     : /workspace
```

### 起動後の 2 行

```bash
echo 'source /workspace/env.sh' >> ~/.bashrc
source /workspace/env.sh
```

### 確認

```bash
python -c "import torch, diffsynth; print(torch.__version__, torch.cuda.get_device_capability())"
du -sh /workspace/modelscope
```

`2.8.0+cu128 (12, 0)` と `18G` が出れば復帰完了。**2 分。**

### 倉庫の中身

```
/workspace/venv          Python 環境（--system-site-packages で torch 継承）
/workspace/env.sh        環境変数一式
/workspace/modelscope    モデル 18GB
/workspace/hf            HuggingFace キャッシュ
/workspace/pipcache      pip キャッシュ
/workspace/constraints.txt  torch/numpy のバージョン固定
/workspace/w0.sh         1 step テストスクリプト
/workspace/runs/         学習結果・ログ
/workspace/DiffSynth-Studio/
```

**残り容量：50GB 中 約 30GB 空き。**
`du -sh /workspace/*` で実測すること（`df` は共有 FS の数字が出るので使えない）。

---

## 11. ハマったポイント（再発防止）

| 症状 | 原因 | 対処 |
|---|---|---|
| `ModuleNotFoundError: librosa` | DiffSynth の依存漏れ。音声を使わなくても import される | `pip install librosa -c /workspace/constraints.txt` |
| コマンドが途中で切れる | **Web Terminal が長い複数行の貼り付けを取りこぼす** | `echo '...' >> file.sh` で小分けに積んでから `bash file.sh` |
| `while true` でプロンプトが返らない | 無限ループなので当然 | `( ... ) &` で裏に回すか、別ターミナルで実行 |
| `df` が 851TB を返す | 共有 FS（MooseFS）全体の数字 | `du -sh /workspace/*` を使う |
| pip が torch を差し替える恐れ | 依存解決の副作用 | `-c /workspace/constraints.txt` を常に付け、**インストール後に必ず torch のバージョンを確認** |

### 貼り付け事故の回避パターン

```bash
echo 'A="--foo bar"' >> script.sh
echo 'A="$A --baz qux"' >> script.sh
# ...
wc -l script.sh      # 行数を数えて欠落を検出
cat script.sh        # 目視確認
bash script.sh
```

**必ず `wc -l` と `cat` で検証してから実行する。**

---

## 12. 進捗

```
W0 環境      : 完了 ✅
Day 4-10     : 静止画完了 / 動画（時間圧縮）が未検証
Day 11-17    : 未着手（Blender エクスポータ）
Day 18-24    : 未着手（Phase 0.0 本番）
Day 25-31    : 未着手（Phase 0.8 ゲート）
```

前回卒論の資産（camera pose エクスポート/検証スクリプト、EXR 多層出力、
法線チェック、既知バグリスト）で Day 11-17 は 7日→4日に短縮できる見込み。

---

## 13. 次にやること

| 優先 | 作業 | GPU | 理由 |
|---|---|---|---|
| **1** | **動画版 VAE テスト**（Arm C の速度上限） | 要る | **Arm C が壊れる可能性が最も高い箇所。結果が Blender のアニメ速度を縛る** |
| 2 | Depth 規約を決める | 不要 | Blender 前に必要。前回卒論のシーンがあれば 15 分 |
| 3 | Blender エクスポータ | 不要 | 1 の結果を待つ |
| 4 | protocol.md §8 に環境固定の項目を追記 | 不要 | 15 分 |

### 動画版 VAE テストとは何か（設計メモ）

**要するに「点が速く動くと VAE を通したときに線になってボケる。どこまで耐えるか測る」。**

```
入力  49 フレーム × 480 × 832
         ↓ VAE encode
潜在  13 フレーム × 60 × 104
       ↑ 4分の1に圧縮。ここが未検証
```

Wan VAE は causal 構造で、**frame 1 は単独、以降は 4 フレームずつ束ねる**（1 + 12×4 = 49）。
→ frame 1 だけ 4 倍の容量を持つ。**静止画テストは事実上その状態で測っていた。**

4 フレームを 1 枚に押し込むと、動く点は**線状の足跡**になる。

```
点は 1 フレームあたり約 17px 動く（16fps で 3 秒画面横断）
潜在 1 枚 = 4 フレーム = 間隔 3 つ
足跡の長さ = 17 × 3 = 51px
```

§16.3 の実測では **40px 離れると identity 47.0%（12 色なので色が重複）。**
→ **51px は崩壊域の向こう側。そのままだと壊れる。**

#### スイープの構造（3 本目）

| | ラウンド1 | ラウンド2 | **ラウンド3** |
|---|---|---|---|
| 変える軸 | 点の半径 | 色の付け方 × 密度 | **点の速度** |
| 固定 | — | 半径 3px | 半径 3px / パレット12色 / d0.3 |
| 結論 | 半径 3px | パレット方式 | **V_max = ?** |

#### 事前予測（外れたら worklog に記録）

| 速度 | identity(20px) 予測 | 根拠 |
|---|---|---|
| 0 px/frame | 88〜92% | 時間圧縮の素の劣化ぶんだけ |
| 5 px/frame | 80〜88% | 足跡 15px。20px 以内 |
| **17 px/frame** | **50〜70%** | **足跡 51px。40px の崩壊域** |
| 34 px/frame | 25〜45% | 足跡 102px。確実に重複 |

**17px で 80% を超えたら「足跡 = 40px 問題」というモデルが誤り。**
その場合は別機序（VAE が動きを補間して代表位置に潰す等）を疑う。

#### 測定の前提条件

**tiling なしで通すこと。** メモリ不足で tiled VAE に切り替わると、
劣化が「時間圧縮のせい」か「tile 境界のせい」か切り分けられなくなる。

→ **80GB 以上の環境で回す。** 29.4GB の実測が出たので PRO 6000 で余裕。

#### 下流への影響

```
V_max が出る
  → protocol.md §13「新たに判明した制約」を更新
  → Blender の物体アニメーション設計に上限速度が入る（Day 11-17）
  → Scene B / C の Target 移動速度が決まる
  → §10 の frame 区間定義（編集区間 12-23 で停止または減速）と整合するか確認
```

**エクスポータを作る前に知っておかないと、作ったアニメが速すぎて全部作り直しになる。**
ラウンド 1〜2 を Blender より前に置いたのと同じロジック。

---

## 14. 未使用ファイル

以下は今回のセッションで未添付。動画版 VAE テストの改修に入る際に必要：

- `make_test_dots.py` … 点画像ジェネレータ
- `measure_vae_roundtrip.py` … VAE 往復の採点スクリプト

---

## 15. 会話の進め方（次回に引き継ぐ）

- 前提から説明して、**1 ステップずつ**進める。まとめて全部出さない
- 結論から先に、カジュアルに。お世辞や誇張は不要
- 図や表で直感的に分かるように
- **予測が外れたときは訂正を明示する**（worklog.md に記録するため）
- **Web Terminal への貼り付けは短く小分けに。長い複数行は落ちる**
