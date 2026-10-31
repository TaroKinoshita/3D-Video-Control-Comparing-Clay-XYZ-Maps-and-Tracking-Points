# 本実験前 統合サマリ — protocol rev.13 相当 / worklog W0–W8

**作成 2026-08-13。Phase 0.8（本実験）に入る直前の全体像。**

出典：`protocol.md` rev.10 本体 + rev.11 / rev.12 / rev.13 差分 + `worklog` W6 / W7 / W8。
本文書は**確定事項の索引**であり、`protocol.md` を置き換えるものではない。数値の一次情報は各節の参照先にある。

---

## 0. 現在の到達点（3行）

1. **準備段階は完了した。** 3 arm の control video（Scene C / Scene C′）、target video、参照画像、学習 metadata、すべて凍結値と一致を確認済み。
2. **本実験は未着手。** 学習 run 予算 5 本のうち 1 本消費（W6 偵察 run）、残 2 本。GPU は W7–W8 で未使用。
3. **次は Phase 0.8**（Arm A / Arm B の学習 2 本 + 推論）。開始前に一度止まる方針を W7 で固定済み。

---

## 1. 実験設計（不変の骨格）

### 1.1 研究質問

| | 内容 | 主張範囲 |
|---|---|---|
| **RQ1** 適応コスト | 3 表現は Wan2.1-VACE-1.3B に対する zero-shot 互換性、相対的な適応速度、共通実用閾値への到達 GPU コストにどう差を示すか | 表現一般の普遍的優劣は主張しない。「Wan2.1-VACE-1.3B へ LoRA で適応させる条件における差」に限定 |
| **RQ2** 制御性能と編集漏出 | Target Entity の物体軌道を反実仮想的に変更したとき、各表現は指定編集をどこまで実現し、Blender GT 上で不変な空間領域へどの程度漏出させるか | 今月は物体軌道編集のみ。カメラ編集と temporal spillover は 9 月以降 |

### 1.2 3 arm

| Arm | 表現 | 密度 | 外観情報 | 3D 情報 | Target 識別 | 時間対応 |
|---|---|---|---|---|---|---|
| **A** | Clay Previs | 密 | あり | 暗黙的 | control 内で明示しない | 暗黙的 |
| **B** | Object-Aware Dense World-XYZ | 密 | なし | 現在の world 位置 | control 内に明示 | 固定点対応なし |
| **C** | Object-Aware 3D Tracking | 疎 | なし | 3D 点軌道 | control 内に明示 | 固定色で明示 |

- **Primary comparison：Arm B vs Arm C**
- Secondary：Arm A vs B/C（Target ラベル条件が異なるため単軸比較ではない）
- **異なるのは control video のみ。** それ以外の全条件を 3 arm で共通にする（§8）。

### 1.3 固定パラメータ（§5）

| 項目 | 値 |
|---|---|
| フレーム数 | **49**（4k+1 でなければならない。50 / 80 は不可） |
| 潜在フレーム数 | 13（参照画像なし）/ **14**（参照画像あり = 本設計の実効値。学習・推論の両経路で 14） |
| 解像度 | **832 × 480** |
| fps | 16 |
| 焦点距離 | 50 mm / センサー幅 36 mm（水平画角 約 39.6°） |
| 画角 | 全条件固定 |

---

## 2. データ設計

### 2.1 割り当て（§33.2）

| データ | 用途 | 状態 |
|---|---|---|
| **Scene C R0S0 Original** | **Train**（3 arm control + tv） | ✅ 完成・凍結 |
| Scene C R0S0 Edited | RQ2 反実仮想テスト | ✅ 完成 |
| Scene C R1S0 / R0S1 | 分布シフト（評価は 9 月） | ✅ 生成のみ |
| **Scene C′ R0S0 Original**（variant `prime`） | **IID テスト** | ✅ **W7 完成** |
| Scene C-noocc | 遮蔽対照 | ❌ 未生成。**偵察 run で暗記に落ちなかったため不要** |
| Scene B | train 2 シーン目 | ❌ 未着手 |

**Edited を train に入れない理由**：モデルが編集後の軌道を暗記し、edited control 入力時の出力が control 追従ではなく記憶の再生になる（§33.3）。

### 2.2 Scene C′ = variant `prime` の定義（rev.12 変更1）

| 軸 | 定義 |
|---|---|
| X | `original` を完全継承（`-0.90 + 0.0375·index`）。`KEYFRAMES["prime"] = KEYFRAMES["original"]` |
| Y | 不変（0）。断面が動かないので制約2（面積不変）が自動で満たされる |
| Z | `location.z = dz_at(index)` |

```
dz_at(i) = DZ_A·(1 − cos(2π·i/DZ_PERIOD))/2   (i ≤ 35)
         = 0.0                                 (i > 35)
DZ_A = 0.85 / DZ_PERIOD = 35.0 / DZ_CLAMP_INDEX = 35
```

**最重要の実装注意（W7 最大の発見）**

> `location.z` に入れるのは **差分 dZ のみ**。Target 幾何中心の Z（−0.80）は**メッシュ頂点に焼かれている**（作法#14）ため、Scene C では `obj.location.z` は 0.0 のまま。
> `scene_c_prime_traj.json` の meta `Z0_note`（"location_z = Z0 + dZ"）は **誤り**。従うと中心が −1.60 に沈み床（top Z = −1.50）を貫通する。
> `Z0_placeholder = -0.770651` は点群重心の v 範囲 [252.3, 434.8] から depth 8.60 で逆算した推定値であり、幾何中心（depth 9.00）とは別物。
> **meta は使わず `keyframes[].dZ` のみ使う。**

### 2.3 Scene C′ の 4 制約（§33.4）

| # | 内容 | 実測余裕 |
|---|---|---|
| 1 | 速度上限 | **4.99%**（11.4014 / 12.0 px/frame）→ `DZ_A` / `DZ_PERIOD` は**凍結扱い** |
| 2 | 面積不変 | Y 不変により自動成立 |
| 3 | 始点一致 | index 0 で Scene C と bit 一致 → `vace_reference_image` が 1 枚に collapse |
| 4 | **遮蔽窓の固定** | index **36–40** を Scene C と揃える。画素レベルで実証済み |

**制約4 が壊れる経路（要注意）**：`build_scene_c.py` の LINEAR 強制ループを `array_index not in (0, 2)` にしないと、Z が BEZIER のまま通り補間オーバーシュートで遮蔽窓の dZ が非ゼロになる。**静かに壊れる。**

### 2.4 Scene C′ の主要実測値

| 項目 | 値 |
|---|---|
| dZ ピーク | **0.8482891**（index 17 / 18）。連続ピーク 0.85 には整数フレームで到達しない |
| Target 中心 v | 342.7161（dZ=0）→ 233.7999（peak）。移動 108.9161 px |
| Target 上端 v（depth 8.60） | 253.4367 → 139.4547。移動 113.9820 px |
| ピーク時頂点 Z（world） | −0.6517109 .. 0.7482891。`NORM_BOX["z"]` 上限 2.60 に対し余裕 **1.85 unit** |
| Target 画素の B 最大（Arm B 8bit） | 177（dZ=0）→ 197（peak）。振れ幅 **20 段階** |
| Target 可視フレーム数 | **44**（49 − 完全遮蔽 5） |

`index 12` と `index 23` の dZ が同値（0.6591312）なのは raised-cosine が f=17.5 を軸に対称で 12 + 23 = 35 だから。

---

## 3. 学習設定（確定）

### 3.1 metadata（rev.13 変更1）

| 対象 | 形式 | 生成 | 内容 |
|---|---|---|---|
| 学習 metadata | jsonl 3 本 | `make_metadata.py` v2 | **original のみ**。240 bytes / 1 行 / 4 列 |
| 資産 manifest | TSV（md5 / size / relpath） | `manifest.py` / `stage_pod_inputs.py` | 転送の bit 一致検証 |

**列名（確定）**：`video` / `prompt` / `vace_video` / `vace_reference_image`

**prompt（3 arm 共通、108 文字、ASCII のみ）**
```
static camera, simple matte geometric solids on a flat floor, plain even lighting, an orange hexagonal prism
```
運動を記述しない。軌道は control が指定する情報であり、prompt に書くと control の仕事を奪う。

**prime の jsonl は作らない。** train は Scene C R0S0 Original のみ（§33.4 a）。C′ は評価側であり、推論は jsonl を読まず frame の list を直接受け取る。

**凍結リテラル（3 arm すべて 240 bytes）**

| arm | sha256（先頭 16） |
|---|---|
| armA | `b98633171d33d47f` |
| armB | `8ed8deedd11f4ef7` |
| armC | `33e531b5c7888662` |

armC は W6 偵察 run で実際に使った `metadata_armC.jsonl` と **bit 一致を確認済み**。

### 3.2 本番の学習コマンド（§36.14）

| フラグ | 値 | 根拠 |
|---|---|---|
| `--dataset_base_path` | `/workspace/w4/mkv` | 1 本しか渡せないので 9 ファイルを集約 |
| `--dataset_metadata_path` | `/workspace/w4/meta/metadata_arm{A,B,C}.jsonl` | |
| `--num_epochs` | **500** | N=1 / A=1 で 500 optimizer steps |
| `--save_steps` | **100** | 20.4 GB → 209 MiB |
| `--enable_tensorboard_log` | 付ける | 横軸は micro-batch 単位 |
| `--use_gradient_checkpointing` | **明示する** | 明示すれば `training_args.json` に `true` と正しく記録される |

変更しないもの（W0 と同一）：
```
--data_file_keys video,vace_video,vace_reference_image
--extra_inputs vace_video,vace_reference_image
--height 480 --width 832 --num_frames 49
--dataset_repeat 1
--learning_rate 1e-4
--lora_base_model vace --lora_rank 32
--lora_target_modules q,k,v,o,ffn.0,ffn.2
--remove_prefix_in_ckpt pipe.vace.
```

### 3.3 DiffSynth 2.0.18 の実装上の事実（§35。推測厳禁）

| 事実 | 帰結 |
|---|---|
| `--num_steps` は**存在しない** | 学習量は `--num_epochs` 駆動 |
| `--batch_size` は**存在しない** | DataLoader が **B=1 固定** |
| `--lora_alpha` は**存在しない** | **alpha = rank** が実装既定 → load 側は `alpha=1` が正しいスケール |
| `--seed` は**存在しない** | 学習側 seed は非固定。**再現性は generation seed のみで担保** |
| `gradient_accumulation_steps` | **1**（既定のまま。g1_armC の実測） |
| `--save_steps` を省くと | epoch ごとに 500 個保存される |
| `ConstantLR` が既定引数で呼ばれる | **最初 5 step の lr が 1/3** になる |
| checkpoint 保存 | `step-{n}.safetensors` のみ。epoch 毎保存は発生しない（実測 0 個） |
| `training_args.json` | 再現性の記録として**不完全**（§36.12） |

### 3.4 seed（§9）

| 種別 | 具体値 |
|---|---|
| generation | G1=0 / G2=1 / G3=2 / G4=3（`pipe(seed=...)` に渡す） |
| training | S1=0 / S2=1 / S3=2 —— ただし `--seed` が無いので **S1=0 は「1 本目」のラベル** |

---

## 4. 推論設定（確定。rev.11 変更4）

`WanVideoPipeline.__call__` の実装既定を採用し、3 点のみ変更する。

| 引数 | 値 | 出所 |
|---|---|---|
| cfg_scale | 5.0 | 実装既定 |
| num_inference_steps | 50 | 実装既定 |
| sigma_shift | 5.0 | 実装既定 |
| negative_prompt | ""（空） | 実装既定、3 arm 統一 |
| denoising_strength / vace_scale | 1.0 / 1.0 | 実装既定 |
| **num_frames** | **49** | 既定 81 から変更 |
| **tiled** | **False** | bit 一致経路（VAE tiled=False）と統一 |
| **seed** | **G1=0** | §9 |

**LoRA load**
```python
pipe.load_lora(pipe.vace, "<output>/step-500.safetensors", alpha=1)
```
- `pipe.dit` ではなく **`pipe.vace`**（学習の `--remove_prefix_in_ckpt pipe.vace.` と対応）
- **`150 tensors are fused by LoRA` が出るか必ず確認。** 出なければ LoRA が乗っておらずベースモデルで推論している（**静かな失敗**）
- control は 49 フレームのリスト、ref は `Image.open(...).convert("RGB").resize((832,480))`

**モデルは実パス直指定**（`ModelConfig(path=...)`）＋ `DIFFSYNTH_SKIP_DOWNLOAD=True`。model_id 形式は redirect 先を探して空振りする。tokenizer も直指定。

---

## 5. 評価設計

### 5.1 ゲート（§37.7）

| | 式 | 面 |
|---|---|---|
| **Control** | median 中心誤差 **≤ 47 px** | Scene C′、GT 可視 **44 フレーム**（0–35 / 41–48） |
| **Coherence** | 個数 ≠ 1 フレーム率 **≤ 2 / 44** | 同上。Tracker failure を分母から除外 |

- 代表指標は **Target center trajectory error**（§12 の筆頭。px 単位で Target の画面サイズから根拠付けできる）。残り 6 指標は**報告のみ**（案 6-C）
- **47 px の錨**：Target 画面面積 28,029 px → 等価円半径 **94.5 px**、画面占有 7.0%
- **2/44 の根拠**：frame 35 / 41 が幾何的に判定困難な部分遮蔽フレーム。その本数だけを予算として認める
- 統計量は **median 主 / p95 副**。検出器の失敗フレームが max を支配するので平均も max も使えない

### 5.2 検出/追跡器（rev.11 変更6）

色セグメンテーション：
```
マスク条件： R>G, G>=B, (R−B)>40, R>80        （Target orange 分離）
Marker(黄) 誤検出対策： 重心が (658.76, 123.47) の ±30px の連結成分を除外
検出： 残りの最大連結成分の重心
遮蔽窓（GT の Target 画素 0）は tracker failure として分母から除外
```

**Evaluator floor**：median 3.05 / **p95 3.89 px**。**p95 ≪ 47 → 関門は有効、保険条項は発動しない。**

### 5.3 暗記/制御の判別（重要）

**T_control 単体は「追従の粗いゲート」で、暗記でも通りうる**（陽性対照で Scene C tv を入れると median 38 で PASS した）。判別は **Scene C center 誤差と C′ center 誤差の比較**で行う（`score_t_control.py` に実装）。

---

## 6. Claim Gate と予算

### 6.1 Claim Gate（§15）

以下の場合、表現固有の優位性を**主張しない**。

```
Arm 間差 ≤ Evaluator floor
Arm 間差 ≤ Pipeline floor
Arm 間差 ≤ 同 Arm 内の設定差
Arm 間差 ≤ Training-run 間変動
Arm 間差 ≤ 数値再現性の床（GPU 間 bf16 差 ≈ 0.3 ポイント）
```

RQ2 ではさらに latent edit-signal magnitude を確認する。入力側の編集信号強度が大きく異なる場合、spillover 差を表現の性質だけで説明しない。

### 6.2 学習 run 予算（rev.12 変更9）

| 項目 | 本数 |
|---|---|
| 総枠（§21） | 5 |
| 消費済み（W6 偵察 run = g1_armC = Arm C の S1） | 1 |
| Phase 0.8 で必要（Arm A / Arm B の S1） | 2 |
| **残余** | **2** |

**Pipeline floor 4 本（Determinism / Encoding / Micro-perturbation / Seed variability）は推論枠であり学習 run 予算を消費しない。**

### 6.3 W6 偵察 run の結果（rev.11 変更5。予測は外れた）

| 項目 | 値 |
|---|---|
| T_control median | **8.55 px**（≤47 PASS。Evaluator floor 3.89 の約 2 倍） |
| 暗記判別 | C′ center 誤差 **8.55** ≪ Scene C center 誤差 **57.47** → **control を読めている** |
| T_coherence | 2/44（閾値ぎりぎり FAIL。tracker_fail は境界 f34/f35） |

**帰結**：暗記に落ちていないため軌道変種の追加は不要。そのまま 3 arm へ進む。未決 #23（N）は**暫定 N=1**。

---

## 7. 成果物と凍結リテラル

### 7.1 content sha の計算方式（rev.12 変更4 で確定）

```
方式B（統一）： sha256( imageio デコードした (T,H,W,3) uint8 の tobytes )
方式A（旧記録）： sha256( str(shape) + tobytes )
```
**両記録値はどちらも正しく、ファイルの異常は存在しない。** Pod との照合に使うのは**方式B**。

### 7.2 mkv 8 本（すべて 49 frames / 832×480 / FFV1 level 3 / gbrp / g=1 / +bitexact）

| 本 | content sha (B) | content sha (A) | bytes |
|---|---|---|---|
| `tv_R0S0_original.mkv` | `6190557123fdc2b3` | `bba8f4b9e280b9e3` | 3,426,110 |
| `armA_original.mkv` | `33cabc1b11b72f45` | `246ed61a47ad39cd` | 2,217,662 |
| `armB_original.mkv` | `d31186af848c80dc` | `ab5eb47f98a1bd74` | 1,874,392 |
| `armC_original.mkv` | `7c08f981503edb34` | `46abbcded632a61b` | 1,088,684 |
| `tv_R0S0_prime.mkv` | `71d0253f9fdc3161` | `8fa4ae4bb827a020` | 3,466,263 |
| `armA_prime.mkv` | `3666302824d68ee5` | `1f273ee25c8b73c8` | 2,224,461 |
| `armB_prime.mkv` | `5a08e10cc228d230` | `77f1cf8b2e21b5f1` | 1,914,588 |
| `armC_prime.mkv` | `fde80efbe48dcb51` | `a7ac42f40f26b82d` | 1,066,438 |

**8 本すべて PNG ↔ mkv の bit 一致を確認済み**（不一致画素 0 / 最大絶対差 0）。検証経路は DiffSynth の `LoadVideo` と同一。

**入力経路の全区間が bit 一致**（§36.6）：
```
Blender PNG → FFV1 mkv → imageio → torchvision resize → UnifiedDataset → DiT
```
「control が壊れていた」という失敗モードは考えなくてよい。

### 7.3 参照画像

`ref_R0S0_original.png` = `tv\R0S0\original\tv_0000.png` をコピーするだけ（動画化しない）。frame 0 の Target 可視点数 72/72、画面面積 28,029 px。

**content sha の方式は未確定**（記録は先頭 24 桁 `df7c595457c81e35b47164ae` のみ）→ 未決 #27。file md5 でのみ照合する。

### 7.4 ファイル配置（rev.12 変更10 + W8 訂正）

| 種別 | 場所 |
|---|---|
| **スクリプト本体の唯一の正** | `C:\Bachelor Thesis\SummerWork\W2\.py\` |
| 成果物（PNG） | `W2\RENDER\{tv,geo,armA,armB}\<variant>\` |
| mkv | `W3\mkv\`（Arm C prime のみ `W4\mkv\`） |
| Pod 集約先 | `W8\pod_upload\mkv\` → `/workspace/w4/mkv` |
| Pod スクリプト | `/workspace/w4/py/` |

**W8 訂正**：`armC_original.mkv` は「Arm C は W4」規則の**例外**で `W3\mkv` にある。規則は W7 制定であり、armC_original は W6 以前の生成物のため規則の外にあった。ファイル自体は正規（sha / bytes 一致）。

---

## 8. 検査体系

### 8.1 自己検査の到達点（Scene C′）

| 段 | スクリプト | 出力 | SELF-CHECK |
|---|---|---|---|
| `.blend` | `build_scene_c.py --variant prime` | `scene_c_prime.blend`（98,219 B） | **PASS 32 / FAIL 0** |
| tv | `render_target_video.py` | 49 枚 | **PASS 104 / FAIL 0** |
| geo | `export_geometry.py` | 147 枚（position/depth/id × 49） | **PASS 150 / FAIL 0** |
| Arm A | `render_arm_a.py` | 49 枚 | **PASS 105 / FAIL 0** |
| Arm B | `encode_arm_b.py` | 49 枚 | **PASS 52 / FAIL 0** |
| metadata | `make_metadata.py` v2 | jsonl 3 本 | **selftest 8/8** |
| 集約 | `stage_pod_inputs.py` | 9 ファイル + manifest | **selftest 9/9** |

### 8.2 検査の空白と是正（W7 変更7 / W8 rev.13 変更3）

**「Target の Z が全フレームでゼロでも全検査が PASS する」状態が W7 で 4 箇所、W8 で 2 箇所判明した。** いずれも静かな失敗であり、実験結果を無言で無効化しうる。

| スクリプト | 空白 | 是正 |
|---|---|---|
| `render_target_video.py` | 位置検査ゼロ | `check_position()`（matrix_world → X / location.z / u / v 突合） |
| `render_arm_a.py` | 同上（clay は全同色なので位置が誤っても「それらしい絵」が出る） | 同上 |
| `export_geometry.py` | `target_f1` が index 0 限定、シルエット照合は u のみ | `target_v_top`（ID マスク最上行 → v） |
| `encode_arm_b.py` | **B チャンネル（Z 軸）が dZ>0 で一度も照合されない** | `target_b_max` + Target 帯 (140,242) 内包確認 |
| `make_metadata.py` v1 | 禁止語に `"prime"` が無い → **定数を prime に書き換えると全検査が PASS** | 禁止語に追加 |
| `make_metadata.py` v1 | 生成バイト列の回帰が無い → **prompt を 3 arm 同時に変えると素通り** | `FROZEN_JSONL` で sha 突合 |

### 8.3 壊したデータでの発火確認（作法）

**W7（レンダ系）**

| 壊し方 | tv 位置 | geo u | geo v_top | armB B max |
|---|---|---|---|---|
| Z 未反映（改修前の挙動） | 34/49 | 23 | 32 | 26/44 |
| `location.z = Z0 + dZ`（JSON の誤指示） | 49/49 | 23 | 44 | 44/44 |
| dZ が半分 | — | 3 | 32 | 24/44 |
| 遮蔽窓の clamp 除去 | — | 13 | — | — |
| **正しいデータ** | **0** | **0** | **0** | **0/44** |

「dZ が半分」は u 判定だけでは 3 件しか捕まらず v_top で 32 件捕まる。**u と v の両方を見る必要がある根拠。**

**W8（metadata 系、8 ケース全て期待どおり）**

| 壊し方 | v1 | v2 |
|---|---|---|
| prompt に空白 1 文字（1 arm） | 検出 | 検出 |
| vace_video の重複 | 検出 | 検出 |
| Edited の混入（1 arm） | 検出 | 検出 |
| prime の混入（1 arm） | 検出 | 検出 |
| **prime の混入（全 arm）** | **素通り** | **検出** |
| **prompt を 3 arm 同時に変更** | **素通り** | **検出** |
| dict の挿入順（発火しないのが正） | 素通り | 素通り |

### 8.4 検査の限界（記録）

- Arm B の B 最大値検査は **+5% のスケール誤差を検出できない**（8bit 量子化に埋もれる）。ただし最大誤差 0.042 unit < 量子化予算 0.044118 unit なので実験結果に影響しない
- 余裕が薄い箇所：制約1（速度）**4.99%** / Arm A `frac_0` **0.32%** / Arm B `target_b_min` 帯下限まで **3 段階**
- 参照画像 PNG は content sha 照合していない（未決 #27）

---

## 9. 環境と既知の罠

| 環境 | 症状 | 対処 |
|---|---|---|
| Blender 5.2.0 LTS | `save_as_mainfile` 後に `EXCEPTION_ACCESS_VIOLATION` | `comfyui_blender` アドオン起因。**`--factory-startup` 必須**。出力は無傷 |
| PowerShell 5.1（日本語 Win） | `Get-Content \| Add-Content` で日本語コメント文字化け | 既定 ANSI。**`cmd /c copy /b a.py + b.py new.py`** を使う |
| PowerShell | `^` が使えない | cmd の継続文字。**1 行で打つ**か バッククォート |
| PowerShell | `& $BL ...` が `BadExpression` | 開き直すと `$BL` が消える。毎回定義する |
| 一般 | 同名スクリプトが複数フォルダに存在 | **`Compare-Object` で中身を確認。タイムスタンプだけで決めない** |
| RunPod | `imageio` が無い（`ModuleNotFoundError`） | `pip install imageio imageio-ffmpeg`。**コンテナ破棄で消えるので再インストールが必要** |
| RunPod | GPU: RTX PRO 6000 Blackwell 96GB / EU-CZ-1 | ピーク VRAM **30,109 MiB**（参照画像あり。129 サンプル全て同値） |

---

## 10. 未決事項

| # | 内容 | 待ち条件 | 本実験を止めるか |
|---|---|---|---|
| 20 | 本番シーンでの色の一意性の実地検証 | 本番シーン構築後 | 止めない |
| 21 | 評価 GT 側で Arm A の Target を同定する方法 | C′ では geo の ID パス（ids==3）で同定可と実測。本番シーンでの方法は未決 | 止めない |
| 23 | train 軌道本数 N | **暫定 N=1**。3 arm の結果で確定 | 止めない |
| 24 | Occluder 除去方法（c-x / c-y / c-z） | 本実験時 | 止めない |
| 25 | T_coherence 2/44 が生成破綻か tracker 限界か | f34/f35 切り分け | 止めない（W7 todo #3） |
| 27 | 参照画像 PNG の content sha 方式 | `sha_probe.py` の PNG 対応 | 止めない |
| — | Pod の python 環境が train.py 実行環境と同一か | 次回 Pod 起動時に `which python` / `diffsynth.__file__` | **確認すべき** |
| — | `pod_upload.tsv` が Pod に未転送 | 次回 Pod 起動時 | 止めない（mkv 4 本は check-sha で画素確認済み） |

**解決済み**：#2（gradient_accumulation_steps=1）/ #13（参照画像）/ #25 旧（step の数え方）/ #26（metadata 形式）/ #26 旧（latent 時間長 14）/ #28（ConstantLR）/ content sha 方式（方式B）

---

## 11. 本実験前チェックリスト

### 11.1 完了済み

- [x] Scene C（original）4 本 + Scene C′（prime）4 本、すべて PNG ↔ mkv bit 一致
- [x] content sha 方式を方式B に統一、8 本すべて凍結値と一致
- [x] metadata jsonl 3 本を生成、armC は W6 の実物と bit 一致
- [x] 9 ファイルを 1 ディレクトリに集約、manifest TSV 生成
- [x] Pod へ転送、`/workspace/w4/mkv` に 9 ファイル確認
- [x] Pod 上で `--check-sha` PASS（mkv 4 本 sha(B) 一致 / shape (49,480,832,3)）
- [x] `/workspace/w4/meta/` に jsonl 3 本を書き出し

### 11.2 次回 Pod 起動時（5 分）

- [ ] `ls /workspace/w4/mkv` / `ls /workspace/w4/meta` —— volume 上に残っているか
- [ ] `pip install imageio imageio-ffmpeg` —— コンテナ破棄で消えていたら
- [ ] `which python` / `python -c "import diffsynth; print(diffsynth.__file__)"` —— **検査した環境と train.py の環境が同一か**
- [ ] `md5sum /workspace/w4/mkv/ref_R0S0_original.png` をローカルと照合
- [ ] `pod_upload.tsv` を転送して `manifest.py verify`

### 11.3 Phase 0.8 開始時

- [ ] 学習コマンドの全フラグを §36.14 と 1 行ずつ突合（特に `--num_epochs 500` / `--save_steps 100` / `--use_gradient_checkpointing` 明示）
- [ ] output dir が空か
- [ ] 推論時に **`150 tensors are fused by LoRA`** が出るか（出なければベースモデルで推論している）
- [ ] `alpha=1` で load しているか、`pipe.vace` を指しているか

---

## 12. 訂正記録の要約（予測外し）

**方針：作業前に予測を置き、外れたものを全て記録する。**

| # | 予測 | 結果 | 外した理由 |
|---|---|---|---|
| W6 | 3 arm すべて T_control を下回り暗記に落ちる | **外れ。control 追従を獲得** | — |
| W7-01 | Arm B は Z を動かすと正規化レンジからクリップ | **外れ。余裕 1.85 unit** | NORM_BOX の実値を見ていなかった |
| W7-02 | W2 が本体、W4 は残骸 | 外れ（結論は W2 で正しい） | **差分の中身を見る前にタイムスタンプで判断した** |
| W7-03 | geo の `target_v_top` 誤差最大は 1.0 未満 | 外れ（1.019） | 縁画素の被覆規則を過小評価 |
| W7-04 | C′ の Arm A sep は 0.87–0.92 に上がる | **外れ。0.8661（Scene C と同一）** | Target が明るくなる効果だけ見た。接地影が消えて**非 Target 側の平均も下がる** |
| W7-05 | Arm A の `frac_0` が dZ ピークで 3% を超える | **外れ。最大 2.684%（index 15）** | **`frac_0` のピークは dZ のピークと一致しない。** X も同時に動くため。**dZ の単変数で予測したのが誤り** |
| W7-06 | `armC_prime.mkv` の sha は W6 と一致 | 一度外し、方式差と特定 | 推測せず 6 方式総当たりして確定させた |
| W8-01 | `armC_original.mkv` は W4\mkv にある | **外れ。W3\mkv にあった** | 配置規則が W7 制定で、W6 以前の生成物が規則の外にあった |

**当たった予測（W7）**：Target 可視 44 フレーム / v_top なしは index 36–40 / 境界は index 5・30 / sep 最大 0.87 未満 / Arm B の B 最大 smoke 全一致 / dZ=0 フレームは bit 一致 15/15 / `png_to_ffv1.py` は無改修で使える / sha は方式B

---

## 13. 作法（実測で得た教訓）

1. **予測を先に置いて外れたら訂正明示。** W7-05 は予測していなければ気づかないまま通過していた
2. **assert は壊したデータで発火確認。** 「Z 未反映」は改修前コードで c1/c3/c4/NORM_BOX の全てを素通りした
3. **API・記録値を推測しない。** sha 不一致は「方式の違いだろう」で済ませず 6 方式を総当たり
4. **一度に一つ決める。** 案1/案2 を都度確定してから実装
5. **回帰を先に取る。** prime を焼く前に必ず original で回して既存成果物と bit 比較
6. **経路分離。** 作る側 = 閉形式 / 照合側 = リテラル（片方だけ直す事故の防止）
7. **タイムスタンプで判断せず `Compare-Object` を先に打つ**（W7-02）
8. **サイズは scale ではなくメッシュ頂点に焼く**（作法#14。これが `location.z = dZ` の理由）
9. **`open()` は必ず `encoding="utf-8"`**
10. **単変数で予測しない**（W7-05。`frac_0` は dZ だけでは決まらない）

---

## 14. 次の手（W7 todo の現状）

| # | 内容 | GPU | 状態 |
|---|---|---|---|
| 1 | 学習マニフェスト / メタデータ | 不要 | ✅ **W8 完了** |
| 2 | protocol 本体を rev.10 → rev.13 に反映 | 不要 | ⬜ 高 |
| 3 | T_coherence f34/f35 の切り分け | 一部 | ⬜ 中 |
| 4 | **Phase 0.8**（Arm A / Arm B 学習 2 本 + 推論） | 必要 | ⬜ **本実験** |
| 5 | Pipeline floor 4 本（推論のみ、学習枠を消費しない） | 必要 | ⬜ 4 と並行可 |
| 6 | Claim Gate（3 arm 差 vs Pipeline floor） | — | ⬜ 4・5 後 |

**準備完了。次は #2（GPU 不要）か #4（本実験）。**
