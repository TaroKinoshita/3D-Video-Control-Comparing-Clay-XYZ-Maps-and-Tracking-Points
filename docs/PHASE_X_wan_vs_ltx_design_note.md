# Phase X — Wan vs LTX Cross-Family Validation 構想計画

**Status: DESIGN FROZEN / NOT STARTED**
**作成 2026-08-13。Main experiment（Phase 0.8）着手前に構想のみ確定。**

---

## 0. これは何か

Main experiment（Wan2.1-VACE-1.3B、3 arm）で観測された representation-level tendency が、
**異なる model family かつ異なる conditioning mechanism** でも観測されるかを探索的に検証する、
追加フェーズの構想。

**現時点で実行しない。** Phase 0.8 → Pipeline floor → Claim Gate が終わるまで着手しない。
本文書は「やるとしたら何をどう測るか」を凍結しておくためのもの。

---

## 1. なぜ Wan2.2 ではなく LTX なのか

### 1.1 答える問いが違う

| 追加対象 | 答える問い | 種別 |
|---|---|---|
| Wan2.1 → Wan2.2 | 同系列で世代を変えても結果は残るか | within-family replication |
| Wan2.1 → LTX | **別系列の video diffusion model でも残るか** | **cross-family replication** |

「この結果は Wan 特有なのでは?」という反論に効くのは後者。本研究が知りたい
「3D control 表現そのものにある程度一般的な性質があるのか」に直接効くのも後者。

### 1.2 Wan 3.0 は選択肢にならない（2026-08-13 確認）

Wan 3.0 は 2026年8月6日に public beta として発表されたが、**API 限定で重みは非公開**。
アプリケーション審査制。オープン重みは **Wan 2.2（2025年8月）で止まっており**、
2.5 / 2.6 / 2.7 はコアの動画モデルの重みを一切公開していない。

**RQ1（LoRA での適応コスト比較）は重みにアクセスして fine-tuning できることが前提。**
API 経由の生成しかできない Wan 3.0 では実験の型自体が成立しない。

したがって「Wan2.1 が古くなるのでは」という懸念に対して、**乗り換え先のオープン重み
モデルは Wan 系列に存在しない**。Wan2.1 を使い続けるのが正しい。

### 1.3 容量クラスを揃える

| モデル | パラメータ |
|---|---|
| Wan2.1-VACE-1.3B（Main） | 1.3B |
| **LTXV 2B 0.9.5** | **2B** |
| LTX-2 | 19B |
| LTX-2.5 | 22B |
| Wan2.2 A14B | A14B / MoE total 約27B |

19B や 22B を持ってくると cross-family validation 以前に**容量差15倍**で交絡する。
LTXV 2B が唯一「同じ小型 open-weight video DiT クラス」に近い。

---

## 2. 却下した案

### 2.1 VACE-LTX-Video-0.9（却下）

「同じ VACE という制御思想を Wan backbone と LTX backbone で比較すればきれいな
replication になる」という案。実際に `ali-vilab/VACE-LTX-Video-0.9` は存在する。

**却下理由：**

- 2025年3月31日リリースの研究検証版。公式自身が VACE 論文の experimental research
  validation 用と位置づけている
- VACE repo で確認できるのは inference path のみ。**学習経路が整備されていない**
- 依存関係が `ltx-video@git+...@ltx-video-0.9.1` で、公式が Wan と競合する可能性を明記

### 2.2 LTX-2 / 2.3 / 2.5（却下）

- 19B / 22B で容量差が大きすぎる
- checkpoint が transformer + VAE + text projection のバンドル構造で、**VAE 世代が 0.9.x とは別物**
- audio-video joint model の要素が入ってくる

### 2.3 Wan2.2（優先度最下位）

技術的には可能（オープン重みが存在する）。ただし within-family replication であり、
cross-family より主張が弱い。時間が余った場合の最後の選択肢。

---

## 3. 採用する構成

```
Main            : Wan2.1-VACE-1.3B + VACE conditioning + DiffSynth-Studio LoRA
Cross-family    : LTXV 2B 0.9.5 + IC-LoRA reference_video conditioning + LTX-Video-Trainer
```

### 3.1 名称と問いの立て方

backbone と conditioning architecture が**同時に変わる**ため、
「Cross-backbone replication」とは呼べない。

**名称：Exploratory Cross-Family Validation**

| | |
|---|---|
| ❌ | 同一の control architecture で backbone だけを変えたとき順位が再現するか |
| ✅ | **異なる video-generation family および conditioning mechanism に移した場合でも、3D control representation に関する定性的傾向が観測されるか** |

交絡を隠さない。これを Discussion に明記する。

### 3.2 既存資産がそのまま使える（2026-08-13 確認）

LTX-Video-Trainer の resolution bucket 制約は「空間が32の倍数」「frames % 8 == 1」。

```
832 = 32 × 26   ✅
480 = 32 × 15   ✅
49  = 8 × 6 + 1 ✅
```

**mkv 8本の作り直しは不要。Arm C の radius=6 / 240 points も変更しない。**

（当初「VACE-LTX は 97×512×768 なので全作り直し」と判断したが、これは VACE-LTX
inference script の default 値であり、LTX-Video-Trainer の制約とは別。§7 の訂正参照。）

### 3.3 LTXV 2B 0.9.5 を選ぶ理由

| 理由 | 内容 |
|---|---|
| 公開 checkpoint が明確 | `ltx-video-2b-v0.9.5.safetensors`（Lightricks/LTX-Video）。将来同じ checkpoint を再取得できる |
| trainer が正式サポート | `model_loader.py` の `LtxvModelVersion` enum に `LTXV_2B_095 = "LTXV_2B_0.9.5"` があり、`hf_repo` は `Lightricks/LTX-Video-0.9.5` |
| VAE と学習経路の世代が一致 | 「VAE だけ 0.9.x で試して本番は LTX-2」というズレを作らない |
| 0.9.6 より安定 | 0.9.6 は `DEV` 命名で、`hf_repo` プロパティが `ValueError` を投げる（専用 HF repo が無い） |

**Limitation として事前に記録する：**

> 2B 0.9.5 was selected to keep model capacity closer to the 1.3B main backbone and to use a
> stable, reproducible public checkpoint, rather than a substantially larger contemporary LTX model.

提出時点（2027年6–7月想定）で約2年前の checkpoint になる点も Discussion に書く。

---

## 4. 実装確認の結果（2026-08-13、公式実装を読んだ）

### 4.1 ⚠️ 最重要：0.9.5 から 13B 0.9.7 まで VAE が共通

`LTX-Video-Trainer/src/ltxv_trainer/model_loader.py` の `load_vae()` に明記：

> `LTXV_2B_095`'s VAE must be loaded from the Diffusers folder-format instead of safetensors.
> This is a special case also for `LTXV_2B_096_DEV` and `LTXV_13B_097_*` which don't have
> standalone HuggingFace repos, but **share the same VAE as `LTXV_2B_095`**.

実装は次の5つを同一分岐に入れ、いずれも `Lightricks/LTX-Video-0.9.5` の `subfolder="vae"` から読む：

```
LTXV_2B_0.9.5 / LTXV_2B_0.9.6_DEV / LTXV_2B_0.9.6_DISTILLED
LTXV_13B_097_DEV / LTXV_13B_097_DISTILLED
```

**帰結：**

| | 内容 |
|---|---|
| Gate L1 の結果が使える範囲 | **LTX-Video 0.9.5 〜 13B 0.9.7 の全系列**（0.9.5 単体ではない） |
| 「どの VAE 世代でテストするか」の未決 | **解決**。0.9.x/13B 系は VAE が1個 |
| Arm C が死んだ場合の主張 | 「LTXV 2B 0.9.5 で死んだ」ではなく「**LTX-Video 0.9.5 以降の全系列で死んだ**」 |
| 範囲外 | LTX-2 / 2.3 / 2.5（別 VAE） |

VAE のロード（`.safetensors` 単体からは `encoder.conv_out.conv.weight` で ValueError。
必ず folder-format を使う）：

```python
from diffusers import AutoencoderKLLTXVideo
vae = AutoencoderKLLTXVideo.from_pretrained(
    "Lightricks/LTX-Video-0.9.5", subfolder="vae", torch_dtype=torch.bfloat16)
```

### 4.2 2B × IC-LoRA にコード上の阻害要因はない

`training_strategies.py` の strategy 選択：

```python
def get_training_strategy(conditioning_config: ConditioningConfig) -> TrainingStrategy:
    conditioning_mode = conditioning_config.mode
    if conditioning_mode == "none":               -> StandardTrainingStrategy
    elif conditioning_mode == "reference_video":  -> ReferenceVideoTrainingStrategy
```

**引数に model version が存在しない。** `ConditioningConfig` のみを受け取るため、
モデルサイズによる分岐が構造的にありえない。

`ReferenceVideoTrainingStrategy.prepare_batch` も generic：

| 要素 | 実装 |
|---|---|
| latent 形状 | dataset から `num_frames` / `height` / `width` を読む。ハードコードなし |
| 連結 | `torch.cat([ref_latents, noisy_target], dim=1)` |
| 座標 | `prepare_video_coordinates(..., sequence_multiplier=2)` |
| channel 数 | ハードコードなし |
| 13B 専用処理 | **無い** |

13B 専用分岐が存在するのは `load_transformer` のみ（13B は Diffusers config が無いため
`_load_ltxv_13b_transformer` で手書き config を注入）。**2B は `from_single_file` で
素直に読める側であり、13B より経路が単純。**

設定は `model.training_mode: "lora"` ＋ `conditioning.mode: "reference_video"` の2項目。
`model_source` は独立した第3の項目。

### 4.3 残る不確実性

| 項目 | 状態 |
|---|---|
| コード上 2B × reference_video を構成できるか | ✅ できる |
| 公開されている IC-LoRA の実例 | ❌ **全部 13B 0.9.7**（depth / pose / canny、example YAML も 13B）。**2B の hyperparameter 実績はゼロ** |
| sequence 倍長のコスト | ⚠️ 公式が「reference と target の token 連結で sequence が倍になり学習・推論時間が大幅に増える」と明記 |
| 実際に走るか | ❓ smoke で確認 |
| **Arm C が VAE を生き残るか** | ❓ **検索では絶対にわからない。forward pass そのものが実験** |

「動かない理由は見つからなかった」が正確な記述。**「誰も 2B で公開していないだけ」** という
状態まで格下げできた。

---

## 5. Gate 設計

### 5.1 なぜ素朴な pixel round-trip では判定できないか

**LTX の VAE は素朴な VAE ではない。** 論文は、空間32×32・時間8倍で128-channel latent へ
圧縮すること、著者自身が「高圧縮では fine detail の表現能力が制限される」「高周波信号は
compressed latent で poorly represented になり得る」と認めていること、そして
**それを補うため decoder に最後の denoising を担当させ pixel space で detail を生成させる**
設計であることを述べている。

したがって `encode → decode → 円が復元された` を見ても：

```
円が復元された
  ├─ latent に点の情報が残っていた
  └─ decoder がそれっぽい高周波を描いた   ← 区別できない
```

**これは W6 でぶつかったのと同じ形の失敗。** 「T_control 単体は追従の粗いゲートで、暗記でも
通りうる（陽性対照で Scene C tv を入れると median 38 で PASS した）」——指標が間違った理由で
通る。あの時は Scene C 誤差と C′ 誤差の比較で判別した。ここでも判別設計が必要。

`posterior.mode()` ＋ `decoder timestep = 0` にしても **decoder は同じ生成的 decoder のまま**。
生成性を減らすだけで消せるとは言えない。

### 5.2 もっと本質的な話：測るべき面が違う

**Arm C の仕事は DiT に情報を届けること。DiT が見るのは latent だけで、decode された
pixel を一度も見ない。**

```
Arm C pixels → VAE encode → latent → DiT      ← 実験で使う経路
                              ↓
                          decode → pixels      ← 実験で一度も通らない経路
```

Wan では pixel round-trip が有効な代理指標だった（Wan の decoder が素朴なので
「pixel が戻った ≒ latent に残っていた」が成立する）。**diffusion decoder ではその
代理関係が壊れる。**

### 5.3 主 Gate は latent 空間の弁別性にする

**摂動を入れて latent が動くかを見る。**

```
Arm C（正）              → encode → z_base
点の色を2つ入れ替え      → encode → z_swap
点を 1 px ずらす         → encode → z_shift
1 点を消す               → encode → z_drop
correspondence を壊す    → encode → z_break

||z_swap − z_base|| などが測定ノイズを超えるか
```

**latent が動かないなら情報は届いていない。** decode が何を描くかに関係なく確定する。
逆に latent が動くなら DiT が読める可能性がある。

これは陰性対照そのもので、W7 でやった「assert は壊したデータで発火確認する」と同型。
摂動の生成は既存の `score_armc.py` の `replay_frame` 規約（radius 6px=113px、
depth 遠い順に塗る）をそのまま叩けばよい。

**対称性：Wan 側でも同じ摂動を流して `||Δz||` を取る。** 「同一の摂動・異なる VAE」という
対称な比較になる。色誤差だと Wan の閾値（palette 最小色間距離 48.1 由来）を LTX に
持ち込む形になるが、latent 弁別性なら両側で同じ量を測れる。

### 5.4 Gate L1 の3層

| 段 | 何を測る | 位置づけ |
|---|---|---|
| **L1a** | **latent 弁別性**（摂動で `\|\|Δz\|\|` が測定ノイズを超えるか） | **主 Gate。DiT が実際に見る面** |
| L1b | `posterior.mode()` ＋ `t=0` の pixel round-trip、color error / palette margin | 参考。Wan の §13.1 との直接比較用 |
| L1c | 実運用 decode 設定での reconstruction | 診断のみ。Wan 側と非対称なので判定に使わない |

### 5.5 L1 の入力と判定範囲

**入力（一切変更しない）**

```
armA_original.mkv / armB_original.mkv / armC_original.mkv / tv_R0S0_original.mkv
832 × 480 × 49
Arm C: radius = 6 px / 240 points / 12色パレット
```

**Go/No-Go は Arm C で判定する。** cross-family を先へ進める最大のボトルネックが
Arm C（疎な点＋固定色 identity）であるため。

**Arm A / B は診断的記録のみ。PASS/FAIL 判定しない。**
palette 最小色間距離 48.1 は Tracking palette 由来で、Arm A / B の表現には対応しない。

| Arm | 記録する指標 |
|---|---|
| Arm B | XYZ reconstruction error / Entity classification と境界漏出 / latent edit signal |
| Arm A | silhouette preservation / shading reconstruction / boundary error |

**併せて保存する latent**

```
z_armC / z_original / z_edited / z_correspondence-broken
```

から次を測る：latent edit magnitude `||z_edited − z_original||` / changed-token fraction /
Target 帯への latent 応答 / correspondence 破壊前後の latent 差。
**実際の latent shape も実測して記録する**（「32×32×8 だからこうなるはず」で議論しない）。

### 5.6 閾値と予測を実行前に凍結する

- **L1a の主判定閾値（`||Δz||` が超えるべき測定ノイズの水準）は測定前に凍結する。**
  §37.8：閾値を測定後に緩めることは事後合わせであり禁止
- L1b の参考値は Wan §13.1 と同じ `color error ≤ 24` / `palette margin ≥ 1.0`
  （Wan 側参照値：V=0 で色誤差 2.29 / margin 10.5、V=17 で 11.30 / 2.13）
- **予測も測定前に記録する**（作法：予測を先に置いて外れたら訂正明示）

---

## 6. 実行順序と依存関係

### 6.1 論理的依存関係と作業順序を分ける

```
論理的依存関係
  L1   : 前提なし
  L2   : L1 PASS ＋ Main Claim Gate を超える §2 の contrast が存在
  Full : L2 PASS

実際の作業順序
  Phase 0.8 → Pipeline floor → Main Claim Gate → （必要なら）L1 → L2 → Full
```

**L1 に論理的な前提はないが、実行は Main を優先する。** scope creep 防止。

### 6.2 Gate L2 — IC-LoRA smoke

**前提条件（両方満たすこと）**

1. **L1 PASS**
2. **§2 の Primary contrast（Arm B vs Arm C）または Secondary contrast（Arm A vs Arm B/C）
   のいずれかが、Main Claim Gate（§15：Evaluator floor / Pipeline floor / 同 Arm 内の設定差 /
   Training-run 間変動 / 数値再現性の床）を超えた**

条件2の contrast は **§2 で既に事前定義済み**であり、Phase 0.8 の結果を見てから
新規に定義しない。順位そのもの（例：B=0.81 > C=0.80 > A=0.60）を再現対象にすると
Pipeline floor 以下の微小差を追いかけることになるため、contrast 単位で扱う。

**Main Claim Gate が「主張不可」に振れた場合、再現すべき対象が存在しないため L2 は実施しない。**

**内容**

```
LTXV 2B 0.9.5 + reference_video conditioning + LoRA、20–50 steps
```

性能を見るのではなく経路の成立だけを確認する：

- Arm control を reference_video として読めるか
- loss が正常か / checkpoint が保存されるか
- inference 時に LoRA をロードできるか（Wan 側の `150 tensors are fused by LoRA` に相当する確認手段を特定する）
- control の有無で output が変わるか
- **trainer preprocessing の exact-input 検査**（§36.6 の罠と同型。LTX 側の preprocessing は
  「アスペクト比を保ってリサイズ → center crop」なので、scale = 1.0 でも resize 経路を
  通る可能性がある）

**参照画像の扱い**：Wan 側は `vace_reference_image = target frame 0`。LTX 側は
`first_frame_conditioning_p = 1.0` にして毎回 frame 0 を conditioning する設計に寄せられる。
完全に同一ではないが「Wan だけ reference image を持つ」という大きな交絡は減らせる。

### 6.3 Full validation

**L2 PASS 後に初めて実施可否を判断する。** 現時点では予約しない。

3 arm × 500 steps から始める必要はなく、まず Arm C のみ step 0 / 100 / 200 程度で
tracking control として成立するかを見る縮小版から入る選択肢がある。

### 6.4 もし Arm C が L1 で死んだ場合の2つの道

**A. Strict replication（推奨）**

`Wan Arm C = radius 6` / `LTX Arm C = radius 6` を絶対に変えない。LTX で死んだら：

> The fixed tracking representation did not survive the LTX latent interface sufficiently
> for a matched cross-family replication.

として終了。これは失敗ではなく「representation の可用性自体が VAE architecture 依存」
という結果。

**B. Backbone-calibrated replication**

どうしても生成実験までやる場合、radius を 6 / 8 / 10 / 12 … と **VAE だけを見て**探索し、
各 backbone で同じ survivability threshold を満たす最小 radius を選ぶ。
**Main output を見る前に決めること。**

変えるのは radius のみ。240点 / point centers / RGB identity / entity codebook /
trajectories / visibility / temporal correspondence は全部同じ。

この場合は strict cross-family replication と呼ばず
**backbone-calibrated cross-family validation** とする。Wan/LTX 間の絶対性能差は比較しない。

---

## 7. 結果がどちらでも thesis は改善する

| L1 の結果 | 得られるもの |
|---|---|
| Arm C が残る | sparse tracking representation が Wan VAE だけに依存した特殊ケースではない、という補助証拠。**ただし入力表現としての可搬性を示すだけで、生成時に利用されることは示さない** |
| Arm C が死ぬ | **実測に裏付けられた limitation。** 「Wan2.1 では survive した同一の tracking control が、**LTX-Video 0.9.5 以降の全系列が共有する VAE** では survivability gate を満たさなかった」。VAE 共通の発見により、単一モデルではなく系列全体の話として書ける |

前者でも後者でも Discussion が1段強くなる。**だから L1 はやる価値がある。**
一方 L2 以降は Claim Gate の結果次第で無意味になりうるので予約しない。

---

## 8. この構想に至る過程で訂正した予測

**方針：作業前に予測を置き、外れたものを全て記録する。**

| # | 予測 | 判定 | 外した理由 |
|---|---|---|---|
| X-01 | radius 6 → LTX では 32/8 倍の 24 px が必要 | **根拠なし** | VAE は 32×32 の average pooling ではない。128-channel latent で情報を保持する設計なので面積比から survival threshold は導けない。**自分の作法（VAE survivability は実測してから決める）を LTX には適用していなかった** |
| X-02 | 240点 × radius 24 は画面面積を超えるので物理的に不可能 | **誤り** | 円は重なる。延べ描画面積が画面を超えても描画自体は可能。問題になるのは union occupancy / latent cell 内 collision / color confusion であり面積の総和ではない |
| X-03 | 832×480×49 は LTX で使えず mkv 全作り直し | **誤り** | LTX-Video-Trainer の bucket 制約（32の倍数 / 8n+1）を既存値が満たす。97×512×768 は VACE-LTX inference script の default 値に過ぎない |
| X-04 | LTX で Control LoRA を学習する経路自体がない | **誤り** | LTX-Video-Trainer に IC-LoRA（reference_video conditioning）が実装済み。ただし VACE-LTX 経由ではない |
| X-05 | pixel round-trip の色誤差で survivability を判定できる | **不十分** | LTX の decoder は pixel space で高周波を生成する設計。「latent に残っていた」と「decoder が描いた」を分離できない。主 Gate は latent 弁別性にすべき |

X-01 〜 X-04 はいずれも**実装・仕様を確認する前に数式で結論を出した**ことが原因。
§36.15 で「実装を読んでから予測を立てたため的中率が改善した」と記録したのと逆をやった。

X-05 は別種で、**Wan で有効だった代理指標を、decoder の性質が違う LTX にそのまま
持ち込もうとした**のが原因。W6 の「T_control 単体は暗記でも通る」と同じ構造。

---

## 9. 調査は打ち切り

公式論文・公式実装から得られる情報はここまで。残っているのは

> **半径 6 px・240点・12色パレットが LTX 0.9.5 の VAE を通るか**

の一点で、これは **trained weights への forward pass そのものが実験**であり、
検索では絶対に出てこない。transfer characteristic は公開されていない。

**これ以上調べても情報は増えない。** #7 は `NOT STARTED` のまま放置しても情報が古くならない。

---

## 10. 一言サマリ

```
Phase X: Exploratory Cross-Family Validation
Candidate: LTXV 2B 0.9.5（VAE は 0.9.5〜13B 0.9.7 で共通）

L1a  latent 弁別性（主 Gate、摂動 → ||Δz||）      前提なし / Main 優先
L1b  t=0 pixel round-trip（参考、Wan §13.1 比較用）
L1c  実運用 decode（診断のみ）
        ↓ Arm C PASS
L2   2B IC-LoRA smoke 20–50 steps
        ↑ 前提：L1 PASS ＋ Claim Gate を超える §2 contrast が存在
        ↓ PASS
Full  実施可否を再判断（予約しない）

既存資産は無変更で使える。mkv 作り直し不要。
Arm C が死んだら strict replication のまま limitation として報告する。
```
