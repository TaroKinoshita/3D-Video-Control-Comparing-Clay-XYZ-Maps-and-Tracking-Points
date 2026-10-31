# protocol rev.10 → rev.11 更新差分（W6 反映、2026-08-10）

偵察 run G1 完了を受けた更新。**確定事項の追加・訂正のみ。設計の骨格（3-arm VACE LoRA、B vs C primary）は不変。**

適用したら本ファイル1行目付近の版表記を **rev.11** に。

---

## 変更1：§33.4 に制約4を明示追加

**現状**：Scene C′ の制約は3本（速度・面積・始点）。

**追加**：制約4を明記。

> **制約4（遮蔽窓の固定）**：Scene C′ の完全遮蔽区間は Scene C と同じ **index 36–40** に揃える。
> 根拠：§12 の T_control 評価面「GT で Target 可視な 44 フレーム（0–35 / 41–48）」と T_coherence の予算「2/44（frame 35/41 が幾何的に判定困難）」は、いずれも Scene C の遮蔽構造にハードコードされている。C′ の遮蔽タイミングがずれると評価面と予算が破綻する。
> 実装上は、C′ 軌道が遮蔽窓 36–40 で dZ=0（Scene C と同一位置）になっていれば自動的に満たされる。

（W6 で採用した「案 A」。案 B＝評価面を C′ 依存に測り直す、は不採用。）

---

## 変更2：§9(Seed設計) に具体値を追記

**現状**：generation seed G1–G4 / training seed S1–S3 はラベルのみ、数値なし。

**追記**：

> **具体値（固定集合）**：
> - generation seed：G1=0, G2=1, G3=2, G4=3（`pipe(seed=...)` に渡す）
> - training seed：S1=0, S2=1, S3=2
>
> **注意**：DiffSynth 2.0.18 の `train.py` に `--seed` は存在しない（実物確認済み）。学習側 seed は非固定であり、S1=0 は「1本目」のラベルとして扱う。再現性は generation seed のみで担保される。

---

## 変更3：§21 未決事項 #2 を解決

**現状**：#2「gradient_accumulation_steps のみ」＝偵察 run 後に決める、未決。

**解決**：

> **#2 解決**：`gradient_accumulation_steps = 1`（g1_armC の training_args.json 実測）。DiffSynth 既定のまま。

未決テーブルから #2 を削除。#23（N）は次項参照。

---

## 変更4：推論設定の新節（§35 系に追加）

> ### 推論設定（偵察 run / Phase 0.8 共通、DiffSynth 2.0.18 実装既定 + 3変更）
>
> `WanVideoPipeline.__call__` の実装既定を採用し、3点のみ変更する。
>
> | 引数 | 値 | 出所 |
> |---|---|---|
> | cfg_scale | 5.0 | 実装既定 |
> | num_inference_steps | 50 | 実装既定 |
> | sigma_shift | 5.0 | 実装既定 |
> | negative_prompt | ""（空） | 実装既定、3 arm 統一 |
> | denoising_strength / vace_scale | 1.0 / 1.0 | 実装既定 |
> | **num_frames** | **49** | 既定81から変更 |
> | **tiled** | **False** | §2.1 の bit一致経路（VAE tiled=False）と統一 |
> | **seed** | **G1=0** | §9 |
>
> **LoRA load**：`pipe.load_lora(pipe.vace, "<output>/step-500.safetensors", alpha=1)`。
> - `pipe.dit` ではなく **`pipe.vace`**（学習の `--remove_prefix_in_ckpt pipe.vace.` と対応）。
> - `alpha=1`（学習が alpha=rank で保存 → load 側 1 が正しいスケール）。
> - **`150 tensors are fused by LoRA` が出るか必ず確認**。出なければ LoRA が乗っておらずベースモデルで推論している（静かな失敗）。
> - control は 49 フレームのリスト、ref は PNG を `Image.open(...).convert("RGB").resize((832,480))`。
>
> **モデルは実パス直指定**（`ModelConfig(path=...)`）＋ `DIFFSYNTH_SKIP_DOWNLOAD=True`。model_id 形式は redirect 先を探して空振りする。tokenizer も `tokenizer_config=ModelConfig(path=".../google/umt5-xxl")` で直指定。

---

## 変更5：§33.7 偵察 run の予測を訂正（結果で上書き）

**現状（rev.10 の予測、2026-08-06）**：
- C′ 上で 3 arm すべてが T_control を下回る
- 出力の中身：C′ control を入れても Scene C の軌道が出る（暗記）
- 帰結：3 arm 差 ≤ Pipeline floor → 主張不可 → N を増やす

**W6 実測（Arm C 代表、S1、500 steps、G1）で訂正**：

> **偵察 run 結果（2026-08-10）**：予測は外れた。**暗記に落ちず、control 追従を獲得した。**
> - T_control median = **8.55 px**（≤47 PASS、Evaluator floor 3.89 の約2倍）
> - 暗記判別：C′ center 誤差 **8.55** ≪ Scene C center 誤差 **57.47** → **control 読めている**
> - T_coherence = 2/44（閾値ぎりぎり FAIL。tracker_fail は境界フレーム f34/f35＝§12 の 2/44 予算に対応）
>
> **帰結（段4）**：暗記に落ちていないため軌道変種の追加は不要。**そのまま 3 arm × S1 へ進む。未決 #23（N）は暫定 N=1（増やさない）。** 3 arm で control 追従性能の差を測り、Claim Gate（3 arm 差 vs Pipeline floor）で主張可否を判定する。
>
> **注記**：T_control 単体は「追従の粗いゲート」で、暗記でも通りうる（陽性対照で Scene C tv を入れると median 38 で PASS した）。暗記/制御の判別は **Scene C center 誤差と C′ center 誤差の比較**で行う（判定スクリプト `score_t_control.py` に実装）。

---

## 変更6：§18-9 の検出/追跡器を確定

**現状**：tracker 未選定（Day 25-31 前）。

**確定**：

> **tracker 確定（色セグメンテーション）**：
> - マスク条件：R>G, G>=B, (R−B)>40, R>80（Target orange 分離）
> - Marker(黄) 誤検出対策：重心が (658.76, 123.47) の ±30px の連結成分を除外
> - 検出：残りの最大連結成分の重心
> - 遮蔽窓（GT の Target 画素0）は tracker failure として分母から除外
>
> **Evaluator floor（§37.8）**：tv_R0S0_original に上記 tracker を当て、ids==3 重心と比較。median 3.05 / **p95 3.89 px**。**p95 ≪ 47 → 関門有効、保険条項は発動しない。**

---

## 変更7：Arm C 生成が Blender 不要である旨を明記（§13 系 or §33）

> **Arm C の control は Blender レンダではなく Python 円盤描画**（`score_armc.py` の `replay_frame` 規約：radius 6px=113px、depth 遠い順に塗る）。点の u,v,depth,visible,rgb は `armc_points.json`（`select_arm_c.py` の bpy 出力）にある。
>
> **Scene C′ の Arm C 化**：Target帯72点を各フレーム `v_new = v_old − (1155.5556/point_depth)·dZ(f)` で v シフト（点ごとの depth）。u/depth/rgb は不変。**visible は f0–35 を Occluder 矩形（u 401.56–567.67 / v 232.78–456.67 / depth 8.00）で再計算、f36–48 は継承**（Scene C で f0–40 は矩形判定=レイキャスト完全一致、f41–48 は Occluder 厚み0.05 の縁で7点ズレのため継承側が正）。
>
> → **偵察 run は Arm C 代表なら Blender ゼロで回せる。Arm A（clay）/ Arm B（position）の C′ control は Blender レンダが必要。**

---

## 変更8：§2.5 の訂正

> - `--use_gradient_checkpointing` を**明示**すれば training_args.json に `true` と正しく記録される（rev.10 の「false と記録される」は非明示時の挙動）。
> - checkpoint の保存は `step-{n}.safetensors` のみ（save_steps 到達時）。**epoch 毎保存は発生しない**（rev.10 で懸念した「epoch-{id} 大量生成」は起きなかった。実測 0 個）。

---

## 変更9：§2.2 / §36.6 の content sha の解釈を注記

> **content sha256 の計算方式に注意**：ローカル W3\mkv・Pod /workspace/w4/mkv・armc_points.json からの再現、の三者で同一の content sha が出る（例：tv=`6190557123fdc2b3`、armC=`7c08f981503edb34`）。frame0 は既存 armC_0000.png と bit 一致。
> **§2.2/§36.6 の記録値（tv=`bba8…`、armC=`46ab…`、ref=`df7c…`）とは異なるが、これは content sha の計算方式の違い**（現行の imageio デコード→RGB(T,H,W,3) tobytes vs 記録時の別方式。bgr0/gbrp は可逆なので格納形式ではなく計算経路の差）**であり、画素は正規**。ファイルの正規性は「三者一致 + frame0 bit一致」で担保する。
> 今後の照合は「imageio デコード RGB の tobytes sha」に統一するのが安全。

---

## 未決事項テーブルの更新後の姿

| # | 内容 | 待ち条件 |
|---|---|---|
| ~~2~~ | ~~gradient_accumulation_steps~~ | **解決＝1** |
| 20 | 本番シーンでの色の一意性の実地検証 | 本番シーン構築後 |
| 21 | 評価 GT 側で Arm A の Target を同定する方法 | arm A 着手時 |
| 23 | train 軌道本数 N | **暫定 N=1**。3 arm の結果で確定 |
| 24 | Occluder 除去方法（c-x / c-y / c-z） | 本実験時 |
| 新 | T_coherence 2/44 が生成破綻か tracker 限界か | f34/f35 切り分け |
