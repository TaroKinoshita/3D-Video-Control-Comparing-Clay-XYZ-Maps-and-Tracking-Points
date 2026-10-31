# protocol rev.13 → rev.14 更新差分（W9 反映、2026-10-31）

Phase 0.8（Arm A / Arm B の学習）に入る前に、**採点ルールの穴を 2 つ埋めた。**
どちらも Arm A / Arm B の結果を見る前に確定しており、事後合わせではない。
設計の骨格（3-arm VACE LoRA、B vs C primary、47 px / 2 フレームの閾値）は不変。

本差分は `protocol.md` 本体に反映済み（§12 / §18 / §37 の該当箇所の書き換え + §41 新設）。

---

## 変更1：T_control の GT 中心を Scene C′ の ID パスから取る（§37.4 / §12）

**旧（rev.13 まで、`score_t_control.py` v1）**

```
u_GT = Scene C の ids==3 重心
v_GT = Scene C の ids==3 重心 − (K / 9.00)·dZ(f)
```

**問題。** Scene C′ では Target が持ち上がって遮蔽板の上端（v = 232.78）を越えるため、
**見えているシルエットそのものが Scene C と違う**（rev.12 変更6 と同じ機序。index 6–29 は u 全幅が可視）。
Scene C で半分隠れている形の重心を上にずらしても、C′ で見えている形の重心にはならない。

| 実測（W9、numpy、`make_gt_center.py`） | 値 |
|---|---|
| v1 GT と C′ 本物の GT の差 | median 2.38 / p95 31.42 / **最大 31.88 px**（index 22–27） |
| 完璧な生成（`tv_R0S0_prime.mkv`）を v1 GT で採点 | median 4.09 / **p95 32.98 px** |
| 同じものを C′ 本物の GT で採点 | median 1.81 / **p95 3.75 px** |
| 例：index 23 の Target 画素数 | Scene C 11,897 → C′ 17,261（u 右端 401.5 → 482.5） |

v1 の GT では、採点側だけで p95 が約 33 px 食われていた。47 px の関門に対して大きく、
arm 間差を潰し、§15 の Evaluator floor も過小に見積もる（Scene C 上で測った 3.89 px は C′ の採点面の床ではなかった）。

**新**

```
GT = Scene C′（geo/prime/id_%04d.exr）の ids==3 のマスク重心
     座標系は tracker と同じ画素 index（+0.5 しない）
```

- 生成：`scripts/pipeline/make_gt_center.py` → **`data/points/gt_target_center_prime.json`**（5,848 bytes、sha256 先頭 16 = `2656edf279cfcb3e`）
- 採点側はこの JSON だけを読む（経路分離。作法 #6）
- 内部検査：可視 44 フレーム / 完全遮蔽 index 36–40 / index ≥ 35 は C′ と Scene C の重心が一致 / index 0 の ID パスは bit 一致 / Marker 重心で EXR の向きを確認
- 暗記判別の参照（Scene C 軌道）も同じ JSON の `u_sc` / `v_sc`（Scene C の ids==3 重心）から取る。v1 と同じ値

---

## 変更2：T_coherence は「本数」で判定する（§37.6 / §12）

**旧の書き方**：「個数 ≠ 1 のフレーム率 ≤ 2 / 44。Tracker failure は分母から除外し別カウント」。

**問題。** 分母から tracker failure を外すと分母が 44 未満になり、率で判定すると
**tracker failure が出るほど基準が厳しくなる。** 偵察 run は個数 ≠ 1 が 2 本（f33 / f41）で
予算どおりだったのに、分母 42 になったため 2/42 = 0.0476 > 2/44 = 0.0455 で FAIL と判定されていた。

2/44 の根拠（§37.6）は「幾何的に判定困難な frame 35 / 41 の **2 本分だけ** 予算を認める」であり、
本数の予算である。

**新**

| 項目 | 規則 |
|---|---|
| 判定 | **個数 ≠ 1 のフレームが ≤ 2 本なら PASS** |
| 対象 | GT 可視 44 フレームのうち tracker が Target を検出できたフレーム |
| tracker failure（検出 0） | **分母にも分子にも入れない。** 別カウントとして本数とフレーム番号を必ず報告する |

決定者：Taro（2026-10-31、Arm A / B の結果を見る前）。
tracker failure の本数に上限は置かない（置く根拠となる測定が無い）。arm 間で本数を並べて報告する。

---

## 変更3：偵察 run（g1_armC）を新ルールで採点し直した（§33.7 / §38）

`scripts/pod/score_t_control_v2.py`（GPU 不要）。tracker は v1 と同一。

| 指標 | v1（rev.11 の記録） | **v2** |
|---|---|---|
| T_control median | 8.55 px | **6.62 px**（PASS） |
| T_control p95 | 42.00 px | **20.36 px** |
| T_control max | 44.29 px | 25.74 px |
| 暗記判別 | C′ 8.55 ≪ Scene C 57.47 | **C′ 6.62 ≪ Scene C 57.47**（control 追従） |
| T_coherence | 2/42 → FAIL | **2 本（f33 / f41）→ PASS** |
| tracker failure | f34 / f35 | f34 / f35（別カウント） |

レポート：`data/reports/w9/score_g1_armC_v2.json`。v1 の 8.55 / 42.00 / f34・f35 は numpy の再実装で再現を確認済み。

**結論は変わらない**（control を読めている）。p95 が半分以下になったのは、v1 の p95 の大部分が GT のズレだったため。

---

## 変更4：Evaluator floor を C′ の採点面で測り直した（§37.8）

| 面 | median | p95 | max | 備考 |
|---|---|---|---|---|
| Scene C の tv、Scene C の GT（rev.11） | 3.05 | 3.89 | 91.25 | 旧記録 |
| **C′ の tv、C′ の GT（W9）** | **1.81** | **3.75** | 91.25 | **これを本実験の床とする** |

p95 3.75 ≪ 47 → **関門は有効、保険条項は発動しない**（結論は rev.11 と同じ）。max 91.25 は index 35（Target が板の縁でほぼ消える）。

---

## 変更5：陽性対照の結果が変わった（記録）

`tv_R0S0_original.mkv`（暗記の模擬）を v2 で採点した。

| | v1 | v2 |
|---|---|---|
| T_control median | 38.35 px（PASS） | **47.15 px（FAIL、閾値ぎりぎり）** |
| 暗記判別 | 暗記寄り | 暗記寄り（vs C′ 47.15 / vs Scene C 3.05） |

**「T_control 単体は暗記でも通りうる」（rev.11 変更5 注記）の判断は維持する。** 47.15 は閾値の 0.15 px 外側にあるだけで、
暗記を T_control で弾けるとは言えない。暗記判別は引き続き Scene C 距離との比較で行う。
`score_t_control_v2.py` は暗記判別を main から常に出力する（v1 は `memorization_probe` が main から呼ばれていなかった）。

---

## 変更6：未決 #27（参照画像 PNG の content sha 方式）を解決（rev.13 変更4）

総当たりの結果、記録値 `df7c595457c81e35b47164ae` は **方式A を shape `(1, 480, 832, 3)` で計算した値**だった。

| 方式 | shape | sha256 先頭 |
|---|---|---|
| **A（`str(shape)` + tobytes）** | **(1, 480, 832, 3)** | **`df7c595457c81e35b47164ae`** ← 記録値と一致 |
| A | (480, 832, 3) | `4c99994a2ddd544a754c4efc` |
| B（tobytes） | どちらでも | `afc440eec3c328600d869ef5` |
| file sha256 | — | `64ccc9dbd502acf5bba861c5` |

併せて `ref_R0S0_original.png` が `tv_R0S0/original/tv_0000.png` および `tv_R0S0_original.mkv` の frame 0 と bit 一致することを確認した。
以後の照合は方式B `afc440eec3c32860` を使える（manifest の md5 照合も継続）。

---

## 変更7：Arm C prime の点データは W4 版が正（README 注記の解決）

`data/points/armc_prime_points_W3.json` と `_W4.json` は v 座標が同一で、**visible が 267 箇所違う**
（W3 = visible 継承のみ、W4 = index 0–35 を Occluder 矩形で再計算）。
自前で円盤描画を再現して `armC_prime.mkv` と比較した結果：

| 点データ | `armC_prime.mkv` との不一致フレーム |
|---|---|
| W3 版 | index 8–30（23 フレーム） |
| **W4 版** | **なし（49/49 bit 一致）** |

**W3 版は使わない。** 同じ再現で `armc_points.json` → `armC_original.mkv` も 49/49 bit 一致した。

---

## 変更8：Phase 0.8 用スクリプト（凍結スクリプト一覧に追加）

| スクリプト | 役割 | 状態 |
|---|---|---|
| `scripts/pipeline/make_gt_center.py` | C′ の GT 中心 JSON を作る | ローカルで実行済み |
| `scripts/pod/score_t_control_v2.py` | T_control / T_coherence / 暗記判別 | selftest PASS（陽性対照 2 本） |
| `scripts/pod/infer_arm.py` | arm 切替式の推論。tokenizer 直指定・LoRA fuse 検査・FFV1 保存 | **GPU 未実行** |
| `scripts/pod/train_arm.sh` | arm 切替式の学習。フラグは g1_armC の `training_args.json` と同一 | **GPU 未実行** |

`train_arm.sh` は `--enable_tensorboard_log` を**付けない**（g1_armC と揃えるため。§36.11 の「付ける」は撤回）。
loss はログから読む。

---

## 変更9：本文の記述ミスの修正

- §13.3 の速度表で、半径 6 の行に §18 の行が混入していた（rev.10 の編集事故）。元の行に戻した
- `scene_c_prime_traj.json` の meta `Z0_note` は誤り（rev.12 変更1）。ファイルは凍結入力なので書き換えず、本文の注意書きで扱う

---

## 未決事項の番号の整理

PRE_EXPERIMENT_SUMMARY と rev.11〜13 は「新 25」「新 27」という番号を使っていたが、§18 にはすでに
解決済みの #25〜#28 があり衝突していた。§18 では次の番号に振り直す。

| サマリ上の番号 | §18 の番号 | 内容 | 状態 |
|---|---|---|---|
| 新 25 | **29** | T_coherence 2/44 が生成破綻か tracker 限界か | 未決（判定規則は変更2 で確定。原因の切り分けが残る） |
| 新 27 | **30** | 参照画像 PNG の content sha 方式 | **解決**（変更6） |
