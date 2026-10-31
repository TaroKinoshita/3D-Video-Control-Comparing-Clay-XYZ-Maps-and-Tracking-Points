# 引継ぎ：卒業論文 3-arm比較実験（Radboud BSc AI）— **W2 完了時点**

## 最初に読んでほしいこと

前のチャットからの引継ぎ。以下を添付してある（無い場合は聞いて）。

- `protocol.md` … **rev.6。** 確定事項。凍結して守るドキュメント（**必須**）
- `worklog_W2_complete_2026-08-05.md` … このファイル
- `worklog_W2a_2026-08-04.md` … W2a の経緯

**W2 は完了。3表現（Arm A / B / C）と Target video（3 split × 2変種）が全部出た。**
次は **Phase 0.0 本番（RunPod / VAE roundtrip）**。

---

## 1. W2 で何が終わったか

| 段階 | 内容 | 成果 |
|---|---|---|
| W2a（8/4） | 配管検証。極小シーンで正解が手計算できる状態を作る | PASS 29 / FAIL 0 |
| W2b Step 1 | Scene C の形状 / カメラ / 光源 / 材質 | PASS 27 / FAIL 0（2変種） |
| W2b Step 2 | アニメーション（Original / Edited） | PASS 27 / FAIL 0 |
| W2b Step 3 | 幾何パス（Position / Depth / ID マスク） | PASS 150 / FAIL 0（2変種） |
| W2c-1 | **Arm B**（Dense World-XYZ） | PASS 52 / FAIL 0（2変種） |
| W2c-2 | **Arm A**（Clay Previs） | PASS 55 / FAIL 0（2変種） |
| W2c-3 | **Arm C**（3D Tracking。240点） | select 19 / render 10、FAIL 0 |
| W2c-4 | Target video（R0S0 / R1S0 / R0S1） | PASS 54 / FAIL 0（各2変種） |

**成果物 882 ファイル。すべて同一スクリプトから bit 一致で再生成できることを機械検査済み。**

---

## 2. 生成物の場所と構成

```
C:\Bachelor Thesis\SummerWork\W2\
  scene_c_original.blend / scene_c_edited.blend
  geo\<variant>\position_%04d.exr  depth_%04d.exr  id_%04d.exr   （frame 1..49）
  armA\<variant>\armA_%04d.png                                    （index 0..48）
  armB\<variant>\armB_%04d.png
  armC\<variant>\armC_%04d.png  +  armC\armc_points.json
  tv\<split>\<variant>\tv_%04d.png                                （R0S0 / R1S0 / R0S1）
```

**入力は frame 番号（1..49）、Control Video と Target video は index 番号（0..48）。**
既知バグ#3（off-by-one）の系統なので、対応を毎回 assert している。

---

## 3. スクリプト一覧

| ファイル | 役割 | Blender |
|---|---|---|
| `scene_c_spec.py` | 寸法・軌道・Style set の定義（正本） | 不要 |
| `scene_c_expect.py` | 照合値の凍結リテラル | 不要 |
| `geo_expect.py` / `armb_expect.py` | 幾何パス / Arm B の照合値 | 不要 |
| `verify_expect.py` | リテラルの検算（**PASS 76**） | 不要 |
| `png_io.py` | 8bit RGB PNG の自前読み書き（PIL と突合済み） | 不要 |
| `armc_palette.py` / `armc_select.py` | パレット / 選定・彩色 | 不要 |
| `armb_encode.py` | Arm B の codebook 適用 | 不要 |
| `test_armb_encode.py` | Arm B 単体テスト（**PASS 35**） | 不要 |
| `test_armc_select.py` | Arm C 単体テスト（**PASS 32**） | 不要 |
| `build_scene_c.py` | Scene C 構築 + SELF-CHECK | 必要 |
| `export_geometry.py` | 幾何パス書き出し | 必要 |
| `encode_arm_b.py` | Arm B 生成 | 必要（EXR 読み） |
| `select_arm_c.py` | Arm C の点選定・彩色 | 必要 |
| `render_arm_c.py` | Arm C の描画 | **不要** |
| `render_arm_a.py` | Arm A クレイ | 必要 |
| `render_target_video.py` | Target video / `--compare` | 必要（compare は不要） |
| `probe_env.py` / `probe_compositor.py` / `probe_masks.py` / `probe_wire.py` | API プローブ | 必要 |

### 実行の型

```
$BL = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
```

**`--factory-startup` を必ず付ける。** アドオンが `save_as_mainfile` で ACCESS_VIOLATION を起こす。

---

## 4. 設計上の重要な決定（W2 で確定したもの）

| # | 決定 | 理由 |
|---|---|---|
| 1 | **正面性フィルタは入れない**（§18-19） | 床の 79% が n·v < 0.2。フィルタを入れると Arm C から床が消え、Position パスで床を持つ Arm B と非対称になる |
| 2 | **点集合と色は2変種で共有**（98フレームの和で選定・彩色） | 変種ごとに選び直すと編集と無関係な差が Control に入る |
| 3 | **点数を入力・分離を出力** | 面積固定なので d を入力にすると点数が決まり Target 割合 0.30 を満たせない |
| 4 | **彩色は使用回数最小の色を選ぶ** | 背景は衝突ペアが 0〜1 本しかなく、素朴な貪欲だと全点が同色になる |
| 5 | **Entity ごとの quota**（DeepObject 8 / Occluder 8 / Marker 2） | 面積比だけだと Wall が支配し Marker 0 点 / DeepObject 6 点になった |
| 6 | **マスクは Emission ID レンダー** | **EEVEE に Object Index パスが存在しない** |
| 7 | **Arm A に Entity ラベルを入れない** | §2 の「Targetラベルなし」の実装。機械検査つき |
| 8 | **Style set は材質のみ。metallic は 0 固定** | `use_raytracing=False` では metallic 面がほぼ黒に落ちる |
| 9 | **`use_raytracing = False`**（Arm A / R0） | True だと再現性が 15 画素崩れる。bit 一致のほうが黒潰れ 1.25% より重要 |
| 10 | **自前 sRGB 符号化** | EXR は線形で出る。線形のまま 8bit にすると動画モデルにとって domain shift |
| 11 | **完全遮蔽5フレーム / 再出現8フレーム** | 速度上限の下で「完全遮蔽」と「完全再出現」は両立しない。予算56pxを分割 |
| 12 | **W2c-5（点数の VAE 判定）を Phase 0.0 に移す** | W2 のスコープは「3種類を出せること」まで。GPU 不要で W2 が完結する |

---

## 5. 訂正した誤り（記録として残す）

| 誤り | 訂正 |
|---|---|
| 「AgX が Position を [0,0.475] に圧縮した。Raw が必須」 | **配線ミスで出た Combined の線形値だった。**真因はソケット型。EXR に view_transform は効かない |
| 「d ≥ 12 なら円は重ならず深度ソートは到達不能」 | **正しくは 13.4143 px。**整数中心 12px は 1 画素共有する |
| 「タブで自己遮蔽させて深度ソートを発動させる」 | 上記が前提だったので**タブは廃止**。ベベル2本に置き換え |
| 「`floor` の誤差は float64 の 1e-9 で直る」 | **mathutils は float32。1e-4 が必要**。しかも落ちる軸が違った |
| 「X-Z 平面の角をベベルする」 | **法線の Y 成分が 0 になり不可視。**X-Y 平面に取る |
| 「Cycles が存在しない」 | **enum が嘘。**直接代入すれば使える |
| 軌道検査の許容値 1e-9 | リテラルが `%+.7f` 丸めなので **1e-5** |

### 予測の的中率

| 段階 | 的中 |
|---|---|
| W2a（Blender API） | 1 / 7 |
| W2b Step 1 の失敗原因 | **0 / 3** |
| W2b Step 2 | 2 / 3 |
| W2b Step 3a（コンポジタ） | 5 / 6 |
| W2b Step 3a-3（Raw / Cycles） | 3 / 5 |
| W2b Step 3a-4（配線） | 3 / 5 |
| W2b Step 3b | 5 / 5 |
| W2c-1（Arm B） | 1 / 1 |
| W2c-3a（Arm C 選定） | 2 / 4 → 3 / 4 → 4 / 5 |
| W2c-2（Arm A） | 3 / 3 → 4 / 6 |
| W2c-4（Target video） | 2 / 3 |

**値・幾何の予測はよく当たる。API と機序の予測は当たらない。**
API は必ずプローブで実測する。

---

## 6. 未解決（優先順）

| # | 項目 | いつ | 状態 |
|---|---|---|---|
| **§18-11** | **点数の確定（VAE roundtrip で色誤差 ≤ 24）** | **Phase 0.0（RunPod）** | 暫定 240 点で生成済み |
| §18-7 | M_invariant の許容値 | W2c で差分ヒストグラム（未実施） | 未決 |
| §18-13 | `vace_reference_image` を3 armでどう扱うか | Day 18前 | 未決 |
| §18-12 | 視差あり条件での V_max | Phase 0.8前 | 未決 |
| §18-14 | VGGT の Depth 規約を実地検証 | Day 25-31 | 未決 |
| §18-2 | `lora_alpha` / batch size / grad accum | 学習環境構築時 | 未決 |
| §18-6 | T_control / T_coherence | Main Test前 | 未決 |
| §18-9 | 検出・追跡器の選定 | Day 25-31前 | 未決 |
| §18-18 | 半径9を採らなかった理由の明文化 | worklog に追記 | **未記録** |
| — | **R 軸が弱い問題** | 9月まで判断保留 | **現状を受け入れて報告する方針** |

### R 軸の問題（新規）

R0S0 基準の 8bit MAE で **R1S0（Cycles）が 6.932、R0S1（set B）が 22.671**。
R1S0 は MAE ≈ 平均シフト（6.926）で、**Renderer shift がほぼ一様な露出差に縮退している。**

原因は Scene C の構成（拡散面のみ / `use_raytracing=False` / 単一点光源 / 閉じた箱）で、
Cycles の強みである鏡面反射・コースティクスが出る余地がない。

**幾何を変えると全 arm の再生成が必要。R1 の評価は9月なので今は動かさない。**
「R 軸は弱いシフト、S 軸は強いシフト」という非対称は結果の解釈に使える。

---

## 7. 次にやること：Phase 0.0 本番（RunPod）

| | 内容 | GPU |
|---|---|---|
| 1 | Pod 起動（§19 の復帰手順。EU-CZ-1 / `test_pod_volume`） | 必要 |
| 2 | Arm C の VAE roundtrip → 色誤差 ≤ 24 の判定 | 必要 |
| 3 | 通らなければ点数を減らして Arm C を再生成（数分） | 不要 |
| 4 | M_invariant の差分ヒストグラム → §18-7 の確定 | 不要 |

**Target 上の局所占有率が 29.40% で警戒線 30% の直下。**
増やす余地はなく、減らす方向のみ。240 → 200 で 24.5% になる。

Terminate で完全停止（Stop はディスク代が月$4残る）。放置1晩 = $17。

---

## 8. 会話の進め方の希望

- 前提から説明して、**1ステップずつ**進めたい。まとめて全部出さないでほしい
- 結論から先に、カジュアルに。お世辞や誇張は不要
- 図や表で直感的に分かるようにしてほしい
- 予測が外れたときは訂正を明示してほしい（worklog に記録するため）
- **コマンドは貼り付け事故を前提に短く区切って出してほしい。プレースホルダを混ぜない**
- 照合用の値は**コードブロックではなく表**で出してほしい
- **スクリプトを修正したら必ずダウンロードリンクを出してほしい**

---

**最後に1つ。** W2 で「エラーも警告も出さずに間違った出力を出す」経路が3つ見つかった
（コンポジタのソケット型 / `floor` の float32 誤差 / `matrix_world` の遅延評価）。
共通するのは**出力がもっともらしく見えること**。
以降も、閉形式で出した照合値との突合を必ず入れる。
