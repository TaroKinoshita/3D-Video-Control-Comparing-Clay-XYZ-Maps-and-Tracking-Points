# protocol rev.11 → rev.12 更新差分（W7 反映、2026-08-12）

Scene C′ の 3 arm control 生成完了を受けた更新。**確定事項の追加・訂正のみ。設計の骨格（3-arm VACE LoRA、B vs C primary）は不変。**

適用したら本ファイル1行目付近の版表記を **rev.12** に。

**前提**：rev.11 の差分（`protocol_rev11_changes.md`）が本体に未適用のままであれば、先に rev.11 を適用してから本ファイルを適用する。

---

## 変更1：variant "prime" を正式定義（§33.4 系に追加）

> ### Scene C′ = variant `"prime"`
>
> `scene_c_spec.VARIANTS = ("original", "edited", "prime")`。
>
> **軌道の定義**
>
> | 軸 | 定義 |
> |---|---|
> | X | `original` を完全継承（`-0.90 + 0.0375·index`）。`KEYFRAMES["prime"] = KEYFRAMES["original"]` |
> | Y | 不変（0）。断面が動かないので制約2（面積不変）が自動で満たされる |
> | Z | `location.z = dz_at(index)`。`dz_at(i) = DZ_A·(1−cos(2π·i/DZ_PERIOD))/2`（i ≤ 35）、`0.0`（i > 35） |
>
> `DZ_A = 0.85` / `DZ_PERIOD = 35.0` / `DZ_CLAMP_INDEX = 35`。
>
> **重大な実装上の注意**：`location.z` に入れるのは **差分 dZ のみ**。Target 幾何中心の Z（= −0.80）は**メッシュ頂点に焼かれており**（作法#14）、Scene C では `obj.location.z` は 0.0 のままである。
> `scene_c_prime_traj.json` の meta `Z0_note`（"location_z = Z0 + dZ"）は **誤り**。同 meta の `Z0_placeholder = -0.770651` は点群重心の v 範囲 [252.3, 434.8] から depth 8.60 で逆算した推定値であり、幾何中心 −0.80（depth 9.00）とは別物。**meta は使わず `keyframes[].dZ` のみ使う。**
>
> **離散ピーク**：連続ピークは f=17.5 で `A = 0.85` だが、整数フレームでは到達しない。実効ピークは **index 17 と 18 の `0.8482891`**。

---

## 変更2：§33.4 制約1（速度）の余裕を明記

> **制約1 の実測余裕は 4.99%。** `DZ_A` / `DZ_PERIOD` は凍結扱いとする。
>
> | 項目 | 値 |
> |---|---|
> | 最大 2D スクリーン速度（depth 8.60、水平＋垂直の合成） | **11.4014 px/frame**（index 8→9 および 26→27） |
> | `VELOCITY["design_limit"]` | 12.0 |
> | 余裕 | 0.5986 px/frame = **4.99%** |
>
> `scene_c_prime_traj.json` の `screen_v_to_next` は名称と異なり **2D 合成速度**である（水平 5.0388 と垂直成分の合成）。
> 振幅を上げる／周期を短くする変更は即座に上限を超えるため行わない。

---

## 変更3：Scene C′ の照合値を凍結（§37 系に追加）

> ### Scene C′ の凍結リテラル
>
> 経路分離（作法#12）：**作る側 = `scene_c_spec.dz_at()` の閉形式 / 照合側 = `scene_c_prime_traj.json` 由来のリテラル**。
>
> | 追加先 | 定数 | 内容 |
> |---|---|---|
> | `scene_c_spec.py` | `DZ_A` / `DZ_PERIOD` / `DZ_CLAMP_INDEX` / `dz_at()` / `TARGET_CENTRE_Z` / `DZ_MAX_SCREEN_VEL` | 式 |
> | `scene_c_expect.py` | `TRAJ_PRIME`（49行：index, frame, X, location_z, u, v）/ `PRIME_SCALARS` / `PRIME_VERTEX_Z_AT_PEAK` / `TOL_Z=1e-5` / `TOL_V=0.01` | リテラル |
> | `geo_expect.py` | `PRIME_TARGET_V_TOP`（49行：index, v_top, band_px）/ `PRIME_OCCLUDER_TOP_V=232.7778` / `PRIME_BAND_SUBPIXEL_PX=1.5` / `PRIME_V_TOP_SCALARS` | リテラル |
> | `armb_expect.py` | `PRIME_TARGET_B_MAX`（49行：index, 期待 max B）/ `PRIME_TOL_ZLEVEL=2` / `PRIME_B_SCALARS` | リテラル |
>
> **突合実測**：式側とリテラル側の最大差は dZ で `4.71e-08`（JSON の 7 桁丸め由来）、v 換算 `6.05e-06 px`。`TOL_PX = 0.01` の内側。
>
> **主要スカラ**
>
> | 項目 | 値 |
> |---|---|
> | dZ ピーク | 0.8482891（index 17 / 18） |
> | Target 中心 v | 342.7161（dZ=0）→ 233.7999（peak）。移動 108.9161 px |
> | Target 上端 v（depth 8.60） | 253.4367（dZ=0）→ 139.4547（peak）。移動 113.9820 px |
> | ピーク時の Target 頂点 Z（world） | −0.6517109 .. **0.7482891** |
> | `NORM_BOX["z"]` 上限 2.60 に対する余裕 | **1.85 unit** |
> | Target 画素の B チャンネル最大（Arm B 8bit） | 177（dZ=0）→ 197（peak）。振れ幅 20 段階 |
>
> 注：`POINTS` の `Target_centre` は v=342.7160 と記録されているが、これは切り捨て表記であり 342.7161 と `TOL_PX` 内で一致する。矛盾ではない。

---

## 変更4：content sha の計算方式を確定（rev.11 変更9 を解決）

rev.11 変更9 は「§2.2/§36.6 の記録値と W6 の三者一致値の差は計算方式の違い」と**推測**していた。W7 で**実測により確定**した。

> ### content sha256 の計算方式（**方式B に統一**）
>
> ```
> content_sha = sha256( imageio.get_reader でデコードした (T,H,W,3) uint8 の tobytes ).hexdigest()
> ```
> **shape を前置しない。** これを「方式B」と呼ぶ。
>
> | 方式 | 定義 | 対応する記録値 |
> |---|---|---|
> | **A** | `sha256(str(shape) + tobytes)` | §2.2 / §36.6 の記録値（tv=`bba8f4b9e280b9e3`, armC=`46abbcded632a61b`） |
> | **B** | `sha256(tobytes)` | W6 の三者一致値（tv=`6190557123fdc2b3`, armC=`7c08f981503edb34`, armC_prime=`fde80efbe48dcb51`） |
>
> **W7 実測**：既知の記録値を持つ 3 本（`tv_R0S0_original.mkv` / `armC_original.mkv` / `armC_prime.mkv`）に対し、**方式B が 3 本すべてで記録値を再現**した。方式 A は §2.2/§36.6 の記録値を再現した。BGR 順（方式D）・CHW 順（方式E）・ファイル sha（方式F）はいずれも不一致。
>
> **結論**：両記録値はどちらも正しく、**ファイルの異常は存在しない**。Pod との照合に使うのは方式B。
>
> `png_to_ffv1.py` は `content_sha256()`（方式A）と `content_sha256_b()`（方式B）の両方を出力する（W7 改修）。診断用に `sha_probe.py`（6 方式を総当たりして記録値と自動照合）を追加。

---

## 変更5：Scene C′ 成果物の sha 表（§36 系に追加）

> ### Scene C′ control / target video（すべて 49 frames / 832×480 / FFV1 level 3 / gbrp / g=1 / +bitexact）
>
> | 本 | ファイル | content sha (B) | content sha (A) | bytes |
> |---|---|---|---|---|
> | tv | `W3\mkv\tv_R0S0_prime.mkv` | `71d0253f9fdc3161` | `8fa4ae4bb827a020` | 3,466,263 |
> | Arm A | `W3\mkv\armA_prime.mkv` | `3666302824d68ee5` | `1f273ee25c8b73c8` | 2,224,461 |
> | Arm B | `W3\mkv\armB_prime.mkv` | `5a08e10cc228d230` | `77f1cf8b2e21b5f1` | 1,914,588 |
> | Arm C | `W4\mkv\armC_prime.mkv` | `fde80efbe48dcb51` | `a7ac42f40f26b82d` | 1,066,438 |
>
> ### Scene C（参照。方式B）
>
> | 本 | content sha (B) |
> |---|---|
> | tv_R0S0_original | `6190557123fdc2b3` |
> | armA_original | `33cabc1b11b72f45` |
> | armB_original | `d31186af848c80dc` |
> | armC_original | `7c08f981503edb34` |
>
> 4 本すべて `png_to_ffv1.py roundtrip` で **PNG ↔ mkv の bit 一致**を確認済み（不一致画素 0 / 最大絶対差 0）。検証経路は DiffSynth の `LoadVideo` と同一（`imageio.get_reader` → `count_frames` → `get_data`）。

---

## 変更6：制約4 の実証と、その副作用としての評価コード修正（§33.4 / §18-9 系）

> **制約4 は画素レベルで実証された。**
>
> | 検査 | 結果 |
> |---|---|
> | `.blend` の `location.z`（index 36–40） | 厳密に 0.0 |
> | geo の ID パス（frame 37–41） | Target 画素 0（`uniq=[1,2,4,5,6]`） |
> | Arm B の画素値（index 36–40） | max 114。Target 帯 (140, 242) の値が一つも存在しない |
> | Target 可視フレーム数 | **44**（49 − 5）。§12 の評価面「GT で Target 可視な 44 フレーム（0–35 / 41–48）」と一致 |
> | tv / Arm B の index 0 および 35–48 | Scene C と **bit 一致 15/15**。index 1–34 は全て相違 |
>
> index 0 の bit 一致により、§33.11 の「制約3 により `vace_reference_image` は 1 枚に collapse する」が実証された。
>
> ### 副作用：`export_geometry.visible_u` の1次元判定が C′ で破綻する
>
> **原因**：C′ では Target が浮いて遮蔽板の上端（world Z = 0.05 / 画面 v = 232.7778）を越える。Target は X-Y 断面を Z に押し出したプリズムであり **どの Z スライスでも u 範囲は同一**なので、板の上に出た帯がシルエット全幅を覆う。u だけの判定では「板で切られる」と誤った期待値を出す。
>
> **影響範囲**：改修前のコードで正しいデータを判定すると **24/49 フレームが FAIL**（index 6–29）。
>
> **修正（W7、案A採用）**：
> ```
> band_px = 232.7778 − v_top(index)
>   band > 0                       -> 可視シルエットは u 全幅
>   band <= 0                      -> 従来の 1D 判定
>   0 < band < 1.5（サブピクセル）  -> 全幅と 1D 判定の**どちらでも許容**（境界フレーム）
> ```
> **境界フレームは index 5 と 30**（band = 0.842 px）。**W7 実測ではラスタライザが「全幅」側に転んだ**（index 30 で u 右端 516.5、板で切った場合の 401.5 ではない）。代替解の許容は今回発動しなかったが、他 split / 解像度での保険として残す。
>
> 案B（prime では検査をスキップ）は不採用。Arm B の元データの唯一の幾何検証を失うため。

---

## 変更7：検査の空白の是正（§37 系に追加）

**W7 で、Target の Z が全フレームでゼロでも全検査が PASS する状態が 4 箇所で判明した。** いずれも「静かな失敗」であり、実験結果を無言で無効化しうる。

> ### 空白の内容と是正
>
> | スクリプト | 空白 | 是正（W7） |
> |---|---|---|
> | `render_target_video.py` | mean / frac_0 / frac_255 / roundtrip のみ。位置検査ゼロ | `check_position()`：`matrix_world` を読み X / location.z / u / v を `TRAJ_PRIME` と突合 |
> | `render_arm_a.py` | 同上（clay は全同色なので位置が誤っていても「それらしい絵」が出る） | 同上 |
> | `export_geometry.py` | `PROBE_PX` の `target_f1` は `frame_invariant=False` で **index 0 限定**（dZ=0 のフレーム）。シルエット照合は u のみ | `target_v_top`：ID マスクの Target 画素の**最上行**を v に変換して `PRIME_TARGET_V_TOP` と突合。Target 上端は遮蔽状態に関係なく必ず可視 |
> | `encode_arm_b.py` | `PROBE_RGB` の `target_f1` も index 0 限定。**B チャンネル（Z 軸）が dZ>0 で一度も照合されない** | `target_b_max`：ID マスクの Target 画素の **B 最大値**を `PRIME_TARGET_B_MAX` と突合。加えて B の [min, max] が Target 帯 (140, 242) の内側であることを確認 |
>
> ### 壊したデータでの発火確認（作法：assert は壊したデータで発火させる）
>
> | 壊し方 | tv の位置検査 | geo の u 判定 | geo の v_top | armB の B 最大 |
> |---|---|---|---|---|
> | Z 未反映（改修前の挙動） | 34/49 | 23 | 32 | 26/44 |
> | `location.z = Z0 + dZ`（JSON の誤指示） | 49/49 | 23 | 44 | 44/44 |
> | dZ が半分 | — | 3 | 32 | 24/44 |
> | 遮蔽窓の clamp 除去 | — | 13 | — | — |
> | 正しいデータ | **0** | **0** | **0** | **0/44** |
>
> 「dZ が半分」は u 判定だけでは 3 件しか捕まらず、v_top で 32 件捕まる。u と v の両方を見る必要がある根拠。
>
> ### 検査の限界（記録）
>
> Arm B の B 最大値検査は **+5% のスケール誤差を検出できない**（8bit 量子化に埋もれる）。ただし最大誤差 0.042 unit は §34.2 の量子化予算 0.044118 unit（1 LSB の半分）より小さいため、実験結果に影響しない。

---

## 変更8：Arm A の凍結パラメータと C′ での分離性実測（§18-21 系）

> ### Arm A（clay）の凍結パラメータ
>
> | 項目 | 値 | 根拠 |
> |---|---|---|
> | energy | 2000 | §7（全 split 共通） |
> | world（環境光） | 0.10 | sweep で world 0.1 / 0.5 が同一結果 |
> | samples | 32 | R0 = EEVEE |
> | view_transform | Standard | — |
> | `use_raytracing` | **False** | True では同一フレーム2回で差分画素 15。決定性が出ない |
> | 材質 | `ArmA_Clay` 単一 / base_color (0.5, 0.5, 0.5) / roughness 0.6 / metallic 0 | §18-21 |
>
> ### C′ での `label_separability` 実測（49 フレーム）
>
> | 項目 | 値 |
> |---|---|
> | 最大 | **0.8661**（frame 36 = index 35、dZ=0） |
> | 最小 | 0.6241（frame 33 = index 32） |
> | 閾値 `ARMA_LABEL_SEP_MAX` | 0.99 |
> | 余裕 | **0.124** |
>
> **最大値は Scene C と同一（0.8661）。C′ の縦揺れは sep の最大値を一つも押し上げなかった。** §18-21（Arm A は暗黙の Entity ラベルを持たない）は C′ でも維持される。
>
> **機序**：Target が浮いて光源（Z=1.50）に近づき Target 側の平均輝度は上がる（index 0 で 0.5917 → index 12 で 0.6199）が、接地影が消えることで**非 Target 側の平均も下がる**（0.4823 → 0.4718）。分離性は分布の重なりで決まるため単調増加しない。
>
> **`frac_0`（真っ黒画素）の余裕は小さい**：最大 **2.684%**（frame 16 = index 15、dZ=0.8079118）、閾値 3%。余裕 0.32%。**dZ のピーク（frame 18 = index 17）とは一致しない** — X も同時に動くため、Target の下面の暗部が遮蔽板に隠れる分だけ減る。今後 C′ 系のシーンを追加する際は `frac_0` を単変数で予測しないこと。

---

## 変更9：§15 Pipeline floor は推論枠（学習 run 予算を消費しない）

> **Pipeline floor 4 本（Determinism / Encoding / Micro-perturbation / Seed variability）は、いずれも同一の LoRA を使い回して生成側のみを変える測定であり、§21 の学習 run 予算（5 本）を消費しない。**
>
> 学習 run 予算の現状：
>
> | 項目 | 本数 |
> |---|---|
> | 総枠（§21） | 5 |
> | 消費済み（W6 偵察 run = g1_armC = Arm C の S1） | 1 |
> | Phase 0.8 で必要（Arm A / Arm B の S1） | 2 |
> | 残余 | **2** |
>
> g1_armC が Arm C の S1 として流用可能であれば、Phase 0.8 の新規学習は 2 本で足りる。

---

## 変更10：環境の既知問題（§2.6 系 or 付録に追加）

> ### Blender 5.2.0 LTS（hash `fbe6228777e7`）
>
> | 症状 | 原因 | 対処 |
> |---|---|---|
> | `save_as_mainfile` 直後に `EXCEPTION_ACCESS_VIOLATION` でプロセスが死ぬ | `comfyui_blender` アドオンの `load_pre` / `load_post` ハンドラが `AttributeError` / `KeyError` を出し、teardown で落ちる | **`--factory-startup` を必須とする。** 出力（.blend / JSON / PNG）は無傷。`--factory-startup` 付きでは `Blender quit` で正常終了 |
> | `--variant prime` が弾かれる | `scene_c_spec.VARIANTS` に prime が無い | spec を先に更新する |
>
> クラッシュ時も `.blend` は正規に書けている。別プロセスで `--factory-startup <blend> --python-expr` で読み直し、`frame_set(18)` の `matrix_world.translation.z == 0.8482891` を確認して検証した。
>
> ### PowerShell 5.1（日本語 Windows）
>
> **`Get-Content | Add-Content` でファイルを追記してはならない。** 既定エンコーディングが ANSI（Shift-JIS）なので UTF-8 の日本語コメントが破壊され、Python が `SyntaxError` / `UnicodeDecodeError` で落ちる。`-Encoding UTF8` を付けても BOM が途中に埋まる。
>
> **正しい追記**：
> ```
> cmd /c copy /b base.py + append.py new.py
> Move-Item new.py base.py -Force
> ```
> `copy /b` はバイト列をそのまま連結するため文字化けが原理的に起きない。追記対象・追記元がともに改行終端で BOM 無しであることを事前に確認する。
>
> ### ファイル配置（W7 で確定）
>
> `C:\Bachelor Thesis\SummerWork\W2\.py\` が **スクリプト本体の唯一の正**（spec / expect / png_io / armc_palette / geo_expect などの依存が全てここにある）。成果物は `W2\RENDER\{tv,geo,armA,armB}\<variant>\`、mkv は `W3\mkv\`（Arm C のみ `W4\mkv\`）。`W4\.py\` は改修版のダウンロード先であり本体ではない。

---

## 変更11：Scene C′ 成果物の自己検査サマリ（§37 系に追加）

> | 段 | スクリプト | 出力 | SELF-CHECK |
> |---|---|---|---|
> | `.blend` | `build_scene_c.py --variant prime` | `scene_c_prime.blend`（98,219 B） | **PASS 32 / FAIL 0** |
> | tv | `render_target_video.py` | `RENDER\tv\R0S0\prime\` 49 枚 | **PASS 104 / FAIL 0** |
> | geo | `export_geometry.py` | `RENDER\geo\prime\` 147 枚（position / depth / id × 49） | **PASS 150 / FAIL 0** |
> | Arm A | `render_arm_a.py` | `RENDER\armA\prime\` 49 枚 | **PASS 105 / FAIL 0** |
> | Arm B | `encode_arm_b.py` | `RENDER\armB\prime\` 49 枚 | **PASS 52 / FAIL 0** |
>
> 補助実測：
> - geo の `target_v_top` 誤差は最大 **1.019 px**（許容 1.5、検査対象 44 フレーム）。縁画素の被覆規則由来の系統誤差であり、既存の u 誤差（0.90 / 0.47 / 0.65 / 0.83）と同オーダー。
> - Arm B の `q_excess` は全フレーム厳密に **0.00e+00**（許容 1.0e-03）。NORM_BOX 外に一切出ていない。
> - Arm B の Target 帯下限側の余裕は小さい：`target_b_min` = 143（index 0、帯下限 140 まで 3 段階）。
> - `original` 系の再レンダは既存成果物と **bit 一致**（tv 5/5、Arm B 5/5）。`target_z()` が `original` で 0.0 を返すため、既存の frozen sha は保たれている。

---

## 未決事項テーブルの更新後の姿

| # | 内容 | 待ち条件 |
|---|---|---|
| ~~2~~ | ~~gradient_accumulation_steps~~ | 解決＝1（rev.11） |
| 20 | 本番シーンでの色の一意性の実地検証 | 本番シーン構築後 |
| 21 | 評価 GT 側で Arm A の Target を同定する方法 | **arm A 着手済み。C′ では geo の ID パス（ids==3）で同定できることを実測。本番シーンでの方法は未決のまま** |
| 23 | train 軌道本数 N | 暫定 N=1。3 arm の結果で確定 |
| 24 | Occluder 除去方法（c-x / c-y / c-z） | 本実験時 |
| 25 | T_coherence 2/44 が生成破綻か tracker 限界か | f34/f35 切り分け |
| 新 26 | 学習マニフェスト / メタデータの形式 | `manifest.py` / `make_metadata.py` の確認後 |
| ~~新（rev.11）~~ | ~~content sha の計算方式~~ | **解決＝方式B（変更4）** |

---

## 適用チェックリスト

- [ ] 版表記を rev.12 に
- [ ] 変更1〜11 を本体の該当節に反映
- [ ] §2.2 / §36.6 の記録値に「方式A」と明記（値は変更しない）
- [ ] 未決 #21 の記述を更新、rev.11 で追加した content sha の項目を削除、#26 を追加
- [ ] `sha_probe.py` を凍結スクリプト一覧に追加
