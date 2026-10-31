# protocol rev.12 → rev.13 更新差分（W8 反映、2026-08-12）

学習マニフェスト / メタデータの確定。**確定事項の追加・訂正のみ。設計の骨格は不変。**

**前提**：rev.11 / rev.12 の差分が本体に未適用であれば、順に適用してから本ファイルを適用する。

---

## 変更1：未決 #26 を解決（§36.7 に追記）

> ### 学習 metadata と manifest の役割分担（確定）
>
> | 対象 | 形式 | 生成 | 内容 |
> |---|---|---|---|
> | 学習 metadata | jsonl 3 本（`metadata_arm{A,B,C}.jsonl`） | `make_metadata.py` v2 | **original のみ**。240 bytes / 1 行 / 4 列 |
> | 資産 manifest | TSV（md5 / size / relpath） | `manifest.py` / `stage_pod_inputs.py` | Pod 転送の bit 一致検証 |
>
> **prime（Scene C′）用の metadata jsonl は作らない。**
> 理由：§33.4 a により train は Scene C R0S0 Original のみ。C′ は評価側であり、
> DiffSynth の推論は jsonl を読まず frame の list を直接受け取る（rev.11 変更4）。
> W7 worklog 7章の「prime の 4 本を DiffSynth が読む形式に組む」は、
> **jsonl 化ではなく base_path への集約と sha 照合**を指すものとして解決した。
>
> **列名は確定済み**（`video` / `prompt` / `vace_video` / `vace_reference_image`。
> rev.10 §36.7、未決 #25 で 2026-08-09 に解決）。
> `make_metadata.py` v1 の docstring にあった「`prompt` は未確認」は古い記述であり v2 で削除した。
>
> **生成物の凍結リテラル**（3 arm すべて 240 bytes）：
>
> | arm | sha256（先頭 16） |
> |---|---|
> | armA | `b98633171d33d47f` |
> | armB | `8ed8deedd11f4ef7` |
> | armC | `33e531b5c7888662` |
>
> armC は W6 の偵察 run で実際に使った `metadata_armC.jsonl` と **bit 一致を確認**した。

---

## 変更2：base_path は 1 本なので集約する（§36.14 に追記）

> `--dataset_base_path` は 1 本しか渡せない（`os.path.join(base, rel)`）。
> 手元の mkv は tv / armA / armB が `W3\mkv`、armC が `W4\mkv` に分かれているため、
> **9 ファイルを 1 ディレクトリに集約してから転送する**（`stage_pod_inputs.py`）。
>
> 相対パスに `..` を入れる案は不採用。`ToAbsolutePath` の解決を跨いで壊れやすく、
> metadata の 3 arm 同一性検査の意味も濁る。
>
> **集約対象（9 件）**
>
> | 用途 | ファイル |
> |---|---|
> | 学習 | `tv_R0S0_original.mkv` / `armA_original.mkv` / `armB_original.mkv` / `armC_original.mkv` / `ref_R0S0_original.png` |
> | 評価（推論の control 入力） | `tv_R0S0_prime.mkv` / `armA_prime.mkv` / `armB_prime.mkv` / `armC_prime.mkv` |
>
> prime を同じディレクトリに置くのは推論時に読むためであり、metadata には現れない
> （`FORBIDDEN_SUBSTRINGS` に `"prime"` を追加して保証）。

---

## 変更3：検査の空白の是正（§37 系に追記。W7 変更7 の続き）

**W8 で `make_metadata.py` v1 に 2 箇所の静かな失敗が判明した。**

> | 空白 | 症状 | 是正（v2） |
> |---|---|---|
> | `FORBIDDEN_SUBSTRINGS` に `"prime"` が無い | 定数（`TARGET_VIDEO` / `ARM_CONTROL`）を prime に書き換えると **全検査が PASS する**。C′ で学習してしまい RQ が無意味になる | 禁止語に `"prime"` を追加 |
> | 生成バイト列の回帰が無い | `PROMPT` を 3 arm 同時に変えると `check_cross_arm_identity` は通る（3 arm で一致しているため）。W6 の run と prompt が違う状態で学習が走る | `FROZEN_JSONL`（sha256 + bytes）と突合する `check_frozen_jsonl()` |
> | 実在確認のみで中身を見ていない | ファイルは在るが別の中身、を検出できない | `--check-sha`：参照先 mkv の content sha（方式B）を凍結値と突合 |
>
> ### 壊したデータでの発火確認（8 ケース、全て期待どおり）
>
> | 壊し方 | v1 | v2 |
> |---|---|---|
> | prompt に空白 1 文字（1 arm） | 検出 | 検出 |
> | vace_video の重複 | 検出 | 検出 |
> | Edited の混入（1 arm） | 検出 | 検出 |
> | prime の混入（1 arm） | 検出 | 検出 |
> | **prime の混入（全 arm）** | **素通り** | **検出** |
> | **prompt を 3 arm 同時に変更** | **素通り** | **検出** |
> | dict の挿入順を変える（発火しないのが正） | 素通り | 素通り |
> | 正常データ | PASS | PASS |
>
> `stage_pod_inputs.py` 側も合成 FFV1 で 9 ケース発火確認済み
> （方式A/B の弁別、1 LSB 差、bytes 不一致、frame 数不一致）。

---

## 変更4：未決 #27 を追加（参照画像の content sha 方式）

> `ref_R0S0_original.png` の content sha は rev.10 §36.5 に**先頭 24 桁
> `df7c595457c81e35b47164ae` のみ**記録されており、方式A / 方式B のどちらかが不明。
> 動画と違い単一 PNG なので `(1,H,W,3)` として計算したのか `(H,W,3)` かも不明。
>
> 暫定：**PNG は file md5（manifest）でのみ照合する。** content sha 照合は行わない。
> 解決条件：`sha_probe.py` を PNG 対応に拡張して総当たりする（mkv で同じ手が効いた）。

---

## 未決事項テーブルの更新後の姿

| # | 内容 | 待ち条件 |
|---|---|---|
| 20 | 本番シーンでの色の一意性の実地検証 | 本番シーン構築後 |
| 21 | 評価 GT 側で Arm A の Target を同定する方法 | 本番シーンでの方法は未決 |
| 23 | train 軌道本数 N | 暫定 N=1。3 arm の結果で確定 |
| 24 | Occluder 除去方法（c-x / c-y / c-z） | 本実験時 |
| 25 | T_coherence 2/44 が生成破綻か tracker 限界か | f34/f35 切り分け |
| ~~26~~ | ~~学習マニフェスト / メタデータの形式~~ | **解決（変更1・2・3）** |
| 新 27 | 参照画像 PNG の content sha 方式 | `sha_probe.py` の PNG 対応後 |

---

## 適用チェックリスト

- [ ] 版表記を rev.13 に
- [ ] 変更1〜4 を本体の該当節に反映
- [ ] 未決 #26 を削除、#27 を追加
- [ ] `make_metadata.py`（v2）/ `stage_pod_inputs.py` を凍結スクリプト一覧に追加
