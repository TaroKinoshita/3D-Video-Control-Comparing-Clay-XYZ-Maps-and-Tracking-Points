# 3D Video Control Comparing

**Wan2.1-VACE-1.3B に LoRA で「3D の制御映像」を覚えさせるとき、制御映像の“表現の仕方”で何が変わるか** を、3 つの表現（Arm）を同じ予算・同じ条件で比べる実験。

2026 年夏（W0–W8）に準備段階まで完了している。**本実験（Phase 0.8）は未着手。**

---

## まずここを読む

1. [`docs/PRE_EXPERIMENT_SUMMARY_v1.md`](docs/PRE_EXPERIMENT_SUMMARY_v1.md) — 本実験直前の全体像（確定事項の索引）
2. [`docs/protocol/`](docs/protocol/) — `protocol_rev10.md`（本体）+ rev11 / 12 / 13 の差分
3. [`docs/worklog/`](docs/worklog/) — 各週の判断理由・予測と外れの記録

## 3 つの Arm

| Arm | 表現 | 中身 |
|---|---|---|
| **A** | Clay Previs | 全物体を同色クレイでレンダーした映像（外観あり・3D は暗黙） |
| **B** | Object-Aware Dense World-XYZ | 各画素の world 座標を RGB に詰めた映像（Target を明示） |
| **C** | Object-Aware 3D Tracking | 物体表面の点を固定色のドットで追跡した疎な映像 |

違うのは **control video だけ**。それ以外（49 frames / 832×480 / 16 fps / prompt / LoRA 設定 / seed）は全 Arm 共通。

## 現在地

| | 状態 |
|---|---|
| Scene C（train）/ Scene C′（IID テスト）の 3 Arm control + target video | ✅ 完成・PNG↔mkv bit 一致確認済み |
| 学習 metadata（jsonl × 3） | ✅ |
| 偵察 run（Arm C, 500 step） | ✅ T_control median 8.55 px、暗記ではなく control を読めている |
| **Phase 0.8：Arm A / B の学習 2 本 + 推論** | ⬜ **次にやること** |
| Pipeline floor 4 本 → Claim Gate | ⬜ |
| Phase X（Wan vs LTX cross-family 検証） | ⬜ 構想のみ（[`docs/PHASE_X_wan_vs_ltx_design_note.md`](docs/PHASE_X_wan_vs_ltx_design_note.md)） |

## フォルダ構成

```
MainProject/
├─ docs/                 計画・記録（protocol, worklog, summary, Phase X 構想）
├─ scripts/
│  ├─ blender/           シーン構築・レンダー・各 Arm の生成（Blender 5.2, --factory-startup 必須）
│  ├─ pipeline/          PNG→FFV1 mkv 変換、metadata 生成、sha 照合、Pod 転送準備、Arm B/C の検証
│  ├─ pod/               RunPod 側：推論・評価（score_t_control / evaluator_floor）
│  └─ vae_survival/      Phase 0.0：Arm C のドットが VAE 圧縮を生き残るかの検証
├─ scenes/               scene_c_original / scene_c_prime / scene_c_edited (.blend)
├─ data/
│  ├─ inputs/            ★ 学習・評価に渡す最終セット（mkv 8 本 + 参照画像）= Pod の /workspace/w4/mkv
│  ├─ meta/              ★ 学習 metadata（metadata_arm{A,B,C}.jsonl）= Pod の /workspace/w4/meta
│  ├─ render/            レンダー PNG（tv / geo / armA / armB / armC × original / edited / prime）
│  ├─ points/            Arm C の点データ・パレット、Scene C′ の軌道
│  └─ reports/           各スクリプトの SELF-CHECK / 実測レポート（json, txt）
├─ manifest/             md5 / size の照合表（転送・再生成の bit 一致検証用）
├─ runs/                 学習 run の記録（training_args / log / 生成結果）。*.safetensors は git 管理外
└─ references/           参考論文 PDF（git 管理外）
```

### 旧フォルダとの対応（docs 内のパスを読むとき用）

docs や script の docstring には旧パス（`C:\Bachelor Thesis\SummerWork\...`）が残っている。読み替えはこれ：

| 旧 | 新 |
|---|---|
| `SummerWork\W2\.py\` | `scripts/blender/`（`.blend` だけ `scenes/`） |
| `SummerWork\W3\.py\` | `scripts/pipeline/` |
| `SummerWork\W4\.py\` + `W4\.gz\*.tar.gz` の中身 | `scripts/pod/` |
| `SummerWork\W1\.py\` | `scripts/vae_survival/` |
| `SummerWork\W8\pod_upload\mkv\`（= `W3\mkv` + `W4\mkv\armC_prime.mkv`） | `data/inputs/` |
| `SummerWork\W3\meta\` | `data/meta/` |
| `SummerWork\W2\RENDER\` | `data/render/` |
| `SummerWork\W2\.json\`, `W3\.json\` | `data/points/`（点・軌道）/ `data/reports/`（レポート） |
| `SummerWork\W4\runs\`, `W4\.log\` | `runs/` |
| `SummerWork\protocol\`, `W*\.md\`, `W*\.mb\` | `docs/` |

`scripts/pipeline/` の一部（`dump_geo_npy.py` など）は `scripts/blender/` のモジュール（`scene_c_spec`, `armb_encode`, `png_io` …）を import する。`--pypath scripts/blender` を渡すか、`PYTHONPATH` に足して使う。

### 注意

- `data/points/armc_prime_points_W3.json` と `_W4.json` の 2 版がある（W7-06 参照）。`armC_prime.mkv` は sha で W6 と同一と確認済みなので mkv 側は問題なし。
- 学習側の数値は推測しない。`docs/PRE_EXPERIMENT_SUMMARY_v1.md` §3–4 の凍結値に従う。

## 整理で外したもの（元フォルダに残っている）

- 7 月の旧方針（Wan2.1-Fun-Control で Depth/Canny/MLSD/Camera を比較、りんごシーン、VidFM3D/CO3D）：`Blender/`, `Test/`, `milestone5/6_notes.md`, `VidFM3D_環境構築ノート.md`
- protocol rev6–9 と rev8 patch（rev10 に統合済み）
- 再生成できる派生物：`.npy`、`.tar/.zip`、`W3/emit`（Arm C の k スイープ）、`W3/vae_armb`、`W2/.exr`（初期プローブ）、`W4/runs/gen_png`
- `.bak`、`__pycache__`、`_selftest_tmp`、重複していた mkv / script
