# 3D Video Control Comparing

**When teaching Wan2.1-VACE-1.3B a "3D control video" via LoRA, what changes depending on how the control video is represented?** This experiment compares three representations (Arms) under the same budget and the same conditions.

The preparation stage was completed in summer 2026 (W0–W8). **The experiments and the thesis draft were completed in November 2026 (W9–W13).** Start with [`docs/thesis/thesis.md`](docs/thesis/thesis.md).

---

## Read this first

1. [`docs/PRE_EXPERIMENT_SUMMARY_v1.md`](docs/PRE_EXPERIMENT_SUMMARY_v1.md) — overall picture right before the main experiment (index of confirmed decisions)
2. [`docs/protocol/`](docs/protocol/) — `protocol.md` (main document, **rev.14**; rev.11–14 diffs are merged as §38–§41) + the individual `protocol_rev1{1,2,3,4}_changes.md` diffs
3. [`docs/worklog/`](docs/worklog/) — weekly rationale for decisions, and records of predictions and where they were wrong

## The three Arms

| Arm | Representation | Content |
|---|---|---|
| **A** | Clay Previs | Video of all objects rendered in a single clay color (appearance present, 3D implicit) |
| **B** | Object-Aware Dense World-XYZ | Video with each pixel's world coordinates packed into RGB (Target made explicit) |
| **C** | Object-Aware 3D Tracking | Sparse video tracking points on object surfaces as fixed-color dots |

**Only the control video differs.** Everything else (49 frames / 832×480 / 16 fps / prompt / LoRA settings / seed) is shared across all Arms.

## Current status

| | Status |
|---|---|
| 3-Arm control + target video for Scene C (train) / Scene C′ (IID test) | ✅ Complete; PNG↔mkv bit-identity confirmed |
| Training metadata (jsonl × 3) | ✅ |
| Scouting run (Arm C, 500 steps) | ✅ T_control median 8.55 px; the model reads the control rather than memorizing |
| Scoring rules fixed before Phase 0.8 (rev.14: GT from Scene C′ ID pass, T_coherence by frame count) | ✅ `scripts/pod/score_t_control_v2.py` |
| Phase 0.8: train Arm A / B (2 runs) + inference | ✅ W9 |
| Floors + Claim Gate (RQ1), RQ2 spillover | ✅ W10–W11 (pre-registered; primary RQ1 and RQ2: no claim) |
| 20-seed follow-up, learning curve, zero-shot, Mask IoU | ✅ W12 (A and B more accurate than C; all arms usable after ~100 steps) |
| **Thesis v1.0** | ✅ [`docs/thesis/thesis.md`](docs/thesis/thesis.md) |
| Pipeline floor (4 runs) → Claim Gate | ⬜ |
| Phase X (Wan vs LTX cross-family validation) | ⬜ Design only ([`docs/PHASE_X_wan_vs_ltx_design_note.md`](docs/PHASE_X_wan_vs_ltx_design_note.md)) |

## Folder structure

```
MainProject/
├─ docs/                 Plans and records (protocol, worklog, summary, Phase X design)
├─ scripts/
│  ├─ blender/           Scene construction, rendering, generation for each Arm (Blender 5.2, --factory-startup required)
│  ├─ pipeline/          PNG→FFV1 mkv conversion, metadata generation, sha checks, Pod transfer prep, Arm B/C verification
│  ├─ pod/               RunPod side: inference and evaluation (score_t_control / evaluator_floor)
│  └─ vae_survival/      Phase 0.0: checks whether Arm C dots survive VAE compression
├─ scenes/               scene_c_original / scene_c_prime / scene_c_edited (.blend)
├─ data/
│  ├─ inputs/            ★ Final set passed to training/evaluation (8 mkv + reference image) = /workspace/w4/mkv on the Pod
│  ├─ meta/              ★ Training metadata (metadata_arm{A,B,C}.jsonl) = /workspace/w4/meta on the Pod
│  ├─ render/            Rendered PNGs (tv / geo / armA / armB / armC × original / edited / prime)
│  ├─ points/            Arm C point data and palette, Scene C′ trajectory
│  └─ reports/           SELF-CHECK / measurement reports from each script (json, txt)
├─ manifest/             md5 / size tables (for bit-identity checks on transfer and regeneration)
├─ runs/                 Training run records (training_args / log / generated results). *.safetensors not tracked by git
├─ archive/              Original tar.gz archives retrieved from the Pod (with md5). Extracted copies live in runs/
└─ references/           Reference paper PDFs (not tracked by git)
```

### Mapping from old folders (for reading paths in docs)

Old paths (`C:\Bachelor Thesis\SummerWork\...`) remain in docs and script docstrings. Map them as follows:

| Old | New |
|---|---|
| `SummerWork\W2\.py\` | `scripts/blender/` (only `.blend` files go to `scenes/`) |
| `SummerWork\W3\.py\` | `scripts/pipeline/` |
| `SummerWork\W4\.py\` + contents of `W4\.gz\*.tar.gz` | `scripts/pod/` |
| `SummerWork\W1\.py\` | `scripts/vae_survival/` |
| `SummerWork\W8\pod_upload\mkv\` (= `W3\mkv` + `W4\mkv\armC_prime.mkv`) | `data/inputs/` |
| `SummerWork\W3\meta\` | `data/meta/` |
| `SummerWork\W2\RENDER\` | `data/render/` |
| `SummerWork\W2\.json\`, `W3\.json\` | `data/points/` (points, trajectories) / `data/reports/` (reports) |
| `SummerWork\W4\runs\`, `W4\.log\` | `runs/` |
| `SummerWork\protocol\`, `W*\.md\`, `W*\.mb\` | `docs/` |

Some scripts in `scripts/pipeline/` (e.g. `dump_geo_npy.py`) import modules from `scripts/blender/` (`scene_c_spec`, `armb_encode`, `png_io`, …). Pass `--pypath scripts/blender` or add it to `PYTHONPATH`.

### Notes

- There are two versions: `data/points/armc_prime_points_W3.json` and `_W4.json` (see W7-06). **Use `_W4.json`.** Re-rendering both shows only W4 reproduces `armC_prime.mkv` bit-for-bit (W3 differs in frames 8–30; protocol §41.7).
- `score_t_control.py` (v1) builds the Scene C′ GT by shifting the Scene C mask centroid and is off by up to 31.9 px. Use `score_t_control_v2.py` with `data/points/gt_target_center_prime.json` (protocol §41.1).
- Do not guess training-side values. Follow the frozen values in `docs/PRE_EXPERIMENT_SUMMARY_v1.md` §3–4.

## Excluded during reorganization (still in the original folder)

- The July former direction (comparing Depth/Canny/MLSD/Camera with Wan2.1-Fun-Control, apple scene, VidFM3D/CO3D): `Blender/`, `Test/`, `milestone5/6_notes.md`, `VidFM3D_環境構築ノート.md`
- protocol rev6–9 and rev8 patches (merged into rev10)
- Regenerable derivatives: `.npy`, `.tar/.zip`, `W3/emit` (Arm C k-sweep), `W3/vae_armb`, `W2/.exr` (early probes), `W4/runs/gen_png`
- `.bak`, `__pycache__`, `_selftest_tmp`, duplicate mkv / scripts
