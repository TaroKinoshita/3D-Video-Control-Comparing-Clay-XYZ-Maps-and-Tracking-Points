#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiffSynth 2.0.18 用 metadata（jsonl）を 3 arm 分生成する。

なぜ生成するのか
  3 arm の metadata は `vace_video` の 1 列だけが違い、他は完全に同一でなければ
  ならない（違えば RQ1 の交絡になる）。手で 3 ファイル書くと prompt に 1 文字の
  差が入っても気づかない。1 つの定数から生成し、差分を assert する。

形式を jsonl にした理由
  csv は pandas.read_csv が型推論するため、空の prompt が NaN（float）になって
  model に float が渡る。jsonl は型推論が無い。

encoding
  UnifiedDataset.load_metadata の open() は encoding を指定していない
  （unified_dataset.py:76, 81）。Pod は Linux で既定 UTF-8 だが、
  環境依存を避けるため PROMPT は ASCII のみとし、書き出し側で検査する。

列名の根拠
  vace_video / vace_reference_image : training_module.py:358 が
      inputs_shared[extra_input] = data[extra_input] を行うため、
      --extra_inputs の名前が metadata の列名そのままになる。
  video : --data_file_keys の既定が "image,video"（parsers.py:9）。
  prompt : **未確認。** examples/train.py が sdist に含まれないため、
      Pod で `python train.py --help` および train.py 本体で確認する。
      列名が違った場合は PROMPT_KEY を変えて再生成する。

使い方（ローカル / PowerShell）:
  python make_metadata.py --outdir "C:\\Bachelor Thesis\\SummerWork\\W3\\meta"
  python make_metadata.py --outdir <DIR> --check-files "C:\\Bachelor Thesis\\SummerWork\\W3\\mkv"
"""
import argparse
import json
import os
import sys

# --------------------------------------------------------------------------
# 全 arm・全 Scene で共有する定数。ここだけを編集する。
# --------------------------------------------------------------------------

# 運動を記述しない。軌道は control が指定する情報であり、prompt に書くと
# control の仕事を奪う（§33.11 の理由1 と同じ論法）。
PROMPT = ("static camera, simple matte geometric solids on a flat floor, "
          "plain even lighting, an orange hexagonal prism")

PROMPT_KEY = "prompt"            # 未確認。Pod で train.py を見て確定する
VIDEO_KEY = "video"
VACE_VIDEO_KEY = "vace_video"
VACE_REF_KEY = "vace_reference_image"

# base_path からの相対パス
TARGET_VIDEO = "tv_R0S0_original.mkv"
REFERENCE_IMAGE = "ref_R0S0_original.png"
ARM_CONTROL = {
    "armA": "armA_original.mkv",
    "armB": "armB_original.mkv",
    "armC": "armC_original.mkv",
}

# §33.4 a: train は Scene C R0S0 の Original のみ。Edited は学習に一切使わない。
FORBIDDEN_SUBSTRINGS = ("edited", "R1S0", "R0S1")


# --------------------------------------------------------------------------
def build_record(arm):
    return {
        VIDEO_KEY: TARGET_VIDEO,
        VACE_VIDEO_KEY: ARM_CONTROL[arm],
        VACE_REF_KEY: REFERENCE_IMAGE,
        PROMPT_KEY: PROMPT,
    }


def check_prompt_ascii():
    try:
        PROMPT.encode("ascii")
    except UnicodeEncodeError as e:
        raise SystemExit(f"[FATAL] PROMPT に非 ASCII 文字がある: {e}\n"
                         "        load_metadata の open() は encoding 未指定のため環境依存になる。")


def check_forbidden(records):
    """Edited / 別 split が混ざっていないか（§33.4 a）。"""
    for arm, rec in records.items():
        for k, v in rec.items():
            if not isinstance(v, str):
                continue
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad.lower() in v.lower():
                    raise SystemExit(
                        f"[FATAL] {arm} の {k} に禁止語 '{bad}' が入っている: {v}\n"
                        "        §33.4 a: train は Scene C R0S0 Original のみ。")


def check_cross_arm_identity(records):
    """vace_video 以外の全列が 3 arm で完全一致することを検査する。"""
    arms = sorted(records)
    keys = {frozenset(records[a].keys()) for a in arms}
    if len(keys) != 1:
        raise SystemExit(f"[FATAL] arm 間で列の集合が違う: {keys}")
    ref = records[arms[0]]
    for k in sorted(ref):
        if k == VACE_VIDEO_KEY:
            continue
        vals = {records[a][k] for a in arms}
        if len(vals) != 1:
            raise SystemExit(f"[FATAL] 列 '{k}' が arm 間で違う: {vals}")
    vals = {records[a][VACE_VIDEO_KEY] for a in arms}
    if len(vals) != len(arms):
        raise SystemExit(f"[FATAL] vace_video が arm 間で重複している: {vals}")
    return True


def check_files_exist(records, base):
    missing = []
    for arm, rec in records.items():
        for k in (VIDEO_KEY, VACE_VIDEO_KEY, VACE_REF_KEY):
            p = os.path.join(base, rec[k])
            if not os.path.isfile(p):
                missing.append(f"{arm}.{k} -> {p}")
    if missing:
        print("[check-files] 見つからないファイル:")
        for m in missing:
            print("   ", m)
        return False
    print(f"[check-files] {len(records) * 3} 件すべて存在（base={base}）")
    return True


def write_jsonl(path, record):
    # UnifiedDataset は 1 行 1 件で読む（unified_dataset.py:79-84）
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def selftest():
    """assert が壊したデータで発火するかを確認する。"""
    print("===== selftest =====")
    cases = []

    r = {a: build_record(a) for a in ARM_CONTROL}
    try:
        check_cross_arm_identity(r)
        cases.append(("正常データ", "PASS", True))
    except SystemExit:
        cases.append(("正常データ", "FAIL", False))

    r = {a: build_record(a) for a in ARM_CONTROL}
    r["armB"][PROMPT_KEY] = PROMPT + " "          # 末尾に空白 1 文字
    try:
        check_cross_arm_identity(r)
        cases.append(("prompt に空白1文字の差", "検出できず", False))
    except SystemExit:
        cases.append(("prompt に空白1文字の差", "検出", True))

    r = {a: build_record(a) for a in ARM_CONTROL}
    r["armC"][VACE_VIDEO_KEY] = ARM_CONTROL["armB"]   # arm が重複
    try:
        check_cross_arm_identity(r)
        cases.append(("vace_video の重複", "検出できず", False))
    except SystemExit:
        cases.append(("vace_video の重複", "検出", True))

    r = {a: build_record(a) for a in ARM_CONTROL}
    r["armA"][VIDEO_KEY] = "tv_R0S0_edited.mkv"
    try:
        check_forbidden(r)
        cases.append(("Edited の混入", "検出できず", False))
    except SystemExit:
        cases.append(("Edited の混入", "検出", True))

    print("| ケース | 結果 | 期待どおり |")
    print("|---|---|---|")
    for name, res, ok in cases:
        print(f"| {name} | {res} | {ok} |")
    ok = all(c[2] for c in cases)
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", help="metadata を書き出すディレクトリ")
    ap.add_argument("--check-files", default=None,
                    help="mkv / png が実在するか確認する base_path")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.outdir:
        ap.error("--outdir が必要（または --selftest）")

    check_prompt_ascii()
    records = {arm: build_record(arm) for arm in sorted(ARM_CONTROL)}
    check_forbidden(records)
    check_cross_arm_identity(records)
    print("[check] prompt は ASCII / 禁止語なし / vace_video 以外は 3 arm 完全一致")

    if a.check_files:
        check_files_exist(records, a.check_files)

    os.makedirs(a.outdir, exist_ok=True)
    print()
    print("| arm | ファイル | vace_video | N |")
    print("|---|---|---|---|")
    for arm in sorted(records):
        out = os.path.join(a.outdir, f"metadata_{arm}.jsonl")
        write_jsonl(out, records[arm])
        print(f"| {arm} | `metadata_{arm}.jsonl` | `{records[arm][VACE_VIDEO_KEY]}` | 1 |")

    print()
    print(f"[prompt] {PROMPT!r}")
    print(f"[prompt] 長さ {len(PROMPT)} 文字（3 arm 共通）")
    print()
    print("学習時に渡すフラグ:")
    print(f'  --data_file_keys "{VIDEO_KEY},{VACE_VIDEO_KEY},{VACE_REF_KEY}"')
    print(f'  --extra_inputs "{VACE_VIDEO_KEY},{VACE_REF_KEY}"')
    print()
    print(f"[注意] '{PROMPT_KEY}' という列名は未確認。Pod で train.py を見て確定する。")


if __name__ == "__main__":
    main()
