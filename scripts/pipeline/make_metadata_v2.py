#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiffSynth 2.0.18 用 metadata（jsonl）を 3 arm 分生成する。  v2（W8, 2026-08-12）

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

列名（**確定済み**）
  video / prompt / vace_video / vace_reference_image。
  protocol rev.10 §36.7 / 未決 #25（2026-08-09 解決、examples/train.py を実物確認）。
  v1 の docstring にあった「prompt は未確認」は古い記述であり v2 で削除した。

学習に使うのは original のみ（**prime は使わない**）
  §33.4 a：train は Scene C R0S0 Original のみ。
  Scene C′（variant prime）は **評価側（推論の control 入力）専用**であり、
  DiffSynth の推論は jsonl を読まずフレームの list を直接受け取る（rev.11 変更4）。
  したがって prime 用の metadata jsonl は存在してはならない。
  v1 の FORBIDDEN_SUBSTRINGS に "prime" が無く、**定数を prime に書き換えると
  全検査が PASS する**（W8 で発火確認済み）。v2 で禁止語に追加した。

v2 の追加検査
  1. 禁止語に "prime" を追加（上記）
  2. FROZEN_JSONL：生成バイト列を既存成果物と sha256 で突合（回帰）
  3. --check-sha：参照先 mkv の content sha（方式B）を凍結リテラルと突合。
     「ファイルは在るが中身が別」を捕まえる。v1 は存在確認のみだった。

使い方（ローカル / PowerShell）:
  python make_metadata_v2.py --selftest
  python make_metadata_v2.py --outdir "C:\\Bachelor Thesis\\SummerWork\\W3\\meta"
  python make_metadata_v2.py --outdir <DIR> --check-files "<POD_UPLOAD>\\mkv" --check-sha
"""
import argparse
import hashlib
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

PROMPT_KEY = "prompt"            # 確定（rev.10 §36.7）
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

# §33.4 a: train は Scene C R0S0 の Original のみ。
# Edited / 別 split / **prime（= Scene C′、評価専用）** は学習に一切使わない。
FORBIDDEN_SUBSTRINGS = ("edited", "R1S0", "R0S1", "prime")

# 生成物の凍結リテラル（v1 が実際に吐いた 240 bytes と bit 一致することを保証する）
# 出所：W6 に Pod で使った metadata_armC.jsonl（240 B）と同一生成規則。
FROZEN_JSONL = {
    "armA": ("b98633171d33d47f", 240),
    "armB": ("8ed8deedd11f4ef7", 240),
    "armC": ("33e531b5c7888662", 240),
}

# content sha（方式B = sha256(imageio デコード RGB (T,H,W,3) uint8 の tobytes)）
# rev.12 変更4 で方式Bに統一。値は rev.12 変更5 の表（先頭 16 桁）。
FROZEN_SHA_B = {
    "tv_R0S0_original.mkv": "6190557123fdc2b3",
    "armA_original.mkv":    "33cabc1b11b72f45",
    "armB_original.mkv":    "d31186af848c80dc",
    "armC_original.mkv":    "7c08f981503edb34",
}
# 参照画像 PNG の content sha は方式が未確定（rev.10 §36.5 に先頭 24 桁
# `df7c595457c81e35b47164ae` のみ記録。方式A/B の判別が付いていない）。
# → PNG は file md5（manifest.py）でのみ照合する。未決事項として残す。
REFERENCE_IMAGE_SHA_METHOD_UNKNOWN = True


# --------------------------------------------------------------------------
def build_record(arm):
    return {
        VIDEO_KEY: TARGET_VIDEO,
        VACE_VIDEO_KEY: ARM_CONTROL[arm],
        VACE_REF_KEY: REFERENCE_IMAGE,
        PROMPT_KEY: PROMPT,
    }


def record_bytes(rec):
    """書き出しと完全に同一のバイト列を返す（write_jsonl と 1 経路に保つ）。"""
    return (json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")


def check_prompt_ascii():
    try:
        PROMPT.encode("ascii")
    except UnicodeEncodeError as e:
        raise SystemExit(f"[FATAL] PROMPT に非 ASCII 文字がある: {e}\n"
                         "        load_metadata の open() は encoding 未指定のため環境依存になる。")


def check_forbidden(records):
    """Edited / 別 split / prime が混ざっていないか（§33.4 a）。"""
    for arm, rec in sorted(records.items()):
        for k, v in sorted(rec.items()):
            if not isinstance(v, str):
                continue
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad.lower() in v.lower():
                    raise SystemExit(
                        f"[FATAL] {arm} の {k} に禁止語 '{bad}' が入っている: {v}\n"
                        "        §33.4 a: train は Scene C R0S0 Original のみ。\n"
                        "        prime（Scene C′）は評価専用。jsonl に入れてはならない。")


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


def check_frozen_jsonl(records):
    """生成バイト列が既存成果物と bit 一致するか（回帰）。"""
    bad = []
    for arm in sorted(records):
        b = record_bytes(records[arm])
        got = hashlib.sha256(b).hexdigest()[:16]
        exp_sha, exp_len = FROZEN_JSONL[arm]
        if (got, len(b)) != (exp_sha, exp_len):
            bad.append(f"{arm}: sha {got}/{exp_sha}  bytes {len(b)}/{exp_len}")
    if bad:
        raise SystemExit("[FATAL] 凍結リテラルと不一致（既存 jsonl を壊す変更）:\n  "
                         + "\n  ".join(bad)
                         + "\n        意図した変更なら FROZEN_JSONL を更新し、"
                           "W6 の run との比較可能性が切れることを worklog に記録する。")
    print("[check] 生成バイト列は凍結リテラルと bit 一致（3/3）")
    return True


def check_files_exist(records, base):
    missing = []
    for arm, rec in sorted(records.items()):
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


def content_sha256_b(path):
    """方式B：imageio デコード RGB (T,H,W,3) uint8 の tobytes の sha256。

    検証経路は DiffSynth の LoadVideo と同一（get_reader → count_frames → get_data）。
    """
    import imageio
    import numpy as np
    rd = imageio.get_reader(path)
    n = rd.count_frames()
    frames = [np.asarray(rd.get_data(i), dtype=np.uint8) for i in range(n)]
    rd.close()
    arr = np.stack(frames, axis=0)
    if arr.ndim != 4 or arr.shape[3] != 3:
        raise SystemExit(f"[FATAL] 想定外の shape: {arr.shape}（{path}）")
    return hashlib.sha256(arr.tobytes()).hexdigest(), arr.shape


def check_content_sha(records, base):
    """参照先 mkv の中身を凍結 sha（方式B）と突合する。"""
    names = sorted({records[a][k] for a in records
                    for k in (VIDEO_KEY, VACE_VIDEO_KEY)})
    bad = []
    print("| ファイル | sha(B) 実測 | 期待 | shape | 判定 |")
    print("|---|---|---|---|---|")
    for nm in names:
        p = os.path.join(base, nm)
        exp = FROZEN_SHA_B.get(nm)
        if exp is None:
            bad.append(f"{nm}: 凍結 sha が未登録")
            continue
        got, shape = content_sha256_b(p)
        ok = got[:16] == exp
        print(f"| {nm} | `{got[:16]}` | `{exp}` | {shape} | {'OK' if ok else 'NG'} |")
        if not ok:
            bad.append(f"{nm}: {got[:16]} != {exp}")
        if shape[0] != 49:
            bad.append(f"{nm}: frames {shape[0]} != 49")
    if bad:
        raise SystemExit("[FATAL] content sha 不一致:\n  " + "\n  ".join(bad))
    print(f"[check-sha] {len(names)} 本すべて方式B で凍結値と一致")
    if REFERENCE_IMAGE_SHA_METHOD_UNKNOWN:
        print(f"[注意] {REFERENCE_IMAGE} の content sha 方式は未確定。"
              "manifest.py の file md5 で照合する。")
    return True


def write_jsonl(path, record):
    # UnifiedDataset は 1 行 1 件で読む（unified_dataset.py:79-84）
    with open(path, "wb") as f:
        f.write(record_bytes(record))


def selftest():
    """assert が壊したデータで発火するかを確認する。"""
    print("===== selftest =====")
    cases = []

    def case(name, fn, expect_fire):
        try:
            fn()
            fired = False
        except SystemExit:
            fired = True
        cases.append((name, "検出" if fired else "素通り", fired == expect_fire))

    def fresh():
        return {a: build_record(a) for a in ARM_CONTROL}

    def all_checks(r):
        check_forbidden(r)
        check_cross_arm_identity(r)
        check_frozen_jsonl(r)

    case("正常データ", lambda: all_checks(fresh()), False)

    def c_prompt():
        r = fresh(); r["armB"][PROMPT_KEY] = PROMPT + " "
        all_checks(r)
    case("prompt に空白1文字の差", c_prompt, True)

    def c_dup():
        r = fresh(); r["armC"][VACE_VIDEO_KEY] = ARM_CONTROL["armB"]
        all_checks(r)
    case("vace_video の重複", c_dup, True)

    def c_edit():
        r = fresh(); r["armA"][VIDEO_KEY] = "tv_R0S0_edited.mkv"
        all_checks(r)
    case("Edited の混入（1 arm）", c_edit, True)

    def c_prime_one():
        r = fresh(); r["armA"][VACE_VIDEO_KEY] = "armA_prime.mkv"
        all_checks(r)
    case("prime の混入（1 arm）", c_prime_one, True)

    def c_prime_all():
        # v1 の穴：定数を prime に書き換えると全検査が素通りした
        r = {"armA": {VIDEO_KEY: "tv_R0S0_prime.mkv",
                      VACE_VIDEO_KEY: "armA_prime.mkv",
                      VACE_REF_KEY: "ref_R0S0_prime.png", PROMPT_KEY: PROMPT},
             "armB": {VIDEO_KEY: "tv_R0S0_prime.mkv",
                      VACE_VIDEO_KEY: "armB_prime.mkv",
                      VACE_REF_KEY: "ref_R0S0_prime.png", PROMPT_KEY: PROMPT},
             "armC": {VIDEO_KEY: "tv_R0S0_prime.mkv",
                      VACE_VIDEO_KEY: "armC_prime.mkv",
                      VACE_REF_KEY: "ref_R0S0_prime.png", PROMPT_KEY: PROMPT}}
        all_checks(r)
    case("prime の混入（全 arm。v1 の穴）", c_prime_all, True)

    def c_prompt_all():
        # prompt を 3 arm 同時に変えると cross-arm では捕まらない → 凍結 sha が捕まえる
        global PROMPT
        old = PROMPT
        try:
            PROMPT = old + " extra words"
            r = {a: {VIDEO_KEY: TARGET_VIDEO, VACE_VIDEO_KEY: ARM_CONTROL[a],
                     VACE_REF_KEY: REFERENCE_IMAGE, PROMPT_KEY: PROMPT}
                 for a in ARM_CONTROL}
            all_checks(r)
        finally:
            PROMPT = old
    case("prompt を 3 arm 同時に変更（v1 の穴）", c_prompt_all, True)

    def c_key_order():
        # 列順は sort_keys=True で正規化されるので dict の順序差では発火しない（仕様）
        r = fresh()
        r["armA"] = {PROMPT_KEY: PROMPT, VACE_REF_KEY: REFERENCE_IMAGE,
                     VACE_VIDEO_KEY: ARM_CONTROL["armA"], VIDEO_KEY: TARGET_VIDEO}
        all_checks(r)
    case("dict の挿入順を変える（発火しないのが正）", c_key_order, False)

    print("| ケース | 結果 | 期待どおり |")
    print("|---|---|---|")
    for name, res, ok in cases:
        print(f"| {name} | {res} | {ok} |")
    ok = all(c[2] for c in cases)
    print(f"\n[selftest] {'PASS' if ok else 'FAIL'}  ({sum(c[2] for c in cases)}/{len(cases)})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", help="metadata を書き出すディレクトリ")
    ap.add_argument("--check-files", default=None,
                    help="mkv / png が実在するか確認する base_path")
    ap.add_argument("--check-sha", action="store_true",
                    help="--check-files の base で mkv の content sha（方式B）も照合する")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.outdir:
        ap.error("--outdir が必要（または --selftest）")
    if a.check_sha and not a.check_files:
        ap.error("--check-sha は --check-files と併用する")

    check_prompt_ascii()
    records = {arm: build_record(arm) for arm in sorted(ARM_CONTROL)}
    check_forbidden(records)
    check_cross_arm_identity(records)
    check_frozen_jsonl(records)
    print("[check] prompt は ASCII / 禁止語なし / vace_video 以外は 3 arm 完全一致")

    if a.check_files:
        if not check_files_exist(records, a.check_files):
            raise SystemExit("[FATAL] 参照先が揃っていない。stage_pod_inputs.py を先に流す。")
        if a.check_sha:
            check_content_sha(records, a.check_files)

    os.makedirs(a.outdir, exist_ok=True)
    print()
    print("| arm | ファイル | vace_video | N | bytes |")
    print("|---|---|---|---|---|")
    for arm in sorted(records):
        out = os.path.join(a.outdir, f"metadata_{arm}.jsonl")
        write_jsonl(out, records[arm])
        print(f"| {arm} | `metadata_{arm}.jsonl` | `{records[arm][VACE_VIDEO_KEY]}` | 1 "
              f"| {os.path.getsize(out)} |")

    print()
    print(f"[prompt] {PROMPT!r}")
    print(f"[prompt] 長さ {len(PROMPT)} 文字（3 arm 共通）")
    print()
    print("学習時に渡すフラグ:")
    print(f'  --data_file_keys "{VIDEO_KEY},{VACE_VIDEO_KEY},{VACE_REF_KEY}"')
    print(f'  --extra_inputs "{VACE_VIDEO_KEY},{VACE_REF_KEY}"')
    print()
    print("[note] 学習は original のみ。prime（Scene C′）は推論側で "
          "frame の list として直接渡す（jsonl は作らない）。")


if __name__ == "__main__":
    main()
