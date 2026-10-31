#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
stage_pod_inputs.py — Pod にアップロードする入力を 1 ディレクトリに集約し、
                      中身を凍結 sha で照合してから manifest を書く。（W8, 2026-08-12）

なぜ必要か
  §36.14 の `--dataset_base_path` は **1 本しか渡せない**（os.path.join(base, rel)）。
  だが手元の mkv は 2 箇所に分かれている：
      tv / armA / armB → W3\mkv
      armC             → W4\mkv
  相対パスに `..` を入れて逃げる案は不採用（ToAbsolutePath の解決を跨いで壊れやすく、
  metadata の 3 arm 同一性の検査も濁る）。**集約が正。**

何を集約するか（9 ファイル）
  学習側（4）: tv_R0S0_original.mkv / armA_original.mkv / armB_original.mkv /
               armC_original.mkv  ＋ ref_R0S0_original.png
  評価側（4）: tv_R0S0_prime.mkv / armA_prime.mkv / armB_prime.mkv / armC_prime.mkv
  prime は **学習には使わない**（§33.4 a）。同じディレクトリに置くのは推論時に
  control として読むためであり、metadata jsonl には一切現れない
  （make_metadata_v2.py の禁止語 "prime" が保証する）。

検査
  1. content sha 方式B（rev.12 変更4 で統一）を凍結リテラルと突合
  2. content sha 方式A も併せて計算し、記録が残っている分だけ突合（二重化）
  3. shape が (49, 480, 832, 3) であること
  4. file bytes が凍結値と一致すること
  5. 通ったら manifest TSV（manifest.py と同形式）を書く
     → アップロード後に Pod 側で `python manifest.py verify` を流せば
       転送で 1 byte でも変われば FAIL する

使い方
  python stage_pod_inputs.py --selftest
  python stage_pod_inputs.py ^
      --w3 "C:\Bachelor Thesis\SummerWork\W3\mkv" ^
      --w4 "C:\Bachelor Thesis\SummerWork\W4\mkv" ^
      --out "C:\Bachelor Thesis\SummerWork\W8\pod_upload\mkv" ^
      --manifest "C:\Bachelor Thesis\SummerWork\W8\manifest\pod_upload.tsv"
"""
import argparse
import hashlib
import os
import shutil
import sys

CHUNK = 1 << 20
EXPECT_SHAPE = (49, 480, 832, 3)

# name -> (source_key, sha_B_16, sha_A_16 or None, bytes or None)
# 出所：rev.12 変更5（方式A/B 併記）、rev.10 §36.5（original の方式A / FFV1 bytes）
ASSETS = {
    # 学習側（original）
    "tv_R0S0_original.mkv": ("w3", "6190557123fdc2b3", "bba8f4b9e280b9e3", 3426110),
    "armA_original.mkv":    ("w3", "33cabc1b11b72f45", "246ed61a47ad39cd", 2217662),
    "armB_original.mkv":    ("w3", "d31186af848c80dc", "ab5eb47f98a1bd74", 1874392),
    # 注意：rev.12 変更10 の配置規則（Arm C のみ W4\mkv）は W7 制定であり、
    # armC_original は W6 以前に生成されたため規則の外にある。実際の場所は W3\mkv。
    "armC_original.mkv":    ("w3", "7c08f981503edb34", "46abbcded632a61b", 1088684),
    # 評価側（prime = Scene C′）
    "tv_R0S0_prime.mkv":    ("w3", "71d0253f9fdc3161", "8fa4ae4bb827a020", 3466263),
    "armA_prime.mkv":       ("w3", "3666302824d68ee5", "1f273ee25c8b73c8", 2224461),
    "armB_prime.mkv":       ("w3", "5a08e10cc228d230", "77f1cf8b2e21b5f1", 1914588),
    "armC_prime.mkv":       ("w4", "fde80efbe48dcb51", "a7ac42f40f26b82d", 1066438),
}

# 参照画像。content sha の方式が未確定（rev.10 §36.5 に先頭 24 桁のみ）。
# → sha 照合はせず、存在確認と file md5（manifest）だけを行う。未決事項。
REFERENCE_PNG = "ref_R0S0_original.png"
REFERENCE_PNG_SRC = "w3"
REFERENCE_PNG_SHA_PREFIX_24 = "df7c595457c81e35b47164ae"   # 方式不明


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def decode_rgb(path):
    """DiffSynth の LoadVideo と同一経路でデコードして (T,H,W,3) uint8 を返す。"""
    import imageio
    import numpy as np
    rd = imageio.get_reader(path)
    n = rd.count_frames()
    frames = [np.asarray(rd.get_data(i), dtype=np.uint8) for i in range(n)]
    rd.close()
    return np.stack(frames, axis=0)


def content_sha(path):
    """(方式B, 方式A, shape) を返す。B = sha256(tobytes) / A = sha256(str(shape)+tobytes)。"""
    arr = decode_rgb(path)
    b = arr.tobytes()
    sha_b = hashlib.sha256(b).hexdigest()
    sha_a = hashlib.sha256(str(arr.shape).encode() + b).hexdigest()
    return sha_b, sha_a, arr.shape


def verify_one(path, spec):
    """1 本を検査して (ok, rows) を返す。"""
    src_key, exp_b, exp_a, exp_bytes = spec
    problems = []
    size = os.path.getsize(path)
    if exp_bytes is not None and size != exp_bytes:
        problems.append(f"bytes {size} != {exp_bytes}")
    sha_b, sha_a, shape = content_sha(path)
    if sha_b[:16] != exp_b:
        problems.append(f"sha(B) {sha_b[:16]} != {exp_b}")
    if exp_a is not None and sha_a[:16] != exp_a:
        problems.append(f"sha(A) {sha_a[:16]} != {exp_a}")
    if tuple(shape) != EXPECT_SHAPE:
        problems.append(f"shape {tuple(shape)} != {EXPECT_SHAPE}")
    return problems, sha_b[:16], sha_a[:16], shape, size


def cmd_stage(a):
    srcdirs = {"w3": a.w3, "w4": a.w4}
    for k, d in srcdirs.items():
        if not os.path.isdir(d):
            raise SystemExit(f"[FATAL] --{k} が存在しない: {d}")

    print("===== 1. 存在確認 =====")
    plan = []
    missing = []
    for name, spec in sorted(ASSETS.items()):
        src = os.path.join(srcdirs[spec[0]], name)
        (plan if os.path.isfile(src) else missing).append((name, src, spec))
    ref_src = os.path.join(srcdirs[REFERENCE_PNG_SRC], REFERENCE_PNG)
    if not os.path.isfile(ref_src):
        missing.append((REFERENCE_PNG, ref_src, None))
    if missing:
        print("[FATAL] 見つからない:")
        for name, src, _ in missing:
            print(f"    {name} -> {src}")
        raise SystemExit(1)
    print(f"9 件すべて存在（mkv 8 + png 1）")

    print()
    print("===== 2. content sha 照合（方式B 主 / 方式A 二重化） =====")
    print("| ファイル | sha(B) | sha(A) | shape | bytes | 判定 |")
    print("|---|---|---|---|---|---|")
    fail = []
    for name, src, spec in plan:
        problems, sb, sa, shape, size = verify_one(src, spec)
        print(f"| {name} | `{sb}` | `{sa}` | {tuple(shape)} | {size:,} | "
              f"{'OK' if not problems else 'NG'} |")
        if problems:
            fail.append((name, problems))
    if fail:
        print()
        for name, problems in fail:
            print(f"[FATAL] {name}: " + " / ".join(problems))
        raise SystemExit("凍結値と不一致。集約を中止した（コピーしていない）。")
    print(f"[OK] 8 本すべて凍結値と一致（方式B / 方式A の両方）")
    print(f"[note] {REFERENCE_PNG} は content sha の方式が未確定"
          f"（記録は先頭 24 桁 {REFERENCE_PNG_SHA_PREFIX_24} のみ）。"
          "file md5 でのみ照合する。")

    print()
    print("===== 3. 集約 =====")
    os.makedirs(a.out, exist_ok=True)
    copied = []
    for name, src, _ in plan + [(REFERENCE_PNG, ref_src, None)]:
        dst = os.path.join(a.out, name)
        shutil.copy2(src, dst)
        if md5_of(src) != md5_of(dst):
            raise SystemExit(f"[FATAL] コピーで壊れた: {name}")
        copied.append((name, dst))
    print(f"{len(copied)} 件を {a.out} に集約（コピー後の md5 一致を確認）")

    print()
    print("===== 4. manifest =====")
    rows = []
    total = 0
    for name, dst in sorted(copied):
        sz = os.path.getsize(dst)
        total += sz
        rows.append((md5_of(dst), sz, name))
    outdir = os.path.dirname(os.path.abspath(a.manifest))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(a.manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write("# manifest.py v1\n")
        f.write("# generated_by: stage_pod_inputs.py\n")
        f.write("# root_basename: %s\n" % os.path.basename(os.path.abspath(a.out)))
        f.write("# files: %d\n" % len(rows))
        f.write("# bytes: %d\n" % total)
        f.write("# columns: md5\tsize\trelpath\n")
        for md5, sz, name in rows:
            f.write("%s\t%d\t%s\n" % (md5, sz, name))
    print(f"書き出し: {a.manifest}（{len(rows)} 件 / {total/1048576:.1f} MB）")

    print()
    print("===== 次の手 =====")
    print(f"  1) {a.out} を Pod の /workspace/w4/mkv に転送")
    print( "  2) Pod 側: python manifest.py verify --root /workspace/w4/mkv "
           "--manifest pod_upload.tsv")
    print( "  3) Pod 側: python make_metadata_v2.py --outdir /workspace/w4/meta "
           "--check-files /workspace/w4/mkv --check-sha")
    print( "  ※ 2 は転送事故の検出、3 は列名と中身の最終ゲート。両方 PASS で準備完了。")
    return 0


def selftest():
    """合成 FFV1 で sha 方式の弁別と assert の発火を確認する。"""
    import tempfile
    import numpy as np
    import imageio
    print("===== selftest =====")
    tmp = tempfile.mkdtemp()
    T, H, W = 5, 16, 24

    def write_ffv1(path, arr):
        wr = imageio.get_writer(path, codec="ffv1", macro_block_size=1,
                               output_params=["-level", "3", "-pix_fmt", "gbrp",
                                              "-g", "1", "-bitexact"])
        for i in range(arr.shape[0]):
            wr.append_data(arr[i])
        wr.close()

    rng = np.random.default_rng(0)
    a0 = rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8)
    p0 = os.path.join(tmp, "base.mkv")
    write_ffv1(p0, a0)

    b0, a0h, sh0 = content_sha(p0)

    # 合成データは 49 frames ではないので、shape 検査だけ一時的に合成側へ向ける。
    # （こうしないと「正しい凍結値」ケースが shape で常に発火して他の検査を隠す）
    global EXPECT_SHAPE
    real_shape = EXPECT_SHAPE
    EXPECT_SHAPE = tuple(sh0)

    rows = []
    rows.append(("FFV1 は往復 bit 一致", np.array_equal(decode_rgb(p0), a0), True))
    rows.append(("方式A と方式B は別値", a0h != b0, True))

    # 1 LSB だけ変えた別ファイル
    a1 = a0.copy(); a1[2, 3, 4, 1] = (int(a1[2, 3, 4, 1]) + 1) % 256
    p1 = os.path.join(tmp, "lsb.mkv")
    write_ffv1(p1, a1)
    b1, _, _ = content_sha(p1)
    rows.append(("1 LSB 差で sha(B) が変わる", b1 != b0, True))

    # 同一内容の再エンコードで sha(B) が一致（内容依存であること）
    p2 = os.path.join(tmp, "again.mkv")
    write_ffv1(p2, a0)
    b2, _, _ = content_sha(p2)
    rows.append(("同一内容の再エンコードで sha(B) 一致", b2 == b0, True))

    # verify_one が壊れた凍結値で発火するか
    ok_spec = ("w3", b0[:16], a0h[:16], os.path.getsize(p0))
    rows.append(("正しい凍結値 -> 問題 0", verify_one(p0, ok_spec)[0] == [], True))
    bad_sha = ("w3", "0" * 16, a0h[:16], os.path.getsize(p0))
    rows.append(("sha(B) を壊す -> 発火", verify_one(p0, bad_sha)[0] != [], True))
    bad_a = ("w3", b0[:16], "0" * 16, os.path.getsize(p0))
    rows.append(("sha(A) を壊す -> 発火", verify_one(p0, bad_a)[0] != [], True))
    bad_bytes = ("w3", b0[:16], a0h[:16], 12345)
    rows.append(("bytes を壊す -> 発火", verify_one(p0, bad_bytes)[0] != [], True))
    EXPECT_SHAPE = (T + 1, H, W, 3)     # frame 数を 1 だけずらす
    fired = any("shape" in x for x in verify_one(p0, ok_spec)[0])
    rows.append(("frame 数を 1 ずらす -> shape 検査が発火", fired, True))
    EXPECT_SHAPE = tuple(sh0)

    print("| ケース | 結果 | 期待どおり |")
    print("|---|---|---|")
    allok = True
    for name, got, exp in rows:
        allok &= (bool(got) == exp)
        print(f"| {name} | {bool(got)} | {bool(got) == exp} |")
    EXPECT_SHAPE = real_shape
    print(f"\n[note] 合成 {tuple(sh0)} で検査した。本番の EXPECT_SHAPE は {EXPECT_SHAPE}。")
    print(f"[selftest] {'PASS' if allok else 'FAIL'}")
    shutil.rmtree(tmp, ignore_errors=True)
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w3", help=r"tv / armA / armB の mkv があるディレクトリ（W3\mkv）")
    ap.add_argument("--w4", help=r"armC の mkv があるディレクトリ（W4\mkv）")
    ap.add_argument("--out", help="集約先（Pod の /workspace/w4/mkv に対応）")
    ap.add_argument("--manifest", default="pod_upload.tsv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    for k in ("w3", "w4", "out"):
        if not getattr(a, k):
            ap.error(f"--{k} が必要（または --selftest）")
    sys.exit(cmd_stage(a))


if __name__ == "__main__":
    main()
