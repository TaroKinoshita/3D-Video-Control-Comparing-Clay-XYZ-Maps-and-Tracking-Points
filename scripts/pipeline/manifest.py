r"""
manifest.py  —  再生成物の md5 マニフェストを作る / 検証する

目的
  PNG / EXR は git に入れない（bit 一致で再生成できるので派生物）。
  代わりに md5 の一覧をコミットして、「再生成できた」を後から検証可能にする。

方針
  - stdlib のみ
  - パス区切りは **常にスラッシュに正規化**（Windows と Linux で同じ結果になる。
    Compress-Archive でバックスラッシュを踏んだのと同じ罠を避ける）
  - ファイルは **相対パスでソート**（走査順に依存しない）
  - `open()` は必ず encoding="utf-8"（作法 #23）

使い方
  # 作る
  python manifest.py write --root "C:\Bachelor Thesis\SummerWork\W2\RENDER" --out manifest/W2_RENDER.tsv

  # 検証する（再生成後に流す）
  python manifest.py verify --root "C:\Bachelor Thesis\SummerWork\W2\RENDER" --manifest manifest/W2_RENDER.tsv

出力形式（TSV。1行 = 1ファイル）
  md5<TAB>size<TAB>relative/path/with/slashes
先頭に `#` のコメント行でメタ情報を書く。
"""
import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone

CHUNK = 1 << 20


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def walk(root, skip_dirs):
    """(relpath_with_slashes, abspath) を相対パス昇順で返す。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for fn in sorted(filenames):
            ap = os.path.join(dirpath, fn)
            rp = os.path.relpath(ap, root).replace(os.sep, "/")
            out.append((rp, ap))
    out.sort(key=lambda t: t[0])
    return out


def cmd_write(a):
    skip = set(a.skip)
    rows = walk(a.root, skip)
    if not rows:
        print("ファイルが見つからない: %s" % a.root)
        return 1

    total = 0
    lines = []
    by_top = {}
    for rp, ap in rows:
        sz = os.path.getsize(ap)
        total += sz
        lines.append("%s\t%d\t%s" % (md5_of(ap), sz, rp))
        top = rp.split("/")[0] if "/" in rp else "."
        d = by_top.setdefault(top, [0, 0])
        d[0] += 1
        d[1] += sz

    outdir = os.path.dirname(os.path.abspath(a.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# manifest.py v1\n")
        f.write("# generated: %s\n" % datetime.now(timezone.utc).isoformat(timespec="seconds"))
        f.write("# root_basename: %s\n" % os.path.basename(os.path.abspath(a.root)))
        f.write("# files: %d\n" % len(rows))
        f.write("# bytes: %d\n" % total)
        f.write("# columns: md5\tsize\trelpath\n")
        for ln in lines:
            f.write(ln + "\n")

    print("%-28s %8s %12s" % ("top-level", "files", "MB"))
    print("-" * 50)
    for k in sorted(by_top):
        n, b = by_top[k]
        print("%-28s %8d %12.1f" % (k, n, b / 1048576))
    print("-" * 50)
    print("%-28s %8d %12.1f" % ("TOTAL", len(rows), total / 1048576))
    print("\n書き出し: %s" % a.out)
    return 0


def read_manifest(path):
    rec = {}
    meta = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("#"):
                if ":" in ln:
                    k, v = ln[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) != 3:
                raise ValueError("行の形式が不正: %r" % ln)
            rec[parts[2]] = (parts[0], int(parts[1]))
    return rec, meta


def cmd_verify(a):
    want, meta = read_manifest(a.manifest)
    skip = set(a.skip)
    have = {rp: ap for rp, ap in walk(a.root, skip)}

    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    mismatch = []
    for rp in sorted(set(want) & set(have)):
        exp_md5, exp_sz = want[rp]
        ap = have[rp]
        sz = os.path.getsize(ap)
        if sz != exp_sz:
            mismatch.append((rp, "size %d != %d" % (sz, exp_sz)))
            continue
        got = md5_of(ap)
        if got != exp_md5:
            mismatch.append((rp, "md5 %s != %s" % (got[:12], exp_md5[:12])))

    print("manifest : %s" % a.manifest)
    if "generated" in meta:
        print("generated: %s" % meta["generated"])
    print("expected : %d files" % len(want))
    print("found    : %d files" % len(have))
    print()
    ok = True
    for label, items in (("MISSING", missing), ("EXTRA", extra)):
        print("%-10s %d" % (label, len(items)))
        for x in items[:10]:
            print("    %s" % x)
        if len(items) > 10:
            print("    ... 他 %d 件" % (len(items) - 10))
        if items and label == "MISSING":
            ok = False
    print("%-10s %d" % ("MISMATCH", len(mismatch)))
    for rp, why in mismatch[:10]:
        print("    %s  (%s)" % (rp, why))
    if len(mismatch) > 10:
        print("    ... 他 %d 件" % (len(mismatch) - 10))
    if mismatch:
        ok = False

    print()
    print("=" * 50)
    print("結果: %s" % ("PASS（bit 一致で再生成できている）" if ok else "FAIL"))
    print("=" * 50)
    if extra and ok:
        print("※ EXTRA はマニフェスト作成後に追加されたファイル。判定には含めない。")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write")
    w.add_argument("--root", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--skip", nargs="*",
                   default=["__pycache__", ".ipynb_checkpoints", ".git"])
    w.set_defaults(func=cmd_write)

    v = sub.add_parser("verify")
    v.add_argument("--root", required=True)
    v.add_argument("--manifest", required=True)
    v.add_argument("--skip", nargs="*",
                   default=["__pycache__", ".ipynb_checkpoints", ".git"])
    v.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
