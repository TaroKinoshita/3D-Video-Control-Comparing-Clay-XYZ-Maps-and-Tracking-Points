"""
test_score_armc.py  —  score_armc.py の単体テスト。GPU / torch / Blender 不要。

確認すること
  1. 恒等（VAE を通さない）なら色誤差が厳密に 0
  2. 既知の摂動を入れると色誤差が閉形式と一致する
  3. 描画の再現が armC PNG と bit 一致する（png_replay）
  4. 中心を他点に奪われた点を検出する（no_buried が FAIL する）
  5. Target 可視点が 0 のフレームで落ちない
  6. Entity 帯の導出が protocol §13.5 の 30/98/157/225 と一致する
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import png_io

ROOT = "/tmp/sctest"
W, H, R, T = 200, 120, 6, 49

NT = [[30, 64, 98], [98, 30, 64], [64, 98, 30], [30, 98, 64]]
TG = [[157, 191, 225], [225, 157, 191], [191, 225, 157], [157, 225, 191]]

npass = nfail = 0


def ck(cond, label):
    global npass, nfail
    print(("PASS  " if cond else "FAIL  ") + label)
    if cond:
        npass += 1
    else:
        nfail += 1


def disc(r):
    return np.array([(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                     if dx * dx + dy * dy <= r * r], dtype=np.int32)


OFF = disc(R)


def build_points(bury=False, occlude_target=True):
    pts, U, V = [], [], []
    pid = 0
    for k, c in enumerate(TG):                        # target 4点
        pts.append({"id": pid, "object": "Target", "pass_index": 3,
                    "band": "target", "rgb": c, "is_target": True})
        U.append(30.0 + 40 * k)
        V.append(30.0)
        pid += 1
    for k, c in enumerate(NT):                        # background 4点
        pts.append({"id": pid, "object": "Wall", "pass_index": 1,
                    "band": "nontarget", "rgb": c, "is_target": False})
        U.append(30.0 + 40 * k)
        V.append(80.0)
        pid += 1
    if bury:
        # background 点を既存点の 3px 隣に置く → 中心が奪われる
        pts.append({"id": pid, "object": "Wall", "pass_index": 1,
                    "band": "nontarget", "rgb": NT[1], "is_target": False})
        U.append(33.0)
        V.append(80.0)
        pid += 1
    n = pid
    frames = {}
    for variant in ("original", "edited"):
        rows = []
        for t in range(T):
            vis = [True] * n
            if occlude_target and 36 <= t <= 40:      # Target 完全遮蔽の再現
                for i in range(len(TG)):
                    vis[i] = False
            rows.append({"index": t, "u": list(U), "v": list(V),
                         "visible": vis, "nv": [0.5] * n,
                         "depth": [9.0 + 0.01 * i for i in range(n)]})
        frames[variant] = rows
    prm = {"pitch": 0.1, "radius": R, "assign_radius": 24.0,
           "n_target": len(TG), "n_background": n - len(TG),
           "d_floor": 13.42, "d_floor_no_overlap": 13.4143,
           "width": W, "height": H,
           "achieved_separation": {"target": 40.0, "background": 40.0,
                                   "global": 40.0}}
    return {"points": pts, "frames": frames, "params": prm}


def render(D):
    """score_armc の replay と同じ規約で PNG を書く（独立実装）。"""
    rgb = np.array([p["rgb"] for p in D["points"]], dtype=np.uint8)
    for variant, rows in D["frames"].items():
        d = os.path.join(ROOT, "armC", variant)
        os.makedirs(d, exist_ok=True)
        for row in rows:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            drawn = []
            for i in range(len(rgb)):
                if not row["visible"][i]:
                    continue
                ci = int(round(row["u"][i] - 0.5))
                cj = int(round(row["v"][i] - 0.5))
                if 0 <= ci < W and 0 <= cj < H:
                    drawn.append((row["depth"][i], i, ci, cj))
            drawn.sort(key=lambda x: (-x[0], x[1]))
            for _dp, i, ci, cj in drawn:
                xs, ys = OFF[:, 0] + ci, OFF[:, 1] + cj
                m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                canvas[ys[m], xs[m]] = rgb[i]
            png_io.write_rgb(os.path.join(d, "armC_%04d.png" % row["index"]),
                             canvas, W, H)


def make_npy(delta_r=0):
    os.makedirs(os.path.join(ROOT, "rt"), exist_ok=True)
    for variant in ("original", "edited"):
        frames = []
        for t in range(T):
            px, w, h = png_io.read_rgb(
                os.path.join(ROOT, "armC", variant, "armC_%04d.png" % t))
            a = np.frombuffer(px, dtype=np.uint8).reshape(h, w, 3).astype(np.int16)
            if delta_r:
                a[:, :, 0] = np.clip(a[:, :, 0] + delta_r, 0, 255)
            frames.append(a.astype(np.uint8))
        np.save(os.path.join(ROOT, "rt", "%s_temporal.npy" % variant),
                np.stack(frames))


def run(extra=()):
    cmd = [sys.executable, os.path.join(HERE, "score_armc.py"),
           "--points", os.path.join(ROOT, "armc_points.json"),
           "--rt", os.path.join(ROOT, "rt"),
           "--armc", os.path.join(ROOT, "armC"),
           "--pngio", HERE,
           "--out", os.path.join(ROOT, "score.json")] + list(extra)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    with open(os.path.join(ROOT, "score.json"), encoding="utf-8") as f:
        rep = json.load(f)
    return p.stdout + p.stderr, rep


def setup(D, delta_r=0):
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT)
    with open(os.path.join(ROOT, "armc_points.json"), "w", encoding="utf-8") as f:
        json.dump(D, f)
    render(D)
    make_npy(delta_r)


def chk(rep, name):
    return {c["name"]: c["ok"] for c in rep["checks"]}.get(name)


# ---- 1. 恒等 ----------------------------------------------------------
setup(build_points())
out, rep = run()
ck(chk(rep, "entity_bands") is True, "entity bands = 30/98/157/225")
ck(chk(rep, "disc_area") is True, "disc area 113")
ck(chk(rep, "png_replay_original_temporal") is True, "replay == PNG (original)")
ck(chk(rep, "png_replay_edited_temporal") is True, "replay == PNG (edited)")
ck(chk(rep, "no_buried_original_temporal") is True, "no buried centers")
r0 = [r for r in rep["results"] if r["variant"] == "original"][0]
ck(r0["overall"]["color_err"] == 0.0,
   "identity roundtrip -> color_err == 0 (got %s)" % r0["overall"]["color_err"])
ck(r0["overall"]["identity"] == 1.0,
   "identity == 100%% (got %s)" % r0["overall"]["identity"])
ck(r0["overall"]["entity_accuracy"] == 1.0, "entity accuracy == 100%")
ck(rep["verdict"]["original"]["pass"] is True, "verdict pass at err 0")

# Target 可視 0 のフレームで n_tgt=0 かつ落ちない
z = [x for x in r0["per_frame"] if 36 <= x["t"] <= 40]
ck(len(z) == 5 and all(x["n_tgt"] == 0 for x in z),
   "frames 36-40 have zero visible target points")
ck(all(x["color_err_tgt"] is None for x in z),
   "target-band metric is None when no target visible")

# ---- 2. 既知の摂動 ----------------------------------------------------
setup(build_points(), delta_r=5)
out, rep = run()
r0 = [r for r in rep["results"] if r["variant"] == "original"][0]
exp = round(5.0 / 3.0, 3)
ck(abs(r0["overall"]["color_err"] - exp) < 1e-9,
   "+5 on R channel -> color_err == %.3f (got %s)" % (exp, r0["overall"]["color_err"]))
ck(r0["overall"]["identity"] == 1.0, "identity survives a 1.67 shift")
ck(rep["verdict"]["original"]["pass"] is True, "1.67 passes threshold 24")

setup(build_points(), delta_r=100)
out, rep = run()
r0 = [r for r in rep["results"] if r["variant"] == "original"][0]
# target 帯は 157..225 なので +100 で 255 にクリップされる。閉形式で出す
gt = np.array(TG + NT, dtype=np.float64)
per = np.abs(np.stack([np.clip(gt[:, 0] + 100, 0, 255), gt[:, 1], gt[:, 2]], 1)
             - gt).mean(axis=1)
exp_t = round(float(per[:4].mean()), 3)
ck(abs(r0["target"]["color_err"] - exp_t) < 1e-9,
   "+100 clipped -> target color_err == %.3f (got %s)"
   % (exp_t, r0["target"]["color_err"]))
ck(rep["verdict"]["original"]["pass"] is False, "large shift fails threshold")

# ---- 3. 中心の上書き --------------------------------------------------
setup(build_points(bury=True))
out, rep = run()
ck(chk(rep, "no_buried_original_temporal") is False,
   "overlapping point fires no_buried")
ck("上書きされた中心 0" not in out, "buried count is nonzero")
ck(chk(rep, "png_replay_original_temporal") is True,
   "replay still matches PNG when points overlap")

print("\nPASS %d / FAIL %d" % (npass, nfail))
sys.exit(1 if nfail else 0)
