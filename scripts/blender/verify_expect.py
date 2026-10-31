"""
scene_c_expect.py のリテラルを閉形式で検算する。

Blender 不要。RunPod でも走る。
これはリテラルの「打ち間違い」を検出するだけ。
規約そのものの検証は Blender 側の SELF-CHECK が担う。
"""
import sys
import scene_c_expect as E
import scene_c_spec as S

K = S.K_PX_PER_UNIT
fail = 0
npass = 0


def proj(x, y, z):
    d = y + 10.0
    return 416.0 + x * K / d, 240.0 - z * K / d, d


print("=" * 68)
print("verify_expect : scene_c_expect.py のリテラル検算")
print("=" * 68)

for label, x, y, z, eu, ev, ed in E.POINTS:
    u, v, d = proj(x, y, z)
    ok = (abs(u - eu) <= E.TOL_PX and abs(v - ev) <= E.TOL_PX
          and abs(d - ed) <= E.TOL_DEPTH)
    if ok:
        npass += 1
    else:
        fail += 1
        print("FAIL %-16s calc u=%.4f v=%.4f d=%.4f | lit u=%.4f v=%.4f d=%.4f"
              % (label, u, v, d, eu, ev, ed))

for d in (8.60, 9.00, 9.40):
    key = "px_per_unit_at_%.2f" % d
    calc = K / d
    if abs(calc - E.SCALARS[key]) <= 1e-3:
        npass += 1
    else:
        fail += 1
        print("FAIL %-16s calc %.4f | lit %.4f" % (key, calc, E.SCALARS[key]))

checks = [
    ("target_v_top", proj(0, -0.60, -0.10)[1]),
    ("target_v_bottom", proj(0, -1.40, -1.50)[1]),
    ("floor_first_visible_depth", abs(S.OBJECTS[1][3] + 0.05) * K / 240.0),
]
for key, calc in checks:
    if abs(calc - E.SCALARS[key]) <= 1e-3:
        npass += 1
    else:
        fail += 1
        print("FAIL %-26s calc %.4f | lit %.4f" % (key, calc, E.SCALARS[key]))

# spec と expect の整合（別ファイルなので食い違いを検出する価値がある）
for name, pidx, cs, z0, z1, _c, _r in S.OBJECTS:
    if E.PASS_INDEX.get(name) != pidx:
        fail += 1
        print("FAIL pass_index mismatch %s spec=%s expect=%s"
              % (name, pidx, E.PASS_INDEX.get(name)))
    else:
        npass += 1
    nv, nf = len(cs) * 2, len(cs) + 2
    if E.EXPECT_TOPOLOGY.get(name) != (nv, nf):
        fail += 1
        print("FAIL topology %s spec=(%d,%d) expect=%s"
              % (name, nv, nf, E.EXPECT_TOPOLOGY.get(name)))
    else:
        npass += 1

# ---------------------------------------------------------------- spec ↔ expect
# 破壊テスト3 で判明した穴：spec の寸法を変えても expect のリテラルが
# 素通しで通ってしまう。両者の幾何的整合をここで縛る。

def obj_verts(name):
    for n, _p, cs, z0, z1, _c, _r in S.OBJECTS:
        if n == name:
            off = S.TARGET_X_START if n == "Target" else 0.0
            return {(round(x + off, 6), round(y, 6), round(z, 6))
                    for x, y in cs for z in (z0, z1)}
    return None


def bbox_centre(name):
    vs = obj_verts(name)
    xs = [v[0] for v in vs]; ys = [v[1] for v in vs]; zs = [v[2] for v in vs]
    return (round((min(xs) + max(xs)) / 2, 6),
            round((min(ys) + max(ys)) / 2, 6),
            round((min(zs) + max(zs)) / 2, 6))


lit = {}
for label, x, y, z, _u, _v, _d in E.POINTS:
    lit[label] = (round(x, 6), round(y, 6), round(z, 6))

# 1. Target の頂点集合が完全一致すること
tgt_lit = {v for k, v in lit.items() if k.startswith("Target_P")}
tgt_spec = obj_verts("Target")
if tgt_lit == tgt_spec:
    npass += 1
else:
    fail += 1
    print("FAIL Target vertex set mismatch")
    print("     only in expect : %s" % sorted(tgt_lit - tgt_spec))
    print("     only in spec   : %s" % sorted(tgt_spec - tgt_lit))

# 2. 頂点リテラルが spec の頂点集合に含まれること
for prefix, oname in (("Occ_", "Occluder"), ("Deep_", "DeepObject"),
                      ("Wall_corner", "Wall")):
    vs = obj_verts(oname)
    for k, v in lit.items():
        if k.startswith(prefix):
            if v in vs:
                npass += 1
            else:
                fail += 1
                print("FAIL %s %s not a vertex of %s" % (k, v, oname))

# 3. 中心リテラルが bbox 中心と一致すること
for k, oname in (("Target_centre", "Target"), ("Marker_centre", "Marker")):
    if lit[k] == bbox_centre(oname):
        npass += 1
    else:
        fail += 1
        print("FAIL %s %s != bbox centre %s" % (k, lit[k], bbox_centre(oname)))

# 4. 正規化ボックスが全ジオメトリを含むこと
allv = [v for n, _p, _cs, _a, _b, _c, _r in S.OBJECTS for v in obj_verts(n)]
for ax, i in (("x", 0), ("y", 1), ("z", 2)):
    lo, hi = S.NORM_BOX[ax]
    vals = [v[i] for v in allv]
    if lo - 1e-9 <= min(vals) and max(vals) <= hi + 1e-9:
        npass += 1
    else:
        fail += 1
        print("FAIL norm_box %s: geometry [%.2f,%.2f] vs box [%.2f,%.2f]"
              % (ax, min(vals), max(vals), lo, hi))

# 5. 遮蔽板が index 37-43 の Target シルエットを覆うこと
def silhouette_u(idx):
    xc = S.TARGET_X_START + (S.TARGET_X_END - S.TARGET_X_START) * idx / 48.0
    us = [proj(xc + x, y, 0.0)[0] for x, y in S.TARGET_CS]
    return min(us), max(us)


occ = obj_verts("Occluder")
occ_u = [proj(v[0], v[1], 0.0)[0] for v in occ]
o_lo, o_hi = min(occ_u), max(occ_u)
i0, i1 = S.OCCLUSION_INDEX
s_lo = min(silhouette_u(i)[0] for i in range(i0, i1 + 1))
s_hi = max(silhouette_u(i)[1] for i in range(i0, i1 + 1))
if o_lo <= s_lo and s_hi <= o_hi:
    npass += 1
    print("occluder covers index %d-%d : target u [%.2f,%.2f] within occ u [%.2f,%.2f]"
          % (i0, i1, s_lo, s_hi, o_lo, o_hi))
else:
    fail += 1
    print("FAIL occluder does not cover index %d-%d : target u [%.2f,%.2f] vs occ [%.2f,%.2f]"
          % (i0, i1, s_lo, s_hi, o_lo, o_hi))

# 6. 深い物体が Target の掃く帯の外にあること
sweep_lo = min(silhouette_u(i)[0] for i in range(0, 49))
sweep_hi = max(silhouette_u(i)[1] for i in range(0, 49))
deep_u = [proj(v[0], v[1], 0.0)[0] for v in obj_verts("DeepObject")]
if max(deep_u) < sweep_lo or min(deep_u) > sweep_hi:
    npass += 1
    print("deep object outside sweep : deep u [%.2f,%.2f] / sweep u [%.2f,%.2f]"
          % (min(deep_u), max(deep_u), sweep_lo, sweep_hi))
else:
    fail += 1
    print("FAIL deep object inside sweep band : deep u [%.2f,%.2f] / sweep [%.2f,%.2f]"
          % (min(deep_u), max(deep_u), sweep_lo, sweep_hi))

# Target の速度制約（Original / catch-up）
dx = S.TARGET_X_END - S.TARGET_X_START
d_front = 10.0 + min(y for _x, y in S.TARGET_CS)
v_orig = dx / 48.0 * K / d_front
print()
print("Target dX               : %.4f unit" % dx)
print("front face depth        : %.4f" % d_front)
print("Original velocity       : %.4f px/frame  (<= 6 ?  %s)"
      % (v_orig, "OK" if v_orig <= 6.0 else "OVER"))
# catch-up の実速度は軌道テーブルから出す（下の「軌道の検算」節）。
# rev.1 にあった「Original の2倍 = 10.0775」は概算で、実際は 9.3023。
if v_orig > 6.0:
    fail += 1

# ---------------------------------------------------------------- 軌道（Step 2）
print()
print("-" * 68)
print("軌道の検算")
print("-" * 68)

# spec のキーフレームと expect のキーフレームが一致すること
for v in S.VARIANTS:
    if S.KEYFRAMES[v] == E.KEYFRAMES[v]:
        npass += 1
    else:
        fail += 1
        print("FAIL keyframes %s spec=%s expect=%s"
              % (v, S.KEYFRAMES[v], E.KEYFRAMES[v]))


def eval_keyframes(kfs, frame):
    """LINEAR 補間で frame の X を求める。Blender の fcurve と独立な経路。"""
    kfs = sorted(kfs)
    if frame <= kfs[0][0]:
        return kfs[0][1]
    if frame >= kfs[-1][0]:
        return kfs[-1][1]
    for (f0, x0), (f1, x1) in zip(kfs, kfs[1:]):
        if f0 <= frame <= f1:
            if f1 == f0:
                return x1
            return x0 + (x1 - x0) * (frame - f0) / (f1 - f0)
    return kfs[-1][1]


for v, table in (("original", E.TRAJ_ORIGINAL), ("edited", E.TRAJ_EDITED)):
    nbad = 0
    for idx, frm, ex, eu in table:
        if frm != idx + 1:
            nbad += 1
            print("FAIL %s index %d: frame %d != index+1" % (v, idx, frm))
            continue
        x = eval_keyframes(S.KEYFRAMES[v], frm)
        u = proj(x, -1.00, 0.0)[0]
        if abs(x - ex) > E.TOL_X or abs(u - eu) > E.TOL_U:
            nbad += 1
            print("FAIL %s index %2d: calc X=%+.7f u=%.4f | lit X=%+.7f u=%.4f"
                  % (v, idx, x, u, ex, eu))
    if nbad == 0:
        npass += 1
        print("%-10s 49 frames OK" % v)
    else:
        fail += 1

# 速度制約（最近隅 = 前面 depth 8.60）
for v, table in (("original", E.TRAJ_ORIGINAL), ("edited", E.TRAJ_EDITED)):
    xs = [r[2] for r in table]
    vmax = max(abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)) * K / d_front
    key = v + "_max"
    ok = (abs(vmax - E.VELOCITY[key]) <= 0.001
          and vmax <= E.VELOCITY["design_limit"])
    if ok:
        npass += 1
        print("%-10s max velocity %.4f px/frame (limit %.1f) OK"
              % (v, vmax, E.VELOCITY["design_limit"]))
    else:
        fail += 1
        print("FAIL %s velocity calc %.4f | lit %.4f | limit %.1f"
              % (v, vmax, E.VELOCITY[key], E.VELOCITY["design_limit"]))

# t_sync 以降で2変種が一致すること
ts = S.EDIT["t_sync"]
diff = [(r[0], abs(r[2] - o[2]))
        for r, o in zip(E.TRAJ_EDITED, E.TRAJ_ORIGINAL) if r[0] >= ts]
if all(d <= 1e-9 for _i, d in diff):
    npass += 1
    print("t_sync=%d 以降で2変種一致 OK (index %d-48)" % (ts, ts))
else:
    fail += 1
    print("FAIL t_sync 以降で不一致: %s" % [(i, d) for i, d in diff if d > 1e-9])

# 遮蔽区間で2変種が一致すること（Edited が早着していない）
i0, i1 = S.OCCLUSION_INDEX
occ_same = all(abs(E.TRAJ_EDITED[i][2] - E.TRAJ_ORIGINAL[i][2]) <= 1e-9
               for i in range(i0, i1 + 1))
if occ_same:
    npass += 1
    print("遮蔽区間 index %d-%d で2変種一致 OK" % (i0, i1))
else:
    fail += 1
    print("FAIL 遮蔽区間 index %d-%d で2変種が不一致" % (i0, i1))

# Edited が Original を追い越さないこと
ahead = [r[0] for r, o in zip(E.TRAJ_EDITED, E.TRAJ_ORIGINAL) if r[2] > o[2] + 1e-9]
if not ahead:
    npass += 1
else:
    fail += 1
    print("FAIL Edited が Original を追い越す index: %s" % ahead)

# ------------------------------------------- 遮蔽の再設計（2026-08-04）
print()
print("-" * 68)
print("遮蔽と再出現")
print("-" * 68)


def sil_x(xc):
    us = [416.0 + (xc + x) * K / (y + 10.0) for x, y in S.TARGET_CS]
    return min(us), max(us)


occ_u = [proj(vv[0], vv[1], 0.0)[0] for vv in obj_verts("Occluder")]
L, R = min(occ_u), max(occ_u)
print("遮蔽板 u = %.2f .. %.2f (幅 %.2f px)" % (L, R, R - L))
wt = sil_x(S.TARGET_X_START)[1] - sil_x(S.TARGET_X_START)[0]
print("Target 幅 = %.2f px" % wt)

full = {}
for name, table in (("original", E.TRAJ_ORIGINAL), ("edited", E.TRAJ_EDITED)):
    idxs = []
    for idx, _f, x, _u in table:
        a, b = sil_x(x)
        if a >= L - 1e-9 and b <= R + 1e-9:
            idxs.append(idx)
    full[name] = (min(idxs), max(idxs)) if idxs else None

# 1. 完全遮蔽区間が spec の宣言と一致すること
for name in ("original", "edited"):
    if full[name] == S.OCCLUSION_INDEX:
        npass += 1
        print("%-10s 完全遮蔽 index %s = spec 宣言 OK" % (name, full[name]))
    else:
        fail += 1
        print("FAIL %s 完全遮蔽 index %s != spec %s"
              % (name, full[name], S.OCCLUSION_INDEX))

# 2. 2変種で完全遮蔽区間が一致（入りも出も）
if full["original"] == full["edited"]:
    npass += 1
    print("2変種の完全遮蔽区間が一致（入り・出ともに） OK")
else:
    fail += 1
    print("FAIL 完全遮蔽区間が2変種で不一致: %s" % full)

# 3. 完全遮蔽が t_sync 以降で始まること（再出現の測定が2変種で同一になる条件）
if full["original"] and full["original"][0] >= S.EDIT["t_sync"]:
    npass += 1
else:
    fail += 1
    print("FAIL 完全遮蔽が t_sync=%d より前に始まる: %s"
          % (S.EDIT["t_sync"], full["original"]))

# 4. 再出現区間と最終可視幅
i1 = full["original"][1] if full["original"] else None
if i1 is not None:
    reap = (i1 + 1, 48)
    if reap == S.REAPPEAR_INDEX:
        npass += 1
        print("再出現 index %s = spec 宣言 OK (%d frames)" % (reap, 48 - i1))
    else:
        fail += 1
        print("FAIL 再出現 index %s != spec %s" % (reap, S.REAPPEAR_INDEX))
    vis48 = sil_x(E.TRAJ_ORIGINAL[48][2])[1] - R
    if vis48 >= S.MIN_FINAL_VISIBLE_PX:
        npass += 1
        print("最終フレームの可視幅 %.2f px (>= %.1f) OK"
              % (vis48, S.MIN_FINAL_VISIBLE_PX))
    else:
        fail += 1
        print("FAIL 最終フレームの可視幅 %.2f px < %.1f"
              % (vis48, S.MIN_FINAL_VISIBLE_PX))
    print("再出現後の可視幅推移: %s"
          % [round(max(0.0, sil_x(E.TRAJ_ORIGINAL[i][2])[1] - R), 1)
             for i in range(i1 + 1, 49)])

# 5. 遮蔽板が Target を鉛直方向に完全に覆うこと
occ_v = [proj(0.0, vv[1], vv[2])[1] for vv in obj_verts("Occluder")]
tgt_v = [proj(0.0, y, z)[1] for x, y in S.TARGET_CS for z in S.TARGET_Z]
if min(occ_v) <= min(tgt_v) and max(tgt_v) <= max(occ_v):
    npass += 1
    print("遮蔽板の v 範囲 [%.1f,%.1f] が Target [%.1f,%.1f] を覆う OK"
          % (min(occ_v), max(occ_v), min(tgt_v), max(tgt_v)))
else:
    fail += 1
    print("FAIL 遮蔽板の v 範囲 [%.1f,%.1f] が Target [%.1f,%.1f] を覆わない"
          % (min(occ_v), max(occ_v), min(tgt_v), max(tgt_v)))

# 最大乖離
devs = [(r[0], abs(r[3] - o[3])) for r, o in zip(E.TRAJ_EDITED, E.TRAJ_ORIGINAL)]
mi, mv = max(devs, key=lambda t: t[1])
if mi == E.MAX_DEVIATION["index"] and abs(mv - E.MAX_DEVIATION["px"]) <= 0.001:
    npass += 1
    print("最大乖離 index %d / %.4f px OK" % (mi, mv))
else:
    fail += 1
    print("FAIL max deviation calc (index %d, %.4f px) | lit %s"
          % (mi, mv, E.MAX_DEVIATION))

print()
print("PASS %d / FAIL %d" % (npass, fail))
sys.exit(1 if fail else 0)
