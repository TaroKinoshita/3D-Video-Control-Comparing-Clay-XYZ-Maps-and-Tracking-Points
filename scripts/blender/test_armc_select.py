"""
test_armc_select.py  —  Arm C の選定・彩色の単体テスト（Blender 不要）
手で答えが出る合成データだけを使う。
"""
import sys
import numpy as np
sys.path.insert(0, ".")
import armc_select as A
import armc_palette as P

npass = fail = 0
def ck(n, ok, d=""):
    global npass, fail
    if ok: npass += 1
    else:
        fail += 1
        print("FAIL %-32s %s" % (n, d))

# ---------- 1) 静止した1直線上の5点。手で答えが出る ----------
F = 3
xs = [0.0, 10.0, 20.0, 30.0, 40.0]
uv = np.zeros((5, F, 2), dtype=np.float64)
for i, x in enumerate(xs):
    uv[i, :, 0] = x
vis = np.ones((5, F), dtype=bool)

sel = A.fps_select(uv, vis, 3)
ach = A.min_separation(uv, vis, sel)
ck("fps_line_seed_is_0", sel[0] == 0, str(sel))
ck("fps_line_order", list(sel) == [0, 4, 2], str(sel))
ck("fps_line_achieved_20", abs(ach - 20.0) < 1e-9, ach)
print("直線5点から3点: 選定 %s / 達成分離 %.1f px（手計算 20.0）" % (list(sel), ach))

sel5 = A.fps_select(uv, vis, 5)
ach5 = A.min_separation(uv, vis, sel5)
ck("fps_all5_achieved_10", abs(ach5 - 10.0) < 1e-9, ach5)

# ---------- 2) 一度も可視でない点は選ばれない ----------
vis2 = vis.copy(); vis2[2, :] = False
sel2 = A.fps_select(uv, vis2, 5)
ck("never_visible_excluded", 2 not in sel2, str(sel2))
ck("never_visible_count", len(sel2) == 4, len(sel2))

# ---------- 3) 同時可視でないペアは距離制約を受けない ----------
uv3 = np.zeros((2, 2, 2)); uv3[0, :, 0] = 0.0; uv3[1, :, 0] = 1.0
vis3 = np.array([[True, False], [False, True]])
d = A.pair_min_distance(uv3, vis3, 0)
ck("disjoint_visibility_inf", np.isinf(d[1]), d[1])
print("同時可視ゼロのペア -> 距離 INF（制約なし） OK")

# ---------- 4) 動く点は「最小」で評価される ----------
uv4 = np.zeros((2, 3, 2))
uv4[0, :, 0] = [0, 0, 0]
uv4[1, :, 0] = [100, 5, 100]      # 中央フレームだけ接近
vis4 = np.ones((2, 3), dtype=bool)
ck("moving_uses_min", abs(A.pair_min_distance(uv4, vis4, 0)[1] - 5.0) < 1e-9)
print("フレームをまたいだ最小値で評価される（100/5/100 -> 5.0） OK")

# ---------- 5) 衝突グラフと彩色 ----------
pairs = A.collision_pairs(uv, vis, [0, 1, 2, 3, 4], 24.0)
# 距離 <=24 のペア: (0,1)(0,2)(1,2)(1,3)(2,3)(2,4)(3,4) = 7本
ck("collision_count_7", len(pairs) == 7, "%d %s" % (len(pairs), pairs))
print("直線5点・半径24 の衝突ペア数 %d（手計算 7）" % len(pairs))

color, nc, failed = A.greedy_color([0, 1, 2, 3, 4], pairs, 12)
ck("coloring_no_failure", not failed, str(failed))
ck("coloring_valid", not A.verify_coloring(color, pairs), A.verify_coloring(color, pairs))
print("彩色 %s / 使用色数 %d" % (color, nc))

# 三角形は3色必要
tri_pairs = [(0, 1), (1, 2), (0, 2)]
c3, n3, f3 = A.greedy_color([0, 1, 2], tri_pairs, 12)
ck("triangle_needs_3", n3 == 3, n3)
ck("triangle_valid", not A.verify_coloring(c3, tri_pairs))

# 色が足りない場合に failed が返る
c2, n2, f2 = A.greedy_color([0, 1, 2], tri_pairs, 2)
ck("insufficient_colors_detected", len(f2) == 1, str(f2))
print("3色必要なグラフに 2 色 -> failed=%s（検出） OK" % f2)

# 13-クリークは12色で塗れない（パレット上限の検出）
n13 = list(range(13))
p13 = [(a, b) for a in n13 for b in n13 if a < b]
c13, nn13, f13 = A.greedy_color(n13, p13, 12)
ck("clique13_fails", len(f13) == 1, str(f13))
print("13-クリークに 12 色 -> failed=%s（パレット上限の検出） OK" % f13)

# ---------- 6) 占有率と円の面積 ----------
area, occ = A.occupancy(uv, vis, [0, 1, 2, 3, 4], 6, 832, 480)
ck("disc_area_113", area == 113, area)
ck("occupancy_value", abs(occ[0] - 5 * 113 / 399360.0) < 1e-12, occ[0])
print("半径6の円の面積 %d px（protocol 113） / 5点の占有率 %.5f%%" % (area, occ[0] * 100))

# ---------- 7) preselected が距離の下限に効く ----------
selp = A.fps_select(uv, vis, 1, pool=[1, 2, 3], preselected=[0])
ck("preselected_pushes_away", selp[0] == 3, str(selp))
print("既存点0 がある状態で pool=[1,2,3] から1点 -> %s（最も遠い3） OK" % list(selp))

# ---------- 8) 決定性 ----------
a1 = A.fps_select(uv, vis, 4)
a2 = A.fps_select(uv, vis, 4)
ck("deterministic", list(a1) == list(a2), "%s vs %s" % (list(a1), list(a2)))

# ---------- 9) パレットとの結合 ----------
cols = P.PALETTE["target"]
ck("palette_12", len(cols) == 12)
assign = {n: cols[c] for n, c in color.items()}
ck("assign_in_band", all(140 <= v <= 242 for c in assign.values() for v in c))
# 隣接ペアの色間距離が最小色間距離以上
dmin = min(P.dist(assign[a], assign[b]) for a, b in pairs)
ck("adjacent_color_dist", dmin >= P.MIN_DIST - 1e-9, "%.4f" % dmin)
print("隣接ペアの最小色間距離 %.4f（>= %.4f） OK" % (dmin, P.MIN_DIST))


# ---------- 10) min_dist で打ち切る ----------
selm = A.fps_select(uv, vis, 5, min_dist=15.0)
ck("min_dist_stops", len(selm) == 3, "%d 点 %s" % (len(selm), list(selm)))
ck("min_dist_holds", A.min_separation(uv, vis, selm) >= 15.0,
   A.min_separation(uv, vis, selm))
print("min_dist=15 -> %d 点で打ち切り（分離 %.1f px）"
      % (len(selm), A.min_separation(uv, vis, selm)))
selm2 = A.fps_select(uv, vis, 5, min_dist=5.0)
ck("min_dist_loose", len(selm2) == 5, len(selm2))

# ---------- 11) balance=True で色が分散する ----------
nodes = list(range(24))
cb, ncb, fb = A.greedy_color(nodes, [], 12, balance=True)
cnt = {}
for v in cb.values():
    cnt[v] = cnt.get(v, 0) + 1
ck("balance_spreads", len(cnt) == 12, str(sorted(cnt.items())))
ck("balance_even", set(cnt.values()) == {2}, str(sorted(cnt.items())))
cu, ncu, fu = A.greedy_color(nodes, [], 12, balance=False)
ck("unbalanced_degenerates", len(set(cu.values())) == 1, str(set(cu.values())))
print("衝突0本・24点: balance=True -> %d色を各2点 / balance=False -> %d色"
      % (len(cnt), len(set(cu.values()))))

# balance でも隣接制約は守る
cb2, _, _ = A.greedy_color([0, 1, 2], tri_pairs, 12, balance=True)
ck("balance_respects_adjacency", not A.verify_coloring(cb2, tri_pairs))
ck("balance_triangle_3", len(set(cb2.values())) == 3, str(cb2))

# ---------- 12) preselected は返り値に含まれない ----------
sp = A.fps_select(uv, vis, 2, pool=[0, 1, 2, 3, 4], preselected=[0])
ck("preselected_excluded", 0 not in sp, str(list(sp)))
ck("preselected_count", len(sp) == 2, len(sp))

print()
print("PASS %d / FAIL %d" % (npass, fail))
sys.exit(1 if fail else 0)
