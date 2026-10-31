"""
armc_select.py  —  Arm C の点選定・衝突グラフ・彩色（bpy 非依存）

protocol §13.4 / §13.5 の規約を実装する。Blender の外でも走るので単体テストできる。

データ配置
    uv  : (M, F, 2) float  各候補点の画面座標。F は全変種のフレームを連結したもの
    vis : (M, F)    bool   可視フラグ

「同時可視かつ距離」はフレームをまたいだ最小値で評価する
（protocol の「一度でも 24px 以内になるペア」）。
点集合は 3D で一度だけ確定させるので、変種のフレームを連結して一括で扱う。

2026-08-05 の変更
    fps_select に min_dist を追加。下限未満しか残らない場合は**点数を減らして止まる**。
      下限未満の点を掴むと達成分離が壊れる（pitch 0.30 で 0.934px まで落ちた）。
    greedy_color に balance を追加。許される色のうち**使用回数が最小のもの**を選ぶ。
      衝突ペアが0本だと全点が color 0 になり identity が原理的に測れなくなる。
"""
import numpy as np

INF = np.inf


def _sub(uv, vis, idx):
    idx = np.asarray([int(x) for x in idx], dtype=int)
    return uv[idx], vis[idx], idx


def pair_min_distance(uv, vis, s):
    """点 s と全候補との「同時可視フレームでの最小画面距離」。
    同時可視フレームが無ければ INF（制約なし）。"""
    dx = uv[:, :, 0] - uv[s, :, 0]
    dy = uv[:, :, 1] - uv[s, :, 1]
    dd = np.sqrt(dx * dx + dy * dy)
    dd = np.where(vis & vis[s], dd, INF)
    return dd.min(axis=1)


def min_separation(uv, vis, idx):
    """与えられた点集合の達成最小分離。選定済み部分だけに切り出して計算する。"""
    if len(idx) < 2:
        return INF
    u, v, _ = _sub(uv, vis, idx)
    best = INF
    for a in range(len(u)):
        dx = u[:, :, 0] - u[a, :, 0]
        dy = u[:, :, 1] - u[a, :, 1]
        dd = np.sqrt(dx * dx + dy * dy)
        dd = np.where(v & v[a], dd, INF)
        d = dd.min(axis=1)
        d[a] = INF
        m = float(d.min())
        if m < best:
            best = m
    return best


def fps_select(uv, vis, n, pool=None, preselected=None, min_dist=None):
    """最遠点サンプリング。

    pool        : 選定対象の候補インデックス
    preselected : 既選定点（距離の下限に効くが返り値には含めない）
    min_dist    : これを下回る点は選ばない。足りなければ n より少なく返す

    種は「可視フレーム数が最多」の点（同数ならインデックス最小）。決定的。
    """
    M = uv.shape[0]
    pool = np.arange(M) if pool is None else np.asarray(pool, dtype=int)
    pool = pool[vis[pool].any(axis=1)]
    pre = [int(x) for x in (preselected or [])]
    taken = set(pre)
    pool = np.array([p for p in pool if int(p) not in taken], dtype=int)
    if len(pool) == 0 or n <= 0:
        return np.array([], dtype=int)

    running = np.full(M, INF)
    for s in pre:
        running = np.minimum(running, pair_min_distance(uv, vis, s))

    chosen = []
    if not pre:
        nvis = vis[pool].sum(axis=1)
        seed = int(pool[int(np.lexsort((pool, -nvis))[0])])
        chosen.append(seed)
        taken.add(seed)
        running = np.minimum(running, pair_min_distance(uv, vis, seed))

    while len(chosen) < n:
        cand = np.array([p for p in pool if int(p) not in taken], dtype=int)
        if len(cand) == 0:
            break
        vals = running[cand]
        best = int(cand[int(np.lexsort((cand, -vals))[0])])
        if min_dist is not None and running[best] < min_dist:
            break                       # 下限を割るなら増やさない
        chosen.append(best)
        taken.add(best)
        running = np.minimum(running, pair_min_distance(uv, vis, best))

    return np.array(chosen, dtype=int)


def collision_pairs(uv, vis, idx, radius):
    """protocol §13.4：一度でも「同時可視かつ radius 以内」になるペア。"""
    if len(idx) < 2:
        return []
    u, v, ids = _sub(uv, vis, idx)
    pairs = []
    for a in range(len(ids)):
        dx = u[:, :, 0] - u[a, :, 0]
        dy = u[:, :, 1] - u[a, :, 1]
        dd = np.sqrt(dx * dx + dy * dy)
        dd = np.where(v & v[a], dd, INF)
        d = dd.min(axis=1)
        for b in range(a + 1, len(ids)):
            if d[b] <= radius:
                pairs.append((int(ids[a]), int(ids[b])))
    return pairs


def greedy_color(nodes, pairs, ncolors, balance=True):
    """次数の大きい順に貪欲彩色（protocol §13.4）。
    balance=True なら許される色のうち使用回数が最小のものを選ぶ。
    衝突ペアが0本のときに全点が同色になるのを防ぐ。
    返り値: {node: color_index}, 使った色数, 塗れなかったノード"""
    adj = {int(n): set() for n in nodes}
    for a, b in pairs:
        a, b = int(a), int(b)
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    order = sorted(adj.keys(), key=lambda n: (-len(adj[n]), n))
    color = {}
    used_count = [0] * ncolors
    failed = []
    for n in order:
        forbidden = {color[m] for m in adj[n] if m in color}
        allowed = [c for c in range(ncolors) if c not in forbidden]
        if not allowed:
            failed.append(n)
            continue
        if balance:
            c = min(allowed, key=lambda x: (used_count[x], x))
        else:
            c = allowed[0]
        color[n] = c
        used_count[c] += 1
    return color, (len(set(color.values())) if color else 0), failed


def verify_coloring(color, pairs):
    bad = []
    for a, b in pairs:
        a, b = int(a), int(b)
        if a in color and b in color and color[a] == color[b]:
            bad.append((a, b, color[a]))
    return bad


def occupancy(uv, vis, idx, radius, width, height):
    """円の面積（dx^2+dy^2 <= r^2 の画素数）と、フレームごとの占有率。"""
    area = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                area += 1
    idx = np.asarray([int(x) for x in idx], dtype=int)
    nvis = vis[idx].sum(axis=0)
    return area, (nvis * area) / float(width * height)
