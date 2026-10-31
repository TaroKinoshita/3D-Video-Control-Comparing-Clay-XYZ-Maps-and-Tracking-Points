# -*- coding: utf-8 -*-
"""score_t_control.py — 偵察run の T_control / T_coherence 判定（GPU不要）。
生成動画に color tracker を当て、C' の GT center と比較する。
GT center: u = Scene C mask重心, v = Scene C mask重心 - (K/9.00)*dZ(f)。
使い方: python score_t_control.py --gen generated.mkv --ids ids_original.npy --traj scene_c_prime_traj.json
陽性対照: --gen に Scene C tv を渡すと「暗記」を模擬し T_control 超過を確認できる。
"""
import argparse, json, numpy as np, imageio.v2 as imageio
from scipy import ndimage
K=1155.5556; CENTER_DEPTH=9.00; MARKER=(658.76,123.47); GATE_CTRL=47.0; GATE_COH=2/44

def track(img):
    R,G,B=img[...,0].astype(int),img[...,1].astype(int),img[...,2].astype(int)
    mask=(R>G)&(G>=B)&((R-B)>40)&(R>80)
    if mask.sum()==0: return None,0
    lab,n=ndimage.label(mask); best=None;bestsz=0;ncomp=0
    for c in range(1,n+1):
        ys,xs=np.where(lab==c); cu,cv=xs.mean(),ys.mean()
        if abs(cu-MARKER[0])<30 and abs(cv-MARKER[1])<30: continue
        ncomp+=1
        if len(xs)>bestsz: bestsz=len(xs); best=(cu,cv)
    return best,ncomp

def gt_center(ids, dZ):
    """Scene C mask重心 -> C' GT center。可視(npx>0)のみ。"""
    g={}
    for f in range(len(ids)):
        m=(ids[f]==3.0); n=int(m.sum())
        if n==0: continue
        u=np.where(m)[1].mean(); v=np.where(m)[0].mean()
        g[f]=(u, v-(K/CENTER_DEPTH)*dZ[f])
    return g

def score(gen, ids, dZ):
    G=gt_center(ids,dZ)
    errs=[]; coh_bad=0; coh_tot=0; fail=[]
    for f in range(len(gen)):
        if f not in G: continue          # 遮蔽窓は面から除外
        det,nc=track(gen[f])
        if det is None: fail.append(f); continue   # tracker failure -> 分母除外
        coh_tot+=1; coh_bad+= (nc!=1)
        errs.append(np.hypot(det[0]-G[f][0], det[1]-G[f][1]))
    errs=np.array(errs)
    return dict(n=len(errs), median=float(np.median(errs)), p95=float(np.percentile(errs,95)),
                coh_rate=coh_bad/coh_tot if coh_tot else 0, fail=fail, G=G)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gen',required=True); ap.add_argument('--ids',required=True)
    ap.add_argument('--traj',required=True)
    a=ap.parse_args()
    rd=imageio.get_reader(a.gen); gen=np.stack([f[...,:3] for f in rd]); rd.close()
    ids=np.load(a.ids)
    tr=json.load(open(a.traj,encoding='utf-8')); dZ={kf['index']:kf['dZ'] for kf in tr['keyframes']}
    dZ=[dZ[i] for i in range(len(ids))]
    r=score(gen,ids,dZ)
    print("scored=%d  T_control median=%.2f p95=%.2f  T_coherence bad-rate=%.4f (%d/44)"%(
        r['n'],r['median'],r['p95'],r['coh_rate'],round(r['coh_rate']*44)))
    print("tracker_fail:",r['fail'])
    print("VERDICT T_control : median %.2f %s %.0f -> %s"%(
        r['median'],'<=' if r['median']<=GATE_CTRL else '>',GATE_CTRL,
        'PASS(control読めてる)' if r['median']<=GATE_CTRL else 'FAIL(暗記の疑い)'))
    print("VERDICT T_coherence: %.4f %s %.4f -> %s"%(
        r['coh_rate'],'<=' if r['coh_rate']<=GATE_COH else '>',GATE_COH,
        'PASS' if r['coh_rate']<=GATE_COH else 'FAIL'))
    return r
if __name__=='__main__': main()

# ---- 暗記判別の追加（Scene C center vs C' center）----
def memorization_probe(gen, ids, dZ):
    """生成が Scene C 軌道(暗記)寄りか C' 軌道(制御)寄りかを定量化。"""
    from scipy import ndimage
    # Scene C center = mask重心そのまま（v シフトなし）
    SC={}; CP={}
    for f in range(len(ids)):
        m=(ids[f]==3.0); n=int(m.sum())
        if n==0: continue
        u=np.where(m)[1].mean(); v=np.where(m)[0].mean()
        SC[f]=(u,v); CP[f]=(u, v-(K/CENTER_DEPTH)*dZ[f])
    esc=[]; ecp=[]
    for f in range(len(gen)):
        if f not in SC: continue
        det,nc=track(gen[f])
        if det is None: continue
        esc.append(np.hypot(det[0]-SC[f][0], det[1]-SC[f][1]))
        ecp.append(np.hypot(det[0]-CP[f][0], det[1]-CP[f][1]))
    esc=np.array(esc); ecp=np.array(ecp)
    return float(np.median(esc)), float(np.median(ecp))
