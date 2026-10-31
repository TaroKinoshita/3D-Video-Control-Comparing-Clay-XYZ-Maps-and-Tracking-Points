# -*- coding: utf-8 -*-
"""armC_prime v2: v シフト + visible を frame0-35 は Occluder矩形で再計算、36-48は継承。"""
import json, copy, numpy as np
K=1155.5556
src=json.load(open('/mnt/user-data/uploads/armc_points.json',encoding='utf-8'))
traj=json.load(open('/mnt/user-data/uploads/scene_c_prime_traj.json',encoding='utf-8'))
dZ={kf['index']:kf['dZ'] for kf in traj['keyframes']}
istgt=np.array([p['is_target'] for p in src['points']])
W,H=src['params']['width'],src['params']['height']
OU=(401.5556,567.6667); OV=(232.7778,456.6667); OD=8.00

def occ_hidden(u,v,d): return (u>=OU[0])&(u<=OU[1])&(v>=OV[0])&(v<=OV[1])&(d>OD+1e-6)
def in_screen(u,v):    return (u>=0)&(u<W)&(v>=0)&(v<H)

out={"points":copy.deepcopy(src['points']),"frames":{"original":[]},
     "params":copy.deepcopy(src['params']),
     "meta":{"derived_from":"armc_points.json","A":traj['meta']['A'],
             "visible_rule":"f0-35: in_screen & ~occ_rect (recomputed); f36-48: inherited"}}
srcO=src['frames']['original']
chg=[]
for i in range(49):
    r=copy.deepcopy(srcO[i])
    u=np.array(r['u']); v=np.array(r['v']); d=np.array(r['depth'])
    vis0=np.array(r['visible'],bool)
    dz=dZ[i]
    vnew=np.where(istgt, v-(K/d)*dz, v)
    if i<=35:
        # Target のみ再計算。non-target は継承
        vis_re = in_screen(u,vnew) & ~occ_hidden(u,vnew,d)
        visnew = np.where(istgt, vis_re, vis0)
    else:
        visnew = vis0
    n_before=int((vis0&istgt).sum()); n_after=int((visnew&istgt).sum())
    if n_before!=n_after: chg.append((i,n_before,n_after))
    r['v']=[round(float(x),5) for x in vnew]
    r['visible']=[bool(x) for x in visnew]
    out['frames']['original'].append(r)

# 制約再突合
dstO=out['frames']['original']
assert max(abs(a-b) for a,b in zip(srcO[0]['v'],dstO[0]['v']))==0.0, "c3 frame0 v"
assert all(np.array(dstO[0]['visible'])==np.array(srcO[0]['visible'])), "c3 frame0 visible"
for i in range(36,41):
    assert int((np.array(dstO[i]['visible'])&istgt).sum())==0, "c4 occ %d"%i
json.dump(out,open('/mnt/user-data/outputs/armc_prime_points.json','w',encoding='utf-8'),ensure_ascii=False)
print("ALL ASSERTS PASS (c3 frame0 identical, c4 occ 35-41 zero)")
print("frames where target visible count changed (idx, Scene C, C'):")
for x in chg: print("  ",x)
