# -*- coding: utf-8 -*-
"""evaluator_floor.py — Evaluator floor 校正（§15/§37.8）。GPU不要。
tracker = 色セグメンテーション(orange) + Marker位置除外 + 最大連結成分重心。
floor = tv から検出した center と ids==3 重心の誤差 p95。
使い方: python evaluator_floor.py --tv tv_R0S0_original.mkv --ids ids_original.npy
"""
import argparse, numpy as np, imageio.v2 as imageio
from scipy import ndimage

MARKER=(658.76,123.47)   # scene_c_expect Marker_centre（固定物・除外）

def track_orange(img):
    R,G,B=img[...,0].astype(int),img[...,1].astype(int),img[...,2].astype(int)
    mask=(R>G)&(G>=B)&((R-B)>40)&(R>80)
    if mask.sum()==0: return None,0
    lab,n=ndimage.label(mask)
    best=None; bestsz=0
    for c in range(1,n+1):
        ys,xs=np.where(lab==c); cu,cv=xs.mean(),ys.mean()
        if abs(cu-MARKER[0])<30 and abs(cv-MARKER[1])<30: continue
        if len(xs)>bestsz: bestsz=len(xs); best=(cu,cv)
    return best,bestsz

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tv',required=True); ap.add_argument('--ids',required=True)
    ap.add_argument('--gate',type=float,default=47.0)
    a=ap.parse_args()
    rd=imageio.get_reader(a.tv); tv=np.stack([f[...,:3] for f in rd]); rd.close()
    ids=np.load(a.ids)
    rows=[]
    for f in range(len(tv)):
        m=(ids[f]==3.0); gtn=int(m.sum())
        cen,tn=track_orange(tv[f])
        if gtn==0: rows.append((f,None,'occluded')); continue
        if cen is None: rows.append((f,None,'tracker_fail')); continue
        gu,gv=np.where(m)[1].mean(),np.where(m)[0].mean()
        rows.append((f,float(np.hypot(cen[0]-gu,cen[1]-gv)),'ok'))
    errs=np.array([r[1] for r in rows if r[2]=='ok'])
    p95=float(np.percentile(errs,95))
    print('scored=%d  median=%.3f  p95=%.3f  p99=%.3f  max=%.3f'%(
        len(errs),np.median(errs),p95,np.percentile(errs,99),errs.max()))
    print('occluded:',[r[0] for r in rows if r[2]=='occluded'])
    print('tracker_fail:',[r[0] for r in rows if r[2]=='tracker_fail'])
    print('VERDICT: floor p95 %.2f %s gate %.1f -> %s'%(
        p95,'<' if p95<a.gate else '>=',a.gate,
        'gate valid' if p95<a.gate else 'RETRACT gate (S37.8)'))

if __name__=='__main__': raise SystemExit(main())
