# -*- coding: utf-8 -*-
"""Pod用: armc_points.json + scene_c_prime_traj.json -> armC_prime.mkv
v シフト + visible再計算(f0-35 Occluder矩形) + replay + FFV1 gbrp。
frame0 が既存 armC_0000.png と bit一致するか検証(あれば)。"""
import json, copy, os, subprocess, numpy as np, imageio.v2 as imageio
K=1155.5556
SRC=os.environ.get('ARMC_POINTS','armc_points.json')
TRAJ=os.environ.get('TRAJ','scene_c_prime_traj.json')
OUT=os.environ.get('OUT','armC_prime.mkv')
REF=os.environ.get('REF_PNG','')  # 任意: armC_0000.png があれば bit照合

src=json.load(open(SRC,encoding='utf-8')); traj=json.load(open(TRAJ,encoding='utf-8'))
dZ={kf['index']:kf['dZ'] for kf in traj['keyframes']}
istgt=np.array([p['is_target'] for p in src['points']])
W,H,R=src['params']['width'],src['params']['height'],src['params']['radius']
OU=(401.5556,567.6667); OV=(232.7778,456.6667); OD=8.00
def occ_hidden(u,v,d): return (u>=OU[0])&(u<=OU[1])&(v>=OV[0])&(v<=OV[1])&(d>OD+1e-6)
def in_screen(u,v): return (u>=0)&(u<W)&(v>=0)&(v<H)
def disc(r): return np.array([(dx,dy) for dy in range(-r,r+1) for dx in range(-r,r+1) if dx*dx+dy*dy<=r*r],np.int32)
off=disc(R); rgb=np.array([p['rgb'] for p in src['points']],np.uint8)

os.makedirs('armC_prime_png',exist_ok=True)
srcO=src['frames']['original']
for i in range(49):
    r=copy.deepcopy(srcO[i]); u=np.array(r['u']); v=np.array(r['v']); d=np.array(r['depth']); vis0=np.array(r['visible'],bool)
    dz=dZ[i]; vnew=np.where(istgt, v-(K/d)*dz, v)
    vis = np.where(istgt, in_screen(u,vnew)&~occ_hidden(u,vnew,d), vis0) if i<=35 else vis0
    canvas=np.zeros((H,W,3),np.uint8); drawn=[]
    for k in range(len(u)):
        if not vis[k]: continue
        ci=int(round(u[k]-0.5)); cj=int(round(vnew[k]-0.5))
        if 0<=ci<W and 0<=cj<H: drawn.append((d[k],k,ci,cj))
    drawn.sort(key=lambda t:(-t[0],t[1]))
    for _,k,ci,cj in drawn:
        xs=off[:,0]+ci; ys=off[:,1]+cj; m=(xs>=0)&(xs<W)&(ys>=0)&(ys<H); canvas[ys[m],xs[m]]=rgb[k]
    imageio.imwrite('armC_prime_png/armC_%04d.png'%i, canvas)

if REF and os.path.exists(REF):
    a=imageio.imread('armC_prime_png/armC_0000.png'); b=imageio.imread(REF)
    assert np.array_equal(a[...,:3],b[...,:3]), "frame0 != existing armC_0000.png"
    print("frame0 bit-match existing PNG: True")

subprocess.run('ffmpeg -y -hide_banner -loglevel error -framerate 16 -i armC_prime_png/armC_%04d.png '
               '-c:v ffv1 -pix_fmt gbrp -g 1 -flags +bitexact -fflags +bitexact '+OUT, shell=True, check=True)
import hashlib
rd=imageio.get_reader(OUT); arr=np.stack([f[...,:3] for f in rd]); rd.close()
print("wrote", OUT, arr.shape)
print("content sha256(first16):", hashlib.sha256(arr.tobytes()).hexdigest()[:16], "(expect fde80efbe48dcb51)")
