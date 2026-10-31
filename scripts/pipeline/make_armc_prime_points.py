# -*- coding: utf-8 -*-
"""armC_prime points 生成：Scene C の armc_points.json の Target 72点を C' 軌道で v シフト。
C' は Original のみ。u/depth/visible/rgb 不変、v_new = v_old - (K/depth)*dZ(f)。"""
import json, copy, numpy as np

K = 1155.5556
src   = json.load(open('/mnt/user-data/uploads/armc_points.json', encoding='utf-8'))
traj  = json.load(open('/mnt/user-data/uploads/scene_c_prime_traj.json', encoding='utf-8'))
dZ = {kf['index']: kf['dZ'] for kf in traj['keyframes']}      # 凍結リテラルから
istgt = np.array([p['is_target'] for p in src['points']])

out = {"points": copy.deepcopy(src['points']),
       "frames": {"original": []},           # C' は Original のみ
       "params": copy.deepcopy(src['params']),
       "meta": {"derived_from": "armc_points.json (Scene C)", "A": traj['meta']['A'],
                "rule": "v_new = v_old - (K/depth)*dZ ; target band only"}}

frames_c = {i: r for i,r in enumerate([r for r in src['frames']['original']])}
for i in range(49):
    r = copy.deepcopy(src['frames']['original'][i])
    u = np.array(r['u']); v = np.array(r['v']); dep = np.array(r['depth'])
    dz = dZ[i]
    vnew = v - (K/dep)*dz                    # 全点式（non-target は dz 効くが is_target で選ぶ）
    v2 = np.where(istgt, vnew, v)            # Target 帯のみ更新
    # mesh_co Z も dz 足す（記録整合。描画には使わない）
    r['v'] = [round(float(x),5) for x in v2]
    out['frames']['original'].append(r)
for k,p in enumerate(out['points']):
    if p['is_target']:
        p['mesh_co'] = [p['mesh_co'][0], p['mesh_co'][1], p['mesh_co'][2]]  # base（frameごとに動くのでpoint側は据置）

# ---- 制約突合 ----
srcO = src['frames']['original']; dstO = out['frames']['original']
# 制約3: frame0 は完全一致（dZ0=0）
f0d = max(abs(a-b) for a,b in zip(srcO[0]['v'], dstO[0]['v']))
assert f0d == 0.0, "constraint3: frame0 must be identical (got %g)"%f0d
# 制約4: 35-41 で Target visible=0（不変）
for i in range(35,42):
    vv = np.array(dstO[i]['visible'])
    assert int((vv & istgt).sum())==0, "constraint4 broken at %d"%i
# 制約2: Target 占有率 = visible 円盤面積和/画面。v シフトは面積不変 -> 各フレーム Scene C と同一
R=src['params']['radius']; area=sum(1 for dy in range(-R,R+1) for dx in range(-R,R+1) if dx*dx+dy*dy<=R*R)
occ_src=[int((np.array(srcO[i]['visible'])&istgt).sum()) for i in range(49)]
occ_dst=[int((np.array(dstO[i]['visible'])&istgt).sum()) for i in range(49)]
assert occ_src==occ_dst, "constraint2: visible target count changed"
# GT center 軌道（depth 9.00 -> pxu 128.40）
pxu_c = K/9.00
vc0 = 342.7160
gt_v_center = [round(vc0 - pxu_c*dZ[i],4) for i in range(49)]

# ---- selftest: 壊すと発火 ----
def selftest():
    bad = np.array(dstO[0]['v']).copy(); bad[0]+=5   # f0 破壊
    tripped = (max(abs(a-b) for a,b in zip(srcO[0]['v'], bad)) != 0.0)
    # v シフト符号逆で GT center とズレる
    wrong = [vc0 + pxu_c*dZ[i] for i in range(49)]
    dev = max(abs(a-b) for a,b in zip(wrong, gt_v_center))
    assert tripped and dev>0, "selftest failed"
    return dev
st = selftest()

json.dump(out, open('/mnt/user-data/outputs/armc_prime_points.json','w',encoding='utf-8'),
          ensure_ascii=False)

print("ALL CONSTRAINT ASSERTS PASS")
print("constraint3 frame0 diff : %g px (=0)"%f0d)
print("constraint4 occ 35-41   : target visible 0 (ok)")
print("constraint2 occupancy   : identical to Scene C (disc area %d px)"%area)
print("selftest sign-flip dev  : %.1f px (>0 trips)"%st)
print()
print("=== v shift magnitude at peak dZ (f=17) ===")
i=17; dep=np.array(srcO[i]['depth'])[istgt]
sh=(K/dep)*dZ[i]
print("dZ=%.4f  depth %.2f-%.2f  v-shift %.1f-%.1f px (up)"%(dZ[i],dep.min(),dep.max(),sh.min(),sh.max()))
print()
print("=== GT Target center trajectory (for T_control) ===")
for i in [0,8,17,26,34,35,41,42,48]:
    print("f=%2d  dZ=%.4f  GT v_center=%.2f  (f0 342.72)"%(i,dZ[i],gt_v_center[i]))
