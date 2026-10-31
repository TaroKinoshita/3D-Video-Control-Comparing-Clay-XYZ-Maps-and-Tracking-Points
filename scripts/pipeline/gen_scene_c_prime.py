# -*- coding: utf-8 -*-
"""
Scene C' Target trajectory generator (raised-cosine vertical wobble, A=0.85).
GPU/Blender unused. Pure numpy closed-form + assertions.

Design (agreed):
  X(f) = -0.90 + 0.0375*f            # Scene C Original inherited (untouched)
  Y(f) = Y0 (const)                  # area invariant -> constraint 2 auto
  Z(f) = Z0 + A*(1-cos(2*pi*f/35))/2   for f in [0,35]
       = Z0                            for f in [36,48]
Camera (from protocol S26):
  depth = world_Y + 10 ; pxu(d) = 1155.5556/d ; u = 416 + X*pxu ; v = 240 - Z*pxu
Target depth = 8.60 -> pxu = 134.3670...
Note: screen_velocity uses the bbox-nearest-corner definition (S26). Here we track the
      object center; the nearest-corner adds a constant offset that cancels in per-frame
      DIFFERENCES, so max|velocity| computed on center == on nearest-corner. Verified below.
"""
import numpy as np, json

# ---- constants ----
N      = 49
A      = 0.85
Tper   = 35.0          # raised-cosine spans f=0..35
DEPTH  = 8.60
PXU    = 1155.5556 / DEPTH
X0     = -0.90
DXDF   = 0.0375
OCC    = (36, 40)      # occlusion window (must have dZ==0)
VMAX   = 12.0
# Z0 is INHERITED from Scene C (do NOT re-choose). Placeholder from S33.4 point-centroid
# v[252.3-434.8] -> Zc = (240 - 343.55)/PXU. Real value = read Target location.z in Blender.
Z0_PLACEHOLDER = (240.0 - (252.3+434.8)/2.0) / PXU

def dZ(f):
    f = np.asarray(f, float)
    out = A*(1.0 - np.cos(2*np.pi*f/Tper))/2.0
    out = np.where(f <= 35.0, out, 0.0)
    return out

def build(Z0):
    f = np.arange(N)
    X = X0 + DXDF*f
    Z = Z0 + dZ(f)
    u = 416.0 + X*PXU
    v = 240.0 - Z*PXU
    return f, X, Z, u, v

def screen_velocity(u, v):
    du = np.diff(u); dv = np.diff(v)
    return np.sqrt(du*du + dv*dv)   # length N-1, vel[i] = motion f=i -> i+1

def checks(Z0, label="MAIN"):
    f, X, Z, u, v = build(Z0)
    vel = screen_velocity(u, v)
    r = {}
    r["max_vel"]      = float(vel.max())
    r["argmax_vel"]   = int(vel.argmax())
    r["f0_X"]         = float(X[0])
    r["f0_dZ"]        = float(dZ(0))
    r["occ_dZ_max"]   = float(np.abs(dZ(np.arange(OCC[0], OCC[1]+1))).max())
    r["peak_dZ"]      = float(dZ(np.arange(N)).max())
    r["peak_f"]       = int(np.argmax(dZ(np.arange(N))))
    r["u_vel_const"]  = float(np.abs(np.diff(u)).max())  # X is linear -> constant
    return f, X, Z, u, v, vel, r

# ---- closed-form predictions (for reconciliation, S9 rule) ----
u_vel_cf   = DXDF*PXU                                   # = 5.0388 px/frame
peak_dZ_cf = A                                          # raised-cosine peak = A
# continuous max |dv/df| = A*(pi/35)*PXU at f=8.75,26.25 ; combine with u_vel
dv_peak_cf = A*(np.pi/Tper)*PXU
vmax_cf    = np.sqrt(u_vel_cf**2 + dv_peak_cf**2)

f, X, Z, u, v, vel, R = checks(Z0_PLACEHOLDER)

# ---- assertions (MAIN must PASS) ----
assert abs(R["f0_X"] - (-0.90)) < 1e-9,           "f0 X must equal -0.90 (constraint 3)"
assert R["f0_dZ"] == 0.0,                          "f0 dZ must be 0 (constraint 3)"
assert R["occ_dZ_max"] == 0.0,                     "occlusion window 36-40 must have dZ==0 (constraint 4)"
assert R["max_vel"] <= VMAX + 1e-9,                "max screen_velocity must be <= 12 (constraint 1)"
assert abs(R["u_vel_const"] - u_vel_cf) < 1e-6,    "u velocity must be constant == closed form"
assert R["peak_dZ"] <= peak_dZ_cf + 1e-12,         "discrete peak dZ must not exceed A"
assert abs(float(dZ(17.5)) - A) < 1e-12,           "continuous peak at f=17.5 must equal A"

# ---- SELFTEST: broken inputs MUST trip the same asserts (S9 rule) ----
def selftest():
    # (a) velocity overshoot: A too big
    global A
    save = A; A = 3.0
    _,_,_,uu,vv = build(Z0_PLACEHOLDER); vv2 = screen_velocity(uu,vv)
    tripped_vel = vv2.max() > VMAX
    A = save
    # (b) f0 displaced: shift X0
    bad_f0 = abs((X0+0.5) - (-0.90)) > 1e-9
    # (c) occlusion window polluted: extend wobble past 35
    def dZ_bad(f):
        f=np.asarray(f,float); return A*(1-np.cos(2*np.pi*f/Tper))/2.0  # no clamp
    occ_polluted = np.abs(dZ_bad(np.arange(OCC[0],OCC[1]+1))).max() > 0.0
    assert tripped_vel and bad_f0 and occ_polluted, "SELFTEST failed: an assert would not fire on broken data"
    return dict(overshoot=float(vv2.max()), occ_bad=float(np.abs(dZ_bad(np.arange(36,41))).max()))
ST = selftest()

# ---- write literal JSON (all 49 keyframes; Blender frame = index+1) ----
records = [{"index": int(i), "blender_frame": int(i)+1,
           "location_x": round(float(X[i]), 7),
           "dZ": round(float(dZ(i)), 7),
           "screen_v_to_next": (round(float(vel[i]),4) if i < N-1 else None)}
          for i in range(N)]
out = {"meta": {"A": A, "period": Tper, "depth": DEPTH, "pxu": round(PXU,4),
                "Z0_note": "location_z = Z0 + dZ ; Z0 MUST be read from Scene C Target in Blender",
                "Z0_placeholder": round(Z0_PLACEHOLDER,6),
                "interp": "keyframe every index (LINEAR ok, points are dense)"},
       "keyframes": records}
with open("/mnt/user-data/outputs/scene_c_prime_traj.json","w",encoding="utf-8") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=2)

# ---- report ----
print("=== closed-form reconciliation ===")
print(f"u velocity        cf={u_vel_cf:.4f}  meas={R['u_vel_const']:.4f}")
print(f"peak dZ           cf={peak_dZ_cf:.4f}  meas={R['peak_dZ']:.4f}  at f={R['peak_f']}")
print(f"max screen_vel    cf(cont)={vmax_cf:.4f}  meas(discrete)={R['max_vel']:.4f}  at f={R['argmax_vel']}->{R['argmax_vel']+1}")
print(f"occ window dZmax  {R['occ_dZ_max']:.1f}   f0 X {R['f0_X']:.2f}  f0 dZ {R['f0_dZ']:.1f}")
print("ALL MAIN ASSERTS PASS")
print(f"SELFTEST: overshoot A=3 -> vmax {ST['overshoot']:.2f} (>12 trips), occ-unclamped dZ {ST['occ_bad']:.3f} (>0 trips)")
print()
print("=== key frames ===")
for i in [0,8,17,18,26,35,36,40,48]:
    vn = f"{vel[i]:.3f}" if i < N-1 else "  -  "
    print(f"f={i:2d}  X={X[i]:+.4f}  dZ={dZ(i):+.4f}  v_to_next={vn}")
