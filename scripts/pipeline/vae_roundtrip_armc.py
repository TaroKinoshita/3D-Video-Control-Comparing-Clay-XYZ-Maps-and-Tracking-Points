r"""
vae_roundtrip_armc.py  —  Phase 0.0 本番（Arm C / RunPod）

armC/<variant>/armC_%04d.png（index 0..48）を Wan VAE に通し、復元を .npy で保存する。
採点はしない（score_armc.py の担当）。

W1 の vae_roundtrip_video.py からの変更
  1. 入力を manifest + speed ループ → variant ループに変更（manifest 不要）
  2. ファイル名 frame_*.png → armC_%04d.png
  3. PNG 読みを PIL → png_io.read_rgb に統一（§14。経路を1本にする）
  4. 出力名 v{V}_{mode}.npy → {variant}_{mode}.npy
  5. peak VRAM の取得 API を実測で選び、どちらを使ったかログに残す（protocol §23 の矛盾）
  6. 入出力の md5 をログに残す（転送検証用。§19）

モード
  temporal  : 49フレーム → 潜在13枚 → 49フレーム（判定に使う）
  framewise : 1フレーム = 潜在1枚。時間圧縮なしの対照条件（§16.9）

使い方
  python vae_roundtrip_armc.py --armc armC --out vae_armc --pngio . --fake-vae
  python vae_roundtrip_armc.py --armc armC --out vae_armc --pngio .
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

N_FRAMES = 49
WIDTH = 832
HEIGHT = 480


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_vae_path(hint=None):
    if hint:
        if not os.path.exists(hint):
            raise FileNotFoundError("指定された VAE が無い: %s" % hint)
        return hint
    roots = ["/workspace/modelscope", "/workspace",
             os.path.expanduser("~/.cache/modelscope")]
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "**", "Wan2.1_VAE.pth"),
                                recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Wan2.1_VAE.pth が見つからない。--vae で明示する。\n"
        "  find /workspace -name 'Wan2.1_VAE.pth'")


def load_vae(path, device, torch_dtype):
    from diffsynth.models.wan_video_vae import WanVideoVAE
    from diffsynth.core.loader.model import load_model
    from diffsynth.utils.state_dict_converters.wan_video_vae import (
        WanVideoVAEStateDictConverter)

    vae = load_model(WanVideoVAE, path, torch_dtype=torch_dtype, device=device,
                     state_dict_converter=WanVideoVAEStateDictConverter)
    # mean / std は nn.Module のバッファではないので .to() で移動しない（§23）
    vae.mean = vae.mean.to(device=device, dtype=torch_dtype)
    vae.std = vae.std.to(device=device, dtype=torch_dtype)
    vae.scale = [vae.mean, 1.0 / vae.std]
    return vae


def load_frames(read_rgb, folder, size, prefix="armC_"):
    """<prefix>%04d.png 0..48 → uint8 (T,H,W,3)。欠番は例外。

    prefix の既定は armC_（Phase 0.0 と挙動を変えない）。
    §18-4 の k スイープでは armB_ を渡す。
    """
    frames = []
    for i in range(N_FRAMES):
        p = os.path.join(folder, "%s%04d.png" % (prefix, i))
        px, w, h = read_rgb(p)
        if (w, h) != size:
            raise ValueError("%s は %dx%d（期待 %dx%d）" % (p, w, h, size[0], size[1]))
        frames.append(np.frombuffer(px, dtype=np.uint8).reshape(h, w, 3))
    return np.stack(frames)


def to_tensor(arr, device, torch_dtype):
    """uint8 (T,H,W,3) → float (1,3,T,H,W) を [-1,1] で（§23）"""
    import torch
    x = torch.from_numpy(arr.copy()).to(device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0)
    return x.to(torch_dtype) / 127.5 - 1.0


def to_uint8(x):
    """float (1,3,T,H,W) in [-1,1] → uint8 (T,H,W,3)。round（§28.2 と揃える）"""
    y = (x[0].float().clamp(-1, 1) + 1.0) * 127.5
    y = y.permute(1, 2, 3, 0).round().clamp(0, 255).to("cpu").numpy()
    return y.astype(np.uint8)


def peak_mib(device):
    """protocol §23 は max_memory_stats が無いと書いているが実測で選ぶ。"""
    import torch
    if hasattr(torch.cuda, "max_memory_stats"):
        try:
            v = torch.cuda.max_memory_stats(device)["allocated_bytes.all.peak"]
            return v / 2 ** 20, "max_memory_stats"
        except Exception:
            pass
    return torch.cuda.max_memory_allocated(device) / 2 ** 20, "max_memory_allocated"


def eps_stats(out, arr):
    """往復前後の 8bit 差分 ε の統計。§18-4 用。GT は使わない。

    out / arr : uint8 (T,H,W,3)
    ε が k で変わるかを見るのが目的なので、分位点を必ず出す
    （平均だけでは保証ラインの破れを見落とす。§18-6 の教訓）。
    """
    d = np.abs(out.astype(np.int16) - arr.astype(np.int16))
    st = {}
    flat = d.reshape(-1)
    st["overall"] = {"mean": float(flat.mean()),
                     "p50": float(np.percentile(flat, 50)),
                     "p95": float(np.percentile(flat, 95)),
                     "p99": float(np.percentile(flat, 99)),
                     "max": int(flat.max())}
    for ci, cn in enumerate("RGB"):
        e = d[:, :, :, ci].reshape(-1)
        st[cn] = {"mean": float(e.mean()),
                  "p99": float(np.percentile(e, 99)),
                  "max": int(e.max())}
    st["frac_gt"] = {str(t): float((flat > t).mean()) for t in (0, 1, 2, 4, 8, 16)}
    # 帯（127.5 の多数決）が反転した画素。k に依存しないはずの量
    b_in = (arr.astype(np.int16) > 127).sum(axis=3) >= 2
    b_out = (out.astype(np.int16) > 127).sum(axis=3) >= 2
    st["band_flips_127_5"] = int((b_in != b_out).sum())
    # 予約帯（124-131）に落ちた画素
    st["reserved_124_131"] = int(((out >= 124) & (out <= 131)).any(axis=3).sum())
    return st


def run_pass(vae, video, mode, device, fake):
    import torch
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        if mode == "temporal":
            lat = vae.encode([video[0]], device=device, tiled=False)
            out = vae.decode(lat, device=device, tiled=False)
        elif mode == "framewise":
            lat = vae.encode_framewise(video, device=device)
            out = vae.decode_framewise(lat, device=device)
        else:
            raise ValueError(mode)
    torch.cuda.synchronize()
    sec = time.time() - t0
    peak, api = peak_mib(device)
    return to_uint8(out), list(lat.shape), peak, sec, api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--armc", required=True, help="armC ディレクトリ（配下に variant）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=["original", "edited"])
    ap.add_argument("--modes", nargs="*", default=["temporal", "framewise"])
    ap.add_argument("--prefix", default="armC_",
                    help="PNG のファイル名 prefix。§18-4 では armB_")
    ap.add_argument("--pngio", default=None, help="png_io.py のあるディレクトリ")
    ap.add_argument("--vae", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--expect-size", default="832x480")
    ap.add_argument("--fake-vae", action="store_true",
                    help="VAE を通さず入力をそのまま出す。配管確認用（CPU可）")
    a = ap.parse_args()

    sys.path.insert(0, a.pngio or os.path.dirname(os.path.abspath(__file__)))
    from png_io import read_rgb

    size = tuple(int(x) for x in a.expect_size.lower().split("x"))
    os.makedirs(a.out, exist_ok=True)

    if a.fake_vae:
        device, dt, vae = "cpu", None, None
        print("!! fake-vae モード。VAE も torch も通していない !!")
    else:
        import torch
        assert torch.cuda.is_available(), "GPU が見えていない"
        device = "cuda"
        dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
        vp = find_vae_path(a.vae)
        print("VAE   : %s" % vp)
        print("dtype : %s / device: %s" % (a.dtype, torch.cuda.get_device_name(0)))
        vae = load_vae(vp, device, dt)

    log = []
    for variant in a.variants:
        vdir = os.path.join(a.armc, variant)
        arr = load_frames(read_rgb, vdir, size, prefix=a.prefix)
        in_md5 = hashlib.md5(arr.tobytes()).hexdigest()
        print("\n%s : %s  pixels-md5 %s" % (variant, arr.shape, in_md5[:12]))
        video = None if a.fake_vae else to_tensor(arr, device, dt)

        for mode in a.modes:
            r = {"variant": variant, "mode": mode, "dtype": a.dtype,
                 "input_shape": list(arr.shape), "input_pixels_md5": in_md5}
            try:
                if a.fake_vae:
                    out, lshape, peak, sec, api = arr.copy(), [], 0.0, 0.0, "fake"
                else:
                    out, lshape, peak, sec, api = run_pass(vae, video, mode,
                                                          device, a.fake_vae)
            except RuntimeError as e:
                r["error"] = str(e)[:300]
                log.append(r)
                print("  %-9s 失敗: %s" % (mode, str(e)[:120]))
                continue

            assert out.shape == arr.shape, "出力 shape 不一致: %s" % (out.shape,)
            name = "%s_%s.npy" % (variant, mode)
            op = os.path.join(a.out, name)
            np.save(op, out)
            n_changed = int((out != arr).any(axis=3).sum())

            r.update({"latent_shape": lshape, "peak_vram_mib": round(peak, 1),
                      "vram_api": api, "seconds": round(sec, 2),
                      "output": name, "output_md5": md5_of(op),
                      "changed_pixels": n_changed,
                      "eps": eps_stats(out, arr)})
            log.append(r)
            e = r["eps"]["overall"]
            print("  %-9s 潜在%-22s VRAM %8.0f MiB  %6.1fs  変化画素 %d"
                  % (mode, lshape, peak, sec, n_changed))
            print("            eps mean %.4f  p95 %.1f  p99 %.1f  max %d"
                  "  帯反転 %d  予約帯 %d"
                  % (e["mean"], e["p95"], e["p99"], e["max"],
                     r["eps"]["band_flips_127_5"], r["eps"]["reserved_124_131"]))
        del video

    jp = os.path.join(a.out, "roundtrip_armc_log.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({"tiled": False, "dtype": a.dtype, "fake_vae": a.fake_vae,
                   "prefix": a.prefix,
                   "n_frames": N_FRAMES, "width": size[0], "height": size[1],
                   "passes": log}, f, indent=2, ensure_ascii=False)
    print("\n保存: %s\nログ: %s" % (a.out, jp))


if __name__ == "__main__":
    main()
