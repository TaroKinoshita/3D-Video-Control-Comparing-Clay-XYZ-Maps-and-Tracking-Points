"""
Wan VAE 往復スクリプト（動画版・GPU）

make_dots_video.py が出したPNG列をVAEに通し、復元結果を .npy で保存する。
採点はしない（score_video.py の担当）。

2つのモードを回す:
    temporal  : 通常。49フレーム → 潜在13枚 → 49フレーム（時間4:1圧縮あり）
    framewise : encode_framewise。1フレーム = 潜在1枚（時間圧縮なし）の対照条件

tiling は常に無効。tiled=False は DiffSynth 2.0.18 の既定値だが明示的に渡す。

使い方:
    python vae_roundtrip_video.py --in vae_test_video --out vae_out
    python vae_roundtrip_video.py --in vae_test_video --out vae_out --speeds 0
    python vae_roundtrip_video.py --in vae_test_video --out vae_out --fake-vae   # 配管確認のみ
"""

import argparse
import glob
import json
import os
import time

import numpy as np
from PIL import Image


def find_vae_path(hint=None):
    """Wan2.1_VAE.pth を探す"""
    if hint:
        if not os.path.exists(hint):
            raise FileNotFoundError(f"指定されたVAEが無い: {hint}")
        return hint
    roots = ["/workspace/modelscope", "/workspace", os.path.expanduser("~/.cache/modelscope")]
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "**", "Wan2.1_VAE.pth"), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Wan2.1_VAE.pth が見つからない。--vae で明示的に指定する。\n"
        "  find /workspace -name 'Wan2.1_VAE.pth'")


def load_vae(path, device, torch_dtype):
    import torch
    from diffsynth.models.wan_video_vae import WanVideoVAE
    from diffsynth.core.loader.model import load_model
    from diffsynth.utils.state_dict_converters.wan_video_vae import (
        WanVideoVAEStateDictConverter)

    vae = load_model(WanVideoVAE, path, torch_dtype=torch_dtype, device=device,
                     state_dict_converter=WanVideoVAEStateDictConverter)

    # mean / std は nn.Module のバッファではないので .to() で移動しない。
    # 明示的に移さないと single_encode で device mismatch になる。
    vae.mean = vae.mean.to(device=device, dtype=torch_dtype)
    vae.std = vae.std.to(device=device, dtype=torch_dtype)
    vae.scale = [vae.mean, 1.0 / vae.std]
    return vae


def load_frames(folder):
    """PNG列 → uint8 (T, H, W, 3)"""
    files = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
    if not files:
        raise FileNotFoundError(f"PNGが無い: {folder}")
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    return arr


def to_tensor(arr, device, torch_dtype):
    """uint8 (T,H,W,3) → float (1,3,T,H,W) を [-1,1] で"""
    import torch
    x = torch.from_numpy(arr).to(device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0)           # (1,3,T,H,W)
    x = x.to(torch_dtype) / 127.5 - 1.0
    return x


def to_uint8(x):
    """float (1,3,T,H,W) in [-1,1] → uint8 (T,H,W,3)"""
    x = (x[0].float().clamp(-1, 1) + 1.0) * 127.5
    x = x.permute(1, 2, 3, 0).round().clamp(0, 255).to("cpu").numpy()
    return x.astype(np.uint8)


def run_pass(vae, video, mode, device, fake):
    """1本ぶんの往復。戻り値は (復元uint8, 潜在shape, peak VRAM MiB, 秒)"""
    import torch

    if fake:
        return to_uint8(video), list(video.shape), 0.0, 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        if mode == "temporal":
            lat = vae.encode([video[0]], device=device, tiled=False)
            out = vae.decode(lat, device=device, tiled=False)
        elif mode == "framewise":
            # encode_framewise はリストではなくテンソル (B,C,T,H,W) を取る
            lat = vae.encode_framewise(video, device=device)
            out = vae.decode_framewise(lat, device=device)
        else:
            raise ValueError(mode)

    torch.cuda.synchronize()
    sec = time.time() - t0
    peak = torch.cuda.max_memory_stats(device)["allocated_bytes.all.peak"] / 2 ** 20
    return to_uint8(out), list(lat.shape), peak, sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="vae_test_video")
    ap.add_argument("--out", dest="outdir", default="vae_out")
    ap.add_argument("--speeds", nargs="*", type=int, default=None)
    ap.add_argument("--modes", nargs="*", default=["temporal", "framewise"])
    ap.add_argument("--vae", default=None, help="Wan2.1_VAE.pth のパス")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--fake-vae", action="store_true",
                    help="VAEを通さず入力をそのまま出す。配管確認用（CPUで動く）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join(args.indir, "manifest.json"), encoding="utf-8") as f:
        gen = json.load(f)
    speeds = args.speeds if args.speeds is not None else [
        s["speed_px_per_frame"] for s in gen["speeds"]]

    if args.fake_vae:
        import torch
        device, dt, vae = "cpu", torch.float32, None
        print("!! fake-vae モード。VAEは通していない !!")
    else:
        import torch
        assert torch.cuda.is_available(), "GPUが見えていない"
        device = "cuda"
        dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        vp = find_vae_path(args.vae)
        print(f"VAE: {vp}\ndtype: {args.dtype} / device: "
              f"{torch.cuda.get_device_name(0)}")
        vae = load_vae(vp, device, dt)

    log = []
    for V in speeds:
        arr = load_frames(os.path.join(args.indir, f"v{V}"))
        assert arr.shape[0] == gen["n_frames"], f"フレーム数不一致: {arr.shape[0]}"
        video = to_tensor(arr, device, dt)

        for mode in args.modes:
            rec = {"speed": V, "mode": mode, "dtype": args.dtype,
                   "input_shape": list(arr.shape)}
            try:
                out, lshape, peak, sec = run_pass(vae, video, mode, device,
                                                  args.fake_vae)
            except RuntimeError as e:
                rec["error"] = str(e)[:300]
                log.append(rec)
                print(f"  V={V:>2} {mode:<9} 失敗: {str(e)[:120]}")
                continue

            assert out.shape == arr.shape, f"出力shape不一致: {out.shape}"
            name = f"v{V}_{mode}.npy"
            np.save(os.path.join(args.outdir, name), out)

            rec.update({"latent_shape": lshape, "peak_vram_mib": round(peak, 1),
                        "seconds": round(sec, 2), "output": name})
            log.append(rec)
            print(f"  V={V:>2} {mode:<9} 潜在{lshape}  "
                  f"VRAM {peak:>8.0f} MiB  {sec:>6.1f}s")

        del video

    out_json = os.path.join(args.outdir, "roundtrip_log.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"tiled": False, "dtype": args.dtype, "passes": log},
                  f, indent=2, ensure_ascii=False)
    print(f"\n保存: {args.outdir}  ログ: {out_json}")


if __name__ == "__main__":
    main()
