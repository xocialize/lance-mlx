#!/usr/bin/env python3
"""Phase 5n / D1b — VAE frame-indexing follow-up.

D1 showed T_latent=1 produces T_decoded=3 frames (not 1 as the spec
`T'=(T-1)*4+1` predicts). t2i.py:286 grabs decoded[0, 0]. Question:
is frame 0 actually the "real" decoded frame, or is it a causal-padding
build-up artifact?

If frame 0 of T_latent=1 has noticeably LESS detail than frames 1/2,
t2i has been silently picking the wrong output frame and quality has
been left on the table this entire time.

Also expand to T_latent=2, 3 to see how T_decoded scales for small T.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VAE_SAFETENSORS = REPO_ROOT.parent / "lance-mlx-models" / "Wan22-VAE-bf16" / "vae.safetensors"
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d1b_vae_frame_indexing"


def load_vae():
    from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder
    vae = Wan22VAEDecoder(z_dim=48, dim=160, dec_dim=256)
    saved = mx.load(str(VAE_SAFETENSORS))
    dec_state = {
        k: v for k, v in saved.items()
        if k.startswith("decoder.") or k.startswith("conv2.")
    }
    vae.load_weights(list(dec_state.items()))
    mx.eval(vae.parameters())
    return vae


def fft_hf(img_hwc):
    gray = img_hwc.mean(axis=-1)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.ones_like(mag)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    return float((mag * mask).sum())


def per_frame_stats(out_mx, label):
    out = np.array(out_mx[0].astype(mx.float32))   # (T', H, W, 3)
    T = out.shape[0]
    print(f"\n  --- {label}: T_decoded={T} ---")
    print(f"  {'frame':>5s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}  {'FFT_HF':>10s}")
    rows = []
    for t in range(T):
        f = out[t]
        rows.append((t, f.mean(), f.std(), f.min(), f.max(), fft_hf(f)))
        print(f"  {t:>5d}  {f.mean():+.4f}  {f.std():.4f}  "
              f"{f.min():+.4f}  {f.max():+.4f}  {fft_hf(f):.2e}")
    return out, rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h_lat = w_lat = 24
    z_dim = 48
    seed = 42

    print(f"=== Phase 5n / D1b — VAE frame-indexing scan ===\n")
    print(f"Loading Wan2.2 VAE decoder ...")
    t0 = time.perf_counter()
    vae = load_vae()
    dtype = vae.conv2.weight.dtype
    print(f"  loaded in {time.perf_counter()-t0:.1f}s; dtype={dtype}")

    # Build T=5 noise; reuse leading slices for the smaller T tests so
    # frame-0 latent content is identical across all runs.
    mx.random.seed(seed)
    base_t5 = mx.random.normal((1, 5, h_lat, w_lat, z_dim)).astype(dtype)

    cases = [
        ("T_latent=1", base_t5[:, 0:1, :, :, :]),
        ("T_latent=2", base_t5[:, 0:2, :, :, :]),
        ("T_latent=3", base_t5[:, 0:3, :, :, :]),
        ("T_latent=5", base_t5),
    ]

    all_results = {}
    from PIL import Image
    def to_u8(arr):
        return Image.fromarray(((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8))

    for label, z in cases:
        print(f"\nDecoding {label} ...  input shape={tuple(z.shape)}")
        t0 = time.perf_counter()
        out = vae(z); mx.eval(out)
        print(f"  decoded in {time.perf_counter()-t0:.2f}s")
        decoded, rows = per_frame_stats(out, label)
        all_results[label] = (decoded, rows)
        # Save each frame.
        sub = OUT_DIR / label.replace("=", "")
        sub.mkdir(exist_ok=True)
        for t in range(decoded.shape[0]):
            to_u8(decoded[t]).save(sub / f"f{t:02d}.png")

    # --- cross-T comparison: which T_latent=1 frame matches T_latent=5 frame 0? ---
    print(f"\n=== Cross-T frame-match: best alignment between T_latent=1 frames and T_latent=5 frame 0 ===")
    out_t1, _ = all_results["T_latent=1"]
    out_t5, _ = all_results["T_latent=5"]
    ref = out_t5[0]   # T_latent=5 frame 0 — assumed "real" decode of latent frame 0
    print(f"  ref = T_latent=5 frame 0,  std={ref.std():.4f}, HF={fft_hf(ref):.2e}")
    for t in range(out_t1.shape[0]):
        f = out_t1[t]
        diff = np.abs(f - ref)
        print(f"  T_latent=1 frame {t}:  "
              f"abs_diff mean={diff.mean():.4f}  max={diff.max():.4f}  "
              f"FFT_HF={fft_hf(f):.2e}")

    # Compare per-frame across all T values: how does "frame 0" detail evolve?
    print(f"\n=== Frame-0 detail across T_latent values ===")
    print(f"  {'T_latent':<10s}  {'T_decoded':<10s}  {'f0 std':<10s}  {'f0 FFT_HF':<12s}  {'f0 mean':<10s}")
    for label, (decoded, rows) in all_results.items():
        f0 = decoded[0]
        print(f"  {label:<10s}  {decoded.shape[0]:<10d}  "
              f"{f0.std():<10.4f}  {fft_hf(f0):<12.2e}  {f0.mean():<+10.4f}")

    print(f"\nSaved per-T per-frame PNGs to: {OUT_DIR}")
    print(f"\n=== Interpretation guide ===")
    print(f"  - If T_latent=1 frame 0 looks WASHED OUT compared to a later")
    print(f"    frame from the same decode, t2i.py:286 is picking the wrong index.")
    print(f"  - If T_latent=1 frame 0 ≈ T_latent=5 frame 0 visually, VAE is")
    print(f"    mode-agnostic on noise and the high-freq diff is just stochastic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
