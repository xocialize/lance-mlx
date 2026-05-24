#!/usr/bin/env python3
"""Phase 5n / D1 — VAE T=1 vs T=N noise decode comparison.

Tests whether the Wan2.2 VAE decoder produces materially different output
for T_latent=1 (image-mode input) vs T_latent>1 (video-mode input) when
the first-frame latent content is byte-identical.

The two pipelines (t2i, t2v) share THE SAME 48-ch Wan2.2 VAE weights but
feed it different temporal shapes. The 3D causal temporal conv has a
strictly-causal kernel — output frame 0 should depend ONLY on latent
frame 0, so T=1-alone and T=5-frame-0 SHOULD be byte-identical if the
conv truly has no lookahead and no global temporal normalization.

If they differ materially:
  - VAE has hidden temporal-mode dependency (e.g. attention over time,
    instance-norm over time axis, padding asymmetry at t=0).
  - That asymmetry is a plausible contributor to the image-vs-video
    quality gap — Lance's image path is essentially asking the VAE to
    decode at T=1 forever, which may be a less-trained regime.

If they are byte-identical:
  - VAE is mode-agnostic. Eliminate this candidate. The quality gap
    lives elsewhere (mrope, CFG renorm, etc.).

Cost: ~15s. No LLM load needed, just the standalone VAE.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VAE_SAFETENSORS = REPO_ROOT.parent / "lance-mlx-models" / "Wan22-VAE-bf16" / "vae.safetensors"
OUT_DIR = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d1_vae_temporal_mode"


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


def fft_high_freq_energy(img_hwc: np.ndarray) -> float:
    """Sum of FFT magnitudes outside the central low-freq region (sharpness proxy)."""
    gray = img_hwc.mean(axis=-1) if img_hwc.shape[-1] == 3 else img_hwc
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8           # mask central low-freq disk
    mask = np.ones_like(mag)
    mask[cy - r:cy + r, cx - r:cx + r] = 0
    return float((mag * mask).sum())


def stats(name, arr_mx) -> np.ndarray:
    arr_np = np.array(arr_mx.astype(mx.float32))
    print(f"  {name:30s}  shape={tuple(arr_mx.shape)}  "
          f"mean={arr_np.mean():+.4f}  std={arr_np.std():.4f}  "
          f"min={arr_np.min():+.4f}  max={arr_np.max():+.4f}")
    return arr_np


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h_lat = w_lat = 24       # → 384² output frames (small, fast)
    z_dim = 48
    seed = 42

    print(f"=== Phase 5n / D1 — VAE temporal-mode comparison ===")
    print(f"  VAE weights: {VAE_SAFETENSORS}")
    print(f"  latent grid: {h_lat}×{w_lat} × {z_dim}ch  (→ {h_lat*16}² output)")
    print(f"  seed: {seed}")
    print(f"  out:  {OUT_DIR}\n")

    t0 = time.perf_counter()
    print(f"Loading Wan2.2 VAE decoder ...")
    vae = load_vae()
    dtype = vae.conv2.weight.dtype
    print(f"  loaded in {time.perf_counter()-t0:.1f}s; decoder dtype={dtype}\n")

    # Build a T=5 noise tensor; slice frame 0 for the T=1 case — guarantees
    # byte-identical first-frame content for the two decodes.
    mx.random.seed(seed)
    noise_t5 = mx.random.normal((1, 5, h_lat, w_lat, z_dim)).astype(dtype)
    noise_t1 = noise_t5[:, 0:1, :, :, :]

    stats("noise_t1 (sliced)", noise_t1)
    stats("noise_t5 (full)",   noise_t5)
    print()

    print(f"Decoding T=1 input ...")
    t0 = time.perf_counter()
    out_t1 = vae(noise_t1); mx.eval(out_t1)
    dt_t1 = time.perf_counter() - t0
    print(f"  decoded in {dt_t1:.2f}s")
    stats("decoded T=1 output", out_t1)

    print(f"\nDecoding T=5 input ...")
    t0 = time.perf_counter()
    out_t5 = vae(noise_t5); mx.eval(out_t5)
    dt_t5 = time.perf_counter() - t0
    print(f"  decoded in {dt_t5:.2f}s")
    stats("decoded T=5 output", out_t5)
    print()

    # Frame-0 comparison — the critical test.
    f0_t1 = np.array(out_t1[0, 0].astype(mx.float32))   # (H, W, 3)
    f0_t5 = np.array(out_t5[0, 0].astype(mx.float32))

    print(f"=== Frame-0 comparison (identical first-frame latent input) ===")
    diff = np.abs(f0_t1 - f0_t5)
    print(f"  shapes:           T=1 {f0_t1.shape}   T=5_f0 {f0_t5.shape}")
    print(f"  per-frame mean:   T=1 {f0_t1.mean():+.4f}   T=5_f0 {f0_t5.mean():+.4f}")
    print(f"  per-frame std:    T=1 {f0_t1.std():.4f}    T=5_f0 {f0_t5.std():.4f}")
    print(f"  abs diff:         mean={diff.mean():.4f}  max={diff.max():.4f}  "
          f"95p={np.percentile(diff, 95):.4f}")

    hf_t1 = fft_high_freq_energy(f0_t1)
    hf_t5_f0 = fft_high_freq_energy(f0_t5)
    print(f"  FFT high-freq E:  T=1 {hf_t1:.2e}   T=5_f0 {hf_t5_f0:.2e}   "
          f"ratio {hf_t5_f0/hf_t1:+.4f}")

    # Compare across T=5 frames to see intra-T temporal smoothing.
    if out_t5.shape[1] > 4:
        f4_t5 = np.array(out_t5[0, 4].astype(mx.float32))   # next latent boundary
        hf_t5_f4 = fft_high_freq_energy(f4_t5)
        f_last = np.array(out_t5[0, -1].astype(mx.float32))
        hf_t5_last = fft_high_freq_energy(f_last)
        print(f"  Within T=5:  f0 std={f0_t5.std():.4f}  "
              f"f4 std={f4_t5.std():.4f}  "
              f"f{out_t5.shape[1]-1} std={f_last.std():.4f}")
        print(f"  Within T=5:  f0 HF={hf_t5_f0:.2e}  "
              f"f4 HF={hf_t5_f4:.2e}  "
              f"f{out_t5.shape[1]-1} HF={hf_t5_last:.2e}")

    # Save images for visual inspection.
    from PIL import Image
    def to_u8(arr):
        return Image.fromarray(((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8))

    to_u8(f0_t1).save(OUT_DIR / "f0_T1_alone.png")
    to_u8(f0_t5).save(OUT_DIR / "f0_T5_frame0.png")
    if out_t5.shape[1] > 4:
        to_u8(f4_t5).save(OUT_DIR / "f4_T5.png")
        to_u8(f_last).save(OUT_DIR / f"flast_T5.png")
    # Difference image, gain-normalized.
    diff_gray = diff.mean(axis=-1)
    diff_u8 = (diff_gray / max(diff_gray.max(), 1e-6) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(diff_u8, mode='L').save(OUT_DIR / "abs_diff_T1_vs_T5frame0.png")

    print(f"\n=== Verdict ===")
    pct_max = diff.max() / max(f0_t1.max() - f0_t1.min(), 1e-6) * 100
    pct_mean = diff.mean() / max(f0_t1.std(), 1e-6) * 100
    if diff.max() < 1e-4:
        verdict = "BYTE-IDENTICAL"
        recommendation = "VAE is strictly mode-agnostic. ELIMINATE Candidate 2."
    elif diff.mean() < 0.01 and abs(hf_t5_f0 / hf_t1 - 1) < 0.05:
        verdict = "NUMERICALLY EQUIVALENT"
        recommendation = "VAE differs only by rounding error. ELIMINATE Candidate 2."
    elif hf_t5_f0 > 1.2 * hf_t1:
        verdict = "T=5 frame-0 HAS MORE DETAIL"
        recommendation = ("VAE prefers temporal context. Test t2i with fake "
                          "t_lat=2 input as candidate fix.")
    elif hf_t5_f0 < 0.8 * hf_t1:
        verdict = "T=5 frame-0 HAS LESS DETAIL"
        recommendation = ("Inverse hypothesis: VAE degrades over time. Could "
                          "explain video < image quality.")
    else:
        verdict = f"DIFFERENT BUT AMBIGUOUS (max diff {pct_max:.1f}% of range)"
        recommendation = "Inspect saved PNGs for visual judgment."

    print(f"  → {verdict}")
    print(f"  → max diff = {pct_max:.2f}% of pixel range, mean = {pct_mean:.2f}% of σ")
    print(f"  → {recommendation}")
    print(f"\nSaved to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
