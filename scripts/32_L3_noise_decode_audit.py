#!/usr/bin/env python3
"""L3 — noise-path decode test (mlx-porting skill pitfall #7 + #10).

Decode pure random Gaussian through the Wan22VAEDecoder at the same
spatial dims as the oracle pass (768×768 × 4 latent frames → 13 video
frames). If the decoded output has:
  - Periodic patterns (stride 2/4/8/16) → spatial-op bug
  - Corner-asymmetric artifacts → likely VAE chunk-boundary or padding bug
  - Color tints (cyan, gray, washed) → groupnorm_eps mismatch
  - Uniform-looking noise → VAE spatial ops are correct; corner-cloud
    residual is upstream of VAE (LLM-side)

This isolates VAE-side bugs from LLM-side bugs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    VAE_WEIGHTS = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors"
    )
    OUT = Path("/tmp/lance_L3_noise_decode")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"┏━━ L3 — VAE noise-path decode audit ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ Decode random Gaussian through Wan22VAEDecoder")
    print(f"┃ Latent dims: T_lat=4, H_lat=48, W_lat=48 (matches 768²×13f)")
    print(f"┃ → Decoded: 13 frames × 768×768")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder, denormalize_latents

    print(f"\n=== Loading VAE decoder ===")
    t0 = time.perf_counter()
    decoder = Wan22VAEDecoder(z_dim=48, dim=160, dec_dim=256)
    saved = mx.load(str(VAE_WEIGHTS))
    dec_state = {
        k: v for k, v in saved.items()
        if k.startswith("decoder.") or k.startswith("conv2.")
    }
    decoder.load_weights(list(dec_state.items()))
    mx.eval(decoder.parameters())
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # Several noise seeds to check determinism + average behavior
    for seed in [42, 43, 7, 13]:
        print(f"\n=== Decode pure-Gaussian latent at seed={seed} ===")
        mx.random.seed(seed)
        # Decoder expects (B, T, H, W, C=48) channels-last — same shape Lance produces
        latent = mx.random.normal((1, 4, 48, 48, 48), loc=0.0, scale=1.0)
        latent = latent.astype(mx.bfloat16)
        # Apply same normalization step as the t2v pipeline does post-flow
        z = denormalize_latents(latent).astype(decoder.conv2.weight.dtype)
        mx.eval(z)
        print(f"  latent shape: {tuple(z.shape)}  dtype={z.dtype}")

        t0 = time.perf_counter()
        decoded = decoder(z)
        mx.eval(decoded)
        print(f"  decoded in {time.perf_counter()-t0:.1f}s  shape={tuple(decoded.shape)}")

        # decoded is (B, T', H', W', 3) per the t2v.py docstring
        decoded_np = np.array(decoded[0].astype(mx.float32))   # (T', H', W', 3)
        # Clamp to [-1, 1] then map to [0, 255]
        decoded_np = np.clip((decoded_np + 1.0) * 127.5, 0, 255).astype(np.uint8)

        print(f"  output stats: shape={decoded_np.shape}")
        print(f"    min={decoded_np.min()}  max={decoded_np.max()}  "
              f"mean={decoded_np.mean():.1f}  std={decoded_np.std():.1f}")

        # Save the midframe + first/last frames for inspection
        for i, label in [(0, "frame00"), (decoded_np.shape[0]//2, "midframe"), (decoded_np.shape[0]-1, "frameN")]:
            Image.fromarray(decoded_np[i]).save(OUT / f"noise_seed{seed}_{label}.png")
        # Filmstrip
        strip = np.concatenate(list(decoded_np), axis=1)  # (H, T*W, 3)
        Image.fromarray(strip).save(OUT / f"noise_seed{seed}_strip.png")
        print(f"  → {OUT}/noise_seed{seed}_strip.png")

        # Spatial-stats check: are different regions of the frame similar?
        mid = decoded_np[decoded_np.shape[0]//2]   # (768, 768, 3)
        h, w = mid.shape[:2]
        regions = {
            "TL": mid[:h//2, :w//2],
            "TR": mid[:h//2, w//2:],
            "BL": mid[h//2:, :w//2],
            "BR": mid[h//2:, w//2:],
        }
        print(f"  midframe quadrant stats (mean, std):")
        for name, region in regions.items():
            print(f"    {name}: mean={region.mean():.1f}  std={region.std():.1f}")

    print(f"\n┏━━ Done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ Inspect: {OUT}/noise_seed*_*.png")
    print(f"┃ Look for:")
    print(f"┃   - Periodic patterns at stride 2/4/8/16 → spatial-op bug")
    print(f"┃   - Corner-asymmetric artifacts → padding/cache-boundary bug")
    print(f"┃   - Quadrant stats deviation > 5% → asymmetry suggests issue")
    print(f"┃   - Uniform noise → VAE is fine; corner-clouds = LLM-side")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
