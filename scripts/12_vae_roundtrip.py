#!/usr/bin/env python3
"""Phase 3.5a — Wan2.2 VAE encoder load + roundtrip smoke test.

Loads BOTH encoder and decoder from our converted bf16 safetensors, encodes
a PIL image at 768×768, decodes back, and reports per-pixel MAD vs the
input. A clean roundtrip means the encoder loads correctly and our latent
normalization is consistent with the decoder.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/12_vae_roundtrip.py \\
        --image tests/fixtures/image_edit/edit_img.jpg \\
        --vae-weights /Volumes/.../Wan22-VAE-bf16/vae.safetensors
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_video.models.wan_2.vae22 import (
    VAE22_MEAN,
    VAE22_STD,
    Wan22VAEDecoder,
    Wan22VAEEncoder,
    denormalize_latents,
)
from PIL import Image


def load_image_as_tensor(path: Path, size: int = 768) -> mx.array:
    """Load PNG/JPG → (1, T=1, H, W, 3) in [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0          # [-1, 1]
    arr = arr[None, None, ...]                                      # (1, 1, H, W, 3)
    return mx.array(arr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/lance_vae_roundtrip"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("┏━━ Phase 3.5a — Wan2.2 VAE roundtrip ━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ image    : {args.image}")
    print(f"┃ vae      : {args.vae_weights}")
    print(f"┃ size     : {args.size}×{args.size}")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # --- Load encoder + decoder ---------------------------------------------
    print("\n=== Loading VAE encoder + decoder ===")
    t0 = time.perf_counter()
    saved = mx.load(str(args.vae_weights))

    encoder = Wan22VAEEncoder(z_dim=48, dim=160)
    enc_state = {
        k: v for k, v in saved.items()
        if k.startswith("encoder.") or k.startswith("conv1.")
    }
    print(f"  encoder keys: {len(enc_state)}")
    encoder.load_weights(list(enc_state.items()))
    mx.eval(encoder.parameters())

    decoder = Wan22VAEDecoder(z_dim=48, dim=160, dec_dim=256)
    dec_state = {
        k: v for k, v in saved.items()
        if k.startswith("decoder.") or k.startswith("conv2.")
    }
    print(f"  decoder keys: {len(dec_state)}")
    decoder.load_weights(list(dec_state.items()))
    mx.eval(decoder.parameters())
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # --- Load + encode image ------------------------------------------------
    print(f"\n=== Encoding {args.image.name} ===")
    img_t = load_image_as_tensor(args.image, size=args.size)
    print(f"  input shape: {tuple(img_t.shape)}  dtype={img_t.dtype}")
    t0 = time.perf_counter()
    mu = encoder(img_t)
    mx.eval(mu)
    enc_t = time.perf_counter() - t0
    print(f"  latent shape: {tuple(mu.shape)}  dtype={mu.dtype}")
    print(f"  latent mean={float(mx.mean(mu)):.4f}  std={float(mx.std(mu)):.4f}")
    print(f"  encode time: {enc_t:.2f}s")

    # --- Decode back --------------------------------------------------------
    print(f"\n=== Decoding back ===")
    t0 = time.perf_counter()
    z = denormalize_latents(mu).astype(decoder.conv2.weight.dtype)
    decoded = decoder(z)
    mx.eval(decoded)
    dec_t = time.perf_counter() - t0
    print(f"  decoded shape: {tuple(decoded.shape)}")
    print(f"  decode time: {dec_t:.2f}s")

    # --- MAD vs input -------------------------------------------------------
    # Decoded shape is (1, T'≥1, H, W, 3); take frame 0.
    dec_img = decoded[0, 0]                                          # (H, W, 3)
    inp_img = img_t[0, 0]                                            # (H, W, 3)
    mad = float(mx.mean(mx.abs(dec_img - inp_img)))
    max_err = float(mx.max(mx.abs(dec_img - inp_img)))
    print(f"\n=== Roundtrip quality ===")
    print(f"  per-pixel MAD ([-1,1] domain): {mad:.4f}")
    print(f"  max abs error: {max_err:.4f}")
    # In u8 domain: mad * 127.5
    print(f"  per-pixel MAD ([0,255] domain): {mad * 127.5:.2f} / 255")

    # --- Save outputs -------------------------------------------------------
    def to_pil(t: mx.array) -> Image.Image:
        arr = np.asarray(t.astype(mx.float32)) if hasattr(t, "astype") else np.asarray(t)
        u8 = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(u8)

    Image.fromarray((((np.asarray(img_t[0, 0].astype(mx.float32))) + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(args.out_dir / "input.png")
    to_pil(dec_img).save(args.out_dir / "roundtrip.png")
    print(f"\n  saved input + roundtrip PNGs to {args.out_dir}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
