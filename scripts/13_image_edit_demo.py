#!/usr/bin/env python3
"""Phase 3.5b — image_edit demo.

Loads Lance + Wan2.2 VAE (encoder + decoder), edits a source image
according to a text instruction, and saves the result PNG.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/13_image_edit_demo.py \\
        --input-image tests/fixtures/image_edit/edit_img.jpg \\
        --instruction "Remove the hat from the painting." \\
        --lance-weights /Volumes/.../Lance-3B-bf16 \\
        --vae-weights /Volumes/.../Wan22-VAE-bf16/vae.safetensors \\
        --out /tmp/lance_edit_remove_hat.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", type=Path, required=True)
    ap.add_argument("--instruction", required=True,
                    help='Edit instruction, e.g. "Remove the hat from the painting."')
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lance-weights", type=Path, required=True)
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/lance_edit.png"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"┏━━ Phase 3.5b — image_edit demo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ input        : {args.input_image}")
    print(f"┃ instruction  : {args.instruction!r}")
    print(f"┃ output size  : {args.width}×{args.height}")
    print(f"┃ steps        : {args.steps}  cfg={args.cfg_scale}  seed={args.seed}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("\n=== Loading image_edit pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.image_edit import ImageEditPipeline
    pipe = ImageEditPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== Generating ===")
    t0 = time.perf_counter()
    img = pipe.generate(
        input_image=args.input_image,
        instruction=args.instruction,
        height=args.height, width=args.width,
        num_steps=args.steps, cfg_scale=args.cfg_scale,
        seed=args.seed, verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  generated in {elapsed:.1f}s ({elapsed/args.steps:.2f}s/step)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"  saved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
