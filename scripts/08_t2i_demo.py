#!/usr/bin/env python3
"""Phase 3b — first text-to-image generation on Lance MLX.

Generates a single image from a prompt using the LanceModel + Wan2.2 VAE
+ flow-matching Euler. No CFG in v1 (Phase 3c follow-up).

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/08_t2i_demo.py \\
        --prompt "A cat holds a poster with rainbow text 'STOP'" \\
        --lance-weights /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16 \\
        --vae-weights   /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors \\
        --output-png    /tmp/lance_t2i.png \\
        --seed 42 --steps 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A cat holds a poster with rainbow text 'STOP'")
    ap.add_argument("--lance-weights", type=Path, required=True)
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--output-png", type=Path, default=Path("/tmp/lance_t2i.png"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0,
                    help="Classifier-free guidance scale. 1.0 disables CFG. "
                         "Lance default is 4.0.")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--scheduler", default="euler", choices=["euler", "dpm"],
                    help="ODE solver. 'euler' = default 30-step Euler. "
                         "'dpm' = DPM-Solver++(2M), ~2.4x faster at ~12 steps.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"=== Loading TextToImagePipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    t1 = time.perf_counter()
    print(f"  loaded in {t1-t0:.1f}s")

    print(f"\n=== Generating ===")
    print(f"  prompt: {args.prompt!r}")
    print(f"  {args.width}x{args.height}, {args.steps} steps, "
          f"scheduler={args.scheduler}, cfg={args.cfg_scale}, seed={args.seed}")
    t0 = time.perf_counter()
    img = pipe.generate(
        args.prompt,
        height=args.height, width=args.width,
        num_steps=args.steps, cfg_scale=args.cfg_scale,
        seed=args.seed, verbose=args.verbose,
        scheduler=args.scheduler,
    )
    t1 = time.perf_counter()
    print(f"  generated in {t1-t0:.1f}s")

    img.save(args.output_png)
    print(f"\n✓ Saved {args.output_png} ({args.output_png.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
