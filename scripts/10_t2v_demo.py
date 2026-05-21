#!/usr/bin/env python3
"""Phase 4a — first text-to-video generation on Lance MLX.

MVP starts conservative (256x256 × 16 frames) to validate the architecture
before scaling up to Lance's default (768x768 × 50 frames). Encodes the
output to MP4 via imageio-ffmpeg.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/10_t2v_demo.py \\
        --prompt "A red panda surfing on a sunny wave." \\
        --lance-weights /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16 \\
        --vae-weights   /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors \\
        --output-mp4    /tmp/lance_t2v.mp4 \\
        --num-frames 16 --height 256 --width 256
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A red panda surfing on a bright sunny wave.")
    ap.add_argument("--lance-weights", type=Path, required=True,
                    help="MUST be Lance_3B_Video (not Lance_3B) for video.")
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--output-mp4", type=Path, default=Path("/tmp/lance_t2v.mp4"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--num-frames", type=int, default=16,
                    help="Pre-VAE frame count. Lance default is 50; MVP uses 16.")
    ap.add_argument("--height", type=int, default=256,
                    help="Lance default is 768; MVP uses 256.")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fps", type=int, default=12,
                    help="Output MP4 fps. Lance default is 12.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"=== Loading TextToVideoPipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== Generating ===")
    print(f"  prompt: {args.prompt!r}")
    print(f"  {args.num_frames}f × {args.width}x{args.height}, "
          f"{args.steps} steps, cfg={args.cfg_scale}, seed={args.seed}")
    t0 = time.perf_counter()
    frames = pipe.generate(
        args.prompt,
        num_frames=args.num_frames, height=args.height, width=args.width,
        num_steps=args.steps, cfg_scale=args.cfg_scale,
        seed=args.seed, verbose=args.verbose,
    )
    t1 = time.perf_counter()
    print(f"  generated {frames.shape[0]} decoded frames in {t1-t0:.1f}s")

    # Encode MP4.
    print(f"\n=== Encoding MP4 → {args.output_mp4} ===")
    import imageio
    with imageio.get_writer(args.output_mp4, fps=args.fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  saved {args.output_mp4} ({args.output_mp4.stat().st_size/1e3:.0f} KB)")
    print(f"  shape: {frames.shape}  fps: {args.fps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
