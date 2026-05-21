#!/usr/bin/env python3
"""Phase 4d video_edit demo — instruction-based video editing.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/15_video_edit_demo.py \\
        --input-video tests/fixtures/video_understanding/phase4e_17f_balls.mp4 \\
        --instruction "Change all the balls to red." \\
        --lance-weights /Volumes/.../Lance-3B-Video-bf16 \\
        --vae-weights   /Volumes/.../Wan22-VAE-bf16/vae.safetensors \\
        --out-mp4 /tmp/video_edit_out.mp4 \\
        --num-frames 17 --height 256 --width 256
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-video", type=Path, required=True)
    ap.add_argument("--instruction", required=True,
                    help='Edit instruction, e.g. "Change all the balls to red."')
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--num-frames", type=int, default=17)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lance-weights", type=Path, required=True)
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--out-mp4", type=Path, default=Path("/tmp/lance_video_edit.mp4"))
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("┏━━ Phase 4d video_edit demo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ input          : {args.input_video}")
    print(f"┃ instruction    : {args.instruction!r}")
    print(f"┃ output dims    : {args.num_frames}f × {args.width}×{args.height}")
    print(f"┃ steps          : {args.steps}  cfg={args.cfg_scale}  seed={args.seed}")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("\n=== Loading VideoEditPipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.video_edit import VideoEditPipeline
    pipe = VideoEditPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== Generating ===")
    t0 = time.perf_counter()
    frames = pipe.generate(
        input_video=args.input_video,
        instruction=args.instruction,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  generated {frames.shape[0]} frames in {elapsed:.1f}s")

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    with imageio.get_writer(args.out_mp4, fps=args.fps, codec="libx264") as writer:
        for f in frames:
            writer.append_data(f)
    print(f"  saved → {args.out_mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
