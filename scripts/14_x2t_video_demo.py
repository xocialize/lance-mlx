#!/usr/bin/env python3
"""x2t_video demo — Lance video VQA / captioning via MLX.

Loads Lance + Qwen2.5-VL ViT, decodes an MP4, runs the existing
UnderstandingPipeline.generate_video() on it.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/14_x2t_video_demo.py \\
        --video tests/fixtures/video_understanding/caption-short-01.mp4 \\
        --question "Offer a succinct account of the culinary process shown in this video." \\
        --lance-weights /Volumes/.../Lance-3B-Video-bf16 \\
        --vit-weights   /Volumes/.../Lance-3B-Video-bf16/vit.safetensors
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--question",
                    default="Describe what happens in this video.")
    ap.add_argument("--num-sample-frames", type=int, default=16,
                    help="Frames to sample from video (must be even for Qwen2.5-VL ViT).")
    ap.add_argument("--target-h", type=int, default=224)
    ap.add_argument("--target-w", type=int, default=224)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lance-weights", type=Path, required=True,
                    help="Use Lance_3B_Video for video tasks per upstream.")
    ap.add_argument("--vit-weights", type=Path, required=True)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("┏━━ x2t_video demo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ video        : {args.video}")
    print(f"┃ question     : {args.question!r}")
    print(f"┃ sample       : {args.num_sample_frames} frames @ {args.target_w}×{args.target_h}")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print("\n=== Loading UnderstandingPipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vit_safetensors=args.vit_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== Generating ===")
    t0 = time.perf_counter()
    answer = pipe.generate_video(
        args.video, args.question,
        num_sample_frames=args.num_sample_frames,
        target_h=args.target_h, target_w=args.target_w,
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_cache,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  generated in {elapsed:.1f}s")
    print(f"\n┏━━ ANSWER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ {answer}")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
