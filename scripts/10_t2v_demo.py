#!/usr/bin/env python3
"""Phase 4a — text-to-video generation on Lance MLX.

The MVP preset stays conservative (256x256 × 16 frames) to validate the
architecture; `--long` switches to a long-clip preset (256x256 × 61 frames with
temporal VAE tiling) that the relay memory mode + shed cascade make fit a 16 GB
Mac. Encodes the output to MP4 via imageio-ffmpeg.

Usage (short MVP):
    HF_HUB_DISABLE_XET=1 uv run python scripts/10_t2v_demo.py \\
        --prompt "A red panda surfing on a sunny wave." \\
        --lance-weights /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16 \\
        --vae-weights   /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors \\
        --output-mp4    /tmp/lance_t2v.mp4 \\
        --num-frames 16 --height 256 --width 256

Usage (LONG clip — 61 frames @ 256², relay + temporal VAE tiling, fits 16 GB):
    HF_HUB_DISABLE_XET=1 uv run python scripts/10_t2v_demo.py \\
        --prompt "A red panda surfing on a sunny wave." \\
        --lance-weights ... --vae-weights ... \\
        --output-mp4 /tmp/lance_t2v_long.mp4 \\
        --long --memory-mode relay --verbose
    # (equivalently, set it explicitly:)
    #   --num-frames 61 --vae-temporal-tile 16 --vae-temporal-overlap 8
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
    # Phase 5g research-brief candidates (github issue #2):
    ap.add_argument("--spatial-merge-size", type=int, default=1,
                    help="P0b: divisor for h/w axes in latent position-id grid. "
                         "Default 1 (legacy). Set to 2 to match upstream Lance's "
                         "`shift_position_ids` and RockTalk's parallel MLX port "
                         "(both divide visual position-ids by spatial_merge_size=2).")
    ap.add_argument("--rope-fp32", action="store_true",
                    help="P0a: compute cos/sin and the q*cos+rotate_half(q)*sin "
                         "rotation in fp32 (mlx-vlm's stock path casts cos/sin to "
                         "bf16 before rotation). Hypothesized to recover "
                         "high-frequency precision in flow-matching velocity.")
    ap.add_argument("--mape-anchor", type=int, default=None,
                    help="Override MaPE temporal anchor (default None = no shift, "
                         "per Phase 5d). Pass 2000 to restore legacy behavior.")
    ap.add_argument("--memory-mode", default=None,
                    choices=["auto", "parallel", "relay"],
                    help="How the three generation phases (prefill→UND, "
                         "denoise→GEN, decode→VAE) share memory (load-time). "
                         "Default None = auto: BINARY parallel (ws ≥ 18 GiB, all "
                         "resident + reusable) | relay (below, incl. 16 GB Macs — "
                         "baton handoff, towers never co-resident + VAE deferred to "
                         "decode, ~10 GB peak, single-shot). Output is identical.")
    ap.add_argument("--und-mode", default=None,
                    choices=["auto", "keep", "free", "deferred"],
                    help="DEPRECATED alias for --memory-mode (keep→parallel, "
                         "deferred/free→relay). Use --memory-mode.")
    ap.add_argument("--long", action="store_true",
                    help="Long-clip preset: 61 frames @ 256² with temporal VAE "
                         "tiling (tile=16, overlap=8). Applied only where the "
                         "corresponding flags are still at their defaults, so "
                         "explicit --num-frames / --vae-temporal-tile still win.")
    ap.add_argument("--vae-temporal-tile", type=int, default=None,
                    help="Temporal VAE tile size in PIXEL frames (÷4 → latent "
                         "tile). Must be ≥16 and divisible by 8. Needed for long "
                         "clips so the whole-clip decode doesn't spike. None = off.")
    ap.add_argument("--vae-temporal-overlap", type=int, default=0,
                    help="Temporal tile overlap in pixel frames. Divisible by 8 "
                         "and < tile (e.g. 8). Default 0.")
    args = ap.parse_args()

    # --long preset: bump only the flags still at their MVP defaults so explicit
    # overrides win. 61f @ 256² + temporal tiling (tile=16, overlap=8) is the
    # validated long single-shot that fits 16 GB under relay + the shed cascade.
    if args.long:
        if args.num_frames == 16:
            args.num_frames = 61
        if args.vae_temporal_tile is None:
            args.vae_temporal_tile = 16
            if args.vae_temporal_overlap == 0:
                args.vae_temporal_overlap = 8

    print(f"=== Loading TextToVideoPipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
        memory_mode=args.memory_mode,
        und_mode=args.und_mode,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s (memory_mode={pipe.memory_mode})")

    print(f"\n=== Generating{' (LONG)' if args.long else ''} ===")
    print(f"  prompt: {args.prompt!r}")
    print(f"  {args.num_frames}f × {args.width}x{args.height}, "
          f"{args.steps} steps, cfg={args.cfg_scale}, seed={args.seed}")
    if args.vae_temporal_tile is not None:
        print(f"  temporal VAE tiling: tile={args.vae_temporal_tile} "
              f"overlap={args.vae_temporal_overlap} (pixel frames)")
    t0 = time.perf_counter()
    if args.verbose:
        print(f"  phase 5g flags: sms={args.spatial_merge_size}, "
              f"rope_fp32={args.rope_fp32}, mape_anchor={args.mape_anchor}")
    frames = pipe.generate(
        args.prompt,
        num_frames=args.num_frames, height=args.height, width=args.width,
        num_steps=args.steps, cfg_scale=args.cfg_scale,
        seed=args.seed, verbose=args.verbose,
        spatial_merge_size=args.spatial_merge_size,
        rope_fp32=args.rope_fp32,
        mape_anchor=args.mape_anchor,
        memory_mode=args.memory_mode,
        vae_temporal_tile=args.vae_temporal_tile,
        vae_temporal_overlap=args.vae_temporal_overlap,
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
