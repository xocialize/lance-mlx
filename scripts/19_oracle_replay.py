#!/usr/bin/env python3
"""Phase 5d — replay Lance Phase 0 t2v oracle at exact PyTorch config.

Generates the oracle prompt 000000 (red panda surfing) at:
  - 768×768 spatial (oracle)
  - 50 frames input → 49 decoded (oracle: Wan2.2 VAE causal-temporal compression)
  - seed 42 (oracle)
  - timestep_shift 3.5, cfg_scale 4.0, num_steps 30 (oracle)
  - MaPE shift removed (--no-mape-shift) — Candidate 0 hypothesis from
    github issue #2.

Compares MAD/structural similarity vs the saved oracle MP4.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/19_oracle_replay.py \\
        --lance-weights /Volumes/.../Lance-3B-Video-bf16 \\
        --vae-weights /Volumes/.../Wan22-VAE-bf16/vae.safetensors \\
        --no-mape-shift \\
        --out-dir /tmp/lance_oracle_replay_candidate0

Hardware: ~2.25 hours on M5 Max at 768×768×50f, n_lat=29952.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ORACLE_DIR = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
    "t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630"
)
ORACLE_PROMPT_FILE = ORACLE_DIR / "prompt.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-weights", type=Path, required=True)
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/lance_oracle_replay"))
    ap.add_argument("--oracle-id", default="000000",
                    help="Oracle prompt id to replay (000000 = red panda surfing).")
    ap.add_argument("--no-mape-shift", action="store_true",
                    help="Skip the MAPE_ANCHOR_VIDEO_GEN=2000 shift. Candidate 0 "
                         "hypothesis from issue #2. RECOMMENDED for this test.")
    ap.add_argument("--num-frames", type=int, default=50)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the exact prompt the oracle used.
    prompts = json.loads(ORACLE_PROMPT_FILE.read_text())
    oracle_mp4_name = f"{args.oracle_id}.mp4"
    if oracle_mp4_name not in prompts:
        print(f"ERROR: {oracle_mp4_name} not in oracle prompt list")
        return 1
    prompt_text = prompts[oracle_mp4_name]
    oracle_mp4 = ORACLE_DIR / oracle_mp4_name

    print("┏━━ Phase 5d — Oracle replay (Candidate 0: MaPE-shift removed) ━━━━━")
    print(f"┃ oracle id   : {args.oracle_id}")
    print(f"┃ prompt      : {prompt_text[:80]}...")
    print(f"┃ oracle ref  : {oracle_mp4}")
    print(f"┃ config      : {args.num_frames}f × {args.width}×{args.height}, "
          f"{args.steps} steps, CFG={args.cfg_scale}, seed={args.seed}")
    print(f"┃ MaPE shift  : {'DISABLED (None)' if args.no_mape_shift else f'2000 (default)'}")
    print(f"┃ out         : {args.out_dir}")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 2. Load pipeline.
    print("\n=== Loading TextToVideoPipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # 3. Generate at oracle config.
    print(f"\n=== Generating {args.oracle_id} replay ===")
    t0 = time.perf_counter()
    frames = pipe.generate(
        prompt_text,
        num_frames=args.num_frames,
        height=args.height, width=args.width,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        mape_anchor=None if args.no_mape_shift else 2000,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  generated {frames.shape[0]} frames in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")

    # 4. Save the output.
    import imageio
    import numpy as np
    from PIL import Image
    mp4_path = args.out_dir / "replay.mp4"
    with imageio.get_writer(mp4_path, fps=12, codec="libx264") as writer:
        for f in frames:
            writer.append_data(f)
    print(f"  saved {mp4_path}")

    # 5. Mid-frame extraction for both MLX replay and oracle.
    mid_replay = frames[frames.shape[0] // 2]
    Image.fromarray(mid_replay).save(args.out_dir / "mid_replay.png")
    print(f"  saved {args.out_dir / 'mid_replay.png'}")

    # Extract corresponding mid from oracle MP4 for direct comparison.
    import imageio.v3 as iio
    oracle_frames = np.array([f for f in iio.imiter(str(oracle_mp4))])
    if oracle_frames.size > 0:
        mid_oracle = oracle_frames[oracle_frames.shape[0] // 2]
        Image.fromarray(mid_oracle).save(args.out_dir / "mid_oracle.png")
        print(f"  saved {args.out_dir / 'mid_oracle.png'} (oracle reference)")

    # 6. Numerical compare — same-frame indices (best-effort, since decoded
    # counts may differ slightly).
    if oracle_frames.size > 0:
        n = min(frames.shape[0], oracle_frames.shape[0])
        mlx_resized = frames[:n]
        ora_resized = oracle_frames[:n]
        # If shapes differ, resize MLX to oracle's spatial dims via PIL.
        if mlx_resized.shape != ora_resized.shape:
            from PIL import Image as _PIL
            target_hw = ora_resized.shape[1:3]
            mlx_resized = np.stack([
                np.asarray(_PIL.fromarray(mlx_resized[i]).resize(
                    (target_hw[1], target_hw[0]), _PIL.LANCZOS
                )) for i in range(n)
            ])
        mad = float(np.abs(mlx_resized.astype(np.float32) - ora_resized.astype(np.float32)).mean())
        print(f"\n=== Comparison vs oracle ===")
        print(f"  MAD (u8 domain, first {n} frames): {mad:.2f} / 255")
        print(f"  oracle frame count: {oracle_frames.shape[0]}")
        print(f"  MLX frame count: {frames.shape[0]}")

    # 7. Write meta.
    meta = {
        "oracle_id": args.oracle_id,
        "oracle_mp4": str(oracle_mp4),
        "prompt": prompt_text,
        "config": {
            "num_frames": args.num_frames,
            "height": args.height, "width": args.width,
            "steps": args.steps, "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "mape_shift_disabled": args.no_mape_shift,
        },
        "wall_clock_s": round(elapsed, 1),
        "mlx_decoded_frames": int(frames.shape[0]),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n✓ Replay complete. View side-by-side:")
    print(f"  open {args.out_dir / 'mid_replay.png'} {args.out_dir / 'mid_oracle.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
