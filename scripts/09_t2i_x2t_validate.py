#!/usr/bin/env python3
"""Phase 3d validation oracle: t2i → x2t_image self-describe loop.

Generates an image from a prompt using Lance MLX, then asks the same model
to describe what's in the image. Reports the prompt and the description
side-by-side so we can score how well t2i preserved the prompt's content.

The understanding head is the empirically-trustworthy half of the model
(Phase 2.1b: read license plates correctly, described Colosseums
factually). So if it tells us our t2i output is "a nighttime urban scene"
when we asked for "a cat with a poster", that's strong evidence the t2i
flow lost the conditioning — independent of whether someone eye-balls
the PNG.

Loads BOTH pipelines (t2i + understanding). Memory peaks ~25 GB resident.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/09_t2i_x2t_validate.py \\
        --prompt "A cat holds a poster with rainbow text 'STOP'" \\
        --lance-weights /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16 \\
        --vae-weights   /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors \\
        --vit-weights   /Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16/vit.safetensors \\
        --output-png    /tmp/lance_validate.png \\
        --label baseline
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A cat holds a poster with rainbow text 'STOP'")
    ap.add_argument("--describe-question",
                    default="What is shown in this image?",
                    help="Question for the x2t_image describe step.")
    ap.add_argument("--lance-weights", type=Path, required=True)
    ap.add_argument("--vae-weights", type=Path, required=True)
    ap.add_argument("--vit-weights", type=Path, required=True)
    ap.add_argument("--output-png", type=Path, default=Path("/tmp/lance_validate.png"))
    ap.add_argument("--label", default="run",
                    help="Label for log lines + result files (e.g. 'baseline', 'cand1').")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--describe-max-tokens", type=int, default=128)
    args = ap.parse_args()

    print(f"┏━━ [{args.label}] t2i → x2t validation ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃  prompt: {args.prompt!r}")
    print(f"┃  {args.width}x{args.height}, {args.steps} steps, cfg={args.cfg_scale}, seed={args.seed}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # --- t2i: generate ----------------------------------------------------
    print(f"\n=== [1/3] Loading t2i pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    t2i = TextToImagePipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vae_safetensors=args.vae_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== [2/3] Generating image ===")
    t0 = time.perf_counter()
    img = t2i.generate(
        args.prompt,
        height=args.height, width=args.width,
        num_steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed,
    )
    t_gen = time.perf_counter() - t0
    print(f"  generated in {t_gen:.1f}s")
    img.save(args.output_png)
    print(f"  saved {args.output_png}")

    # Free t2i to reduce memory pressure before loading understanding.
    del t2i
    gc.collect()

    # --- x2t_image: describe ----------------------------------------------
    print(f"\n=== [3/3] Loading understanding pipeline + describing ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    x2t = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vit_safetensors=args.vit_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    img_for_x2t = Image.open(args.output_png).convert("RGB")
    description = x2t.generate(
        img_for_x2t, args.describe_question,
        max_new_tokens=args.describe_max_tokens, prompt_style="lance",
    )
    t_desc = time.perf_counter() - t0
    print(f"  described in {t_desc:.1f}s")

    # --- Verdict ---------------------------------------------------------
    print(f"\n┏━━ [{args.label}] verdict ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃  prompt:      {args.prompt!r}")
    print(f"┃  description: {description!r}")
    print(f"┃  timing:      t2i={t_gen:.1f}s, describe={t_desc:.1f}s")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
