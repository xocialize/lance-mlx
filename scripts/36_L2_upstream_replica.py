#!/usr/bin/env python3
"""L2-impl — test upstream-correct position-ID convention vs Phase 5j.

Per the L2 audit (notes/L2_upstream_position_ids_audit.md), upstream
inference uses get_rope_index which does:
  - sms=2 divisor on h/w grid coords
  - base = text_len + st_idx (text-position anchor)

Phase 5j uses sms=1 + base=0 (empirically validated, dramatically
better than legacy). The upstream-correct combo is unbenchmarked at
production scale; Phase 5g V1 tested sms=2+base=text_len at 256² and
got "subject loss" but subjects barely resolve at 256² even at baseline.

Three variants at 768²×13f (the production scale we know works):

  V0 — Phase 5j default:         sms=1, base=0
  V1 — upstream replica:         sms=2, base=text_len (None=legacy)
  V2 — half-replica probe:       sms=2, base=0

If V1 matches or exceeds V0, the upstream-correct config wins (more
diff-friendly to upstream + may close corner-cloud and motion-direction
residuals). If V1 regresses, our port has an additional divergence to
investigate.

V2 helps disambiguate whether sms or base is the dominant axis.

Red-panda-surfing oracle prompt, seed=43 (matching the Phase 5k full
oracle pass), 30 steps, CFG=4, MaPE=None.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> int:
    LANCE_WEIGHTS = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16"
    )
    VAE_WEIGHTS = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors"
    )
    ORACLE_PROMPT_FILE = Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
        "t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630/prompt.json"
    )
    OUT = Path("/tmp/lance_L2_upstream_replica")
    OUT.mkdir(parents=True, exist_ok=True)

    PROMPT = json.loads(ORACLE_PROMPT_FILE.read_text())["000000.mp4"]

    variants = [
        ("V0_phase5j",          {"spatial_merge_size": 1, "latent_pos_base": 0}),
        ("V1_upstream",         {"spatial_merge_size": 2, "latent_pos_base": None}),
        ("V2_sms2_base0_probe", {"spatial_merge_size": 2, "latent_pos_base": 0}),
    ]

    print(f"┏━━ L2-impl — upstream-replica position-IDs A/B @ 768²×13f ━━━━━━━━━━")
    print(f"┃ prompt: {PROMPT[:70]}...")
    print(f"┃ config: 768×768 × 13f, 30 steps, CFG=4.0, seed=43, MaPE=None")
    for name, ovr in variants:
        print(f"┃   {name:24s} sms={ovr['spatial_merge_size']}, "
              f"base={ovr['latent_pos_base']!r}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    import mlx.core as mx
    import numpy as np
    import imageio
    from PIL import Image

    print(f"\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    md5s = {}
    mid_imgs = {}
    for name, ovr in variants:
        print(f"\n=== {name} (sms={ovr['spatial_merge_size']}, "
              f"base={ovr['latent_pos_base']!r}) ===")
        t0 = time.perf_counter()
        frames = pipe.generate(
            PROMPT,
            num_frames=13, height=768, width=768,
            num_steps=30, cfg_scale=4.0,
            seed=43, verbose=False,
            mape_anchor=None,
            spatial_merge_size=ovr["spatial_merge_size"],
            latent_pos_base=ovr["latent_pos_base"],
        )
        dt = time.perf_counter() - t0
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        mp4 = OUT / f"{name}.mp4"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))
        mid = int(frames.shape[0] // 2)
        png = OUT / f"{name}_midframe.png"
        Image.fromarray(np.asarray(frames[mid])).save(png)
        md5s[name] = hashlib.md5(png.read_bytes()).hexdigest()
        mid_imgs[name] = Image.open(png)
        print(f"  → {mp4}  ({mp4.stat().st_size/1e3:.0f} KB)")
        print(f"  → {png}  md5={md5s[name][:16]}")

    # Build 3-row compare grid (one row per variant)
    print(f"\n=== Building compare grid ===")
    W, H = mid_imgs[variants[0][0]].size
    label_h = 32
    margin = 8
    grid = Image.new("RGB", (W + 2*margin, 3*(H + label_h + margin) + margin), "black")
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i, (name, ovr) in enumerate(variants):
        y = margin + i * (H + label_h + margin)
        label = f"{name}  sms={ovr['spatial_merge_size']}  base={ovr['latent_pos_base']!r}"
        draw.text((margin + 4, y), label, fill="white", font=font)
        grid.paste(mid_imgs[name], (margin, y + label_h))
    grid_path = OUT / "compare_grid.png"
    grid.save(grid_path)
    print(f"  → {grid_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for name, _ in variants:
        print(f"┃ {name:24s} md5={md5s[name]}")
    print(f"┃")
    print(f"┃ Inspect {grid_path}")
    print(f"┃ If V1 matches/beats V0 visually → switch to upstream-replica default.")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
