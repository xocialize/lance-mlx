#!/usr/bin/env python3
"""Phase 5j — does latent_pos_base=0 fix the watercolor on the red panda prompt?

Hypothesis from Phase 5i.2 bisect: the watercolor aesthetic is triggered by
prompt-length-dependent drift of the latent block's mrope position-IDs.
When `base = text_len_before_latents` is large (long verbose prompt with
role tags), the latent grid coords land far from where Lance was trained.

Fix: anchor latent grid to origin (base=0) regardless of prompt length.

This script runs the red-panda prompt at our standard config (which produces
watercolor) with `latent_pos_base=None` (legacy) vs `latent_pos_base=0` (fix
hypothesis). It ALSO runs the ocean prompt for both — if the fix is correct,
ocean should stay sharp.

Variants (256² × 17f, 30 steps, CFG=4.0, seed=42 — our standard config):
  V0_redpanda_legacy : long prompt, base=text_len (CURRENT: watercolor)
  V1_redpanda_fix    : long prompt, base=0           (FIX HYPOTHESIS)
  V2_ocean_legacy    : short prompt, base=text_len (CURRENT: should still work)
  V3_ocean_fix       : short prompt, base=0           (FIX should not regress)

Wall-clock: ~4 × 50s ≈ 3.5 min
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


ORACLE_PROMPT_FILE = Path(
    "/Volumes/DEV_VOL1/VideoResearch/lance-mlx/tests/fixtures/results/"
    "t2v_sample_ts30_tts3.5_seed42_cfg4.0_kvcache_20260520_091630/prompt.json"
)


def main() -> int:
    LANCE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16")
    VAE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors")
    OUT_DIR = Path("/tmp/lance_phase5j")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    RED_PANDA_PROMPT = json.loads(ORACLE_PROMPT_FILE.read_text())["000000.mp4"]
    OCEAN_PROMPT = "a calm ocean wave rolling onto a sandy beach"

    # (name, prompt, latent_pos_base)
    variants = [
        ("V0_redpanda_legacy", RED_PANDA_PROMPT, None),
        ("V1_redpanda_FIX",    RED_PANDA_PROMPT, 0),
        ("V2_ocean_legacy",    OCEAN_PROMPT,     None),
        ("V3_ocean_FIX",       OCEAN_PROMPT,     0),
    ]

    print(f"┏━━ Phase 5j — latent_pos_base=0 fix hypothesis test ━━━━━━━━━━━━━━━")
    print(f"┃ scale: 256×256 × 17f (T_lat=5), 30 steps, CFG=4.0, seed=42")
    print(f"┃ variants:")
    for name, p, base in variants:
        print(f"┃   {name:22s} prompt={p[:40]!r}..., base={base}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Loading pipeline ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS,
        vae_safetensors=VAE_WEIGHTS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    import imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    md5s = {}
    strips = {}
    for name, prompt, base in variants:
        print(f"\n=== {name} (latent_pos_base={base}) ===")
        t0 = time.perf_counter()
        try:
            frames = pipe.generate(
                prompt,
                num_frames=17, height=256, width=256,
                num_steps=30, cfg_scale=4.0,
                seed=42, verbose=False,
                mape_anchor=None,
                spatial_merge_size=1,
                rope_fp32=False,
                prompt_format="ours",
                latent_pos_base=base,
            )
        except Exception as e:
            print(f"  GENERATION FAILED: {e!r}")
            continue
        dt = time.perf_counter() - t0
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        # MP4
        mp4 = OUT_DIR / f"{name}.mp4"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))

        # Filmstrip
        n = int(frames.shape[0])
        strip_w = 256 * n
        strip = Image.new('RGB', (strip_w, 256))
        for i, fr in enumerate(np.asarray(frames)):
            strip.paste(Image.fromarray(fr), (i * 256, 0))
        strip_path = OUT_DIR / f"{name}_strip.png"
        strip.save(strip_path)
        md5s[name] = hashlib.md5(strip_path.read_bytes()).hexdigest()
        strips[name] = strip
        print(f"  → {mp4}  ({mp4.stat().st_size/1e3:.0f} KB)")
        print(f"  → {strip_path}  md5={md5s[name][:16]}")

    # Verdict grid: 4 rows, each filmstrip scaled to same width
    print(f"\n=== Building verdict grid ===")
    if not strips:
        print(f"  No strips to grid; abort")
        return 1
    max_w = max(s.width for s in strips.values())
    target_h = 144
    label_h = 28
    margin = 8
    grid_w = max_w + 2*margin
    grid_h = len(strips) * (target_h + label_h + margin) + margin
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    y = margin
    for name, _, _ in variants:
        if name not in strips:
            continue
        s = strips[name]
        scaled_w = int(s.width * target_h / s.height)
        scaled = s.resize((scaled_w, target_h))
        x_off = margin + (max_w - scaled_w) // 2
        draw.text((margin + 4, y), f"{name}  md5={md5s[name][:16]}",
                  fill='white', font=font)
        grid.paste(scaled, (x_off, y + label_h))
        y += target_h + label_h + margin
    grid_path = OUT_DIR / "verdict_grid.png"
    grid.save(grid_path)
    print(f"  → {grid_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for name, _, base in variants:
        if name in md5s:
            print(f"┃ {name:22s} (base={base})  md5={md5s[name]}")
    print(f"┃")
    print(f"┃ Expectations:")
    print(f"┃   V0_redpanda_legacy: watercolor (current bug)")
    print(f"┃   V1_redpanda_FIX:    SHARP if hypothesis correct")
    print(f"┃   V2_ocean_legacy:    sharp (control — short prompt, already works)")
    print(f"┃   V3_ocean_FIX:       sharp (regression check — fix shouldn't break this)")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
