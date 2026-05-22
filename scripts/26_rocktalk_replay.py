#!/usr/bin/env python3
"""Phase 5i — replay RockTalk's exact ocean-wave config through OUR pipeline.

From `RockTalk/Lance-3B-Video-MLX` HF model card:
    Prompt:  "a calm ocean wave rolling onto a sandy beach"
    Size:    256 × 256
    Frames:  T_lat=3 → 9 frames
    Steps:   24
    CFG:     4
    Seed:    0 (per their `sample_t2i` example code snippet)

This gives us an apples-to-apples comparison point:
  - RT's published sample (from THEIR pipeline) is at
    samples/ocean_wave_9frames.png in the RockTalk/Lance-3B-Video-MLX HF repo
    — already downloaded to /Volumes/DEV_VOL1/VideoResearch/rocktalk-weights/
  - OUR pipeline output (this script) — what we produce on the same input

If OUR output is watercolor on this exact config too → confirms the bug
is in OUR forward-pass code, AND gives us a small target to debug against
(9-frame, low-resolution, simple subject = easier to inspect than the
red-panda case).

If OUR output is sharp on this exact config → suggests the bug is
config-dependent (e.g., prompt-length sensitivity or seed quirk).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    LANCE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16")
    VAE_WEIGHTS = Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Wan22-VAE-bf16/vae.safetensors")
    OUT_DIR = Path("/tmp/lance_phase5i")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PROMPT = "a calm ocean wave rolling onto a sandy beach"
    # T_lat=3 → num_frames = (3-1)*4 + 1 = 9
    NUM_FRAMES = 9
    HEIGHT = WIDTH = 256
    NUM_STEPS = 24
    CFG = 4.0
    SEED = 0

    print(f"┏━━ Phase 5i — RockTalk's ocean-wave config × OUR pipeline ━━━━━━━━━━")
    print(f"┃ prompt   : {PROMPT!r}")
    print(f"┃ size     : {HEIGHT}×{WIDTH}, {NUM_FRAMES} frames (T_lat=3)")
    print(f"┃ steps    : {NUM_STEPS}")
    print(f"┃ CFG      : {CFG}")
    print(f"┃ seed     : {SEED}")
    print(f"┃ template : 'ours' (legacy verbose, but also testing 'rocktalk' below)")
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

    runs = []
    for label, fmt in [("ours_tmpl", "ours"), ("rocktalk_tmpl", "rocktalk")]:
        print(f"\n=== Generating ({label}, prompt_format={fmt!r}) ===")
        t0 = time.perf_counter()
        frames = pipe.generate(
            PROMPT,
            num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
            num_steps=NUM_STEPS, cfg_scale=CFG,
            seed=SEED, verbose=False,
            mape_anchor=None,
            spatial_merge_size=1,
            rope_fp32=False,
            prompt_format=fmt,
        )
        dt = time.perf_counter() - t0
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        # MP4
        mp4 = OUT_DIR / f"{label}.mp4"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))
        print(f"  → {mp4} ({mp4.stat().st_size/1e3:.0f} KB)")

        # Strip — concatenate all 9 frames horizontally (matches RT's
        # ocean_wave_9frames.png layout).
        strip_h = HEIGHT
        strip_w = WIDTH * NUM_FRAMES
        strip = Image.new('RGB', (strip_w, strip_h))
        for i, fr in enumerate(np.asarray(frames)):
            strip.paste(Image.fromarray(fr), (i * WIDTH, 0))
        strip_path = OUT_DIR / f"{label}_9frames.png"
        strip.save(strip_path)
        print(f"  → {strip_path}")

        runs.append((label, frames, strip_path))

    # Side-by-side compare: RT's published strip vs each of our two strips.
    rt_strip = Path("/Volumes/DEV_VOL1/VideoResearch/rocktalk-weights/samples/ocean_wave_9frames.png")
    if not rt_strip.exists():
        # Pull it from HF cache if exists
        alt = Path.home() / ".cache" / "huggingface" / "hub"
        for p in alt.rglob("ocean_wave_9frames.png"):
            rt_strip = p
            break

    if rt_strip.exists():
        print(f"\n=== Building 3-row compare grid vs RT's published sample ===")
        rt_img = Image.open(rt_strip).convert("RGB")
        # Resize to match if needed
        if rt_img.size != (strip_w, strip_h):
            print(f"  RT strip size: {rt_img.size} (will resize to match {strip_w}x{strip_h})")
            rt_img = rt_img.resize((strip_w, strip_h))

        label_h = 30
        margin = 8
        rows = [("RT published (their pipeline)", rt_img)]
        for label, _, sp in runs:
            rows.append((f"OURS ({label})", Image.open(sp)))

        grid_w = strip_w + 2*margin
        grid_h = len(rows) * (strip_h + label_h + margin) + margin
        grid = Image.new('RGB', (grid_w, grid_h), 'black')
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
        except Exception:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(grid)
        y = margin
        for label, img in rows:
            draw.text((margin + 4, y), label, fill='white', font=font)
            grid.paste(img, (margin, y + label_h))
            y += strip_h + label_h + margin

        grid_path = OUT_DIR / "rt_vs_ours_grid.png"
        grid.save(grid_path)
        print(f"  → {grid_path}")
    else:
        print(f"\n  RT published sample not found at {rt_strip}")
        print(f"  Skipping 3-row compare; inspect /tmp/lance_phase5i/*.png manually.")

    print(f"\n┏━━ Done ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ Outputs: {OUT_DIR}/")
    print(f"┃   {OUT_DIR}/ours_tmpl_9frames.png")
    print(f"┃   {OUT_DIR}/rocktalk_tmpl_9frames.png")
    if rt_strip.exists():
        print(f"┃   {OUT_DIR}/rt_vs_ours_grid.png   ← the headline comparison")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
