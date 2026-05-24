#!/usr/bin/env python3
"""Phase 5n / D3 — t2v mape_anchor sweep.

Last live candidate after D1 (VAE) and D2 (CFG renorm) were both refuted.

Asymmetry: t2i unconditionally applies a +1000 t-axis MaPE shift via
hardcoded `MAPE_ANCHOR_IMAGE_GEN`. t2v applies NOTHING by default
(Phase 5d removed the legacy +2000 default → None). The two pipelines
feed structurally different t-axis position-IDs to the SAME shared
mrope kernel, and the t2v side has never been tested at the value t2i
uses (1000) — only at None (current good) and 2000 (legacy bad).

Phase 5d removed the +2000 shift because it caused painterly smearing.
But the empirical sweep was binary (on/off, 2000 vs None). Intermediate
values were never tested. It's plausible that:
  - None is the local optimum within the range [0, 2000]
  - OR there's a sweet spot at ~1000 (matching t2i's working regime)
  - OR the optimum depends on n_lat and shifts at different scales

This script runs t2v.generate() at small scale (256²×9f, ~30s per run)
for 5 mape_anchor values, saves a representative frame from each, and
builds a comparison grid for visual judgment.

Cost: ~3 min (load + 5 generations + grid).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LANCE_VIDEO_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-Video-bf16"
VAE_SAFETENSORS     = LANCE_VIDEO_WEIGHTS / "vae.safetensors"
OUT_DIR             = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d3_t2v_mape_anchor_sweep"


# A prompt with multiple discriminative features so prompt-adherence is
# visible. "Red fire truck on a snowy street" — color (red), object
# (truck — not car/bus), context (snow — white ground), action (parked).
PROMPT = ("A red fire truck parked on a snowy street, photorealistic, sharp focus.")

# Anchors to sweep. None = no shift (current default). 1000 = match t2i.
# 2000 = restore legacy (KNOWN to regress per Phase 5d). 500/1500 fill out.
ANCHORS = [None, 500, 1000, 1500, 2000]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 5n / D3 — t2v mape_anchor sweep ===")
    print(f"  prompt:   {PROMPT!r}")
    print(f"  scale:    256² × 9f  (n_lat = 3*16*16 = 768)")
    print(f"  anchors:  {ANCHORS}")
    print(f"  out:      {OUT_DIR}\n")

    print(f"Loading TextToVideoPipeline ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2v import TextToVideoPipeline
    pipe = TextToVideoPipeline.from_pretrained(
        lance_weights_dir=LANCE_VIDEO_WEIGHTS,
        vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

    from PIL import Image, ImageDraw, ImageFont
    frame_outputs = []

    for anchor in ANCHORS:
        label = f"anchor={anchor}"
        print(f"──── {label} ────")
        t0 = time.perf_counter()
        frames_u8 = pipe.generate(
            prompt=PROMPT,
            height=256, width=256, num_frames=9,
            num_steps=30, cfg_scale=4.0, seed=42,
            mape_anchor=anchor,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        # frames shape: (T_decoded, H, W, 3) uint8 — keep a middle frame
        # to show what the model actually produced (skip causal-padding
        # build-up frames at the start).
        T = frames_u8.shape[0]
        mid = T // 2
        first = frames_u8[0]
        middle = frames_u8[mid]
        # save individual PNGs
        sub = OUT_DIR / f"anchor_{anchor}"
        sub.mkdir(exist_ok=True)
        for i in range(T):
            Image.fromarray(frames_u8[i]).save(sub / f"f{i:02d}.png")
        # stats
        std_first = float(first.std())
        std_mid = float(middle.std())
        mean_first = float(first.mean())
        mean_mid = float(middle.mean())
        print(f"  generated in {dt:.1f}s; T_decoded={T}")
        print(f"  frame 0:    mean={mean_first:.2f}  std={std_first:.2f}")
        print(f"  frame {mid:>2d}:   mean={mean_mid:.2f}  std={std_mid:.2f}")
        frame_outputs.append((label, frames_u8, mid))

    # Build comparison grid.
    print(f"\n──── Building comparison grid ────")
    H = 256
    rows = len(frame_outputs)
    # 2 columns: frame 0 (causal-padding start) and mid frame (representative)
    cols = 2
    pad = 32
    margin = 12
    grid_w = cols * H + (cols + 1) * margin
    grid_h = rows * (H + pad) + (rows + 1) * margin
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()
    col_headers = ["frame 0 (decode start)", "middle frame"]
    for i, label in enumerate(col_headers):
        x = margin + i * (H + margin)
        draw.text((x + 6, 4), label, fill='gray', font=font)
    for r, (label, frames, mid) in enumerate(frame_outputs):
        y = margin + r * (H + pad + margin) + pad
        for c, frame in enumerate([frames[0], frames[mid]]):
            x = margin + c * (H + margin)
            grid.paste(Image.fromarray(frame), (x, y))
            if c == 0:
                draw.text((x + 6, y - pad + 4), label,
                          fill='yellow', font=font)
    grid_path = OUT_DIR / "_compare_grid.png"
    grid.save(grid_path)
    print(f"  saved: {grid_path}")

    print(f"\n=== Verdict guidance ===")
    print(f"  - Look for: subject (RED FIRE TRUCK), context (SNOW),")
    print(f"    sharpness/contrast, color saturation.")
    print(f"  - If anchor=1000 visibly outperforms anchor=None (current default):")
    print(f"    → mrope asymmetry is the gap; ship anchor=1000 as new default.")
    print(f"  - If anchor=None is best: this candidate is also refuted, and the")
    print(f"    quality gap is from something we haven't identified yet.")
    print(f"  - If anchor=2000 reproduces Phase 5d's painterly: confirms the prior")
    print(f"    finding but doesn't help.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
