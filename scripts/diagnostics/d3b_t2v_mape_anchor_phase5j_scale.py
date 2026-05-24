#!/usr/bin/env python3
"""Phase 5n / D3b — t2v mape_anchor sweep at Phase-5j-sensitive scale.

D3 at 256²×9f + short prompt showed no anchor sensitivity. Phase 5j
proved the mrope/position-ID effect requires *verbose prompts* and
larger scale (their fix landed at 256²×17f with the red-panda oracle).
This rerun uses the exact same test conditions to maximize sensitivity.

Anchors tested at this scale:
  - None  (current default, Phase 5d)
  - 1000  (match t2i's value, NEVER TESTED)
  - 2000  (legacy default, KNOWN regression per Phase 5d)

Drops 500/1500 since D3 showed they're indistinguishable at small scale.
Cost: ~3 min (3 generations × ~50s each + load + grid).
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
OUT_DIR             = REPO_ROOT / "notes" / "phase5n_diagnostics" / "d3b_t2v_mape_anchor_phase5j_scale"

# The canonical Phase 5j oracle prompt — verbose, ~80 tokens, proven to
# show position-ID sensitivity per Phase 5i.2 bisect.
PROMPT = (
    "A medium-close shot shows a red panda wearing a gold-trimmed cap "
    "and travel satchel on a bright seaside wave with a painted "
    "surfboard, foam spray, and a glowing summer sky. Subject fills "
    "frame; premium detail, clear focus, lively eyes, readable motion. "
    "tracking shot. It rides the wave, lifts one paw in balance, and "
    "laughs as spray catches the light."
)

ANCHORS = [None, 1000, 2000]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 5n / D3b — t2v mape_anchor at Phase-5j-sensitive scale ===")
    print(f"  scale:    256² × 17f  (t_lat=5, n_lat=1280)")
    print(f"  prompt:   <Phase 5j canonical red-panda oracle, {len(PROMPT.split())} words>")
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
    runs = []

    for anchor in ANCHORS:
        label = f"anchor={anchor}"
        print(f"──── {label} ────")
        t0 = time.perf_counter()
        frames_u8 = pipe.generate(
            prompt=PROMPT,
            height=256, width=256, num_frames=17,
            num_steps=30, cfg_scale=4.0, seed=42,
            mape_anchor=anchor,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        T = frames_u8.shape[0]
        print(f"  generated in {dt:.1f}s; T_decoded={T}")
        mid = T // 2
        print(f"  frame 0 :  mean={float(frames_u8[0].mean()):.2f}  std={float(frames_u8[0].std()):.2f}")
        print(f"  frame{mid:>2d} :  mean={float(frames_u8[mid].mean()):.2f}  std={float(frames_u8[mid].std()):.2f}")
        print(f"  frame{T-1:>2d} :  mean={float(frames_u8[T-1].mean()):.2f}  std={float(frames_u8[T-1].std()):.2f}")
        # Save all frames as individual PNGs.
        sub = OUT_DIR / f"anchor_{anchor}"
        sub.mkdir(exist_ok=True)
        for i in range(T):
            Image.fromarray(frames_u8[i]).save(sub / f"f{i:02d}.png")
        runs.append((label, frames_u8, mid))

    # Wide comparison grid: 3 anchors × 3 selected frames (early, mid, late),
    # upscaled for readability.
    print(f"\n──── Building comparison grid ────")
    cell = 320
    rows, cols = len(runs), 3
    pad = 26
    margin = 12
    grid_w = cols * cell + (cols + 1) * margin
    grid_h = rows * (cell + pad + margin) + margin + pad
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()
    for ci, hdr in enumerate(["frame 2 (early)", "middle frame", "last frame"]):
        x = margin + ci * (cell + margin)
        draw.text((x + 4, 4), hdr, fill='gray', font=font)
    for r, (label, frames, mid) in enumerate(runs):
        y = margin + pad + r * (cell + pad + margin)
        picks = [frames[2], frames[mid], frames[-1]]
        for c, frame in enumerate(picks):
            x = margin + c * (cell + margin)
            grid.paste(Image.fromarray(frame).resize((cell, cell), Image.LANCZOS), (x, y))
            if c == 0:
                draw.text((x + 4, y - pad + 4), label,
                          fill='yellow' if label == 'anchor=None' else 'white',
                          font=font)
    out = OUT_DIR / "_compare_grid.png"
    grid.save(out)
    print(f"  saved: {out}")

    print(f"\n=== What to look for ===")
    print(f"  - WATERCOLOR / PAINTERLY → that's the legacy MaPE-shift bug;")
    print(f"    expect at anchor=2000 (Phase 5d findings).")
    print(f"  - SHARPNESS of fur, hat detail, water spray, eyes.")
    print(f"  - PROMPT ADHERENCE: gold-trimmed cap visible? satchel? hands? "
          f"paw motion?")
    print(f"  - If anchor=1000 produces noticeably sharper / more prompt-aligned")
    print(f"    output than anchor=None: ship mape_anchor=1000 as new t2v default.")
    print(f"  - If all three produce similar output: mrope asymmetry isn't the")
    print(f"    differential, and we need a wider hypothesis search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
