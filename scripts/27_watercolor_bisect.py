#!/usr/bin/env python3
"""Phase 5i.2 — bisect which axis triggers the watercolor aesthetic.

Baseline (from Phase 5i replay — known PHOTOREAL):
    prompt:  "a calm ocean wave rolling onto a sandy beach"
    size:    256×256
    frames:  9 (T_lat=3)
    steps:   24
    CFG:     4.0
    seed:    0
    template: ours

Each variant changes EXACTLY ONE axis. The one that flips photoreal →
watercolor isolates the bug surface.

  V0 — BASELINE (re-run for sanity)
  VA — prompt content: swap to red-panda long prompt
  VB — frame count:    T_lat=3 → T_lat=5 (17 frames)
  VC — seed:           0 → 42
  VD — step count:     24 → 30

Wall-clock estimate: ~25-30s/variant × 5 = ~2.5 min total.
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
    OUT_DIR = Path("/tmp/lance_phase5i2")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OCEAN_PROMPT = "a calm ocean wave rolling onto a sandy beach"
    RED_PANDA_PROMPT = json.loads(ORACLE_PROMPT_FILE.read_text())["000000.mp4"]

    # Baseline config (RT's published config — known sharp)
    BASE = dict(
        prompt=OCEAN_PROMPT, num_frames=9, height=256, width=256,
        num_steps=24, cfg_scale=4.0, seed=0,
    )

    # Each variant: name, overrides on BASE
    variants = [
        ("V0_baseline_repeat",   {}),
        ("VA_red_panda_prompt",  {"prompt": RED_PANDA_PROMPT}),
        ("VB_T_lat_5_17f",       {"num_frames": 17}),
        ("VC_seed_42",           {"seed": 42}),
        ("VD_30_steps",          {"num_steps": 30}),
    ]

    print(f"┏━━ Phase 5i.2 — watercolor-trigger bisect ━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ baseline (RT-config, known photoreal):")
    for k, v in BASE.items():
        if k == "prompt":
            print(f"┃   {k:14s} = {v[:60]!r}")
        else:
            print(f"┃   {k:14s} = {v}")
    print(f"┃ variants (each flips ONE axis vs baseline):")
    for name, ovr in variants:
        if not ovr:
            print(f"┃   {name:24s} = (rerun baseline)")
        else:
            for k, v in ovr.items():
                if k == "prompt":
                    print(f"┃   {name:24s} = prompt → '{v[:50]}...'")
                else:
                    print(f"┃   {name:24s} = {k} → {v}")
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
    for name, ovr in variants:
        cfg = {**BASE, **ovr}
        print(f"\n=== {name} ===")
        for k, v in ovr.items():
            print(f"  override: {k}={v if k != 'prompt' else v[:60]!r}")
        t0 = time.perf_counter()
        frames = pipe.generate(
            cfg["prompt"],
            num_frames=cfg["num_frames"], height=cfg["height"], width=cfg["width"],
            num_steps=cfg["num_steps"], cfg_scale=cfg["cfg_scale"],
            seed=cfg["seed"], verbose=False,
            mape_anchor=None, spatial_merge_size=1, rope_fp32=False,
            prompt_format="ours",
        )
        dt = time.perf_counter() - t0
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        # MP4
        mp4 = OUT_DIR / f"{name}.mp4"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))
        # Filmstrip
        strip_w = cfg["width"] * int(frames.shape[0])
        strip_h = cfg["height"]
        strip = Image.new('RGB', (strip_w, strip_h))
        for i, fr in enumerate(np.asarray(frames)):
            strip.paste(Image.fromarray(fr), (i * cfg["width"], 0))
        strip_path = OUT_DIR / f"{name}_strip.png"
        strip.save(strip_path)
        md5s[name] = hashlib.md5(strip_path.read_bytes()).hexdigest()
        strips[name] = strip
        print(f"  → {mp4}  ({mp4.stat().st_size/1e3:.0f} KB)")
        print(f"  → {strip_path}  md5={md5s[name][:16]}")

    # Build the verdict grid — each row is one variant, all scaled to same width
    print(f"\n=== Building verdict grid ===")
    max_w = max(s.width for s in strips.values())
    target_h = 96  # uniform display height
    label_h = 26
    margin = 8
    grid_w = max_w + 2*margin
    grid_h = len(strips) * (target_h + label_h + margin) + margin
    grid = Image.new('RGB', (grid_w, grid_h), 'black')
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    y = margin
    for name, _ in variants:
        s = strips[name]
        # Scale to uniform height
        scaled_w = int(s.width * target_h / s.height)
        scaled = s.resize((scaled_w, target_h))
        x_off = margin + (max_w - scaled_w) // 2  # center horizontally
        draw.text((margin + 4, y), f"{name}  md5={md5s[name][:12]}", fill='white', font=font)
        grid.paste(scaled, (x_off, y + label_h))
        y += target_h + label_h + margin
    grid_path = OUT_DIR / "verdict_grid.png"
    grid.save(grid_path)
    print(f"  → {grid_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for name in md5s:
        print(f"┃ {name:26s} md5={md5s[name]}")
    print(f"┃")
    print(f"┃ Open {grid_path} and identify which row shows watercolor.")
    print(f"┃ That variant's changed axis = the bug trigger.")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
