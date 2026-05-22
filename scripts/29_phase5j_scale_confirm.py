#!/usr/bin/env python3
"""Phase 5j confirmation — does the fix hold at user-reference scale 480×704×17f?

Phase 5j proved `latent_pos_base=0` fixes watercolor at 256²×17f. This
re-runs the red-panda-surfing oracle prompt at 480×704×17f (the user's
reference scale that exposed the gap most clearly) with both base=None
(legacy/broken) and base=0 (fix). If V1 is sharp at 480×704, the fix is
robust to scale and we can ship it as the new default.

Wall-clock: ~3-4 min per variant at 480×704×17f.
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
    OUT_DIR = Path("/tmp/lance_phase5j_scale")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PROMPT = json.loads(ORACLE_PROMPT_FILE.read_text())["000000.mp4"]

    variants = [
        ("V0_480x704_legacy", None),
        ("V1_480x704_FIX",    0),
    ]

    print(f"┏━━ Phase 5j confirmation @ 480×704×17f ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompt: {PROMPT[:70]}...")
    print(f"┃ scale : 480×704 × 17f, 30 steps, CFG=4.0, seed=42, MaPE=None")
    print(f"┃ variants:")
    for name, base in variants:
        print(f"┃   {name:22s} latent_pos_base={base}")
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
    for name, base in variants:
        print(f"\n=== {name} (latent_pos_base={base}) ===")
        t0 = time.perf_counter()
        frames = pipe.generate(
            PROMPT,
            num_frames=17, height=480, width=704,
            num_steps=30, cfg_scale=4.0,
            seed=42, verbose=False,
            mape_anchor=None,
            latent_pos_base=base,
        )
        dt = time.perf_counter() - t0
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        mp4 = OUT_DIR / f"{name}.mp4"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))

        # Save mid frame as PNG for inspection
        mid = int(frames.shape[0] // 2)
        png = OUT_DIR / f"{name}_midframe.png"
        Image.fromarray(np.asarray(frames[mid])).save(png)
        md5s[name] = hashlib.md5(png.read_bytes()).hexdigest()
        strips[name] = Image.fromarray(np.asarray(frames[mid]))
        print(f"  → {mp4}  ({mp4.stat().st_size/1e3:.0f} KB)")
        print(f"  → {png}  md5={md5s[name][:16]}")

    # 2-row vertical compare grid
    if len(strips) == 2:
        print(f"\n=== Building 2-row compare grid ===")
        a = strips["V0_480x704_legacy"]
        b = strips["V1_480x704_FIX"]
        W, H = a.size
        label_h = 28
        margin = 8
        grid = Image.new('RGB', (W + 2*margin, 2*(H + label_h + margin) + margin), 'black')
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
        except Exception:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(grid)
        for i, (label, img) in enumerate([
            ("V0 LEGACY (base=text_len) — expected: watercolor", a),
            ("V1 FIX    (base=0)         — expected: photoreal", b),
        ]):
            y = margin + i * (H + label_h + margin)
            draw.text((margin + 4, y), label, fill='white', font=font)
            grid.paste(img, (margin, y + label_h))
        grid_path = OUT_DIR / "compare_grid.png"
        grid.save(grid_path)
        print(f"  → {grid_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for name, base in variants:
        if name in md5s:
            print(f"┃ {name:22s} (base={base})  md5={md5s[name]}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
