#!/usr/bin/env python3
"""Phase 5h — test RockTalk's minimal chat template vs ours.

From `RockTalk/Lance-3B-Video-MLX` HF model card:
    `<|im_start|> [prompt tokens] <|im_end|> <|vision_start|>
     [N latent placeholders] <|vision_end|>`

No system/user/assistant role tags, no T2V_INSTRUCTION prefix. This is
THEIR working pipeline's template. Since Phase 5f proved the bug is in
our pipeline code (RT weights × our pipeline = blurry baseline), the
chat-template difference is the most concrete piece of pipeline-code
divergence between RT (sharp) and us (blurry).

Test at 256²×17f red-panda-surfing seed=42. Compare against V0_baseline
from `scripts/23_gutcheck_phase5g.py`.
"""
from __future__ import annotations

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
    OUT_ROOT = Path("/tmp/lance_phase5h")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    GUT_DIR = Path("/tmp/lance_phase5g")  # for V0_baseline.png to compare against

    prompts = json.loads(ORACLE_PROMPT_FILE.read_text())
    prompt = prompts["000000.mp4"]

    variants = [
        ("V0_ours_template",     "ours"),
        ("V1_rocktalk_template", "rocktalk"),
    ]

    print(f"┏━━ Phase 5h chat-template A/B at 256²×17f ━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"┃ prompt: {prompt[:80]}...")
    print(f"┃ scale : 17f × 256×256, 30 steps, CFG=4.0, seed=42")
    print(f"┃ flags : MaPE=None, sms=1, rope_fp32=False")
    print(f"┃ variants:")
    for name, fmt in variants:
        print(f"┃   {name}: prompt_format={fmt}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"\n=== Loading pipeline (shared across variants) ===")
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
    import hashlib

    md5s = {}
    for name, fmt in variants:
        print(f"\n=== {name} (prompt_format={fmt!r}) ===")
        t0 = time.perf_counter()
        frames = pipe.generate(
            prompt,
            num_frames=17, height=256, width=256,
            num_steps=30, cfg_scale=4.0,
            seed=42, verbose=False,
            mape_anchor=None,
            spatial_merge_size=1,
            rope_fp32=False,
            prompt_format=fmt,
        )
        dt = time.perf_counter() - t0
        print(f"  generated {frames.shape[0]} frames in {dt:.1f}s")

        mp4 = OUT_ROOT / f"{name}.mp4"
        with imageio.get_writer(mp4, fps=12, codec="libx264") as w:
            for fr in frames:
                w.append_data(np.asarray(fr))
        print(f"  → {mp4} ({mp4.stat().st_size/1e3:.0f} KB)")

        mid = int(frames.shape[0] // 2)
        png = OUT_ROOT / f"{name}_midframe.png"
        Image.fromarray(np.asarray(frames[mid])).save(png)
        md5s[name] = hashlib.md5(png.read_bytes()).hexdigest()
        print(f"  → {png} (md5={md5s[name]})")

    # Side-by-side compare grid.
    print(f"\n=== Building compare grid ===")
    imgs = [(label, Image.open(OUT_ROOT / f"{label}_midframe.png"))
            for label, _ in variants]
    W, H = imgs[0][1].size
    pad = 30
    margin = 12
    grid = Image.new('RGB', (2*W + 3*margin, H + pad + 2*margin), 'black')
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i, (label, img) in enumerate(imgs):
        x = margin + i * (W + margin)
        y = margin + pad
        grid.paste(img, (x, y))
        draw.text((x+5, y - pad + 5), label, fill='white', font=font)
    grid_path = OUT_ROOT / "compare_grid.png"
    grid.save(grid_path)
    print(f"  → {grid_path}")

    print(f"\n┏━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for label, md5 in md5s.items():
        print(f"┃ {label:24s} md5={md5}")
    if len(set(md5s.values())) == 1:
        print(f"┃ → BYTE-IDENTICAL. Chat template made no difference.")
    else:
        print(f"┃ → DIFFERENT. Chat template changed pixels — visually inspect:")
        print(f"┃   {grid_path}")
    print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


if __name__ == "__main__":
    sys.exit(main())
