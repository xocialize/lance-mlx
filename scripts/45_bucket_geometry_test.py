#!/usr/bin/env python3
"""Test the upstream-geometry hypothesis for the x2t_image chart-read gap.

Upstream Lance (bytedance/Lance) preprocesses the ViT input for x2t_image
with `image_768res` → resolution_vit=672 via VideoTransform(mode="bucket"):
  1. BucketResize(max_area=672², buckets from [21:9,16:9,4:3,1:1,3:4,9:16],
     stride=16) — torchvision RandomResizedCrop(scale=(1,1), ratio=(r,r)),
     which for our images deterministically CENTER-CROPS to the bucket AR
     then resizes (bicubic) to the bucket dims.
  2. DivisibleCrop(28) — center crop to 28-divisible dims.
  3. CLIP mean/std normalize (same as HF processor — not a differentiator).

All MLX runs used HF smart-resize (native AR, no crop). This script
replicates the upstream geometry offline with PIL and feeds the result to
OUR UnderstandingPipeline. If case-02 flips "43"→"29%", geometry is the
root cause of the MLX-vs-PyTorch oracle gap.

Run:
    cd /Volumes/DEV_ARCHIVE/lance-mlx && HF_HUB_DISABLE_XET=1 \
    uv run python /tmp/rocktalk-code/test_bucket_geometry.py --cases 01,02,04
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

LANCE = Path("/Volumes/DEV_ARCHIVE/lance-mlx")

ASPECT_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
MAX_AREA = 672 * 672
STRIDE = 16  # VideoTransform default stride_spatial


def init_buckets(aspect_ratio_names, max_area, stride):
    """Verbatim port of upstream BucketResize.init_buckets."""
    height_factor = width_factor = stride
    buckets, bucket_ratios = [], []
    for name in aspect_ratio_names:
        w, h = (int(v) for v in name.split(":"))
        aspect_ratio = w / h

        resize_width1 = math.sqrt(max_area * aspect_ratio)
        bucket_width1 = round(resize_width1 / width_factor) * width_factor
        resize_height1 = bucket_width1 / aspect_ratio
        bucket_height1 = round(resize_height1 / height_factor) * height_factor
        bucket_ratio1 = bucket_width1 / bucket_height1
        bucket_area1 = bucket_width1 * bucket_height1

        resize_height2 = math.sqrt(max_area / aspect_ratio)
        bucket_height2 = round(resize_height2 / height_factor) * height_factor
        resize_width2 = bucket_height2 * aspect_ratio
        bucket_width2 = round(resize_width2 / width_factor) * width_factor
        bucket_ratio2 = bucket_width2 / bucket_height2
        bucket_area2 = bucket_width2 * bucket_height2

        if abs(bucket_ratio1 - aspect_ratio) < abs(bucket_ratio2 - aspect_ratio):
            bucket_width, bucket_height = bucket_width1, bucket_height1
        elif abs(bucket_ratio1 - aspect_ratio) > abs(bucket_ratio2 - aspect_ratio):
            bucket_width, bucket_height = bucket_width2, bucket_height2
        else:
            if abs(bucket_area1 - max_area) <= abs(bucket_area2 - max_area):
                bucket_width, bucket_height = bucket_width1, bucket_height1
            else:
                bucket_width, bucket_height = bucket_width2, bucket_height2

        buckets.append((bucket_height, bucket_width))
        bucket_ratios.append(bucket_width / bucket_height)
    return buckets, np.array(bucket_ratios)


def bucket_geometry(img: Image.Image) -> Image.Image:
    """Replicate upstream's vit-stream geometry: BucketResize + DivisibleCrop(28)."""
    W, H = img.size
    buckets, ratios = init_buckets(ASPECT_RATIOS, MAX_AREA, STRIDE)
    idx = int(np.abs((W / H) - ratios).argmin())
    bh, bw = buckets[idx]
    r = bw / bh

    # RandomResizedCrop(scale=(1,1), ratio=(r,r)) deterministic behavior:
    # the sampled crop (full area at AR r) only fits if sqrt(area*r) <= W and
    # sqrt(area/r) <= H; otherwise torchvision falls back to a center crop at
    # the clamped ratio. Replicate both branches.
    area = W * H
    cw, ch = int(round(math.sqrt(area * r))), int(round(math.sqrt(area / r)))
    if not (0 < cw <= W and 0 < ch <= H):
        in_ratio = W / H
        if in_ratio < r:
            cw = W
            ch = int(round(W / r))
        elif in_ratio > r:
            ch = H
            cw = int(round(H * r))
        else:
            cw, ch = W, H
    left = (W - cw) // 2
    top = (H - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize((bw, bh), Image.Resampling.BICUBIC)

    # DivisibleCrop(28) — center crop to 28-divisible dims.
    dw, dh = (bw // 28) * 28, (bh // 28) * 28
    left = (bw - dw) // 2
    top = (bh - dh) // 2
    img = img.crop((left, top, left + dw, top + dh))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="01,02,04")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lance-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16"))
    ap.add_argument("--vit-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16/vit.safetensors"))
    args = ap.parse_args()

    oracle_dir = sorted(LANCE.glob("tests/fixtures/results/x2t_image_sample_*"))[0]
    oracle = json.loads((oracle_dir / "result.json").read_text())

    print("=== Loading OUR pipeline (bf16) ===")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.understanding import UnderstandingPipeline
    pipe = UnderstandingPipeline.from_pretrained(
        lance_weights_dir=args.lance_weights,
        vit_safetensors=args.vit_weights,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    for case in args.cases.split(","):
        fname = f"image-understanding-case-{case}.png"
        entry = next(e for e in oracle if e["image"].endswith(fname))
        img = Image.open(LANCE / "tests/fixtures/images" / fname).convert("RGB")
        pre = bucket_geometry(img)
        print(f"\n=== case {case}: {img.size} → bucket geometry {pre.size} ===")
        print(f"  Q: {entry['question']!r}")

        t0 = time.perf_counter()
        answer = pipe.generate(pre, entry["question"],
                               max_new_tokens=args.max_new_tokens)
        print(f"  bucket-geom answer: {answer!r}  ({time.perf_counter()-t0:.1f}s)")
        print(f"  PyTorch oracle:     {entry['answer']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
