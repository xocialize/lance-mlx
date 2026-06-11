"""Upstream-exact ViT preprocessing for x2t (understanding) tasks.

Replicates bytedance/Lance's validation preprocessing byte-exactly by using
the same torch/torchvision ops the upstream `ValidationDataset` composes
(`data/transforms.py` + `data/video/transforms/*` + `data/data_utils.py`,
Apache-2.0). This is NOT the HF Qwen2.5-VL smart-resize path — upstream:

  1. Loads the image, compositing RGBA onto white, converting to RGB.
  2. `BucketResize(max_area=resolution_vit², aspect-ratio buckets
     [21:9,16:9,4:3,1:1,3:4,9:16] at stride 16)` — implemented with
     torchvision `RandomResizedCrop(scale=(1,1), ratio=(r,r))`, which is a
     deterministic center-crop to the bucket AR + bicubic resize.
  3. `DivisibleCrop(28)` — center crop to 28-divisible dims.
  4. `Normalize(CLIP mean/std)` on the [0,1] tensor.
  5. Duplicates the single frame (T: 1 → 2) AFTER normalization.
  6. `patchify_video_with_merge(C,T,H,W; p=14, tp=2, ms=2)` → (N, 1176).

`resolution_vit` mapping (upstream `validation_dataset.py`):
  video_192p/image_256res → 224 · image_512res → 448 · image_768res → 672 ·
  video_360p → 476 · video_480p → 616.
The Phase-0 oracle capture ran x2t_image with RESOLUTION=video_480p → 616.

Returns HF-layout pixel_values + grid_thw, drop-in compatible with
mlx-vlm's `VisionModel.__call__(pixel_values, grid_thw)`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
ASPECT_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]

# RESOLUTION preset → resolution_vit (upstream validation_dataset.py)
RESOLUTION_VIT = {
    "video_192p": 224, "image_256res": 224,
    "image_512res": 448,
    "image_768res": 672,
    "video_360p": 476,
    "video_480p": 616,
}


def load_image_upstream(path) -> Image.Image:
    """Upstream image load: palette→RGBA, RGBA composited onto white, RGB."""
    image = Image.open(path)
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    else:
        image = image.convert("RGB")
    return image


def preprocess_und_image(
    image: Image.Image,
    resolution: str = "video_480p",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the full upstream vit-stream pipeline on a PIL image.

    Returns:
        pixel_values: float32 (num_patches, 1176) in HF/mlx-vlm layout.
        grid_thw:     int32 (1, 3) = (t=1, h_patches, w_patches).
    """
    import torch
    from torchvision.transforms import (
        Compose, InterpolationMode, Normalize, RandomResizedCrop,
    )
    from torchvision.transforms.functional import center_crop, to_tensor

    resolution_vit = RESOLUTION_VIT[resolution]
    max_area = resolution_vit ** 2

    buckets, bucket_ratios = _init_buckets(ASPECT_RATIOS, max_area, stride=16)
    W, H = image.size
    idx = int(np.abs((W / H) - bucket_ratios).argmin())
    bh, bw = buckets[idx]
    bucket_ratio = bw / bh

    # Upstream BucketResize: torchvision RandomResizedCrop with scale=(1,1)
    # and a fixed ratio — deterministic given the input size. BICUBIC comes
    # from NaResize's default interpolation, passed through.
    resizer = RandomResizedCrop(
        size=(bh, bw), scale=(1, 1), ratio=(bucket_ratio, bucket_ratio),
        interpolation=InterpolationMode.BICUBIC,
    )
    t = to_tensor(resizer(image))                      # (C, bh, bw) in [0,1]

    # DivisibleCrop(28)
    ch, cw = t.shape[-2] - (t.shape[-2] % 28), t.shape[-1] - (t.shape[-1] % 28)
    t = center_crop(t, (ch, cw))

    t = Normalize(CLIP_MEAN, CLIP_STD)(t)              # CLIP stats

    # (C,H,W) → (C,T,H,W) with the frame duplicated (upstream: vit_video
    # stream for element_dtype=="image" does video_tensor.repeat(1,2,1,1)).
    video = t.unsqueeze(1).repeat(1, 2, 1, 1)          # (C, 2, H, W)

    patches = _patchify_video_with_merge(video, 14, 2, merge_size=2)
    gh, gw = ch // 14, cw // 14
    grid_thw = np.array([[1, gh, gw]], dtype=np.int32)
    return patches.numpy().astype(np.float32), grid_thw


def _init_buckets(aspect_ratio_names, max_area, stride):
    """Verbatim port of upstream BucketResize.init_buckets."""
    import math
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


def _patchify_video_with_merge(video, spatial_patch_size, temporal_patch_size,
                               merge_size=2):
    """Verbatim port of upstream data_utils.patchify_video_with_merge
    (einops rearrange replaced with permute)."""
    video = video.permute(1, 0, 2, 3)                  # C T H W → T C H W
    T, C, H, W = video.shape
    p, tp, ms = spatial_patch_size, temporal_patch_size, merge_size

    gt, gh, gw = T // tp, H // p, W // p
    video = video.reshape(gt, tp, C, gh // ms, ms, p, gw // ms, ms, p)
    video = video.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = video.reshape(gt * gh * gw, C * tp * p * p)
    return patches
