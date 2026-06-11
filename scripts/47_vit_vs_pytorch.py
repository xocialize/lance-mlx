#!/usr/bin/env python3
"""ViT parity: mlx-vlm VisionModel vs HF PyTorch Qwen2.5-VL vision tower.

The MLX ViT (mlx-vlm, pinned f2e19dec) has only ever been verified against
the Python MLX port itself — never against PyTorch. This script loads the
SAME vit.safetensors into both towers, feeds byte-identical upstream-exact
pixels, runs both in fp32, and reports per-stage cosine / max-abs-diff on
the final merged features.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/47_vit_vs_pytorch.py --case 02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="02")
    ap.add_argument("--resolution", default="image_768res")
    ap.add_argument("--vit-weights", type=Path,
                    default=Path("/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-Video-bf16/vit.safetensors"))
    args = ap.parse_args()

    from lance_mlx.pipeline.upstream_und_preprocess import (
        load_image_upstream, preprocess_und_image,
    )
    img = load_image_upstream(
        f"tests/fixtures/images/image-understanding-case-{args.case}.png")
    pixels, grid = preprocess_und_image(img, resolution=args.resolution)
    print(f"case-{args.case}: pixels {pixels.shape}, grid {grid.tolist()}")

    # ---- PyTorch (HF) tower, fp32 ----------------------------------------
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLVisionConfig,
    )
    cfg = json.loads(Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16/config.json"
    ).read_text())
    vc = dict(cfg["vision_config"])
    vc.setdefault("in_chans", vc.pop("in_channels", 3))
    hf_cfg = Qwen2_5_VLVisionConfig(**{k: v for k, v in vc.items()
                                       if k != "model_type"})
    with torch.device("cpu"):
        pt_model = Qwen2_5_VisionTransformerPretrainedModel(hf_cfg)
    from safetensors.torch import load_file
    sd = load_file(str(args.vit_weights))
    sd = {k.removeprefix("vision_tower."): v for k, v in sd.items()}
    missing, unexpected = pt_model.load_state_dict(sd, strict=False)
    print(f"PT load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:6])
    if unexpected:
        print("  unexpected:", unexpected[:6])
    pt_model = pt_model.float().eval()

    with torch.no_grad():
        pt_out = pt_model(
            hidden_states=torch.from_numpy(pixels).float(),
            grid_thw=torch.from_numpy(grid.astype(np.int64)),
        )
        if not isinstance(pt_out, torch.Tensor):
            # transformers >= 5: merged features live in pooler_output
            # (last_hidden_state is the pre-merger stream).
            pt_out = pt_out.pooler_output
        pt_out = pt_out.numpy()
    print(f"PT features: {pt_out.shape}, norm {np.linalg.norm(pt_out):.2f}")

    # ---- MLX (mlx-vlm) tower, fp32 ----------------------------------------
    import mlx.core as mx
    import inspect
    from mlx_vlm.models.qwen2_5_vl.config import VisionConfig
    from mlx_vlm.models.qwen2_5_vl.vision import VisionModel
    fields = set(inspect.signature(VisionConfig).parameters)
    mv = dict(cfg["vision_config"])
    if "in_chans" in mv:
        mv["in_channels"] = mv.pop("in_chans")
    vision_cfg = VisionConfig(**{k: v for k, v in mv.items() if k in fields})
    mlx_model = VisionModel(vision_cfg)
    saved = mx.load(str(args.vit_weights))
    saved = {k.removeprefix("vision_tower."): v for k, v in saved.items()}
    saved = mlx_model.sanitize(saved)
    saved = {k: v.astype(mx.float32) for k, v in saved.items()}
    mlx_model.load_weights(list(saved.items()))
    mx.eval(mlx_model.parameters())

    mlx_out = np.array(
        mlx_model(mx.array(pixels, dtype=mx.float32), mx.array(grid)),
        copy=False,
    )
    print(f"MLX features: {mlx_out.shape}, norm {np.linalg.norm(mlx_out):.2f}")

    # ---- Compare -----------------------------------------------------------
    a, b = pt_out.reshape(-1), mlx_out.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    mad = float(np.abs(pt_out - mlx_out).max())
    # per-token worst cosine
    ta = pt_out / np.linalg.norm(pt_out, axis=-1, keepdims=True)
    tb = mlx_out / np.linalg.norm(mlx_out, axis=-1, keepdims=True)
    per_tok = (ta * tb).sum(-1)
    print(f"\noverall cosine: {cos:.6f}  max|diff|: {mad:.4f}")
    print(f"per-token cosine: min {per_tok.min():.6f}  "
          f"p1 {np.percentile(per_tok, 1):.6f}  mean {per_tok.mean():.6f}")
    return 0


if __name__ == "__main__":
    sys = __import__("sys")
    sys.exit(main())
