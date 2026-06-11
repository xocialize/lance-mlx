#!/usr/bin/env python3
"""Stage-by-stage ViT bisect: mlx-vlm vs HF PyTorch, fp32, same weights+pixels.

Follows the E6 method: compare every intermediate (window_index, cu_seqlens,
rotary, patch_embed, each of the 32 blocks, merger) — the first stage below
~0.9999 names the divergent op.

Usage:
    HF_HUB_DISABLE_XET=1 uv run python scripts/48_vit_stage_bisect.py --case 02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def tok_min_cos(a: np.ndarray, b: np.ndarray) -> float:
    ta = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    tb = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return float((ta * tb).sum(-1).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="02")
    ap.add_argument("--resolution", default="image_768res")
    ap.add_argument("--cpu", action="store_true",
                    help="pin MLX to the CPU stream (GPU fp32 matmul has "
                         "~8e-4 rel error on M5; CPU isolates algorithm "
                         "from backend accumulation noise)")
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

    cfg = json.loads(Path(
        "/Volumes/DEV_VOL1/VideoResearch/lance-mlx-models/Lance-3B-bf16/config.json"
    ).read_text())

    # ======================= PyTorch side (hooked) ==========================
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLVisionConfig,
    )
    vc = dict(cfg["vision_config"])
    vc.setdefault("in_chans", vc.pop("in_channels", 3))
    hf_cfg = Qwen2_5_VLVisionConfig(**{k: v for k, v in vc.items()
                                       if k != "model_type"})
    pt_model = Qwen2_5_VisionTransformerPretrainedModel(hf_cfg)
    from safetensors.torch import load_file
    sd = load_file(str(args.vit_weights))
    sd = {k.removeprefix("vision_tower."): v for k, v in sd.items()}
    pt_model.load_state_dict(sd, strict=False)
    pt_model = pt_model.float().eval()

    pt_grid = torch.from_numpy(grid.astype(np.int64))
    pt_stages: dict[str, np.ndarray] = {}
    hooks = []
    hooks.append(pt_model.patch_embed.register_forward_hook(
        lambda m, i, o: pt_stages.__setitem__("patch_embed", o.detach().numpy())))
    for bi, blk in enumerate(pt_model.blocks):
        hooks.append(blk.register_forward_hook(
            (lambda bi: lambda m, i, o:
             pt_stages.__setitem__(f"block_{bi:02d}", o.detach().numpy()))(bi)))
    hooks.append(pt_model.merger.register_forward_hook(
        lambda m, i, o: pt_stages.__setitem__("merger", o.detach().numpy())))

    with torch.no_grad():
        pt_out = pt_model(hidden_states=torch.from_numpy(pixels).float(),
                          grid_thw=pt_grid)
        pt_final = (pt_out if isinstance(pt_out, torch.Tensor)
                    else pt_out.pooler_output).numpy()
    for h in hooks:
        h.remove()

    pt_window_index, pt_cu_win = pt_model.get_window_index(pt_grid)
    pt_window_index = pt_window_index.numpy()
    pt_cu_win = torch.unique_consecutive(
        torch.tensor(pt_cu_win) if not isinstance(pt_cu_win, torch.Tensor)
        else pt_cu_win).numpy()
    pt_rot = pt_model.rot_pos_emb(pt_grid).numpy()

    # ======================= MLX side (inline) =============================
    import inspect
    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
        print("(MLX pinned to CPU stream)")
    from mlx_vlm.models.qwen2_5_vl.config import VisionConfig
    from mlx_vlm.models.qwen2_5_vl.vision import VisionModel
    fields = set(inspect.signature(VisionConfig).parameters)
    mv = dict(cfg["vision_config"])
    if "in_chans" in mv:
        mv["in_channels"] = mv.pop("in_chans")
    vision_cfg = VisionConfig(**{k: v for k, v in mv.items() if k in fields})
    m = VisionModel(vision_cfg)
    saved = mx.load(str(args.vit_weights))
    saved = {k.removeprefix("vision_tower."): v for k, v in saved.items()}
    saved = m.sanitize(saved)
    saved = {k: v.astype(mx.float32) for k, v in saved.items()}
    m.load_weights(list(saved.items()))
    mx.eval(m.parameters())

    grid_mx = mx.array(grid)
    hidden = m.patch_embed(mx.array(pixels, dtype=mx.float32))
    mlx_stages = {"patch_embed": np.array(hidden, copy=False)}

    rotary = m.rot_pos_emb(grid_mx)
    window_index, cu_window_seqlens = m.get_window_index(grid_mx)
    # dedup (as VisionModel.__call__ does)
    seen, idx = set(), []
    for i, x in enumerate(cu_window_seqlens.tolist()):
        if x not in seen:
            seen.add(x)
            idx.append(i)
    cu_window_seqlens = cu_window_seqlens[mx.array(idx, dtype=mx.int32)]

    mlx_window_index = np.array(window_index, copy=False)
    mlx_cu_win = np.array(cu_window_seqlens, copy=False)
    mlx_rot = np.array(rotary, copy=False)

    print("\n--- index/rotary parity ---")
    print(f"window_index equal: {np.array_equal(pt_window_index, mlx_window_index)}")
    print(f"cu_window_seqlens equal: "
          f"{np.array_equal(pt_cu_win.astype(np.int64), mlx_cu_win.astype(np.int64))}"
          f"  (pt {len(pt_cu_win)} mlx {len(mlx_cu_win)})")
    print(f"rotary cosine: {cos(pt_rot, mlx_rot):.8f}")

    seq_len = hidden.shape[0]
    smu = m.spatial_merge_unit
    hidden = hidden.reshape(seq_len // smu, smu, -1)[window_index, :, :].reshape(seq_len, -1)
    rotary_w = rotary.reshape(seq_len // smu, smu, -1)[window_index, :, :].reshape(seq_len, -1)

    cu_seqlens = []
    for i in range(grid_mx.shape[0]):
        sl = grid_mx[i, 1] * grid_mx[i, 2]
        cu_seqlens.append(mx.repeat(sl, grid_mx[i, 0]))
    cu_seqlens = mx.concatenate(cu_seqlens)
    cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
    cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

    for bi, blk in enumerate(m.blocks):
        cu_now = cu_seqlens if bi in m.fullatt_block_indexes else cu_window_seqlens
        hidden = blk(hidden, cu_seqlens=cu_now, rotary_pos_emb=rotary_w)
        mx.eval(hidden)
        mlx_stages[f"block_{bi:02d}"] = np.array(hidden, copy=False)

    merged = m.merger(hidden)
    mlx_stages["merger"] = np.array(merged, copy=False)
    reverse = mx.argsort(window_index, axis=0)
    mlx_final = np.array(merged[reverse, :], copy=False)

    # ======================= Compare ========================================
    print("\n--- stage cosines (overall / per-token-min) ---")
    names = (["patch_embed"] + [f"block_{i:02d}" for i in range(len(m.blocks))]
             + ["merger"])
    for n in names:
        a, b = pt_stages[n], mlx_stages[n]
        if a.shape != b.shape:
            print(f"{n}: SHAPE MISMATCH pt{a.shape} mlx{b.shape}")
            continue
        marker = ""
        c, t = cos(a, b), tok_min_cos(a, b)
        if t < 0.999:
            marker = "  <-- "
        print(f"{n}: {c:.6f} / {t:.6f}{marker}")

    print(f"\nfinal (post-reverse): {cos(pt_final, mlx_final):.6f} / "
          f"{tok_min_cos(pt_final, mlx_final):.6f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
