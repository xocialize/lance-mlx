"""Modality-Aware Positional Encoding (MaPE) — verified against upstream source.

Upstream lives in `bytedance/Lance` at `data/common.py::shift_position_ids` —
NOT in the model code. It's a pure position-ID transform applied BEFORE mRoPE.
Behavior (verified 2026-05-19):

    - Modality 4 (image-gen) tokens: RE-ANCHORED so the segment's first
      temporal position becomes 1000. Subsequent positions follow at their
      original relative spacing.
    - Modality 3 (video-gen) tokens: RE-ANCHORED so segment starts at 2000.
    - Modality 1 (understanding ref) tokens: positions COPIED from modality-2
      (image-edit reference) when their counts match — for image-edit anchoring.
    - All other modalities (0=text, 2=clean-VAE-reference): untouched.

Key clarifications vs the original research/scaffold guess:
    - Offsets are HARDCODED CONSTANTS {1000, 2000}, NOT learned parameters.
      There are NO MaPE tensors in the safetensors — `llm_config.json` has
      no MaPE keys at all; the entire mechanism lives in this data utility.
    - Behavior is RE-ANCHORING (overwrite the segment start position), NOT
      additive offsetting. The paper text reads as "+ Δ_m" but the code
      computes `delta = 1000 - first_position` and adds that, which is
      mathematically a re-anchor.

This module is therefore a ~20-LOC pure function — no nn.Module, no parameters.
"""

from __future__ import annotations

import mlx.core as mx

# Verified modality-ID semantics in upstream `i_sample_modality`:
#   0 = text
#   1 = understanding-image / understanding-video ref (re-anchors to match modality 2)
#   2 = image-edit clean reference (untouched; serves as anchor for modality 1)
#   3 = video generation (re-anchored to 2000)
#   4 = image generation (re-anchored to 1000)
MODALITY_TEXT = 0
MODALITY_UND_REF = 1
MODALITY_EDIT_CLEAN_REF = 2
MODALITY_VIDEO_GEN = 3
MODALITY_IMAGE_GEN = 4

# Hardcoded per upstream `data/common.py::shift_position_ids`.
ANCHOR_IMAGE_GEN: int = 1000  # modality 4
ANCHOR_VIDEO_GEN: int = 2000  # modality 3


def shift_position_ids_mape(
    position_ids: mx.array,  # (B, 3, T) — (t, h, w) coordinates per token
    modality_ids: mx.array,  # (T,) int in {0..4} per upstream semantics
) -> mx.array:
    """Apply Lance MaPE re-anchoring to a position grid.

    Mirrors upstream `data/common.py::shift_position_ids` with `pro_type == 10`
    (the only branch with non-trivial body). Operates on the TEMPORAL axis
    (position_ids[:, 0, :]) only; h, w untouched.

    Args:
        position_ids: (B, 3, T) integer positions; layout matches Qwen2.5-VL
                      mRoPE expectation `[t, h, w]` along the 3-axis.
        modality_ids: (T,) per-token modality label.

    Returns:
        (B, 3, T) — temporal axis re-anchored per the rules above; other axes
        passed through unchanged.
    """
    out = mx.array(position_ids)  # copy
    t_axis = out[:, 0, :]  # (B, T) — temporal coordinates

    img_mask = modality_ids == MODALITY_IMAGE_GEN
    vid_mask = modality_ids == MODALITY_VIDEO_GEN

    # MLX 0.31 does not support boolean-mask indexing (`t_axis[:, img_mask]`),
    # so use argmax to find the first True position. argmax on a 0/1 int mask
    # returns the smallest index where the value is 1 (ties resolve left-most).
    # One scalar sync per re-anchor is acceptable at the prep stage.

    # Image-gen: re-anchor first image-gen position to ANCHOR_IMAGE_GEN.
    if bool(mx.any(img_mask).item()):
        first_img = int(mx.argmax(img_mask.astype(mx.int32)).item())
        img_first = t_axis[:, first_img:first_img + 1]  # (B, 1)
        shift = ANCHOR_IMAGE_GEN - img_first
        t_axis = mx.where(img_mask[None, :], t_axis + shift, t_axis)

    # Video-gen: re-anchor to ANCHOR_VIDEO_GEN.
    if bool(mx.any(vid_mask).item()):
        first_vid = int(mx.argmax(vid_mask.astype(mx.int32)).item())
        vid_first = t_axis[:, first_vid:first_vid + 1]
        shift = ANCHOR_VIDEO_GEN - vid_first
        t_axis = mx.where(vid_mask[None, :], t_axis + shift, t_axis)

    # Modality-1 alignment to modality-2 — upstream does this only when the
    # counts match (used for image-edit anchoring). Implement in the pipeline
    # layer that has segment-construction context; not generic enough here.
    # See `Lance.forward` upstream for the canonical site.

    out[:, 0, :] = t_axis
    return out
