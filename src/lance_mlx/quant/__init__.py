"""Phase 5c-3 — AWQ quantization for Lance.

MLX port of Reza2kn/lance-quant's AWQ recipe. Provides alpha-search
+ scale fusion in MLX so we can quantize Lance to INT4 with intact
prompt-following (naive INT4/INT8 destroys image quality per Phase 5c-2).

Pipeline:
  1. collect activation statistics via forward hooks (calibration)
  2. for each fusion group (norm + consumer linears), run alpha-search
     to find the per-channel scale s that minimizes quantization error
  3. apply scale: norm.weight /= s, consumer.weight *= s (still bf16)
  4. hand the modified model to mlx-lm's nn.quantize for INT4 packing
"""
from lance_mlx.quant.awq import (
    awq_search_scale,
    apply_scale_to_norm_and_consumers,
    FUSION_GROUPS,
    NO_FUSE_LINEARS,
    QUANT_SUFFIXES,
)
from lance_mlx.quant.calibrate import (
    ActStats,
    ActStatsLinear,
    install_act_stats,
    save_act_stats,
    load_act_stats,
)

__all__ = [
    "awq_search_scale",
    "apply_scale_to_norm_and_consumers",
    "FUSION_GROUPS",
    "NO_FUSE_LINEARS",
    "QUANT_SUFFIXES",
    "ActStats",
    "ActStatsLinear",
    "install_act_stats",
    "save_act_stats",
    "load_act_stats",
]
