"""MLX port of Reza2kn/lance-quant's AWQ alpha-search + scale fusion.

PyTorch reference: notes/phase5n_diagnostics/phase5c3_awq_port/_reza_awq_apply.py

The math:

  Per fusion group {norm → consumer linears that share the same input
  activation distribution}, we want to find a per-input-channel scale s
  such that:
    1. The consumer weights' quantization error is reduced (multiplying
       by s spreads the dynamic range better per group)
    2. The norm's output magnitudes are preserved (divide by s upstream)

  Mathematically equivalent to upstream activation × downstream weight
  pre/post the linear transform — the scale fuses into the preceding
  norm, so no extra ops at inference.

  Alpha balances activation-aware vs weight-aware scaling:
    s = (act_mean^alpha / w_max^(1-alpha)).clamp(1e-5)
    s = s / sqrt(s.max() * s.min())  # geomean ≈ 1

  alpha=0 → pure weight-magnitude balancing (no activation info used)
  alpha=1 → pure activation-magnitude balancing (no weight info)
  best   → balance chosen by grid search to minimize quant error

We use synthetic input `x = randn(512, in_features) * act_mean` rather
than real activations — this is what Reza2kn does and is sufficient
for the error-minimization signal.

The output (s) is then applied:
  norm.weight        /= s           # upstream absorbs the inverse
  consumer.weight    *= s.reshape(1, -1)   # downstream applies it
                                            (per-input-channel column scale)

After scale fusion the model is mathematically equivalent (modulo
numerical precision) but the consumers' weight distribution is
quantization-friendlier. Hand the modified model to nn.quantize() to
produce the final INT4 packed format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx


# ============================================================================
# Fusion groups (from Reza2kn/lance-quant/scripts/awq_apply.py:FUSION_GROUPS)
# ============================================================================

# Each entry: norm submodule → list of consumer linear submodules (all sharing
# the same activation distribution coming into them).
FUSION_GROUPS: dict[str, list[str]] = {
    "input_layernorm":                  ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "input_layernorm_moe_gen":          ["self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen", "self_attn.v_proj_moe_gen"],
    "post_attention_layernorm":         ["mlp.gate_proj", "mlp.up_proj"],
    "post_attention_layernorm_moe_gen": ["mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj"],
}

# Linears that don't have a clean fuse target → plain per-group INT4.
NO_FUSE_LINEARS: list[str] = [
    "self_attn.o_proj", "self_attn.o_proj_moe_gen",
    "mlp.down_proj",    "mlp_moe_gen.down_proj",
]

# All quant-target Linear suffixes (for activation-hook installation).
QUANT_SUFFIXES: tuple[str, ...] = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
)


# ============================================================================
# Core alpha-search
# ============================================================================


@dataclass
class AWQScaleResult:
    """Output of awq_search_scale: the per-input-channel scale + telemetry."""
    s: mx.array              # (in_features,) — the per-channel scale
    best_alpha: float        # winning alpha ∈ [0, 1]
    best_err: float          # quant error at best_alpha
    err_curve: list[float]   # per-alpha error (n_grid + 1 entries)


def _per_group_fake_quant(w_scaled: mx.array, n_bit: int, group_size: int) -> mx.array:
    """Asymmetric per-group quantize → dequantize. Returns dequantized weight
    of the same shape as w_scaled. Matches Reza's PyTorch fake-quant exactly.

    Algorithm:
      n_groups = in_features // group_size
      w_grp    = w.reshape(out, n_groups, group_size)
      max_v, min_v = per-group max / min
      scale    = (max - min) / qmax
      zero     = round(-min / scale)
      q        = clamp(round(w/scale + zero), 0, qmax)
      w_dq     = (q - zero) * scale
    """
    out_features, in_features = w_scaled.shape
    assert in_features % group_size == 0, (
        f"in_features={in_features} not divisible by group_size={group_size}"
    )
    n_groups = in_features // group_size
    w_grp = w_scaled.reshape(out_features, n_groups, group_size)

    max_v = w_grp.max(axis=-1, keepdims=True)   # (out, n_groups, 1)
    min_v = w_grp.min(axis=-1, keepdims=True)
    qmax = (1 << n_bit) - 1
    sc = mx.maximum((max_v - min_v) / qmax, 1e-5)
    z = mx.round(-min_v / sc)
    q = mx.clip(mx.round(w_grp / sc + z), 0, qmax)
    w_dq = (q - z) * sc
    return w_dq.reshape(out_features, in_features)


def awq_search_scale(
    consumer_weights: Sequence[mx.array],
    act_means: Sequence[mx.array | None],
    *,
    n_bit: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
    seed: int = 0xC0DE,
    return_curve: bool = False,
) -> AWQScaleResult | None:
    """Find the per-input-channel scale s that minimizes per-output INT
    quantization error, summed across all consumers in this fusion group.

    Args:
        consumer_weights: list of weight tensors, each (out_features_i, in_features).
            All consumers in a fusion group share the same in_features.
        act_means: list of (in_features,) tensors of mean(|activations|) per
            consumer. None entries are dropped. If all None → returns None.
        n_bit: target bit-width (default 4).
        group_size: quantization group size (default 128).
        n_grid: alpha grid resolution (default 20 → tests 21 alpha values).
        seed: RNG seed for synthetic input (default matches PyTorch reference).
        return_curve: include per-alpha error curve in result.

    Returns:
        AWQScaleResult, or None if no valid activation data was provided.
    """
    in_features = consumer_weights[0].shape[1]
    # All consumers must agree on in_features.
    for w in consumer_weights:
        assert w.shape[1] == in_features, "fusion group consumers disagree on in_features"

    # Average activation magnitudes across consumers (drops None entries).
    valid_acts = [a for a in act_means if a is not None and float(a.sum()) > 0.0]
    if not valid_acts:
        return None
    act = mx.stack([a.astype(mx.float32) for a in valid_acts], axis=0).mean(axis=0)
    act = mx.maximum(act, 1e-5)

    # Synthetic input scaled by per-channel activation magnitudes.
    mx.random.seed(seed)
    x = mx.random.normal((512, in_features)) * act

    # Baseline outputs (full precision).
    ws_f32 = [w.astype(mx.float32) for w in consumer_weights]
    org_outs = [x @ w.T for w in ws_f32]
    mx.eval(x, org_outs)

    # Per-output-channel max-abs of each weight, then average across consumers.
    w_max_list = [
        mx.maximum(mx.max(mx.abs(w), axis=0), 1e-5)   # (in_features,)
        for w in ws_f32
    ]
    w_max = mx.stack(w_max_list, axis=0).mean(axis=0)
    mx.eval(w_max)

    best_alpha = 0.0
    best_err = float("inf")
    err_curve: list[float] = []

    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = mx.maximum(act ** alpha / w_max ** (1.0 - alpha), 1e-5)
        s = s / mx.sqrt(s.max() * s.min())      # normalize geom-mean ≈ 1

        err_total = 0.0
        for w, org in zip(ws_f32, org_outs):
            w_scaled = w * s.reshape(1, -1)
            w_dq = _per_group_fake_quant(w_scaled, n_bit, group_size)
            # The runtime trick: at inference time, x → x/s (absorbed into
            # preceding norm), then matmul with w_dq. Compute the equivalent
            # here to score this alpha.
            out = (x / s.reshape(1, -1)) @ w_dq.T
            err_total += float(((out - org) ** 2).mean())

        err_curve.append(err_total)
        if err_total < best_err:
            best_err = err_total
            best_alpha = alpha

    # Recompute s at the winning alpha.
    s = mx.maximum(act ** best_alpha / w_max ** (1.0 - best_alpha), 1e-5)
    s = s / mx.sqrt(s.max() * s.min())
    mx.eval(s)

    return AWQScaleResult(
        s=s.astype(mx.float32),
        best_alpha=best_alpha,
        best_err=best_err,
        err_curve=err_curve if return_curve else [],
    )


def apply_scale_to_norm_and_consumers(
    norm_weight: mx.array,
    consumer_weights: Sequence[mx.array],
    s: mx.array,
) -> tuple[mx.array, list[mx.array]]:
    """Apply AWQ scale: norm.weight /= s, consumer.weight *= s on input axis.

    The pair of operations cancel out mathematically at inference (the
    norm output is divided by s, then immediately multiplied by s via the
    consumer column scale). The point is that the consumer weights now have
    a quantization-friendlier distribution (per-group dynamic ranges are
    more uniform).

    Returns new tensors (does NOT mutate inputs).
    """
    s_f32 = s.astype(mx.float32)
    new_norm = (norm_weight.astype(mx.float32) / s_f32).astype(norm_weight.dtype)
    new_consumers = [
        (w.astype(mx.float32) * s_f32.reshape(1, -1)).astype(w.dtype)
        for w in consumer_weights
    ]
    return new_norm, new_consumers
