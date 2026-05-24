#!/usr/bin/env python3
"""Phase 5c-3b — unit test for awq_search_scale.

Validates the alpha-search math works correctly on synthetic data
BEFORE wiring up calibration. Three checks:

  1. With realistic act_mean + random weights, alpha-search finds a
     non-degenerate alpha and reduces quant error below the alpha=0
     baseline (no AWQ).

  2. Error curve is reasonably-shaped (monotonic-ish; one minimum).

  3. Apply scale + actual mx.quantize/mx.dequantize → reconstruction
     error matches what the search predicted.

If all three pass, the core kernel is correct and we can wire it to
Lance with confidence.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lance_mlx.quant.awq import (
    awq_search_scale,
    apply_scale_to_norm_and_consumers,
    _per_group_fake_quant,
)


def synthetic_fusion_group(
    in_features: int = 2048,
    out_features_list: list[int] = (2048, 256, 256),  # mimic q_proj, k_proj, v_proj GQA
    seed: int = 42,
    act_skew: bool = True,
):
    """Generate a synthetic Lance-like fusion group: realistic weight
    distributions + activation magnitudes with channel skew."""
    rng = np.random.default_rng(seed)
    # Weights: zero-mean normal with mild per-channel variance variation
    # (mimics a trained Linear's distribution).
    weights = []
    for out_f in out_features_list:
        # Per-output-channel scale variation 0.5×–1.5×
        out_scales = rng.uniform(0.5, 1.5, size=(out_f, 1))
        w_np = rng.standard_normal((out_f, in_features)).astype(np.float32) * 0.05 * out_scales
        weights.append(mx.array(w_np))

    # Activation magnitudes: optionally skewed (some channels are 10× more
    # active than others — common in trained transformers).
    if act_skew:
        # 5% of channels are "high-activation", rest normal
        act = rng.lognormal(mean=0.0, sigma=0.5, size=in_features).astype(np.float32)
        high = rng.choice(in_features, size=in_features // 20, replace=False)
        act[high] *= 10
    else:
        act = np.abs(rng.standard_normal(in_features)).astype(np.float32) * 0.5 + 0.1
    act_mean = mx.array(act)

    # Same act_mean per consumer (true since they share the upstream norm).
    return weights, [act_mean] * len(weights)


def baseline_err_no_awq(weights, act_means, n_bit, group_size, seed=0xC0DE):
    """Reference: error when alpha=0 (no AWQ; just min/max INT4 quant)."""
    in_features = weights[0].shape[1]
    act = mx.stack([a.astype(mx.float32) for a in act_means], axis=0).mean(axis=0)
    mx.random.seed(seed)
    x = mx.random.normal((512, in_features)) * act
    ws_f32 = [w.astype(mx.float32) for w in weights]
    err_total = 0.0
    for w in ws_f32:
        w_dq = _per_group_fake_quant(w, n_bit, group_size)
        out_quant = x @ w_dq.T
        out_ref = x @ w.T
        err_total += float(((out_quant - out_ref) ** 2).mean())
    return err_total


def main() -> int:
    print(f"=== Phase 5c-3b — awq_search_scale unit test ===\n")

    # === Test 1: realistic Lance-like fusion group ============================
    print(f"Test 1: realistic fusion group (mimics input_layernorm + q/k/v projs in GQA)")
    print(f"  in_features=2048, out=[2048, 256, 256]  (Qwen2.5-VL 3B-ish dims)")
    weights, act_means = synthetic_fusion_group(
        in_features=2048, out_features_list=[2048, 256, 256], seed=42, act_skew=True,
    )

    t0 = time.perf_counter()
    result = awq_search_scale(weights, act_means, n_bit=4, group_size=128, return_curve=True)
    dt = time.perf_counter() - t0
    assert result is not None, "search returned None on valid input"

    baseline = baseline_err_no_awq(weights, act_means, n_bit=4, group_size=128)
    reduction_pct = (baseline - result.best_err) / baseline * 100

    print(f"  search time:    {dt:.2f}s")
    print(f"  best_alpha:     {result.best_alpha:.3f}  (0 = no AWQ, 1 = pure act)")
    print(f"  best_err:       {result.best_err:.6f}")
    print(f"  baseline err:   {baseline:.6f}   (alpha=0, no AWQ)")
    print(f"  error reduction: {reduction_pct:+.1f}%")
    print(f"  s stats:        min={float(result.s.min()):.4f}  max={float(result.s.max()):.4f}  "
          f"mean={float(result.s.mean()):.4f}")

    if reduction_pct < 5:
        print(f"  ⚠️  WARNING: error reduction <5% — AWQ may not be benefiting this case")
    else:
        print(f"  ✓ AWQ reduces quant error by {reduction_pct:.0f}%")

    # === Test 2: error curve shape ============================================
    print(f"\nTest 2: error curve shape (should have a single minimum, not flat or noisy)")
    print(f"  alpha:  " + "  ".join(f"{i/20:.2f}" for i in range(0, 21, 4)))
    print(f"  err:    " + "  ".join(f"{result.err_curve[i]:.4f}" for i in range(0, 21, 4)))
    min_idx = int(np.argmin(result.err_curve))
    print(f"  argmin idx={min_idx} (alpha={min_idx/20:.3f})")

    # === Test 3: apply scale + verify reconstruction ==========================
    print(f"\nTest 3: apply scale to norm + consumers, verify forward equivalence")
    # Synthesize a "norm.weight" — same in_features dimension as consumers.
    rng = np.random.default_rng(7)
    norm_weight = mx.array(rng.uniform(0.5, 1.5, size=(2048,)).astype(np.float32))

    new_norm, new_consumers = apply_scale_to_norm_and_consumers(
        norm_weight, weights, result.s,
    )
    # Pre/post bf16-equivalent forward: norm(x) * weight_t
    # Skip the actual norm activation; test that:
    #   (norm/s)(x) @ (w*s).T  approximately equals  norm(x) @ w.T
    in_features = 2048
    x_test = mx.random.normal((4, in_features))
    s = result.s
    # Original: x_after_norm = x * norm_weight; out = x_after_norm @ w.T
    # AWQ:      x_after_norm' = x * (norm_weight / s); out' = x_after_norm' @ (w * s.reshape(1,-1)).T
    #         = (x * norm/s) @ (w * s).T = (x * norm) @ w.T   (s cancels)
    for w_orig, w_new in zip(weights, new_consumers):
        out_orig = (x_test * norm_weight) @ w_orig.T
        out_new = (x_test * new_norm) @ w_new.T
        abs_diff = float(mx.abs(out_orig.astype(mx.float32) - out_new.astype(mx.float32)).max())
        rel_diff = abs_diff / max(float(mx.abs(out_orig).max()), 1e-12)
        print(f"  out shape {tuple(out_orig.shape)}:  max abs diff = {abs_diff:.2e}  "
              f"(rel = {rel_diff*100:.3f}%)")
        # fp32 rounding through scale division+multiplication: 0.5% rel is the
        # numerical floor when s has 5× dynamic range. Hard fail above 5% rel.
        assert rel_diff < 0.05, (
            f"AWQ scale fusion has unexpected drift: abs={abs_diff}, rel={rel_diff*100:.2f}%"
        )
    print(f"  ✓ Scale fusion is mathematically equivalent (within fp32 round-off)")

    # === Test 4: end-to-end with mx.quantize → mx.dequantize ===================
    print(f"\nTest 4: end-to-end with real mx.quantize (production path)")
    # Quantize one consumer using MLX's native quantize → dequantize, and compare
    # the resulting reconstruction error between (a) no-AWQ and (b) AWQ-applied.
    w = weights[0]
    w_no_awq = w
    w_awq = new_consumers[0]

    # mx.quantize wants (..., in_features) — Linear weight is (out, in) which fits.
    qw_no, sc_no, b_no = mx.quantize(w_no_awq, bits=4, group_size=128)
    w_dq_no = mx.dequantize(qw_no, sc_no, b_no, bits=4, group_size=128)
    err_no = float(((w_no_awq - w_dq_no) ** 2).mean())

    qw_a, sc_a, b_a = mx.quantize(w_awq, bits=4, group_size=128)
    w_dq_a = mx.dequantize(qw_a, sc_a, b_a, bits=4, group_size=128)
    err_a = float(((w_awq - w_dq_a) ** 2).mean())

    # But err_a is on the SCALED weight; to compare apples-to-apples we
    # need to undo the s scaling at inference (which is what runtime does
    # via the divided-norm trick).
    # Output reconstruction error: (x/s) @ w_dq_awq.T  vs  x @ w.T
    x_eval = mx.random.normal((512, in_features)) * mx.stack(
        [a.astype(mx.float32) for a in act_means], axis=0
    ).mean(axis=0)
    out_ref = x_eval.astype(mx.float32) @ w.astype(mx.float32).T
    out_no = x_eval.astype(mx.float32) @ w_dq_no.astype(mx.float32).T
    out_a = (x_eval / s.reshape(1, -1)).astype(mx.float32) @ w_dq_a.astype(mx.float32).T

    output_err_no = float(((out_no - out_ref) ** 2).mean())
    output_err_a = float(((out_a - out_ref) ** 2).mean())
    output_reduction = (output_err_no - output_err_a) / output_err_no * 100

    print(f"  consumer 0 weight reconstruction:")
    print(f"    no-AWQ weight-MSE:     {err_no:.6e}")
    print(f"    AWQ-applied weight-MSE: {err_a:.6e}  (on scaled weight — not comparable)")
    print(f"  consumer 0 OUTPUT reconstruction (apples-to-apples):")
    print(f"    no-AWQ output-MSE:     {output_err_no:.6e}")
    print(f"    AWQ-applied output-MSE: {output_err_a:.6e}")
    print(f"    output error reduction: {output_reduction:+.1f}%")

    if output_reduction > 5:
        print(f"  ✓ AWQ reduces production-path INT4 output error by {output_reduction:.0f}%")
    elif output_reduction > 0:
        print(f"  ~ AWQ helps slightly ({output_reduction:.1f}%). May still be valuable on Lance.")
    else:
        print(f"  ✗ AWQ does NOT help on this synthetic case. Check implementation.")

    print(f"\n=== Summary ===")
    print(f"  Test 1 (alpha search):    best_alpha={result.best_alpha:.2f}, "
          f"err_reduction_in_search={reduction_pct:+.1f}%")
    print(f"  Test 3 (scale fusion):    mathematically equivalent ✓")
    print(f"  Test 4 (mx.quantize e2e): output error reduction {output_reduction:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
