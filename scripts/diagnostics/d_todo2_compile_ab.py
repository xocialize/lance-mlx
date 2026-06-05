#!/usr/bin/env python3
"""TODO-2 A/B harness — does mx.compile() on the per-step forward help t2i?

Measures wall-clock per-step on t2i at small scale (256², few steps) with
mx.compile applied to lance_model.__call__ vs without. Cold-call,
warm-call-1, warm-call-2 timings to separate JIT trace cost from
steady-state benefit.

Decision rule for TODO-2:
  - steady-state speedup >= 5% AND first-call delta is amortized by
    second generation: ship as opt-in kwarg, default OFF (warm-up matters
    for single-shot usage)
  - steady-state speedup < 5%: document as "investigated, no practical
    win" and close TODO-2 without adding a kwarg

No code changes to pipeline files unless the empirical data justifies it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[2]
LANCE_WEIGHTS = REPO_ROOT.parent / "lance-mlx-models" / "Lance-3B-bf16"
VAE_SAFETENSORS = LANCE_WEIGHTS / "vae.safetensors"

PROMPT = "A photorealistic tabby cat sitting in a sunlit garden."
HEIGHT = WIDTH = 256
NUM_STEPS = 10        # short loop — separates per-step from overhead
SEED = 42


def time_generate(pipe, label: str) -> float:
    mx.clear_cache()
    t0 = time.perf_counter()
    _ = pipe.generate(
        prompt=PROMPT, height=HEIGHT, width=WIDTH,
        num_steps=NUM_STEPS, cfg_scale=4.0, seed=SEED, verbose=False,
    )
    mx.eval(_)
    dt = time.perf_counter() - t0
    print(f"  {label:<32s}  {dt:>6.2f}s  ({dt/NUM_STEPS*1000:.0f} ms/step)")
    return dt


def main() -> int:
    print(f"=== TODO-2 — mx.compile A/B on t2i (256², {NUM_STEPS} steps) ===\n")

    print(f"Loading TextToImagePipeline (baseline, lance_model uncompiled) ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS, vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

    print(f"── Baseline (no mx.compile) ──")
    t_base_1 = time_generate(pipe, "baseline call 1 (cold)")
    t_base_2 = time_generate(pipe, "baseline call 2 (warm)")
    t_base_3 = time_generate(pipe, "baseline call 3 (warm)")

    # Swap lance_model.__call__ for a compiled wrapper.
    # mx.compile caches per-shape; first call traces, subsequent calls reuse.
    print(f"\n── Compiled (mx.compile around lance_model) ──")
    print(f"Wrapping lance_model.__call__ with mx.compile ...")
    original_call = pipe.lance_model.__call__
    compiled_call = mx.compile(original_call)

    # Monkey-patch __call__ at the bound-method level via __dict__.
    # nn.Module's __call__ is class-level; we replace on the instance.
    import types
    pipe.lance_model.__call__ = compiled_call

    try:
        t_comp_1 = time_generate(pipe, "compiled call 1 (cold + JIT)")
        t_comp_2 = time_generate(pipe, "compiled call 2 (warm)")
        t_comp_3 = time_generate(pipe, "compiled call 3 (warm)")
    finally:
        # Restore baseline so the pipeline isn't permanently mutated.
        pipe.lance_model.__call__ = original_call

    # ── Verdict ────────────────────────────────────────────────────────────
    base_warm = (t_base_2 + t_base_3) / 2
    comp_warm = (t_comp_2 + t_comp_3) / 2
    speedup_warm = (base_warm - comp_warm) / base_warm * 100
    first_call_overhead = (t_comp_1 - t_base_1) / t_base_1 * 100

    print(f"\n=== Summary ===")
    print(f"  baseline warm avg:   {base_warm:.2f}s")
    print(f"  compiled warm avg:   {comp_warm:.2f}s")
    print(f"  warm-state speedup:  {speedup_warm:+.1f}%")
    print(f"  first-call overhead: {first_call_overhead:+.1f}%  ({t_comp_1-t_base_1:+.1f}s on call 1)")

    print(f"\n=== Verdict ===")
    if speedup_warm >= 5:
        print(f"  ✅ Steady-state speedup ≥ 5%. Worth shipping as opt-in kwarg.")
        if first_call_overhead < 50:
            print(f"     First-call overhead is small enough that default=True is reasonable.")
        else:
            print(f"     First-call overhead is substantial — ship default=False to protect single-shot users.")
    elif speedup_warm >= 0:
        print(f"  ~ Marginal warm-state gain ({speedup_warm:.1f}%); not worth the integration noise.")
        print(f"     Document as 'investigated, no practical win for Lance' and close TODO-2.")
    else:
        print(f"  ❌ Compiled is SLOWER ({-speedup_warm:.1f}% regression). Don't ship.")
        print(f"     Possible causes: mx.compile trace breaks the nn.Module __call__ dispatch,")
        print(f"     or per-step shape varies and forces re-tracing every step.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
