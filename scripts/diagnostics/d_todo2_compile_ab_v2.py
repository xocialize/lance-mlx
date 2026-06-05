#!/usr/bin/env python3
"""TODO-2 A/B v2 — properly isolated mx.compile test on lance_model forward.

v1 monkey-patched `pipe.lance_model.__call__` at the instance level, which
doesn't intercept `self.lance_model(...)` syntax (Python dispatches through
type(...).__call__, not the instance attribute). v2 calls the compiled
wrapper directly so we know mx.compile is actually in the loop.

Builds the same per-step inputs the pipeline would build, then times:
  - 5 baseline forwards (uncompiled)
  - 5 compiled forwards via mx.compile(model)

Eliminates the pipeline plumbing as a confound.
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
N_FORWARDS = 6   # cond + uncond × 3 — enough to amortize JIT cost


def main() -> int:
    print(f"=== TODO-2 v2 — isolated mx.compile A/B on lance_model forward ===\n")

    print(f"Loading TextToImagePipeline ...")
    t0 = time.perf_counter()
    from lance_mlx.pipeline.t2i import TextToImagePipeline
    pipe = TextToImagePipeline.from_pretrained(
        lance_weights_dir=LANCE_WEIGHTS, vae_safetensors=VAE_SAFETENSORS,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

    # Build the per-step inputs ONCE via the existing pipeline state path.
    print(f"Building per-step inputs (cond state) ...")
    from lance_mlx.pipeline.t2i import T2I_INSTRUCTION
    h_lat = w_lat = HEIGHT // 16
    n_lat = h_lat * w_lat
    state = pipe._prepare_state(
        prompt=PROMPT, instruction=T2I_INSTRUCTION,
        n_lat=n_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
        latent_pos_base=None,
    )
    # Skip the cache-using "optimized" path (PR #6) so we exercise the
    # baseline lance_model.__call__ that compile_steps would target.
    if "caches" in state:
        state = dict(state); state["caches"] = None

    # Synthesize the inputs that _step_velocity would feed to lance_model.
    max_side = 64
    lpe_indices = mx.array(
        [r * max_side + c for r in range(h_lat) for c in range(w_lat)],
        dtype=mx.int32,
    )
    mx.random.seed(0xC0DE)
    latents = mx.random.normal((1, 1, h_lat, w_lat, 48)).astype(
        pipe.lance_model.embed_tokens.weight.dtype
    )
    t = mx.array(0.5, dtype=mx.float32)

    latents_flat = latents.reshape(1, n_lat, 48)
    pe = pipe.lance_model.latent_pos_embed(lpe_indices)[None, ...]
    t_emb = pipe.lance_model.time_embedder(t.reshape(1)).reshape(1, 1, -1)
    lat_embed = pipe.lance_model.vae_in_proj(latents_flat) + pe + t_emb

    # Build the inputs_embeds matching the baseline path.
    import numpy as np
    target_dtype = state["text_embeds"].dtype
    base = np.array(state["text_embeds"].astype(mx.float32))
    ins = np.array(lat_embed.astype(mx.float32))
    pos_np = np.array(state["latent_positions_arr"])
    base[:, pos_np, :] = ins
    inputs_embeds = mx.array(base).astype(target_dtype)
    mx.eval(inputs_embeds, state["text_embeds"], state["position_ids"],
            state["position_group"], state["mask"])
    print(f"  inputs ready (T={inputs_embeds.shape[1]})\n")

    # ── Baseline forward ──────────────────────────────────────────────────
    def baseline_fwd():
        h = pipe.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=state["position_ids"],
            position_group=state["position_group"],
            mask=state["mask"],
        )
        mx.eval(h)
        return h

    # ── Compiled forward ──────────────────────────────────────────────────
    @mx.compile
    def compiled_fwd():
        return pipe.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=state["position_ids"],
            position_group=state["position_group"],
            mask=state["mask"],
        )

    # Warm both to amortize Metal first-call.
    print(f"── Metal pre-warm ──")
    _ = baseline_fwd()
    print(f"  baseline pre-warm done\n")

    print(f"── Baseline ({N_FORWARDS} forwards) ──")
    times_base = []
    for i in range(N_FORWARDS):
        t0 = time.perf_counter()
        _ = baseline_fwd()
        dt = time.perf_counter() - t0
        times_base.append(dt)
        print(f"  forward {i+1}:  {dt*1000:>6.0f} ms")

    print(f"\n── Compiled ({N_FORWARDS} forwards; first call traces) ──")
    times_comp = []
    for i in range(N_FORWARDS):
        t0 = time.perf_counter()
        out = compiled_fwd()
        mx.eval(out)
        dt = time.perf_counter() - t0
        times_comp.append(dt)
        marker = " (JIT trace)" if i == 0 else ""
        print(f"  forward {i+1}:  {dt*1000:>6.0f} ms{marker}")

    # Output equivalence check.
    h_base = baseline_fwd()
    h_comp = compiled_fwd()
    mx.eval(h_base, h_comp)
    max_diff = float(mx.abs(h_base - h_comp).max())
    print(f"\n── Output equivalence ──")
    print(f"  max|baseline - compiled|: {max_diff:.2e}  "
          f"({'BIT-IDENTICAL' if max_diff == 0.0 else 'within tolerance' if max_diff < 1e-3 else 'DIVERGES'})")

    # Summary — skip first compiled call (JIT trace), compare steady state.
    base_mean = sum(times_base) / len(times_base)
    comp_steady = sum(times_comp[1:]) / len(times_comp[1:])
    speedup = (base_mean - comp_steady) / base_mean * 100
    jit_overhead = times_comp[0] - base_mean

    print(f"\n=== Summary ===")
    print(f"  baseline mean:       {base_mean*1000:>6.0f} ms/forward")
    print(f"  compiled steady:     {comp_steady*1000:>6.0f} ms/forward "
          f"(forwards 2-{N_FORWARDS}, excludes JIT trace)")
    print(f"  steady-state speedup: {speedup:+.1f}%")
    print(f"  JIT trace overhead:  {jit_overhead*1000:+.0f} ms one-time")

    print(f"\n=== Verdict ===")
    if max_diff > 1e-3:
        print(f"  ❌ Output diverges (max|Δ|={max_diff:.2e}). Don't ship.")
    elif speedup >= 5:
        print(f"  ✅ Steady-state speedup ≥ 5%. Worth shipping as opt-in kwarg.")
    elif speedup >= 0:
        print(f"  ~ Marginal gain ({speedup:+.1f}%). Skip integration; not worth the surface area.")
    else:
        print(f"  ❌ Compiled is SLOWER ({-speedup:.1f}% regression). Don't ship.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
