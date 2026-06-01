"""Portable equivalence test for the single-node generation optimization.

This is the CI-friendly regression guard for the optimized generate() path used
by BOTH the t2i PR and the t2v PR — the mechanism lives in the shared
``LanceModel`` methods (``prefill_prefix`` / ``gen_loop_forward`` /
``free_und_tower``), which t2v reuses unchanged, so one test covers both.

It builds a TINY, randomly-initialized Lance backbone from a ``TextConfig`` —
**no model weights, no tokenizer, no VAE, no downloads** — and asserts that the
optimized path reproduces the naive full-sequence forward for the latent (GEN)
tokens:

    baseline:   one full forward over [text prefix ‖ latent block] with the
                asymmetric block mask (text causal-only; latent attends to all
                prefix + all latent), then GEN final-norm + flow head.

    optimized:  prefill_prefix(text) -> free_und_tower() -> gen_loop_forward(latent),
                then GEN final-norm + flow head.

The two are exact in exact arithmetic: text tokens never attend to latents, so
the per-layer text-prefix K/V evolve independently of the latents and the
prefill reproduces them layer-for-layer; the latent hidden states then match by
induction. We run in fp32 and assert agreement to 1e-4.

This exercises the prefix-cache concat, the GEN-only routing, the unmasked
latent SDPA, and that ``free_und_tower`` actually deletes the UND weights while
``gen_loop_forward`` still runs.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model.lance_llm import LanceModel

# Tiny dims. head_dim = hidden_size // n_heads = 16, so mrope_section must sum to
# head_dim/2 = 8 ([2,3,3]). GQA with n_kv_heads < n_heads exercises the broadcast.
HIDDEN, N_HEADS, N_KV, LAYERS = 64, 4, 2, 3
TEXT_PAD, NOISY_VAE = 0, 3   # position_group buckets: TEXT -> UND, NOISY_VAE -> GEN


def _tiny_config() -> TextConfig:
    return TextConfig(
        model_type="qwen2_5_vl",
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        intermediate_size=128,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        rms_norm_eps=1e-6,
        vocab_size=128,
        rope_theta=1e6,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
    )


def _block_mask(P: int, n_lat: int) -> mx.array:
    """Asymmetric additive mask (1,1,T,T): text query i attends text key j<=i
    (causal, never latent); latent query attends to every prefix + latent key.
    Mirrors pipeline _build_block_mask: allowed = (i>=j) | (both in latent)."""
    T = P + n_lat
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    both_latent = (i >= P) & (j >= P)
    allowed = (i >= j) | both_latent
    add = np.where(allowed, 0.0, -np.inf).astype(np.float32)
    return mx.array(add).reshape(1, 1, T, T)


def _causal_mask(P: int) -> mx.array:
    m = np.triu(np.full((P, P), -np.inf, dtype=np.float32), k=1)
    return mx.array(m).reshape(1, 1, P, P)


@pytest.mark.parametrize("P,n_lat", [(5, 6), (3, 16)])  # image-like and bigger (video-like) latent grids
def test_optimized_matches_baseline_velocity(P: int, n_lat: int):
    mx.random.seed(0)
    cfg = _tiny_config()
    model = LanceModel(cfg, num_latent_positions=64)

    B, T = 1, P + n_lat
    prefix = mx.random.normal((B, P, HIDDEN))
    latent = mx.random.normal((B, n_lat, HIDDEN))
    full = mx.concatenate([prefix, latent], axis=1)

    pos = mx.broadcast_to(mx.arange(T).reshape(1, 1, T), (3, B, T))  # (3,B,T) mRoPE coords
    pg_full = mx.array([TEXT_PAD] * P + [NOISY_VAE] * n_lat)
    pg_prefix = mx.array([TEXT_PAD] * P)

    # --- baseline: one full forward, then GEN head on the latent tokens --------
    h_full = model(
        inputs_embeds=full, position_ids=pos, position_group=pg_full,
        mask=_block_mask(P, n_lat), cache=None,
    )
    v_base = model.llm2vae(h_full[:, P:, :])
    mx.eval(v_base)  # materialize BEFORE freeing the UND tower (graph references it)

    # --- optimized: prefill prefix -> free UND -> gen-only latent loop ----------
    caches = model.prefill_prefix(
        prefix, position_ids=pos[:, :, :P], position_group=pg_prefix,
        mask=_causal_mask(P),
    )
    freed = model.free_und_tower()
    h_lat = model.gen_loop_forward(latent, position_ids=pos[:, :, P:], caches=caches)
    v_opt = model.llm2vae(h_lat)
    mx.eval(v_opt)

    # free_und_tower really deleted the UND weights (and gen_loop still ran).
    assert not hasattr(model.layers[0], "mlp")
    assert not hasattr(model.layers[0].self_attn, "q_proj")
    assert not hasattr(model, "lm_head")
    assert "freed_bytes" in freed

    assert v_opt.shape == v_base.shape == (B, n_lat, 48)
    max_abs = float(mx.abs(v_opt - v_base).max())
    assert max_abs < 1e-4, f"optimized vs baseline velocity diverged: max|Δ|={max_abs:.2e}"


@pytest.mark.parametrize("P,n_lat", [(5, 6)])
def test_gen_loop_with_und_resident_matches_baseline(P: int, n_lat: int):
    """free_und=False path: the GEN-only loop is gated on the prefix caches, NOT
    on the UND tower being freed, so it must reproduce the baseline velocity with
    the UND weights STILL RESIDENT (no free_und_tower call). This is the
    guarantee the auto-detect relies on — keeping the tower never changes output."""
    mx.random.seed(0)
    cfg = _tiny_config()
    model = LanceModel(cfg, num_latent_positions=64)

    B, T = 1, P + n_lat
    prefix = mx.random.normal((B, P, HIDDEN))
    latent = mx.random.normal((B, n_lat, HIDDEN))
    full = mx.concatenate([prefix, latent], axis=1)
    pos = mx.broadcast_to(mx.arange(T).reshape(1, 1, T), (3, B, T))
    pg_full = mx.array([TEXT_PAD] * P + [NOISY_VAE] * n_lat)
    pg_prefix = mx.array([TEXT_PAD] * P)

    h_full = model(
        inputs_embeds=full, position_ids=pos, position_group=pg_full,
        mask=_block_mask(P, n_lat), cache=None,
    )
    v_base = model.llm2vae(h_full[:, P:, :])

    caches = model.prefill_prefix(
        prefix, position_ids=pos[:, :, :P], position_group=pg_prefix,
        mask=_causal_mask(P),
    )
    # NO free_und_tower() — UND stays resident (the free_und=False / keep case).
    h_lat = model.gen_loop_forward(latent, position_ids=pos[:, :, P:], caches=caches)
    v_opt = model.llm2vae(h_lat)
    mx.eval(v_base, v_opt)

    # The UND weights are still present (we did NOT free them).
    assert hasattr(model.layers[0], "mlp")
    assert hasattr(model.layers[0].self_attn, "q_proj")
    max_abs = float(mx.abs(v_opt - v_base).max())
    assert max_abs < 1e-4, f"UND-resident gen loop diverged: max|Δ|={max_abs:.2e}"


@pytest.mark.parametrize("P,n_lat", [(5, 6), (3, 16)])
def test_und_only_prefill_is_bit_identical(P: int, n_lat: int):
    """Deferred mode: the UND-only prefill must capture K/V bit-identical to the
    full-routed prefill. The prefix is all-TEXT, so the full path's
    ``mx.where(gen_mask, *_moe_gen(x), *und(x))`` returns the UND branch verbatim
    (no blending) — and the UND-only path computes exactly that branch while never
    touching a ``*_moe_gen`` weight (so it would not materialize a deferred GEN
    tower). Same op, same inputs → max|Δ| = 0."""
    mx.random.seed(0)
    cfg = _tiny_config()
    model = LanceModel(cfg, num_latent_positions=64)

    B = 1
    prefix = mx.random.normal((B, P, HIDDEN))
    pos_pre = mx.broadcast_to(mx.arange(P).reshape(1, 1, P), (3, B, P))
    pg_prefix = mx.array([TEXT_PAD] * P)

    caches_full = model.prefill_prefix(
        prefix, position_ids=pos_pre, position_group=pg_prefix,
        mask=_causal_mask(P), und_only=False,
    )
    caches_und = model.prefill_prefix(
        prefix, position_ids=pos_pre, position_group=pg_prefix,
        mask=_causal_mask(P), und_only=True,
    )
    for cf, cu in zip(caches_full, caches_und):
        dk = float(mx.abs(cf.k_pre - cu.k_pre).max())
        dv = float(mx.abs(cf.v_pre - cu.v_pre).max())
        assert dk == 0.0 and dv == 0.0, f"und-only prefix K/V diverged: dk={dk:.2e} dv={dv:.2e}"


@pytest.mark.parametrize("P,n_lat", [(5, 6), (3, 16)])
def test_deferred_handoff_matches_baseline(P: int, n_lat: int):
    """The full deferred handoff — UND-only prefill → free_und_tower(deferred=True)
    (release UND WITHOUT the eager eval that would materialize GEN) →
    materialize_gen_tower() → GEN-only loop — reproduces the naive baseline velocity.
    Exercises the exact ordering the deferred pipeline uses."""
    mx.random.seed(0)
    cfg = _tiny_config()
    model = LanceModel(cfg, num_latent_positions=64)

    B, T = 1, P + n_lat
    prefix = mx.random.normal((B, P, HIDDEN))
    latent = mx.random.normal((B, n_lat, HIDDEN))
    full = mx.concatenate([prefix, latent], axis=1)
    pos = mx.broadcast_to(mx.arange(T).reshape(1, 1, T), (3, B, T))
    pg_full = mx.array([TEXT_PAD] * P + [NOISY_VAE] * n_lat)
    pg_prefix = mx.array([TEXT_PAD] * P)

    h_full = model(
        inputs_embeds=full, position_ids=pos, position_group=pg_full,
        mask=_block_mask(P, n_lat), cache=None,
    )
    v_base = model.llm2vae(h_full[:, P:, :])
    mx.eval(v_base)  # materialize before freeing UND (graph references it)

    caches = model.prefill_prefix(
        prefix, position_ids=pos[:, :, :P], position_group=pg_prefix,
        mask=_causal_mask(P), und_only=True,
    )
    freed = model.free_und_tower(deferred=True)
    gen = model.materialize_gen_tower()
    h_lat = model.gen_loop_forward(latent, position_ids=pos[:, :, P:], caches=caches)
    v_opt = model.llm2vae(h_lat)
    mx.eval(v_opt)

    # UND deleted; GEN materialized + loop ran.
    assert not hasattr(model.layers[0].self_attn, "q_proj")
    assert hasattr(model.layers[0].self_attn, "q_proj_moe_gen")
    assert "freed_bytes" in freed and "materialized_bytes" in gen
    max_abs = float(mx.abs(v_opt - v_base).max())
    assert max_abs < 1e-4, f"deferred handoff velocity diverged: max|Δ|={max_abs:.2e}"


@pytest.mark.parametrize("P,n_lat", [(5, 6), (3, 16)])
def test_free_gen_tower_after_velocity_preserves_output(P: int, n_lat: int):
    """The shed cascade's second half: ``free_gen_tower`` runs AFTER the denoise
    loop has produced the final velocity/latents and BEFORE the VAE decode, which
    never touches the MoT backbone. So shedding the GEN tower CANNOT change the
    output — the velocity is already materialized — it only reclaims the ~5.5 GB
    that would otherwise sit resident through decode.

    This asserts the contract directly: compute v_opt via the deferred handoff,
    materialize it, then ``free_gen_tower`` and confirm (a) v_opt is byte-identical
    to the baseline (unchanged by the shed), (b) every ``*_moe_gen`` weight + the
    GEN final norm + flow head + embeddings are gone, and (c) the backbone is now
    spent (a second ``gen_loop_forward`` raises)."""
    mx.random.seed(0)
    cfg = _tiny_config()
    model = LanceModel(cfg, num_latent_positions=64)

    B, T = 1, P + n_lat
    prefix = mx.random.normal((B, P, HIDDEN))
    latent = mx.random.normal((B, n_lat, HIDDEN))
    full = mx.concatenate([prefix, latent], axis=1)
    pos = mx.broadcast_to(mx.arange(T).reshape(1, 1, T), (3, B, T))
    pg_full = mx.array([TEXT_PAD] * P + [NOISY_VAE] * n_lat)
    pg_prefix = mx.array([TEXT_PAD] * P)

    h_full = model(
        inputs_embeds=full, position_ids=pos, position_group=pg_full,
        mask=_block_mask(P, n_lat), cache=None,
    )
    v_base = np.array(model.llm2vae(h_full[:, P:, :]).astype(mx.float32))

    caches = model.prefill_prefix(
        prefix, position_ids=pos[:, :, :P], position_group=pg_prefix,
        mask=_causal_mask(P), und_only=True,
    )
    model.free_und_tower(deferred=True)
    model.materialize_gen_tower()
    h_lat = model.gen_loop_forward(latent, position_ids=pos[:, :, P:], caches=caches)
    v_opt = model.llm2vae(h_lat)
    mx.eval(v_opt)                       # the loop output is now materialized
    v_opt_np = np.array(v_opt.astype(mx.float32))

    # Shed the GEN tower — exactly what the pipeline does right before decode.
    shed = model.free_gen_tower()

    # (a) Output is unchanged: the velocity computed before the shed is preserved.
    assert np.array_equal(v_opt_np, np.array(v_opt.astype(mx.float32)))
    assert float(np.abs(v_opt_np - v_base).max()) < 1e-4

    # (b) Both towers + heads are gone — the backbone is fully shed.
    attn0 = model.layers[0].self_attn
    assert not hasattr(attn0, "q_proj")           # UND (freed earlier)
    assert not hasattr(attn0, "q_proj_moe_gen")   # GEN (freed now)
    assert not hasattr(model.layers[0], "mlp_moe_gen")
    for attr in ("norm_moe_gen", "llm2vae", "embed_tokens",
                 "vae_in_proj", "latent_pos_embed", "time_embedder"):
        assert not hasattr(model, attr), f"{attr} should be freed"
    assert "freed_bytes" in shed and shed["freed_bytes"] >= 0
    assert getattr(model, "_gen_freed", False) is True

    # (c) The backbone is spent — a second denoise pass cannot run.
    with pytest.raises(Exception):
        model.gen_loop_forward(latent, position_ids=pos[:, :, P:], caches=caches)


def test_resolve_und_mode_bands(monkeypatch):
    """resolve_und_mode: explicit modes verbatim, deprecated free_und alias, and
    the auto-detect bands (keep / free / deferred) + unresolved fail-safe."""
    import lance_mlx.model.lance_llm as m
    GiB = 1024**3

    # 1. Explicit modes returned verbatim (no probe).
    for mode in ("keep", "free", "deferred"):
        assert m.resolve_und_mode(mode) == mode
    with pytest.raises(ValueError):
        m.resolve_und_mode("bogus")

    # 2. Deprecated free_und alias (only when und_mode is None/"auto").
    assert m.resolve_und_mode(None, free_und=True) == "free"
    assert m.resolve_und_mode(None, free_und=False) == "keep"
    assert m.resolve_und_mode("auto", free_und=True) == "free"
    # Explicit und_mode wins over the deprecated alias.
    assert m.resolve_und_mode("deferred", free_und=False) == "deferred"

    # 3. Auto-detect is BINARY off the device budget — keep or deferred, NEVER free.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 21 * GiB)
    assert m.resolve_und_mode("auto") == "keep"          # 24 GB+ machine
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 14 * GiB)
    assert m.resolve_und_mode(None) == "deferred"        # 16 GB Mac (ws ~14 GiB) — was free
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 10 * GiB)
    assert m.resolve_und_mode(None) == "deferred"        # 12 GB-and-below box
    # Exactly at the keep pivot (18 GiB) → keep; just below → deferred.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 18 * GiB)
    assert m.resolve_und_mode(None) == "keep"
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 18 * GiB - 1)
    assert m.resolve_und_mode(None) == "deferred"
    # CPU-only backend → host-RAM fallback.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: None)
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: 8 * GiB)
    assert m.resolve_und_mode(None) == "deferred"
    # Nothing resolves → fail-safe to non-destructive keep.
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: None)
    assert m.resolve_und_mode(None) == "keep"
    # Auto NEVER yields "free" across the whole budget range.
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: None)
    for gib in (4, 8, 12, 14, 16, 17, 18, 21, 32, 64):
        monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda gib=gib: gib * GiB)
        assert m.resolve_und_mode("auto") in ("keep", "deferred")


def test_resolve_memory_mode_bands(monkeypatch):
    """resolve_memory_mode: explicit modes verbatim, deprecated und_mode/free_und
    aliases (with a DeprecationWarning), and the binary auto bands + fail-safe."""
    import warnings

    import lance_mlx.model.lance_llm as m
    GiB = 1024**3

    # 1. Explicit modes returned verbatim (no probe, no warning).
    for mode in ("parallel", "relay"):
        assert m.resolve_memory_mode(mode) == mode
    with pytest.raises(ValueError):
        m.resolve_memory_mode("bogus")

    # 2. Deprecated und_mode alias (only when memory_mode is None/"auto") — warns.
    with pytest.warns(DeprecationWarning):
        assert m.resolve_memory_mode(None, und_mode="keep") == "parallel"
    with pytest.warns(DeprecationWarning):
        assert m.resolve_memory_mode(None, und_mode="deferred") == "relay"
    with pytest.warns(DeprecationWarning):
        assert m.resolve_memory_mode("auto", und_mode="free") == "relay"
    with pytest.raises(ValueError):
        m.resolve_memory_mode(None, und_mode="bogus")
    # Explicit memory_mode wins over the deprecated alias (no warning needed).
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here would fail
        assert m.resolve_memory_mode("relay", und_mode="keep") == "relay"

    # 3. Deprecated free_und bool alias — warns; und_mode takes precedence over it.
    with pytest.warns(DeprecationWarning):
        assert m.resolve_memory_mode(None, free_und=True) == "relay"
    with pytest.warns(DeprecationWarning):
        assert m.resolve_memory_mode(None, free_und=False) == "parallel"
    with pytest.warns(DeprecationWarning):
        assert m.resolve_memory_mode(None, und_mode="keep", free_und=True) == "parallel"

    # 4. Auto-detect is BINARY off the device budget — parallel or relay.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 21 * GiB)
    assert m.resolve_memory_mode("auto") == "parallel"   # 24 GB+ machine
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 14 * GiB)
    assert m.resolve_memory_mode(None) == "relay"        # 16 GB Mac (ws ~14 GiB)
    # Exactly at the pivot (18 GiB) → parallel; just below → relay.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 18 * GiB)
    assert m.resolve_memory_mode(None) == "parallel"
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 18 * GiB - 1)
    assert m.resolve_memory_mode(None) == "relay"
    # CPU-only backend → host-RAM fallback.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: None)
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: 8 * GiB)
    assert m.resolve_memory_mode(None) == "relay"
    # Nothing resolves → fail-safe to non-destructive parallel.
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: None)
    assert m.resolve_memory_mode(None) == "parallel"


def test_materialize_vae_defers_then_loads():
    """VAE deferral (relay): the decoder is left lazy at load and materialized at
    decode. ``_materialize_vae`` is a no-op (returns None) when not deferred, and
    when deferred it evals the decoder params + flips the flag exactly once.

    Weights-free: a tiny nn.Module stands in for the Wan2.2 decoder, and the
    pipeline method is exercised unbound (no real model/VAE/tokenizer load). The
    real relay-vs-parallel byte-identity is covered by the monitored val script."""
    from lance_mlx.pipeline.t2i import TextToImagePipeline

    class _FakeVAE:
        """Stand-in for Wan22VAEDecoder: just needs a ``parameters()`` tree for
        ``mx.eval`` (the method never touches the decoder's structure)."""
        def parameters(self):
            return {"conv2": {"weight": mx.zeros((256, 256))}}

    # (a) Not deferred → no-op, returns None, leaves the (absent) flag alone.
    parallel = type("P", (), {"_vae_deferred": False, "vae_decoder": _FakeVAE()})()
    assert TextToImagePipeline._materialize_vae(parallel) is None
    assert parallel._vae_deferred is False

    # (b) Deferred → evals params, flips flag to False, returns stats once.
    relay = type("R", (), {"_vae_deferred": True, "vae_decoder": _FakeVAE()})()
    stats = TextToImagePipeline._materialize_vae(relay)
    assert stats is not None and "materialized_bytes" in stats and "active_after" in stats
    assert relay._vae_deferred is False
    # Idempotent: a second call is a no-op now that the flag is cleared.
    assert TextToImagePipeline._materialize_vae(relay) is None


def test_resolve_free_und_manual_override(monkeypatch):
    """True/False always win — the device probe is never consulted."""
    import lance_mlx.model.lance_llm as m

    # Force the probes to "unknown" so only the override can decide.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: None)
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: None)
    assert m.resolve_free_und(True) is True
    assert m.resolve_free_und(False) is False


def test_resolve_free_und_auto(monkeypatch):
    """None auto-detects from the device budget, falls back to host RAM, and
    fail-safes to non-destructive (keep) when nothing resolves."""
    import lance_mlx.model.lance_llm as m

    GiB = 1024**3
    # Constrained device budget (e.g. 16 GB Mac → ~14 GiB working set) → free.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 14 * GiB)
    assert m.resolve_free_und(None) is True
    # Ample device budget (24 GB+ machine) → keep resident.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: 21 * GiB)
    assert m.resolve_free_und(None) is False
    # No device info (CPU-only backend) → fall back to host RAM; low → free.
    monkeypatch.setattr(m, "_device_memory_budget_bytes", lambda: None)
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: 8 * GiB)
    assert m.resolve_free_und(None) is True
    # Nothing resolves → fail-safe to non-destructive (keep the tower).
    monkeypatch.setattr(m, "_total_ram_bytes", lambda: None)
    assert m.resolve_free_und(None) is False
