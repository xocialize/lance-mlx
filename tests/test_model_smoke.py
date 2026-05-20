"""Shape-only smoke tests for Lance model scaffolds.

Confirms each scaffold module instantiates, exposes the safetensors-key-matching
attribute names, and produces the right output shapes for plausible inputs.
Does NOT validate numerical correctness — that's Phase 2/3 work against the
Phase 0 parity oracle.

The point of these is to catch regressions in scaffold dimensions/key naming
during refactors. They run in <1 second total.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from lance_mlx.model import (
    FlowHead,
    LatentPosEmbed,
    TimestepEmbedder,
    VAEInputProjection,
)


# ----------------------------- FlowHead (D1) ------------------------------

def test_flow_head_default_dims():
    """FlowHead defaults: Linear(2048, 48, bias=True). Verified against
    Lance_3B safetensors `llm2vae.weight [48, 2048]` + `llm2vae.bias [48]`."""
    fh = FlowHead()
    assert fh.llm2vae.weight.shape == (48, 2048)
    assert fh.llm2vae.bias.shape == (48,)


def test_flow_head_forward_shape():
    """(B, T, 2048) → (B, T, 48) velocity prediction."""
    fh = FlowHead()
    h = mx.zeros((1, 4, 2048))
    v = fh(h)
    assert v.shape == (1, 4, 48)


def test_flow_head_bias_is_true():
    """Regression test for D1: HANDOFF.md initially said bias=False but the
    actual safetensors has both weight AND bias. If this fails, someone
    reverted D1 without reading the empirical evidence."""
    fh = FlowHead()
    assert fh.llm2vae.bias is not None, (
        "llm2vae must have a bias — see notes/phase1a_keys.md D1"
    )


# --------------------- VAEInputProjection (D3, new) -----------------------

def test_vae_input_projection_default_dims():
    """Linear(48, 2048, bias=True). Verified against `vae2llm.weight
    [2048, 48]` + `vae2llm.bias [2048]`."""
    vp = VAEInputProjection()
    assert vp.vae2llm.weight.shape == (2048, 48)
    assert vp.vae2llm.bias.shape == (2048,)


def test_vae_input_projection_forward_shape():
    """(B, T_vae, 48) → (B, T_vae, 2048)."""
    vp = VAEInputProjection()
    x = mx.zeros((1, 16, 48))
    out = vp(x)
    assert out.shape == (1, 16, 2048)


def test_vae_input_projection_symmetric_with_flow_head():
    """vae2llm and llm2vae are SHAPE-symmetric (not weight-shared).
    Composing them on zeros should produce a 2048→48 round-trip of the
    correct shape (semantic correctness is tested in Phase 3 parity)."""
    vp = VAEInputProjection()
    fh = FlowHead()
    x = mx.zeros((1, 4, 48))
    h = vp(x)         # (1, 4, 2048)
    v = fh(h)         # (1, 4, 48)
    assert v.shape == x.shape


# ---------------------- LatentPosEmbed (D4, new) --------------------------

def test_latent_pos_embed_default_shape():
    """pos_embed [max_latent_size**2, hidden_size] = [4096, 2048] with shipped
    `--max_latent_size 64`. Verified against `latent_pos_embed.pos_embed
    [4096, 2048]`."""
    lpe = LatentPosEmbed()
    assert lpe.pos_embed.shape == (4096, 2048)


def test_latent_pos_embed_lookup_shape():
    """Indexing N positions returns (N, hidden_size)."""
    lpe = LatentPosEmbed()
    positions = mx.array([0, 1, 64, 4095], dtype=mx.int32)
    out = lpe(positions)
    assert out.shape == (4, 2048)


def test_latent_pos_embed_custom_size():
    """Smaller grids should also work — useful for tests that don't need 64x64."""
    lpe = LatentPosEmbed(max_latent_size=8, hidden_size=64)
    assert lpe.pos_embed.shape == (64, 64)


# -------------------- TimestepEmbedder (existing, smoke) ------------------

def test_timestep_embedder_default_shape():
    """proj_in: (256 → 2048), proj_out: (2048 → 2048)."""
    te = TimestepEmbedder()
    assert te.proj_in.weight.shape == (2048, 256)
    assert te.proj_out.weight.shape == (2048, 2048)


def test_timestep_embedder_forward_shape():
    """(B,) timesteps → (B, hidden_size) embedding."""
    te = TimestepEmbedder()
    t = mx.array([0.0, 0.5, 1.0])
    emb = te(t)
    assert emb.shape == (3, 2048)
