"""Weights-free equivalence guard for the MD per-window velocity refactor.

`t2v_multidiff._window_velocity` originally inlined its own GEN-only arm forward
+ CFG renorm block; it now reuses `TextToVideoPipeline._step_velocity` per arm
(the production single-shot path) + the shared `_md_common.cfg_combine`. The two
must be EXACTLY equivalent — same ops in the same order on the same values. This
test keeps the ORIGINAL inline implementation as the oracle and asserts the
refactored path reproduces it on a tiny random-init LanceModel (no weights, no
tokenizer, no VAE — same harness as tests/test_prefix_equiv.py).
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model.lance_llm import LanceModel
from lance_mlx.pipeline.t2v import TextToVideoPipeline
from lance_mlx.pipeline.t2v_multidiff import _window_velocity

HIDDEN, N_HEADS, N_KV, LAYERS = 64, 4, 2, 3
TEXT_PAD = 0


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


def _causal_mask(P: int) -> mx.array:
    m = np.triu(np.full((P, P), -np.inf, dtype=np.float32), k=1)
    return mx.array(m).reshape(1, 1, P, P)


class _ShimPipe:
    """Bare pipe exposing lance_model + the REAL TextToVideoPipeline._step_velocity
    (unbound method reuse — no tokenizer/VAE/__init__ needed)."""
    _step_velocity = TextToVideoPipeline._step_velocity

    def __init__(self, model: LanceModel):
        self.lance_model = model


def _arm_inline_oracle(lm: LanceModel, state: dict, z_win, t, lpe_indices,
                       W, h_lat, w_lat):
    """The pre-refactor _window_velocity arm, verbatim (the oracle)."""
    n_lat_win = W * h_lat * w_lat
    pe = lm.latent_pos_embed(lpe_indices)[None, ...]
    t_emb = lm.time_embedder(t.reshape(1)).reshape(1, 1, -1)
    lat_flat = z_win.reshape(1, n_lat_win, 48)
    lat_embed = lm.vae_in_proj(lat_flat) + pe + t_emb
    h = lm.gen_loop_forward(
        lat_embed.astype(state["text_embeds"].dtype),
        position_ids=state["position_ids_lat"],
        caches=state["caches"],
    )
    return lm.llm2vae(h).reshape(1, W, h_lat, w_lat, 48)


def _setup(seed=0, P=5, W=2, h=2, w=2):
    mx.random.seed(seed)
    model = LanceModel(_tiny_config(), num_latent_positions=64)
    pipe = _ShimPipe(model)
    n_lat = W * h * w

    def make_state():
        prefix = mx.random.normal((1, P, HIDDEN))
        pos = mx.broadcast_to(mx.arange(P + n_lat).reshape(1, 1, -1),
                              (3, 1, P + n_lat))
        caches = model.prefill_prefix(
            prefix, position_ids=pos[:, :, :P],
            position_group=mx.array([TEXT_PAD] * P), mask=_causal_mask(P))
        # exactly the keys _step_velocity's cached branch reads
        return {"caches": caches, "text_embeds": prefix,
                "position_ids_lat": pos[:, :, P:]}

    cond, uncond = make_state(), make_state()
    z_win = mx.random.normal((1, W, h, w, 48))
    t = mx.array(0.7321)
    lpe = mx.arange(n_lat, dtype=mx.int32)   # in-table; identical for both paths
    return pipe, model, cond, uncond, z_win, t, lpe, (W, h, w)


def test_window_velocity_cfg_matches_inline_oracle():
    pipe, model, cond, uncond, z_win, t, lpe, (W, h, w) = _setup()
    got = _window_velocity(
        pipe, z_win=z_win, t=t, lpe_indices=lpe,
        cond_state=cond, uncond_state=uncond, cfg_scale_step=4.0,
        cfg_renorm_type="channel", cfg_renorm_min=0.0, W=W, h_lat=h, w_lat=w)
    # oracle: inline arms + the ORIGINAL inline channel-renorm block
    v_c = _arm_inline_oracle(model, cond, z_win, t, lpe, W, h, w)
    v_u = _arm_inline_oracle(model, uncond, z_win, t, lpe, W, h, w)
    v_cfg = v_u + 4.0 * (v_c - v_u)
    nc = mx.sqrt(mx.sum(v_c * v_c, axis=-1, keepdims=True))
    nf = mx.sqrt(mx.sum(v_cfg * v_cfg, axis=-1, keepdims=True))
    exp = v_cfg * mx.clip(nc / (nf + 1e-8), 0.0, 1.0)
    mx.eval(got, exp)
    max_abs = float(mx.abs(got - exp).max())
    assert max_abs == 0.0, f"refactored window velocity diverged: max|Δ|={max_abs:.2e}"


def test_window_velocity_no_cfg_matches_cond_arm():
    pipe, model, cond, uncond, z_win, t, lpe, (W, h, w) = _setup(seed=1)
    # cfg_scale_step == 1.0 -> the uncond arm must not change the output
    got = _window_velocity(
        pipe, z_win=z_win, t=t, lpe_indices=lpe,
        cond_state=cond, uncond_state=uncond, cfg_scale_step=1.0,
        cfg_renorm_type="channel", cfg_renorm_min=0.0, W=W, h_lat=h, w_lat=w)
    exp = _arm_inline_oracle(model, cond, z_win, t, lpe, W, h, w)
    mx.eval(got, exp)
    assert float(mx.abs(got - exp).max()) == 0.0
