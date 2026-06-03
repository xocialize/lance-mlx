"""Dual-expert Mixture-of-Transformer-Experts backbone for Lance.

Architecture (VERIFIED against upstream source 2026-05-19 — supersedes the
original scaffold's open questions):

- 36 transformer layers (`num_hidden_layers=36`)
- Hidden 2048, intermediate 11008, 16 attention heads, 2 KV heads (GQA 8:1),
  `head_dim = 128`. Standard Qwen2.5-VL-3B dimensions.
- mRoPE with `rope_theta=1e6`, `mrope_section=[16, 24, 24]`. MaPE re-anchoring
  is applied to position_ids BEFORE the layer stack (see `mape.py`); no
  per-layer MaPE module.

Resolved questions (verified against `modeling/lance/qwen2_navit.py`):

1. **QKV projections are DUPLICATED per expert, NOT shared.** Each MoT layer
   holds two full attention substrates: `{q,k,v,o}_proj` for UND and
   `{q,k,v,o}_proj_moe_gen` for GEN. Upstream's `PackedAttentionMoT.__init__`
   creates the UND set via `super().__init__()` and ADDS the `_moe_gen`
   siblings. The shell flag `--copy_init_moe true` populates the GEN side
   from UND at load time.

2. **Per-expert QK-Norms: 4 RMSNorms per layer, 144 total across 36 layers.**
   `q_norm`, `k_norm`, `q_norm_moe_gen`, `k_norm_moe_gen` — each over
   `head_dim=128`. Tiny in params, but separate state-dict entries each.
   ⚠ NOTE: mlx-vlm's stock `Attention` does NOT have QK-norms — we add all
   four ourselves on top of the inherited q/k/v/o_proj.

   ⚠ Phase-1a empirical correction (2026-05-20): the **final** RMSNorm is
   ALSO per-expert. `model.norm` (UND) and `model.norm_moe_gen` (GEN) are
   BOTH present in the safetensors, each [2048]. Total RMSNorm count is
   therefore 146, not 144. Applied at the end of the layer stack, routed
   by `position_group` per-token (UND tokens → `self.norm`, GEN tokens
   → `self.norm_moe_gen`).

3. **Routing is strict per-token; NO cross-expert blending.** Each token
   passes through exactly one expert's input-layernorm → attention → MLP path
   and the result is written back via index assignment to a zero-init buffer
   which is then added to the residual. `freeze_und` optionally `.detach()`s
   UND outputs for fine-tuning GEN — not relevant for inference.

4. **LM head is UNTIED at runtime** despite `llm_config.json` saying
   `tie_word_embeddings: true`. `inference_lance.sh` passes
   `--tie_word_embeddings false`; the code calls `untie_lm_head()` after
   weight load. The safetensors contains a distinct `lm_head.weight` tensor
   (confirm in Phase-0 weight inspection).

5. **Per-expert prefixes in safetensors** follow the `_moe_gen` suffix
   pattern: UND keys carry no suffix (inherited from Qwen2 layer naming),
   GEN keys are siblings with `_moe_gen` appended (e.g. `q_proj_moe_gen`,
   `mlp_moe_gen`, `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`).

v1 implementation strategy (correctness-first, 2026-05-20):

Both expert paths are computed on ALL tokens at each routing point, then
merged with `mx.where`. This produces the same numerical output as upstream's
gather/scatter pattern but avoids scatter-assignment which complicates MLX's
functional autograd. The cost is 2× FLOPs on the dominant MLP — for inference
on Apple Silicon this is currently dwarfed by attention SDP at typical Lance
sequence lengths (8K–20K tokens for image/video gen). Optimization to
gather/scatter (or sorted-modality slicing) is a Phase 5 task once the
correctness baseline is validated against the Phase 0 oracle.

Subclassing strategy (verified to be feasible):

- We subclass `mlx_vlm.models.qwen2_5_vl.language.{Attention, Qwen2VLDecoderLayer}`
  with small deltas. Upstream's commit is pinned in `pyproject.toml`.
- `apply_multimodal_rotary_pos_emb` in mlx-vlm is a free function consuming
  `position_ids` — the clean seam for MaPE (pre-shift `position_ids` before
  the layer stack; no need to override the rotary embedding itself).

Class layout:

- `LanceMoTAttention(Attention)`: adds `q_proj_moe_gen`/`k_proj_moe_gen`/
  `v_proj_moe_gen`/`o_proj_moe_gen` and 4 QK-norms; routed `__call__`
  that takes `position_group` and merges per-token via `mx.where`.

- `LanceMoTLayer(Qwen2VLDecoderLayer)`: adds `mlp_moe_gen`,
  `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen`; routed
  forward that dispatches by `position_group`.

- `LanceModel`: full backbone — 36 LanceMoTLayer + embeddings + UNTIED
  lm_head + per-expert final RMSNorms + flow head + VAE bridge + latent
  pos embed + timestep embedder. NOT IMPLEMENTED THIS SESSION (Phase 1d).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_vlm.models.qwen2_5_vl.config import TextConfig
from mlx_vlm.models.qwen2_5_vl.language import (
    MLP,
    Attention,
    Qwen2VLDecoderLayer,
    apply_multimodal_rotary_pos_emb,
)

from .flow_head import FlowHead
from .latent_pos_embed import LatentPosEmbed
from .routing import expert_mask_from_position_group
from .time_embedder import TimestepEmbedder
from .vae_bridge import VAEInputProjection


# --- memory_mode auto-detect (device-budget pivot: parallel vs relay) -------
# A Lance generation runs three phases over DISJOINT weight sets — prefill→UND
# tower, denoise→GEN tower, decode→VAE decoder. memory_mode picks whether those
# components co-reside (parallel: reusable, needs headroom) or hand off one at a
# time (relay: never co-resident, fits a 16 GB Mac but single-shot). auto chooses
# between them from the device memory budget.

# Auto pivot: at/above this budget a machine can hold all components resident
# (~13.5 GB sustained, ~14 GB prefill peak) alongside normal co-resident apps, so
# auto runs parallel (reusable, no teardown). Below it, auto relays (towers never
# co-resident, ~10 GB peak). 18 GiB ≈ a 24 GB+ machine; a 16 GB Mac reports
# ws=14.0 GiB → relay.
_PARALLEL_MIN_BUDGET_BYTES = 18 * 1024**3


def _device_memory_budget_bytes() -> int | None:
    """Best-effort GPU/accelerator memory budget in bytes — VRAM on a discrete
    GPU (CUDA), unified RAM on Apple Silicon (Metal). This is the figure that
    governs whether the model can stay resident, and unlike system RAM it is
    correct on NON-UNIFIED memory: a Linux box with a discrete GPU holds the
    model in VRAM regardless of how much host RAM it has. Returns None on a
    CPU-only backend (no device to query), where the caller falls back to host
    RAM."""
    try:
        info_fn = getattr(mx, "device_info", None)
        if info_fn is None:
            metal = getattr(mx, "metal", None)
            info_fn = getattr(metal, "device_info", None) if metal is not None else None
        if info_fn is None:
            return None
        info = info_fn()
        for key in ("max_recommended_working_set_size", "memory_size"):
            val = info.get(key)
            if val:
                return int(val)
    except Exception:
        pass
    return None


def _total_ram_bytes() -> int | None:
    """Total host RAM in bytes — CPU-backend fallback only (used when there is no
    accelerator device to query). None if it cannot be determined."""
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (ValueError, AttributeError, OSError):
        pass
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        return int(out.strip())
    except Exception:
        return None


def resolve_memory_mode(memory_mode: str | None) -> str:
    """Resolve the memory strategy to one of ``"parallel" | "relay"``.

    The three phases of a Lance generation use DISJOINT weight sets — prefill→UND
    tower, denoise→GEN tower, decode→VAE decoder — and ``memory_mode`` chooses
    whether they co-reside or hand off:

    - ``"parallel"`` — all three components co-resident; nothing is freed. The
      pipeline is REUSABLE across calls (no teardown). For machines with headroom
      (24 GB+), where holding ~13.5 GB resident alongside normal apps is fine.
    - ``"relay"`` — a baton handoff: the two MoT towers are never co-resident and
      the VAE decoder is not loaded until decode. Load UND+shared (GEN lazy) →
      UND-only prefill → free UND → materialize GEN → denoise loop → shed GEN →
      materialize VAE → decode. Peak ≈ the single heaviest phase (~9-10 GB), so
      it fits a 16 GB Mac, but it is SINGLE-SHOT (the backbone is destroyed; the
      pipeline reloads on the next call). The default below the budget pivot.
    - ``"auto"``/``None`` — device budget: ``ws ≥ 18 GiB`` → parallel; below →
      relay. Keys off the GPU/accelerator budget so it is correct on non-unified
      memory (VRAM on a discrete GPU, unified RAM on Apple Silicon), with a
      host-RAM fallback on a CPU-only backend. Unresolved budget → parallel
      (fail-safe: non-destructive, reusable).

    ``relay`` is a **load-time** decision (the GEN tower and VAE are loaded lazily),
    so it must be resolved at ``from_pretrained``. Requesting ``relay`` at
    ``generate`` against a parallel-loaded model cannot un-materialize the towers;
    the pipeline falls back to an eager UND-free (same output + shed, no prefill-peak
    win)."""
    if memory_mode is not None and memory_mode != "auto":
        if memory_mode not in ("parallel", "relay"):
            raise ValueError(
                f"memory_mode must be 'auto'|'parallel'|'relay', got {memory_mode!r}"
            )
        return memory_mode
    budget = _device_memory_budget_bytes()
    if budget is None:
        budget = _total_ram_bytes()
    if budget is None:
        return "parallel"  # unknown device → non-destructive, reusable
    return "parallel" if budget >= _PARALLEL_MIN_BUDGET_BYTES else "relay"


def _broadcast_mask(position_group: mx.array, target_dtype) -> mx.array:
    """(T,) int position_group → (1, T, 1) bool mask for per-token routing.

    True == route to GEN expert; False == route to UND expert.
    Reshape lets it broadcast cleanly against (B, T, D)-shaped projections.
    """
    e_mask = expert_mask_from_position_group(position_group)  # (T,) int 0/1
    return (e_mask.reshape(1, -1, 1) == 1)


class LanceMoTAttention(Attention):
    """mlx-vlm Attention + `_moe_gen` projection siblings + 4 per-expert QK-norms.

    On top of stock Attention (q/k/v/o_proj + rotary_emb), adds:
        - q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen, o_proj_moe_gen
        - q_norm, k_norm, q_norm_moe_gen, k_norm_moe_gen (each RMSNorm over head_dim)

    The routed `__call__` takes a `position_group` tensor (per-token modality
    bucket). Tokens where `position_group >= CLEAN_VAE` (i.e., 2 or 3) route to
    GEN-side projections and norms; tokens 0/1 route to UND.

    Attention SDP itself is SHARED — there is one packed sequence and all
    tokens attend to all tokens. Only the *projections* and *norms* are
    duplicated per expert.
    """

    def __init__(self, args: TextConfig):
        super().__init__(args)  # q/k/v/o_proj + rotary_emb
        dim = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv_heads = args.num_key_value_heads or n_heads
        head_dim = dim // n_heads
        eps = args.rms_norm_eps

        # GEN-side projections (mirror UND with same dims/biases)
        self.q_proj_moe_gen = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(n_heads * head_dim, dim, bias=False)

        # 4 per-expert QK-norms (added on top of stock Attention, which has none).
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.q_norm_moe_gen = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm_moe_gen = nn.RMSNorm(head_dim, eps=eps)

        # P0a candidate (issue #2 / Phase 5g): when True, compute cos/sin and
        # the q*cos + rotate_half(q)*sin rotation in fp32 instead of inheriting
        # mlx-vlm's `Qwen2RotaryEmbedding`'s downcast (language.py:73 does
        # `cos.astype(x.dtype)` → bf16 in our run). Default: False (preserves
        # legacy behavior).
        self._rope_fp32 = False

        # Phase 5m / Issue #1: when True, promote Q/K/V to fp32 through the
        # entire RoPE + SDP path (downcast before o_proj). Hypothesis: at long
        # sequence lengths (n_lat ≥ 11,520) bf16 attention accumulation
        # produces silent semantic drift. Independent of _rope_fp32 (which
        # downcasts before SDP). Set via `LanceModel.set_attention_fp32(True)`.
        # Default: False.
        self._attention_fp32 = False

    def __call__(
        self,
        x: mx.array,                       # (B, L, D)
        position_group: mx.array,          # (T=L,) modality bucket
        mask: mx.array | None = None,
        cache=None,
        position_ids: mx.array | None = None,
        gen_only: bool = False,
        und_only: bool = False,
    ) -> mx.array:
        # Single-node t2i optimization: every loop token is NOISY_VAE→GEN, so
        # call ONLY the GEN projections/norms (no mx.where, no UND weights) and
        # attend over [cached text-prefix K/V ‖ fresh latent K/V]. This both
        # halves projection FLOPs and is the precondition for freeing the UND
        # tower (mx.where would otherwise evaluate the now-deleted UND branch).
        if gen_only:
            return self._gen_only_forward(x, cache, position_ids)
        # Deferred-load prefill: the prefix is all-TEXT (gen_mask all-False), so
        # the full routed mx.where always selects the UND branch — but mx.where is
        # not short-circuit, so it would still evaluate q_proj_moe_gen(x) etc. and
        # MATERIALIZE the deferred GEN tower. The UND-only path computes ONLY the
        # UND projections/norms — numerically identical K/V, zero _moe_gen touch.
        if und_only:
            return self._und_only_forward(x, mask, cache, position_ids)

        B, L, D = x.shape
        # (1, L, 1) bool — True = GEN expert
        gen_mask = _broadcast_mask(position_group, x.dtype)

        # --- Per-expert Q/K/V projection (both paths, merged) -----------------
        queries = mx.where(gen_mask, self.q_proj_moe_gen(x), self.q_proj(x))
        keys    = mx.where(gen_mask, self.k_proj_moe_gen(x), self.k_proj(x))
        values  = mx.where(gen_mask, self.v_proj_moe_gen(x), self.v_proj(x))

        # Reshape to (B, n_heads_or_kv_heads, L, head_dim)
        queries = queries.reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        keys    = keys.reshape   (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values  = values.reshape (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # --- Per-expert QK-norm (over head_dim, applied to (B, H, L, head_dim)) ---
        # Reshape mask for the new layout: (1, 1, L, 1)
        gen_mask_qk = gen_mask.reshape(1, 1, L, 1)
        queries = mx.where(gen_mask_qk, self.q_norm_moe_gen(queries), self.q_norm(queries))
        keys    = mx.where(gen_mask_qk, self.k_norm_moe_gen(keys),    self.k_norm(keys))

        # --- Position-aware rotary (uses post-MaPE position_ids from upstream) ----
        kv_seq_len = keys.shape[-2]
        if position_ids is None:
            offset = cache.offset if cache is not None else 0
            kv_seq_len += offset + (1 if cache is not None else 0)
            position_ids = mx.arange(L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        else:
            kv_seq_len += (cache.offset + 1) if cache is not None else 0

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        # Issue #1 / Phase 5m: attention_fp32 promotes Q/K/V to fp32 through
        # the entire RoPE + SDP path, downcasting only before o_proj. At long
        # sequence lengths (n_lat ≥ 11,520 e.g. 768²×17f) bf16 softmax
        # accumulation in SDP can produce silent semantic drift even when
        # latent stats are numerically bounded. fp32 attention is a candidate
        # mitigation. Independent of `_rope_fp32` (which only upcasts the
        # rotary calc and downcasts back before SDP).
        original_q_dtype = queries.dtype
        if self._attention_fp32:
            values_fp32 = values.astype(mx.float32)
            cos, sin = self.rotary_emb(values_fp32, position_ids)
            q_fp32 = queries.astype(mx.float32)
            k_fp32 = keys.astype(mx.float32)
            q_rot, k_rot = apply_multimodal_rotary_pos_emb(
                q_fp32, k_fp32, cos, sin, unqueeze_dim=1
            )
            # KEEP fp32 through SDP — do NOT downcast q/k/v before SDP call
            queries = q_rot
            keys = k_rot
            values = values_fp32
        elif self._rope_fp32:
            # P0a (Phase 5g): fp32 RoPE rotation only — downcast before SDP.
            values_fp32 = values.astype(mx.float32)
            cos, sin = self.rotary_emb(values_fp32, position_ids)
            q_fp32 = queries.astype(mx.float32)
            k_fp32 = keys.astype(mx.float32)
            q_rot, k_rot = apply_multimodal_rotary_pos_emb(
                q_fp32, k_fp32, cos, sin, unqueeze_dim=1
            )
            queries = q_rot.astype(original_q_dtype)
            keys = k_rot.astype(original_q_dtype)
        else:
            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin, unqueeze_dim=1
            )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # --- Shared SDP attention (full sequence, no per-expert split) -----------
        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Downcast back if we promoted to fp32 (o_proj weights are bf16).
        if self._attention_fp32 and output.dtype != original_q_dtype:
            output = output.astype(original_q_dtype)

        # --- Per-expert output projection (both paths, merged) -------------------
        return mx.where(gen_mask, self.o_proj_moe_gen(output), self.o_proj(output))

    def _gen_only_forward(
        self,
        x: mx.array,                       # (B, n_lat, D) — latent tokens ONLY
        cache,                             # PrefixKVCache (loop mode) — frozen prefix
        position_ids: mx.array,            # (3, B, n_lat) — latent grid coords (post-MaPE)
    ) -> mx.array:
        """GEN-only attention over latent queries, attending to [prefix ‖ latent].

        Mirrors the bf16/_rope_fp32/_attention_fp32 branches of ``__call__`` but
        uses ONLY the ``*_moe_gen`` weights. The cached prefix K/V is already
        post-RoPE (captured during prefill) and is NOT re-rotated here. No mask
        is needed: latent queries are allowed to attend to every prefix key
        (causal: text precedes latents) and every latent key (bidirectional);
        the dropped trailing ``<|vision_end|>`` token was masked-off and discarded
        in the baseline, so excluding it from the keys is exact.
        """
        B, L, _ = x.shape

        queries = self.q_proj_moe_gen(x)
        keys    = self.k_proj_moe_gen(x)
        values  = self.v_proj_moe_gen(x)

        queries = queries.reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        keys    = keys.reshape   (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values  = values.reshape (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.q_norm_moe_gen(queries)
        keys    = self.k_norm_moe_gen(keys)

        original_q_dtype = queries.dtype
        if self._attention_fp32:
            values = values.astype(mx.float32)
            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries.astype(mx.float32), keys.astype(mx.float32), cos, sin, unqueeze_dim=1
            )
        elif self._rope_fp32:
            values_fp32 = values.astype(mx.float32)
            cos, sin = self.rotary_emb(values_fp32, position_ids)
            q_rot, k_rot = apply_multimodal_rotary_pos_emb(
                queries.astype(mx.float32), keys.astype(mx.float32), cos, sin, unqueeze_dim=1
            )
            queries = q_rot.astype(original_q_dtype)
            keys = k_rot.astype(original_q_dtype)
        else:
            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin, unqueeze_dim=1
            )

        # Concat the frozen (post-RoPE) text-prefix K/V in front of the fresh
        # latent K/V. PrefixKVCache does NOT mutate the stored prefix.
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Full (unmasked) attention: latent queries see every prefix + latent key.
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=None
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self._attention_fp32 and output.dtype != original_q_dtype:
            output = output.astype(original_q_dtype)
        return self.o_proj_moe_gen(output)

    def _und_only_forward(
        self,
        x: mx.array,                       # (B, P, D) — text-prefix tokens ONLY
        mask: mx.array | None,             # (P, P) causal additive mask
        cache,                             # PrefixKVCache (capturing) or None
        position_ids: mx.array | None,     # (3, B, P)
    ) -> mx.array:
        """UND-only attention for the deferred-mode prefill (mirror of the full
        routed ``__call__`` UND branch). Computes ONLY ``q/k/v/o_proj`` and
        ``q/k_norm`` — never a ``*_moe_gen`` weight — so it does NOT materialize
        the deferred GEN tower. Bit-identical to the full-routed prefill because
        the prefix is all-TEXT: ``mx.where(gen_mask, …_moe_gen, …und)`` selects the
        UND branch everywhere, and the discarded GEN branch never affects the K/V.
        Carries the causal prefix mask and uses the same SDP wrapper as the full
        path (unlike ``_gen_only_forward``, which is unmasked latent attention)."""
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys    = self.k_proj(x)
        values  = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        keys    = keys.reshape   (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values  = values.reshape (B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys    = self.k_norm(keys)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        original_q_dtype = queries.dtype
        if self._attention_fp32:
            values = values.astype(mx.float32)
            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries.astype(mx.float32), keys.astype(mx.float32), cos, sin, unqueeze_dim=1
            )
        elif self._rope_fp32:
            values_fp32 = values.astype(mx.float32)
            cos, sin = self.rotary_emb(values_fp32, position_ids)
            q_rot, k_rot = apply_multimodal_rotary_pos_emb(
                queries.astype(mx.float32), keys.astype(mx.float32), cos, sin, unqueeze_dim=1
            )
            queries = q_rot.astype(original_q_dtype)
            keys = k_rot.astype(original_q_dtype)
        else:
            cos, sin = self.rotary_emb(values, position_ids)
            queries, keys = apply_multimodal_rotary_pos_emb(
                queries, keys, cos, sin, unqueeze_dim=1
            )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self._attention_fp32 and output.dtype != original_q_dtype:
            output = output.astype(original_q_dtype)
        return self.o_proj(output)


class LanceMoTLayer(Qwen2VLDecoderLayer):
    """mlx-vlm Qwen2VLDecoderLayer + `_moe_gen` siblings.

    Adds to the stock decoder layer:
        - `self.self_attn` replaced by LanceMoTAttention (per-expert q/k/v/o + QK-norms)
        - `self.mlp_moe_gen`, sibling SwiGLU MLP for GEN tokens
        - `self.input_layernorm_moe_gen`, second pre-attention RMSNorm
        - `self.post_attention_layernorm_moe_gen`, second post-attention RMSNorm

    The routed forward pattern (mirrors upstream `Qwen2MoTDecoderLayer.forward_train`):

        r = self_attn(  mx.where(gen, input_layernorm_moe_gen(x),     input_layernorm(x)),
                        position_group, ...)
        h = x + r
        r = mx.where(gen, mlp_moe_gen(post_attention_layernorm_moe_gen(h)),
                          mlp        (post_attention_layernorm(h))             )
        return h + r
    """

    def __init__(self, args: TextConfig):
        super().__init__(args)  # self_attn (Attention), mlp, input_layernorm, post_attention_layernorm

        # Replace stock Attention with our routed subclass (uses same args).
        self.self_attn = LanceMoTAttention(args)

        # GEN-side delta.
        self.mlp_moe_gen = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm_moe_gen        = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,                     # (B, T, D)
        position_group: mx.array,        # (T,) modality bucket
        mask: mx.array | None = None,
        cache=None,
        position_ids: mx.array | None = None,
        gen_only: bool = False,
        und_only: bool = False,
    ) -> mx.array:
        # GEN-only residual block for the t2i denoise loop: only the *_moe_gen
        # norms/attention/MLP run — no mx.where, no UND weights referenced.
        if gen_only:
            h_norm = self.input_layernorm_moe_gen(x)
            r = self.self_attn(h_norm, position_group, mask, cache, position_ids, gen_only=True)
            h = x + r
            h_norm2 = self.post_attention_layernorm_moe_gen(h)
            return h + self.mlp_moe_gen(h_norm2)

        # UND-only residual block for the deferred-mode prefill: only the UND
        # norms/attention/MLP run — no mx.where, no *_moe_gen weights referenced,
        # so the deferred GEN tower stays un-materialized. Numerically identical
        # to the full-routed path on an all-TEXT prefix.
        if und_only:
            h_norm = self.input_layernorm(x)
            r = self.self_attn(h_norm, position_group, mask, cache, position_ids, und_only=True)
            h = x + r
            h_norm2 = self.post_attention_layernorm(h)
            return h + self.mlp(h_norm2)

        # (1, T, 1) bool — True = GEN expert
        gen_mask = _broadcast_mask(position_group, x.dtype)

        # === Pre-attention: per-expert input_layernorm, then routed attention ===
        h_norm = mx.where(
            gen_mask,
            self.input_layernorm_moe_gen(x),
            self.input_layernorm(x),
        )
        r = self.self_attn(h_norm, position_group, mask, cache, position_ids)
        h = x + r

        # === Post-attention: per-expert post_attention_layernorm + MLP ===========
        h_norm2 = mx.where(
            gen_mask,
            self.post_attention_layernorm_moe_gen(h),
            self.post_attention_layernorm(h),
        )
        mlp_out = mx.where(
            gen_mask,
            self.mlp_moe_gen(h_norm2),
            self.mlp(h_norm2),
        )
        return h + mlp_out


class LanceModel(nn.Module):
    """Full Lance LLM backbone.

    Layout (matches the safetensors keys from `scripts/02_convert.py`):

        embed_tokens                  Embedding(vocab, hidden)
        layers[0..N-1]                LanceMoTLayer × num_hidden_layers
        norm                          RMSNorm(hidden)        — UND final
        norm_moe_gen                  RMSNorm(hidden)        — GEN final
        lm_head                       Linear(hidden, vocab, bias=False) — UNTIED
        vae_in_proj.vae2llm           Linear(48, hidden, bias=True)  — VAE → LLM
        latent_pos_embed.pos_embed    (4096|126976, hidden) parameter
        time_embedder.proj_in/out     Linear pair for sinusoidal-timestep MLP
        llm2vae                       Linear(hidden, 48, bias=True)  — LLM → VAE velocity (flow head)

    Does NOT include the ViT — that lives at the Pipeline orchestrator level
    (see `notes/phase1b_converter_design.md` for the placement rationale).

    `__call__` runs the transformer stack and per-expert final norm. The
    output heads (`self.lm_head`, `self.llm2vae`) are exposed as attributes;
    callers apply them on the position subsets they care about. This avoids
    burning the lm_head matmul (311 M params, expensive) on GEN positions
    where its output is discarded.

    Caller patterns (pipeline modules, not LanceModel itself):

        # x2t_image (VQA): all UND, take logits at last position.
        h = model(input_ids=tokens, position_ids=pids, position_group=groups)
        logits = model.lm_head(h[:, -1:, :])

        # t2i (image-gen flow step): mixed UND + NOISY_VAE.
        # Caller pre-builds inputs_embeds = text_emb || (vae_in_proj(latents) + latent_pos_embed + time_embedder(t)).
        h = model(inputs_embeds=embeds, position_ids=pids, position_group=groups)
        velocity = model.llm2vae(h[:, vae_idx, :])   # (B, n_vae, 48)
    """

    def __init__(self, args: TextConfig, num_latent_positions: int = 4096):
        """
        Args:
            args: mlx-vlm's TextConfig (matches Qwen2.5-VL-3B dimensions for Lance_3B).
            num_latent_positions: size of the `latent_pos_embed.pos_embed` table.
                4096 for Lance_3B (image, 64x64 spatial grid).
                126976 for Lance_3B_Video (4096 × 31 temporal slots).
                On load from a converted checkpoint, this gets overwritten with
                the actual tensor; the value here only sizes the fresh-init buffer.
        """
        super().__init__()
        self.args = args
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [LanceMoTLayer(args) for _ in range(args.num_hidden_layers)]

        # Per-expert final RMSNorms (146 RMSNorms total per Phase 1a inspection).
        self.norm         = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm_moe_gen = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Untied LM head — runtime override of llm_config.json's tie_word_embeddings: true.
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # Phase-1a empirical additions (not in original handoff).
        self.vae_in_proj      = VAEInputProjection(latent_channels=48, hidden_size=args.hidden_size)
        self.latent_pos_embed = LatentPosEmbed(num_positions=num_latent_positions, hidden_size=args.hidden_size)
        self.time_embedder    = TimestepEmbedder(hidden_size=args.hidden_size)
        self.llm2vae          = FlowHead(hidden_size=args.hidden_size, latent_channels=48)

    # ----- Phase 5g (issue #2 P0a) — runtime RoPE precision toggle ----------
    def set_rope_fp32(self, enabled: bool) -> None:
        """Toggle fp32 RoPE rotation across all 36 LanceMoTAttention modules.

        When True, each LanceMoTAttention computes cos/sin and applies the
        rotation `q*cos + rotate_half(q)*sin` in fp32 instead of inheriting
        mlx-vlm's bf16 downcast at `qwen2_5_vl/language.py:73`. This is the
        P0a candidate from the Phase 5e research brief — hypothesized to
        recover high-frequency precision in the flow-matching velocity field
        (manifests as soft water/fur/paws when off).
        """
        for layer in self.layers:
            layer.self_attn._rope_fp32 = bool(enabled)

    # ----- Phase 5m (issue #1) — runtime attention precision toggle ----------
    def set_attention_fp32(self, enabled: bool) -> None:
        """Toggle fp32 attention (Q/K/V through RoPE + SDP) across all 36
        LanceMoTAttention modules.

        When True, Q/K/V are promoted to fp32 before RoPE rotation and stay
        in fp32 through `scaled_dot_product_attention`; output is downcast
        before o_proj. Hypothesized fix for issue #1 silent semantic drift at
        n_lat ≥ 11,520 (768²×17f+) where bf16 attention accumulation degrades
        output without numerical blowup. Independent of `_rope_fp32` (which
        downcasts before SDP).
        """
        for layer in self.layers:
            layer.self_attn._attention_fp32 = bool(enabled)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        *,
        position_ids: mx.array,
        position_group: mx.array,
        mask: mx.array | None = None,
        cache: list | None = None,
    ) -> mx.array:
        """Run embeddings → 36 LanceMoTLayers → per-expert final norm.

        Args:
            input_ids:      (B, T) int token IDs, OR None if inputs_embeds is given.
            inputs_embeds:  (B, T, hidden_size) pre-built embeddings, OR None if input_ids.
                            For mixed-modality (text + ViT + VAE), caller builds these
                            using `model.embed_tokens`, `model.vae_in_proj`,
                            `model.latent_pos_embed`, `model.time_embedder` as helpers.
            position_ids:   (3, B, T) post-MaPE position coordinates for mRoPE.
            position_group: (T,) int per-token modality bucket {0:TEXT, 1:VIT, 2:CLEAN_VAE, 3:NOISY_VAE}.
            mask:           optional attention mask (causal by default if None).
            cache:          optional list of KVCache per layer (decoder steps).

        Returns:
            (B, T, hidden_size) — final hidden states with per-expert RMSNorm applied.
            Caller applies `self.lm_head(h[:, und_idx, :])` for next-token logits
            and `self.llm2vae(h[:, gen_idx, :])` for flow-matching velocity.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("LanceModel.__call__: provide either input_ids or inputs_embeds")
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, position_group, mask, c, position_ids)

        # Per-expert final norm.
        gen_mask = _broadcast_mask(position_group, h.dtype)
        h = mx.where(gen_mask, self.norm_moe_gen(h), self.norm(h))
        return h

    # ===== Single-node t2i optimization: prefix-prefill + gen-only loop ======
    #
    # Usage (per CFG arm), see pipeline/t2i.py:
    #   caches = model.prefill_prefix(prefix_embeds, position_ids=..., position_group=..., mask=...)
    #   model.free_und_tower()                          # AFTER all arms prefilled
    #   for step in ...:                                # 60 cheap latent-only passes
    #       h_lat = model.gen_loop_forward(lat_embeds, position_ids=pids_lat, caches=caches)
    #       velocity = model.llm2vae(h_lat)

    def prefill_prefix(
        self,
        prefix_embeds: mx.array,           # (B, P, D) — clean text-prefix embeddings ONLY
        *,
        position_ids: mx.array,            # (3, B, P)
        position_group: mx.array,          # (P,) — all TEXT (→ UND); routes via mx.where
        mask: mx.array,                    # (P, P) causal additive mask
        und_only: bool = False,
    ) -> list:
        """Run the forward over the text prefix ONCE, capturing each layer's
        post-RoPE K/V into a per-layer ``PrefixKVCache``. The hidden output is
        discarded — only the K/V matter. Must run BEFORE ``free_und_tower`` (the
        prefix tokens route to the UND expert).

        ``und_only`` (deferred-load mode): route each layer through the UND-only
        path so no ``*_moe_gen`` weight is touched and the deferred GEN tower stays
        un-materialized. The captured K/V are bit-identical to the full-routed
        prefill (the prefix is all-TEXT, so ``mx.where`` selects UND everywhere)."""
        from .prefix_cache import PrefixKVCache

        caches = [PrefixKVCache() for _ in range(len(self.layers))]
        h = prefix_embeds
        for layer, c in zip(self.layers, caches):
            h = layer(h, position_group, mask, c, position_ids, und_only=und_only)
        for c in caches:
            c.freeze()
        return caches

    def gen_loop_forward(
        self,
        lat_embeds: mx.array,              # (B, n_lat, D) — latent embeddings ONLY
        *,
        position_ids: mx.array,            # (3, B, n_lat) — latent grid coords (post-MaPE)
        caches: list,                      # per-layer PrefixKVCache (frozen)
    ) -> mx.array:
        """One denoise pass: 36 GEN-only layers over the latent tokens, attending
        to the cached text prefix, then the GEN final norm. Returns the latent
        hidden states (B, n_lat, D) for ``llm2vae``. Touches ZERO UND weights."""
        h = lat_embeds
        for layer, c in zip(self.layers, caches):
            h = layer(h, None, None, c, position_ids, gen_only=True)
        return self.norm_moe_gen(h)

    def free_und_tower(self, deferred: bool | None = None) -> dict:
        """Delete the UND-tower weights to reclaim ~5.5 GB. KEEPS embed_tokens,
        every ``*_moe_gen`` module, the final GEN norm, and the VAE/flow heads.

        MUST be called AFTER the prefix prefill (eager relay: full-routed prefill,
        which evaluates the UND branch; deferred relay: UND-only prefill) and BEFORE
        any ``gen_loop_forward`` (which never references UND). Inference-only: do NOT
        call before saving a checkpoint.

        ``deferred`` (None → read ``self._gen_deferred``): in deferred-load mode the
        GEN tower is still lazy/mmap-backed here, so we must NOT ``mx.eval(self.
        parameters())`` — that would materialize GEN while the just-freed UND buffers
        may not yet be released (a transient ~11 GB both-towers blip). Instead release
        UND first (del → gc → clear_cache); the caller then runs ``materialize_gen_tower``.
        In eager relay mode GEN is already resident, so the eval is the same cheap
        consistency flush as before."""
        import gc
        if deferred is None:
            deferred = bool(getattr(self, "_gen_deferred", False))

        before = mx.get_active_memory()
        for layer in self.layers:
            attn = layer.self_attn
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                if hasattr(attn, attr):
                    delattr(attn, attr)
            for attr in ("mlp", "input_layernorm", "post_attention_layernorm"):
                if hasattr(layer, attr):
                    delattr(layer, attr)
        # UND final norm (GEN loop uses norm_moe_gen). lm_head is unused in t2i
        # (only llm2vae runs) — free it too for another ~0.6 GB.
        for attr in ("norm", "lm_head"):
            if hasattr(self, attr):
                delattr(self, attr)

        gc.collect()
        if not deferred:
            # GEN already materialized → flush the graph before releasing buffers.
            mx.eval(self.parameters())
        mx.clear_cache()   # return the freed UND Metal buffers to the OS (mlx 0.31.2)
        after = mx.get_active_memory()
        return {"freed_bytes": before - after, "active_after": after}

    def materialize_gen_tower(self) -> dict:
        """Eval the deferred GEN-tower (``*_moe_gen``) params, pulling them from the
        retained safetensors mmap into Metal buffers (~5.5 GB). Call AFTER
        ``free_und_tower(deferred=True)`` (UND released first, so the two towers are
        never co-resident) and BEFORE the GEN-only denoise loop. The mmap reference
        is dropped once materialized. No-op-ish if GEN was never deferred."""
        from mlx.utils import tree_flatten

        before = mx.get_active_memory()
        gen = [v for k, v in tree_flatten(self.parameters()) if "_moe_gen" in k]
        mx.eval(gen)
        # GEN now lives in Metal buffers; the mmap'd safetensors can be released.
        if getattr(self, "_saved_ref", None) is not None:
            self._saved_ref = None
        self._gen_deferred = False
        after = mx.get_active_memory()
        return {"materialized_bytes": after - before, "active_after": after}

    def free_gen_tower(self) -> dict:
        """Delete the GEN-tower weights (every ``*_moe_gen`` module + the GEN final
        norm + the flow/VAE-input heads + token embeddings) to reclaim ~5.5 GB
        AFTER the denoise loop has produced the final latents and BEFORE the VAE
        decode.

        The VAE decoder is a SEPARATE module (``pipeline.vae_decoder``) that never
        references the MoT backbone — once the latents are computed, the entire
        Lance LLM is dead weight for the rest of generate(). Shedding it here drops
        the decode peak from ~11.6 GB (GEN resident + decode transient) to ~5-6 GB
        (VAE-only + transient), and that decode peak is then flat in frame count.

        This is the GEN half of the shed cascade — the symmetric partner of
        ``free_und_tower``:

            prefill(UND) → free_und_tower → denoise loop(GEN) → free_gen_tower → decode(VAE)

        Each phase holds only the weights it uses, so the peak ≈ the single
        heaviest phase rather than the sum. Architecture property of the MoT split,
        so it applies to t2i and t2v identically.

        ONE-WAY and DESTRUCTIVE: the backbone cannot run again after this. Call ONLY
        in relay (single-shot) mode — which already deleted UND, so this adds no
        reuse constraint that wasn't already there — and NEVER in parallel mode
        (which preserves the full model for reuse). Inference-only. The final
        ``latents`` must already be ``mx.eval``'d (the denoise loop does this every
        step), so no pending graph references the weights being freed."""
        import gc

        before = mx.get_active_memory()
        for layer in self.layers:
            attn = layer.self_attn
            for attr in ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen",
                         "o_proj_moe_gen", "q_norm_moe_gen", "k_norm_moe_gen"):
                if hasattr(attn, attr):
                    delattr(attn, attr)
            for attr in ("mlp_moe_gen", "input_layernorm_moe_gen",
                         "post_attention_layernorm_moe_gen"):
                if hasattr(layer, attr):
                    delattr(layer, attr)
        # GEN final norm, flow head, GEN-input helpers, and the token embedding are
        # all unused once the latents exist (decode runs on pipeline.vae_decoder
        # alone). norm/lm_head are already gone in relay (free_und_tower deleted
        # them) — guarded so this is also safe if ever called standalone.
        for attr in ("norm_moe_gen", "llm2vae", "embed_tokens",
                     "vae_in_proj", "latent_pos_embed", "time_embedder",
                     "norm", "lm_head"):
            if hasattr(self, attr):
                delattr(self, attr)

        gc.collect()
        mx.clear_cache()   # return the freed GEN Metal buffers to the OS
        self._gen_freed = True
        after = mx.get_active_memory()
        return {"freed_bytes": before - after, "active_after": after}
