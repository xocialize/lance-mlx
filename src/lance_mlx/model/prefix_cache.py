"""PrefixKVCache — frozen text-prefix K/V for the Lance t2i single-node optimization.

Lance t2i runs 30 Euler steps × 2 CFG arms = 60 transformer passes over the SAME
clean text prefix plus a per-step-changing block of latent tokens. The block mask
(t2i.py `_build_block_mask`) is asymmetric: text queries are causal-only and NEVER
attend to latent keys, so text-prefix hidden states — hence the text-prefix K/V at
every layer — are bit-identical across all 60 passes. This cache computes the
prefix K/V ONCE (during a prefill forward) and replays it for every loop pass.

Two phases:
  * capture (prefill): the first ``update_and_fetch(keys, values)`` call stores the
    post-RoPE text-prefix K/V and returns them unchanged, so the prefill forward
    runs ordinary prefix self-attention. Call ``freeze()`` afterwards.
  * loop (gen-only): each ``update_and_fetch(k_lat, v_lat)`` returns
    ``concat([k_pre, k_lat])`` along the sequence axis WITHOUT mutating the stored
    prefix — the same frozen prefix is reused for every step.

Deliberately does NOT inherit mlx_lm's ``KVCache`` (whose append-and-grow-offset
semantics would accumulate latent K/V across steps, exploding the offset). It has
no ``.bits`` attribute, so mlx's SDP treats it as a plain (non-quantized) cache.
"""

from __future__ import annotations

import mlx.core as mx


class PrefixKVCache:
    """Per-layer cache holding a frozen prefix; concats fresh latent K/V per step."""

    def __init__(self) -> None:
        self.k_pre: mx.array | None = None   # (B, n_kv_heads, P, head_dim), post-RoPE
        self.v_pre: mx.array | None = None
        self._capturing = True

    @property
    def offset(self) -> int:
        """Prefix length once captured (0 before). Informational — the t2i path
        always passes ``position_ids`` explicitly, so attention never relies on it."""
        return 0 if self.k_pre is None else int(self.k_pre.shape[2])

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Capture phase: store + return the prefix. Loop phase: return
        ``concat([prefix, latent])`` without mutating the stored prefix."""
        if self._capturing:
            self.k_pre = keys
            self.v_pre = values
            return keys, values
        return (
            mx.concatenate([self.k_pre, keys], axis=2),
            mx.concatenate([self.v_pre, values], axis=2),
        )

    def freeze(self) -> "PrefixKVCache":
        """Switch from capture to loop mode and pin the prefix tensors."""
        self._capturing = False
        if self.k_pre is not None:
            mx.eval(self.k_pre, self.v_pre)
        return self
