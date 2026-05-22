# L2 — Upstream position-IDs audit (2026-05-22)

**Triggered by:** L2 in the polish-phase plan, post-Phase-5j.

**Source:** Cloned `bytedance/Lance` at `/tmp/lance-upstream` (shallow clone,
HEAD as of 2026-05-22).

## What we found

### 1. Two position-ID paths in upstream

`upstream/modeling/lance/qwen2_navit.py:864-872`:

```python
if self.apply_qwen_2_5_vl_pos_emb:                    # OPTIONAL path
    packed_position_embeddings = self.rotary_emb(
        packed_sequence.unsqueeze(0), packed_position_ids
    )
else:                                                  # DEFAULT (config) path
    cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
    cos = cos.squeeze(0); sin = sin.squeeze(0)
    packed_position_embeddings = (cos, sin)
```

- **`True` path**: 3D mrope from `get_rope_index` — what Qwen2.5-VL normally does
- **`False` path** (config default): 1D Qwen2 RoPE with **all visual tokens
  in a block sharing ONE position id**

The data prep (`upstream/data/datasets_custom/validation_dataset.py:457`) builds
the 1D shared positions:

```python
self.sample["packed_position_ids"].extend(
    [curr_rope_id] * (num_vid_tokens + num_special_tokens)
)  # NOTE: 为什么rope固定?  ← "Why is rope fixed?" — upstream's own TODO
curr_rope_id += 1
```

### 2. Inference always uses the `True` path

`upstream/inference_lance.sh:128` explicitly sets `--apply_qwen_2_5_vl_pos_emb true`.
`upstream/lance_gradio_t2v_v2t.py:200` hardcodes `apply_qwen_2_5_vl_pos_emb=True`.

The `False` config default is the legacy training-time value, dead at inference.
**Our 3D-mrope port is architecturally correct.**

### 3. `get_rope_index` does TWO things we don't

Upstream calls `mlx_vlm/models/qwen2_5_vl/language.py::get_rope_index`
(or the matching PyTorch original):

```python
# lines 341-345
llm_grid_t, llm_grid_h, llm_grid_w = (
    t.item(),
    h.item() // spatial_merge_size,   # ← DIVIDES BY sms (=2)
    w.item() // spatial_merge_size,
)
# ... visual h-axis: arange(llm_grid_h) + text_len + st_idx
```

A) Divides h/w by `spatial_merge_size=2`
B) Anchors visual block at `text_len + st_idx` (text-position anchor)

`shift_position_ids` (`upstream/data/common.py:46`) further re-anchors the
t-axis for image-gen / video-gen modalities — but only if
`attn_mode in ["full_noise", "full"]`. For pure t2v (`attn_mode = "noise"`),
this gate doesn't fire. Confirmed our Phase 5d default `mape_anchor=None`
is upstream-correct.

### 4. Where our port diverges

Our `_build_position_ids` (across t2v.py / image_edit.py / video_edit.py):

| Upstream `get_rope_index` | Our port (Phase 5j t2v.py default) |
|---|---|
| `sms = 2` divisor on h/w grid | `sms = 1` (no divisor) |
| Visual base = `text_len + st_idx` | Visual base = `0` (Phase 5j fix) |

The combination we run is **`sms=1, base=0`**. The upstream inference
combination is **`sms=2, base=text_len`**.

### 5. The Phase 5g test gap

Phase 5g tested:

| Variant | sms | base | MD5 | Visual |
|---|---|---|---|---|
| V0 (legacy) | 1 | text_len | `2ca49d9…` | watercolor |
| V1          | 2 | text_len | `e612884…` | **subject loss** |
| V2 (rope_fp32) | 1 | text_len | `2ca49d9…` | identical to V0 |
| V3 (rope_fp32 + sms=2) | 2 | text_len | `e612884…` | identical to V1 |

We did NOT test **`sms=2 + base=0`** — the unexplored combination.

Possible explanation for V1's subject loss: when we set `sms=2` but kept
`MAX_LATENT_SIDE=64` in our `latent_pos_embed` lookup, the LPE indices
still spanned the full 64×64 grid while the mrope coords were halved.
Mismatch between additive LPE and mrope coords → broken spatial alignment
→ subject loss. The fix here would be to ALSO halve the LPE-index range,
or use a different `MAX_LATENT_SIDE` aligned with sms.

## What this means for our shipped Phase 5j fix

**Our Phase 5j fix (`sms=1, base=0`) is NOT an exact upstream replica.**
The empirical result (photoreal output across 8 oracle prompts at 768²×13f)
suggests our combination produces a *different but equally-good* position-ID
configuration that:

- Keeps visual coords in a small numerical range (0..30 for 480×704)
- Avoids the prompt-length drift that caused watercolor in the legacy

The "correct" upstream-matching configuration (`sms=2, base=text_len`)
remains untested. If it produces equal or better output than our Phase 5j
fix, it would be the more faithful port. If it produces worse output (as
suggested by V1's subject loss), our fix is an empirically-derived
alternative that happens to work better.

## Worth-testing combinations

1. **`sms=2, base=text_len` with halved LPE range** — closest to upstream
   inference. Requires changing `MAX_LATENT_SIDE` from 64 to 32 (or some
   sms-aware divisor), then re-running the oracle pass.

2. **`sms=2, base=0` (untested)** — the simple completion of Phase 5g's
   matrix. May reveal whether the sms divisor alone changes much when
   base is anchored at origin.

3. **Direct call to mlx-vlm's `get_rope_index`** — replace our
   `_build_position_ids` with the canonical function. Most faithful port.
   Risk: subtle behavior differences (e.g. how it handles vision_start
   token-id detection, attention_mask conventions) might break things.

## Practical recommendation

**For shipping: keep Phase 5j as-is.** It's empirically validated across
8 oracle prompts and the L1 lesson reminded us that propagating fixes
without per-pipeline validation is risky.

**For continued investigation (post-v0.5.0):** test option 1 above. If it
produces equivalent quality, swap in as the new default (closer to
upstream); if not, document our Phase 5j approach as a viable variant
and continue.

The motion-direction residual (left-to-right instead of forward-into-depth)
that the user observed at seed=42 may relate to one of these untested
combinations — t-axis handling in the canonical `get_rope_index` differs
from our scheme in subtle ways that could affect temporal coherence.

## Also confirmed: L4 (TimestepEmbedder t-scale) is correct

Upstream `modeling_utils.py::TimestepEmbedder.forward` calls
`timestep_embedding(t, dim)` directly with t ∈ [0, 1] (no scaling).
Our scaffold matches exactly. Research P2 candidate refuted with
upstream-source evidence.

## Files cross-referenced

- `/tmp/lance-upstream/modeling/lance/lance.py` lines 227, 241, 249, 599-625, 657
- `/tmp/lance-upstream/modeling/lance/qwen2_navit.py` lines 864-872
- `/tmp/lance-upstream/modeling/lance/modeling_utils.py` lines 110-146
- `/tmp/lance-upstream/data/common.py::shift_position_ids` (gated, no-op for noise)
- `/tmp/lance-upstream/data/datasets_custom/validation_dataset.py:457` (1D shared rope, training only)
- `/tmp/lance-upstream/data/data_utils.py::get_flattened_position_ids_extrapolate_video` (matches our LPE indices)
- `/tmp/lance-upstream/inference_lance.sh:128` (`--apply_qwen_2_5_vl_pos_emb true`)
- `/tmp/lance-upstream/lance_gradio_t2v_v2t.py:200` (`apply_qwen_2_5_vl_pos_emb=True`)
- `/Volumes/DEV_VOL1/VideoResearch/lance-mlx/.venv/.../mlx_vlm/models/qwen2_5_vl/language.py::get_rope_index` (canonical 3D-mrope builder)

---

## L2-impl follow-up (2026-05-22) — upstream-correct config tested + REJECTED

Empirically tested the upstream-correct config at 768²×13f, red-panda-
surfing oracle prompt, seed=43, 30 steps, CFG=4, MaPE=None:

| Variant | sms | base | Result |
|---|---|---|---|
| V0 — Phase 5j (current default) | 1 | 0 | **Photoreal CGI** ✓ (production baseline) |
| V1 — upstream replica | 2 | text_len | Cartoon/illustrated, stylized — REGRESSED |
| V2 — disambiguation probe | 2 | 0 | Cartoon/illustrated, stylized — REGRESSED |

**Verdict: `sms=2` is the regression axis.** Both V1 and V2 share `sms=2`
and both regress vs V0. The `base` value (text_len vs 0) is secondary.

**This contradicts the upstream convention.** Our port empirically REQUIRES
`sms=1` (full grid) to produce photoreal output, even though upstream Lance
was trained with `sms=2` (per `get_rope_index`). Some interaction in our
forward pass — likely in mrope freq scaling, LPE/mrope alignment, or
LanceMoTAttention's per-tower QK-norms — makes halved-grid h/w coords
land out of training distribution for our specific port.

### Open hypotheses for the sms=2 regression

1. **Mrope freq scaling**: inherited from mlx-vlm `Qwen2RotaryEmbedding`,
   uses upstream's exact formula. Freq calc itself should match.
2. **LPE / mrope mismatch**: both we and upstream use raw (sms=1) coords
   for LPE — should match upstream. Yet upstream produces clean output
   at sms=2 and we don't.
3. **LanceMoTAttention QK-norm interaction** (most likely): our per-tower
   QK-norms (q_norm, k_norm, q_norm_moe_gen, k_norm_moe_gen) normalize
   per-head over head_dim=128. At sms=2 with halved h/w spread, the
   post-rotary QK values land in a different magnitude regime that may
   interact poorly with checkpoint-trained norm scales.
4. Some other axis-dependent op we haven't identified.

### Verdict

**Ship Phase 5j as the default.** The port's bf16 ceiling is lower than
full upstream parity, and the gap is small enough (corner clouds + motion
direction at short clips) to accept as a production residual. Closer-to-
upstream alternatives empirically don't improve output for our port.

**Next leverage is on the quantization track** (DWQ calibrated against
the Phase 5j-correct bf16 baseline). Further bf16 optimization would
need PyTorch parity tests (layer-by-layer activation comparison) — a
multi-day effort with unclear ROI given current production quality.

Files cross-referenced in this follow-up:
- `scripts/36_L2_upstream_replica.py` — the 3-variant test harness
- `/tmp/lance_L2_upstream_replica/compare_grid.png` — visual evidence
- V0 md5: `45631d9f8b6254e2d73b74572149aa4b`
- V1 md5: `4c5d6c6bd23b8e9dc7cd305d9bb81593`
- V2 md5: `b872b56b0196ba056167bda656edf270`
