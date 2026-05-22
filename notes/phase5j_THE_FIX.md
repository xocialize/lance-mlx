# Phase 5j — 🎯 THE WATERCOLOR FIX

**Date:** 2026-05-21
**Status:** ✅ **FIX CONFIRMED at 256²×17f. Default behavior changed.**
**Commit:** (pending)

## TL;DR

The painterly/watercolor aesthetic on t2v was caused by
**prompt-length-dependent drift of the latent block's mrope position-IDs**.

In `_build_position_ids`, the latent block's (t, h, w) grid coordinates were
anchored to `base = text_len_before_latents`, so the latent token grid lived
at coords `[base, base+t_lat) × [base, base+h_lat) × [base, base+w_lat)`.
For our verbose chat template wrapping a long prompt (`system\n{T2V_INSTRUCTION}
<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
<|vision_start|>`), `base` was 80–100. Lance's GEN tower trains against
Qwen2.5-VL's mrope convention where visual tokens occupy the **3D-mrope grid
ORIGIN** (h ∈ [0, h_grid), w ∈ [0, w_grid)) — not concatenated with the text
sequence. The drift pushed every visual position out of the training
distribution, manifesting as watercolor smearing of high-frequency detail.

**Fix:** anchor the latent block at `base = 0` regardless of prompt length.
One-line conceptual change, gated by a new kwarg `latent_pos_base: int | None
= 0` (default is now the fix; pass `None` to restore legacy/broken behavior
for reproducibility).

## How we got here

This is the resolution of GitHub issue #2 — the t2v port-quality gap that
survived seven phases of investigation (4b, 4c paused, 5d MaPE-shift, 5e
research handoff, 5f RockTalk weights test, 5g P0a/P0b candidates, 5h chat
template). Each phase narrowed the bug surface:

| Phase | Hypothesis | Outcome |
|---|---|---|
| 4b/4c | Frame-count threshold (n_lat ceiling) | Real but narrower than thought (issue #1) |
| 5d   | MaPE temporal shift to t=2000 | Removing improved 256²×13f but residual gap on water/textures |
| 5e   | Deep-research handoff (4 ranked candidates) | Returned P0a/P0b/P1/P2 |
| 5f   | RockTalk weights × our pipeline | **Byte-identical output → bug is in OUR pipeline, not converter** |
| 5g   | P0a (fp32 RoPE), P0b (sms=2 divisor) | Both refuted: P0a no-op, P0b makes worse |
| 5h   | RockTalk's minimal chat template | Different but worse (subject loss) |
| 5i   | RT-config replay (ocean prompt, T_lat=3) | **Our pipeline produces SHARP output on this config** |
| 5i.2 | Single-axis bisect of trigger | **Prompt-length is the trigger axis** |
| **5j**   | **`latent_pos_base=0`** | **🎯 PHOTOREAL** |

## The bug

`src/lance_mlx/pipeline/t2v.py::_build_position_ids` legacy logic:

```python
base = text_len_before_latents              # 80-100 for verbose prompts
for idx, token_pos in enumerate(latent_positions):
    f = idx // (h_lat * w_lat)
    r = (idx % (h_lat * w_lat)) // w_lat
    c = (idx % (h_lat * w_lat)) % w_lat
    pos[0, 0, token_pos] = base + f         # t-axis    DRIFT
    pos[1, 0, token_pos] = base + r         # h-axis    DRIFT
    pos[2, 0, token_pos] = base + c         # w-axis    DRIFT
```

Upstream Qwen2.5-VL convention for visual tokens (per `get_rope_index` in
HF transformers and `mlx_vlm/models/qwen2_5_vl/language.py`): visual tokens
occupy the **3D-mrope grid origin**, e.g. for a 32×32 image patch grid the
positions are `(0, [0,32), [0,32))` — NOT continued from the preceding text
sequence. Text positions before/after are independent because mrope's three
axes give each token a unique combined coordinate, so duplicate position-IDs
across text and visual axes don't conflict.

Our port concatenated the latent grid onto the text sequence — a port-side
deviation that grew the drift linearly with prompt length.

## Why every prior bisect missed it

- **Phase 3e fixed t2i** with the same `base + f/r/c` formula — image at
  T_lat=1 is just one frame; small spread; model tolerates the drift.
- **Phase 4b/4c** focused on frame-count threshold; that's a separate bug
  (issue #1).
- **Phase 5d** noticed position-related issues but blamed the MaPE temporal
  shift (correctly removed). MaPE removal helped at small scales but the
  remaining h/w drift still bit on long prompts.
- **Phase 5g P0b** tried `sms=2` divisor — RIGHT INSTINCT (position-ID
  spread), WRONG AXIS. Halving the latent grid coords didn't help because
  the drift was the `+ base` offset, not the magnitude of `r`/`c` themselves.
- **Phase 5h** tried RT's minimal chat template — shorter prompt = smaller
  `base` = less drift, but also removed instruction guidance Lance is
  trained against. Net: different bad.
- **Phase 5i (ocean wave)** finally exposed the trigger: simple short
  prompt → small base → no drift → sharp output. Then the 5i.2 single-axis
  bisect at the same other config flipped to watercolor only when we
  swapped prompts. That's when the position-ID concatenation hypothesis
  became testable.

## Empirical verdict

**Phase 5j at 256²×17f red-panda-surfing oracle prompt, seed=42, 30 steps,
CFG=4.0, MaPE=None:**

| Variant | latent_pos_base | midframe MD5 | Visual |
|---|---|---|---|
| V0_redpanda_legacy | text_len (~90) | `a20138f…` | **watercolor** (the bug) |
| **V1_redpanda_FIX**    | **0** | `1b802f2…` | **🎯 PHOTOREAL** |
| V2_ocean_legacy    | text_len (~30) | `7583224…` | sharp (control) |
| V3_ocean_FIX       | 0 | `5983697…` | sharp (no regression) |

Output strips:
- `/tmp/lance_phase5j/V0_redpanda_legacy_strip.png` — watercolor red panda
- `/tmp/lance_phase5j/V1_redpanda_FIX_strip.png` — **photoreal red panda
  with clearly defined straw hat, fur texture, beach scene structure**
- `/tmp/lance_phase5j/V2_ocean_legacy_strip.png` — sharp ocean wave
- `/tmp/lance_phase5j/V3_ocean_FIX_strip.png` — sharp ocean wave

Scale confirmation at 480×704×17f (Phase 5j.2, in progress at commit time):
`/tmp/lance_phase5j_scale/compare_grid.png` — TBD.

## Code change

`src/lance_mlx/pipeline/t2v.py`:

1. `_build_position_ids` gains `latent_pos_base: int | None = None` kwarg.
   `None` = legacy behavior (`base = text_len_before_latents`); integer =
   override (typically `0`).
2. `_prepare_state` propagates the kwarg.
3. `generate` signature: **`latent_pos_base: int | None = 0`** (DEFAULT
   CHANGED — the fix is now on by default). Pass `None` to restore legacy.

Backwards-compatibility note: callers who had been pinning specific
`latent_pos_base=None` behavior in tests can do so explicitly. The flag
is plumbed cleanly all the way through `_prepare_state` and
`_build_position_ids`.

## Why we didn't propagate to t2i / image_edit

Those pipelines work today at production quality (Phase 3e finished t2i, 3.5
finished image_edit; both validated visually and on the oracle dataset).
The reason they tolerate the legacy `base = text_len` drift is presumably
small T_lat (=1 → single frame, no t-axis spread) plus more forgiving
training-time distribution for single-image generation. **We deliberately
do NOT change t2i.py or image_edit.py in this commit** — would be a
gratuitous risk to working pipelines. If those pipelines show any latent
quality issues in the future, the same `latent_pos_base=0` flag can be
added there as a one-line port.

(`video_edit.py` and `x2t_video.py` similarly are NOT touched in this
commit — they use the same `_build_position_ids` pattern from t2v.py so
share the fix when invoked through TextToVideoPipeline.generate, but
direct callers won't get the new default until plumbed.)

## Open question

Why does our pipeline work in t2i and at Phase 5i's ocean-wave RT-config
test (short prompt) but fail on the verbose prompt? The bisect data is
unambiguous (prompt length triggers the failure) — but the explanation
"latent grid coords drift out of training distribution" is one hypothesis
among several. Alternative: maybe Lance's training-time data also wrapped
the latent block in text and the issue is something more subtle (e.g. the
ACTUAL position-IDs of the chat template tokens, not the offset). The
photoreal output with `base=0` is the empirical proof regardless of which
specific theoretical model explains it.

## Next steps

1. Confirmation at 480×704×17f (running now).
2. Confirmation at 768²×50f (Lance reference scale, was issue #1's
   "pure noise" failure case — does the position-ID fix help at that
   scale, or does the n_lat-ceiling bug from issue #1 dominate?).
3. Update HF model card for `mlx-community/Lance-3B-Video-bf16` to upgrade
   status from "🟡 functional, port-quality under investigation" to "🟢
   production" (assuming scale confirmations hold).
4. Update GitHub issue #2 with the verdict, close as fixed.
5. Update top-level README to reflect t2v reaching production.
6. (Optional, future) Apply `latent_pos_base=0` flag to image_edit /
   video_edit / x2t pipelines for consistency.
