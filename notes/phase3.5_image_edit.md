# Phase 3.5 — image_edit pipeline

**Status: ✅ Functional end-to-end on Apple Silicon, 2026-05-21.**

Lance's instruction-based image editing now runs in MLX. Two oracle cases
("Remove the hat from the painting." and "Add a pearl necklace around the
woman's neck.") both produced content-correct edits on the first run.

## Architecture

`ImageEditPipeline` (`src/lance_mlx/pipeline/image_edit.py`) composes:
- **Wan22VAEEncoder** — input image → clean latent z_clean (1, 1, 48, 48, 48 @ 768²)
- **LanceModel** — dual-stream MoT with flow head
- **Wan22VAEDecoder** — denoised latent → output PIL image
- **Flow loop** — 30 Euler steps, timestep_shift=3.5, linear schedule

### Token sequence layout
```
<|im_start|>system\n{edit_instruction}<|im_end|>\n
<|im_start|>user\n<|vision_start|>{video_pad × 2304 (clean ref)}<|vision_end|>{user_text}<|im_end|>\n
<|im_start|>assistant\n<|vision_start|>{video_pad × 2304 (noisy target)}<|vision_end|>
```

Total ≈ 4688 tokens at 768², of which 2 × 2304 = 4608 are latent tokens.

### Embedding scatter
- **Clean ref positions:** `vae_in_proj(z_clean) + latent_pos_embed + time_embedder(0)`
- **Noisy target positions:** `vae_in_proj(z_t) + latent_pos_embed + time_embedder(t)`

The clean ref's `time_embedder(0)` is the upstream's signal that the latent
is "clean" (not being denoised). Per `lance_lance.py:644-648`:
```python
timestep = torch.zeros(x_t.shape[0])
timestep[current_vae_mse_indexes_local_in_vae] = timestep_  # ONLY at noisy
```

### Position IDs
Both blocks cover the same 48×48 grid spatially. Per upstream's
`shift_position_ids` rule `position_ids[modality==1] = position_ids[modality==2]`
(noise inherits clean ref positions when counts match — exactly the case for
edit), both blocks share the SAME (t, h, w) per-cell.

After MaPE re-anchor to ANCHOR_IMAGE_GEN=1000, both latent blocks have
t-axis = 1000 across all cells, with h, w varying spatially.

### Attention mask
Per upstream `create_sparse_mask` (after the "full_noise → full" rewrite):
- Causal default
- Bidirectional WITHIN clean ref block
- Bidirectional WITHIN noisy target block
- Cross-block: causal (target sees clean ref since later in sequence; clean
  cannot see target acausally)

### Routing
Standard `PositionGroup` taxonomy used everywhere:
- TEXT → UND expert
- CLEAN_VAE → GEN expert
- NOISY_VAE → GEN expert

## MVP simplifications

1. **No ViT semantic stream of input.** Upstream's `tiv2v_sample` runs the
   input image through BOTH the Qwen2.5-VL ViT AND the VAE encoder; we only
   use the VAE path. Clean VAE conditioning alone was sufficient for the two
   tested cases. If quality suffers on harder edits, Phase 3.6 adds the ViT
   stream.

2. **CFG = text only.** Lance's `cfg_type=0` (matched Phase 0). Uncond branch
   keeps the clean ref image but empties the user text instruction.

3. **Same anchor for both latent blocks.** We anchor both clean and noisy
   to t=1000. Upstream technically leaves modality 2 (clean ref) untouched
   and only triggers a copy when modality 1 (noise) is present. The net
   effect is equivalent since the copy makes them identical.

## Results

| case | instruction | output | notes |
|---|---|---|---|
| 0001 | Remove the hat from the painting. | `tests/fixtures/image_edit/lance_edit_remove_hat.png` | Hat gone, hair revealed, identity preserved, signature preserved |
| 0002 | Add a pearl necklace around the woman's neck. | `tests/fixtures/image_edit/lance_edit_add_necklace.png` | Pearl strand added, hat preserved (not in instruction), identity preserved |

Phase 0 oracle for case 0001 was generated with **Lance_3B_Video** (painterly
aesthetic, per Phase 4c findings). Our MLX result uses **Lance_3B** (image
specialist) → photo-realistic aesthetic. Content correctness matches in both.

## Performance (M5 Max 128GB)

- VAE encoder: ~0.5s for 768² image
- VAE decoder: ~1.6s
- Lance forward (one step, CFG=4.0, T=4688): ~6.7s
- 30-step generation: ~201s total (≈6.7s/step)

CFG doubles the forward cost since cond + uncond are run separately.
For Phase 5 optimization: KV cache for the text prefix would amortize the
text-tower cost across steps (~30% of the forward today).

## Latent statistics convergence

Starting noise: mean ~0, std ~1.0
After 30 steps: mean ~-0.087, std ~0.580

`z_clean` reference: mean = -0.067, std = 0.602.

The denoised latent converges to the same distribution as the encoded
reference — visual evidence of a well-conditioned flow.

## Files

- `src/lance_mlx/pipeline/image_edit.py` — ImageEditPipeline (~430 LOC)
- `scripts/12_vae_roundtrip.py` — encoder/decoder load smoke test
- `scripts/13_image_edit_demo.py` — CLI driver
- `tests/fixtures/image_edit/edit_img.jpg` — upstream source painting
- `tests/fixtures/image_edit/lance_edit_remove_hat.png` — our case 0001 output
- `tests/fixtures/image_edit/lance_edit_add_necklace.png` — our case 0002 output

## Deferred to Phase 3.6 / Phase 5

- **Add ViT semantic stream of input image.** Currently skipped — clean VAE
  alone proved sufficient on simple-attribute edits (hat/necklace). Harder
  edits ("change pose", "3D cartoon style") may benefit from the semantic
  prior.
  *Benefit:* Better identity preservation on style transfers; potentially
  more accurate referent resolution for complex instructions.

- **KV cache for text + clean ref prefix.** They don't change across steps,
  yet we recompute them every step. ~30% step-time speedup.
  *Benefit:* Faster per-edit latency, especially at lower step counts.

- **CFG channel-norm exploration.** Lance's upstream default is global renorm
  (matches t2i). Channel renorm may help fine detail preservation.
  *Benefit:* Sharper edits on fine-grained features (jewelry, text).

- **Multi-block batching.** Cond + uncond CFG arms run sequentially today.
  Could batch them as B=2.
  *Benefit:* ~2× generation throughput with same memory headroom.
