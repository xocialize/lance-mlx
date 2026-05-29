"""t2i — text-to-image generation via Lance + Wan2.2 VAE.

Phase 3b MVP. Composes:
  - LanceModel (our MoE backbone with the flow head)
  - Wan22VAEDecoder (mlx-video; loads from converted Lance VAE checkpoint)
  - Linear flow-matching scheduler with Lance's timestep_shift=3.5

Algorithmic overview (30 Euler steps, no CFG in v1):

  for t in schedule(1.0 → 0.0, 30 steps):
      embed[noisy_positions] = vae_in_proj(z) + latent_pos_embed[grid_idx] + time_embedder(t)
      h = lance_model(inputs_embeds, position_ids, position_group)
      velocity = llm2vae(h[noisy_positions])         # (B, N_lat, 48)
      z = z - velocity * dt                          # Lance integrates t=1→0
  image = vae_decoder(denormalize(z))

Position handling per upstream Lance:
  - Text tokens get sequential 1D positions broadcast to 3 axes
  - Latent tokens get 3D grid positions (t=0, h=row, w=col) starting from text_len
  - MaPE re-anchors the latent temporal axis to 1000 (modality 4 = image-gen)

Latent grid layout for 768² output:
  - VAE downsample 16× → 48×48 latent grid per frame, 2304 tokens
  - 48-channel latent (Lance bundled VAE z_dim=48)
  - T=1 (single image)

No CFG in v1 — single conditional forward per step. Phase 3c will add
classifier-free guidance (text-guided minus unconditional × 4.0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_video.models.wan_2.vae22 import (
    VAE22_MEAN,
    VAE22_STD,
    Wan22VAEDecoder,
    denormalize_latents,
)
from mlx_vlm.models.qwen2_5_vl.config import TextConfig
from PIL import Image

from lance_mlx.model import LanceModel
from lance_mlx.model.flow_head import timestep_schedule
from lance_mlx.model.routing import PositionGroup
from lance_mlx.scheduler.solvers import DPMSolverPlusPlus2M


# Upstream Lance's t2i system-prompt instruction (from data/common.py
# `generate_system_prompt('t2i', 'image')`).
T2I_INSTRUCTION = (
    "Describe the image by detailing the color, quantity, text, shape, "
    "size, texture, spatial relationships of the objects and background:"
)

# MaPE anchor: per `data/common.py::shift_position_ids`, modality 4
# (image_gen) re-anchors the temporal axis of latent tokens to 1000.
MAPE_ANCHOR_IMAGE_GEN = 1000

# VAE constants for Lance's bundled Wan2.2 VAE.
VAE_LATENT_CHANNELS = 48
VAE_SPATIAL_DOWNSAMPLE = 16   # 768x768 image → 48x48 latent grid


class TextToImagePipeline:
    """Lance t2i — text prompt → PIL image via flow-matching."""

    def __init__(
        self,
        lance_model: LanceModel,
        vae_decoder: Wan22VAEDecoder,
        processor,                         # AutoProcessor (provides tokenizer)
        text_config: TextConfig,
        image_pad_token_id: int,
        video_pad_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
    ):
        self.lance_model = lance_model
        self.vae_decoder = vae_decoder
        self.processor = processor
        self.text_config = text_config
        self.image_pad_token_id = image_pad_token_id
        self.video_pad_token_id = video_pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

    @classmethod
    def from_pretrained(
        cls,
        lance_weights_dir: Path | str,
        vae_safetensors: Path | str,
        hf_processor_repo: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ) -> "TextToImagePipeline":
        lance_weights_dir = Path(lance_weights_dir)
        vae_safetensors = Path(vae_safetensors)

        # 1. Processor (tokenizer + chat template — no images here so the
        #    image_processor is unused but the tokenizer is essential).
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_processor_repo)
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        # 2. LanceModel (quantization-aware: applies nn.quantize if
        #    config.json has a 'quantization' block).
        from lance_mlx.model._loader import build_text_config, load_lance_model
        cfg = json.loads((lance_weights_dir / "config.json").read_text())
        text_cfg = build_text_config(cfg)
        lance_model = load_lance_model(lance_weights_dir)

        # 3. VAE decoder (skip encoder since t2i doesn't need it).
        vae_decoder = Wan22VAEDecoder(z_dim=VAE_LATENT_CHANNELS, dim=160, dec_dim=256)
        saved_vae = mx.load(str(vae_safetensors))
        # Only the decoder.* and conv2.* keys; the encoder.* + conv1.* belong
        # to the encoder which we don't load for pure t2i.
        dec_state = {
            k: v for k, v in saved_vae.items()
            if k.startswith("decoder.") or k.startswith("conv2.")
        }
        vae_decoder.load_weights(list(dec_state.items()))
        mx.eval(vae_decoder.parameters())

        return cls(
            lance_model=lance_model,
            vae_decoder=vae_decoder,
            processor=processor,
            text_config=text_cfg,
            image_pad_token_id=image_pad_id,
            video_pad_token_id=video_pad_id,
            vision_start_token_id=vision_start_id,
            vision_end_token_id=vision_end_id,
        )

    # ------------------------------------------------------------------ generate

    def generate(
        self,
        prompt: str,
        *,
        height: int = 768,
        width: int = 768,
        num_steps: int = 30,
        timestep_shift: float = 3.5,
        cfg_scale: float = 4.0,           # 1.0 disables CFG; Lance default 4.0
        cfg_renorm_type: str = "channel",  # Phase 5m fix: changed from 'global' to 'channel'.
                                          # 'channel' restores high-frequency detail at high n_lat
                                          # (768²×17f silent regression closed) without regressing
                                          # smaller scales (production-validated equivalence at
                                          # 768²×13f). Pass 'global' to restore legacy default.
        cfg_renorm_min: float = 0.0,      # Lance default: 0.0 (never upscale).
        seed: int = 42,
        verbose: bool = False,
        instruction: str = T2I_INSTRUCTION,
        latent_pos_base: int | None = None,
        scheduler: str = "euler",
    ) -> Image.Image:
        """Generate a single image from a text prompt.

        Args:
            prompt: text description.
            height/width: must be divisible by VAE_SPATIAL_DOWNSAMPLE=16. Default 768.
            num_steps: number of Euler steps. Default 30 per Lance config.
            timestep_shift: linear-schedule shift. Default 3.5 per Lance config.
            cfg_scale: classifier-free guidance scale. 1.0 = no CFG (single
                conditional forward per step). Lance config uses 4.0 — runs
                a second unconditional forward (empty user prompt) and blends
                velocities: v = v_uncond + cfg_scale * (v_cond - v_uncond).
            seed: RNG seed for noise init.
            verbose: print per-step latent stats.
            instruction: system-prompt instruction. Default = Lance t2i convention.
            scheduler: integration scheme. "euler" (default, 30 steps) or "dpm"
                (DPM-Solver++(2M), ~12 steps, ~2.4× faster at equivalent quality).

        Returns:
            PIL.Image (RGB).
        """
        if scheduler not in ("euler", "dpm"):
            raise ValueError(f"Unknown scheduler {scheduler!r}. Use 'euler' or 'dpm'.")
        assert height % VAE_SPATIAL_DOWNSAMPLE == 0
        assert width % VAE_SPATIAL_DOWNSAMPLE == 0
        h_lat = height // VAE_SPATIAL_DOWNSAMPLE
        w_lat = width // VAE_SPATIAL_DOWNSAMPLE
        n_lat = h_lat * w_lat                      # 2304 for 768²

        # --- Pre-build per-prompt state ---------------------------------
        # Each "state" packages the prompt-dependent tensors that don't
        # change across timesteps — input_ids, text_embeds, position_ids,
        # position_group, latent_positions. We build TWO states (with
        # prompt + empty prompt) so the flow loop can do CFG by calling
        # the model twice per step.
        cond_state = self._prepare_state(
            prompt=prompt, instruction=instruction,
            n_lat=n_lat, h_lat=h_lat, w_lat=w_lat, verbose=verbose,
            latent_pos_base=latent_pos_base,
        )
        if cfg_scale > 1.0:
            uncond_state = self._prepare_state(
                prompt="", instruction=instruction,
                n_lat=n_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
                latent_pos_base=latent_pos_base,
            )
            if verbose:
                print(f"  CFG enabled, scale={cfg_scale}, "
                      f"uncond tokens={uncond_state['T']}, cond tokens={cond_state['T']}")
        else:
            uncond_state = None

        # latent_pos_embed indices (shared across cond/uncond).
        max_side = 64  # Lance_3B latent_pos_embed is 64×64 = 4096
        lpe_indices = mx.array(
            [r * max_side + c for r in range(h_lat) for c in range(w_lat)],
            dtype=mx.int32,
        )

        # --- Init noise -------------------------------------------------
        mx.random.seed(seed)
        latents = mx.random.normal((1, 1, h_lat, w_lat, VAE_LATENT_CHANNELS))
        latents_dtype = self.lance_model.embed_tokens.weight.dtype
        latents = latents.astype(latents_dtype)

        # --- Flow loop -------------------------------------------------
        sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
        if verbose:
            print(f"  schedule: {[round(float(sched[i]), 4) for i in range(min(6, num_steps+1))]} ...")

        solver = DPMSolverPlusPlus2M() if scheduler == "dpm" else None

        for step in range(num_steps):
            t = sched[step]
            dt = sched[step] - sched[step + 1]

            v_cond = self._step_velocity(
                state=cond_state, latents=latents, t=t,
                lpe_indices=lpe_indices, n_lat=n_lat,
                h_lat=h_lat, w_lat=w_lat,
            )
            if uncond_state is not None:
                v_uncond = self._step_velocity(
                    state=uncond_state, latents=latents, t=t,
                    lpe_indices=lpe_indices, n_lat=n_lat,
                    h_lat=h_lat, w_lat=w_lat,
                )
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

                # CFG renormalization per upstream Lance (lance.py:712-725).
                # Without this, cfg_scale=4 makes |v_cfg| ~4x |v_cond|, causing
                # latents to overshoot and decode to a blurry/noisy mean. The
                # renorm clamps |v_cfg| to at most |v_cond|, restoring proper
                # step magnitudes.
                if cfg_renorm_type == "global":
                    norm_cond = mx.sqrt(mx.sum(v_cond * v_cond))   # scalar
                    norm_cfg  = mx.sqrt(mx.sum(v_cfg  * v_cfg))    # scalar
                    ratio = norm_cond / (norm_cfg + 1e-8)
                    scale = mx.clip(ratio, cfg_renorm_min, 1.0)
                    velocity = v_cfg * scale
                elif cfg_renorm_type == "channel":
                    # Per-channel norm: shape (1, 1, h, w, 1) so each spatial
                    # cell's channel-vector is rescaled independently.
                    norm_cond = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
                    norm_cfg  = mx.sqrt(mx.sum(v_cfg  * v_cfg,  axis=-1, keepdims=True))
                    ratio = norm_cond / (norm_cfg + 1e-8)
                    scale = mx.clip(ratio, cfg_renorm_min, 1.0)
                    velocity = v_cfg * scale
                else:   # "none" or anything else — no renorm
                    velocity = v_cfg
            else:
                velocity = v_cond

            if solver is not None:
                latents = solver.step(velocity, latents, dt)
            else:
                latents = latents - velocity * dt
            mx.eval(latents)

            if verbose:
                lat_np = latents.astype(mx.float32)
                print(f"  step {step+1}/{num_steps} t={float(t):.4f} dt={float(dt):.4f} "
                      f"  mean={float(mx.mean(lat_np)):.3f}  std={float(mx.std(lat_np)):.3f}")

        # --- 10. VAE decode --------------------------------------------
        if verbose:
            print(f"  VAE decode ...")
        # latents are in normalized space; decoder wants denormalized.
        z = denormalize_latents(latents).astype(self.vae_decoder.conv2.weight.dtype)
        decoded = self.vae_decoder(z)                            # (1, T', H', W', 3)
        mx.eval(decoded)

        # Take frame 0 (Wan2.2 VAE produces T'≥1 frames from causal padding).
        img_t = decoded[0, 0]                                    # (H', W', 3)
        import numpy as np
        img_np = np.array(img_t).astype(np.float32)
        # Map [-1, 1] → [0, 255] uint8.
        img_u8 = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_u8)

    # ----- per-prompt state assembly ---------------------------------------

    def _prepare_state(
        self,
        *,
        prompt: str,
        instruction: str,
        n_lat: int,
        h_lat: int,
        w_lat: int,
        verbose: bool,
        latent_pos_base: int | None = None,
    ) -> dict:
        """Pack the prompt-dependent state needed for one CFG-arm of the flow.

        Returns dict with: T, input_ids, text_embeds, latent_positions_arr,
        position_ids, position_group.
        """
        video_pad_str = "<|video_pad|>" * n_lat
        text = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<|vision_start|>{video_pad_str}<|vision_end|>"
        )

        tokenizer = self.processor.tokenizer
        input_ids = mx.array(
            [tokenizer(text, add_special_tokens=False)["input_ids"]],
            dtype=mx.int32,
        )
        T = input_ids.shape[1]
        if verbose:
            print(f"  prompt tokens: {T} ({T - n_lat} text + {n_lat} latent)")

        ids_list = input_ids[0].tolist()
        latent_positions = [
            i for i, v in enumerate(ids_list) if v == self.video_pad_token_id
        ]
        assert len(latent_positions) == n_lat, (
            f"expected {n_lat} latent positions, found {len(latent_positions)}"
        )
        latent_positions_arr = mx.array(latent_positions, dtype=mx.int32)
        first_latent_pos = latent_positions[0]
        text_len_before_latents = first_latent_pos
        if verbose:
            print(f"  latent positions span tokens {first_latent_pos}..{first_latent_pos + n_lat - 1}")

        position_ids = self._build_position_ids(
            T=T, n_lat=n_lat, h_lat=h_lat, w_lat=w_lat,
            text_len_before_latents=text_len_before_latents,
            latent_positions=latent_positions,
            latent_pos_base=latent_pos_base,
        )

        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)
        position_group = self._scatter_set(
            position_group, latent_positions_arr, int(PositionGroup.NOISY_VAE)
        )

        text_embeds = self.lance_model.embed_tokens(input_ids)   # (1, T, D)

        # Build the Lance attention mask: causal everywhere PLUS bidirectional
        # within the noisy-VAE token block. Per upstream `data/data_utils.py
        # create_sparse_mask`, the mask is `causal_mask OR full_and_noise_mask`
        # — tokens in the same "noise" segment all see each other regardless
        # of order. WITHOUT this, the noisy-VAE position 0 of a 2304-token
        # image grid can only see itself + text → no spatial context →
        # consistent painterly/blurry outputs.
        mask = self._build_block_mask(T, latent_positions, dtype=text_embeds.dtype)

        return {
            "T": T,
            "input_ids": input_ids,
            "text_embeds": text_embeds,
            "latent_positions_arr": latent_positions_arr,
            "position_ids": position_ids,
            "position_group": position_group,
            "mask": mask,
        }

    @staticmethod
    def _build_block_mask(T: int, latent_positions: list[int], dtype) -> mx.array:
        """Causal OR bidirectional-within-latent-block additive mask (T, T).

        Returns float mask with 0 where attention is allowed and -1e9
        (effectively -inf) where blocked. Shape (T, T) — broadcasts to
        (B, H, T, T) in SDP.
        """
        i = mx.arange(T)[:, None]      # (T, 1)
        j = mx.arange(T)[None, :]      # (1, T)
        lat_start = latent_positions[0]
        lat_end = latent_positions[-1] + 1
        in_lat_q = (i >= lat_start) & (i < lat_end)
        in_lat_kv = (j >= lat_start) & (j < lat_end)
        bidirectional = in_lat_q & in_lat_kv
        allowed = (i >= j) | bidirectional
        neg_inf = mx.array(-1e9, dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        return mx.where(allowed, zero, neg_inf)

    def _step_velocity(
        self,
        *,
        state: dict,
        latents: mx.array,
        t: mx.array,
        lpe_indices: mx.array,
        n_lat: int,
        h_lat: int,
        w_lat: int,
    ) -> mx.array:
        """One forward pass: returns velocity reshaped to latent grid.

        Args:
            state: from `_prepare_state` (text_embeds, position_ids, etc.)
            latents: current (B=1, T=1, h_lat, w_lat, C=48)
            t: scalar mx.array, current timestep
            lpe_indices: precomputed latent_pos_embed gather indices.

        Returns:
            velocity: (1, 1, h_lat, w_lat, C=48)
        """
        # CRITICAL: timestep_embed is added ONLY at VAE positions per upstream
        # Lance (modeling/lance/lance.py line ~668-670):
        #     vae_embed = self.vae2llm(x_t) + timestep_embed + latent_pos_embed
        #     current_sequence[current_vae_token_indexes_local] = vae_embed
        # Adding it via broadcast over the whole inputs_embeds polluted the text
        # conditioning with timestep noise and caused prompt-collapse on t2i
        # (Phase 3b/3c output looked like generic urban scenes regardless of CFG).
        latents_flat = latents.reshape(1, n_lat, VAE_LATENT_CHANNELS)
        pe = self.lance_model.latent_pos_embed(lpe_indices)[None, ...]   # (1, n_lat, D)
        t_emb = self.lance_model.time_embedder(t.reshape(1)).reshape(1, 1, -1)  # (1, 1, D)
        lat_embed = (
            self.lance_model.vae_in_proj(latents_flat)                   # (1, n_lat, D)
            + pe
            + t_emb                                                       # broadcast over n_lat
        )

        # Scatter the VAE embed (which already includes timestep) into the
        # text-embedded sequence ONLY at the latent positions. Text positions
        # keep their pure token embeddings.
        inputs_embeds = self._scatter_embeds(
            state["text_embeds"], lat_embed, state["latent_positions_arr"],
        )

        h = self.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=state["position_ids"],
            position_group=state["position_group"],
            mask=state["mask"],     # causal + bidirectional-within-latent block
        )
        h_lat_pos = h[:, state["latent_positions_arr"], :]
        velocity_flat = self.lance_model.llm2vae(h_lat_pos)
        return velocity_flat.reshape(1, 1, h_lat, w_lat, VAE_LATENT_CHANNELS)

    # ----- helpers -----

    @staticmethod
    def _scatter_set(arr: mx.array, idx: mx.array, value: int) -> mx.array:
        """Return arr with arr[idx] = value (functional, no in-place)."""
        # MLX doesn't have first-class scatter for this; use a numpy roundtrip
        # since the data is small (positions are ints, 1D).
        import numpy as np
        out_np = np.array(arr)
        out_np[np.array(idx)] = value
        return mx.array(out_np)

    @staticmethod
    def _scatter_embeds(
        base: mx.array,           # (B, T, D)
        inserts: mx.array,        # (B, N, D)
        positions: mx.array,      # (N,) int — positions in T where inserts go
    ) -> mx.array:
        """Return `base` with `inserts` slotted at `positions` along the T axis."""
        # Numpy roundtrip via fp32 because numpy can't directly buffer bf16
        # (PEP 3118 mismatch). Cast to fp32 → np → scatter → mx → back to
        # original dtype. base is sized (1, ~2400, 2048) ~10 MB at bf16,
        # acceptable for once-per-step embed assembly.
        import numpy as np
        target_dtype = base.dtype
        out_np = np.array(base.astype(mx.float32))
        ins_np = np.array(inserts.astype(mx.float32))
        pos_np = np.array(positions)
        out_np[:, pos_np, :] = ins_np
        return mx.array(out_np).astype(target_dtype)

    def _build_position_ids(
        self,
        *,
        T: int,
        n_lat: int,
        h_lat: int,
        w_lat: int,
        text_len_before_latents: int,
        latent_positions: list[int],
        latent_pos_base: int | None = None,
    ) -> mx.array:
        """Build (3, 1, T) position_ids with text + 3D latent grid + MaPE re-anchor.

        Layout per upstream Lance:
          - Token positions 0..text_len_before_latents-1: sequential 1D, broadcast to 3 axes.
          - Token positions latent_positions: 3D grid coords (t=0, h=row, w=col).
          - All other positions (e.g. trailing vision_end): sequential, continuing from
            text_len_before_latents.

        Then apply MaPE: temporal axis of latent positions → all anchored to 1000.

        `latent_pos_base`: experimental hook mirroring t2v.py Phase 5j.
        **Default is `None` (legacy = text-position anchor at
        `text_len_before_latents`) — production-validated since Phase 3e.**
        Setting to `0` anchors latent grid at origin (matches t2v.py's
        Phase 5j fix convention). t2i has a single trailing latent block
        like t2v, so this is a safe experimental hook. Whether it improves
        output for image generation is an open question — t2i has been
        producing photoreal output with the legacy default for months.
        """
        import numpy as np
        pos = np.zeros((3, 1, T), dtype=np.int32)
        # Default: sequential 1D, broadcast to 3 axes (fallback for any
        # non-latent position).
        seq = np.arange(T, dtype=np.int32)
        pos[0, 0, :] = seq
        pos[1, 0, :] = seq
        pos[2, 0, :] = seq

        # Override at latent positions with 3D grid (t=0, h=row, w=col).
        # The anchor: either text_len_before_latents (legacy) or fixed
        # origin (Phase 5j-style).
        base = text_len_before_latents if latent_pos_base is None else int(latent_pos_base)
        for idx, token_pos in enumerate(latent_positions):
            r = idx // w_lat
            c = idx % w_lat
            pos[0, 0, token_pos] = base + 0      # t-axis, all 0 in the grid (t=1)
            pos[1, 0, token_pos] = base + r      # h-axis
            pos[2, 0, token_pos] = base + c      # w-axis

        # Tokens AFTER the latent block (vision_end) should continue counting
        # from the max position used so far. The max image grid position is
        # base + max(r, c) = base + max(h_lat-1, w_lat-1).
        max_grid = max(h_lat, w_lat) - 1
        after_latents_start = latent_positions[-1] + 1
        if after_latents_start < T:
            tail_len = T - after_latents_start
            tail = base + max_grid + 1 + np.arange(tail_len, dtype=np.int32)
            pos[:, 0, after_latents_start:] = tail[None, :]

        # MaPE re-anchor: temporal axis of latent positions → all anchored to 1000.
        # Per upstream `shift_position_ids`: shift = 1000 - first_latent_t_pos.
        first_latent_t = pos[0, 0, latent_positions[0]]
        shift = MAPE_ANCHOR_IMAGE_GEN - int(first_latent_t)
        for token_pos in latent_positions:
            pos[0, 0, token_pos] += shift

        return mx.array(pos)
