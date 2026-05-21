"""image_edit — instruction-based image editing via Lance + Wan2.2 VAE.

Phase 3.5 MVP. Derives from `t2i.py` but adds a CLEAN VAE-encoded reference
of the input image, so the same flow loop denoises the target latent while
conditioning on the source image.

Algorithmic overview (per upstream `Lance.validation_gen`, simplified):

  z_clean = vae_encoder(input_image)                   # (1, 1, h_lat, w_lat, 48)
  z_t = randn_like(z_clean)                            # noisy target init

  for t in schedule(1.0 → 0.0, 30 steps):
      # Both blocks get pos_embed + vae_in_proj; only target gets time_embed(t).
      # Clean block uses time_embed(0).
      embed[clean_positions] = vae_in_proj(z_clean) + latent_pos_embed + time_embedder(0)
      embed[noisy_positions] = vae_in_proj(z_t)     + latent_pos_embed + time_embedder(t)
      h = lance_model(embed, position_ids, position_group, mask)
      velocity = llm2vae(h[noisy_positions])           # (B, N_lat, 48)
      z_t = z_t - velocity * dt
  image = vae_decoder(denormalize(z_t))

Token sequence layout:

  <|im_start|>system\n{edit_instruction}<|im_end|>\n
  <|im_start|>user\n<|vision_start|>{video_pad × n_lat (clean ref)}<|vision_end|>{user_text}<|im_end|>\n
  <|im_start|>assistant\n<|vision_start|>{video_pad × n_lat (noisy target)}<|vision_end|>

Attention mask: causal default, bidirectional within EACH of the two latent
blocks (clean ref and noisy target are bidirectional internally; noisy target
sees clean ref causally; everything sees prior text causally).

Position IDs: both latent blocks get 3D grid (t-axis re-anchored to 1000 via
MaPE, h/w from spatial grid). Both blocks share the SAME positional structure
since they cover the same image grid spatially. This mirrors upstream's
"modality 1 := modality 2 positions" rule, which aligns target with clean ref.

MVP simplification vs full upstream: SKIP the Qwen2.5-VL ViT semantic stream
of the input image. Clean VAE conditioning alone should provide the visual
anchor for what to preserve from the input. If quality suffers, Phase 3.6
will add the ViT stream.

Default weights: Lance_3B (image specialist) for crystal-clear output.
Lance_3B_Video also works and is what Phase 0 oracle used, but produces
the same painterly aesthetic seen in Phase 4c findings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx_video.models.wan_2.vae22 import (
    Wan22VAEDecoder,
    Wan22VAEEncoder,
    denormalize_latents,
)
from mlx_vlm.models.qwen2_5_vl.config import TextConfig
from PIL import Image

from lance_mlx.model import LanceModel
from lance_mlx.model.flow_head import timestep_schedule
from lance_mlx.model.routing import PositionGroup


# Upstream Lance's image_edit system-prompt instruction (from data/common.py
# `generate_system_prompt("edit", "image")`).
EDIT_INSTRUCTION = (
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)

# MaPE anchor: per `data/common.py::shift_position_ids`, image-gen tokens
# re-anchor the temporal axis to 1000.
MAPE_ANCHOR_IMAGE_GEN = 1000

# VAE constants for Lance's bundled Wan2.2 VAE.
VAE_LATENT_CHANNELS = 48
VAE_SPATIAL_DOWNSAMPLE = 16   # 768x768 image → 48x48 latent grid


class ImageEditPipeline:
    """Lance image_edit — input image + text instruction → edited PIL image."""

    def __init__(
        self,
        lance_model: LanceModel,
        vae_encoder: Wan22VAEEncoder,
        vae_decoder: Wan22VAEDecoder,
        processor,
        text_config: TextConfig,
        image_pad_token_id: int,
        video_pad_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
    ):
        self.lance_model = lance_model
        self.vae_encoder = vae_encoder
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
    ) -> "ImageEditPipeline":
        lance_weights_dir = Path(lance_weights_dir)
        vae_safetensors = Path(vae_safetensors)

        # 1. Processor.
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_processor_repo)
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        # 2. LanceModel.
        cfg = json.loads((lance_weights_dir / "config.json").read_text())
        text_cfg = TextConfig(
            model_type=cfg["model_type"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            vocab_size=cfg["vocab_size"],
            num_key_value_heads=cfg.get("num_key_value_heads"),
            max_position_embeddings=cfg.get("max_position_embeddings", 128000),
            rope_theta=cfg.get("rope_theta", 1e6),
            rope_scaling=cfg.get("rope_scaling"),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )
        saved_lance = mx.load(str(lance_weights_dir / "model.safetensors"))
        num_latent_positions = saved_lance["latent_pos_embed.pos_embed"].shape[0]
        lance_model = LanceModel(text_cfg, num_latent_positions=num_latent_positions)
        lance_model.load_weights(list(saved_lance.items()))
        mx.eval(lance_model.parameters())

        # 3. VAE encoder + decoder (both needed for edit).
        saved_vae = mx.load(str(vae_safetensors))

        vae_encoder = Wan22VAEEncoder(z_dim=VAE_LATENT_CHANNELS, dim=160)
        enc_state = {
            k: v for k, v in saved_vae.items()
            if k.startswith("encoder.") or k.startswith("conv1.")
        }
        vae_encoder.load_weights(list(enc_state.items()))
        mx.eval(vae_encoder.parameters())

        vae_decoder = Wan22VAEDecoder(z_dim=VAE_LATENT_CHANNELS, dim=160, dec_dim=256)
        dec_state = {
            k: v for k, v in saved_vae.items()
            if k.startswith("decoder.") or k.startswith("conv2.")
        }
        vae_decoder.load_weights(list(dec_state.items()))
        mx.eval(vae_decoder.parameters())

        return cls(
            lance_model=lance_model,
            vae_encoder=vae_encoder,
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
        input_image: Image.Image | Path | str,
        instruction: str,
        *,
        height: int = 768,
        width: int = 768,
        num_steps: int = 30,
        timestep_shift: float = 3.5,
        cfg_scale: float = 4.0,
        cfg_renorm_type: str = "global",
        cfg_renorm_min: float = 0.0,
        seed: int = 42,
        verbose: bool = False,
        system_prompt: str = EDIT_INSTRUCTION,
    ) -> Image.Image:
        """Generate an edited image.

        Args:
            input_image: PIL image, Path, or str path to source image.
            instruction: edit instruction (e.g. "Remove the hat from the painting.")
            height/width: output dimensions, divisible by 16. Default 768.
            num_steps: Euler steps. Default 30.
            timestep_shift: linear-schedule shift. Default 3.5.
            cfg_scale: text CFG scale. Lance default 4.0.
            seed: RNG seed for noise init.
            verbose: print per-step latent stats.
            system_prompt: edit-style system instruction.

        Returns:
            PIL.Image (RGB).
        """
        assert height % VAE_SPATIAL_DOWNSAMPLE == 0
        assert width % VAE_SPATIAL_DOWNSAMPLE == 0
        h_lat = height // VAE_SPATIAL_DOWNSAMPLE
        w_lat = width // VAE_SPATIAL_DOWNSAMPLE
        n_lat = h_lat * w_lat

        # --- Encode source image to clean VAE latent --------------------
        if verbose:
            print(f"  VAE-encoding source image to {h_lat}×{w_lat}×{VAE_LATENT_CHANNELS} latent ...")
        src_t = self._load_image_tensor(input_image, height=height, width=width)
        z_clean = self.vae_encoder(src_t)                         # (1, 1, h_lat, w_lat, 48)
        mx.eval(z_clean)
        z_clean = z_clean.astype(self.lance_model.embed_tokens.weight.dtype)
        if verbose:
            print(f"  z_clean: shape={tuple(z_clean.shape)} "
                  f"mean={float(mx.mean(z_clean)):.3f} std={float(mx.std(z_clean)):.3f}")

        # --- Pre-build per-prompt state ---------------------------------
        cond_state = self._prepare_state(
            instruction=instruction, system_prompt=system_prompt,
            n_lat=n_lat, h_lat=h_lat, w_lat=w_lat, verbose=verbose,
        )
        if cfg_scale > 1.0:
            uncond_state = self._prepare_state(
                instruction="", system_prompt=system_prompt,
                n_lat=n_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
            )
            if verbose:
                print(f"  CFG enabled, scale={cfg_scale}, "
                      f"cond T={cond_state['T']} uncond T={uncond_state['T']}")
        else:
            uncond_state = None

        # latent_pos_embed indices for one image grid (shared by clean + noisy).
        max_side = 64  # Lance_3B latent_pos_embed is 64×64 = 4096
        lpe_indices = mx.array(
            [r * max_side + c for r in range(h_lat) for c in range(w_lat)],
            dtype=mx.int32,
        )

        # --- Init noise -------------------------------------------------
        mx.random.seed(seed)
        z_t = mx.random.normal((1, 1, h_lat, w_lat, VAE_LATENT_CHANNELS))
        z_t = z_t.astype(z_clean.dtype)

        # --- Flow loop --------------------------------------------------
        sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
        if verbose:
            print(f"  schedule: {[round(float(sched[i]), 4) for i in range(min(6, num_steps+1))]} ...")

        for step in range(num_steps):
            t = sched[step]
            dt = sched[step] - sched[step + 1]

            v_cond = self._step_velocity(
                state=cond_state, z_t=z_t, z_clean=z_clean, t=t,
                lpe_indices=lpe_indices, n_lat=n_lat,
                h_lat=h_lat, w_lat=w_lat,
            )
            if uncond_state is not None:
                v_uncond = self._step_velocity(
                    state=uncond_state, z_t=z_t, z_clean=z_clean, t=t,
                    lpe_indices=lpe_indices, n_lat=n_lat,
                    h_lat=h_lat, w_lat=w_lat,
                )
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
                if cfg_renorm_type == "global":
                    norm_cond = mx.sqrt(mx.sum(v_cond * v_cond))
                    norm_cfg = mx.sqrt(mx.sum(v_cfg * v_cfg))
                    ratio = norm_cond / (norm_cfg + 1e-8)
                    scale = mx.clip(ratio, cfg_renorm_min, 1.0)
                    velocity = v_cfg * scale
                elif cfg_renorm_type == "channel":
                    norm_cond = mx.sqrt(mx.sum(v_cond * v_cond, axis=-1, keepdims=True))
                    norm_cfg = mx.sqrt(mx.sum(v_cfg * v_cfg, axis=-1, keepdims=True))
                    ratio = norm_cond / (norm_cfg + 1e-8)
                    scale = mx.clip(ratio, cfg_renorm_min, 1.0)
                    velocity = v_cfg * scale
                else:
                    velocity = v_cfg
            else:
                velocity = v_cond

            z_t = z_t - velocity * dt
            mx.eval(z_t)

            if verbose:
                z_np = z_t.astype(mx.float32)
                print(f"  step {step+1}/{num_steps} t={float(t):.4f} dt={float(dt):.4f} "
                      f"  mean={float(mx.mean(z_np)):.3f}  std={float(mx.std(z_np)):.3f}")

        # --- VAE decode -------------------------------------------------
        if verbose:
            print(f"  VAE decode ...")
        z = denormalize_latents(z_t).astype(self.vae_decoder.conv2.weight.dtype)
        decoded = self.vae_decoder(z)
        mx.eval(decoded)
        img_t = decoded[0, 0]
        img_np = np.array(img_t.astype(mx.float32))
        img_u8 = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_u8)

    # ----- per-prompt state assembly ---------------------------------------

    def _prepare_state(
        self,
        *,
        instruction: str,
        system_prompt: str,
        n_lat: int,
        h_lat: int,
        w_lat: int,
        verbose: bool,
    ) -> dict:
        """Pack state for one CFG arm. Two latent blocks: clean ref + noisy target."""
        video_pad_str = "<|video_pad|>" * n_lat
        # User block: clean ref image + edit text instruction.
        # Assistant block: noisy target placeholder.
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{video_pad_str}<|vision_end|>{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n<|vision_start|>{video_pad_str}<|vision_end|>"
        )
        tokenizer = self.processor.tokenizer
        input_ids = mx.array(
            [tokenizer(text, add_special_tokens=False)["input_ids"]],
            dtype=mx.int32,
        )
        T = input_ids.shape[1]

        # Two latent blocks: first occurrence = clean ref, second = noisy target.
        ids_list = input_ids[0].tolist()
        video_pad_positions = [
            i for i, v in enumerate(ids_list) if v == self.video_pad_token_id
        ]
        assert len(video_pad_positions) == 2 * n_lat, (
            f"expected {2 * n_lat} video_pad tokens, found {len(video_pad_positions)}"
        )
        clean_positions = video_pad_positions[:n_lat]
        noisy_positions = video_pad_positions[n_lat:]
        if verbose:
            print(f"  prompt tokens: {T}  "
                  f"clean=[{clean_positions[0]}..{clean_positions[-1]}]  "
                  f"noisy=[{noisy_positions[0]}..{noisy_positions[-1]}]")

        clean_positions_arr = mx.array(clean_positions, dtype=mx.int32)
        noisy_positions_arr = mx.array(noisy_positions, dtype=mx.int32)

        position_ids = self._build_position_ids(
            T=T, h_lat=h_lat, w_lat=w_lat,
            clean_positions=clean_positions,
            noisy_positions=noisy_positions,
        )

        # PositionGroup: TEXT default, CLEAN_VAE at clean, NOISY_VAE at noisy.
        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)
        position_group = self._scatter_set(
            position_group, clean_positions_arr, int(PositionGroup.CLEAN_VAE)
        )
        position_group = self._scatter_set(
            position_group, noisy_positions_arr, int(PositionGroup.NOISY_VAE)
        )

        text_embeds = self.lance_model.embed_tokens(input_ids)

        mask = self._build_block_mask(
            T, clean_positions, noisy_positions, dtype=text_embeds.dtype
        )

        return {
            "T": T,
            "input_ids": input_ids,
            "text_embeds": text_embeds,
            "clean_positions_arr": clean_positions_arr,
            "noisy_positions_arr": noisy_positions_arr,
            "position_ids": position_ids,
            "position_group": position_group,
            "mask": mask,
        }

    @staticmethod
    def _build_block_mask(
        T: int,
        clean_positions: list[int],
        noisy_positions: list[int],
        dtype,
    ) -> mx.array:
        """Causal OR (bidir-in-clean) OR (bidir-in-noisy) additive mask (T, T).

        Per upstream `data/data_utils.py::create_sparse_mask`, after the
        "full_noise → full" rewrite, both clean ref and noisy target blocks
        are bidirectional internally. Cross-block attention is causal only
        (target sees clean ref since it comes later; clean ref cannot see
        target because that would be acausal).
        """
        i = mx.arange(T)[:, None]
        j = mx.arange(T)[None, :]
        c_start, c_end = clean_positions[0], clean_positions[-1] + 1
        n_start, n_end = noisy_positions[0], noisy_positions[-1] + 1
        in_clean_q = (i >= c_start) & (i < c_end)
        in_clean_kv = (j >= c_start) & (j < c_end)
        in_noisy_q = (i >= n_start) & (i < n_end)
        in_noisy_kv = (j >= n_start) & (j < n_end)
        bidir_clean = in_clean_q & in_clean_kv
        bidir_noisy = in_noisy_q & in_noisy_kv
        allowed = (i >= j) | bidir_clean | bidir_noisy
        neg_inf = mx.array(-1e9, dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        return mx.where(allowed, zero, neg_inf)

    def _step_velocity(
        self,
        *,
        state: dict,
        z_t: mx.array,           # current noisy latent (1, 1, h_lat, w_lat, 48)
        z_clean: mx.array,       # clean encoded ref (1, 1, h_lat, w_lat, 48)
        t: mx.array,
        lpe_indices: mx.array,
        n_lat: int,
        h_lat: int,
        w_lat: int,
    ) -> mx.array:
        """One forward pass: assemble both latent blocks, return velocity at noisy."""
        z_clean_flat = z_clean.reshape(1, n_lat, VAE_LATENT_CHANNELS)
        z_t_flat = z_t.reshape(1, n_lat, VAE_LATENT_CHANNELS)

        pe = self.lance_model.latent_pos_embed(lpe_indices)[None, ...]   # (1, n_lat, D)

        # Clean ref block: time_embedder at t=0 (clean conditioning).
        # Upstream initializes timestep[clean_positions] = 0 and only sets
        # timestep[noisy_positions] = t_step (see lance_lance.py:644-648).
        t_zero = mx.zeros((1,), dtype=t.dtype)
        t_emb_clean = self.lance_model.time_embedder(t_zero).reshape(1, 1, -1)
        clean_embed = (
            self.lance_model.vae_in_proj(z_clean_flat) + pe + t_emb_clean
        )

        # Noisy target block: time_embedder at current t.
        t_emb_noisy = self.lance_model.time_embedder(t.reshape(1)).reshape(1, 1, -1)
        noisy_embed = (
            self.lance_model.vae_in_proj(z_t_flat) + pe + t_emb_noisy
        )

        # Scatter both into the text-embedded sequence.
        inputs_embeds = self._scatter_two_blocks(
            state["text_embeds"],
            clean_embed, state["clean_positions_arr"],
            noisy_embed, state["noisy_positions_arr"],
        )

        h = self.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=state["position_ids"],
            position_group=state["position_group"],
            mask=state["mask"],
        )
        h_noisy = h[:, state["noisy_positions_arr"], :]
        velocity_flat = self.lance_model.llm2vae(h_noisy)
        return velocity_flat.reshape(1, 1, h_lat, w_lat, VAE_LATENT_CHANNELS)

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _load_image_tensor(
        image: Image.Image | Path | str,
        height: int = 768,
        width: int = 768,
    ) -> mx.array:
        """Load image → (1, T=1, H, W, 3) in [-1, 1]."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        arr = arr[None, None, ...]
        return mx.array(arr)

    @staticmethod
    def _scatter_set(arr: mx.array, idx: mx.array, value: int) -> mx.array:
        """Return arr with arr[idx] = value (functional)."""
        out_np = np.array(arr)
        out_np[np.array(idx)] = value
        return mx.array(out_np)

    @staticmethod
    def _scatter_two_blocks(
        base: mx.array,                # (1, T, D)
        clean_block: mx.array,         # (1, n_lat, D)
        clean_pos: mx.array,           # (n_lat,) int
        noisy_block: mx.array,         # (1, n_lat, D)
        noisy_pos: mx.array,           # (n_lat,) int
    ) -> mx.array:
        """Insert both latent blocks into the text-embedded sequence."""
        target_dtype = base.dtype
        out_np = np.array(base.astype(mx.float32))
        c_np = np.array(clean_block.astype(mx.float32))
        n_np = np.array(noisy_block.astype(mx.float32))
        cp = np.array(clean_pos)
        npn = np.array(noisy_pos)
        out_np[:, cp, :] = c_np
        out_np[:, npn, :] = n_np
        return mx.array(out_np).astype(target_dtype)

    def _build_position_ids(
        self,
        *,
        T: int,
        h_lat: int,
        w_lat: int,
        clean_positions: list[int],
        noisy_positions: list[int],
    ) -> mx.array:
        """Build (3, 1, T) position_ids with both latent blocks placed as 3D grids.

        Both clean ref and noisy target cover the SAME image grid spatially,
        so they share the same h/w structure. Per upstream's "modality 1 :=
        modality 2 positions" rule, they ALSO share the same temporal anchor.

        MaPE re-anchor: both blocks' t-axis → 1000 (image-gen anchor).
        """
        pos = np.zeros((3, 1, T), dtype=np.int32)
        seq = np.arange(T, dtype=np.int32)
        pos[0, 0, :] = seq
        pos[1, 0, :] = seq
        pos[2, 0, :] = seq

        # Both latent blocks use the SAME spatial grid relative to their start.
        # Clean ref starts at clean_positions[0], grid base = first text-len.
        # We use the clean block's text-len base for BOTH blocks' h/w grid
        # so they spatially align. The MaPE shift sets t-axis to 1000 for both.
        clean_base = clean_positions[0]
        for idx, token_pos in enumerate(clean_positions):
            r = idx // w_lat
            c = idx % w_lat
            pos[0, 0, token_pos] = clean_base + 0
            pos[1, 0, token_pos] = clean_base + r
            pos[2, 0, token_pos] = clean_base + c

        for idx, token_pos in enumerate(noisy_positions):
            r = idx // w_lat
            c = idx % w_lat
            # Same h/w as clean ref (spatial alignment per upstream rule).
            pos[0, 0, token_pos] = clean_base + 0
            pos[1, 0, token_pos] = clean_base + r
            pos[2, 0, token_pos] = clean_base + c

        # Tokens AFTER the noisy block (vision_end at end) continue counting.
        max_grid = max(h_lat, w_lat) - 1
        after_clean = clean_positions[-1] + 1
        before_noisy = noisy_positions[0]
        # Gap between clean and noisy: vision_end + chat-template tokens.
        # Continue sequential counts from clean_base + max_grid + 1.
        if before_noisy > after_clean:
            gap_len = before_noisy - after_clean
            tail = clean_base + max_grid + 1 + np.arange(gap_len, dtype=np.int32)
            pos[:, 0, after_clean:before_noisy] = tail[None, :]
        # Tokens after noisy block:
        after_noisy = noisy_positions[-1] + 1
        if after_noisy < T:
            # Continue counting from the highest position used so far.
            # (Both blocks max at clean_base + max_grid; sequential tail goes from there + gap_len.)
            gap_len = before_noisy - after_clean
            tail_start = clean_base + max_grid + 1 + gap_len
            tail_len = T - after_noisy
            tail = tail_start + np.arange(tail_len, dtype=np.int32)
            pos[:, 0, after_noisy:] = tail[None, :]

        # MaPE re-anchor: BOTH latent blocks' t-axis → 1000.
        first_latent_t = pos[0, 0, clean_positions[0]]
        shift = MAPE_ANCHOR_IMAGE_GEN - int(first_latent_t)
        for token_pos in clean_positions + noisy_positions:
            pos[0, 0, token_pos] += shift

        return mx.array(pos)
