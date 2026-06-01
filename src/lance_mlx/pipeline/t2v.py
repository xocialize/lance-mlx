"""t2v — text-to-video generation via Lance_3B_Video + Wan2.2 VAE.

Phase 4a MVP. Extends the Phase 3e t2i pipeline pattern with:

  - Lance_3B_Video LLM checkpoint (126976-entry latent_pos_embed table
    covering 31 temporal × 64×64 spatial positions).
  - 3D latent grid (T_lat, h_lat, w_lat) instead of (1, h_lat, w_lat).
  - 3D position-ID construction: t-axis varies per frame.
  - MaPE re-anchor to 2000 (modality 3 = video_gen) instead of 1000.
  - MP4 output via imageio-ffmpeg.

All other Phase 3 learnings carry over:
  - System prompt = Lance t2v instruction (`generate_system_prompt('t2v', 'video')`)
  - Image-as-video convention: `<|video_pad|>` for latent placeholder
  - Timestep embed added ONLY at VAE positions (Phase 3d fix)
  - Custom mask: causal OR bidirectional-within-latent-block (Phase 3e fix)
  - CFG with renormalization (`cfg_renorm_type='global'`)

VAE compression for Wan2.2 (per HANDOFF):
  - Temporal 4× (50 frames → 13 latent frames, via T_lat = (T-1)//4 + 1)
  - Spatial 16× (768x768 → 48x48 latent)
  - 48-channel latent (Lance bundled VAE z_dim=48)

For 768x768 × 50 frames: 13 × 48 × 48 = 29952 latent tokens. Very large.
This MVP uses smaller dims (256x256 × 16 frames → 1280 latent tokens) for
fast iteration before scaling up.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_video.models.wan_2.vae22 import (
    Wan22VAEDecoder,
    denormalize_latents,
)
from mlx_vlm.models.qwen2_5_vl.config import TextConfig

from lance_mlx.model import LanceModel
from lance_mlx.model.lance_llm import resolve_memory_mode
from lance_mlx.model.flow_head import timestep_schedule
from lance_mlx.model.routing import PositionGroup


# Upstream Lance's t2v system-prompt instruction
# (from data/common.py::generate_system_prompt('t2v', 'video')).
T2V_INSTRUCTION = (
    "Describe the video by detailing the color, quantity, visible text, "
    "shape, size, texture, spatial relationships and motion/camera "
    "movements of the objects and background:"
)

# MaPE anchor for video_gen (modality 3) per upstream `shift_position_ids`.
MAPE_ANCHOR_VIDEO_GEN = 2000

# VAE constants for Lance's bundled Wan2.2 VAE.
VAE_LATENT_CHANNELS = 48
VAE_SPATIAL_DOWNSAMPLE = 16
VAE_TEMPORAL_DOWNSAMPLE = 4    # First chunk = 1 frame, rest = 4 frames each

# Lance_3B_Video latent_pos_embed table dims (per Phase 1a inspection).
MAX_LATENT_SIDE = 64                   # spatial max per axis
MAX_LATENT_FRAMES = 31                 # temporal max
MAX_NUM_LATENT_POSITIONS = MAX_LATENT_FRAMES * MAX_LATENT_SIDE * MAX_LATENT_SIDE   # = 126976


class TextToVideoPipeline:
    """Lance t2v — text prompt → MP4 video via flow-matching."""

    def __init__(
        self,
        lance_model: LanceModel,
        vae_decoder: Wan22VAEDecoder,
        processor,
        text_config: TextConfig,
        image_pad_token_id: int,
        video_pad_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        memory_mode: str = "parallel",
        vae_deferred: bool = False,
    ):
        self.lance_model = lance_model
        self.vae_decoder = vae_decoder
        self.processor = processor
        self.text_config = text_config
        self.image_pad_token_id = image_pad_token_id
        self.video_pad_token_id = video_pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        # Resolved memory strategy ("parallel"|"relay"). "relay" is a load-time
        # decision (GEN tower + VAE decoder loaded lazily), so it is fixed here;
        # generate() honors it. See resolve_memory_mode for the auto-detect bands.
        self.memory_mode = memory_mode
        # In relay mode the VAE decoder is loaded lazily and materialized at decode
        # (it is dead weight through prefill + the whole denoise loop otherwise).
        self._vae_deferred = vae_deferred

    @property
    def und_mode(self) -> str:
        """Deprecated read-only alias for ``memory_mode`` (parallel→"keep",
        relay→"deferred"). Kept one cycle for callers/scripts that read it."""
        return "keep" if self.memory_mode == "parallel" else "deferred"

    @classmethod
    def from_pretrained(
        cls,
        lance_weights_dir: Path | str,
        vae_safetensors: Path | str,
        hf_processor_repo: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        memory_mode: str | None = None,
        und_mode: str | None = None,
    ) -> "TextToVideoPipeline":
        """Loads Lance_3B_Video LLM + Wan2.2 VAE decoder + tokenizer.

        `memory_mode` ("auto"|"parallel"|"relay", default auto): how the three
        generation phases (prefill→UND, denoise→GEN, decode→VAE) share memory.
        "relay" is the only mode that changes the LOAD itself — it loads only
        UND+shared (GEN tower lazy) AND leaves the VAE decoder lazy (~1.1 GB), so
        neither inflates the prefill/denoise peak; GEN materializes after the
        UND-only prefill (~5-10 s) and VAE at decode. The towers never co-reside
        (~9-10 GB peak, fits a 16 GB Mac) but the pipeline is SINGLE-SHOT. "auto"
        is BINARY — "parallel" (ws ≥ 18 GiB, a 24 GB+ machine that holds everything
        resident + reusable) or "relay" (below, incl. 16 GB Macs). `und_mode` is a
        deprecated alias (keep→parallel, deferred/free→relay). Mirrors
        t2i.from_pretrained; the tower mechanics are shared in LanceModel."""
        lance_weights_dir = Path(lance_weights_dir)
        vae_safetensors = Path(vae_safetensors)
        resolved_mode = resolve_memory_mode(memory_mode, und_mode=und_mode)

        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_processor_repo)
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        # LanceModel (must be Lance_3B_Video for video generation — the
        # latent_pos_embed table needs 126976 entries for temporal coverage).
        # Quantization-aware loader applies nn.quantize if config says so.
        from lance_mlx.model._loader import build_text_config, load_lance_model
        cfg = json.loads((lance_weights_dir / "config.json").read_text())
        text_cfg = build_text_config(cfg)
        # Peek at latent_pos_embed shape for the warning, then load.
        _saved_peek = mx.load(str(lance_weights_dir / "model.safetensors"))
        num_latent_positions = _saved_peek["latent_pos_embed.pos_embed"].shape[0]
        del _saved_peek
        if num_latent_positions != MAX_NUM_LATENT_POSITIONS:
            print(f"WARNING: latent_pos_embed has {num_latent_positions} entries; "
                  f"video pipeline expects {MAX_NUM_LATENT_POSITIONS} (= 31×64×64). "
                  f"Are you using Lance_3B_Video weights?")
        lance_model = load_lance_model(
            lance_weights_dir,
            defer_gen_tower=(resolved_mode == "relay"),
        )

        vae_decoder = Wan22VAEDecoder(z_dim=VAE_LATENT_CHANNELS, dim=160, dec_dim=256)
        saved_vae = mx.load(str(vae_safetensors))
        dec_state = {
            k: v for k, v in saved_vae.items()
            if k.startswith("decoder.") or k.startswith("conv2.")
        }
        vae_decoder.load_weights(list(dec_state.items()))
        # In relay mode leave the decoder lazy/mmap-backed: it is ~1.1 GB of dead
        # weight through prefill + the whole denoise loop and is only touched at
        # decode (generate() materializes it then). `load_weights` only assigns
        # refs, so skipping the eval keeps it out of the binding loop ceiling.
        vae_deferred = (resolved_mode == "relay")
        if not vae_deferred:
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
            memory_mode=resolved_mode,
            vae_deferred=vae_deferred,
        )

    def _materialize_vae(self) -> dict | None:
        """Eval the deferred VAE-decoder params (relay mode), pulling them from the
        retained safetensors mmap into Metal buffers (~1.1 GB). Called at the TOP of
        the decode block — AFTER the GEN tower is shed — so the decoder is loaded
        exactly when its phase begins, never co-resident with the towers. No-op (and
        returns None) in parallel mode (VAE already eager at load)."""
        if not getattr(self, "_vae_deferred", False):
            return None
        before = mx.get_active_memory()
        mx.eval(self.vae_decoder.parameters())
        self._vae_deferred = False
        after = mx.get_active_memory()
        return {"materialized_bytes": after - before, "active_after": after}

    # ------------------------------------------------------------------ generate

    def generate(
        self,
        prompt: str,
        *,
        num_frames: int = 16,                # 50 per Lance default; smaller for MVP
        height: int = 256,                   # 768 per Lance default; smaller for MVP
        width: int = 256,
        num_steps: int = 30,
        timestep_shift: float = 3.5,
        cfg_scale: float = 4.0,
        cfg_renorm_type: str = "channel",  # Phase 5m fix: changed from 'global' to 'channel'.
                                          # Resolves Issue #1 — 'global' computes scalar L2 over
                                          # the full velocity tensor; at high n_lat (e.g. 768²×17f,
                                          # n_lat=11520) the L2 cap silently over-suppresses
                                          # high-frequency detail. 'channel' computes per-channel
                                          # L2 so pathological channels clamp without dragging
                                          # the aggregate signal down. Equivalent at small scales
                                          # (768²×13f production-validated). Pass 'global' to
                                          # restore legacy default.
        cfg_renorm_min: float = 0.0,
        cfg_interval: tuple[float, float] | None = None,
        seed: int = 42,
        verbose: bool = False,
        instruction: str = T2V_INSTRUCTION,
        mape_anchor: int | None = None,
        cfg_uncond_mode: str = "empty_prompt",
        spatial_merge_size: int = 1,
        rope_fp32: bool = False,
        attention_fp32: bool = False,
        prompt_format: str = "ours",
        latent_pos_base: int | None = 0,
        optimized: bool = True,
        memory_mode: str | None = None,
        und_mode: str | None = None,
        free_und: bool | None = None,
        tile_vae: bool = True,
        vae_tile_px: int = 256,
        vae_tile_overlap_px: int = 64,
        vae_temporal_tile: int | None = None,
        vae_temporal_overlap: int = 0,
    ) -> mx.array:
        """`mape_anchor`: temporal-anchor value for latent t-axis positions.
        **Default changed to None on 2026-05-21** after Phase 5d scale bisect
        (github issue #2) showed that no-shift produces photorealistic
        prompt-aligned output at every practical scale (256² to 768²×13f,
        n_lat ≤ 9216) where the old default (2000) produced painterly
        smearing. The shift was a port-side deviation from upstream
        `shift_position_ids` (whose gate never fires for pure t2v).
        Pass `mape_anchor=2000` to restore legacy behavior. At very high
        n_lat (≥ ~12k, e.g. 768²×17f or larger) outputs may degrade —
        coherence threshold is around n_lat=11,520. The 768²×50f oracle
        scale still has a separate second bug under investigation.

        `cfg_interval`: (lo, hi) tuple — CFG fires only when `lo < t <= hi`,
        else falls to cfg_scale=1.0 (no CFG) for that step. Upstream Lance
        default per `config_factory.py` is `[0.4, 1.0]`. Pass None to apply
        CFG at every step (legacy MLX port behavior — likely a contributor
        to the painterly aesthetic bug per github issue #2 Candidate 1b).

        `cfg_uncond_mode`: 'empty_prompt' (legacy) feeds the full chat-template
        sequence with `prompt=''` through the LLM for the uncond branch.
        'no_text' matches upstream Lance (per `lance_lance.py:627-630` and
        `uncond_forward`): the uncond branch DROPS all text positions and
        feeds only the latent block through the LLM. Upstream's CFG direction
        is "with text vs no text at all" rather than "with prompt vs empty
        prompt"; the latter under-amplifies fine-detail features. **Candidate 3
        in issue #2.**

        `spatial_merge_size`: divisor for h/w axes in `_build_position_ids`'s
        latent grid. Default `1` (legacy). Set to `2` to match upstream Lance's
        `data/common.py::shift_position_ids` and RockTalk's parallel MLX port
        (both divide visual position-ids by spatial_merge_size=2). **P0b
        candidate from issue #2 / Phase 5g research brief.**

        `rope_fp32`: when True, compute cos/sin and the
        `q*cos + rotate_half(q)*sin` rotation in fp32 across all 36 attention
        layers (mlx-vlm's stock path casts cos/sin to bf16 at
        `qwen2_5_vl/language.py:73` before the rotation). Default False
        (legacy bf16 path). **P0a candidate from issue #2 / Phase 5g.**

        `latent_pos_base`: anchor (origin) for the latent block's (t, h, w)
        mrope grid coords. **Default 0 (Phase 5j fix, 2026-05-21):** latent
        grid always starts at origin regardless of prompt length, matching
        Qwen2.5-VL's training convention where visual tokens use 3D-mrope
        grid origin (not concatenated with text positions). The Phase 5i.2
        bisect proved long verbose prompts trigger watercolor while short
        prompts produce sharp output at the same other config — the trigger
        was prompt-length-dependent drift of latent block position-IDs.
        Pass `None` to restore legacy `base=text_len_before_latents`
        behavior (watercolor on long prompts). Phase 5j A/B at 256²×17f
        on the red-panda-surfing oracle prompt: legacy = watercolor,
        base=0 = PHOTOREAL. The fix that closes the painterly aesthetic gap.

        `memory_mode` ("parallel"|"relay"|"auto"|None): generate-time override of
        the pipeline's load-time memory strategy. None (default) honors the mode
        resolved at `from_pretrained`. "relay" frees UND after prefill, sheds GEN
        before decode, and materializes the VAE only at decode (single-shot,
        ~9-10 GB peak); "parallel" keeps everything resident (reusable). The low
        relay prefill peak needs the model to have been LOADED relay (GEN+VAE lazy);
        against a parallel-loaded model relay falls back to an eager UND-free (same
        output + shed, no prefill-peak win). Output is identical across modes — the
        GEN-only loop is gated on the prefix caches, not the towers. `und_mode`
        (keep/free/deferred) and `free_und` (bool) are DEPRECATED aliases
        (keep→parallel, deferred/free→relay; True→relay, False→parallel).
        Precedence: memory_mode > und_mode > free_und > pipeline's resolved mode.
        """
        if cfg_interval is None:
            # Legacy behavior: CFG at every step. Effectively cfg_interval=[-inf, +inf].
            cfg_lo, cfg_hi = float("-inf"), float("inf")
        else:
            cfg_lo, cfg_hi = float(cfg_interval[0]), float(cfg_interval[1])

        # P0a (issue #2 / Phase 5g) — fp32 RoPE rotation in all 36 attention layers.
        # Default off (legacy bf16 path); set True to test the research-brief
        # candidate that bf16 rotation perturbs flow-matching velocity precision.
        self.lance_model.set_rope_fp32(bool(rope_fp32))
        self.lance_model.set_attention_fp32(bool(attention_fp32))
        """Generate a video as (T_decoded, H, W, 3) uint8-compatible mx.array.

        Caller is responsible for encoding to MP4 (see scripts/10_t2v_demo.py
        for the imageio-ffmpeg path).
        """
        assert height % VAE_SPATIAL_DOWNSAMPLE == 0
        assert width % VAE_SPATIAL_DOWNSAMPLE == 0
        h_lat = height // VAE_SPATIAL_DOWNSAMPLE
        w_lat = width // VAE_SPATIAL_DOWNSAMPLE
        # Wan2.2 VAE temporal compression: T frames → ((T-1)//4 + 1) latent frames.
        # First chunk = 1 frame, remaining chunks of 4. Inverse on decode side.
        t_lat = (num_frames - 1) // VAE_TEMPORAL_DOWNSAMPLE + 1
        n_lat = t_lat * h_lat * w_lat

        assert t_lat <= MAX_LATENT_FRAMES, (
            f"t_lat={t_lat} exceeds MAX_LATENT_FRAMES={MAX_LATENT_FRAMES}"
        )

        if verbose:
            print(f"  video dims: {num_frames}f × {height}×{width}")
            print(f"  latent dims: {t_lat}f × {h_lat}×{w_lat} = {n_lat} tokens")

        # --- Build per-prompt states (cond + uncond) --------------------
        cond_state = self._prepare_state(
            prompt=prompt, instruction=instruction,
            n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=verbose,
            mape_anchor=mape_anchor, uncond_no_text=False,
            spatial_merge_size=spatial_merge_size,
            prompt_format=prompt_format,
            latent_pos_base=latent_pos_base,
        )
        if cfg_scale > 1.0:
            uncond_state = self._prepare_state(
                prompt="", instruction=instruction,
                n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat, verbose=False,
                mape_anchor=mape_anchor,
                uncond_no_text=(cfg_uncond_mode == "no_text"),
                spatial_merge_size=spatial_merge_size,
                prompt_format=prompt_format,
                latent_pos_base=latent_pos_base,
            )
            if verbose:
                print(f"  CFG enabled, scale={cfg_scale}, mode={cfg_uncond_mode}, "
                      f"uncond tokens={uncond_state['T']}, cond tokens={cond_state['T']}")
        else:
            uncond_state = None

        # --- Single-node optimization: prefill the text prefix K/V once per CFG
        # arm, then handle the MoT towers per `memory_mode` BEFORE the denoise loop.
        # The loop runs GEN-only over the latent tokens against the cached prefix
        # in every mode — the fast path is gated on the caches, not on the tower
        # being gone — so the velocity is identical across parallel/relay.
        #   parallel — full-routed prefill, both towers + VAE resident (reusable).
        #   relay    — UND-only prefill (GEN never touched), free UND, then
        #              materialize GEN; towers never co-resident (~10 GB peak),
        #              single-shot. If the model was loaded parallel, relay falls
        #              back to an eager UND-free (full-routed prefill → free UND):
        #              same output + shed, just no prefill-peak win.
        # resolve_memory_mode picks the auto default from the device budget; the
        # deprecated und_mode/free_und kwargs still map onto parallel/relay.
        # Identical structure to t2i; the only video-specific difference is the
        # latent block spans t_lat×h_lat×w_lat tokens instead of h_lat×w_lat (the
        # tower mechanics in LanceModel are modality-agnostic → pure pipeline mirror).
        # Whether to shed the GEN tower before decode (the second half of the
        # shed cascade). Set once the effective mode is known; relay only.
        shed_gen_before_decode = False
        if optimized:
            # Effective memory mode. Precedence: a generate-time memory_mode override
            # > deprecated und_mode/free_und aliases > the pipeline's load-time mode.
            if memory_mode is not None:
                effective = resolve_memory_mode(memory_mode)
            elif und_mode is not None:
                effective = resolve_memory_mode(None, und_mode=und_mode)
            elif free_und is not None:
                effective = resolve_memory_mode(None, free_und=free_und)
            else:
                effective = self.memory_mode  # resolved "parallel"|"relay" at load

            # relay's low prefill peak requires the GEN tower to have been LOADED
            # lazily. Against a parallel-loaded model we cannot un-materialize GEN,
            # so fall back to an eager UND-free: full-routed prefill (which
            # materializes GEN as before) + free_und_tower(). Same output + shed.
            gen_deferred = bool(getattr(self.lance_model, "_gen_deferred", False))
            relay = (effective == "relay")
            relay_deferred = relay and gen_deferred

            # In the deferred relay path the prefill MUST be UND-only — the full-
            # routed mx.where would evaluate the *_moe_gen branch and materialize
            # the deferred GEN tower (defeating the deferral). Bit-identical on an
            # all-TEXT prefix (t2v's prefix is all TEXT for both uncond modes).
            und_only_prefill = relay_deferred
            cond_state["caches"] = self.lance_model.prefill_prefix(
                cond_state["prefix_embeds"],
                position_ids=cond_state["position_ids_prefix"],
                position_group=cond_state["position_group_prefix"],
                mask=cond_state["mask_prefix"],
                und_only=und_only_prefill,
            )
            if uncond_state is not None:
                uncond_state["caches"] = self.lance_model.prefill_prefix(
                    uncond_state["prefix_embeds"],
                    position_ids=uncond_state["position_ids_prefix"],
                    position_group=uncond_state["position_group_prefix"],
                    mask=uncond_state["mask_prefix"],
                    und_only=und_only_prefill,
                )

            if relay_deferred:
                # Release UND FIRST (del → gc → clear_cache), THEN materialize GEN —
                # the two towers are never co-resident, so the peak stays ~10 GB.
                free_stats = self.lance_model.free_und_tower(deferred=True)
                gen_stats = self.lance_model.materialize_gen_tower()
                if verbose:
                    print(f"  [opt] relay (deferred): UND-only prefill → freed UND "
                          f"{free_stats['freed_bytes']/1e9:.2f} GB "
                          f"(active {free_stats['active_after']/1e9:.2f} GB) → "
                          f"materialized GEN {gen_stats['materialized_bytes']/1e9:.2f} GB "
                          f"(active {gen_stats['active_after']/1e9:.2f} GB)")
            elif relay:
                # Model loaded parallel; eager free (GEN already resident).
                stats = self.lance_model.free_und_tower()
                if verbose:
                    print(f"  [opt] relay (eager free; model loaded parallel): freed "
                          f"UND {stats['freed_bytes']/1e9:.2f} GB, active now "
                          f"{stats['active_after']/1e9:.2f} GB")
            elif verbose:  # parallel
                print(f"  [opt] prefilled prefix; towers kept resident "
                      f"(memory_mode='parallel')")

            # relay already deleted UND (single-shot), so shedding GEN before decode
            # adds no new reuse constraint. parallel preserves the full model for
            # reuse → no shed.
            shed_gen_before_decode = relay

        # latent_pos_embed indices: flat into (max_frames × max_side × max_side).
        # For grid cell (frame, row, col): idx = frame*64² + row*64 + col.
        lpe_indices = mx.array(
            [
                f * (MAX_LATENT_SIDE ** 2) + r * MAX_LATENT_SIDE + c
                for f in range(t_lat)
                for r in range(h_lat)
                for c in range(w_lat)
            ],
            dtype=mx.int32,
        )

        # --- Init noise -------------------------------------------------
        mx.random.seed(seed)
        latents = mx.random.normal((1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS))
        latents_dtype = self.lance_model.embed_tokens.weight.dtype
        latents = latents.astype(latents_dtype)

        # --- Flow loop ---------------------------------------------------
        sched = timestep_schedule(num_steps=num_steps, shift=timestep_shift)
        if verbose:
            print(f"  schedule: {[round(float(sched[i]), 4) for i in range(min(6, num_steps+1))]} ...")

        for step in range(num_steps):
            t = sched[step]
            dt = sched[step] - sched[step + 1]
            # Per upstream Lance: CFG fires only inside cfg_interval; outside, scale collapses to 1.0.
            t_scalar = float(t.item()) if hasattr(t, "item") else float(t)
            cfg_active = (t_scalar > cfg_lo) and (t_scalar <= cfg_hi)
            cfg_scale_step = cfg_scale if cfg_active else 1.0

            v_cond = self._step_velocity(
                state=cond_state, latents=latents, t=t,
                lpe_indices=lpe_indices,
                n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
            )
            if uncond_state is not None and cfg_scale_step > 1.0:
                v_uncond = self._step_velocity(
                    state=uncond_state, latents=latents, t=t,
                    lpe_indices=lpe_indices,
                    n_lat=n_lat, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
                )
                v_cfg = v_uncond + cfg_scale_step * (v_cond - v_uncond)

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

            latents = latents - velocity * dt
            mx.eval(latents)

            if verbose:
                lat_np = latents.astype(mx.float32)
                print(f"  step {step+1}/{num_steps} t={float(t):.4f} dt={float(dt):.4f}  "
                      f"mean={float(mx.mean(lat_np)):.3f}  std={float(mx.std(lat_np)):.3f}")

        # --- Shed the GEN tower before decode ---------------------------
        # The VAE decoder never touches the MoT backbone, so holding GEN resident
        # through decode only inflates the peak. In relay mode the final latents
        # are already materialized → drop the dead prefix caches and free the GEN
        # tower. Decode peak then ≈ VAE-only + transient (~5-6 GB), flat in frame
        # count, instead of GEN(~5.5 GB) + transient (~11.6 GB).
        if shed_gen_before_decode:
            cond_state["caches"] = None
            if uncond_state is not None:
                uncond_state["caches"] = None
            shed_stats = self.lance_model.free_gen_tower()
            if verbose:
                print(f"  [opt] shed GEN tower before decode: freed "
                      f"{shed_stats['freed_bytes']/1e9:.2f} GB, active now "
                      f"{shed_stats['active_after']/1e9:.2f} GB")

        # --- VAE decode -------------------------------------------------
        if verbose:
            print(f"  [pre-decode] peak so far={mx.get_peak_memory()/1e9:.2f} GB "
                  f"active={mx.get_active_memory()/1e9:.2f} GB")
            mx.reset_peak_memory()   # isolate the decode-only peak for reporting
            print(f"  VAE decode ...")
        # relay: load the VAE decoder now (it was left lazy at from_pretrained so it
        # never inflated the prefill/loop peak). No-op in parallel (eager at load).
        vae_stats = self._materialize_vae()
        if verbose and vae_stats is not None:
            print(f"  [opt] materialized VAE decoder: "
                  f"{vae_stats['materialized_bytes']/1e9:.2f} GB "
                  f"(active {vae_stats['active_after']/1e9:.2f} GB)")
        z = denormalize_latents(latents).astype(self.vae_decoder.conv2.weight.dtype)
        if tile_vae:
            # Spatial tiling bounds the decoder's peak activation, exactly as in
            # t2i: at 768² the whole decode peaks high, but 256px tiles (16
            # latent, 64px=4-latent overlap, trapezoidal blend) hold the decode
            # transient to a few GB. decode_tiled self-falls-back to a whole
            # decode when small enough (at 256²: h_lat=16 ≤ 256//16, no tiling),
            # so the 256² win is free_und-driven and tiling only bites at 768².
            # Temporal tiling (vae_temporal_tile) only engages for long videos
            # (t_lat > tile//4); at t_lat=4 (≤13f) it can't and needn't fire.
            from mlx_video.models.wan_2.tiling import (
                SpatialTilingConfig,
                TemporalTilingConfig,
                TilingConfig,
            )
            spatial_cfg = SpatialTilingConfig(
                tile_size_in_pixels=vae_tile_px,
                tile_overlap_in_pixels=vae_tile_overlap_px,
            )
            temporal_cfg = None
            if vae_temporal_tile is not None:
                temporal_cfg = TemporalTilingConfig(
                    tile_size_in_frames=vae_temporal_tile,
                    tile_overlap_in_frames=vae_temporal_overlap,
                )
            tcfg = TilingConfig(
                spatial_config=spatial_cfg, temporal_config=temporal_cfg)
            decoded = self.vae_decoder.decode_tiled(z, tiling_config=tcfg)
        else:
            decoded = self.vae_decoder(z)         # (1, T', H', W', 3) in [-1, 1]
        mx.eval(decoded)
        if verbose:
            print(f"  [decode] peak={mx.get_peak_memory()/1e9:.2f} GB "
                  f"active={mx.get_active_memory()/1e9:.2f} GB "
                  f"(tile_vae={tile_vae}, tile_px={vae_tile_px})")

        # Convert to uint8 frames (T', H', W', 3). VAE may add extra leading
        # padding frames; we keep them all so the caller can decide.
        import numpy as np
        frames_t = decoded[0]                     # (T', H', W', 3)
        frames_np = np.array(frames_t.astype(mx.float32))
        frames_u8 = ((frames_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return frames_u8

    # ------ per-prompt state assembly ----------------------------------------

    def _prepare_state(
        self,
        *,
        prompt: str,
        instruction: str,
        n_lat: int,
        t_lat: int,
        h_lat: int,
        w_lat: int,
        verbose: bool,
        mape_anchor: int | None = MAPE_ANCHOR_VIDEO_GEN,
        uncond_no_text: bool = False,
        spatial_merge_size: int = 1,
        prompt_format: str = "ours",
        latent_pos_base: int | None = None,
    ) -> dict:
        """Pack the prompt-dependent state needed for one CFG-arm of the flow.

        `uncond_no_text=True` builds a text-stripped sequence containing only
        the latent block (per upstream Lance's `uncond_split_pro_new`,
        `lance_lance.py:755+`, which selects positions where
        `i_sample_modality != 0`, i.e. non-text positions only). Used for the
        CFG-uncond arm. The CFG direction becomes `(v_text - v_no_text)` rather
        than `(v_text - v_empty_prompt)` which under-amplifies fine-detail
        features.
        """
        video_pad_str = "<|video_pad|>" * n_lat
        if uncond_no_text:
            # Minimal sequence — just the latent block, no chat template.
            # Wrap in vision_start/vision_end so the latent block is still
            # delimited (these are also non-text modality positions upstream).
            text = f"<|vision_start|>{video_pad_str}<|vision_end|>"
        elif prompt_format == "rocktalk":
            # Phase 5h: RockTalk's minimal chat template per their HF card:
            # `<|im_start|> [prompt tokens] <|im_end|> <|vision_start|>
            #  [N latent placeholders] <|vision_end|>`
            # No system/user/assistant role tags, no T2V_INSTRUCTION prefix.
            # This is THEIR working pipeline's template — the minimal wrap
            # may be critical for not over-shifting latent position-IDs into
            # out-of-distribution territory relative to training.
            text = (
                f"<|im_start|>{prompt}<|im_end|>"
                f"<|vision_start|>{video_pad_str}<|vision_end|>"
            )
        else:
            # 'ours' (legacy): full chat template with system + user +
            # assistant role tags and the T2V_INSTRUCTION prefix.
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

        position_ids = self._build_position_ids(
            T=T, t_lat=t_lat, h_lat=h_lat, w_lat=w_lat,
            text_len_before_latents=text_len_before_latents,
            latent_positions=latent_positions,
            mape_anchor=mape_anchor,
            spatial_merge_size=spatial_merge_size,
            latent_pos_base=latent_pos_base,
        )

        position_group = mx.full((T,), int(PositionGroup.TEXT), dtype=mx.int32)
        position_group = self._scatter_set(
            position_group, latent_positions_arr, int(PositionGroup.NOISY_VAE)
        )

        text_embeds = self.lance_model.embed_tokens(input_ids)
        mask = self._build_block_mask(T, latent_positions, dtype=text_embeds.dtype)

        # --- Prefix/latent slices for the single-node optimization ----------
        # Sequence layout is [text prefix (P)] [latent block (n_lat)] [tail
        # <|vision_end|>]. The prefix-KV path requires the latent block to be
        # contiguous so it equals position_ids[:, :, P:lat_end]; assert it.
        # (Holds for both uncond modes: 'empty_prompt' has a full chat-template
        # prefix; 'no_text' has just <|vision_start|> as the 1-token prefix.)
        P = first_latent_pos                       # text-prefix length
        lat_end = latent_positions[-1] + 1
        assert latent_positions == list(range(P, lat_end)), (
            "prefix-KV optimization requires a contiguous latent block; "
            f"got {len(latent_positions)} positions spanning {P}..{lat_end - 1}"
        )

        return {
            "T": T,
            "input_ids": input_ids,
            "text_embeds": text_embeds,
            "latent_positions_arr": latent_positions_arr,
            "position_ids": position_ids,
            "position_group": position_group,
            "mask": mask,
            # optimization slices (text prefix only / latent block only):
            "P": P,
            "prefix_embeds": text_embeds[:, :P, :],
            "position_ids_prefix": position_ids[:, :, :P],
            "position_group_prefix": position_group[:P],
            "mask_prefix": mask[:P, :P],
            "position_ids_lat": position_ids[:, :, P:lat_end],
            "caches": None,   # filled by generate() via prefill_prefix
        }

    def _step_velocity(
        self,
        *,
        state: dict,
        latents: mx.array,
        t: mx.array,
        lpe_indices: mx.array,
        n_lat: int,
        t_lat: int,
        h_lat: int,
        w_lat: int,
    ) -> mx.array:
        """One forward pass; returns velocity reshaped to (1, t_lat, h_lat, w_lat, C)."""
        latents_flat = latents.reshape(1, n_lat, VAE_LATENT_CHANNELS)
        pe = self.lance_model.latent_pos_embed(lpe_indices)[None, ...]
        t_emb = self.lance_model.time_embedder(t.reshape(1)).reshape(1, 1, -1)
        lat_embed = self.lance_model.vae_in_proj(latents_flat) + pe + t_emb

        # Optimized path: feed ONLY the latent tokens through the GEN-only stack,
        # attending to the per-arm cached text prefix. No full-sequence assembly,
        # no scatter, no UND weights, no dense (T,T) mask. Equivalent to the
        # baseline (text-prefix K/V are step-invariant; the dropped trailing
        # <|vision_end|> token is causally after the latents and never feeds
        # them). Cast lat_embed (fp32 — the timestep embed promotes it) down to
        # the model dtype to match the baseline's _scatter_embeds precision.
        if state.get("caches") is not None:
            h_lat_pos = self.lance_model.gen_loop_forward(
                lat_embed.astype(state["text_embeds"].dtype),
                position_ids=state["position_ids_lat"],
                caches=state["caches"],
            )
            velocity_flat = self.lance_model.llm2vae(h_lat_pos)
            return velocity_flat.reshape(1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS)

        inputs_embeds = self._scatter_embeds(
            state["text_embeds"], lat_embed, state["latent_positions_arr"],
        )

        h = self.lance_model(
            inputs_embeds=inputs_embeds,
            position_ids=state["position_ids"],
            position_group=state["position_group"],
            mask=state["mask"],
        )
        h_lat_pos = h[:, state["latent_positions_arr"], :]
        velocity_flat = self.lance_model.llm2vae(h_lat_pos)
        return velocity_flat.reshape(1, t_lat, h_lat, w_lat, VAE_LATENT_CHANNELS)

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _scatter_set(arr: mx.array, idx: mx.array, value: int) -> mx.array:
        import numpy as np
        out_np = np.array(arr)
        out_np[np.array(idx)] = value
        return mx.array(out_np)

    @staticmethod
    def _scatter_embeds(
        base: mx.array,
        inserts: mx.array,
        positions: mx.array,
    ) -> mx.array:
        import numpy as np
        target_dtype = base.dtype
        out_np = np.array(base.astype(mx.float32))
        ins_np = np.array(inserts.astype(mx.float32))
        pos_np = np.array(positions)
        out_np[:, pos_np, :] = ins_np
        return mx.array(out_np).astype(target_dtype)

    @staticmethod
    def _build_block_mask(T: int, latent_positions: list[int], dtype) -> mx.array:
        """Causal OR bidirectional-within-latent-block additive mask (T, T).
        Same pattern as t2i — the noisy-VAE positions need full mutual
        attention to denoise coherently."""
        i = mx.arange(T)[:, None]
        j = mx.arange(T)[None, :]
        lat_start = latent_positions[0]
        lat_end = latent_positions[-1] + 1
        in_lat_q = (i >= lat_start) & (i < lat_end)
        in_lat_kv = (j >= lat_start) & (j < lat_end)
        bidirectional = in_lat_q & in_lat_kv
        allowed = (i >= j) | bidirectional
        neg_inf = mx.array(-1e9, dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        return mx.where(allowed, zero, neg_inf)

    def _build_position_ids(
        self,
        *,
        T: int,
        t_lat: int,
        h_lat: int,
        w_lat: int,
        text_len_before_latents: int,
        latent_positions: list[int],
        mape_anchor: int | None = MAPE_ANCHOR_VIDEO_GEN,
        spatial_merge_size: int = 1,
        latent_pos_base: int | None = None,
    ) -> mx.array:
        """Build (3, 1, T) position_ids with 3D grid for latent positions.

        Layout: latent token i (in flat row-major (t, h, w) order) gets:
          - t-axis: base + frame_idx     (BEFORE MaPE shift)
          - h-axis: base + (row_idx // spatial_merge_size)
          - w-axis: base + (col_idx // spatial_merge_size)
        Then MaPE re-anchors the t-axis of latent positions:
          - shift = 2000 - first_latent_t_axis_position  (modality 3 = video_gen)
          - applied uniformly to all latent positions

        `spatial_merge_size`: divisor for h/w axes (P0b candidate from issue #2).
        Default `1` = no merging (legacy). Upstream Qwen2.5-VL convention is
        `sms=2` (see `data/common.py::shift_position_ids`). RockTalk's parallel
        MLX port also uses `sms=2`. Setting to 2 halves the spatial position-id
        spread, which matches the trained mrope convention for visual tokens
        and may close residual fine-detail gap on water/textures.

        `latent_pos_base`: anchor (origin) for the latent block's (t, h, w)
        grid coords. **None (default, legacy):** `base = text_len_before_latents`
        — the latent grid starts where text ends, so latent coords drift with
        prompt length. **0 (Phase 5i.2 hypothesis):** `base = 0` — latent grid
        always starts at origin regardless of prompt length, matching
        Qwen2.5-VL's training convention where visual tokens use 3D-mrope
        grid origin (not concatenated with text positions). The Phase 5i.2
        bisect showed long prompts trigger watercolor while short prompts
        produce sharp output at the same other config — strong signal that
        prompt-length-dependent position-ID drift is the bug.
        """
        import numpy as np
        pos = np.zeros((3, 1, T), dtype=np.int32)
        seq = np.arange(T, dtype=np.int32)
        pos[0, 0, :] = seq
        pos[1, 0, :] = seq
        pos[2, 0, :] = seq

        sms = max(1, int(spatial_merge_size))
        base = text_len_before_latents if latent_pos_base is None else int(latent_pos_base)
        for idx, token_pos in enumerate(latent_positions):
            f = idx // (h_lat * w_lat)
            r = (idx % (h_lat * w_lat)) // w_lat
            c = (idx % (h_lat * w_lat)) % w_lat
            pos[0, 0, token_pos] = base + f
            pos[1, 0, token_pos] = base + (r // sms)
            pos[2, 0, token_pos] = base + (c // sms)

        # Tokens after the latent block (vision_end) continue from the max.
        max_grid = max(t_lat - 1, (h_lat - 1) // sms, (w_lat - 1) // sms)
        after_latents_start = latent_positions[-1] + 1
        if after_latents_start < T:
            tail_len = T - after_latents_start
            tail = base + max_grid + 1 + np.arange(tail_len, dtype=np.int32)
            pos[:, 0, after_latents_start:] = tail[None, :]

        # MaPE re-anchor: optionally re-anchor the t-axis of latent positions
        # to `mape_anchor`. Pass None to skip — this matches upstream's
        # `shift_position_ids` behavior for pure t2v (its gate
        # `attn_mode in ['full_noise','full']` never fires for `'noise'`-only
        # samples, so upstream does NOT re-anchor t2v positions). Under
        # investigation as Candidate 0 in github issue #2.
        if mape_anchor is not None:
            first_latent_t = pos[0, 0, latent_positions[0]]
            shift = int(mape_anchor) - int(first_latent_t)
            for token_pos in latent_positions:
                pos[0, 0, token_pos] += shift

        return mx.array(pos)
