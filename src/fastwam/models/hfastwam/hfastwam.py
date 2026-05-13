"""Hierarchical Fast World-Action Model (H-FastWAM).

Three-expert composition on top of :class:`fastwam.models.wan22.mot.MoT`::

    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Language  │  │   Video    │  │   Action   │
    │  Expert    │  │  Expert    │  │  Expert    │
    │ (random)   │  │ (Wan2.2)   │  │ (Fastwam)  │
    └─────┬──────┘  └──────┬─────┘  └─────┬──────┘
          └─── shared multi-modal self-attention ───┘
                    (structured mask)

Token ordering inside MoT: ``[language ‖ video ‖ action]``.

Unified Vision Encoder
----------------------
All visual input flows through a **single** encoder (DINO or VAE) into
the video expert. The language expert has **no** dedicated image encoder.
Instead, it obtains visual grounding by attending to the video expert's
**first-frame tokens** (clean, t=0) through the shared MoT self-attention.
This eliminates a redundant SigLIP encoder and ensures language conditions
on the same features that drive video generation.

Knowledge Insulation
--------------------
The ``language`` expert is trained only by its own teacher-forced CE loss
(𝓛_lang). To prevent the video / action flow-matching losses from
leaking into the language weights, the language K/V is **detached** when
video or action queries attend to it. The language expert's own Q
attending to its own K/V is *not* detached, so it still trains from
𝓛_lang. This is enabled via
``MoT.forward(..., detach_kv_experts={"language"})``.

Attention mask (row = query, col = key)
---------------------------------------
Within language block (task / subtask tokens):
  - task ↔ task:     bidirectional
  - subtask → task + prev subtask: causal over subtask

Cross-expert:
  - language → video first frame: **allowed** (visual grounding).
  - language → video rest / action: blocked.
  - video → language (all):   allowed (subtask conditioning).
  - video → video:            ``video_expert.build_video_to_video_mask``
  - video → action:           blocked.
  - action → language (all):  allowed.
  - action → video first frame only: allowed (anti-leakage).
  - action → action:          bidirectional.

Weight init
-----------
- Language expert: fully random. Phase ``language_video`` warm-start is
  recommended so the language expert reaches a reasonable subtask
  distribution before adding the action expert.
- Video expert: continues to load from Wan2.2-TI2V-5B + optional
  ``pretrain_checkpoint``.
- Action expert: continues to load from ``action_dit_pretrained_path``
  (unchanged from fastwam).

Training phases::

    language_video  →  full
    (lang + video)     (lang + video + action)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.visual_encoder import BaseVisualEncoder, build_visual_encoder

from .language_expert import CROSS_ENTROPY_IGNORE_INDEX, LanguageExpert

logger = logging.getLogger(__name__)


class HFastWAM(nn.Module):
    """Hierarchical Fast World-Action Model (3-expert MoT)."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        language_expert: LanguageExpert,
        video_expert: nn.Module,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        tokenizer=None,
        text_dim: int = 4096,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        # Schedulers
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        # Loss weights
        loss_lambda_language: float = 1.0,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        # Training phase & gradient policy
        training_phase: str = "full",
        knowledge_insulation: bool = True,
        # Optional DINO/VAE override for video expert
        visual_encoder=None,
    ):
        super().__init__()
        self._validate_expert_shapes(language_expert, video_expert, action_expert)
        self._validate_mot_membership(mot, language_expert, video_expert, action_expert)

        self.language_expert = language_expert
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_dim = int(text_dim)
        self.torch_dtype = torch_dtype

        # Video-expert visual encoder (DINO / V-JEPA2) or VAE fallback
        self.use_visual_encoder = isinstance(visual_encoder, BaseVisualEncoder)
        self.visual_encoder = visual_encoder if self.use_visual_encoder else vae

        # Proprio → video/action context via a learned token
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.dit = self.mot  # trainer/optimizer compat

        # Schedulers
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift,
        )

        self.loss_lambda_language = float(loss_lambda_language)
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self._training_phase = training_phase
        self.knowledge_insulation = bool(knowledge_insulation)

        self.device_str = device
        self.to(device)

        logger.info(
            "HFastWAM: phase=%s, KI=%s, λ_lang=%.2f λ_vid=%.2f λ_act=%.2f, experts=%s, "
            "unified_vision=True (language sees video first-frame via MoT attention)",
            training_phase, self.knowledge_insulation,
            self.loss_lambda_language, self.loss_lambda_video, self.loss_lambda_action,
            list(self.mot.expert_order),
        )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_expert_shapes(lang, video, action):
        # All three experts MUST share attn-space shape or MoT can't concat.
        if int(lang.num_heads) != int(video.num_heads) or int(action.num_heads) != int(video.num_heads):
            raise ValueError(
                f"num_heads mismatch: lang={lang.num_heads}, vid={video.num_heads}, act={action.num_heads}."
            )
        if int(lang.attn_head_dim) != int(video.attn_head_dim) or int(action.attn_head_dim) != int(video.attn_head_dim):
            raise ValueError(
                f"attn_head_dim mismatch: lang={lang.attn_head_dim}, "
                f"vid={video.attn_head_dim}, act={action.attn_head_dim}."
            )
        if int(len(lang.blocks)) != int(len(video.blocks)) or int(len(action.blocks)) != int(len(video.blocks)):
            raise ValueError(
                f"num_layers mismatch: lang={len(lang.blocks)}, "
                f"vid={len(video.blocks)}, act={len(action.blocks)}."
            )

    @staticmethod
    def _validate_mot_membership(mot, lang, video, action):
        if set(mot.expert_order) != {"language", "video", "action"}:
            raise ValueError(
                f"H-FastWAM expects MoT with experts {{language, video, action}}, got {mot.expert_order}."
            )
        if mot.mixtures["language"] is not lang:
            raise ValueError("MoT['language'] must be the same module as language_expert.")
        if mot.mixtures["video"] is not video:
            raise ValueError("MoT['video'] must be the same module as video_expert.")
        if mot.mixtures["action"] is not action:
            raise ValueError("MoT['action'] must be the same module as action_expert.")

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def training_phase(self) -> str:
        return self._training_phase

    @training_phase.setter
    def training_phase(self, phase: str):
        valid = {"language_video", "full"}
        if phase not in valid:
            raise ValueError(f"Invalid training phase: {phase}. Must be one of {valid}")
        self._training_phase = phase
        logger.info("Training phase set to: %s", phase)

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------ #
    # Encoders
    # ------------------------------------------------------------------ #
    def _encode_video_latents(self, video: torch.Tensor, tiled: bool = False):
        """Encode [B, 3, T, H, W] → video-expert latents."""
        if self.use_visual_encoder:
            return self.visual_encoder.encode(video, device=self.device)
        with torch.no_grad():
            return self.vae.encode(video, device=self.device, tiled=tiled)

    @torch.no_grad()
    def _encode_first_frame(self, image: torch.Tensor, tiled: bool = False) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B, 3, H, W], got {tuple(image.shape)}")
        video = image.to(device=self.device, dtype=self.torch_dtype).unsqueeze(2)
        if self.use_visual_encoder:
            return self.visual_encoder.encode(video, device=self.device)
        return self.vae.encode(video, device=self.device, tiled=tiled)

    # ------------------------------------------------------------------ #
    # Dummy cross-attention context for video/action pre_dit
    # ------------------------------------------------------------------ #
    def _make_dummy_text_context(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single zero text token + empty mask.

        ``video_expert.pre_dit`` and ``action_expert.pre_dit`` both require
        ``context``/``context_mask`` arguments because Wan2.2 originally
        used a text encoder. Under the 3-expert MoT design, language
        conditioning happens via shared **self-attention** inside MoT, not
        via cross-attention. We pass a 1-token dummy here and then feed
        ``context_all["video"]=None`` / ``context_all["action"]=None``
        into :meth:`MoT.forward` so that
        :func:`MoT._apply_expert_post_block` skips cross-attention.
        """
        dummy_ctx = torch.zeros(
            (batch_size, 1, self.text_dim),
            dtype=self.torch_dtype, device=self.device,
        )
        dummy_mask = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=self.device,
        )
        return dummy_ctx, dummy_mask

    # ------------------------------------------------------------------ #
    # Structured attention mask
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_full_attention_mask(
        self,
        task_len: int,
        subtask_len: int,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the full ``[S_lang + S_vid + S_act]²`` mask.

        See module docstring for the rule set.
        """
        S_lang = task_len + subtask_len
        S_vid = int(video_seq_len)
        S_act = int(action_seq_len)
        S = S_lang + S_vid + S_act
        mask = torch.zeros((S, S), dtype=torch.bool, device=device)

        # Block ranges
        l_s, l_e = 0, S_lang
        v_s, v_e = S_lang, S_lang + S_vid
        a_s, a_e = S_lang + S_vid, S

        # ---- Language rows -------------------------------------------- #
        # Language self-attention (task + subtask)
        mask[l_s:l_e, l_s:l_e] = LanguageExpert.build_language_rows(
            task_len=task_len, subtask_len=subtask_len, device=device,
        )
        # Language → video FIRST FRAME only (visual grounding)
        first_frame_tokens = min(int(video_tokens_per_frame), S_vid)
        mask[l_s:l_e, v_s:v_s + first_frame_tokens] = True
        # Language → action: blocked (default False).

        # ---- Video rows ----------------------------------------------- #
        # video → language (all) — subtask is the condition
        mask[v_s:v_e, l_s:l_e] = True
        # video → video (per video_expert rule)
        mask[v_s:v_e, v_s:v_e] = self.video_expert.build_video_to_video_mask(
            video_seq_len=S_vid,
            video_tokens_per_frame=int(video_tokens_per_frame),
            device=device,
        )
        # video → action: blocked (default).

        # ---- Action rows ---------------------------------------------- #
        # action → language (all)
        mask[a_s:a_e, l_s:l_e] = True
        # action → first video frame only (anti-leakage)
        mask[a_s:a_e, v_s:v_s + first_frame_tokens] = True
        # action → action: bidirectional
        mask[a_s:a_e, a_s:a_e] = True

        return mask

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    def _validate_sample(self, sample: dict) -> None:
        for required in ("task_token_ids", "subtask_token_ids"):
            if required not in sample:
                raise ValueError(f"H-FastWAM training needs sample['{required}'].")
        if "video" not in sample:
            raise ValueError("H-FastWAM always needs sample['video'] (unified vision).")
        if self._training_phase == "full" and "action" not in sample:
            raise ValueError("phase='full' needs sample['action'].")

    def training_loss(
        self, sample: dict, tiled: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """Training loss with shared MoT self-attention.

        Phase ``language_video`` trains language + video only (action
        branch is dropped from MoT — we use a 2-expert view for efficiency).
        Phase ``full`` runs all three experts through MoT in one pass.
        """
        self._validate_sample(sample)
        loss_dict: dict[str, float] = {}
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        task_ids = sample["task_token_ids"].to(self.device)
        subtask_ids = sample["subtask_token_ids"].to(self.device)
        B = task_ids.shape[0]

        # ---------- Prepare language pre_dit ---------- #
        lang_pre = self.language_expert.pre_dit(
            task_token_ids=task_ids,
            subtask_token_ids=subtask_ids,
        )
        lang_segments = lang_pre["segments"]
        S_lang = lang_segments["task_len"] + lang_segments["subtask_len"]

        # ---------- Prepare video pre_dit ---------- #
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"sample['video'] must be [B,3,T,H,W], got {tuple(video.shape)}")

        input_latents = self._encode_video_latents(video, tiled=tiled)
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=B, device=self.device, dtype=input_latents.dtype,
        )
        noisy_latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video,
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        if fuse_flag:
            noisy_latents[:, :, 0:1] = input_latents[:, :, 0:1]

        dummy_ctx, dummy_mask = self._make_dummy_text_context(B)
        video_pre = self.video_expert.pre_dit(
            x=noisy_latents,
            timestep=timestep_video,
            context=dummy_ctx,
            context_mask=dummy_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])

        # ---------- Action branch (full phase only) ---------- #
        if self._training_phase == "full":
            action = sample["action"].to(device=self.device, dtype=self.torch_dtype)
            if action.ndim != 3:
                raise ValueError(f"sample['action'] must be [B,T,a_dim], got {tuple(action.shape)}")
            noise_action = torch.randn_like(action)
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=B, device=self.device, dtype=action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(
                action, noise_action, timestep_action,
            )
            target_action = self.train_action_scheduler.training_target(
                action, noise_action, timestep_action,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=dummy_ctx,
                context_mask=dummy_mask,
            )

            tokens_out = self._run_mot_three_experts(
                lang_pre=lang_pre,
                video_pre=video_pre,
                action_pre=action_pre,
                task_len=lang_segments["task_len"],
                subtask_len=lang_segments["subtask_len"],
                video_tokens_per_frame=video_tokens_per_frame,
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
            lang_output = self.language_expert.post_dit(tokens_out["language"], lang_pre)
            loss_language = self.language_expert.language_loss(
                logits=lang_output.logits, subtask_token_ids=subtask_ids,
            )
        else:
            # language_video: run MoT with only {language, video}.
            tokens_out = self._run_mot_two_experts_lv(
                lang_pre=lang_pre,
                video_pre=video_pre,
                task_len=lang_segments["task_len"],
                subtask_len=lang_segments["subtask_len"],
                video_tokens_per_frame=video_tokens_per_frame,
            )
            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            lang_output = self.language_expert.post_dit(tokens_out["language"], lang_pre)
            loss_language = self.language_expert.language_loss(
                logits=lang_output.logits, subtask_token_ids=subtask_ids,
            )
            pred_action = None
            target_action = None
            timestep_action = None

        # ---------- Losses ---------- #
        total_loss = total_loss + self.loss_lambda_language * loss_language
        loss_dict["loss_language"] = self.loss_lambda_language * float(loss_language.detach().item())

        loss_video = self._compute_video_loss(
            pred_video=pred_video,
            target_video=target_video,
            fuse_flag=fuse_flag,
            timestep_video=timestep_video,
        )
        total_loss = total_loss + self.loss_lambda_video * loss_video
        loss_dict["loss_video"] = self.loss_lambda_video * float(loss_video.detach().item())

        if pred_action is not None:
            action_is_pad = sample.get("action_is_pad", None)
            loss_action = self._compute_action_loss(
                pred_action=pred_action,
                target_action=target_action,
                timestep_action=timestep_action,
                action_is_pad=action_is_pad,
            )
            total_loss = total_loss + self.loss_lambda_action * loss_action
            loss_dict["loss_action"] = self.loss_lambda_action * float(loss_action.detach().item())

        return total_loss, loss_dict

    # ------------------------------------------------------------------ #
    # Specialized forward helpers
    # ------------------------------------------------------------------ #
    def _run_mot_three_experts(
        self,
        lang_pre: dict,
        video_pre: dict,
        action_pre: dict,
        task_len: int,
        subtask_len: int,
        video_tokens_per_frame: int,
    ) -> dict:
        video_seq_len = int(video_pre["tokens"].shape[1])
        action_seq_len = int(action_pre["tokens"].shape[1])

        attention_mask = self._build_full_attention_mask(
            task_len=task_len, subtask_len=subtask_len,
            video_seq_len=video_seq_len, action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )

        detach_set = {"language"} if self.knowledge_insulation else None

        return self.mot(
            embeds_all={
                "language": lang_pre["tokens"],
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "language": lang_pre["freqs"],
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "language": None,  # no cross-attn in language expert
                "video": None,     # cross-attn deliberately skipped
                "action": None,
            },
            t_mod_all={
                "language": lang_pre["t_mod"],
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            detach_kv_experts=detach_set,
        )

    def _run_mot_two_experts_lv(
        self,
        lang_pre: dict,
        video_pre: dict,
        task_len: int,
        subtask_len: int,
        video_tokens_per_frame: int,
    ) -> dict:
        """2-expert MoT pass over {language, video} for language_video phase."""
        video_seq_len = int(video_pre["tokens"].shape[1])

        mask_full = self._build_full_attention_mask(
            task_len=task_len, subtask_len=subtask_len,
            video_seq_len=video_seq_len, action_seq_len=0,
            video_tokens_per_frame=video_tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        # Temporarily build a lightweight MoT view on just {language, video}.
        tmp_mot = MoT(
            mixtures={"language": self.language_expert, "video": self.video_expert},
            mot_checkpoint_mixed_attn=self.mot.mot_checkpoint_mixed_attn,
        )
        detach_set = {"language"} if self.knowledge_insulation else None

        return tmp_mot(
            embeds_all={"language": lang_pre["tokens"], "video": video_pre["tokens"]},
            attention_mask=mask_full,
            freqs_all={"language": lang_pre["freqs"], "video": video_pre["freqs"]},
            context_all={"language": None, "video": None},
            t_mod_all={"language": lang_pre["t_mod"], "video": video_pre["t_mod"]},
            detach_kv_experts=detach_set,
        )

    # ------------------------------------------------------------------ #
    # Loss helpers
    # ------------------------------------------------------------------ #
    def _compute_video_loss(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        fuse_flag: bool,
        timestep_video: torch.Tensor,
    ) -> torch.Tensor:
        if fuse_flag:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        per_sample = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none",
        ).mean(dim=(1, 2, 3, 4))
        w = self.train_video_scheduler.training_weight(timestep_video).to(
            device=per_sample.device, dtype=per_sample.dtype,
        )
        return (per_sample * w).mean()

    def _compute_action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(self.device)
            token_loss = F.mse_loss(
                pred_action.float(), target_action.float(), reduction="none",
            ).mean(dim=-1)  # [B, T]
            valid = (~action_is_pad).to(dtype=torch.float32)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            per_sample = (token_loss * valid).sum(dim=1) / valid_sum
        else:
            per_sample = F.mse_loss(
                pred_action.float(), target_action.float(), reduction="none",
            ).mean(dim=(1, 2))
        w = self.train_action_scheduler.training_weight(timestep_action).to(
            device=per_sample.device, dtype=per_sample.dtype,
        )
        return (per_sample * w).mean()

    # ------------------------------------------------------------------ #
    # Inference (action-only)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def infer_action(
        self,
        image: torch.Tensor,
        task_token_ids: torch.Tensor,
        action_horizon: int,
        subtask_token_ids: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_subtask_tokens: int = 64,
    ) -> dict:
        """Inference: language AR-generates subtask, then 1-shot video + action denoising.

        Language generates subtask by attending to video expert's first-frame
        tokens through MoT. During generation, video tokens are **frozen**
        (first frame only, t=0) — they just provide K/V for language.
        """
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError("infer_action requires video_attention_mask_mode='first_frame_causal'.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"image must be [1,3,H,W] or [3,H,W], got {tuple(image.shape)}")

        image = image.to(device=self.device, dtype=self.torch_dtype)
        task_token_ids = task_token_ids.to(self.device)

        # Encode first frame into video latents (for both subtask gen and action)
        first_frame_latents = self._encode_first_frame(image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        dummy_ctx, dummy_mask = self._make_dummy_text_context(1)

        # Video pre_dit for first frame (t=0, used as visual grounding)
        timestep_zero = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype, device=self.device,
        )
        video_pre_ff = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_zero,
            context=dummy_ctx,
            context_mask=dummy_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_tokens_per_frame = int(video_pre_ff["meta"]["tokens_per_frame"])

        # ---- Step 1: language AR generation (or use given subtask) ---- #
        if subtask_token_ids is None:
            subtask_ids_used = self._generate_subtask(
                task_ids=task_token_ids,
                video_pre=video_pre_ff,
                video_tokens_per_frame=video_tokens_per_frame,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                max_new_tokens=max_subtask_tokens,
            )
        else:
            subtask_ids_used = subtask_token_ids.to(self.device)

        # ---- Step 2: Full 3-expert MoT for action denoising ---------- #
        # Re-run language pre_dit with finalised subtask for consistent K/V
        lang_pre = self.language_expert.pre_dit(
            task_token_ids=task_token_ids,
            subtask_token_ids=subtask_ids_used,
        )
        seg = lang_pre["segments"]

        # Noisy action init
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        infer_timesteps, infer_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device, dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=dummy_ctx,
                context_mask=dummy_mask,
            )

            tokens_out = self._run_mot_three_experts(
                lang_pre=lang_pre,
                video_pre=video_pre_ff,
                action_pre=action_pre,
                task_len=seg["task_len"],
                subtask_len=seg["subtask_len"],
                video_tokens_per_frame=video_tokens_per_frame,
            )
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta, latents_action,
            )

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "subtask_tokens": subtask_ids_used[0].detach().cpu(),
        }

    # ------------------------------------------------------------------ #
    # Subtask generation (language + video 2-expert MoT)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _generate_subtask(
        self,
        task_ids: torch.Tensor,
        video_pre: dict,
        video_tokens_per_frame: int,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive subtask decoding via 2-expert MoT (language + video).

        Language expert sees the video expert's first-frame tokens via MoT
        shared attention. Video tokens are **frozen** (just providing K/V).
        """
        B = task_ids.shape[0]
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            lang_pre = self.language_expert.pre_dit(
                task_token_ids=task_ids,
                subtask_token_ids=generated,
            )
            seg = lang_pre["segments"]

            # Build 2-expert attention mask (language + video, no action)
            video_seq_len = int(video_pre["tokens"].shape[1])
            mask = self._build_full_attention_mask(
                task_len=seg["task_len"],
                subtask_len=seg["subtask_len"],
                video_seq_len=video_seq_len,
                action_seq_len=0,
                video_tokens_per_frame=video_tokens_per_frame,
                device=lang_pre["tokens"].device,
            )

            # Run 2-expert MoT (language + video)
            tmp_mot = MoT(
                mixtures={"language": self.language_expert, "video": self.video_expert},
                mot_checkpoint_mixed_attn=self.mot.mot_checkpoint_mixed_attn,
            )
            detach_set = {"language"} if self.knowledge_insulation else None

            tokens_out = tmp_mot(
                embeds_all={"language": lang_pre["tokens"], "video": video_pre["tokens"]},
                attention_mask=mask,
                freqs_all={"language": lang_pre["freqs"], "video": video_pre["freqs"]},
                context_all={"language": None, "video": None},
                t_mod_all={"language": lang_pre["t_mod"], "video": video_pre["t_mod"]},
                detach_kv_experts=detach_set,
            )

            logits_last = self.language_expert.step_logits(
                tokens_after_mot=tokens_out["language"],
                task_len=seg["task_len"],
            )  # [B, 1, vocab]

            if temperature > 0:
                logits_last = logits_last / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                    logits_last[logits_last < v[:, :, [-1]]] = -float("inf")
                probs = F.softmax(logits_last.float(), dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            else:
                next_token = logits_last.argmax(dim=-1)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------ #
    # Checkpoint I/O
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str, optimizer=None, step=None):
        payload = {
            "language_expert": self.language_expert.state_dict(),
            "mot": self.mot.state_dict(),
            "training_phase": self._training_phase,
            "torch_dtype": str(self.torch_dtype),
            "step": step,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.use_visual_encoder:
            payload["visual_encoder"] = self.visual_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str, optimizer=None):
        payload = torch.load(path, map_location=self.device)

        def _log(name, missing, unexpected):
            logger.info(
                "Loaded %s (missing=%d, unexpected=%d).",
                name, len(missing), len(unexpected),
            )

        if "language_expert" in payload:
            _log("language_expert", *self.language_expert.load_state_dict(
                payload["language_expert"], strict=False,
            ))
        if "mot" in payload:
            _log("mot", *self.mot.load_state_dict(payload["mot"], strict=False))
        elif "dit" in payload:
            logger.warning("Legacy ckpt: loading 'dit' into video_expert only.")
            _log("video_expert (legacy dit)", *self.video_expert.load_state_dict(
                payload["dit"], strict=False,
            ))
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if self.use_visual_encoder and "visual_encoder" in payload:
            _log("visual_encoder", *self.visual_encoder.load_state_dict(
                payload["visual_encoder"], strict=False,
            ))
        if "training_phase" in payload:
            self._training_phase = payload["training_phase"]
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained_fastwam(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = False,
        redirect_common_files: bool = True,
        video_dit_config: dict | None = None,
        action_dit_config: dict | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = True,
        skip_video_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        visual_encoder_config: dict | None = None,
        training_phase: str = "full",
        loss_config: dict | None = None,
        video_scheduler: dict | None = None,
        action_scheduler: dict | None = None,
        proprio_dim: int | None = None,
        knowledge_insulation: bool = True,
        fastwam_checkpoint: str | None = None,
        pretrain_checkpoint: str | None = None,
        # Language expert config
        language_vocab_size: int = 32000,
        language_ffn_dim: Optional[int] = None,
        language_max_task_len: int = 128,
        language_max_subtask_len: int = 128,
    ):
        """Build H-FastWAM: 3-expert MoT with a random-init language expert.

        The language expert gets visual grounding through MoT shared attention
        with the video expert's first-frame tokens. No separate image encoder.
        """
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required.")
        loss_config = loss_config or {}
        video_scheduler = video_scheduler or {}
        action_scheduler = action_scheduler or {}
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required.")

        # Visual encoder (DINO/V-JEPA2) for video expert (optional)
        dino_visual_encoder = None
        if visual_encoder_config is not None:
            ve_cfg = dict(visual_encoder_config)
            encoder_type = ve_cfg.pop("encoder_type", "dino")
            dino_visual_encoder = build_visual_encoder(
                encoder_type=encoder_type, torch_dtype=torch_dtype, **ve_cfg,
            ).to(device=device)

        # Wan2.2 components (video expert + VAE + tokenizer)
        components = load_wan22_ti2v_5b_components(
            device=device, torch_dtype=torch_dtype,
            model_id=model_id, tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            skip_video_dit_load_from_pretrain=skip_video_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
            skip_vae_load=(dino_visual_encoder is not None),
        )
        video_expert = components.dit

        # Action expert (fastwam-style init)
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config or {},
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device, torch_dtype=torch_dtype,
        )
        # Strict shape matching with the video expert
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT num_heads must match video expert.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT attn_head_dim must match video expert.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT num_layers must match video expert.")

        # Language expert (randomly initialised — visual grounding via MoT attention)
        lang_hidden = int(video_expert.blocks[0].hidden_dim)
        lang_ffn = int(language_ffn_dim) if language_ffn_dim is not None else int(
            video_expert.blocks[0].ffn_dim
        )
        language_expert = LanguageExpert(
            hidden_dim=lang_hidden,
            num_heads=int(video_expert.num_heads),
            attn_head_dim=int(video_expert.attn_head_dim),
            ffn_dim=lang_ffn,
            num_layers=int(len(video_expert.blocks)),
            vocab_size=int(language_vocab_size),
            max_task_len=int(language_max_task_len),
            max_subtask_len=int(language_max_subtask_len),
            eps=1e-6,
            use_gradient_checkpointing=bool(mot_checkpoint_mixed_attn),
            dtype=torch_dtype,
        ).to(device=device)

        # 3-expert MoT (order matters: language | video | action)
        mot = MoT(
            mixtures={
                "language": language_expert,
                "video": video_expert,
                "action": action_expert,
            },
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            language_expert=language_expert,
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
            video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
            video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
            action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
            action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
            action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
            loss_lambda_language=float(loss_config.get("lambda_language", 1.0)),
            loss_lambda_video=float(loss_config.get("lambda_video", 1.0)),
            loss_lambda_action=float(loss_config.get("lambda_action", 1.0)),
            training_phase=training_phase,
            knowledge_insulation=knowledge_insulation,
            visual_encoder=dino_visual_encoder,
        )

        # Optional: resume video expert from a fastwam pretrain ckpt.
        ckpt_path = fastwam_checkpoint or pretrain_checkpoint
        if ckpt_path is not None:
            logger.info("Loading fastwam pretrain checkpoint: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device)
            if "mot" in ckpt:
                missing, unexpected = model.mot.load_state_dict(ckpt["mot"], strict=False)
                logger.info(
                    "Merged fastwam MoT into 3-expert MoT (missing=%d, unexpected=%d).",
                    len(missing), len(unexpected),
                )
            elif "dit" in ckpt:
                missing, unexpected = model.video_expert.load_state_dict(ckpt["dit"], strict=False)
                logger.info(
                    "Loaded legacy DiT into video_expert (missing=%d, unexpected=%d).",
                    len(missing), len(unexpected),
                )
            if model.use_visual_encoder and "visual_encoder" in ckpt:
                missing, unexpected = model.visual_encoder.load_state_dict(
                    ckpt["visual_encoder"], strict=False,
                )
                logger.info(
                    "Loaded visual_encoder (missing=%d, unexpected=%d).",
                    len(missing), len(unexpected),
                )
            del ckpt

        return model

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
