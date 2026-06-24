"""
FastWAM Action-Only model: ActionDiT without video expert/MoT.

Uses VAE-encoded condition image as visual context for ActionDiT,
bypassing the 5B video generation model entirely.

Architecture:
  - VAE encodes condition image → latent [B, C, 1, H, W]
  - ImageProjector: spatial tokens → [B, N_img_tokens, text_dim]
  - ActionDiT receives: [image_tokens; text_tokens] as context
  - Flow-matching training on action only (no video loss)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class ImageProjector(nn.Module):
    """Project VAE latents into context tokens for ActionDiT.

    Mirrors the video_expert's patch_embedding (Conv3d stride=(1,2,2)) to produce
    the same number of tokens (3×14×14=588), but replaces the 30-layer video
    transformer with a simple MLP projection.

    Input: VAE latent [B, C=48, T_lat=3, H_lat=28, W_lat=28]
           (9 condition frames → VAE temporal compression → 3 latent frames)
    Output: [B, 588, text_dim=4096]
    """

    def __init__(
        self,
        vae_channels: int = 48,
        text_dim: int = 4096,
        patch_size: tuple = (1, 2, 2),  # same as video_expert patch_embedding
    ):
        super().__init__()
        patch_dim = vae_channels * patch_size[0] * patch_size[1] * patch_size[2]
        # Conv3d patchify: same as video_expert.patch_embedding but projects to text_dim
        self.patch_conv = nn.Conv3d(
            vae_channels, text_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # Extra MLP for capacity (since we don't have 30 transformer layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, vae_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vae_latent: [B, 48, T_lat, H_lat, W_lat] (e.g. [B, 48, 3, 28, 28])
        Returns:
            [B, T_lat * (H_lat//2) * (W_lat//2), text_dim]  = [B, 588, 4096]
        """
        # Patchify: [B, 48, 3, 28, 28] → [B, 4096, 3, 14, 14]
        x = self.patch_conv(vae_latent)
        B, C, T, H, W = x.shape
        # Reshape to tokens: [B, T*H*W, C]
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        x = self.proj(x)
        return x


class FastWAMActionOnly(nn.Module):
    """Action-only model: ActionDiT + VAE image encoder, no video expert."""

    def __init__(
        self,
        action_expert: ActionDiT,
        vae,
        image_projector: ImageProjector,
        text_dim: int = 4096,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.action_expert = action_expert
        self.vae = vae
        self.image_projector = image_projector
        self.text_dim = text_dim

        # For trainer compatibility: optimizer uses model.dit
        self.dit = action_expert

        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_config(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        redirect_common_files: bool = True,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_action: float = 1.0,
        n_image_tokens: int = 64,
        # Unused but accepted for config compatibility
        **kwargs,
    ):
        """Create action-only model. Only loads VAE + ActionDiT (no video expert)."""

        # Load only VAE — pass a minimal dummy dit_config to satisfy the loader,
        # but skip pretrained weights (DiT won't be used anyway)
        _dummy_dit_config = {
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48,
            "hidden_dim": 3072,
            "ffn_dim": 14336,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 48,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "eps": 1e-6,
            "seperated_timestep": True,
            "require_clip_embedding": False,
            "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": False,
            "video_attention_mask_mode": "first_frame_causal",
            "action_conditioned": False,
            "action_dim": 3,
            "action_group_causal_mask_mode": "group_diagonal",
        }
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=_dummy_dit_config,
            skip_dit_load_from_pretrain=True,  # Don't load 5B video weights
            load_text_encoder=False,
        )
        vae = components.vae
        # Free the dummy DiT to save memory
        del components.dit

        # ActionDiT
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        text_dim = int(action_dit_config["text_dim"])

        # Image projector (VAE latent → context tokens, same patchify as video expert)
        image_projector = ImageProjector(
            vae_channels=48,
            text_dim=text_dim,
            patch_size=(1, 2, 2),  # same as video_expert.patch_embedding
        ).to(device=device, dtype=torch_dtype)

        model = cls(
            action_expert=action_expert,
            vae=vae,
            image_projector=image_projector,
            text_dim=text_dim,
            device=device,
            torch_dtype=torch_dtype,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_action=loss_lambda_action,
        )

        logger.info(
            f"FastWAMActionOnly: ActionDiT params={sum(p.numel() for p in action_expert.parameters())/1e6:.0f}M, "
            f"ImageProjector params={sum(p.numel() for p in image_projector.parameters())/1e6:.1f}M, "
            f"n_image_tokens=588 (3×14×14, same as video expert)"
        )
        return model

    @torch.no_grad()
    def _encode_condition_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Encode all 9 condition frames using VAE.

        Same as the full model: 9 RGB frames → VAE temporal compression → 3 latent frames.

        Args:
            video: [B, C=3, T=17, H, W] in [-1, 1] (9 cond + 8 future)
        Returns:
            vae_latent: [B, 48, 3, 28, 28] (3 temporal latent frames)
        """
        # Take condition frames (first 9): [B, 3, 9, H, W]
        n_cond_frames = 9
        cond_video = video[:, :, :n_cond_frames]
        z = self.vae.encode(
            cond_video.to(device=self.device, dtype=self.torch_dtype),
            device=self.device,
        )
        if isinstance(z, list):
            z = z[0]
        if z.ndim == 4:
            z = z.unsqueeze(0)
        # z shape: [B, 48, T_lat, H_lat, W_lat] where T_lat = (9+3)//4 = 3
        return z

    def _build_context(
        self,
        video: torch.Tensor,
        text_context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build combined image+text context for ActionDiT.

        Returns:
            context: [B, N_img + N_text, text_dim]
            context_mask: [B, N_img + N_text]
        """
        # Encode condition frames (all 9 → 3 latent frames)
        vae_latent = self._encode_condition_frames(video)
        img_tokens = self.image_projector(vae_latent)  # [B, 588, text_dim]

        # Concat: [image_tokens, text_tokens]
        context = torch.cat([img_tokens, text_context], dim=1)

        # Build mask
        B = img_tokens.shape[0]
        img_mask = torch.ones(B, img_tokens.shape[1], dtype=torch.bool, device=context.device)
        if context_mask is None:
            text_mask = torch.ones(B, text_context.shape[1], dtype=torch.bool, device=context.device)
        else:
            text_mask = context_mask
        full_mask = torch.cat([img_mask, text_mask], dim=1)

        return context, full_mask

    def training_loss(self, sample) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute action-only training loss (no video loss)."""
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype)
        context = sample["context"].to(device=self.device, dtype=self.torch_dtype)
        context_mask = sample.get("context_mask")
        if context_mask is not None:
            context_mask = context_mask.to(device=self.device)

        batch_size = action.shape[0]

        # Build context with image tokens
        full_context, full_mask = self._build_context(video, context, context_mask)

        # Flow-matching: add noise to action
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # ActionDiT forward (standalone, no MoT)
        pred_dict = self.action_expert(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=full_context,
            context_mask=full_mask,
        )
        pred_action = pred_dict["action"]

        # Action loss with d_theta weighting
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)  # [B, T]
        action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_action * loss_action

        loss_dict = {
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_video": 0.0,
        }

        # Waypoint metrics (diagnostic)
        with torch.no_grad():
            _num_t = float(self.train_action_scheduler.num_train_timesteps)
            sigma = (timestep_action.float() / _num_t).to(pred_action.device)
            sigma_b = sigma.view(-1, *([1] * (noisy_action.ndim - 1)))
            pred_wp = noisy_action.float() - sigma_b * pred_action.float()
            gt_wp = action.float()
            err = pred_wp - gt_wp
            xy_l2 = err[..., 0:2].pow(2).sum(-1).sqrt()
            _DENORM = 4.0
            loss_dict["wp_xy_mae_m"] = float(xy_l2.mean().item()) / _DENORM
            loss_dict["wp_xy_first_mae_m"] = float(xy_l2[:, 0].mean().item()) / _DENORM
            loss_dict["wp_xy_endpoint_mae_m"] = float(xy_l2[:, -1].mean().item()) / _DENORM
            loss_dict["wp_dtheta_mae_rad"] = float(err[..., 2].abs().mean().item())
            loss_dict["wp_moving_flag_mae"] = float(err[..., 3].abs().mean().item())
            loss_dict["wp_xy_mae_smallsig_m"] = float('nan')

        return loss_total, loss_dict

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save trainable weights (action_expert + image_projector)."""
        payload = {
            "action_expert": self.action_expert.state_dict(),
            "image_projector": self.image_projector.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)
        logger.info(f"Saved action-only checkpoint to {path} (step={step})")

    def load_checkpoint(self, path, optimizer=None):
        """Load trainable weights (action_expert + image_projector)."""
        payload = torch.load(path, map_location=self.device)
        if "action_expert" in payload:
            self.action_expert.load_state_dict(payload["action_expert"], strict=False)
            logger.info("Loaded `action_expert` weights from checkpoint.")
        elif "mot" in payload:
            # Legacy: full-model checkpoint — try to load action expert from mot
            logger.warning("Loading legacy full-model checkpoint; extracting action_expert keys.")
            action_keys = {k.replace("action_expert.", ""): v
                          for k, v in payload["mot"].items()
                          if k.startswith("action_expert.")}
            if action_keys:
                self.action_expert.load_state_dict(action_keys, strict=False)
        if "image_projector" in payload:
            self.image_projector.load_state_dict(payload["image_projector"], strict=True)
            logger.info("Loaded `image_projector` weights from checkpoint.")
        else:
            logger.warning("Checkpoint has no `image_projector` weights; keeping random init.")
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @torch.no_grad()
    def infer(
        self,
        input_image: torch.Tensor,
        num_frames: int = 17,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        action_horizon: Optional[int] = None,
        action_dim: int = 4,
        # Accept and ignore full-model kwargs for compatibility
        prompt: Optional[str] = None,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Inference interface compatible with trainer.evaluate().

        Returns {"video": list_of_pil_frames, "action": tensor [1, T, D]}.
        Video is a dummy (black frames) since this model doesn't generate video.
        """
        from PIL import Image

        self.eval()
        B = 1  # eval is always batch_size=1

        # Move inputs to model device
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)

        # Build a fake "video" tensor from input_image for _encode_condition_frames
        # input_image: [1, 3, H, W] — repeat to 9 condition frames
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        # Create [1, 3, 17, H, W] with the input image repeated for cond frames
        H, W = input_image.shape[-2:]
        cond_frames = input_image.unsqueeze(2).expand(-1, -1, 9, -1, -1)  # [1, 3, 9, H, W]
        # Pad to 17 frames for _encode_condition_frames (it only uses first 9)
        pad_frames = torch.zeros(1, 3, 8, H, W, device=self.device, dtype=self.torch_dtype)
        video_tensor = torch.cat([cond_frames, pad_frames], dim=2)  # [1, 3, 17, H, W]

        # Context
        if context is not None:
            context = context.to(device=self.device, dtype=self.torch_dtype)
            if context.ndim == 2:
                context = context.unsqueeze(0)  # [1, seq_len, dim]
            if context_mask is not None:
                context_mask = context_mask.to(device=self.device)
                if context_mask.ndim == 1:
                    context_mask = context_mask.unsqueeze(0)
        else:
            # Dummy context (zeros)
            context = torch.zeros(1, 256, self.text_dim, device=self.device, dtype=self.torch_dtype)
            context_mask = torch.ones(1, 256, dtype=torch.bool, device=self.device)

        # Infer action
        _action_horizon = action_horizon if action_horizon is not None else 8
        pred_action = self.infer_action(
            video=video_tensor,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            action_dim=action_dim,
            action_horizon=_action_horizon,
        )  # [1, T, D]

        # Dummy video output (black frames) for compatibility with trainer eval
        dummy_frames = [Image.new("RGB", (W, H), (0, 0, 0)) for _ in range(num_frames)]

        return {
            "video": dummy_frames,
            "action": pred_action,  # [1, T, D]
        }

    @torch.no_grad()
    def infer_action(
        self,
        video: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        action_dim: int = 4,
        action_horizon: int = 8,
    ) -> torch.Tensor:
        """Run multi-step denoising to predict action."""
        self.eval()
        B = video.shape[0]

        # Build context
        full_context, full_mask = self._build_context(video, context, context_mask)

        # Start from pure noise
        action = torch.randn(B, action_horizon, action_dim, device=self.device, dtype=self.torch_dtype)

        # Build inference schedule
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=self.torch_dtype,
        )

        for t, delta in zip(timesteps, deltas):
            t_batch = t.unsqueeze(0).expand(B)
            pred_dict = self.action_expert(
                action_tokens=action,
                timestep=t_batch,
                context=full_context,
                context_mask=full_mask,
            )
            velocity = pred_dict["action"]
            action = self.infer_action_scheduler.step(velocity, delta, action)

        return action
