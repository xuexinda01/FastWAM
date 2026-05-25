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
    """Project VAE latent of a single frame into context tokens for ActionDiT.

    VAE latent shape: [B, C=16, 1, H_lat, W_lat] (e.g. 16×28×28 for 224px)
    Output: [B, n_tokens, text_dim]
    """

    def __init__(
        self,
        vae_channels: int = 16,
        latent_h: int = 28,
        latent_w: int = 28,
        text_dim: int = 4096,
        n_tokens: int = 64,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        # Spatial pooling: [B, C, H, W] → [B, C, h, w] where h*w = n_tokens
        pool_h = 8
        pool_w = n_tokens // pool_h  # 64/8=8
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        # Project: [B, n_tokens, C] → [B, n_tokens, text_dim]
        self.proj = nn.Sequential(
            nn.Linear(vae_channels, text_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(text_dim, text_dim),
        )

    def forward(self, vae_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vae_latent: [B, C, 1, H, W] or [B, C, H, W]
        Returns:
            [B, n_tokens, text_dim]
        """
        if vae_latent.ndim == 5:
            vae_latent = vae_latent[:, :, 0]  # [B, C, H, W]
        x = self.pool(vae_latent)  # [B, C, pool_h, pool_w]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, n_tokens, C]
        x = self.proj(x)  # [B, n_tokens, text_dim]
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

        # Image projector (VAE latent → context tokens)
        image_projector = ImageProjector(
            vae_channels=16,
            latent_h=28,  # 224/8
            latent_w=28,
            text_dim=text_dim,
            n_tokens=n_image_tokens,
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
            f"n_image_tokens={n_image_tokens}"
        )
        return model

    @torch.no_grad()
    def _encode_condition_image(self, video: torch.Tensor) -> torch.Tensor:
        """Encode the first frame of video using VAE.

        Args:
            video: [B, C=3, T, H, W] in [-1, 1]
        Returns:
            vae_latent: [B, 16, 1, H_lat, W_lat]
        """
        # Take first frame: [B, 3, H, W]
        first_frame = video[:, :, 0]
        # VAE expects [B, C, T, H, W]
        frame_5d = first_frame.unsqueeze(2)  # [B, 3, 1, H, W]
        z = self.vae.encode(
            frame_5d.to(device=self.device, dtype=self.torch_dtype),
            device=self.device,
        )
        if isinstance(z, list):
            z = z[0]
        if z.ndim == 4:
            z = z.unsqueeze(0)
        return z  # [B, 16, 1, H_lat, W_lat]

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
        # Encode condition image
        vae_latent = self._encode_condition_image(video)
        img_tokens = self.image_projector(vae_latent)  # [B, N_img, text_dim]

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
