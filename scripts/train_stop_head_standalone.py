"""
Standalone Stop Head Training Script for FastWAM

This script enables fine-tuning of the ActionDiT stop head while keeping
all other components frozen. The stop head learns to predict when the
navigation agent has reached its goal (stop=1) vs. when it's still moving (stop=0).

Key Features:
- Loads pre-trained ActionDiT with random stop_head initialization
- Freezes all backbone weights (DiT blocks, action_encoder, text_embedding)
- Trains only stop_head weights via binary cross-entropy loss
- Supports gradient accumulation and mixed precision training
- Integrates with Weights & Biases for experiment tracking
- Saves checkpoints with stop_head state dict

Usage:
    python train_stop_head_standalone.py \
        --data_dir /path/to/lerobot/datasets \
        --output_dir ./runs/stop_head_finetune \
        --num_epochs 10 \
        --batch_size 32
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs
import wandb
from tqdm import tqdm

# Add FastWAM src to path
FASTWAM_ROOT = Path(__file__).parent.parent / "src"
if str(FASTWAM_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTWAM_ROOT))

from fastwam.datasets.lerobot.nav_video_dataset import NavVideoDataset
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class StopHeadTrainer:
    """Trainer for ActionDiT stop head fine-tuning."""

    def __init__(
        self,
        action_dit: ActionDiT,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "./runs/stop_head",
        use_wandb: bool = False,
        wandb_project: str = "fastwam-stop-head",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
    ):
        """
        Initialize the stop head trainer.
        
        Args:
            action_dit: Pre-trained ActionDiT with stop_head
            train_dataloader: Training dataset loader
            val_dataloader: Optional validation dataset loader
            learning_rate: Learning rate for stop_head
            num_epochs: Number of training epochs
            warmup_steps: Warmup steps for learning rate schedule
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            output_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            device: Device to train on
            torch_dtype: Torch dtype for training
            seed: Random seed
        """
        self.action_dit = action_dit
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        self.device = device
        self.torch_dtype = torch_dtype

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator
        project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            automatic_checkpoint_naming=False,
            total_limit=3,  # Keep last 3 checkpoints
        )
        kwargs = InitProcessGroupKwargs(timeout=180)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision="bf16",
            project_config=project_config,
            kwargs_handlers=[kwargs],
        )

        # Setup optimizer
        self._setup_optimizer()

        # Setup learning rate scheduler
        self._setup_scheduler()

        # Wandb setup
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=wandb_project,
                name=f"stop_head_{Path(output_dir).name}",
                config={
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": train_dataloader.batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "warmup_steps": warmup_steps,
                    "max_grad_norm": max_grad_norm,
                },
            )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        logger.info(f"Initialized StopHeadTrainer")
        logger.info(f"  Action DIT stop_head trainable: {hasattr(self.action_dit, 'stop_head')}")
        logger.info(f"  Output directory: {self.output_dir}")

    def _setup_optimizer(self):
        """Setup optimizer for stop_head only."""
        if not hasattr(self.action_dit, "stop_head"):
            raise ValueError(
                "action_dit must have stop_head. "
                "Ensure it was created with predict_stop=True"
            )

        # Get stop_head parameters
        stop_head_params = list(self.action_dit.stop_head.parameters())
        if not stop_head_params:
            raise ValueError("stop_head has no parameters to optimize")

        logger.info(f"Stop head parameters to train: {len(stop_head_params)}")
        total_params = sum(p.numel() for p in stop_head_params)
        logger.info(f"Total trainable parameters: {total_params:,}")

        self.optimizer = torch.optim.AdamW(
            stop_head_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        num_training_steps = len(self.train_dataloader) * self.num_epochs
        num_warmup_steps = self.warmup_steps

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            ))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def _freeze_backbone(self):
        """Freeze all backbone components, keep only stop_head trainable."""
        # Freeze everything
        for param in self.action_dit.parameters():
            param.requires_grad = False

        # Unfreeze stop_head
        if hasattr(self.action_dit, "stop_head"):
            for param in self.action_dit.stop_head.parameters():
                param.requires_grad = True

        logger.info("Backbone frozen, stop_head trainable")

    def _compute_stop_loss(
        self,
        stop_logits: torch.Tensor,
        action_is_pad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for stop prediction.
        
        Args:
            stop_logits: Model predictions [B, T, 1] (logits, not probabilities)
            action_is_pad: Ground truth labels [B, T] boolean
                - True: trajectory ended (stop=1)
                - False: trajectory ongoing (stop=0)
        
        Returns:
            BCE loss scalar
        """
        # Remove extra dimension from logits if present
        if stop_logits.ndim == 3:
            stop_logits = stop_logits.squeeze(-1)  # [B, T]

        # Convert boolean to float
        stop_labels = action_is_pad.float()  # [B, T]

        # Binary cross-entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            stop_logits, stop_labels, reduction="mean"
        )
        return loss

    def _encode_text_context(
        self,
        sample: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract or compute text context from sample.
        
        Args:
            sample: Sample dictionary with "context" and "context_mask"
        
        Returns:
            context: [B, L, text_dim]
            context_mask: [B, L]
        """
        context = sample["context"]  # [B, L, 4096]
        context_mask = sample["context_mask"]  # [B, L]

        if context.ndim == 2:
            # Handle single-sample case: [L, 4096] -> [1, L, 4096]
            context = context.unsqueeze(0)
            context_mask = context_mask.unsqueeze(0)

        return context, context_mask

    def _forward_action_dit(
        self,
        action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through action DIT.
        
        Args:
            action: Action tokens [B, T, action_dim]
            context: Text context [B, L, text_dim]
            context_mask: Context attention mask [B, L]
            timestep: Diffusion timestep [B]. If None, samples random timesteps.
        
        Returns:
            Dictionary with "action" [B, T, action_dim] and optionally "stop" [B, T, 1]
        """
        batch_size = action.shape[0]

        # Sample random timesteps for training
        if timestep is None:
            timestep = torch.randint(
                0, 1000, (batch_size,), device=action.device, dtype=torch.long
            )

        # Forward through action DIT
        with torch.no_grad():  # Backbone frozen
            output = self.action_dit(
                action_tokens=action,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
            )

        return output

    def train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            batch: Training batch with action, action_is_pad, context, context_mask
        
        Returns:
            loss: Scalar loss
            metrics: Dictionary with loss metrics
        """
        # Extract batch components
        action = batch["action"].to(self.device)  # [B, 32, 4]
        action_is_pad = batch["action_is_pad"].to(self.device)  # [B, 32]
        context, context_mask = self._encode_text_context(batch)
        context = context.to(self.device)
        context_mask = context_mask.to(self.device)

        # Forward pass
        output = self._forward_action_dit(action, context, context_mask)

        # Check if stop head output is available
        if "stop" not in output:
            raise ValueError(
                "action_dit output missing 'stop' key. "
                "Ensure action_dit was created with predict_stop=True"
            )

        stop_logits = output["stop"]  # [B, T, 1] or [B, T]

        # Compute stop loss
        loss = self._compute_stop_loss(stop_logits, action_is_pad)

        # Metrics
        metrics = {
            "loss": float(loss.detach().item()),
        }

        return loss, metrics

    def val_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single validation step.
        
        Args:
            batch: Validation batch
        
        Returns:
            loss: Scalar loss
            metrics: Dictionary with metrics
        """
        with torch.no_grad():
            loss, metrics = self.train_step(batch)
        return loss, metrics

    def train(self):
        """Main training loop."""
        logger.info("Starting stop head training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Total batches per epoch: {len(self.train_dataloader)}")

        self._freeze_backbone()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self._train_epoch()

            # Validation
            if self.val_dataloader is not None:
                self._validate()

            # Save checkpoint
            self._save_checkpoint(epoch)

        logger.info("Training completed!")
        self._save_final_checkpoint()

    def _train_epoch(self):
        """Train for one epoch."""
        self.action_dit.train()
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            disable=not self.accelerator.is_main_process,
        )

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss, metrics = self.train_step(batch)

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient clipping
            self.accelerator.clip_grad_norm_(
                self.action_dit.stop_head.parameters(),
                self.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Metrics
            epoch_loss += metrics["loss"]
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % 10 == 0 and self.accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

                if self.use_wandb:
                    wandb.log({
                        "train/loss": metrics["loss"],
                        "train/avg_loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/global_step": self.global_step,
                    })

    def _validate(self):
        """Run validation."""
        self.action_dit.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_dataloader,
                desc="Validation",
                disable=not self.accelerator.is_main_process,
            )

            for batch in progress_bar:
                loss, _ = self.val_step(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / max(1, num_batches)

        if self.accelerator.is_main_process:
            logger.info(f"Epoch {self.epoch} - Validation Loss: {avg_val_loss:.4f}")
            if self.use_wandb:
                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/epoch": self.epoch,
                })

            # Track best validation loss
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                logger.info(f"Best validation loss: {avg_val_loss:.4f}")

    def _save_checkpoint(self, epoch: int):
        """Save checkpoint."""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = self.output_dir / f"checkpoint_epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save stop_head state dict
        stop_head_path = checkpoint_dir / "stop_head.pt"
        torch.save(self.action_dit.stop_head.state_dict(), stop_head_path)

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        state_path = checkpoint_dir / "training_state.pt"
        torch.save(state, state_path)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def _save_final_checkpoint(self):
        """Save final checkpoint."""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = self.output_dir / "final"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save stop_head state dict
        stop_head_path = checkpoint_dir / "stop_head.pt"
        torch.save(self.action_dit.stop_head.state_dict(), stop_head_path)

        # Save full model config
        config_path = checkpoint_dir / "config.json"
        config = {
            "model": "action_dit",
            "predict_stop": True,
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.epoch,
            "global_step": self.global_step,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved final checkpoint to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ActionDiT stop head for navigation tasks"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Specific scene name to train on (optional)",
    )
    
    # Model arguments
    parser.add_argument(
        "--action_dit_pretrained_path",
        type=str,
        default="/tmp/fastwam_checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
        help="Path to pretrained ActionDiT checkpoint",
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for stop_head",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps for learning rate schedule",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs/stop_head",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fastwam-stop-head",
        help="Weights & Biases project name",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    logger.info("=== FastWAM Stop Head Training ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load datasets
    logger.info("Loading datasets...")
    dataset_dirs = [args.data_dir] if args.data_dir else []
    
    train_dataset = NavVideoDataset(
        dataset_dirs=dataset_dirs,
        camera_keys=["125cm_0deg", "125cm_30deg"],
        num_frames=33,
        n_history_frames=8,
        n_future_video_frames=8,
        video_size=[224, 224],
        context_len=256,
        sample_stride=4,
        terminal_oversample_ratio=3.0,
    )
    
    logger.info(f"Loaded {len(train_dataset)} training samples")
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Load ActionDiT with stop head
    logger.info("Loading ActionDiT with stop head...")
    action_dit_config = {
        "hidden_dim": 1024,
        "action_dim": 4,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1e-6,
        "use_gradient_checkpointing": False,
        "predict_stop": True,  # Enable stop head
    }
    
    action_dit = ActionDiT.from_pretrained(
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=args.action_dit_pretrained_path,
        skip_dit_load_from_pretrain=False,
        device=args.device,
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"Loaded ActionDiT with stop_head: {hasattr(action_dit, 'stop_head')}")
    
    # Create trainer
    trainer = StopHeadTrainer(
        action_dit=action_dit,
        train_dataloader=train_dataloader,
        val_dataloader=None,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        device=args.device,
        torch_dtype=torch.bfloat16,
        seed=args.seed,
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
