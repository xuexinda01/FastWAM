# Stop Head Training Guide

## Overview

This guide explains how to use the standalone stop head training script to fine-tune the ActionDiT stop head for navigation tasks. The stop head learns to predict when a navigation agent has reached its goal (stop prediction).

**Key Concept**: The stop head outputs `1` when the trajectory should terminate (goal reached) and `0` when the agent should continue moving.

---

## Architecture & Concepts

### What is the Stop Head?

The stop head is an optional binary classification head added to the ActionDiT (Action Diffusion Transformer). It predicts:
- **1.0 (logit → sigmoid → 1)**: Trajectory should stop (goal reached)
- **0.0 (logit → sigmoid → 0)**: Trajectory should continue (keep moving)

### Data Flow

```
Raw action sequence [32 steps]
    ↓
ActionDiT pre_dit() → Embeddings + time modulation
    ↓
DiT blocks (FROZEN - not trained)
    ↓
ActionHead → Action predictions (not trained)
StopHead → Stop predictions (TRAINED ← here)
    ↓
Loss computation:
  - BCE(stop_pred, action_is_pad)
  - action_is_pad encodes ground truth stops
```

### Ground Truth Encoding

The dataset provides `action_is_pad` field:
- **False** (0): Step is valid/ongoing (stop = 0)
- **True** (1): Step is padded/goal-reached (stop = 1)

This comes from the trajectory structure:
```python
# If trajectory has 20 steps and we need 32:
steps 0-19: action_is_pad = False  (real trajectory)
steps 20-31: action_is_pad = True   (padded beyond end = reached goal)
```

---

## Setup & Installation

### 1. Verify Dependencies

```bash
# Check that these are installed:
pip list | grep -E "torch|accelerate|tqdm|wandb"

# Required packages:
# - torch >= 2.0.0
# - accelerate >= 0.25.0
# - tqdm
# - wandb (optional, for experiment tracking)
```

### 2. Check FastWAM Installation

```bash
# Verify the script can find FastWAM modules:
python -c "from fastwam.models.wan22.action_dit import ActionDiT; print('✓ FastWAM importable')"
```

### 3. Prepare Dataset

The script uses LeRobot format navigation data. You need:
- Dataset directory with parquet files
- Pre-computed T5 text embeddings

```bash
# Expected structure:
/path/to/lerobot/datasets/
├── scene_1/
│   ├── data/chunk-000/
│   │   └── episode_000000.parquet  # Pose data
│   ├── videos/chunk-000/
│   │   ├── observation.images.rgb.125cm_0deg/
│   │   │   └── episode_000000_*.jpg
│   │   └── observation.images.rgb.125cm_30deg/
│   │       └── episode_000000_*.jpg
│   └── meta/episodes.jsonl         # Episode metadata
└── scene_2/
    └── ...
```

### 4. Prepare Pre-computed Text Embeddings

The dataset requires pre-cached T5 embeddings. Generate them first:

```bash
python scripts/precompute_nav_text_embeds.py \
    --data_dir /path/to/lerobot/datasets \
    --output_dir ./text_embeddings
```

Set `text_embedding_cache_dir` in your config or modify the dataset loading call.

---

## Quick Start

### Minimal Example

```bash
python scripts/train_stop_head_standalone.py \
    --data_dir /path/to/lerobot/datasets \
    --output_dir ./runs/stop_head_v1 \
    --batch_size 16 \
    --num_epochs 5
```

### With Weights & Biases Logging

```bash
python scripts/train_stop_head_standalone.py \
    --data_dir /path/to/lerobot/datasets \
    --output_dir ./runs/stop_head_v1 \
    --batch_size 16 \
    --num_epochs 10 \
    --use_wandb \
    --wandb_project my-navigation-project
```

### With Custom Configuration

```bash
python scripts/train_stop_head_standalone.py \
    --data_dir /path/to/lerobot/datasets \
    --output_dir ./runs/stop_head_v1 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_epochs 20 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0 \
    --use_wandb
```

---

## Command Line Arguments

### Data Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | **Required** | Path to LeRobot dataset directory |
| `--scene_name` | str | None | Specific scene to train on (optional) |

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--action_dit_pretrained_path` | str | `/tmp/fastwam_checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | Path to pretrained ActionDiT |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_epochs` | int | 10 | Number of training epochs |
| `--batch_size` | int | 32 | Training batch size |
| `--learning_rate` | float | 1e-4 | Learning rate for stop_head |
| `--num_workers` | int | 4 | Data loading workers |
| `--gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `--warmup_steps` | int | 500 | Warmup steps for LR schedule |
| `--max_grad_norm` | float | 1.0 | Max gradient norm for clipping |

### Output Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `./runs/stop_head` | Directory to save checkpoints |
| `--use_wandb` | flag | False | Enable Weights & Biases logging |
| `--wandb_project` | str | `fastwam-stop-head` | W&B project name |

### Other Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | str | `cuda` | Device to train on (`cuda` or `cpu`) |

---

## Training Loop Details

### What Gets Trained

✅ **Trained:**
- `stop_head.norm` — LayerNorm parameters
- `stop_head.proj` — Linear projection layer
- `stop_head.modulation` — Time modulation parameters

❌ **Frozen:**
- `action_encoder` — Input embedding layer
- `text_embedding` — Text encoder layers
- `time_embedding` — Timestep encoder
- All DiT blocks (30 layers)
- `head` (action head)

### Loss Function

```python
# Binary cross-entropy with logits
BCE = F.binary_cross_entropy_with_logits(
    stop_logits,           # [B, T, 1]
    action_is_pad.float()  # [B, T]
)
```

The loss penalizes:
- Predicting stop=0 when the trajectory actually ended
- Predicting stop=1 when the trajectory is still ongoing

### Optimization Strategy

1. **Optimizer**: AdamW with weight decay
   - Learning rate: 1e-4 (configurable)
   - Beta1: 0.9, Beta2: 0.999
   - Weight decay: 0.01

2. **Learning Rate Schedule**: Warmup → Cosine Decay
   - Linear warmup for first `warmup_steps` steps
   - Cosine decay for remaining steps
   - Minimum LR: 0 (annealed to zero at end of training)

3. **Gradient Clipping**: Max norm = 1.0
   - Prevents unstable gradients
   - Applied only to stop_head parameters

---

## Checkpoints & Outputs

### Directory Structure

```
./runs/stop_head_v1/
├── checkpoint_epoch_0/
│   ├── stop_head.pt          # Stop head weights
│   └── training_state.pt     # Optimizer + scheduler state
├── checkpoint_epoch_1/
│   ├── stop_head.pt
│   └── training_state.pt
├── ...
└── final/
    ├── stop_head.pt          # Best stop head weights
    └── config.json           # Training metadata
```

### Checkpoint Contents

**stop_head.pt:**
```python
{
    "norm.weight": torch.Tensor,       # [1024]
    "norm.bias": torch.Tensor,         # [1024]
    "proj.weight": torch.Tensor,       # [1, 1024]
    "proj.bias": torch.Tensor,         # [1]
    "modulation": torch.Tensor,        # [1, 2, 1024]
}
```

**config.json:**
```json
{
    "model": "action_dit",
    "predict_stop": true,
    "best_val_loss": 0.45,
    "final_epoch": 9,
    "global_step": 5000
}
```

---

## Using Trained Checkpoints

### Loading Stop Head Weights

```python
import torch
from fastwam.models.wan22.action_dit import ActionDiT

# Create ActionDiT with stop head
action_dit = ActionDiT(
    hidden_dim=1024,
    action_dim=4,
    ffn_dim=4096,
    num_heads=24,
    attn_head_dim=128,
    num_layers=30,
    text_dim=4096,
    freq_dim=256,
    eps=1e-6,
    predict_stop=True,
)

# Load trained stop head
checkpoint_path = "./runs/stop_head_v1/final/stop_head.pt"
stop_head_state = torch.load(checkpoint_path, map_location="cpu")
action_dit.stop_head.load_state_dict(stop_head_state)

print("✓ Stop head weights loaded")
```

### Inference

```python
import torch.nn.functional as F

# Forward pass
output = action_dit(
    action_tokens=noisy_actions,       # [B, T, 4]
    timestep=timestep,                 # [B]
    context=text_context,              # [B, L, 4096]
    context_mask=context_mask,         # [B, L]
)

# Extract predictions
action_pred = output["action"]         # [B, T, 4]
stop_logits = output["stop"]           # [B, T, 1]

# Convert to probabilities
stop_probs = torch.sigmoid(stop_logits)  # [B, T, 1] in (0, 1)
```

### Integration with FastWAM

```python
from fastwam.models.wan22.fastwam import FastWAM

# FastWAM automatically uses stop head if available
fastwam = FastWAM(...)  # with action_expert that has predict_stop=True

# In training_loss():
loss_dict = fastwam.training_loss(batch)
# Returns dict with:
# - "loss_video": video generation loss
# - "loss_action": action MSE loss
# - "loss_stop": stop prediction loss (if stop head enabled)
```

---

## Monitoring Training

### Console Output

```
Epoch 0: 100%|██████████| 1000/1000 [10:23<00:00,  1.61it/s, loss=0.4521, lr=5.00e-05]
Epoch 1: 100%|██████████| 1000/1000 [10:15<00:00,  1.62it/s, loss=0.3892, lr=4.80e-05]
...
INFO - Saved checkpoint to ./runs/stop_head_v1/checkpoint_epoch_0
INFO - Validation Loss: 0.4234
```

### Weights & Biases Dashboard

When using `--use_wandb`, you can view:

1. **Loss Curves**
   - `train/loss` — Per-batch loss
   - `train/avg_loss` — Per-epoch average
   - `val/loss` — Validation loss
   - `train/learning_rate` — LR schedule

2. **Metrics Table**
   - Batch number
   - Epoch
   - Loss values
   - Gradient norms (if logging enabled)

3. **System Metrics**
   - GPU memory usage
   - Training throughput
   - Wallclock time

---

## Troubleshooting

### Error: "stop_head has no parameters"

**Cause**: ActionDiT created without `predict_stop=True`

**Solution**:
```python
# Wrong:
action_dit = ActionDiT(...)  # predict_stop defaults to False

# Correct:
action_dit = ActionDiT(..., predict_stop=True)
```

### Error: "action_dit output missing 'stop' key"

**Cause**: Model wasn't created with stop head enabled

**Solution**: Check the trainer initialization and verify `predict_stop=True` in config

### CUDA Out of Memory

**Solutions**:
1. Reduce `--batch_size` (e.g., 16 instead of 32)
2. Reduce `--gradient_accumulation_steps` if using it
3. Ensure no other processes using GPU

### Very High or NaN Losses

**Causes**:
- Learning rate too high
- Gradient explosion
- Mismatched tensor shapes

**Solutions**:
1. Reduce `--learning_rate` (e.g., 5e-5 instead of 1e-4)
2. Increase `--warmup_steps` for gentler start
3. Check ground truth labels (`action_is_pad`) are correct

### Slow Training Speed

**Optimization**:
1. Increase `--batch_size` if GPU memory allows
2. Increase `--num_workers` for data loading
3. Use `pin_memory=True` (already enabled in script)
4. Run on GPU with sufficient VRAM

---

## Advanced Usage

### Multi-GPU Training

The script uses Accelerate which supports multi-GPU training:

```bash
# Single GPU (default)
python scripts/train_stop_head_standalone.py ...

# Multi-GPU via accelerate
accelerate launch --multi_gpu \
    scripts/train_stop_head_standalone.py ...

# Custom config
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
    scripts/train_stop_head_standalone.py ...
```

### Gradient Accumulation

Use to simulate larger batches without more VRAM:

```bash
python scripts/train_stop_head_standalone.py \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    # Effective batch = 8 * 4 = 32
```

### Custom Learning Rate Schedule

Modify `_setup_scheduler()` in the trainer class to use different schedules:

```python
# Constant learning rate:
self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

# Linear decay:
self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
    start_factor=1.0, end_factor=0.0, total_iters=1000)

# Step decay:
self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
    step_size=5, gamma=0.1)
```

### Mixing with Full Model Training

To train stop head together with other components:

```python
# Modify _setup_optimizer to include more parameters
# For example, to also train text_embedding:
stop_head_params = list(self.action_dit.stop_head.parameters())
text_emb_params = list(self.action_dit.text_embedding.parameters())
trainable_params = stop_head_params + text_emb_params
```

---

## Performance Metrics

### Expected Performance

On typical navigation datasets:
- **Initial loss**: 0.6-0.7 (random initialization)
- **After 1 epoch**: 0.45-0.55
- **After 5 epochs**: 0.35-0.45
- **After 10 epochs**: 0.25-0.35

### Benchmarks

Training times on single A100 GPU:
- **Batch size 32**: ~10 min/epoch for 1000 steps
- **Batch size 64**: ~7 min/epoch (with good throughput)
- **Full training (10 epochs)**: ~1.5-2 hours

Memory usage:
- **Batch size 32**: ~15 GB VRAM
- **Batch size 16**: ~10 GB VRAM

---

## Best Practices

1. **Start with small batch size** to ensure training works before scaling
2. **Use validation set** to monitor overfitting (implement val_dataloader)
3. **Save multiple checkpoints** — keep first, best, and final
4. **Monitor learning rate** — ensure it's decaying properly
5. **Check gradient norms** — add logging if training is unstable
6. **Use seed** for reproducibility in experiments
7. **Log to wandb** for easy experiment tracking

---

## Configuration Examples

### Conservative (Stable)

```bash
python scripts/train_stop_head_standalone.py \
    --data_dir data/ \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 20 \
    --warmup_steps 1000 \
    --max_grad_norm 1.0
```

### Aggressive (Faster)

```bash
python scripts/train_stop_head_standalone.py \
    --data_dir data/ \
    --batch_size 64 \
    --learning_rate 2e-4 \
    --num_epochs 5 \
    --warmup_steps 100 \
    --max_grad_norm 2.0
```

### Production (Robust)

```bash
python scripts/train_stop_head_standalone.py \
    --data_dir data/ \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --warmup_steps 2000 \
    --max_grad_norm 1.0 \
    --use_wandb \
    --wandb_project production-models
```

---

## Related Documentation

- **ActionDiT Architecture**: See `src/fastwam/models/wan22/action_dit.py`
- **Dataset Format**: See `src/fastwam/datasets/lerobot/nav_video_dataset.py`
- **Full FastWAM Training**: See `README.md` and `scripts/train.py`
- **Stop Head Tests**: See `test_action_dit_stop_head.py`

---

## Support & Debugging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Model State

```python
# Verify stop head is trainable
print(f"Stop head trainable params: {sum(p.numel() for p in action_dit.stop_head.parameters() if p.requires_grad)}")

# Verify backbone is frozen
print(f"DiT trainable params: {sum(p.numel() for p in action_dit.blocks[0].parameters() if p.requires_grad)}")
```

### Visualize Loss Curves

After training, analyze with:
```python
import json
import matplotlib.pyplot as plt

# Load training logs from wandb or checkpoint
# Plot loss_stop vs epoch
```

