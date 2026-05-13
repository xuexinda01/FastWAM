# FastWAM Conditioning: Quick Reference Guide

## TL;DR - The 5-Minute Overview

### How First Frame Conditioning Works

1. **Encode**: Video → VAE/Visual Encoder → Latents `[B, z_dim, T_lat, H_lat, W_lat]`
2. **Extract**: First frame latents `= latents[:, :, 0:1]` (kept clean)
3. **Freeze**: During training, replace noised first frame with clean: `latents[:, :, 0:1] = first_frame_latents`
4. **Per-Token Timestep**: First frame tokens get `t=0` (clean), future frames get `t=t_sample` (noisy)
5. **Attention Mask**: Action tokens can ONLY attend to first frame video tokens

### Why This Design?

| Aspect | Implementation | Benefit |
|--------|-----------------|---------|
| **Hard Constraint** | Frozen latent injection | Prevents model from forgetting initial condition |
| **Per-Token Timestep** | t=0 for frame 0, t for rest | Teaches model that first frame is clean reference |
| **Attention Mask** | Actions see only first frame | Prevents action loss from corrupting video |
| **Knowledge Insulation** | Detach video K/V for action loss | Action gradients don't update video params |

---

## Complete Data Flow: From Dataset to Loss

```
Dataset Video [B, 3, T, H, W]
    ↓ (VAE/Visual Encoder)
Latents [B, z_dim, T_lat, H_lat, W_lat]
    ↓ (Split & Freeze First Frame)
Noisy Latents (first frame frozen, rest noised)
    ↓ (Patchify + Per-Token Timestep)
Video Tokens [B, T_lat×h×w, hidden_dim]
    ↓ (Expert Pre-DiT)
Video/Action Tokens + Embeddings
    ↓ (MoT Forward Pass - 30 Layers)
Mixed Attention Output
    ↓ (Expert Post-DiT)
Predictions [B, T-1, dims]  (first frame excluded from loss)
    ↓ (Loss Computation)
MSE(pred, target) × timestep_weight
```

---

## MoT Architecture: The Key Innovation

### Two Experts
- **Video Expert** (WanVideoDiT): Processes video latent tokens
- **Action Expert** (ActionDiT): Processes action tokens

### Joint Attention Mechanism

```python
# Each layer:
1. Q/K/V computed independently by each expert
2. Q_cat = [Q_video; Q_action]  # Concatenate
3. K_cat = [K_video; K_action]  # All see all K
4. V_cat = [V_video; V_action]
5. Attention = softmax(Q_cat·K_cat^T/√d + mask)·V_cat
6. Split back to video_out, action_out
7. Optional: Detach video K/V for action gradients (knowledge insulation)
```

### Attention Pattern (After Mask)
```
        Video₀  Video₁  Action
Video₀   ✓       ✗       ✗      (Causal: sees ≤ itself)
Video₁   ✓       ✓       ✗      (Can see earlier frames, no actions)
Action   ✓       ✗       ✓      (Only first frame + self, knowledge barrier)
```

---

## Key Code Locations

### First Frame Conditioning
- **Frozen Injection**: `fastwam.py` line 541-542
  ```python
  if inputs["first_frame_latents"] is not None:
      latents[:, :, 0:1] = inputs["first_frame_latents"]
  ```
  
- **Per-Token Timestep**: `wan_video_dit.py` line 546
  ```python
  token_timesteps[:, 0, :] = 0  # First frame always has t=0
  ```

### MoT Mixed Attention
- **Mixed Attention**: `mot.py` lines 544-612
  - Video sees all K/V normally
  - Action sees detached video K/V + own K/V
  
- **Attention Mask**: `fastwam.py` lines 460-481
  - Action-to-video: first frame only
  - Video-to-video: causal (no future peek)
  - Action-to-action: full connectivity

### Loss Computation
- **Video Loss**: `fastwam.py` lines 483-520
  - Skip first frame: `pred_video = pred_video[:, :, 1:]`
  - Only predict future frames (first frame is frozen anchor)

---

## Design Constraints (Why One Expert Can't Do It)

| Constraint | Why Needed |
|-----------|-----------|
| **Video & Action Must Be Separate Experts** | Action prediction requires knowledge of current state (via first frame), but shouldn't corrupt video training |
| **Actions Can Only See First Frame** | If actions see future frames, they leak information about what video should generate → ill-posed problem |
| **Knowledge Insulation** | Action loss gradients shouldn't update video parameters; only video loss should drive video expert |
| **Per-Token Timestep** | Lets model distinguish "clean reference" (t=0) from "noisy prediction targets" (t>0) |

---

## Common Misconceptions

### ❌ "Extra conditioning images get concatenated like first frame"
**Reality**: No, only first frame is frozen. Other conditioning goes via text embeddings (cross-attention).

### ❌ "Action expert is a separate inference path"
**Reality**: No, action expert is trained jointly with video expert. Shared attention layer allows knowledge transfer.

### ❌ "First frame doesn't get gradients"
**Reality**: It does! But only from action loss, not video loss. First frame is frozen in video loss but visible to action queries.

### ❌ "All frames have the same timestep"
**Reality**: No! First frame = t=0, future frames = t_sample. This is crucial for the model to understand the conditioning mechanism.

---

## Extending the Architecture

If you wanted to add extra conditioning images:

1. **Encode each image** to latent space (like first frame)
2. **Create separate tokens** for each conditioning image
3. **Extend attention mask** to allow all video frames to attend to all conditioning images
4. **Add to context** (via cross-attention embeddings)
5. **Adjust loss** to only predict non-conditioned frames

**Currently not implemented** - the model is designed for single-frame conditioning only.

---

## File Structure Reference

```
fastwam.py                    # Main model
  ├─ build_inputs()           # VAE encoding, first frame extraction
  ├─ _encode_video_latents()  # VAE/visual encoder forward pass
  ├─ training_loss()          # Forward pass + loss computation
  ├─ _compute_video_loss_per_sample()  # Video loss (skips first frame)
  ├─ _build_mot_attention_mask()  # Attention pattern definition
  ├─ infer_joint()            # Video + action inference
  └─ infer_action()           # Action-only inference (fast)

mot.py                        # Mixture of Transformers
  ├─ forward()                # Main mixed attention forward pass
  ├─ prefill_video_cache()    # Precompute video K/V for action inference
  └─ forward_action_with_video_cache()  # Action inference using cached video

wan_video_dit.py              # Video expert
  ├─ pre_dit()                # Patchify, timestep embedding, context
  ├─ post_dit()               # Unpatchify
  └─ forward()                # DiT blocks

action_dit.py                 # Action expert
  ├─ pre_dit()                # Action encoding, timestep embedding
  ├─ post_dit()               # Action projection
  └─ forward()                # DiT blocks

trainer.py                    # Training orchestration
  └─ train()                  # Main training loop
```

---

## Quick Debugging Checklist

- [ ] First frame latents NOT included in loss? (Check `pred_video[:, :, 1:]`)
- [ ] First frame latents frozen during training? (Check `latents[:, :, 0:1] = ...`)
- [ ] Action tokens can't see future video frames? (Check attention mask)
- [ ] Video K/V detached when computing action loss? (Check `detach_video_for_action`)
- [ ] Per-token timestep correctly shaped? (Should be `[B, T*tokens_per_frame]`)
- [ ] First frame token timestep = 0? (Check `token_timesteps[:, 0, :] = 0`)

---

## Performance Tips

- **Action-Only Inference**: Use `infer_action()` instead of `infer_joint()`
  - Precomputes video K/V once, reuses across all action steps
  - ~10-20x faster than joint inference
  
- **Gradient Checkpointing**: Enable `use_gradient_checkpointing=True` in DiT config
  - Saves ~40% memory at ~10% speed cost
  
- **VAE Tiling**: Use `tiled=True` with `tile_size` for high-res videos
  - Prevents OOM on large videos

