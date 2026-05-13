# Wan2.2 FastWAM Model: Image/Video Conditioning in Diffusion Process

## Overview

The FastWAM model is a **Mixture of Transformers (MoT)** world model with **video and action experts** that handles video generation and action prediction jointly. The model uses a sophisticated conditioning mechanism where the first frame is encoded as a frozen reference to guide the video generation process.

---

## 1. First Frame (Condition Image) Encoding & Injection

### 1.1 Dataset Sample → Model Forward Flow

#### Stage 1: Dataset Sample Preparation
```python
# In trainer.py (train method, line 778)
sample = next(data_iter)  # Sample from dataset
```

The sample dict contains:
- `sample["video"]`: [B, 3, T, H, W] - Full video (first frame + future frames)
- `sample["context"]`: [B, L, D] - Text embeddings
- `sample["context_mask"]`: [B, L] - Attention mask for text
- `sample["action"]`: [B, action_horizon, action_dim] - Ground truth actions
- `sample["proprio"]` (optional): Proprioceptive state
- `sample["action_is_pad"]` (optional): Padding mask for actions
- `sample["image_is_pad"]` (optional): Padding mask for frames

#### Stage 2: Input Building & VAE Encoding
```python
# In fastwam.py, build_inputs() method (line 351-457)

# Extract first frame for potential fusion
first_frame_latents = None
fuse_flag = False
if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
    # Encode FULL video to latents
    input_latents = self._encode_video_latents(input_video, tiled=tiled)
    # Extract first frame latents separately
    first_frame_latents = input_latents[:, :, 0:1]  # [B, C, 1, H_lat, W_lat]
    fuse_flag = True
```

**Key Point**: The VAE (or visual encoder) encodes the entire video:
- VAE encoding: `[B, 3, T, H, W] → [B, z_dim, T_lat, H_lat, W_lat]`
- Visual encoder (DINO/VJEPA2): Same output shape for backward compatibility

**Encoding Options**:
1. **VAE encoder** (default):
   ```python
   z = self.vae.encode(
       video_tensor,
       device=self.device,
       tiled=tiled,
       tile_size=tile_size,
       tile_stride=tile_stride,
   )
   ```
   
2. **Visual encoder** (DINO/VJEPA2 - frozen backbone + trainable projection):
   ```python
   z = self.visual_encoder.encode(video_tensor, device=self.device)
   # Only the MLP projection (not the frozen backbone) gets gradients
   ```

#### Stage 3: Training Loss Computation
```python
# In fastwam.py, training_loss() method (line 522-643)

# Add noise to latents for diffusion
noise_video = torch.randn_like(input_latents)
timestep_video = self.train_video_scheduler.sample_training_t(...)
latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

# CRITICAL: Freeze first frame (0:1) in the noised latents
if inputs["first_frame_latents"] is not None:
    latents[:, :, 0:1] = inputs["first_frame_latents"]  # No noise on first frame!
    fuse_flag = True
```

**This is the key conditioning mechanism**:
- The first frame latent is **kept clean** (not noised)
- All future frame latents are noised and denoised during training
- This forces the model to use the first frame as a hard conditioning anchor

### 1.2 First Frame Flow Through DiT (Diffusion Transformer)

#### Pre-DiT Processing
```python
# In wan_video_dit.py, pre_dit() method (line 509-620)

# Video latents: [B, z_dim, T, H_lat, W_lat]
# T = number of latent frames (T_lat = (num_video_frames - 1) // temporal_downsample_factor + 1)

x = self.patchify(x)  # Convert latents to patch tokens
# x: [B, C, T, H, W] → patchified tokens

# Key: Separated timestep handling when fuse_vae_embedding_in_latents=True
if self.seperated_timestep and fuse_vae_embedding_in_latents:
    # Create per-frame timestep embeddings
    token_timesteps = torch.ones(
        (batch_size, x.shape[2], tokens_per_frame),  # [B, T, tokens_per_frame]
        dtype=timestep.dtype,
        device=timestep.device,
    ) * timestep.view(batch_size, 1, 1)
    
    # CRITICAL: First frame gets timestep=0 (clean/uncorrupted signal)
    token_timesteps[:, 0, :] = 0  # First frame always has t=0
    
    # Reshape to [B, T*tokens_per_frame]
    token_timesteps = token_timesteps.reshape(batch_size, -1)
    token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps)  # [B*T*tokens_per_frame, D]
    t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)  # [B, T*tokens_per_frame, D]
    t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))  # [B, T*tokens_per_frame, 6, D]
```

**Critical Design**:
- **Per-token timestep modulation**: Each token gets its own timestep embedding
- **First frame = t=0**: Clean image signal, not a denoising target
- **Future frames = t=t_sample**: Noised signals to be denoised

#### Concat with Context (No Extra Conditioning)
```python
# In wan_video_dit.py, pre_dit() (line 558-599)

context = self.text_embedding(context)  # [B, L, D]
context_len = context.shape[1]

# If action conditioning enabled, append action embeddings to context
if self.action_conditioned and action is not None:
    action_len = action.shape[1]
    action_emb = self.action_embedding(action)  # [B, action_len, D]
    action_pos_embed = sinusoidal_embedding_1d(
        self.hidden_dim, 
        torch.arange(action_len, device=action_emb.device)
    )
    action_emb = action_emb + action_pos_embed.unsqueeze(0)
    context = torch.cat([context, action_emb], dim=1)  # Extend context
```

**No explicit "extra conditioning images"**: 
- Only first frame via frozen latent injection
- Action sequences appended to text context for cross-attention
- **No mechanism for second/third frame conditioning via cross-attention**

### 1.3 Latent Injection Summary

```
Dataset Video [B, 3, T, H, W]
    ↓
VAE/Visual Encoder
    ↓
Latents [B, z_dim, T, H_lat, W_lat]
    ↓
Split:
  - first_frame_latents = latents[:, :, 0:1]  (keep clean)
  - rest_latents = latents[:, :, 1:]  (add noise)
    ↓
During Training:
  - Noised latents[:, :, 1:] = noise + (1-α)^0.5 * rest_latents
  - Frozen latents[:, :, 0:1] = first_frame_latents (NO noise)
    ↓
Through DiT with per-token timestep:
  - First frame tokens: t_mod with t=0
  - Future frame tokens: t_mod with t=t_sample (noised)
```

---

## 2. MoT (Mixture of Transformers) Design

### 2.1 Expert Architecture

```python
# In mot.py, __init__() (line 15-56)

class MoT(nn.Module):
    def __init__(self, mixtures: Dict[str, nn.Module], mot_checkpoint_mixed_attn: bool = True):
        self.mixtures = nn.ModuleDict(mixtures)  # {'video': expert, 'action': expert}
        self.expert_order = list(self.mixtures.keys())  # ['video', 'action']
```

**Two Experts**:
1. **Video Expert** (WanVideoDiT): Processes video latent tokens
   - Input: Patchified latents [B, T*H*W, D]
   - Output: Denoised latents [B, T*H*W, D]
   
2. **Action Expert** (ActionDiT): Processes action tokens
   - Input: Action embeddings [B, action_horizon, D]
   - Output: Predicted actions [B, action_horizon, action_dim]

**Architectural Constraints** (enforced in `from_wan22_pretrained`):
```python
if int(action_expert.num_heads) != int(video_expert.num_heads):
    raise ValueError("num_heads must match")
if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
    raise ValueError("attn_head_dim must match")
if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
    raise ValueError("num_layers must match")
```

### 2.2 Joint Attention Mechanism

#### Building Experts' Q/K/V
```python
# In mot.py, forward() (line 447-638)

for layer_idx in range(self.num_layers):
    # For each expert (video, action):
    for name in self.expert_order:
        expert = self.mixtures[name]
        block = expert.blocks[layer_idx]
        x = tokens_all[name]  # Current expert tokens
        
        # Build Q/K/V for this expert
        q, k, v = self._build_expert_attention_io(
            expert=expert,
            block=block,
            x=x,
            freqs=freqs_all[name],
            t_mod=t_mod_all[name],
        )
        
        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)
```

#### Mixed Attention (Concatenated)
```python
# In mot.py, forward() (line 544-612)

# STEP 1: Concatenate all Q/K/V
q_cat = torch.cat(q_chunks, dim=1)  # [B, Sv + Sa, D]
k_cat = torch.cat(k_chunks, dim=1)  # [B, Sv + Sa, D]
v_cat = torch.cat(v_chunks, dim=1)  # [B, Sv + Sa, D]

# STEP 2: Special handling for video-to-action gradient flow
if detach_video_for_action and len(self.expert_order) == 2:
    # Video queries attend to EVERYTHING
    video_seq_len = seq_lens[0]
    video_mask = attention_mask[:video_seq_len, :total_seq]
    video_mixed = self._mixed_attention(
        q_cat=q_chunks[0],  # Video Q
        k_cat=k_cat,        # All K (video + action)
        v_cat=v_cat,        # All V (video + action)
        attention_mask=video_mask,
    )
    
    # Action queries attend to DETACHED video K/V + normal action K/V
    k_cat_detached = torch.cat([k_chunks[0].detach(), k_chunks[1]], dim=1)
    v_cat_detached = torch.cat([v_chunks[0].detach(), v_chunks[1]], dim=1)
    action_mask = attention_mask[video_seq_len:, :total_seq]
    action_mixed = self._mixed_attention(
        q_cat=q_chunks[1],       # Action Q
        k_cat=k_cat_detached,    # Detached video K + action K
        v_cat=v_cat_detached,    # Detached video V + action V
        attention_mask=action_mask,
    )
    
    mixed = torch.cat([video_mixed, action_mixed], dim=1)
```

**Key Feature: Knowledge Insulation**
- When `action_loss_detach_video_expert=True`:
  - Video tokens see action tokens normally (full gradients)
  - Action tokens see detached video tokens (isolated from video loss)
  - This prevents action loss from directly updating video parameters

### 2.3 Attention Mask Structure

```python
# In fastwam.py, _build_mot_attention_mask() (line 460-481)

def _build_mot_attention_mask(
    self,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    total_seq_len = video_seq_len + action_seq_len
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
    
    # Video-to-video: causal (can attend to first frame and same/earlier frames)
    mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(...)
    
    # Action-to-action: full (all action tokens attend to each other)
    mask[video_seq_len:, video_seq_len:] = True
    
    # Action-to-video: ONLY first frame
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    mask[video_seq_len:, :first_frame_tokens] = True  # Actions can see first frame
    
    return mask
```

**Attention Pattern**:
```
        Video₀  Video₁  Action
Video₀   ✓       ✗       ✗
Video₁   ✓       ✓       ✗
Action   ✓       ✗       ✓     (Only first frame of video)
```

**Semantic**:
- Video frames attend causally (can't peek into future)
- Action tokens see first frame (frozen condition)
- Action tokens don't see future video frames (prevents information leak)

### 2.4 Forward Pass in Training

```python
# In fastwam.py, training_loss() (line 553-603)

# Pre-process video and action
video_pre = self.video_expert.pre_dit(
    x=latents,                    # Noised latents with frozen first frame
    timestep=timestep_video,      # Per-token timestep (0 for first frame)
    context=context,              # Text embeddings
    context_mask=context_mask,
    action=action,                # Ground truth actions (for context)
    fuse_vae_embedding_in_latents=fuse_flag,
)

action_pre = self.action_expert.pre_dit(
    action_tokens=noisy_action,   # Noised actions
    timestep=timestep_action,
    context=context,
    context_mask=context_mask,
)

video_tokens = video_pre["tokens"]  # [B, video_seq_len, D]
action_tokens = action_pre["tokens"]  # [B, action_seq_len, D]

# Mixed attention over all tokens
tokens_out = self.mot(
    embeds_all={
        "video": video_tokens,
        "action": action_tokens,
    },
    attention_mask=attention_mask,
    freqs_all={
        "video": video_pre["freqs"],
        "action": action_pre["freqs"],
    },
    context_all={
        "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
        "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
    },
    t_mod_all={
        "video": video_pre["t_mod"],
        "action": action_pre["t_mod"],
    },
    detach_video_for_action=self.action_loss_detach_video_expert,
)

# Post-process outputs
pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
```

---

## 3. Training Data Flow: From Dataset to Loss

### 3.1 Complete Training Step

```
1. Dataset Sample
   ├─ video: [B, 3, T, H, W]
   ├─ action: [B, T-1, action_dim]
   ├─ context: [B, L, D]
   └─ context_mask: [B, L]
       ↓
2. Encoder
   ├─ Encode video to latents: [B, z_dim, T_lat, H_lat, W_lat]
   ├─ Extract first frame: [B, z_dim, 1, H_lat, W_lat]
   └─ Keep in build_inputs()
       ↓
3. Noise Scheduling
   ├─ Sample timestep: t ~ U(0, 1000)
   ├─ Add noise to all latents
   ├─ FREEZE first frame (replace with clean latent)
   └─ Create target (what to predict)
       ↓
4. DiT Pre-processing
   ├─ Patchify latents: [B, z_dim, T_lat, H_lat, W_lat] → [B, T_lat*h*w, D]
   ├─ Per-token timestep (0 for first frame, t for rest)
   ├─ Embed context + actions
   └─ Build RoPE frequencies
       ↓
5. MoT Forward
   ├─ Video expert: Process video tokens with first-frame anchor
   ├─ Action expert: Process action tokens
   ├─ Joint attention: Mixed Q/K/V with first-frame-only action access
   └─ Post-process outputs
       ↓
6. Loss Computation
   ├─ Video loss: MSE(pred_video, target_video)
   │  └─ EXCLUDE first frame (don't predict it, use clean)
   ├─ Action loss: MSE(pred_action, target_action)
   ├─ Weighted by timestep importance
   └─ Total loss = λ_video * loss_video + λ_action * loss_action
       ↓
7. Backward & Optimize
   ├─ Gradient computation
   ├─ Optional action loss detaches video K/V
   ├─ Gradient clipping
   └─ Optimizer step
```

### 3.2 Key Loss Computation Details

```python
# In fastwam.py, _compute_video_loss_per_sample() (line 483-520)

include_initial_video_step = inputs["first_frame_latents"] is None

if inputs["first_frame_latents"] is not None:
    # Remove first frame from predictions and targets
    pred_video = pred_video[:, :, 1:]        # Skip first frame
    target_video = target_video[:, :, 1:]    # Skip first frame

loss_video_token = F.mse_loss(
    pred_video.float(),
    target_video.float(),
    reduction="none"
).mean(dim=(1, 3, 4))  # Average over channels, H, W

# Apply per-frame masking if available
if image_is_pad is not None:
    # ... create mask for valid frames ...
    valid = (~video_is_pad).to(device=loss_video_token.device)
    valid_sum = valid.sum(dim=1).clamp(min=1.0)
    loss_per_sample = (loss_video_token * valid).sum(dim=1) / valid_sum
else:
    loss_per_sample = loss_video_token.mean(dim=1)

# Weight by timestep importance
video_weight = self.train_video_scheduler.training_weight(timestep_video)
loss_video = (loss_per_sample * video_weight).mean()
```

**Loss Design**:
- **First frame**: Never predicted (frozen anchor)
- **Future frames**: Predicted via denoising
- **Timestep weighting**: Later timesteps (more noise) get higher weight
- **Padding mask**: Ignore padded/invalid frames

---

## 4. Conditioning Mechanisms: Current vs Potential

### 4.1 Current Conditioning Mechanisms

| Mechanism | Implementation | Data Flow |
|-----------|-----------------|-----------|
| **First Frame (Hard)** | Frozen latent injection | Latent space (VAE/visual encoder output) |
| **Text Prompt** | Cross-attention in DiT blocks | Embedded via text encoder → context |
| **Action Sequence** | Appended to context for video expert | Embedded via action embedding → context extension |
| **Per-Token Timestep** | Token-level time modulation (t=0 for frame 0) | Via `t_mod` parameter to DiT blocks |

### 4.2 No Extra Image Conditioning Mechanism

**Findings**:
1. **Single first frame only**: The model only uses the first frame as conditioning
2. **No second-frame conditioning**: No mechanism to provide a second image
3. **No cross-attention from additional images**: All additional conditioning goes through text + action context
4. **No image-to-image mode**: The architecture is not designed for multi-image conditioning

**Why this design**:
- Video generation is inherently sequential
- First frame is sufficient anchor for spatiotemporal consistency
- Additional frames would be redundant (the model generates them)
- Reduces computational overhead

### 4.3 Potential for Extra Conditioning

To add extra conditioning images, one would need to:

1. **Encode additional images** to latent space
2. **Create separate expert or cross-attention pathway**
3. **Add to attention mask and sequence** (increase `attention_mask` dimensions)
4. **Modify context building** to include image embeddings

Example pseudo-code:
```python
# NOT in current codebase - hypothetical addition

if extra_images is not None:
    extra_latents = self.visual_encoder.encode(extra_images)  # [B, C, 1, h, w] × N
    # Would need new cross-attention pathway
```

---

## 5. Key Design Insights

### 5.1 Why First Frame Gets Timestep=0

From `wan_video_dit.py`, line 546:
```python
token_timesteps[:, 0, :] = 0  # First frame always has t=0
```

**Rationale**:
- `t=0` means "fully denoised" or "clean signal"
- The first frame isn't being denoised; it's a fixed anchor
- DiT blocks interpret `t_mod` differently for `t=0`:
  - No noise prediction is needed
  - Acts as a stable reference for later frames
  - Allows model to learn video coherence relative to first frame

### 5.2 Frozen First Frame Injection

From `fastwam.py`, line 541-542:
```python
if inputs["first_frame_latents"] is not None:
    latents[:, :, 0:1] = inputs["first_frame_latents"]
```

**Why critical**:
- During forward pass: Clean first frame is always clean (no noise added/removed)
- Gradients can't flow through first frame during video loss
- But gradients DO flow during action loss (action sees first frame via attention)
- This creates a **one-way information flow**: first frame → action; action ↛ first frame

### 5.3 Action Loss Knowledge Insulation

From `mot.py`, line 556-574:
```python
if detach_video_for_action:
    # Video sees action normally
    video_mixed = flash_attention(q_chunks[0], k_cat, v_cat, ...)
    
    # Action sees DETACHED video K/V
    k_cat_detached = torch.cat([k_chunks[0].detach(), k_chunks[1]], dim=1)
    v_cat_detached = torch.cat([v_chunks[0].detach(), v_chunks[1]], dim=1)
    action_mixed = flash_attention(q_chunks[1], k_cat_detached, v_cat_detached, ...)
```

**Design Goal**:
- Prevents action loss from corrupting video representation
- Video loss still drives all video parameters
- Action loss only directly drives action parameters
- Allows independent scaling of loss weights

---

## 6. Inference Modes

### 6.1 Joint Inference (Video + Action)

```python
# fastwam.py, infer_joint() (line 801-978)

# 1. Encode first image to latents
first_frame_latents = self._encode_input_image_latents_tensor(input_image)

# 2. Freeze first frame latents during denoising loop
latents_video = torch.randn(...)  # Initialize noise
latents_video[:, :, 0:1] = first_frame_latents.clone()

# 3. Denoise both video and action jointly
for step_t_video, step_delta_video, ... in zip(...):
    pred_video, pred_action = self._predict_joint_noise(...)
    latents_video = scheduler.step(pred_video, step_delta_video, latents_video)
    latents_action = scheduler.step(pred_action, step_delta_action, latents_action)
    
    # Keep first frame frozen throughout
    latents_video[:, :, 0:1] = first_frame_latents.clone()
```

### 6.2 Action-Only Inference (Fast)

```python
# fastwam.py, infer_action() (line 981-1123)

# 1. Prefill video cache (just first frame)
timestep_video = torch.zeros(...)  # t=0
video_pre = self.video_expert.pre_dit(
    x=first_frame_latents,
    timestep=timestep_video,
    ...
)
video_kv_cache = self.mot.prefill_video_cache(
    video_tokens=video_pre["tokens"],
    ...
)

# 2. Denoise actions using cached video
for step_t_action, step_delta_action in zip(...):
    pred_action = self._predict_action_noise_with_cache(
        latents_action=latents_action,
        timestep_action=timestep_action,
        video_kv_cache=video_kv_cache,  # Reuse video
        ...
    )
    latents_action = scheduler.step(pred_action, ...)
```

**Key Optimization**:
- Precompute video expert K/V once
- Reuse across all action denoising steps
- Significant speedup for action-only inference

---

## 7. Summary: Condition Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Input: First Frame Image [B, 3, H, W]                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ VAE/Visual Encoder (frozen backbone + trainable projection) │
│ Output: First Frame Latents [B, z_dim, 1, h_lat, w_lat]   │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────┐
        │ Training: Add Gaussian Noise to other frames        │
        │ Inference: Keep clean                               │
        │ → Frozen latent: latents[:, :, 0:1] = first_frame   │
        └─────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ DiT Pre-processing                                            │
│ - Patchify latents: [B, z_dim, T, h_lat, w_lat] → tokens   │
│ - Per-token timestep: t=0 for first frame, t=t_sample rest  │
│ - Embed context (text + actions)                            │
└──────────────────────────────────────────────────────────────┘
                              ↓
    ┌───────────────────────────────────────────────────┐
    │        MoT: Mixed Attention Layer L times         │
    │                                                   │
    │  Video Expert Q/K/V (all frames)                 │
    │        ↓                                           │
    │  ┌─────────────────────────────────────┐          │
    │  │ Mixed Attention with Action Expert  │          │
    │  │ - All Q can attend to all K/V       │          │
    │  │ - Action Q sees first frame only    │          │
    │  └─────────────────────────────────────┘          │
    │        ↓                                           │
    │  Cross-attention: Query context embeddings        │
    │  (Text for both; Actions for video)               │
    │        ↓                                           │
    │  Updated Tokens → Next Layer                      │
    └───────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Post-DiT                                                      │
│ - Unpatchify: tokens → [B, z_dim, T-1, h_lat, w_lat]        │
│   (First frame is not predicted, was frozen)                 │
│ - Predict actions: [B, T-1, action_dim]                      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Loss Computation                                              │
│ - Video MSE: Predict only T-1 frames (skip first)            │
│ - Action MSE: Predict action sequence                        │
│ - Total: λ_video × loss_video + λ_action × loss_action      │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. File References

| Component | File | Key Methods |
|-----------|------|-------------|
| Main Model | `fastwam.py` | `build_inputs()`, `training_loss()`, `infer_joint()`, `infer_action()` |
| MoT | `mot.py` | `forward()`, `prefill_video_cache()`, `forward_action_with_video_cache()` |
| Video DiT | `wan_video_dit.py` | `pre_dit()`, `patchify()`, `post_dit()` |
| Action DiT | `action_dit.py` | `pre_dit()`, `post_dit()` |
| Trainer | `trainer.py` | `train()`, `_set_dit_only_train_mode()` |
| Visual Encoder | `visual_encoder.py` | `BaseVisualEncoder`, `encode()` |

