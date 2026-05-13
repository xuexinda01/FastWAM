# FastWAM Conditioning: Practical Examples & Use Cases

## Example 1: Tracing a Training Step

### Scenario: What happens when I call `trainer.train()`?

**Step 1: Get batch from DataLoader**
```python
sample = next(data_iter)  # fastwam.py trainer.py line 778

# sample contains:
{
    'video': torch.randn(2, 3, 9, 224, 448),      # [B=2, C=3, T=9, H=224, W=448]
    'action': torch.randn(2, 32, 7),              # [B=2, horizon=32, dim=7]
    'context': torch.randn(2, 128, 4096),         # [B=2, seq_len=128, d=4096]
    'context_mask': torch.ones(2, 128),           # [B=2, seq_len]
    'proprio': torch.randn(2, 8),                 # [B=2, dim=8]
}
```

**Step 2: Build Inputs (fastwam.py lines 351-457)**
```python
inputs = self.build_inputs(sample)

# Inside build_inputs():
# 1. Encode video with VAE
input_latents = self._encode_video_latents(video)  # [2, 16, 3, 28, 56]
#    (3 = 9 frames with temporal downsample 4x, 28 = 224/8, 56 = 448/8)

# 2. Extract first frame
first_frame_latents = input_latents[:, :, 0:1]   # [2, 16, 1, 28, 56]

# 3. Encode proprio
proprio_emb = self.proprio_encoder(sample['proprio'][:, 0])  # [2, 4096]

# 4. Append proprio to context
inputs['context'] = torch.cat([context, proprio_emb.unsqueeze(1)], dim=1)
#                            [2, 128, 4096] + [2, 1, 4096] = [2, 129, 4096]

# 5. Store first frame for later
inputs['first_frame_latents'] = first_frame_latents  # [2, 16, 1, 28, 56]
```

**Step 3: Add Noise (fastwam.py lines 522-543)**
```python
# Sample timestep
timestep_video = self.train_video_scheduler.sample_training_t()  # shape: [2]
# e.g., timestep_video = tensor([300, 750])

# Add noise to all frames
noise_video = torch.randn_like(input_latents)  # [2, 16, 3, 28, 56]
latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
#  Now: latents = √ᾱ_t * input_latents + √(1-ᾱ_t) * noise_video
#       latents has shape [2, 16, 3, 28, 56] but is NOISY

# CRITICAL: Freeze first frame (replace noised version with clean)
if inputs["first_frame_latents"] is not None:
    latents[:, :, 0:1] = inputs["first_frame_latents"]
    # Now latents[:,  :, 0:1] is CLEAN again!
    # latents[:, :, 1:] is still NOISY

# Similarly for actions
noise_action = torch.randn_like(sample['action'])  # [2, 32, 7]
action_latents = self.train_action_scheduler.add_noise(...)  # [2, 32, 7] NOISY
```

**Step 4: Video Expert Pre-DiT (wan_video_dit.py lines 509-620)**
```python
video_pre = self.video_expert.pre_dit(
    x=latents,                    # [2, 16, 3, 28, 56] (latents with frozen first frame)
    timestep=timestep_video,      # [2]
    context=inputs['context'],    # [2, 129, 4096]
    context_mask=context_mask,    # [2, 129]
    action=sample['action'],      # [2, 32, 7] (used only for context, not conditioning)
    fuse_vae_embedding_in_latents=True,
)

# Inside pre_dit():
# 1. Patchify: [2, 16, 3, 28, 56] → [2, 3072, 3, 14, 28]
#    (Using Conv3d kernel=[1,2,2] to fuse spatial patches)
#    → reshape to [2, 1176, 3072]
#      (3 frames × 14×28 = 1176 tokens, each dim 3072)

# 2. Create per-token timestep
token_timesteps = torch.full((2, 1176), timestep_video).reshape(2, -1)
# [2, 1176] where each value is either 300 or 750 (from timestep_video)

# 3. CRITICAL: First frame tokens get t=0
token_timesteps[:, :392] = 0  # First frame only (392 = 1×14×28 tokens/frame)
# Now token_timesteps looks like: [0, 0, ..., 0(392 times), 300, 300, ..., 750, 750, ...]

# 4. Create time embeddings
t_emb = sinusoidal_embedding_1d(freq_dim=256, t=token_timesteps)  # [2, 1176, 256]
t = self.time_embedding(t_emb)  # [2, 1176, 3072]
t_mod = self.time_projection(t).unflatten(2, (6, 3072))
#       [2, 1176, 6, 3072]  (6 modulation parameters: shift_msa, scale_msa, gate_msa, etc.)

# 5. Embed context
context_emb = self.text_embedding(inputs['context'])  # [2, 129, 3072]

# Output
video_pre = {
    'tokens': [2, 1176, 3072],      # Patchified video tokens
    'freqs': [1176, 1, 128],         # 3D RoPE frequencies
    't_mod': [2, 1176, 6, 3072],    # Per-token timestep modulation
    'context': [2, 129, 3072],       # Embedded text context
    'context_mask': [2, 1176, 129],  # Attention mask for cross-attention
    ...
}
```

**Step 5: Action Expert Pre-DiT (action_dit.py lines 226-299)**
```python
action_pre = self.action_expert.pre_dit(
    action_tokens=action_latents,      # [2, 32, 7] (noisy actions)
    timestep=timestep_action,          # [2] (e.g., [500, 250])
    context=inputs['context'],         # [2, 129, 4096]
    context_mask=context_mask,         # [2, 129]
)

# Inside pre_dit():
# 1. Embed actions
tokens = self.action_encoder(action_latents)  # Linear(7→1024): [2, 32, 1024]

# 2. Create time embeddings
t_emb = sinusoidal_embedding_1d(freq_dim=256, t=timestep_action)  # [2, 256]
t = self.time_embedding(t_emb)  # [2, 1024]
t_mod = self.time_projection(t).unflatten(1, (6, 1024))
#       [2, 6, 1024]  (shared across all 32 action tokens)

# 3. Embed context
context_emb = self.text_embedding(context)  # [2, 129, 1024]

# Output
action_pre = {
    'tokens': [2, 32, 1024],        # Action tokens
    'freqs': [32, 1, 128],           # 1D RoPE frequencies
    't_mod': [2, 6, 1024],           # Timestep modulation (shared)
    'context': [2, 129, 1024],       # Embedded context
    'context_mask': [2, 129],        # Context attention mask
    ...
}
```

**Step 6: MoT Forward (mot.py lines 447-638)**
```python
# Build attention mask
attention_mask = self._build_mot_attention_mask(
    video_seq_len=1176,      # Number of video tokens
    action_seq_len=32,       # Number of action tokens
    video_tokens_per_frame=392,  # Tokens per frame (14×28)
)
# attention_mask shape: [1208, 1208]
# Structure:
#   [Video-to-Video causal] [Video-to-Action blocked]
#   [Action-to-FirstFrame]  [Action-to-Action full]

# Run 30 MoT layers
tokens_out = self.mot(
    embeds_all={
        'video': video_pre['tokens'],      # [2, 1176, 3072]
        'action': action_pre['tokens'],    # [2, 32, 1024]
    },
    attention_mask=attention_mask,         # [1208, 1208]
    freqs_all={
        'video': video_pre['freqs'],       # [1176, 1, 128]
        'action': action_pre['freqs'],     # [32, 1, 128]
    },
    t_mod_all={
        'video': video_pre['t_mod'],       # [2, 1176, 6, 3072]
        'action': action_pre['t_mod'],     # [2, 6, 1024]
    },
    context_all={...},  # Text context dicts
    detach_video_for_action=True,  # Optional knowledge insulation
)

# Output:
# tokens_out = {
#     'video': [2, 1176, 3072],  # Updated video tokens
#     'action': [2, 32, 1024],   # Updated action tokens
# }
```

**Step 7: Post-DiT & Loss (fastwam.py lines 603-643)**
```python
# Decode predictions
pred_video = self.video_expert.post_dit(tokens_out['video'], video_pre)
#            Unpatchify [2, 1176, 3072] → [2, 16, 3, 28, 56]

pred_action = self.action_expert.post_dit(tokens_out['action'], action_pre)
#            Linear [2, 32, 1024] → [2, 32, 7]

# Compute targets
target_video = self.train_video_scheduler.training_target(
    input_latents, noise_video, timestep_video
)  # [2, 16, 3, 28, 56]

target_action = self.train_action_scheduler.training_target(
    sample['action'], noise_action, timestep_action
)  # [2, 32, 7]

# VIDEO LOSS: Skip first frame!
pred_video_loss = pred_video[:, :, 1:]      # [2, 16, 2, 28, 56]
target_video_loss = target_video[:, :, 1:]  # [2, 16, 2, 28, 56]

loss_video_mse = F.mse_loss(pred_video_loss, target_video_loss, reduction='none')
#                 [2, 16, 2, 28, 56]
loss_video_mse = loss_video_mse.mean(dim=[1, 3, 4])  # [2, 2]

# Apply mask & weight
loss_video = (loss_video_mse * video_weight).mean()  # scalar

# ACTION LOSS: Full sequence
loss_action_mse = F.mse_loss(pred_action, target_action, reduction='none')
#                 [2, 32, 7]
loss_action_mse = loss_action_mse.mean(dim=[1, 2])  # [2]

loss_action = (loss_action_mse * action_weight).mean()  # scalar

# TOTAL LOSS
loss = loss_video + loss_action  # scalar
```

**Step 8: Backward & Optimize**
```python
loss.backward()  # Compute gradients

# Gradient flow:
# - Video loss updates: video_expert, action_expert (via query), mot
# - Action loss updates: action_expert, action query in mot, (mot only, video K/V detached)

optimizer.step()  # Update parameters
```

---

## Example 2: Debugging "First Frame Not Conditioned"

### Problem: Model ignores first frame, treats it like other frames

### Debugging Checklist

```python
# 1. Check: First frame latents are being extracted
print("First frame shape:", first_frame_latents.shape)  # Should be [B, C, 1, H, W]
assert first_frame_latents.shape[2] == 1  # Must be only 1 frame

# 2. Check: First frame is being frozen
print("Before freeze - latents[0, 0, 0:1, 0, 0]:", latents[0, 0, 0:1, 0, 0])
latents[:, :, 0:1] = first_frame_latents
print("After freeze - latents[0, 0, 0:1, 0, 0]:", latents[0, 0, 0:1, 0, 0])
# Should be exactly equal to first_frame_latents[0, 0, 0:1, 0, 0]

# 3. Check: Per-token timestep is correct
print("token_timesteps shape:", token_timesteps.shape)  # [B, T*tokens_per_frame]
print("First tokens timestep:", token_timesteps[:, :392].unique())  # Should be all 0
print("Other tokens timestep:", token_timestep[:, 392:].unique())  # Should be t_sample

# 4. Check: Loss excludes first frame
# In _compute_video_loss_per_sample():
print("pred_video shape before slice:", pred_video.shape)  # [B, C, T, H, W]
pred_video = pred_video[:, :, 1:]  # Skip first frame!
print("pred_video shape after slice:", pred_video.shape)  # [B, C, T-1, H, W]

# 5. Check: Attention mask blocks action-to-future-frames
mask = self._build_mot_attention_mask(...)
print("Action row of mask:")
print(mask[-32:, :392])   # Should be all True (can see first frame)
print(mask[-32:, 392:])   # Should be all False (can't see other frames)
```

---

## Example 3: Optimizing Inference Speed

### Use Case: Generating actions for real-time robot control

**SLOW WAY (Joint Inference):**
```python
# Both video and actions generated
pred_video, pred_action = model.infer_joint(
    input_image=image,
    context=prompt_embedding,
    num_steps=50,
)
# Time: ~30-50 seconds per inference
```

**FAST WAY (Action-Only Inference):**
```python
# Actions only - first frame is input, not output
pred_action = model.infer_action(
    input_image=image,
    context=prompt_embedding,
    num_steps=50,
)
# Time: ~2-5 seconds per inference (10-20x faster!)

# How it works internally:
# 1. Encode first frame to latents [1, C, 1, h, w]
# 2. Pre-compute video expert output
# 3. Cache video K/V from first 15 layers
# 4. For each action denoising step:
#    - Only compute action expert
#    - Reuse cached video K/V from step 3
#    - Skip video expert computation entirely!
```

---

## Example 4: Understanding Attention Pattern

### Scenario: What can each token attend to?

```
Token Population:
- Frame 0: 392 video tokens (patchified first frame)
- Frame 1: 392 video tokens (patchified frame 1)
- Frame 2: 392 video tokens (patchified frame 2)
- Actions: 32 action tokens

Query Position → What it can see (Key/Value):

Frame 0 Video Token (e.g., token 0):
  ✓ Can see: Frame 0 (itself and other tokens in frame 0)
  ✗ Cannot see: Frames 1, 2 (future frames - causal mask)
  ✗ Cannot see: Action tokens (video-to-action blocked)

Frame 1 Video Token (e.g., token 500):
  ✓ Can see: Frame 0 (all tokens, clean condition)
  ✓ Can see: Frame 1 (all tokens, self and earlier in same frame)
  ✗ Cannot see: Frame 2 (future frame - causal mask)
  ✗ Cannot see: Action tokens (video-to-action blocked)

Action Token (e.g., action token 0):
  ✓ Can see: Frame 0 only! (tokens 0-391, the first-frame-only mask)
  ✗ Cannot see: Frames 1, 2 (future frames blocked)
  ✓ Can see: Other action tokens (self-attention)

Why this pattern?
- Video is causal: Can't peek into future you're supposed to generate
- Actions see only first frame: Can't leak information about future video
- Together: Clean separation of concerns between video generation and action prediction
```

---

## Example 5: Adding Extra Conditioning Images (NOT IMPLEMENTED)

### Scenario: You want to condition on 3 key frames instead of 1

### Steps to Implement:

```python
# Step 1: Encode additional images to latent space
extra_frames = [frame_2, frame_5, frame_8]  # 3 conditioning frames
extra_latents = []
for frame in extra_frames:
    # Use same VAE/visual encoder as first frame
    frame_latent = self.visual_encoder.encode(frame.unsqueeze(0).unsqueeze(0))
    extra_latents.append(frame_latent)  # Each: [1, C, 1, h_lat, w_lat]

# Step 2: Create separate token stream for conditioning frames
# Concatenate all conditioning frames: [3, C, 1, h_lat, w_lat]
conditioning_latents = torch.cat(extra_latents, dim=2)  # [1, C, 3, h_lat, w_lat]

# Patchify: [1, C, 3, h_lat, w_lat] → [1, 3*tokens_per_frame, D]
#  = [1, 1176, 3072]  (3 frames × 392 tokens/frame)

# Step 3: Extend attention mask to include conditioning
# Current mask: [1176+32, 1176+32] = video + action
# New mask: [1176+1176+32, 1176+1176+32] = cond + video + action

# Conditioning tokens:
#   - Can attend to: all video + action (broadcast condition)
#   - Cannot attend to: future frames (causal for generation)

# Step 4: Modify context building
# Instead of just text context, add conditioning embeddings:
context = torch.cat([text_context, conditioning_embeddings], dim=1)

# Step 5: Adjust loss
# Only predict non-conditioned frames:
pred_video = pred_video[:, :, 3:]  # Skip first 3 frames (conditioned)
target_video = target_video[:, :, 3:]  # Only last 6 frames are targets
```

---

## Example 6: Understanding Knowledge Insulation

### Scenario: Why detach video K/V when computing action loss?

**WITHOUT Detach (Problem):**
```python
# Forward pass (same for both)
k_video = self.video_expert.wk(video_tokens)
v_video = self.video_expert.wv(video_tokens)
k_action = self.action_expert.wk(action_tokens)
v_action = self.action_expert.wv(action_tokens)

# Attention (concatenated)
k_cat = [k_video ; k_action]
v_cat = [v_video ; v_action]
output = attention(q_action, k_cat, v_cat)  # Action queries see video

# Backward with action_loss
action_loss.backward()

# Problem: Gradients flow through action_loss → output → k_video
#          This means action_loss is TRAINING the video expert!
# Result: Video expert is trained by BOTH video_loss and action_loss
#         Action loss might corrupt video generation quality
```

**WITH Detach (Solution):**
```python
# Forward pass
k_video = self.video_expert.wk(video_tokens)
v_video = self.video_expert.wv(video_tokens)
k_action = self.action_expert.wk(action_tokens)
v_action = self.action_expert.wv(action_tokens)

# Attention (video normal, action with detached video)
video_output = attention(q_video, [k_video ; k_action], [v_video ; v_action])
k_cat_detached = [k_video.detach() ; k_action]  # Detach video K!
v_cat_detached = [v_video.detach() ; v_action]  # Detach video V!
action_output = attention(q_action, k_cat_detached, v_cat_detached)

# Backward with both losses
(video_loss + action_loss).backward()

# Result with video path:
#   video_loss → video_output → k_video → gradients flow ✓
#   (video_loss trains video expert)

# Result with action path:
#   action_loss → action_output → k_video.detach() → NO gradients ✗
#   (action_loss does NOT train video expert)

# Benefit: Video expert only trained by video_loss
#          Action loss doesn't interfere with video generation
```

