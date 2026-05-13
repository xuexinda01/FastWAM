# FastWAM Architecture: Visual Diagrams

## 1. End-to-End Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE TRAINING PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│   DataLoader Sample         │
├─────────────────────────────┤
│ video: [B, 3, T, H, W]      │
│ action: [B, T-1, 7]         │
│ context: [B, L, 4096]       │
│ proprio: [B, 8]             │
└──────────────┬──────────────┘
               │
        ┌──────▼──────┐
        │  VAE Encode │ (frozen)
        │  latents    │
        └──────┬──────┘
               │
    ┌──────────▼──────────┐
    │ latents: [B,C,T,H,W]│
    │ Extract first frame │
    │ first_frame_latents │
    └──────────┬──────────┘
               │
        ┌──────▼──────────────────────────┐
        │ Add Gaussian Noise              │
        │ latents_noised = √(ᾱ)x + √(1-ᾱ)ε  │
        └──────┬───────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ FREEZE First Frame              │
        │ latents_noised[:, :, 0:1]       │
        │   = first_frame_latents         │
        └──────┬───────────────────────────┘
               │
    ┌──────────▼──────────────┐
    │ Video Expert pre_dit()  │
    │ - Patchify              │
    │ - Per-token timestep    │
    │   t[0] = 0, t[1:] = t   │
    │ - Time modulation       │
    │ - Context embedding     │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────┐
    │ Action Expert pre_dit() │
    │ - Action embedding      │
    │ - Time modulation       │
    │ - Context embedding     │
    └──────────┬──────────────┘
               │
    ┌──────────▼─────────────────────────┐
    │ MoT Forward (30 Layers)             │
    ├─────────────────────────────────────┤
    │ Layer i:                            │
    │  1. Video Q/K/V (with RoPE)         │
    │  2. Action Q/K/V (with RoPE)        │
    │  3. Concatenate: [Q_v; Q_a], ...    │
    │  4. Mixed Attention + Mask          │
    │     - Actions see only frame 0      │
    │     - Video sees all + action       │
    │  5. Split outputs                   │
    │  6. Cross-Attention (text)          │
    │  7. FFN                             │
    └──────────┬──────────────────────────┘
               │
    ┌──────────▼──────────────┐
    │ Video Expert post_dit() │
    │ - Unpatchify            │
    │ pred_video: [B,C,T,H,W] │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────┐
    │ Action Expert post_dit()│
    │ - Action head           │
    │ pred_action: [B,T-1,7]  │
    └──────────┬──────────────┘
               │
    ┌──────────▼──────────────────────────┐
    │ Compute Losses                      │
    ├─────────────────────────────────────┤
    │ Video Loss:                         │
    │  - Skip first frame                 │
    │  - pred_video = pred_video[:,:,1:]  │
    │  - MSE(pred, target)                │
    │  - × timestep_weight                │
    │                                     │
    │ Action Loss:                        │
    │  - MSE(pred_action, target_action)  │
    │  - × timestep_weight                │
    └──────────┬──────────────────────────┘
               │
    ┌──────────▼──────────────┐
    │ Backward + Optimize     │
    │ gradient_checkpointing? │
    │ loss.backward()         │
    │ optimizer.step()        │
    └──────────────────────────┘
```

---

## 2. First Frame Conditioning Mechanism

```
┌───────────────────────────────────────────────────────────────────────────┐
│                    FROZEN FIRST FRAME CONDITIONING                        │
└───────────────────────────────────────────────────────────────────────────┘

Input Video: [B, 3, T_orig, H, W]
│
├─ Frame 0 (condition)
├─ Frame 1 (target)
├─ Frame 2 (target)
└─ ... Frame T-1 (target)

        │
        ▼ VAE Encode
        
Latents: [B, z_dim, T_lat, H_lat, W_lat]
│
├─ Frame 0 latent [B, z_dim, 1, H_lat, W_lat] ◄─ Extract & Save
├─ Frame 1 latent [B, z_dim, 1, H_lat, W_lat]
├─ Frame 2 latent [B, z_dim, 1, H_lat, W_lat]
└─ ... Frame T-1 latent

        │
        ▼ Add Noise (Training)
        
Noised Latents (Before Freezing):
│
├─ Frame 0: noisy_latent_0 = √ᾱ·x₀ + √(1-ᾱ)·ε₀  ◄─ Will be replaced
├─ Frame 1: noisy_latent_1 = √ᾱ·x₁ + √(1-ᾱ)·ε₁  ◄─ Prediction target
└─ ... all frames noised

        │
        ▼ FREEZE (Replace first frame)
        
Noised Latents (After Freezing):
│
├─ Frame 0: x₀ (CLEAN) ◄─────── Frozen from earlier
├─ Frame 1: √ᾱ·x₁ + √(1-ᾱ)·ε₁  ◄─ Still noisy
└─ ... all others still noisy

        │
        ▼ DiT Pre-processing
        
Token Timesteps:
│
├─ Frame 0 tokens: t = 0 ◄─────── Clean signal (t=0 means "fully denoised")
├─ Frame 1 tokens: t = t_sample  ◄─ Noisy signal (needs denoising)
└─ ... all others

        │
        ▼ Through 30 MoT Layers
        
During Forward Pass:
│
├─ Frame 0 tokens: See t_mod for t=0 (minimal noise prediction signal)
├─ Frame 1+ tokens: See t_mod for t_sample (strong noise prediction signal)
└─ Model learns to preserve frame 0 and denoise frames 1+

        │
        ▼ Post-DiT Unpatchify
        
Predictions:
│
├─ pred_latent_0: Not used for loss (frozen)
├─ pred_latent_1: Target for MSE loss
└─ ... all others

        │
        ▼ Loss Computation
        
loss = MSE(pred_latent[1:], target_latent[1:]) × weight(t)

Only predicting frames AFTER the frozen first frame!
```

---

## 3. MoT Mixed Attention Mechanism

```
┌──────────────────────────────────────────────────────────────────────────┐
│              MIXTURE OF TRANSFORMERS: MIXED ATTENTION                    │
└──────────────────────────────────────────────────────────────────────────┘

For Layer i (0..29):

Step 1: Independent Expert Pre-Attention
┌────────────────────┐         ┌────────────────────┐
│  Video Expert      │         │  Action Expert     │
├────────────────────┤         ├────────────────────┤
│ x_v: [B,1176,3072] │         │ x_a: [B,32,1024]   │
│                    │         │                    │
│ Q_v = RoPE(RMSNorm │         │ Q_a = RoPE(RMSNorm │
│        (Wq·x_v))   │         │        (Wq·x_a))   │
│ K_v = ...          │         │ K_a = ...          │
│ V_v = Wv·x_v       │         │ V_a = Wv·x_a       │
└────────────────────┘         └────────────────────┘
        │                               │
        ▼                               ▼
     Q_v [B,1176,3072]          Q_a [B,32,3072]
     K_v [B,1176,3072]          K_a [B,32,3072]
     V_v [B,1176,3072]          V_a [B,32,3072]

Step 2: Concatenate
     Q_cat = [Q_v ; Q_a]  →  [B, 1208, 3072]
     K_cat = [K_v ; K_a]  →  [B, 1208, 3072]
     V_cat = [V_v ; V_a]  →  [B, 1208, 3072]

Step 3: Mixed Attention with Mask
     attn_weights = softmax(Q_cat·K_cat^T/√d + MASK)
     
     MASK structure [1208 × 1208]:
     ┌──────────────────────┬────────────┐
     │ Video-to-Video       │ Video-Auth │
     │ (causal mask)        │  (blocked) │
     │ [1176, 1176]         │ [1176, 32] │
     ├──────────────────────┼────────────┤
     │ Action-to-Video      │ Action-Auth│
     │ (first frame only)    │ (all)      │
     │ [32, 1176→392 first] │ [32, 32]   │
     └──────────────────────┴────────────┘
     
     mixed = attn_weights · V_cat  →  [B, 1208, 3072]

Step 4: Optional Knowledge Insulation
     If detach_video_for_action:
         # Recompute action part with detached video
         k_cat_detached = [K_v.detach(); K_a]
         v_cat_detached = [V_v.detach(); V_a]
         action_attn = softmax(Q_a·k_cat_detached^T) · v_cat_detached

Step 5: Split Outputs
     mixed_video = mixed[:1176]     →  [B, 1176, 3072]
     mixed_action = mixed[1176:]    →  [B, 32, 3072]

Step 6: Project Back to Expert Dimensions
     ┌────────────────────┐         ┌────────────────────┐
     │  Video Expert      │         │  Action Expert     │
     ├────────────────────┤         ├────────────────────┤
     │ x_v' = Wo(mixed)   │         │ x_a' = Wo(mixed)   │
     │ x_v' [B,1176,3072] │         │ x_a' [B,32,3072]   │
     │                    │         │                    │
     │ Cross-Attention    │         │ Cross-Attention    │
     │ + Text Context     │         │ + Text Context     │
     │ [B,1176,3072]      │         │ [B,32,1024]        │
     │                    │         │                    │
     │ FFN                │         │ FFN                │
     │ [B,1176,3072]      │         │ [B,32,1024]        │
     └────────────────────┘         └────────────────────┘
           │                              │
           ▼                              ▼
     x_v_out [B,1176,3072]         x_a_out [B,32,1024]
     (to next layer)               (to next layer)
```

---

## 4. Attention Pattern Visualization

```
┌──────────────────────────────────────────────────────────────┐
│         ATTENTION CONNECTIVITY AFTER MASK APPLICATION       │
└──────────────────────────────────────────────────────────────┘

Frame 0: [■ ■ ■ ■] (392 tokens from patchified frame 0)
Frame 1: [■ ■ ■ ■] (392 tokens from patchified frame 1)
Frame 2: [■ ■ ■ ■] (392 tokens from patchified frame 2)
Action:  [● ● ● ●] (32 action tokens)

Query (Can attend to):      Key/Value (What it sees):
────────────────────────────────────────────────────────
Frame 0: [■ ■ ■ ■]    →    [■ ■ ■ ■ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗]
                            self    F1   F2   Action (blocked)

Frame 1: [■ ■ ■ ■]    →    [■ ■ ■ ■ ■ ■ ■ ■ ✗ ✗ ✗ ✗]
                            F0    F1   F2   Action (blocked)

Frame 2: [■ ■ ■ ■]    →    [■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■]
                            F0    F1   F2   Action (blocked)

Action:  [● ● ● ●]    →    [■ ■ ■ ■ ✗ ✗ ✗ ✗ ● ● ● ●]
                            F0only  F1✗  F2✗   self

Meaning:
  ■  = Video token can attend normally
  ✗  = Blocked (cannot attend)
  ●  = Action token
  F0 = Frame 0 (first frame, frozen condition)
  F1 = Frame 1 (future frame)
  F2 = Frame 2 (future frame)

Key Insight:
  - Videos never see action tokens (prevents action from corrupting video)
  - Actions only see Frame 0 (prevents information leakage about future)
  - Videos see all past/current frames (causal)
  - Actions see themselves (action-to-action full attention)
```

---

## 5. Per-Token Timestep Assignment

```
┌──────────────────────────────────────────────────────────┐
│           PER-TOKEN TIMESTEP FOR CONDITIONING           │
└──────────────────────────────────────────────────────────┘

Sample timestep: t_sample = 500 (out of 1000 max)

Frame Structure:
  Frame 0: [token_0_0] [token_0_1] [token_0_2] ... [token_0_391]
  Frame 1: [token_1_0] [token_1_1] [token_1_2] ... [token_1_391]
  Frame 2: [token_2_0] [token_2_1] [token_2_2] ... [token_2_391]

Token Timestep Assignment:
  token_0_*  →  t = 0      ◄─ Frame 0 timestep = 0 (CLEAN/REFERENCE)
  token_1_*  →  t = 500    ◄─ Frame 1 timestep = t_sample (NOISY/TARGET)
  token_2_*  →  t = 500    ◄─ Frame 2 timestep = t_sample (NOISY/TARGET)

Semantic Meaning:
  t = 0   = "This frame is fully denoised, don't predict noise for it"
  t = 500 = "This frame is very noisy, predict how much noise to remove"

Time Modulation (t_mod):
  Per-token sinusoidal position encoding + MLP projection
  
  Shape: [B, num_tokens, 6, hidden_dim]
         where 6 = (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
  
  For t=0 tokens:   t_mod emphasizes the "clean" modulation
  For t=500 tokens: t_mod emphasizes the "noisy" modulation
  
Example (simplified):
  token_0_0: t_mod = sinusoidal(256, 0) → Linear → [shift=0, scale=1, gate=1]
                                           (minimal modification)
  
  token_1_0: t_mod = sinusoidal(256, 500) → Linear → [shift=.5, scale=.7, gate=.3]
                                             (significant modification)

The model learns:
  - For t=0: Use shift/scale/gate near identity (preserve clean frame)
  - For t>0: Use shift/scale/gate to predict denoising direction
```

---

## 6. Knowledge Insulation in Action Loss

```
┌──────────────────────────────────────────────────────────┐
│        KNOWLEDGE INSULATION: ACTION LOSS PREVENTS         │
│     ACTION GRADIENTS FROM CORRUPTING VIDEO PARAMETERS     │
└──────────────────────────────────────────────────────────┘

During Training:

Step 1: Mixed Attention Forward
  ┌────────────────────────────────────────────┐
  │ Build Q/K/V from each expert               │
  │ Q_video, K_video, V_video (trainable)      │
  │ Q_action, K_action, V_action (trainable)   │
  └────────────────────────────────────────────┘

Step 2a: Video Attention (Normal)
  video_output = attention(
      Q=Q_video,
      K=[K_video ; K_action],      ◄─ Full K
      V=[V_video ; V_action],      ◄─ Full V
      mask=video_mask
  )
  Gradients flow: video_loss ← video_output ← K_action, V_action
                                                ↓ (if detach_video_for_action=True)
                                            Will be DETACHED in action path

Step 2b: Action Attention (With Detach)
  IF detach_video_for_action:
      # Detach video K/V to block action gradients
      action_output = attention(
          Q=Q_action,
          K=[K_video.detach() ; K_action],   ◄─ Video K/V detached
          V=[V_video.detach() ; V_action],
          mask=action_mask
      )
  ELSE:
      # Normal (allows gradients through)
      action_output = attention(
          Q=Q_action,
          K=[K_video ; K_action],
          V=[V_video ; V_action],
          mask=action_mask
      )

Step 3: Compute Losses
  video_loss = MSE(pred_video, target_video)
  action_loss = MSE(pred_action, target_action)

Step 4: Backward with Knowledge Insulation
  WITHOUT detach (action_loss can corrupt video):
      video_grad ← video_loss
      action_grad ← action_loss → video_loss (propagates through K_video, V_video)
      PROBLEM: Action gradients modify video parameters!

  WITH detach (action_loss isolated):
      video_grad ← video_loss
      action_grad ← action_loss → K_video.detach() (gradient blocked)
      ✓ Action loss doesn't modify video parameters
      ✓ Only video loss trains video expert

Result:
  - Video Expert: Trained by video_loss + action_loss (through video query)
  - Action Expert: Trained by action_loss only (through Q_action)
  - Video K/V: Only modified by video_loss gradients (blocked from action)
  
Benefit:
  - Prevents action task from interfering with video generation
  - Allows independent tuning of loss weights
  - Ensures video expert focuses on video quality
```

---

## 7. Inference Modes Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│            INFERENCE MODES: JOINT vs ACTION-ONLY                 │
└──────────────────────────────────────────────────────────────────┘

MODE 1: Joint Inference (infer_joint)
╔════════════════════════════════════════════════════════════════╗
║ Generate both video AND actions together                       ║
╠════════════════════════════════════════════════════════════════╣
║ Input:  first_frame [1,3,H,W]  context [1,L,D]                ║
║                                                                ║
║ for each denoising step:                                       ║
║   1. Encode first frame to latents                             ║
║   2. Add noise to all frames + actions                         ║
║   3. Run through MoT (shared attention)                        ║
║   4. Predict noise for both video and action                  ║
║   5. Denoise: latents -= pred_noise                           ║
║   6. Re-freeze first frame: latents[:,0] = first_frame        ║
║   7. Repeat                                                    ║
║                                                                ║
║ Computation: O(steps × video_tokens × action_tokens)          ║
║ Speed: SLOW (20-50 steps × 2 experts)                         ║
╚════════════════════════════════════════════════════════════════╝

MODE 2: Action-Only Inference (infer_action) - OPTIMIZED
╔════════════════════════════════════════════════════════════════╗
║ Generate actions only, reuse first frame for condition        ║
╠════════════════════════════════════════════════════════════════╣
║ Input:  first_frame [1,3,H,W]  context [1,L,D]                ║
║                                                                ║
║ Setup (one-time):                                              ║
║   1. Encode first frame to latents [1,C,1,h,w]               ║
║   2. Run through video expert pre_dit                         ║
║   3. Run first 15 layers to compute K/V_video cache           ║
║   4. Store: kv_cache = {k_0, v_0, k_1, v_1, ..., k_14, v_14}║
║                                                                ║
║ for each denoising step:                                       ║
║   1. Add noise to actions                                     ║
║   2. Action expert pre_dit                                    ║
║   3. Mixed attention using CACHED video K/V                  ║
║      - Skip video expert computation entirely!               ║
║      - Just compute action output                             ║
║   4. Remaining layers (15-29)                                 ║
║   5. Action post_dit → pred_action                           ║
║   6. Denoise: action_latents -= pred_noise                   ║
║                                                                ║
║ Computation: O(steps × action_tokens) [video cached]          ║
║ Speed: FAST (20-50 steps × 1 expert, video cached)            ║
║ Speedup: 10-20x faster than joint                             ║
╚════════════════════════════════════════════════════════════════╝

When to use each:
  ✓ use infer_joint() when you want both video AND actions
  ✓ use infer_action() when you only want actions (RECOMMENDED)
    - Much faster
    - Same quality (video is conditioning input, not output)
```

---

## 8. Loss Computation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION DETAILS                      │
└──────────────────────────────────────────────────────────────────┘

Prediction Outputs from MoT:
  pred_video [B, C, T_lat, H_lat, W_lat]
  pred_action [B, action_horizon, action_dim]

Target Outputs (From training_target):
  target_video [B, C, T_lat, H_lat, W_lat]
  target_action [B, action_horizon, action_dim]

═══════════════════════════════════════════════════════════════════

VIDEO LOSS COMPUTATION:

Step 1: Skip First Frame
  pred_video_loss = pred_video[:, :, 1:]       # [B, C, T_lat-1, H, W]
  target_video_loss = target_video[:, :, 1:]   # [B, C, T_lat-1, H, W]
  
  ✓ First frame NOT included in loss
  ✗ Cannot modify first frame during training

Step 2: Compute MSE
  loss_per_token = (pred_video_loss - target_video_loss) ** 2
  loss_per_token = loss_per_token.mean(dim=[1,3,4])  # Average over C, H, W
  → shape: [B, T_lat-1]

Step 3: Apply Frame Mask (if available)
  if image_is_pad is not None:
      valid = (~image_is_pad[:, 1:])  # Exclude first frame
      loss_per_sample = (loss_per_token * valid).sum(dim=1) / valid.sum(dim=1)
  else:
      loss_per_sample = loss_per_token.mean(dim=1)  # [B]

Step 4: Apply Timestep Weight
  weight_video = scheduler.training_weight(t_sample)
  # Higher weight for t closer to T (more noise harder to denoise)
  loss_video = (loss_per_sample * weight_video).mean()  # scalar

═══════════════════════════════════════════════════════════════════

ACTION LOSS COMPUTATION:

Step 1: Compute MSE
  loss_action = (pred_action - target_action) ** 2
  loss_action = loss_action.mean(dim=[1,2])  # Average over action_horizon, action_dim
  → shape: [B]

Step 2: Apply Action Mask (if available)
  if action_is_pad is not None:
      valid = (~action_is_pad)
      loss_action_per_sample = (loss_action * valid).sum(dim=1) / valid.sum(dim=1)
  else:
      loss_action_per_sample = loss_action

Step 3: Apply Timestep Weight
  weight_action = scheduler.training_weight(t_sample)
  loss_action = (loss_action_per_sample * weight_action).mean()  # scalar

═══════════════════════════════════════════════════════════════════

TOTAL LOSS:

  loss_total = λ_video * loss_video + λ_action * loss_action
  
  where:
    λ_video = weight for video loss (default 1.0)
    λ_action = weight for action loss (default 1.0)

═══════════════════════════════════════════════════════════════════

Backward Pass:

  loss_total.backward()
  
  Gradient Flow:
    → video_loss gradients
        ↓ through video expert parameters
        ↓ through MoT parameters
        ↓ through video_pre state
    
    → action_loss gradients
        ↓ through action expert parameters
        ↓ through MoT parameters (with optional detach)
        ↓ through action_pre state

Key Point: First frame frozen in loss doesn't mean no gradients through it!
  - First frame latents don't get updated by video_loss (frozen)
  - But first frame embeddings CAN receive action_loss gradients
  - This creates one-way information flow: first_frame → action
```

