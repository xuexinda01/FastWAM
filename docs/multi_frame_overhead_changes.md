# Multi-Frame History + Overhead Conditioning - Code Changes

## Date: 2026-05-10

## Overview
Modified FastWAM architecture to support multi-frame history conditioning + overhead camera view for better navigation action prediction.

**New Architecture:**
- Input: 9 frames 125cm_0deg (8 uniformly sampled history + current) + 1 frame 125cm_30deg (overhead) + text
- Output: 8 future video frames (0deg) + 32 action waypoints
- Total VAE input: 17 frames (T%4==1 ✓) → 5 latent frames (3 frozen condition + 2 generated)

---

## Files Modified

### 1. `src/fastwam/datasets/lerobot/nav_video_dataset.py` — **COMPLETE REWRITE**

**Changes:**
- Single camera (125cm_0deg) for main video instead of dual-camera horizontal concat
- Separate overhead frame (125cm_30deg) as independent conditioning input
- History sampling: 8 frames uniformly sampled from [0, current_idx-1] + current frame = 9 condition frames
- Future: 8 frames with stride=4 after current frame for video generation
- Terminal oversampling: last 20% of trajectory is oversampled 3x for learning stop behavior
- Output `video` shape: [C, 17, H, W] (was [C, 9, 224, 448])
- Output `overhead` shape: [C, H, W] (new field)
- Output `video_size`: [224, 224] (was [224, 448] for dual camera)
- Added `n_cond_frames` to sample dict

### 2. `configs/data/nav_vln.yaml`

**Changes:**
- `video_size: [224, 224]` (was [224, 448])
- `concat_multi_camera: "none"` (was "horizontal")  
- Added: `n_history_frames: 8`
- Added: `n_future_video_frames: 8`
- Added: `terminal_oversample_ratio: 3.0`

### 3. `src/fastwam/models/wan22/wan_video_dit.py`

**Changes:**
- `patch_embedding` input channels: `in_dim * 2` (96 instead of 48) to accept overhead channel concat
- Added `overhead_conditioning = True` flag
- Added `initialize_overhead_channels_from_pretrained()` method for loading old checkpoints (zero-init new channels)
- `pre_dit()`: Added `n_cond_latent_frames` parameter; sets timestep=0 for ALL condition latent frames (was only frame 0)

### 4. `src/fastwam/models/wan22/fastwam.py`

**Changes in `build_inputs()`:**
- Computes `n_cond_latent_frames = (n_cond_frames + 3) // 4` (=3 for 9 condition frames)
- Extracts `condition_latents` (first 3 latent frames, frozen during diffusion)
- Encodes overhead frame through VAE separately → `overhead_latent` [B, z_dim, 1, H', W']
- Returns new keys: `condition_latents`, `n_cond_latent_frames`, `overhead_latent`
- Removed old `first_frame_latents` logic

**Changes in `training_loss()`:**
- Freezes ALL condition latent frames: `latents[:, :, :n_cond_latent_frames] = condition_latents`
- Broadcasts overhead latent to full temporal extent and concatenates along channel dim
- Passes `latents_with_overhead` [B, 2*z_dim, T_lat, H', W'] to video expert
- Passes `n_cond_latent_frames` to video_expert.pre_dit() and attention mask
- Video loss computed only on generated frames (strips first n_cond_latent_frames)

**Changes in `_build_mot_attention_mask()`:**
- Added `n_cond_latent_frames` parameter (default=3)
- Action tokens now attend to ALL condition frame tokens (was only first frame)
- `cond_tokens = n_cond_latent_frames * tokens_per_frame`

---

## Architecture Diagram

```
INPUT (dataset):
  video: [B, 3, 17, 224, 224]   ← 9 history(0deg) + 8 future(0deg)
  overhead: [B, 3, 224, 224]     ← 1 frame (30deg) at current timestep
  action: [B, 32, 3]             ← relative (x, y, theta) waypoints

VAE ENCODING:
  video → [B, 48, 5, 28, 28]     ← 17 frames → 5 latent frames
  overhead → [B, 48, 1, 28, 28]  ← 1 frame → 1 latent frame

LATENT PREPARATION:
  Freeze condition: latents[:, :, 0:3] = condition_latents (no noise)
  Broadcast overhead: [B, 48, 1, 28, 28] → [B, 48, 5, 28, 28]
  Concat: [B, 48+48, 5, 28, 28] = [B, 96, 5, 28, 28]

MODEL (WanVideoDiT):
  patch_embedding: Conv3d(96 → 3072, kernel=[1,2,2], stride=[1,2,2])
  → tokens: [B, 5*14*14, 3072] = [B, 980, 3072]
  Timestep: condition frames get t=0, generated frames get sampled t

MoT ATTENTION:
  Video expert: group-causal attention (frame i sees frames ≤ i)
  Action expert: attends to condition frame tokens (frames 0-2 = 588 tokens)

OUTPUT:
  Video loss: MSE on latent frames 3-4 only (generated)
  Action loss: MSE on 32-step waypoint prediction
```

---

## Weight Initialization (Loading Pretrained)

When loading from old checkpoint (48-ch patch_embedding):
```python
model.video_expert.initialize_overhead_channels_from_pretrained(old_state_dict)
# Result: first 48 channels = pretrained weights, channels 48-95 = zeros
# Model starts from pretrained behavior (overhead has no effect initially)
```

---

## Key Design Rationale

1. **Channel concat for overhead** (not cross-attention): Simple, proven in SVD, gives overhead info to ALL tokens at ALL positions
2. **Zero-init overhead channels**: Training starts from pretrained performance, gradually learns to use overhead
3. **3 frozen latent frames**: Encodes full 9-frame history, provides rich trajectory context
4. **Action attends to ALL condition frames**: Action expert sees full motion history (not just current frame)
5. **Terminal oversampling 3x**: Forces model to learn "stop at goal" behavior

---

## Stop Prediction Head (Optional) — Added 2026-05-11

### Overview
Added an optional explicit stop prediction head to the action expert. When enabled (`predict_stop: true`), the model outputs a binary stop signal at each of the 32 action steps, supervised by `action_is_pad` (padded steps = past trajectory end = should stop).

Two modes available:
- **Implicit stop** (`predict_stop: false`): Model learns to output near-zero actions at trajectory end. Inference uses action magnitude threshold.
- **Explicit stop** (`predict_stop: true`): Separate binary head predicts stop probability. Inference uses sigmoid threshold.

### Files Modified

#### `src/fastwam/models/wan22/action_dit.py`
- Added `StopHead` class (lines 32-44): Binary classification head with time modulation (matches ActionHead design)
- `ActionDiT.__init__`: Added `predict_stop` parameter, conditionally creates `self.stop_head`
- `ActionDiT.post_dit()`: Returns `Dict[str, Tensor]` instead of plain Tensor
  - `"action"`: [B, T, action_dim] — action noise prediction (always)
  - `"stop"`: [B, T, 1] — stop logits (only if predict_stop=True)
- `ACTION_BACKBONE_SKIP_PREFIXES` includes `"stop_head."` — stop head is not loaded from backbone pretrained weights

#### `src/fastwam/models/wan22/fastwam.py`
- `training_loss()`: Added stop loss computation after action loss:
  ```python
  if "stop" in pred_action_dict and action_is_pad is not None:
      stop_logits = pred_action_dict["stop"].squeeze(-1)  # [B, T]
      stop_labels = action_is_pad.float()  # padded = goal reached = stop
      loss_stop = F.binary_cross_entropy_with_logits(stop_logits, stop_labels)
      loss_total += loss_stop
  ```
- `loss_dict` includes `"loss_stop"` when active
- All `action_expert.post_dit()` call sites updated to handle Dict return

#### `configs/model/fastwam_nav.yaml`
- Added `predict_stop: true` to `action_dit_config`

### Architecture

```
Action Expert Hidden States [B, 32, 1024]
    │
    ├── self.head (Linear) → action noise pred [B, 32, 3]
    │
    └── self.stop_head (StopHead with time modulation) → stop logits [B, 32, 1]
         │
         └── BCE Loss with action_is_pad as labels
```

### Supervision Signal
- `action_is_pad[t] = True` → frame t is beyond trajectory end → stop_label = 1.0
- `action_is_pad[t] = False` → frame t is within trajectory → stop_label = 0.0

### Inference Usage
```python
pred_action_dict = model.action_expert.post_dit(tokens, pre_state)
actions = pred_action_dict["action"]  # [B, 32, 3]
if "stop" in pred_action_dict:
    stop_probs = torch.sigmoid(pred_action_dict["stop"])  # [B, 32, 1]
    # Stop when probability > 0.5
    stop_step = (stop_probs.squeeze(-1) > 0.5).float().argmax(dim=1)
```

### Backward Compatibility
- `predict_stop: false` (default) → no StopHead created, post_dit returns dict with only "action" key
- Existing checkpoints load without issue (stop_head weights simply missing → randomly initialized)
- Training without stop head: loss_dict won't contain "loss_stop"

### Prediction Steps Discussion
Current: 32 action steps. Could reduce to fewer (e.g., 10) for faster inference, but this would require:
- Changing `num_frames` and temporal structure
- Retraining from scratch
- Recommendation: keep 32 for now, consider reducing later based on eval results

---

## Action Dim 3→4: Integrated Moving Flag — Added 2026-05-11

### Overview
Changed action representation from `[x, y, theta]` (3D) to `[x, y, theta, moving_flag]` (4D).
- `moving_flag = 1.0`: agent is still moving (within trajectory)
- `moving_flag = 0.0`: agent has reached goal / should stop (beyond trajectory end)

The stop signal is now **part of the diffusion action output**, supervised jointly with the waypoints.

**Note**: Stop Head (predict_stop) is disabled (`predict_stop: false`) in favor of this simpler integrated approach.

### Files Modified

#### `src/fastwam/datasets/lerobot/nav_video_dataset.py`
- `_get()` method: Appends 4th dimension to action tensor
  ```python
  moving_flag = (~action_is_pad).astype(np.float32).reshape(-1, 1)  # [32, 1]
  actions_with_flag = np.concatenate([actions, moving_flag], axis=1)  # [32, 4]
  ```
- Output `action` shape: `[32, 4]` (was `[32, 3]`)

#### `configs/model/fastwam_nav.yaml`
- `action_dit_config.action_dim: 4` (was 3)
- `predict_stop: false` (stop head disabled)

#### `src/fastwam/models/wan22/fastwam.py`
- `training_loss()`: Removed `action_is_pad` masking from action loss
  - All 32 steps now contribute to loss (including stopped steps)
  - The model must learn to predict `moving_flag=0` for stopped steps
  - Previously: padded steps were excluded from loss (model never saw stop signal)

### Semantic of 4th Dimension

| Step within trajectory | x | y | theta | moving_flag |
|---|---|---|---|---|
| Normal movement | rel_x | rel_y | rel_theta | **1.0** |
| Beyond trajectory end | last_x (repeated) | last_y | last_theta | **0.0** |

### Inference Usage
```python
pred_actions = model.infer(...)  # [32, 4]
for t in range(32):
    x, y, theta, moving = pred_actions[t]
    if moving < 0.5:  # threshold
        print(f"Stop at step {t}")
        break
    execute_action(x, y, theta)
```

### Why Not a Separate Head?
- Simpler: no extra module, no extra loss term
- The diffusion process naturally handles continuous [0, 1] values
- The moving_flag is highly correlated with (x,y,theta) magnitude — makes sense to predict jointly
- During denoising, all 4 dims converge together — the model sees the full picture

### Backward Compatibility
- Old checkpoints (action_dim=3) won't load into new model (action_dim=4) without adaptation
- This is a breaking change — requires training from scratch with the new action dim

---

## Standalone Stop Head (Independent Training) — Added 2026-05-11

### Overview

独立的 Stop 预测头，**完全独立于主训练流程**训练。直接使用冻结的 VAE 编码器和预缓存的 T5 文本特征，只训练一个轻量级分类头。

**核心设计：**
- 输入与主训练一致：9帧0deg视频(8历史+1当前) + 1帧30deg下倾 + 文本
- 输出：0/1 二分类（距离轨迹终点 ≤5 步则标记为 1）
- 冻结：VAE 全部冻结，不影响任何其他模块
- 只更新 stop head 自身参数（~1.4M）

### Why Standalone?

- 可以在主训练运行的同时，利用 GPU 剩余显存并行训练
- 不需要加载 6B MoT / DiT — 只需 VAE (~100M)
- 不需要 diffusion 过程 — 直接 BCE 分类
- 1 张 A100 即可，~2-4 小时训练完成

### Architecture

```
输入 (与主训练一致):
  ┌─ 9帧 0deg 视频 [B, 3, 9, 224, 224]     ← 8帧均匀采样历史 + 1帧当前
  │     ↓ 冻结VAE (9%4=1 ✓ → 3 latent frames)
  │     → [B, 48, 3, 28, 28]
  │     → spatiotemporal avg pool → [B, 48]
  │     → Linear+GELU → [B, 512]  (可训练)
  │
  ├─ 1帧 30deg 下倾图 [B, 3, 224, 224]       ← 当前时刻
  │     ↓ 冻结VAE
  │     → [B, 48, 1, 28, 28]
  │     → spatial avg pool → [B, 48]
  │     → Linear+GELU → [B, 256]  (可训练)
  │
  └─ 文本 T5 特征 [B, 256, 4096]             ← 预缓存
        → masked mean pool → [B, 4096]

融合 + 分类:
  concat([B, 4096], [B, 512], [B, 256]) = [B, 4864]
  → text_proj [4096→256] + video_proj [512→256] + overhead_proj [256→256]
  → concat [B, 768]
  → classifier MLP [768→256→1]
  → BCE Loss with stop_label
```

### Supervision Signal

```
stop_label = 1  if  (episode_length - 1 - current_idx) <= 5   (距终点≤5步)
stop_label = 0  otherwise
```

正样本过采样 3x（因为大部分帧远离终点，正样本稀少）。

### Files

| File | Purpose |
|------|---------|
| `scripts/train_stop_head.py` | 训练脚本（含模型定义 + dataset + training loop） |
| `scripts/run_stop_head_on_cluster.sh` | 在当前集群节点上后台训练的启动脚本 |
| `scripts/run_stop_head_train.sh` | 通用启动脚本（支持单/多GPU） |
| `scripts/infer_stop_head.py` | 推理脚本 + `StopHeadNavigationHelper` 封装类 |

### Training

```bash
# 在当前 8x8 集群节点上后台运行 (利用剩余显存 ~41GB, 只需 ~5-8GB)
nohup bash scripts/run_stop_head_on_cluster.sh > logs/stop_head.log 2>&1 &

# 查看训练进度
tail -f logs/stop_head.log
```

### Resource Comparison

| | 主训练 (MoT 6B) | Stop Head (standalone) |
|---|---|---|
| 可训练参数 | ~6B | ~1.4M |
| GPU需求 | 64 x A100 (ZeRO-2) | 1 x A100 (剩余显存即可) |
| 训练时间 | 数天 | 2-4 小时 |
| 显存需求 | 40GB+ per GPU | ~5-8 GB |
| 数据IO | 17帧视频+overhead | 10帧 (9+1) |
| 依赖 | DiT + VAE + T5 + diffusion | 仅 VAE (冻结) |

### Inference Usage

```python
from scripts.infer_stop_head import StopHeadNavigationHelper

helper = StopHeadNavigationHelper(
    checkpoint_path="runs/stop_head/best_stop_head.pt",
    vae_path="/tmp/fastwam_checkpoints",
    text_embedding_cache_dir="text_embeds_cache/nav_vln",
)

# 在导航循环中:
should_stop = helper.should_stop(
    history_frames=[img0, img1, ..., img7],  # 8 PIL Images
    current_frame=current_img,               # PIL Image
    overhead_frame=overhead_img,             # PIL Image
    instruction="Go to the kitchen and turn left",
)
if should_stop:
    print("Navigation complete!")
```

### Checkpoint Format

```python
{
    "step": int,
    "epoch": int,
    "model_state_dict": {
        "video_pool_proj": OrderedDict,     # Linear(48→512) + GELU
        "overhead_pool_proj": OrderedDict,  # Linear(48→256) + GELU
        "stop_head": OrderedDict,           # 分类头 MLP
    },
    "metrics": {"loss", "accuracy", "precision", "recall", "f1"},
    "args": dict,  # 训练参数
}
```

### Relationship to Other Stop Approaches

本项目有 3 种 stop 机制，按时间顺序：

1. **Explicit StopHead in ActionDiT** (`predict_stop: true`) — 在 MoT action expert 内部加分类头，与 diffusion 联合训练。已被方案3取代。
2. **Action dim 4 (moving_flag)** (`action_dim: 4`) — 当前主训练使用的方案。将 stop 信号作为 action 的第4维，通过 diffusion 隐式学习。
3. **Standalone Stop Head** (本节) — 独立训练的轻量分类器。可作为方案2的补充/对照：
   - 如果 moving_flag 不够准确，可以用 standalone stop head 做二次确认
   - 可以用两者的 ensemble: `final_stop = moving_flag < 0.5 OR stop_head_prob > 0.5`
   - 独立评估 stop 预测质量（precision/recall/F1）

---

## FastWAM 整体架构详解 — Added 2026-05-11

### 模型全局结构

```
FastWAM
├── VAE (WanVideoVAE, ~700M) .............. ❄️ 冻结
│   └── encode: [B, 3, T, H, W] → [B, 48, T_lat, H/8, W/8]
│   └── decode: latent → pixel video
│
├── T5 Text Encoder (~5B) ................. ❄️ 冻结 (或不加载, 用预缓存)
│   └── encode: text → [B, 256, 4096]
│
└── MoT (model.dit, ~6B) .................. 🔥 可训练
    ├── Video Expert (WanVideoDiT, 30层, hidden=3072, ~5B)
    │   ├── patch_embedding: Conv3d(96→3072, k=[1,2,2])
    │   ├── blocks[0..29]: DiTBlock (self-attn + cross-attn + FFN)
    │   └── post_dit: unpatchify → latent noise prediction
    │
    └── Action Expert (ActionDiT, 30层, hidden=1024, ~1B)
        ├── action_encoder: Linear(4→1024)
        ├── text_embedding: Linear(4096→1024) + GELU + Linear(1024→1024)
        ├── blocks[0..29]: DiTBlock (self-attn + cross-attn + FFN)
        └── head: Linear(1024→4) → action noise prediction
```

### MoT (Mixture of Transformers) 核心机制

**MoT 不是一个独立模块，而是把两个 Expert 的 Transformer 层"交织在一起"跑。**

两个 Expert 各自有 30 层 DiT blocks，但它们的 self-attention 被融合成**联合 attention**：

```python
for layer_idx in range(30):  # 逐层交织
    # 1. 各 Expert 独立计算 Q, K, V (用各自的 blocks[layer_idx])
    q_video, k_video, v_video = video_expert.blocks[layer_idx].qkv(video_tokens)   # [B, 980, 3072]
    q_action, k_action, v_action = action_expert.blocks[layer_idx].qkv(action_tokens) # [B, 32, 1024]
    
    # 2. ★ 核心：拼在一起做联合 attention ★
    q_cat = cat([q_video, q_action])    # [B, 1012, head_dim]
    k_cat = cat([k_video, k_action])    # [B, 1012, head_dim]
    v_cat = cat([v_video, v_action])    # [B, 1012, head_dim]
    mixed = flash_attention(q_cat, k_cat, v_cat, mask=attention_mask)
    
    # 3. 拆回来，各 Expert 独立做 cross-attn(to text) + FFN
    video_tokens = video_expert.blocks[layer_idx].post(mixed[:, :980], text_ctx_3072)
    action_tokens = action_expert.blocks[layer_idx].post(mixed[:, 980:], text_ctx_1024)
```

**注意**: Video Expert 和 Action Expert 的 hidden_dim 不同 (3072 vs 1024)，但它们的 `num_heads=24, attn_head_dim=128` 相同，所以 Q/K/V 在 attention 空间维度对齐 (24*128=3072 for video, 但 action 只用 24 heads 的子集? 不，action 也是 24*128=3072 的 attention 空间，只是 hidden_dim 不同，QKV projection 会映射到相同的 head_dim 空间)。

### Attention Mask 控制信息流

```
                    Video Tokens (980)           Action Tokens (32)
                ┌────────────────────────────┬──────────────────────┐
Video Tokens    │ group-causal               │        ✗             │
(980 queries)   │ (帧i只看≤i的帧)             │  (看不到 action)     │
                ├────────────────────────────┼──────────────────────┤
Action Tokens   │ 看 condition 帧 tokens     │                      │
(32 queries)    │ (前3 latent帧=588 tokens)  │   全连接 (互相看)     │
                └────────────────────────────┴──────────────────────┘
```

- **Video → Video**: Group-causal (帧 i 的 tokens 只能 attend to 帧 ≤ i)
- **Video → Action**: ✗ 看不到 (单向)
- **Action → Video**: ✓ 只能看 condition 帧 (前3 latent frames = 3×14×14 = 588 video tokens)
- **Action → Action**: ✓ 全连接 (32 tokens 互相看)

### Video/Action/Text 如何统一到同一空间

FastWAM **从不做 global avg pool**，而是把所有东西变成 **token 序列**，通过 attention 融合：

#### Video 通路

```
原始视频 [B, 3, 17, 224, 224]
    ↓ VAE encode (冻结)
Video latent [B, 48, 5, 28, 28]
    + Overhead latent broadcast [B, 48, 5, 28, 28]
    = [B, 96, 5, 28, 28]  (channel concat)
    ↓ patch_embedding: Conv3d(96, 3072, kernel=[1,2,2], stride=[1,2,2])
    
    每个 2×2 空间 patch (96通道×4像素=384个数)
    → 3072个卷积核各算一个加权和 → 1个 3072 维 token
    
    时间: 5帧, 空间: 14×14 = 196 patches/帧
    总计: 5 × 196 = 980 tokens, 每个 3072 维
    
Video tokens: [B, 980, 3072]
```

#### Action 通路

```
Action (加噪后) [B, 32, 4]  (x, y, theta, moving_flag)
    ↓ action_encoder: Linear(4, 1024)
Action tokens: [B, 32, 1024]

每个 action step 就是一个 1024 维 token
```

#### Text 通路 (作为 cross-attention context)

```
T5 text embedding [B, 256, 4096] (预缓存)

For Video Expert:
    → 不做变换，直接作为 cross-attention 的 context
    → 每层 DiT block 里: video tokens cross-attend to [B, 256, 4096]
    → cross-attn 内部: Linear(4096→3072) 做 K/V projection

For Action Expert:
    → text_embedding: Linear(4096→1024) + GELU + Linear(1024→1024)
    → context: [B, 256, 1024]
    → 每层 DiT block 里: action tokens cross-attend to [B, 256, 1024]
```

#### 融合总结

| 模态 | 表示方式 | 维度 | 融合方式 |
|------|---------|------|---------|
| Video | patch tokens (保留全部空间+时间位置) | [B, 980, 3072] | 联合 self-attention (MoT) + cross-attn to text |
| Action | per-step tokens | [B, 32, 1024] | 联合 self-attention (MoT) + cross-attn to text |
| Text | sequence tokens | [B, 256, 4096] | 作为 cross-attention context (不参与 self-attn) |
| Overhead | channel concat 到 video latent | 融入 video 通路 | 通过 patch_embedding 编码进 video tokens |

**关键**: 所有 Expert 的 `num_heads=24, attn_head_dim=128` 相同，所以在 attention 计算时 Q/K/V 都在同一个 `24 heads × 128 dim` 空间中，不同 hidden_dim (3072 vs 1024) 只是各自 FFN 和 cross-attention 的宽度不同。

### Diffusion 去噪循环

**训练时**：没有循环。每次只做一步前向：
```python
# training_loss() 内部
t_video = sample_random_timestep()     # 随机采样 t
t_action = sample_random_timestep()    # video 和 action 各自独立采样 t

noisy_video = add_noise(clean_video_latent, noise, t_video)
noisy_action = add_noise(clean_action, noise, t_action)

pred_video_noise = model.forward(noisy_video, noisy_action, t_video, t_action, text)
pred_action_noise = model.forward(...)  # 同一次 forward 同时输出

loss = MSE(pred_video_noise, target) + MSE(pred_action_noise, target)
```

**推理时**：多步去噪循环 (如 50 步)，入口在 `_predict_joint_noise`：
```python
latents_video = torch.randn(...)   # 纯噪声初始化
latents_action = torch.randn(...)

for t in scheduler.timesteps:  # 50, 49, 48, ... 1
    pred_v, pred_a = model._predict_joint_noise(
        latents_video, latents_action, t, t, context, context_mask, ...
    )
    latents_video = scheduler.step(pred_v, t, latents_video)    # 去一步噪
    latents_action = scheduler.step(pred_a, t, latents_action)  # 去一步噪

# 最终
generated_video = VAE.decode(latents_video)
predicted_actions = latents_action  # [B, 32, 4]
```

### 训练时冻结/可训练一览

| 模块 | 参数量 | 训练状态 | 说明 |
|------|--------|---------|------|
| VAE (WanVideoVAE) | ~700M | ❄️ 冻结 | 编码/解码视频，不更新 |
| T5 Text Encoder | ~5B | ❄️ 不加载 | 使用预缓存 embedding |
| Video Expert (WanVideoDiT) | ~5B | 🔥 可训练 | 从 Wan2.2-TI2V-5B 预训练权重初始化 |
| Action Expert (ActionDiT) | ~1B | 🔥 可训练 | 从预训练 backbone 初始化 |
| MoT (容器) | 0 (无自身参数) | — | 只是编排两个 Expert 的 forward |

训练代码核心逻辑 (`trainer.py`):
```python
model.requires_grad_(False)       # 先全部冻结
model.dit.requires_grad_(True)    # 只解冻 MoT = video_expert + action_expert
# model.dit 就是 model.mot
```

### Standalone Stop Head vs FastWAM 主模型的视觉处理对比

| | FastWAM 主模型 | Standalone Stop Head |
|---|---|---|
| 视觉表示 | 每个 patch 一个 token (980 tokens×3072维) | global avg pool → 1个向量 48 维 |
| 保留信息 | 全部空间+时间位置 (RoPE) | 全部 pool 掉 |
| 文本融合 | cross-attention (每层, 每个 token 独立) | mean pool → concat |
| 融合深度 | 30 层 Transformer 逐层交互 | 单层 MLP |
| 可训练参数 | ~6B | ~1.48M |

这解释了为什么 stop head 的视觉信息利用可能不充分 — 它把 FastWAM 精心保留的 112,896 维视觉信息压缩成了 48 维。

