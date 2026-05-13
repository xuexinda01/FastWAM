# FastWAM 训练数据处理 & 模型前向全流程

## 整体架构

```
原始数据(磁盘) → Dataset类 → Processor预处理 → DataLoader → training_loss()
                                                               ├─ VAE编码
                                                               ├─ Proprio编码
                                                               ├─ Video Expert (pre_dit)
                                                               ├─ Action Expert (pre_dit)
                                                               ├─ MoT混合注意力(30层)
                                                               ├─ Video Expert (post_dit)
                                                               ├─ Action Expert (post_dit)
                                                               └─ Loss计算
```

---

## 1. 数据加载 (从磁盘读取)

| 步骤 | 文件 | 说明 |
|------|------|------|
| LeRobot 数据读取 | `src/fastwam/datasets/lerobot/base_lerobot_dataset.py` | 从4个 libero 数据集目录读取 parquet/video |
| 顶层 Dataset 封装 | `src/fastwam/datasets/lerobot/robot_video_dataset.py` | 采样视频帧、拼接多摄像头、加载文本缓存 |
| 数据配置 | `configs/data/libero_2cam.yaml` | 指定数据路径、摄像头配置 |

**原始数据：**
- 图像：2 摄像头 × 33 帧 × 512×512 RGB
- 动作：32 步 × 7 维 (6 EEF + 1 gripper)
- 状态：33 步 × 8 维 (6 EEF + 2 gripper)
- 任务描述：文本字符串

**数据集目录（4个）：**
1. `data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot`
2. `data/libero_mujoco3.3.2/libero_object_no_noops_lerobot`
3. `data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot`
4. `data/libero_mujoco3.3.2/libero_10_no_noops_lerobot`

---

## 2. 数据预处理 (Processor)

| 步骤 | 文件 | 说明 |
|------|------|------|
| 图像变换 | `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` | Resize 到 224×224，ToTensor |
| 归一化 | `src/fastwam/datasets/lerobot/utils/normalizer.py` | action/state min-max 归一化 |
| Action-State 合并 | `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` | ConcatLeftAlign 合并 |
| 视频帧采样 | `src/fastwam/datasets/lerobot/robot_video_dataset.py` | 33帧每4帧采1→9帧，双摄像头水平拼接 224×448 |
| 文本嵌入加载 | `src/fastwam/datasets/lerobot/robot_video_dataset.py` | 预计算的 T5 嵌入，shape [128, 4096] |

**Processor 输入 sample：**
```python
{
    "instruction": str,           # 任务描述
    "images": {
        "image": Tensor,          # [33, 3, 512, 512] (摄像头1)
        "wrist_image": Tensor     # [33, 3, 512, 512] (摄像头2)
    },
    "action": {"default": Tensor},  # [32, 7] 原始动作
    "state": {"default": Tensor},   # [33, 8] 原始状态
    "image_is_pad": Tensor,       # [33]
    "action_is_pad": Tensor,      # [32]
    "state_is_pad": Tensor,       # [33]
}
```

**DataLoader 最终输出 sample dict：**
```python
{
    "video":          [B, 3, 9, 224, 448],   # 视频（两摄像头水平拼接）
    "action":         [B, 32, 7],            # 归一化动作
    "proprio":        [B, 32, 8],            # 归一化本体感知
    "context":        [B, 128, 4096],        # T5 文本嵌入
    "context_mask":   [B, 128],              # 文本 attention mask
    "image_is_pad":   [B, 9],               # 视频帧 padding mask
    "action_is_pad":  [B, 32],              # 动作 padding mask
}
```

---

## 3. 模型前向传播 (`training_loss`)

### 3.1 VAE 视频编码

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/wan_video_vae.py` | Wan2.2 视频 VAE |
| `src/fastwam/models/wan22/fastwam.py:301` | 调用入口 |

```
video [B, 3, 9, 224, 448] → VAE Encoder → latents [B, 48, T_lat, 28, 56]
（空间8×下采样，时间也有压缩）
```

### 3.2 本体感知编码 (Proprio Encoder)

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/fastwam.py:426` | proprio 编码后拼接到 text context |

```
proprio [B, 8] → Linear → proprio_embed [B, 4096] → 拼接到 context 末尾
```

### 3.3 加噪（Flow Matching 扩散）

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/schedulers/` | 连续流匹配调度器 |

```
video_latents + noise → noisy_latents   (依据 timestep_video)
action + noise        → noisy_action     (依据 timestep_action)
```

- 调度器类型：WanContinuousFlowMatchScheduler
- Video/Action 各自独立采样 timestep
- train_shift=5.0, num_steps=1000

### 3.4 Video Expert Pre-DIT

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/wan_video_dit.py` | Wan2.2 视频 DiT（30层, dim=3072, heads=24） |

```
noisy_latents [B, 48, T_lat, 28, 56]
  → Patchify [1,2,2] 时空patch
  → video_tokens [B, S_video, 3072]
  + RoPE 位置编码
  + 时间嵌入（sinusoidal + MLP）
  + 文本 cross-attn 准备
```

### 3.5 Action Expert Pre-DIT

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/action_dit.py` | 动作 DiT（30层, dim=1024, heads=24） |

```
noisy_action [B, 32, 7]
  → Linear(7 → 1024)
  → action_tokens [B, 32, 1024]
  + 时间嵌入（sinusoidal + MLP）
  + 文本嵌入（Linear + GELU + Linear）
```

### 3.6 MoT 混合注意力（核心）

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/mot.py` | Mixture of Transformers，30层联合注意力 |

**注意力 mask 规则：**

| Source → Target | 可见性 |
|-----------------|--------|
| Video → Video | 因果注意力（frame causal） |
| Action → Action | 全注意力 |
| Action → Video 首帧 | 可见 |
| Video → Action | 不可见 |

**每层计算流程：**
```
1. 各 Expert 独立计算 Q, K, V
2. 应用 RoPE 旋转位置编码
3. 时间调制（shift, scale, gate）
4. 拼接 [video_tokens, action_tokens] → 联合注意力（带 mask）
5. 拆分回各 Expert
6. 各 Expert 独立文本 Cross-Attention
7. 各 Expert 独立 FFN + 残差连接
```

### 3.7 Expert Post-DIT

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/wan_video_dit.py` | Video detokenize |
| `src/fastwam/models/wan22/action_dit.py` | Action 输出投影 |

```
mixed_video_tokens  → Unpatchify → pred_video  [B, 48, T_lat, 28, 56]
mixed_action_tokens → Linear(1024 → 7) → pred_action [B, 32, 7]
```

### 3.8 Loss 计算

| 文件 | 说明 |
|------|------|
| `src/fastwam/models/wan22/fastwam.py:609-643` | MSE + 扩散权重 + padding mask |

```python
# Video Loss
video_loss = weighted_MSE(pred_video, target_video) × video_weight
# 考虑 image_is_pad mask 排除 padding 帧

# Action Loss
action_loss = weighted_MSE(pred_action, target_action) × action_weight
# 考虑 action_is_pad mask 排除 padding 步

# 总 Loss
loss_total = 1.0 * loss_video + 1.0 * loss_action
```

**注意：** `action_loss_detach_video_expert=True`，即反传 action loss 时 detach video expert 的梯度。

---

## 4. 数据 shape 变化全览

```
磁盘 (LeRobot):
  images: [2 cameras, 33 frames, 512×512 RGB]
  actions: [32 timesteps, 7-dim]
  state: [33 timesteps, 8-dim]

    ↓ RobotVideoDataset.__getitem__()

Processor 输入:
  pixel_values: [2, 33, 3, 224, 224]
  action: [32, 7]
  proprio: [33, 8]

    ↓ FastWAMProcessor.preprocess() (归一化)

Processor 输出:
  pixel_values: [2, 33, 3, 224, 224]
  action: [32, 7]  (normalized)
  proprio: [33, 8] (normalized)

    ↓ 帧采样 + 摄像头拼接 + Collate (batch_size=16)

DataLoader 输出:
  video: [16, 3, 9, 224, 448]
  action: [16, 32, 7]
  proprio: [16, 32, 8]
  context: [16, 128, 4096]
  context_mask: [16, 128]

    ↓ VAE Encoding

  video_latents: [16, 48, T_lat, 28, 56]

    ↓ Expert Pre-DIT Tokenization

  video_tokens: [16, S_video, 3072]
  action_tokens: [16, 32, 1024]

    ↓ MoT (30 layers of mixed attention)

  mixed_video_tokens: [16, S_video, 3072]
  mixed_action_tokens: [16, 32, 1024]

    ↓ Expert Post-DIT

  pred_video: [16, 48, T_lat, 28, 56]
  pred_action: [16, 32, 7]

    ↓ Loss Computation

  loss_video: scalar
  loss_action: scalar
  loss_total = loss_video + loss_action
```

---

## 5. 关键文件索引

### 模型架构

| 组件 | 文件 | 说明 |
|------|------|------|
| FastWAM 主模型 | `src/fastwam/models/wan22/fastwam.py` | 协调 video/action experts，`training_loss()` 入口 |
| Video Expert (DiT) | `src/fastwam/models/wan22/wan_video_dit.py` | 30层视频扩散 Transformer (来自 Wan2.2) |
| Action Expert | `src/fastwam/models/wan22/action_dit.py` | 30层动作扩散 Transformer |
| MoT 混合注意力 | `src/fastwam/models/wan22/mot.py` | Mixture of Transformers，跨 expert 联合注意力 |
| VAE | `src/fastwam/models/wan22/wan_video_vae.py` | 视频编码器（8× 空间下采样） |
| Visual Encoder | `src/fastwam/models/wan22/visual_encoder.py` | DINO/V-JEPA2 替代 VAE（frozen backbone） |
| 扩散调度器 | `src/fastwam/models/wan22/schedulers/` | 连续流匹配调度 |

### 数据处理

| 组件 | 文件 | 说明 |
|------|------|------|
| Dataset 顶层 | `src/fastwam/datasets/lerobot/robot_video_dataset.py` | PyTorch Dataset，帧采样 + 多摄像头拼接 |
| Base Dataset | `src/fastwam/datasets/lerobot/base_lerobot_dataset.py` | 磁盘数据读取 |
| LeRobot 接口 | `src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py` | LeRobot 库集成 |
| Processor | `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` | 图像变换 + 归一化 + 合并 |
| Normalizer | `src/fastwam/datasets/lerobot/utils/normalizer.py` | Action/State 归一化 |
| Transforms | `src/fastwam/datasets/lerobot/transforms/` | 图像/动作变换 |

### 训练

| 组件 | 文件 | 说明 |
|------|------|------|
| 训练入口 | `scripts/train.py` | Hydra 配置 + 入口 |
| Runtime | `src/fastwam/runtime.py` | 模型/数据实例化 |
| Trainer | `src/fastwam/trainer.py` | 训练循环、评估、checkpoint |

### 配置

| 文件 | 说明 |
|------|------|
| `configs/train.yaml` | 基础训练配置 |
| `configs/task/libero_uncond_2cam224_1e-4.yaml` | 任务覆盖（lr, batch_size 等） |
| `configs/model/fastwam.yaml` | 模型架构配置 |
| `configs/data/libero_2cam.yaml` | 数据加载配置 |

---

## 6. 关键超参数

| 类别 | 参数 | 值 |
|------|------|-----|
| 视频 | 输入帧数 | 33 → 采样9帧 |
| 视频 | 采样步长 | 4 |
| 视频 | 分辨率 | 224×448 (H×W) |
| 视频 | 摄像头拼接 | 水平拼接 |
| 动作 | Action horizon | 32 |
| 动作 | Action dim | 7 |
| 模型 | Video DiT layers | 30 |
| 模型 | Video hidden dim | 3072 |
| 模型 | Action DiT layers | 30 |
| 模型 | Action hidden dim | 1024 |
| 模型 | Text embedding dim | 4096 |
| 训练 | Batch size | 16 |
| 训练 | Learning rate | 1e-4 |
| 训练 | Warmup | 5% of total steps |
| 训练 | LR scheduler | Cosine annealing |
| 训练 | Max grad norm | 1.0 |
| 训练 | Mixed precision | BF16 |
| 扩散 | Video timesteps | 1000 |
| 扩散 | Action timesteps | 1000 |
| 扩散 | Shift | 5.0 |
| 扩散 | 类型 | 连续流匹配 |
| 归一化 | Action delta mask | [T,T,T,T,T,T,F]（EEF用delta，gripper用绝对值） |
