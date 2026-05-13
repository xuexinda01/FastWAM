# FastWAM 适配导航任务改造方案

## 1. 背景

FastWAM 是一个基于 DiT (Diffusion Transformer) 的 world-action model，采用 MoT (Mixture of Transformers) 架构将 Video Expert 和 Action Expert 结合，通过 flow matching 同时预测视频和动作。原设计用于机械臂操作任务（LIBERO benchmark）。

现在需要将其改造为支持 **视觉语言导航 (VLN)** 任务，使用 InternNav 的导航轨迹数据进行训练。

## 2. 核心差异对比

| | FastWAM (机械臂操作) | 导航任务 (目标) |
|---|---|---|
| **动作维度** | 7 (eef_pose 6 + gripper 1) | 3 (x, y, theta) |
| **动作表示** | delta/absolute 末端位姿 | 相对于当前位置的局部坐标轨迹点 |
| **预测步数** | 32步 | 32步 (保持一致) |
| **状态(proprio)** | 8维 (eef 6 + gripper 2) | 无 |
| **图像输入** | 2路相机 224x224，水平拼接成 224x448 | 2路相机 (平视+俯视) 224x224，水平拼接成 224x448 |
| **文本条件** | 操作指令 (如 "pick up the cup") | 导航指令 (如 "Exit the bedroom, enter the bathroom") |
| **数据格式** | LeRobot (parquet + mp4视频) | LeRobot (parquet + jpg图像目录) |

## 3. 导航数据分析

### 3.1 数据路径
```
/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r/
├── 17DRP5sb8fy/           # 场景名
│   ├── meta/
│   │   ├── info.json       # 数据集元信息
│   │   ├── episodes.jsonl  # 每个episode的指令和长度
│   │   └── tasks.jsonl
│   ├── data/chunk-000/
│   │   └── episode_XXXXXX.parquet  # 动作+pose数据
│   └── videos/chunk-000/
│       ├── observation.images.rgb.125cm_0deg/   # 平视RGB图
│       ├── observation.images.rgb.125cm_30deg/  # 俯视30度RGB图
│       ├── observation.images.depth.125cm_0deg/ # 深度图
│       └── ...
├── 1LXtFkjw3qL/
├── ...
```

### 3.2 数据字段 (parquet)

| 字段 | 类型 | shape | 说明 |
|---|---|---|---|
| `action` | int32 | [1] | 离散动作: -1=init, 0=STOP, 1=前进, 2=左转, 3=右转 |
| `pose.125cm_0deg` | float32 | [4, 4] | 相机在世界坐标系的4x4齐次变换矩阵 |
| `pose.125cm_30deg` | float32 | [4, 4] | 俯视相机pose |
| `goal.125cm_0deg` | int32 | [2] | 目标像素坐标 |
| `frame_index` | int64 | [1] | 帧索引 |
| `episode_index` | int64 | [1] | episode索引 |

### 3.3 图像文件命名
```
episode_{ep_idx:06d}_{frame_idx}.jpg
```
例如: `episode_000000_0.jpg`, `episode_000000_1.jpg`, ...

### 3.4 Pose 坐标系分析

- **坐标系**: 世界坐标系，z轴为高度方向（恒定1.25m），x-y为地面平面
- **Rotation矩阵第3列** (z轴) 表示相机前进方向在世界坐标系的投影
- 导航运动主要发生在 x-y 平面，高度恒定

### 3.5 相对动作计算方法

从 pose 序列计算相对 (x, y, theta) 轨迹：

```python
def compute_relative_actions(poses, start_idx, num_steps):
    """将绝对pose序列转换为相对于start_idx帧的局部坐标动作"""
    T_base = poses[start_idx]  # 4x4
    actions = []
    for j in range(start_idx + 1, min(start_idx + num_steps + 1, len(poses))):
        T_rel = np.linalg.inv(T_base) @ poses[j]
        local_pos = T_rel[:3, 3]     # 局部位移
        R_rel = T_rel[:3, :3]
        theta = np.arctan2(R_rel[0, 2], R_rel[2, 2])  # yaw角
        actions.append([local_pos[0], local_pos[2], theta])
    return np.array(actions)  # [num_steps, 3]
```

验证结果示例 (episode_000000, 从frame 0开始的32步):
```
[[0.0000, 0.0000, 0.2618],   # 原地转向
 [0.0000, 0.0000, 0.5236],
 [0.0000, 0.0000, 0.7854],
 [0.1768, 0.1768, 0.7854],   # 开始位移
 [0.1768, 0.1768, 1.0472],
 [0.3933, 0.3018, 1.0472],
 ...]
```

## 4. 改造方案

### 4.1 需要新建的文件

| 文件 | 说明 |
|---|---|
| `configs/data/nav_vln.yaml` | 导航数据配置 |
| `configs/model/fastwam_nav.yaml` | 导航模型配置 (action_dim=3) |
| `configs/task/nav_vln_1e-4.yaml` | 训练超参配置 |
| `src/fastwam/datasets/lerobot/nav_video_dataset.py` | 导航数据集类 |
| `src/fastwam/datasets/lerobot/processors/nav_processor.py` | 导航数据处理器 |

### 4.2 模型层面改动

**改动极小，仅通过配置调整：**

1. **action_dim: 7 → 3**
   - `ActionDiT.action_encoder`: `nn.Linear(3, 1024)` (原 `nn.Linear(7, 1024)`)
   - `ActionDiT.head`: `nn.Linear(1024, 3)` (原 `nn.Linear(1024, 7)`)
   - `video_dit_config.action_dim`: 3

2. **proprio_dim: 8 → null**
   - 不创建 `proprio_encoder`（已有 None 判断逻辑）
   - `context` 不拼接 proprio embedding

3. **预训练权重复用策略**
   - Video Expert (Wan2.2 5B DiT): **完全复用**（视频生成能力通用）
   - Action Expert backbone (30层DiT blocks): **复用**（通过 `action_dit_pretrained_path` 加载）
   - `action_encoder` 和 `head`: **随机初始化**（维度从7→3不匹配，已有 `skip_prefixes` 机制自动处理）

### 4.3 数据集类设计 (NavVideoDataset)

核心逻辑：
1. 扫描所有场景目录，加载 episodes.jsonl 获取指令和长度
2. 构建索引：(scene, episode_idx, start_frame) 三元组
3. `__getitem__` 返回：
   - `video`: [3, T_video, 224, 448] — 双相机拼接的RGB视频
   - `action`: [32, 3] — 相对 (x, y, theta) 轨迹
   - `action_is_pad`: [32] — padding mask
   - `context`: [128, 4096] — 预缓存的文本 embedding
   - `context_mask`: [128] — 文本 mask
   - `image_is_pad`: [T_video] — 视频帧padding mask

### 4.4 视频帧采样策略

与原 FastWAM 一致：
- `num_frames = 33` (32 action steps + 1 initial frame)
- `action_video_freq_ratio = 4` → 视频帧数 = (33-1)/4 + 1 = 9 帧
- 视频帧在动作序列中均匀采样：indices [0, 4, 8, 12, 16, 20, 24, 28, 32]

### 4.5 文本 Embedding 缓存

需要预计算 VLN 指令的 T5 text embedding 并缓存，与原有流程一致。

## 5. 验证计划

1. 单条数据 debug：确保 NavVideoDataset 能正确读取并输出正确 shape
2. 单 GPU forward pass：确保模型能正常计算 loss
3. 训练 100 步：确认 loss 下降正常
4. 推理验证：给定导航指令和起始帧，检查模型输出轨迹合理性
