"""
Standalone Stop Head Training Script.

设计思路:
- 输入 (与主训练完全一致):
    - 9帧0deg视频: 8帧历史(uniformly sampled) + 1帧当前 → VAE编码 → [B, 48, 3, 28, 28]
    - 1帧30deg下倾图: 当前时刻 → VAE编码 → [B, 48, 1, 28, 28]
    - 文本: T5 预缓存特征 [B, 256, 4096]
- 输出: 0/1 二分类 (当前位置距离轨迹点结束在5个点以内则记作1)
- 冻结: VAE 全部冻结，只训练 StopHead 参数
- 训练量: 相比主训练轻量得多 — 不需要 6B MoT，不需要 diffusion，不需要生成未来帧

训练完全独立于现有的 run_nav_vln_8x8.sh 主训练流程。

Usage:
    # 单GPU训练
    python scripts/train_stop_head.py

    # 多GPU训练 (如4卡)
    torchrun --nproc_per_node=4 scripts/train_stop_head.py

    # 指定参数
    python scripts/train_stop_head.py \
        --batch_size 32 \
        --lr 1e-3 \
        --num_epochs 20 \
        --output_dir ./runs/stop_head_v1
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Stop Head Model
# ═══════════════════════════════════════════════════════════════════════════════

class StopHeadModel(nn.Module):
    """
    Stop 预测头 — 融合视觉 latent 和文本特征进行二分类。

    输入:
        - text_feat: [B, text_dim] — pooled T5 text embedding
        - video_feat: [B, video_feat_dim] — pooled VAE video latent (9帧历史+当前)
        - overhead_feat: [B, overhead_feat_dim] — pooled VAE overhead latent

    输出:
        - logits: [B, 1] — stop probability logit

    Architecture:
        text_feat ─────→ [Linear → GELU] ──┐
                                            │
        video_feat ────→ [Linear → GELU] ──├──→ [concat] → [MLP] → logit
                                            │
        overhead_feat ─→ [Linear → GELU] ──┘
    """

    def __init__(
        self,
        text_dim: int = 4096,
        video_feat_dim: int = 512,
        overhead_feat_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(video_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.overhead_proj = nn.Sequential(
            nn.Linear(overhead_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        video_feat: torch.Tensor,
        overhead_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, 1]
        """
        t = self.text_proj(text_feat)         # [B, hidden_dim]
        v = self.video_proj(video_feat)       # [B, hidden_dim]
        o = self.overhead_proj(overhead_feat) # [B, hidden_dim]
        combined = torch.cat([t, v, o], dim=-1)  # [B, hidden_dim*3]
        logits = self.classifier(combined)  # [B, 1]
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset: Stop Prediction — 与主训练一致的输入
# ═══════════════════════════════════════════════════════════════════════════════

class StopPredictionDataset(Dataset):
    """
    Stop预测数据集 — 复用 NavVideoDataset 的数据源。

    每个sample输出 (与主训练一致的视觉输入):
        - video: [3, 9, H, W] — 8帧历史 + 1帧当前 (0deg), 范围[-1,1]
        - overhead: [3, H, W] — 1帧下倾图 (30deg), 范围[-1,1]
        - context: [context_len, 4096] — 预计算的T5特征
        - context_mask: [context_len] — 文本mask
        - stop_label: 0 或 1

    标注逻辑:
        stop_label = 1  if  (episode_length - 1 - current_idx) <= stop_threshold
        stop_label = 0  otherwise
    """

    def __init__(
        self,
        dataset_dirs: list,
        camera_keys: list = None,
        n_history_frames: int = 8,
        video_size: list = None,
        text_embedding_cache_dir: str = None,
        context_len: int = 256,
        sample_stride: int = 2,
        stop_threshold: int = 5,
        balance_ratio: float = 3.0,
    ):
        super().__init__()
        if camera_keys is None:
            camera_keys = ["125cm_0deg", "125cm_30deg"]
        if video_size is None:
            video_size = [224, 224]

        self.primary_camera = camera_keys[0]   # 125cm_0deg
        self.overhead_camera = camera_keys[1]  # 125cm_30deg
        self.n_history_frames = n_history_frames  # 8
        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.stop_threshold = stop_threshold

        import hashlib
        import torchvision.transforms.functional as transforms_F
        from PIL import Image
        self._hashlib = hashlib
        self._transforms_F = transforms_F
        self._Image = Image

        self.samples = []
        self._build_index(dataset_dirs, sample_stride, balance_ratio)

        # 统计
        n_pos = sum(1 for s in self.samples if s["stop_label"] == 1)
        n_neg = len(self.samples) - n_pos
        print(f"[StopPredictionDataset] Total samples: {len(self.samples)}, "
              f"positive (stop=1): {n_pos} ({100*n_pos/max(len(self.samples),1):.1f}%), "
              f"negative (stop=0): {n_neg}")

    def _build_index(self, dataset_dirs, sample_stride, balance_ratio):
        """构建样本索引，带有正样本过采样。"""
        for dataset_dir in dataset_dirs:
            if not os.path.isdir(dataset_dir):
                continue

            scene_names = sorted([
                d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ])

            for scene_name in scene_names:
                scene_path = os.path.join(dataset_dir, scene_name)
                episodes_file = os.path.join(scene_path, "meta", "episodes.jsonl")
                if not os.path.isfile(episodes_file):
                    continue

                episodes = []
                with open(episodes_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            episodes.append(json.loads(line))

                for ep_info in episodes:
                    ep_idx = ep_info["episode_index"]
                    ep_length = ep_info["length"]
                    tasks = ep_info.get("tasks", [])
                    instruction = tasks[0] if tasks else ""

                    # 需要至少 n_history_frames+1 帧才能构成有效输入
                    min_current_idx = self.n_history_frames
                    if ep_length <= min_current_idx:
                        continue

                    max_current_idx = ep_length - 1

                    # 正常采样 (与主训练一致的 stride)
                    for current_idx in range(min_current_idx, max_current_idx + 1, sample_stride):
                        steps_to_end = (ep_length - 1) - current_idx
                        stop_label = 1 if steps_to_end <= self.stop_threshold else 0

                        self.samples.append({
                            "scene_path": scene_path,
                            "episode_idx": ep_idx,
                            "current_idx": current_idx,
                            "episode_length": ep_length,
                            "instruction": instruction,
                            "stop_label": stop_label,
                        })

                    # 末端重采样: 最后20%轨迹额外重采样 (stride=1), 与主训练一致
                    terminal_start = max(min_current_idx, int(ep_length * 0.8))
                    for current_idx in range(terminal_start, max_current_idx + 1):
                        steps_to_end = (ep_length - 1) - current_idx
                        stop_label = 1 if steps_to_end <= self.stop_threshold else 0

                        n_extra = int(balance_ratio) - 1
                        for _ in range(n_extra):
                            self.samples.append({
                                "scene_path": scene_path,
                                "episode_idx": ep_idx,
                                "current_idx": current_idx,
                                "episode_length": ep_length,
                                "instruction": instruction,
                                "stop_label": stop_label,
                            })

    def __len__(self):
        return len(self.samples)

    def _get_history_indices(self, current_idx: int) -> list:
        """均匀采样 n_history_frames 个历史帧索引, 和主训练一致。"""
        if current_idx <= 0:
            return [0] * self.n_history_frames
        if current_idx < self.n_history_frames:
            indices = np.linspace(0, current_idx - 1, self.n_history_frames, dtype=int).tolist()
        else:
            indices = np.linspace(0, current_idx - 1, self.n_history_frames, dtype=int).tolist()
        return indices

    def _load_and_resize_frame(self, scene_path, camera_key, episode_idx, frame_idx):
        """加载并resize一帧到 [3, H, W], 范围[-1, 1]。"""
        img_dir = os.path.join(
            scene_path, "videos", "chunk-000",
            f"observation.images.rgb.{camera_key}"
        )
        img_path = os.path.join(img_dir, f"episode_{episode_idx:06d}_{frame_idx}.jpg")
        img = self._Image.open(img_path).convert("RGB")
        img_tensor = self._transforms_F.to_tensor(img)  # [3, H, W] in [0,1]
        img_tensor = self._transforms_F.resize(
            img_tensor, self.video_size,
            interpolation=self._transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        img_tensor = img_tensor * 2.0 - 1.0  # → [-1,1]
        return img_tensor

    def _get_cached_text_context(self, prompt):
        """加载预计算的T5 text embedding。"""
        if self.text_embedding_cache_dir is None:
            context = torch.zeros(self.context_len, 4096)
            context_mask = torch.ones(self.context_len, dtype=torch.bool)
            return context, context_mask

        hashed = self._hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            self.text_embedding_cache_dir,
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not os.path.exists(cache_path):
            context = torch.zeros(self.context_len, 4096)
            context_mask = torch.ones(self.context_len, dtype=torch.bool)
            return context, context_mask

        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        context = payload["context"]       # [context_len, 4096]
        context_mask = payload["mask"].bool()  # [context_len]
        return context, context_mask

    def _get(self, idx):
        sample_info = self.samples[idx]
        scene_path = sample_info["scene_path"]
        episode_idx = sample_info["episode_idx"]
        current_idx = sample_info["current_idx"]
        episode_length = sample_info["episode_length"]
        instruction = sample_info["instruction"]
        stop_label = sample_info["stop_label"]

        # ─── 加载 9 帧 0deg 视频 (8 history + 1 current) ───
        history_indices = self._get_history_indices(current_idx)
        all_indices = history_indices + [current_idx]  # 8 + 1 = 9 帧

        video_frames = []
        for fidx in all_indices:
            frame = self._load_and_resize_frame(
                scene_path, self.primary_camera, episode_idx, fidx
            )
            video_frames.append(frame)

        video = torch.stack(video_frames, dim=1)  # [3, 9, H, W]

        # ─── 加载 1 帧 30deg 下倾图 (当前时刻) ───
        overhead = self._load_and_resize_frame(
            scene_path, self.overhead_camera, episode_idx, current_idx
        )  # [3, H, W]

        # ─── 文本特征 ───
        prompt = f"A video recorded from a navigation agent's point of view executing the following instruction: {instruction}"
        context, context_mask = self._get_cached_text_context(prompt)

        return {
            "video": video,                    # [3, 9, H, W], range [-1,1]
            "overhead": overhead,              # [3, H, W], range [-1,1]
            "context": context,                # [context_len, 4096]
            "context_mask": context_mask,       # [context_len]
            "stop_label": torch.tensor(stop_label, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as e:
            # fallback
            fallback_idx = np.random.randint(len(self))
            try:
                return self._get(fallback_idx)
            except Exception:
                # 极端情况: 返回全零
                return {
                    "video": torch.zeros(3, 9, self.video_size[0], self.video_size[1]),
                    "overhead": torch.zeros(3, self.video_size[0], self.video_size[1]),
                    "context": torch.zeros(self.context_len, 4096),
                    "context_mask": torch.ones(self.context_len, dtype=torch.bool),
                    "stop_label": torch.tensor(0.0),
                }


# ═══════════════════════════════════════════════════════════════════════════════
# Full Stop Prediction Module (frozen VAE + trainable head)
# ═══════════════════════════════════════════════════════════════════════════════

class StopPredictor(nn.Module):
    """
    完整的 Stop 预测器。

    流程:
        1. video [B, 3, 9, H, W] → 冻结VAE → [B, 48, 3, 28, 28] → pool → [B, video_feat_dim]
        2. overhead [B, 3, H, W] → 冻结VAE → [B, 48, 1, 28, 28] → pool → [B, overhead_feat_dim]
        3. text [B, L, 4096] → mean pool → [B, 4096]
        4. 三者融合 → StopHeadModel → logit [B, 1]

    冻结: VAE 全部冻结
    可训练: video_pool_proj + overhead_pool_proj + StopHeadModel
    """

    def __init__(
        self,
        vae,
        text_dim: int = 4096,
        vae_latent_dim: int = 48,
        video_feat_dim: int = 512,
        overhead_feat_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vae = vae
        self.vae_latent_dim = vae_latent_dim

        # 冻结 VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # ─── 可训练的 pooling projections ───
        # Video: VAE输出 [B, 48, 3, 28, 28] → spatial-temporal avg pool → [B, 48]
        #         → Linear → [B, video_feat_dim]
        self.video_pool_proj = nn.Sequential(
            nn.Linear(vae_latent_dim, video_feat_dim),
            nn.GELU(),
        )

        # Overhead: VAE输出 [B, 48, 1, 28, 28] → spatial avg pool → [B, 48]
        #            → Linear → [B, overhead_feat_dim]
        self.overhead_pool_proj = nn.Sequential(
            nn.Linear(vae_latent_dim, overhead_feat_dim),
            nn.GELU(),
        )

        # ─── 分类头 ───
        self.stop_head = StopHeadModel(
            text_dim=text_dim,
            video_feat_dim=video_feat_dim,
            overhead_feat_dim=overhead_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        用冻结VAE编码视频。

        Args:
            video: [B, 3, 9, H, W] in [-1, 1]
                   9帧满足 T%4==1 (9%4=1 ✓), VAE时间压缩后 → 3 latent frames
        Returns:
            z: [B, 48, 3, 28, 28]
        """
        z = self.vae.encode(video, device=video.device)
        return z

    @torch.no_grad()
    def encode_overhead(self, overhead: torch.Tensor) -> torch.Tensor:
        """
        用冻结VAE编码下倾图。

        Args:
            overhead: [B, 3, H, W] in [-1, 1]
        Returns:
            z: [B, 48, 1, 28, 28]
        """
        # VAE expects [B, 3, T, H, W]
        overhead_video = overhead.unsqueeze(2)  # [B, 3, 1, H, W]
        z = self.vae.encode(overhead_video, device=overhead.device)
        return z

    def pool_text(self, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling of text features: [B, L, D] → [B, D]."""
        mask = context_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (context * mask).sum(dim=1)       # [B, D]
        counts = mask.sum(dim=1).clamp(min=1.0)    # [B, 1]
        return summed / counts

    def forward(self, video, overhead, context, context_mask):
        """
        Args:
            video: [B, 3, 9, H, W] — 8帧历史 + 1当前帧
            overhead: [B, 3, H, W] — 下倾图
            context: [B, L, 4096] — 文本特征
            context_mask: [B, L] — 文本mask
        Returns:
            logits: [B, 1]
        """
        # 1. VAE编码 (冻结, no_grad)
        video_latent = self.encode_video(video)       # [B, 48, 3, 28, 28]
        overhead_latent = self.encode_overhead(overhead)  # [B, 48, 1, 28, 28]

        # 2. Pool视觉特征
        # Video: spatial-temporal average pooling → [B, 48]
        video_pooled = video_latent.mean(dim=[2, 3, 4]).float()  # [B, 48]
        video_feat = self.video_pool_proj(video_pooled)           # [B, video_feat_dim]

        # Overhead: spatial average pooling → [B, 48]
        overhead_pooled = overhead_latent.squeeze(2).mean(dim=[2, 3]).float()  # [B, 48]
        overhead_feat = self.overhead_pool_proj(overhead_pooled)   # [B, overhead_feat_dim]

        # 3. Pool文本特征
        text_feat = self.pool_text(context, context_mask)  # [B, 4096]

        # 4. 分类
        logits = self.stop_head(text_feat.float(), video_feat, overhead_feat)  # [B, 1]
        return logits

    def trainable_parameters(self):
        """只返回需要训练的参数。"""
        params = []
        params += list(self.video_pool_proj.parameters())
        params += list(self.overhead_pool_proj.parameters())
        params += list(self.stop_head.parameters())
        return params

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_params(self):
        return sum(p.numel() for p in self.parameters())

    def train(self, mode=True):
        """Override: VAE 永远保持 eval mode。"""
        super().train(mode)
        self.vae.eval()
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train Stop Head (standalone)")
    # Data
    parser.add_argument("--dataset_dirs", type=str, nargs="+", default=[
        "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r",
        "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/rxr",
        "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/scalevln",
    ])
    parser.add_argument("--text_embedding_cache_dir", type=str,
                        default="/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/text_embeds_cache/nav_vln")
    parser.add_argument("--camera_keys", type=str, nargs=2,
                        default=["125cm_0deg", "125cm_30deg"])
    parser.add_argument("--n_history_frames", type=int, default=8)
    parser.add_argument("--video_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--context_len", type=int, default=256)
    parser.add_argument("--stop_threshold", type=int, default=5,
                        help="距离终点<=N步标记为stop=1")
    parser.add_argument("--sample_stride", type=int, default=2,
                        help="数据采样步长 (1=每帧都用, 2=隔一帧)")
    parser.add_argument("--balance_ratio", type=float, default=3.0,
                        help="正样本过采样倍数")

    # Model
    parser.add_argument("--vae_path", type=str,
                        default="/tmp/fastwam_checkpoints",
                        help="VAE checkpoint 路径 (DIFFSYNTH_MODEL_BASE_PATH)")
    parser.add_argument("--video_feat_dim", type=int, default=512)
    parser.add_argument("--overhead_feat_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size (9帧视频比单帧大, 建议32)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every_steps", type=int, default=300,
                        help="每N步保存一次checkpoint")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="使用 bfloat16 混合精度")

    # Output
    parser.add_argument("--output_dir", type=str, default="./runs/stop_head")

    return parser.parse_args()


def load_vae(vae_path, device="cpu", dtype=torch.bfloat16):
    """加载 Wan2.2 VAE 编码器 (只加载VAE, 不加载DiT/T5)。"""
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", vae_path)

    from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components

    # loader 强制要求 dit_config，但我们跳过 DiT 权重加载 (skip_dit_load_from_pretrain=True)
    # 这样 DiT 只随机初始化（非常快），我们只取 VAE
    dummy_dit_config = {
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
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 3,
    }

    components = load_wan22_ti2v_5b_components(
        device=device,
        torch_dtype=dtype,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        dit_config=dummy_dit_config,
        skip_dit_load_from_pretrain=True,  # 不加载6B DiT权重，只随机初始化（我们不需要）
        load_text_encoder=False,           # 不加载T5 (用缓存)
    )
    vae = components.vae
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # 释放不需要的 DiT（节省内存）
    del components.dit
    del components
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return vae


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine LR schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataloader, device, dtype):
    """评估模型在验证集上的表现。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for batch in dataloader:
        video = batch["video"].to(device, dtype=dtype)
        overhead = batch["overhead"].to(device, dtype=dtype)
        context = batch["context"].to(device, dtype=dtype)
        context_mask = batch["context_mask"].to(device)
        labels = batch["stop_label"].to(device)

        logits = model(video, overhead, context, context_mask).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_loss += loss.item() * labels.shape[0]
        total_correct += (preds == labels).sum().item()
        total_samples += labels.shape[0]

        total_tp += ((preds == 1) & (labels == 1)).sum().item()
        total_fp += ((preds == 1) & (labels == 0)).sum().item()
        total_fn += ((preds == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ─── Distributed setup ───
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.fp16 else torch.float32
    is_main = local_rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 70)
        print(" Stop Head Standalone Training")
        print(" 输入: 9帧0deg视频(8历史+1当前) + 1帧30deg下倾 + 文本")
        print("=" * 70)
        print(f" Device: {device}, World size: {world_size}")
        print(f" Stop threshold: {args.stop_threshold} steps")
        print(f" History frames: {args.n_history_frames}")
        print(f" Batch size (per GPU): {args.batch_size}")
        print(f" Learning rate: {args.lr}")
        print(f" Epochs: {args.num_epochs}")
        print(f" Output: {args.output_dir}")
        print("=" * 70)

    # ─── Seed ───
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # ─── Dataset ───
    if is_main:
        print("\n[1/4] Building dataset...")

    full_dataset = StopPredictionDataset(
        dataset_dirs=args.dataset_dirs,
        camera_keys=args.camera_keys,
        n_history_frames=args.n_history_frames,
        video_size=args.video_size,
        text_embedding_cache_dir=args.text_embedding_cache_dir,
        context_len=args.context_len,
        sample_stride=args.sample_stride,
        stop_threshold=args.stop_threshold,
        balance_ratio=args.balance_ratio,
    )

    # Train/Val split (90%/10%)
    n_total = len(full_dataset)
    n_val = max(int(n_total * 0.1), 100)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if is_main:
        print(f"   Train: {n_train}, Val: {n_val}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ─── Load VAE (frozen) ───
    if is_main:
        print("\n[2/4] Loading frozen VAE encoder...")
        print(f"   VAE path: {args.vae_path}")
        print(f"   (Only VAE loaded — NO DiT, NO T5 text encoder)")

    vae = load_vae(args.vae_path, device="cpu", dtype=dtype)

    # ─── Build model ───
    if is_main:
        print("\n[3/4] Building StopPredictor model...")

    model = StopPredictor(
        vae=vae,
        text_dim=4096,
        vae_latent_dim=48,
        video_feat_dim=args.video_feat_dim,
        overhead_feat_dim=args.overhead_feat_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = model.to(device)

    if is_main:
        n_trainable = model.num_trainable_params()
        n_total_params = model.num_total_params()
        print(f"   Trainable params: {n_trainable:,} (~{n_trainable/1e6:.2f}M)")
        print(f"   Total params (incl. frozen VAE): {n_total_params:,} (~{n_total_params/1e6:.1f}M)")
        print(f"   VAE is FROZEN — only head parameters are updated")

    # DDP
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=False,  # 所有参数都参与forward
        )
        raw_model = model.module
    else:
        raw_model = model

    # ─── Optimizer (只更新 head 参数) ───
    if is_main:
        print("\n[4/4] Setting up optimizer...")

    trainable_params = raw_model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    if is_main:
        print(f"   Total training steps: {total_steps}")
        print(f"   Warmup steps: {args.warmup_steps}")

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # ─── Training ───
    if is_main:
        print("\n" + "=" * 70)
        print(" Starting training...")
        print("=" * 70)

    global_step = 0
    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(args.num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            video = batch["video"].to(device, dtype=dtype, non_blocking=True)
            overhead = batch["overhead"].to(device, dtype=dtype, non_blocking=True)
            context = batch["context"].to(device, dtype=dtype, non_blocking=True)
            context_mask = batch["context_mask"].to(device, non_blocking=True)
            labels = batch["stop_label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=args.fp16, dtype=dtype):
                logits = model(video, overhead, context, context_mask).squeeze(-1)  # [B]
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Stats
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == labels).sum().item()

            epoch_loss += loss.item() * labels.shape[0]
            epoch_correct += correct
            epoch_samples += labels.shape[0]
            global_step += 1

            # Logging
            if is_main and global_step % args.log_every == 0:
                avg_loss = epoch_loss / max(epoch_samples, 1)
                avg_acc = epoch_correct / max(epoch_samples, 1)
                elapsed = time.time() - start_time
                lr_now = scheduler.get_last_lr()[0]
                samples_per_sec = epoch_samples / max(elapsed, 1)
                print(
                    f"  [epoch {epoch+1}/{args.num_epochs}] "
                    f"step {global_step}/{total_steps} | "
                    f"loss={avg_loss:.4f} | acc={avg_acc:.4f} | "
                    f"lr={lr_now:.2e} | "
                    f"{samples_per_sec:.0f} samples/s | "
                    f"elapsed={elapsed:.0f}s"
                )

            # Eval
            if is_main and global_step % args.eval_every == 0:
                val_metrics = evaluate(raw_model, val_loader, device, dtype)
                print(
                    f"  [EVAL] step {global_step} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"acc={val_metrics['accuracy']:.4f} | "
                    f"P={val_metrics['precision']:.4f} | "
                    f"R={val_metrics['recall']:.4f} | "
                    f"F1={val_metrics['f1']:.4f}"
                )
                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    save_path = os.path.join(args.output_dir, "best_stop_head.pt")
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": {
                            "video_pool_proj": raw_model.video_pool_proj.state_dict(),
                            "overhead_pool_proj": raw_model.overhead_pool_proj.state_dict(),
                            "stop_head": raw_model.stop_head.state_dict(),
                        },
                        "metrics": val_metrics,
                        "args": vars(args),
                    }, save_path)
                    print(f"  [SAVE] New best F1={best_f1:.4f} → {save_path}")
                model.train()

            # 每 N 步保存 checkpoint
            if is_main and global_step % args.save_every_steps == 0:
                save_path = os.path.join(args.output_dir, f"stop_head_step{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "model_state_dict": {
                        "video_pool_proj": raw_model.video_pool_proj.state_dict(),
                        "overhead_pool_proj": raw_model.overhead_pool_proj.state_dict(),
                        "stop_head": raw_model.stop_head.state_dict(),
                    },
                    "args": vars(args),
                }, save_path)
                print(f"  [SAVE] Checkpoint step {global_step} → {save_path}")

        # End of epoch
        if is_main:
            epoch_avg_loss = epoch_loss / max(epoch_samples, 1)
            epoch_avg_acc = epoch_correct / max(epoch_samples, 1)
            print(f"\n  === Epoch {epoch+1} done === "
                  f"loss={epoch_avg_loss:.4f}, acc={epoch_avg_acc:.4f}\n")

    # ─── Final save ───
    if is_main:
        save_path = os.path.join(args.output_dir, "stop_head_final.pt")
        torch.save({
            "step": global_step,
            "epoch": args.num_epochs,
            "model_state_dict": {
                "video_pool_proj": raw_model.video_pool_proj.state_dict(),
                "overhead_pool_proj": raw_model.overhead_pool_proj.state_dict(),
                "stop_head": raw_model.stop_head.state_dict(),
            },
            "args": vars(args),
        }, save_path)
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f" Training complete!")
        print(f" Total time: {total_time/3600:.2f} hours")
        print(f" Best val F1: {best_f1:.4f}")
        print(f" Final checkpoint: {save_path}")
        print("=" * 70)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
