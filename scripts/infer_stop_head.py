"""
Stop Head Inference Script.

加载训练好的 stop head checkpoint，给定当前观测 (历史视频+下倾+文本) 预测是否停止。
可以集成到 navigation inference pipeline 中。

Usage:
    python scripts/infer_stop_head.py \
        --checkpoint ./runs/stop_head/best_stop_head.pt \
        --scene_path /path/to/scene \
        --episode_idx 0 \
        --current_idx 50 \
        --instruction "Go to the kitchen and turn left"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_stop_head import StopPredictor, load_vae


def load_stop_predictor(checkpoint_path, vae_path, device="cuda", dtype=torch.bfloat16):
    """加载训练好的 StopPredictor。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args_dict = ckpt.get("args", {})
    video_feat_dim = args_dict.get("video_feat_dim", 512)
    overhead_feat_dim = args_dict.get("overhead_feat_dim", 256)
    hidden_dim = args_dict.get("hidden_dim", 256)
    dropout = args_dict.get("dropout", 0.1)

    # Load VAE
    vae = load_vae(vae_path, device="cpu", dtype=dtype)

    # Build model
    model = StopPredictor(
        vae=vae,
        text_dim=4096,
        vae_latent_dim=48,
        video_feat_dim=video_feat_dim,
        overhead_feat_dim=overhead_feat_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # Load trained weights
    model_state = ckpt["model_state_dict"]
    model.video_pool_proj.load_state_dict(model_state["video_pool_proj"])
    model.overhead_pool_proj.load_state_dict(model_state["overhead_pool_proj"])
    model.stop_head.load_state_dict(model_state["stop_head"])

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_stop(
    model,
    video: torch.Tensor,
    overhead: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    threshold: float = 0.5,
    device="cuda",
    dtype=torch.bfloat16,
):
    """
    预测是否应该停止。

    Args:
        model: StopPredictor
        video: [3, 9, H, W] or [B, 3, 9, H, W] — 8历史+1当前
        overhead: [3, H, W] or [B, 3, H, W] — 下倾图
        context: [L, D] or [B, L, D] — 文本特征
        context_mask: [L] or [B, L]
        threshold: stop 概率阈值

    Returns:
        stop_prob: float
        should_stop: bool
    """
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if overhead.ndim == 3:
        overhead = overhead.unsqueeze(0)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)

    video = video.to(device, dtype=dtype)
    overhead = overhead.to(device, dtype=dtype)
    context = context.to(device, dtype=dtype)
    context_mask = context_mask.to(device)

    logits = model(video, overhead, context, context_mask)
    prob = torch.sigmoid(logits).item()
    should_stop = prob > threshold
    return prob, should_stop


class StopHeadNavigationHelper:
    """
    封装好的 helper class，可直接集成到 navigation pipeline。

    Usage:
        helper = StopHeadNavigationHelper(
            checkpoint_path="runs/stop_head/best_stop_head.pt",
            vae_path="/tmp/fastwam_checkpoints",
        )

        # 在导航循环中:
        should_stop = helper.should_stop(
            history_frames=[img0, img1, ..., img7],  # 8 PIL Images
            current_frame=current_img,               # PIL Image
            overhead_frame=overhead_img,              # PIL Image
            instruction="Go to the kitchen",
        )
    """

    def __init__(
        self,
        checkpoint_path: str,
        vae_path: str,
        text_embedding_cache_dir: str = None,
        context_len: int = 256,
        video_size=(224, 224),
        threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.video_size = list(video_size)
        self.threshold = threshold
        self.context_len = context_len
        self.text_embedding_cache_dir = text_embedding_cache_dir

        self.model = load_stop_predictor(checkpoint_path, vae_path, device, self.dtype)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """PIL Image → [3, H, W] tensor in [-1, 1]."""
        t = transforms_F.to_tensor(img)
        t = transforms_F.resize(t, self.video_size, antialias=True)
        t = t * 2.0 - 1.0
        return t

    def _load_text_embedding(self, instruction: str):
        """加载预缓存的文本特征。"""
        import hashlib
        prompt = f"A video recorded from a navigation agent's point of view executing the following instruction: {instruction}"

        if self.text_embedding_cache_dir:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = os.path.join(
                self.text_embedding_cache_dir,
                f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
            )
            if os.path.exists(cache_path):
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                return payload["context"], payload["mask"].bool()

        return torch.zeros(self.context_len, 4096), torch.ones(self.context_len, dtype=torch.bool)

    @torch.no_grad()
    def should_stop(
        self,
        history_frames: list,
        current_frame,
        overhead_frame,
        instruction: str,
    ) -> bool:
        """
        判断是否应该停止。

        Args:
            history_frames: list of 8 PIL Images (历史帧)
            current_frame: PIL Image (当前帧)
            overhead_frame: PIL Image (下倾图)
            instruction: str (导航指令)

        Returns:
            should_stop: bool
        """
        # Build video tensor: 8 history + 1 current = 9 frames
        frames = []
        for img in history_frames:
            frames.append(self._pil_to_tensor(img))
        frames.append(self._pil_to_tensor(current_frame))
        video = torch.stack(frames, dim=1)  # [3, 9, H, W]

        overhead = self._pil_to_tensor(overhead_frame)  # [3, H, W]
        context, context_mask = self._load_text_embedding(instruction)

        prob, stop = predict_stop(
            self.model, video, overhead, context, context_mask,
            threshold=self.threshold, device=self.device, dtype=self.dtype,
        )
        return stop

    @torch.no_grad()
    def get_stop_probability(
        self,
        history_frames: list,
        current_frame,
        overhead_frame,
        instruction: str,
    ) -> float:
        """返回停止概率 (0~1)。"""
        frames = []
        for img in history_frames:
            frames.append(self._pil_to_tensor(img))
        frames.append(self._pil_to_tensor(current_frame))
        video = torch.stack(frames, dim=1)

        overhead = self._pil_to_tensor(overhead_frame)
        context, context_mask = self._load_text_embedding(instruction)

        prob, _ = predict_stop(
            self.model, video, overhead, context, context_mask,
            threshold=self.threshold, device=self.device, dtype=self.dtype,
        )
        return prob


def main():
    parser = argparse.ArgumentParser(description="Stop Head Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="/tmp/fastwam_checkpoints")
    parser.add_argument("--scene_path", type=str, required=True,
                        help="Scene directory path")
    parser.add_argument("--episode_idx", type=int, required=True)
    parser.add_argument("--current_idx", type=int, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--n_history_frames", type=int, default=8)
    parser.add_argument("--text_embedding_cache_dir", type=str,
                        default="/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/text_embeds_cache/nav_vln")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--context_len", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Loading model from: {args.checkpoint}")
    model = load_stop_predictor(args.checkpoint, args.vae_path, device, dtype)

    # ─── 加载帧 ───
    video_size = [224, 224]
    primary_camera = "125cm_0deg"
    overhead_camera = "125cm_30deg"

    def load_frame(camera, frame_idx):
        img_dir = os.path.join(
            args.scene_path, "videos", "chunk-000",
            f"observation.images.rgb.{camera}"
        )
        img_path = os.path.join(img_dir, f"episode_{args.episode_idx:06d}_{frame_idx}.jpg")
        img = Image.open(img_path).convert("RGB")
        t = transforms_F.to_tensor(img)
        t = transforms_F.resize(t, video_size, antialias=True)
        t = t * 2.0 - 1.0
        return t

    # History indices (uniform sampling)
    current_idx = args.current_idx
    if current_idx < args.n_history_frames:
        history_indices = np.linspace(0, max(current_idx - 1, 0), args.n_history_frames, dtype=int).tolist()
    else:
        history_indices = np.linspace(0, current_idx - 1, args.n_history_frames, dtype=int).tolist()

    # Load 9 frames (8 history + 1 current)
    frames = []
    for fidx in history_indices:
        frames.append(load_frame(primary_camera, fidx))
    frames.append(load_frame(primary_camera, current_idx))
    video = torch.stack(frames, dim=1)  # [3, 9, H, W]

    # Load overhead
    overhead = load_frame(overhead_camera, current_idx)  # [3, H, W]

    # Load text embedding
    import hashlib
    prompt = f"A video recorded from a navigation agent's point of view executing the following instruction: {args.instruction}"
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = os.path.join(
        args.text_embedding_cache_dir,
        f"{hashed}.t5_len{args.context_len}.wan22ti2v5b.pt"
    )
    if os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        context = payload["context"]
        context_mask = payload["mask"].bool()
    else:
        print(f"[WARN] Text embedding not cached: {cache_path}")
        context = torch.zeros(args.context_len, 4096)
        context_mask = torch.ones(args.context_len, dtype=torch.bool)

    # Predict
    prob, should_stop = predict_stop(
        model, video, overhead, context, context_mask,
        threshold=args.threshold, device=device, dtype=dtype,
    )

    print(f"\n{'='*50}")
    print(f" Instruction: {args.instruction}")
    print(f" Scene: {args.scene_path}")
    print(f" Episode: {args.episode_idx}, Step: {current_idx}")
    print(f" History frames: {history_indices}")
    print(f"{'='*50}")
    print(f" Stop probability: {prob:.4f}")
    print(f" Should stop (threshold={args.threshold}): {'YES ■' if should_stop else 'NO →'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
