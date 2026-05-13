"""
Navigation Video Dataset for FastWAM (Multi-Frame History + Overhead Conditioning).

Reads VLN trajectory data in LeRobot format (parquet + jpg images) and produces
samples compatible with the FastWAM training pipeline.

Architecture:
  - 9 condition frames (125cm_0deg): 8 uniformly sampled history + current frame
  - 8 future frames (125cm_0deg): for video generation training
  - 1 overhead frame (125cm_30deg): at current timestep, separate conditioning
  - Total 0deg video: 17 frames (T%4==1 ✓) → 5 VAE latent frames
  - Action: 32 relative waypoints starting from current frame

Each sample contains:
  - video: [C, 17, H, W] — 0deg single-camera RGB video (9 cond + 8 future)
  - overhead: [C, H, W] — 30deg overhead frame at current timestep
  - action: [32, 3] — relative (x, y, theta) trajectory waypoints
  - action_is_pad: [32] — padding mask for action
  - context: [context_len, text_dim] — cached T5 text embedding
  - context_mask: [context_len] — text mask
  - image_is_pad: [17] — video frame padding mask
  - n_cond_frames: int — number of condition frames (9)
"""

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT = "A video recorded from a navigation agent's point of view executing the following instruction: {task}"


class NavVideoDataset(torch.utils.data.Dataset):
    """
    Dataset for VLN navigation trajectories with multi-frame history + overhead conditioning.

    Args:
        dataset_dirs: List of scene root directories.
        camera_keys: [primary_camera, overhead_camera], e.g. ["125cm_0deg", "125cm_30deg"].
        num_frames: Total trajectory steps spanned per sample (default 33 = 32 action steps + 1).
        n_history_frames: Number of past frames to include as condition (default 8).
        n_future_video_frames: Number of future frames for video generation (default 8).
        video_size: [H, W] for each single-camera frame.
        text_embedding_cache_dir: Path to pre-computed text embeddings.
        context_len: Text context length.
        sample_stride: Stride for sampling start frames within episodes.
        terminal_oversample_ratio: How much more to sample near trajectory end (default 3.0).
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        camera_keys: List[str] = None,
        num_frames: int = 33,
        n_history_frames: int = 8,
        n_future_video_frames: int = 8,
        action_video_freq_ratio: int = 4,
        video_size: List[int] = None,
        concat_multi_camera: str = "none",
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 256,
        sample_stride: int = 4,
        terminal_oversample_ratio: float = 3.0,
        # Unused kwargs for compatibility with hydra config
        **kwargs,
    ):
        super().__init__()
        if camera_keys is None:
            camera_keys = ["125cm_0deg", "125cm_30deg"]
        if video_size is None:
            video_size = [224, 224]

        self.primary_camera = camera_keys[0]  # 125cm_0deg
        self.overhead_camera = camera_keys[1] if len(camera_keys) > 1 else camera_keys[0]
        self.camera_keys = camera_keys
        self.num_frames = num_frames  # trajectory span (33 = 32 action steps + 1)
        self.n_history_frames = n_history_frames  # 8
        self.n_future_video_frames = n_future_video_frames  # 8
        self.action_video_freq_ratio = action_video_freq_ratio  # 4
        self.video_size = video_size  # [224, 224] single camera
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.sample_stride = sample_stride
        self.terminal_oversample_ratio = terminal_oversample_ratio

        # Action parameters
        self.num_action_steps = num_frames - 1  # 32

        # Total video frames for VAE: history + current + future = 9 + 8 = 17
        self.n_cond_frames = n_history_frames + 1  # 9 (frozen in latent space)
        self.total_video_frames = self.n_cond_frames + n_future_video_frames  # 17
        assert self.total_video_frames % 4 == 1, (
            f"Total video frames must satisfy T%4==1 for VAE, got {self.total_video_frames}"
        )

        # Future frame stride: span 32 action steps with n_future_video_frames frames
        self.future_frame_stride = self.num_action_steps // n_future_video_frames  # 4

        # Build index
        self.samples = []
        self._build_index(dataset_dirs)
        logger.info(
            f"NavVideoDataset: {len(self.samples)} samples from {len(dataset_dirs)} dataset dirs, "
            f"primary_cam={self.primary_camera}, overhead_cam={self.overhead_camera}, "
            f"n_history={n_history_frames}, n_future_video={n_future_video_frames}, "
            f"total_video_frames={self.total_video_frames}, action_steps={self.num_action_steps}"
        )

    def _build_index(self, dataset_dirs: List[str]):
        """Scan all scenes and episodes to build sample index with terminal oversampling."""
        for dataset_dir in dataset_dirs:
            if not os.path.isdir(dataset_dir):
                logger.warning(f"Dataset dir not found: {dataset_dir}")
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

                    # Need at least n_history_frames + 1 (current) to form a valid sample
                    min_current_idx = self.n_history_frames
                    if ep_length <= min_current_idx:
                        continue

                    # Create samples: current_idx ranges from n_history_frames to ep_length-1
                    # The "current frame" is where the agent is NOW; history is behind, future is ahead
                    max_current_idx = ep_length - 1

                    for current_idx in range(min_current_idx, max_current_idx + 1, self.sample_stride):
                        self.samples.append({
                            "scene_path": scene_path,
                            "episode_idx": ep_idx,
                            "current_idx": current_idx,
                            "episode_length": ep_length,
                            "instruction": instruction,
                        })

                    # Terminal oversampling: add extra samples near the end
                    # (last 20% of trajectory, with stride=1)
                    terminal_start = max(min_current_idx, int(ep_length * 0.8))
                    for current_idx in range(terminal_start, max_current_idx + 1):
                        # Add extra copies based on terminal_oversample_ratio
                        n_extra = int(self.terminal_oversample_ratio) - 1
                        for _ in range(n_extra):
                            self.samples.append({
                                "scene_path": scene_path,
                                "episode_idx": ep_idx,
                                "current_idx": current_idx,
                                "episode_length": ep_length,
                                "instruction": instruction,
                            })

    def __len__(self):
        return len(self.samples)

    def _load_image(self, scene_path: str, camera_key: str, episode_idx: int, frame_idx: int) -> torch.Tensor:
        """Load a single image as tensor [C, H, W] in [0, 1]."""
        img_dir = os.path.join(
            scene_path, "videos", "chunk-000",
            f"observation.images.rgb.{camera_key}"
        )
        img_path = os.path.join(img_dir, f"episode_{episode_idx:06d}_{frame_idx}.jpg")
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms_F.to_tensor(img)  # [C, H, W] in [0, 1]
        return img_tensor

    def _load_and_resize_frame(self, scene_path: str, camera_key: str, episode_idx: int, frame_idx: int) -> torch.Tensor:
        """Load and resize a single frame to target video_size."""
        img = self._load_image(scene_path, camera_key, episode_idx, frame_idx)
        img = transforms_F.resize(
            img, self.video_size,
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return img  # [C, H, W] in [0, 1]

    def _load_poses(self, scene_path: str, episode_idx: int, camera_key: str) -> np.ndarray:
        """Load all poses for an episode. Returns [N, 4, 4] array."""
        parquet_path = os.path.join(
            scene_path, "data", "chunk-000",
            f"episode_{episode_idx:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
        poses_raw = df[f"pose.{camera_key}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])  # [N, 4, 4]
        return poses

    def _compute_relative_actions(
        self, poses: np.ndarray, current_idx: int, num_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute relative (x, y, theta) actions from current frame.

        Returns:
            actions: [num_steps, 3] relative (x, y, theta) waypoints.
            is_pad: [num_steps] boolean mask (True = padded beyond trajectory end).
        """
        T_base = poses[current_idx]
        T_base_inv = np.linalg.inv(T_base)

        actions = np.zeros((num_steps, 3), dtype=np.float32)
        is_pad = np.ones(num_steps, dtype=bool)

        for i in range(num_steps):
            frame_j = current_idx + i + 1
            if frame_j >= len(poses):
                # Beyond trajectory end: repeat last valid relative pose (= stay)
                if i > 0:
                    actions[i] = actions[i - 1]
                continue

            T_rel = T_base_inv @ poses[frame_j]
            local_pos = T_rel[:3, 3]
            R_rel = T_rel[:3, :3]
            theta = np.arctan2(R_rel[0, 2], R_rel[2, 2])

            actions[i] = [local_pos[0], local_pos[2], theta]
            is_pad[i] = False

        return actions, is_pad

    def _get_history_indices(self, current_idx: int) -> List[int]:
        """
        Uniformly sample n_history_frames indices from [0, current_idx-1].
        If trajectory is shorter than n_history_frames, repeat first frame.
        """
        if current_idx <= 0:
            return [0] * self.n_history_frames

        if current_idx < self.n_history_frames:
            # Not enough frames: sample what we have, pad with first frame
            available = list(range(current_idx))
            # Uniformly sample from available, then pad
            indices = np.linspace(0, current_idx - 1, self.n_history_frames, dtype=int).tolist()
        else:
            # Uniformly sample 8 frames from [0, current_idx-1]
            indices = np.linspace(0, current_idx - 1, self.n_history_frames, dtype=int).tolist()

        return indices

    def _get_future_indices(self, current_idx: int, episode_length: int) -> List[int]:
        """
        Get n_future_video_frames indices after current, with stride to span 32 action steps.
        If beyond episode end, clamp to last valid frame.
        """
        indices = []
        for i in range(1, self.n_future_video_frames + 1):
            fidx = current_idx + i * self.future_frame_stride
            fidx = min(fidx, episode_length - 1)  # clamp to trajectory end
            indices.append(fidx)
        return indices

    def _get(self, idx: int) -> dict:
        """Get a single sample."""
        sample_info = self.samples[idx]
        scene_path = sample_info["scene_path"]
        episode_idx = sample_info["episode_idx"]
        current_idx = sample_info["current_idx"]
        episode_length = sample_info["episode_length"]
        instruction = sample_info["instruction"]

        # --- Frame indices ---
        history_indices = self._get_history_indices(current_idx)  # [8]
        future_indices = self._get_future_indices(current_idx, episode_length)  # [8]
        # All 0deg frame indices: history(8) + current(1) + future(8) = 17
        all_0deg_indices = history_indices + [current_idx] + future_indices

        # --- Load 0deg video frames (17 frames) ---
        video_frames = []
        image_is_pad = []
        for fidx in all_0deg_indices:
            if fidx >= episode_length:
                # Pad with last valid frame
                if video_frames:
                    video_frames.append(video_frames[-1].clone())
                else:
                    video_frames.append(torch.zeros(3, self.video_size[0], self.video_size[1]))
                image_is_pad.append(True)
            else:
                frame = self._load_and_resize_frame(scene_path, self.primary_camera, episode_idx, fidx)
                video_frames.append(frame)
                image_is_pad.append(False)

        video = torch.stack(video_frames, dim=0)  # [17, C, H, W]
        video = video * 2.0 - 1.0  # [0,1] → [-1,1]
        video = video.permute(1, 0, 2, 3)  # [C, 17, H, W]

        # --- Load overhead frame (30deg at current timestep) ---
        if current_idx < episode_length:
            overhead = self._load_and_resize_frame(scene_path, self.overhead_camera, episode_idx, current_idx)
        else:
            overhead = torch.zeros(3, self.video_size[0], self.video_size[1])
        overhead = overhead * 2.0 - 1.0  # [0,1] → [-1,1]
        # overhead shape: [C, H, W]

        # --- Actions (32 steps from current frame) ---
        poses = self._load_poses(scene_path, episode_idx, self.primary_camera)
        actions, action_is_pad = self._compute_relative_actions(poses, current_idx, self.num_action_steps)

        # --- Text context ---
        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, context_mask = self._get_cached_text_context(prompt)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        # --- Tensors ---
        # Append moving flag as 4th dimension: 1.0=moving, 0.0=stopped (reached goal)
        moving_flag = (~action_is_pad).astype(np.float32).reshape(-1, 1)  # [32, 1]
        actions_with_flag = np.concatenate([actions, moving_flag], axis=1)  # [32, 4]
        action_tensor = torch.from_numpy(actions_with_flag).float()  # [32, 4]
        action_is_pad_tensor = torch.from_numpy(action_is_pad).bool()  # [32]
        image_is_pad_tensor = torch.tensor(image_is_pad, dtype=torch.bool)  # [17]

        data = {
            "video": video,                         # [C, 17, H, W]
            "overhead": overhead,                   # [C, H, W]
            "action": action_tensor,                # [32, 4]
            "action_is_pad": action_is_pad_tensor,  # [32]
            "context": context,                     # [context_len, 4096]
            "context_mask": context_mask,            # [context_len]
            "image_is_pad": image_is_pad_tensor,    # [17]
            "n_cond_frames": self.n_cond_frames,    # 9
            "prompt": prompt,
        }
        return data

    def _get_cached_text_context(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load pre-computed text embedding from cache."""
        if self.text_embedding_cache_dir is None:
            context = torch.zeros(self.context_len, 4096)
            context_mask = torch.ones(self.context_len, dtype=torch.bool)
            return context, context_mask
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            logger.warning(
                f"Missing text embedding cache (using zeros): {cache_path}. "
                "Run scripts/precompute_nav_text_embeds.py to pre-compute all embeddings."
            )
            context = torch.zeros(self.context_len, 4096)
            context_mask = torch.ones(self.context_len, dtype=torch.bool)
            return context, context_mask
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]  # [context_len, text_dim]
        context_mask = payload["mask"].bool()  # [context_len]
        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            logger.warning(f"Error processing sample idx {idx}: {e}")
            logger.warning(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
