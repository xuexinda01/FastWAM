"""
Navigation Video Dataset for FastWAM.

Reads VLN trajectory data in LeRobot format (parquet + jpg images) and produces
samples compatible with the FastWAM training pipeline.

Architecture:
  - 9 condition frames (125cm_0deg): 8 uniformly sampled history + current frame
  - 8 future frames (125cm_0deg): stride=2 over 16-frame action horizon
  - Total 0deg video: 17 frames (T%4==1 ✓) → 5 VAE latent frames
  - Action: predict_step_num waypoints from cubic spline resampling of 16-frame segment

Video and action are aligned:
  - Action horizon = 16 frames from current frame
  - Video future = 8 frames at stride 2, covering the same 16-frame span
  - Near trajectory end, both video and action use shorter remaining trajectory

Each sample contains:
  - video: [C, 17, H, W] — 0deg single-camera RGB video (9 cond + 8 future)
  - action: [predict_step_num, 4] — relative (dx, dy, d_theta, moving_flag)
  - action_is_pad: [predict_step_num] — padding mask for action
  - context: [context_len, text_dim] — cached T5 text embedding
  - context_mask: [context_len] — text mask
  - image_is_pad: [17] — video frame padding mask
  - n_cond_frames: int — number of condition frames (9)
"""

import hashlib
import json
import os
import traceback
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image
from scipy.interpolate import CubicSpline

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT = "A video recorded from a navigation agent's point of view executing the following instruction: {task}"


# =============================================================================
# Trajectory processing utilities
# =============================================================================


def get_trajectory_relative_to_frame(extrinsics: np.ndarray, camera_deg: float = 0) -> np.ndarray:
    """
    Calculate trajectory poses (x, y, yaw) relative to the first frame.

    Args:
        extrinsics: Sequence of 4x4 extrinsic matrices, shape (N, 4, 4).
        camera_deg: Camera pitch angle in degrees.

    Returns:
        relative_xyyaw: shape (N, 3) — (x, y, yaw) relative to frame[0].
    """
    T_camera2robot = np.array(
        [[[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    T_robot2camera = np.array(
        [[[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    if camera_deg is not None and camera_deg != 0:
        camera_rad = np.radians(camera_deg)
        T_deg = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
                    [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
        T_robot2camera = np.matmul(T_robot2camera, T_deg)
        T_camera2robot = np.linalg.inv(T_robot2camera)

    extrinsics_robot = np.matmul(extrinsics, T_camera2robot)

    T_ref = extrinsics_robot[0]
    T_ref_inv = np.linalg.inv(T_ref)

    relative_to_ref = np.matmul(T_ref_inv[np.newaxis, :, :], extrinsics_robot)

    relative_translations = relative_to_ref[:, :2, 3]
    relative_yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])

    relative_xyyaw = np.concatenate((relative_translations, relative_yaws.reshape(-1, 1)), axis=-1)
    return relative_xyyaw


def smooth_and_resample_trajectory(points: np.ndarray, sample_length: int = 33, interval: float = 0.1) -> np.ndarray:
    """
    Smooth trajectory with cubic spline and resample at equal distance intervals.

    Args:
        points: (M, 2) array of x,y waypoints.
        sample_length: Number of output points.
        interval: Distance between consecutive output points (meters).

    Returns:
        resampled: (sample_length, 2) array.
    """
    total_distance = sample_length * interval

    if len(points) == 0:
        return np.zeros((sample_length, 2))

    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))

    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_distances = np.cumsum(segment_lengths)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)

    if len(points) > 3:
        cs_x = CubicSpline(cumulative_distances, points[:, 0])
        cs_y = CubicSpline(cumulative_distances, points[:, 1])

        dense_distances = np.linspace(0, cumulative_distances[-1], max(50, len(points) * 2))
        x_smooth = cs_x(dense_distances)
        y_smooth = cs_y(dense_distances)
        smoothed_points = np.column_stack((x_smooth, y_smooth))

        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_segment_lengths = np.sqrt(np.sum(smooth_diff**2, axis=1))
        smooth_cumulative_distances = np.cumsum(smooth_segment_lengths)
        smooth_cumulative_distances = np.insert(smooth_cumulative_distances, 0, 0)
    else:
        smoothed_points = points
        smooth_cumulative_distances = cumulative_distances

    target_distances = np.linspace(0, total_distance, sample_length)

    resampled = np.zeros((sample_length, 2))

    for i, target_dist in enumerate(target_distances):
        if target_dist >= smooth_cumulative_distances[-1]:
            resampled[i] = smoothed_points[-1]
            continue

        segment_idx = np.searchsorted(smooth_cumulative_distances, target_dist, side='right') - 1
        start_dist = smooth_cumulative_distances[segment_idx]
        end_dist = smooth_cumulative_distances[segment_idx + 1]
        t = (target_dist - start_dist) / (end_dist - start_dist + 1e-8)

        resampled[i] = smoothed_points[segment_idx] + t * (
            smoothed_points[segment_idx + 1] - smoothed_points[segment_idx]
        )

    return resampled


def xy_to_delta_xyt(xy_actions: np.ndarray) -> np.ndarray:
    """Convert absolute (x, y) positions to relative (dx, dy, delta_yaw)."""
    vectors = np.diff(xy_actions, axis=0)
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])

    delta_yaw = np.diff(yaw)
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    delta_yaw = np.concatenate([[yaw[0]], delta_yaw])

    delta_xyt = np.concatenate([vectors, delta_yaw[:, None]], axis=1)
    return delta_xyt


def interpolate_and_resample_trajectory(
    absolute_trajectories: np.ndarray, predict_step_num: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: filter static points → cubic spline → equal-distance resample.

    Args:
        absolute_trajectories: (N, 3) relative (x, y, yaw) from get_trajectory_relative_to_frame.
        predict_step_num: Number of output action steps.

    Returns:
        resampled_trajectories: (predict_step_num + 1, 2) resampled xy positions.
        resampled_relative_poses: (predict_step_num, 3) delta (dx, dy, d_yaw) actions.
    """
    start_point = np.array([[0.0, 0.0]])

    traj = absolute_trajectories[..., :2]
    steps = traj[1:] - traj[:-1]
    steps_sq = (steps**2).sum(axis=-1)
    mask = steps_sq > 0.05

    filtered_traj = traj[1:][mask]
    filtered_traj = np.concatenate([start_point, filtered_traj], axis=0)

    resampled_trajectories = smooth_and_resample_trajectory(filtered_traj, sample_length=predict_step_num + 1)
    resampled_relative_poses = xy_to_delta_xyt(resampled_trajectories)

    resampled_relative_poses[:, 0:2] *= 4  # normalization factor

    return resampled_trajectories, resampled_relative_poses


def clip_or_pad(arr: np.ndarray, fixed_len: int) -> np.ndarray:
    """Clip or zero-pad array to fixed length along dim 0."""
    T, D = arr.shape
    if T >= fixed_len:
        return arr[:fixed_len]
    else:
        pad = np.zeros((fixed_len - T, D), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)


# =============================================================================
# Dataset
# =============================================================================


class NavVideoDataset(torch.utils.data.Dataset):
    """
    Navigation video dataset with aligned video and action horizons.

    Design:
      - Action horizon: 16 frames from current frame (or less near trajectory end)
      - Video future: 8 frames at stride 2, covering the same 16-frame span
      - Action labels: cubic spline interpolation on the 16-frame segment, resampled
        to predict_step_num waypoints
      - Near trajectory end: shorter action/video horizon (terminal oversampling)

    Args:
        dataset_dirs: List of scene root directories.
        camera_keys: [primary_camera, overhead_camera].
        num_frames: Action horizon in raw frames (default 16).
        n_history_frames: Number of past frames as condition (default 8).
        n_future_video_frames: Number of future video frames (default 8).
        video_size: [H, W] for each frame.
        text_embedding_cache_dir: Path to pre-computed text embeddings.
        context_len: Text context length.
        sample_stride: Stride for sampling start frames within episodes.
        terminal_oversample_ratio: Extra sampling ratio near trajectory end.
        predict_step_num: Number of action waypoints output (after spline resampling).
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        camera_keys: List[str] = None,
        num_frames: int = 16,
        n_history_frames: int = 8,
        n_future_video_frames: int = 8,
        action_video_freq_ratio: int = 2,
        video_size: List[int] = None,
        concat_multi_camera: str = "none",
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 256,
        sample_stride: int = 4,
        terminal_oversample_ratio: float = 3.0,
        predict_step_num: int = 32,
        min_goal_len: int = 3,
        **kwargs,
    ):
        super().__init__()
        if camera_keys is None:
            camera_keys = ["125cm_0deg", "125cm_30deg"]
        if video_size is None:
            video_size = [224, 224]

        self.primary_camera = camera_keys[0]
        self.overhead_camera = camera_keys[1] if len(camera_keys) > 1 else camera_keys[0]
        self.camera_keys = camera_keys
        self.action_horizon = num_frames  # 16 raw frames for action
        self.n_history_frames = n_history_frames  # 8
        self.n_future_video_frames = n_future_video_frames  # 8
        self.action_video_freq_ratio = action_video_freq_ratio  # 2: stride for video
        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.sample_stride = sample_stride
        self.terminal_oversample_ratio = terminal_oversample_ratio
        self.predict_step_num = predict_step_num
        self.min_goal_len = min_goal_len

        # Action output dimension: (dx, dy, d_theta, moving_flag)
        self.action_dim = 4
        self.num_action_steps = predict_step_num

        # Video layout: history + current + future = 9 + 8 = 17
        self.n_cond_frames = n_history_frames + 1  # 9
        self.total_video_frames = self.n_cond_frames + n_future_video_frames  # 17
        assert self.total_video_frames % 4 == 1, (
            f"Total video frames must satisfy T%4==1 for VAE, got {self.total_video_frames}"
        )

        # Future video stride: 16 frames / 8 video frames = stride 2
        self.future_frame_stride = action_video_freq_ratio  # 2

        # Camera pitch for coordinate transform
        self._camera_deg = self._parse_camera_deg(self.overhead_camera)

        # Build index
        self.samples = []
        self._build_index(dataset_dirs)
        logger.info(
            f"NavVideoDataset: {len(self.samples)} samples, "
            f"action_horizon={self.action_horizon}, future_stride={self.future_frame_stride}, "
            f"predict_step_num={predict_step_num}, n_future_video={n_future_video_frames}"
        )

    @staticmethod
    def _parse_camera_deg(camera_key: str) -> float:
        """Extract pitch degrees from camera key like '125cm_30deg'."""
        parts = camera_key.replace("deg", "").split("_")
        for part in parts:
            if part.isdigit() and int(part) <= 90:
                deg = int(part)
                if deg > 0:
                    return float(deg)
        return 0.0

    def _build_index(self, dataset_dirs: List[str]):
        """
        Build sample index (fast, only reads jsonl metadata).
        """
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

                    if ep_length < self.n_history_frames + 2:
                        continue

                    # Regular samples with stride
                    num_rounds = ep_length // self.sample_stride
                    for n in range(num_rounds + 1):
                        start_frame_id = n * self.sample_stride
                        if start_frame_id >= ep_length - 1:
                            continue
                        self.samples.append({
                            "scene_path": scene_path,
                            "episode_idx": ep_idx,
                            "start_frame_id": start_frame_id,
                            "episode_length": ep_length,
                            "instruction": instruction,
                        })

                    # Terminal oversampling: near-end samples get extra copies
                    # Supports fractional ratios via probabilistic extra copy:
                    #   ratio=0.5 → each terminal frame gets 1 extra copy with 50% probability (1.5x total)
                    #   ratio=1.0 → always 1 extra copy (2x total)
                    #   ratio=2.0 → always 2 extra copies (3x total)
                    terminal_start = max(0, ep_length - 5)
                    # Use hashlib (not Python's hash()) for deterministic seed across all ranks.
                    # Python's hash() is randomized per-process (PYTHONHASHSEED), which causes
                    # different ranks to build datasets of different lengths → training crash.
                    _seed_str = f"{scene_path}_{ep_idx}".encode()
                    _seed = int(hashlib.md5(_seed_str).hexdigest(), 16) & 0xFFFFFFFF
                    rng = np.random.default_rng(seed=_seed)
                    for current_idx in range(terminal_start, ep_length):
                        n_extra_int = int(self.terminal_oversample_ratio)
                        frac_part = self.terminal_oversample_ratio - n_extra_int
                        n_extra = n_extra_int + (1 if rng.random() < frac_part else 0)
                        for _ in range(n_extra):
                            self.samples.append({
                                "scene_path": scene_path,
                                "episode_idx": ep_idx,
                                "start_frame_id": current_idx,
                                "episode_length": ep_length,
                                "instruction": instruction,
                            })

        logger.info(f"Index built: {len(self.samples)} samples from {len(dataset_dirs)} dataset dirs.")

    def __len__(self):
        return len(self.samples)

    def _load_and_resize_frame(self, scene_path: str, camera_key: str, episode_idx: int, frame_idx: int) -> torch.Tensor:
        """Load and resize a single frame to target video_size."""
        img_dir = os.path.join(
            scene_path, "videos", "chunk-000",
            f"observation.images.rgb.{camera_key}"
        )
        img_path = os.path.join(img_dir, f"episode_{episode_idx:06d}_{frame_idx}.jpg")
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms_F.to_tensor(img)
        img_tensor = transforms_F.resize(
            img_tensor, self.video_size,
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return img_tensor

    def _load_poses(self, scene_path: str, episode_idx: int, camera_key: str) -> np.ndarray:
        """Load all poses for an episode. Returns [N, 4, 4] array."""
        parquet_path = os.path.join(
            scene_path, "data", "chunk-000",
            f"episode_{episode_idx:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
        poses_raw = df[f"pose.{camera_key}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])
        return poses

    def _compute_spline_actions(
        self, poses: np.ndarray, start_idx: int, end_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute action labels using cubic spline interpolation.

        Takes poses[start_idx:end_idx], converts to relative coords,
        filters static points, applies cubic spline, resamples to predict_step_num.

        Returns:
            actions: [predict_step_num, 3] — (dx, dy, d_theta) per step.
            is_pad: [predict_step_num] — True where padded.
        """
        segment_poses = poses[start_idx:end_idx]
        segment_len = len(segment_poses)

        if segment_len < 2:
            actions = np.zeros((self.predict_step_num, 3), dtype=np.float32)
            is_pad = np.ones(self.predict_step_num, dtype=bool)
            return actions, is_pad

        discrete_traj = get_trajectory_relative_to_frame(segment_poses, camera_deg=self._camera_deg)

        _, resampled_actions = interpolate_and_resample_trajectory(discrete_traj, self.predict_step_num)
        resampled_actions = clip_or_pad(resampled_actions, self.predict_step_num)

        is_pad = np.zeros(self.predict_step_num, dtype=bool)

        traj_xy = discrete_traj[:, :2]
        steps = traj_xy[1:] - traj_xy[:-1]
        n_moving = (np.sum(steps**2, axis=1) > 0.05).sum()
        if n_moving < 1:
            is_pad[1:] = True

        return resampled_actions.astype(np.float32), is_pad

    def _get_history_indices(self, start_frame_id: int) -> List[int]:
        """Uniformly sample n_history_frames indices from [0, start_frame_id-1]."""
        if start_frame_id <= 0:
            return [0] * self.n_history_frames
        indices = np.linspace(0, start_frame_id - 1, self.n_history_frames, dtype=int).tolist()
        return indices

    def _get_future_indices(self, start_frame_id: int, end_frame_id: int, episode_length: int) -> List[int]:
        """
        Get n_future_video_frames indices covering the action horizon with stride 2.
        Aligned with action: both cover start+1 to start+16 (or shorter near end).
        """
        indices = []
        for i in range(1, self.n_future_video_frames + 1):
            fidx = start_frame_id + i * self.future_frame_stride
            fidx = min(fidx, episode_length - 1)
            indices.append(fidx)
        return indices

    def _get(self, idx: int) -> dict:
        """Get a single sample."""
        sample_info = self.samples[idx]
        scene_path = sample_info["scene_path"]
        episode_idx = sample_info["episode_idx"]
        start_frame_id = sample_info["start_frame_id"]
        episode_length = sample_info["episode_length"]
        instruction = sample_info["instruction"]

        # Action end frame: start + 16, clamped to episode end
        # Near terminal, this naturally becomes shorter
        end_frame_id = min(start_frame_id + self.action_horizon + 1, episode_length)

        # --- Frame indices ---
        history_indices = self._get_history_indices(start_frame_id)  # [8]
        future_indices = self._get_future_indices(start_frame_id, end_frame_id, episode_length)  # [8]
        all_0deg_indices = history_indices + [start_frame_id] + future_indices  # 17

        # --- Load video frames (17 frames) ---
        video_frames = []
        image_is_pad = []
        for fidx in all_0deg_indices:
            if fidx >= episode_length:
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

        # --- Actions: cubic spline on 16-frame segment ---
        poses = self._load_poses(scene_path, episode_idx, self.overhead_camera)
        actions, action_is_pad = self._compute_spline_actions(poses, start_frame_id, end_frame_id)

        # --- Stop label: mark moving_flag=0 for steps near goal (monotonic: once stopped, stays stopped) ---
        # Bug fix: independent per-step check caused physically impossible "early stop, late move" labels.
        # Fix: find the FIRST step within 1m of goal, then mark ALL steps from that index onward as stop.
        goal_pos = poses[-1, :3, 3]  # episode 终点的全局坐标
        actual_end = min(start_frame_id + self.action_horizon, episode_length - 1)
        first_stop_step = self.predict_step_num  # default: no near-goal stop
        for i in range(self.predict_step_num):
            frac = (i + 1) / self.predict_step_num
            interp_frame = min(int(start_frame_id + frac * (actual_end - start_frame_id)), episode_length - 1)
            pos_i = poses[interp_frame, :3, 3]
            dist_to_goal = np.linalg.norm(goal_pos - pos_i)
            if dist_to_goal < 1.0:  # reduced from 2.0m → 1.0m
                first_stop_step = i
                break  # monotonic: all subsequent steps will also be stop
        action_is_pad[first_stop_step:] = True  # mark first_stop_step and all later steps as stop

        # --- Text context ---
        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, context_mask = self._get_cached_text_context(prompt)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        # --- Assemble output ---
        moving_flag = (~action_is_pad).astype(np.float32).reshape(-1, 1)
        actions_with_flag = np.concatenate([actions, moving_flag], axis=1)  # [predict_step_num, 4]
        action_tensor = torch.from_numpy(actions_with_flag).float()
        action_is_pad_tensor = torch.from_numpy(action_is_pad).bool()
        image_is_pad_tensor = torch.tensor(image_is_pad, dtype=torch.bool)

        data = {
            "video": video,                         # [C, 17, H, W]
            "action": action_tensor,                # [predict_step_num, 4]
            "action_is_pad": action_is_pad_tensor,  # [predict_step_num]
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
        context = payload["context"]
        context_mask = payload["mask"].bool()
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
