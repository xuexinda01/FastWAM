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
import pickle
import time
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
    """[LEGACY] Convert absolute (x, y) positions to relative (dx, dy, delta_yaw).
    
    NOTE (2026-05-20): This function is kept for backward compatibility but is
    NO LONGER USED by the current `_compute_spline_actions`. The new pipeline
    uses real robot yaw from pose matrices (see `_compute_spline_actions_v2`)
    instead of inferring yaw from xy displacement direction. This function
    has the well-known limitation that pure in-place rotations are completely
    invisible to the resulting `delta_yaw` (which is just `arctan2(Δy, Δx)`).
    """
    vectors = np.diff(xy_actions, axis=0)              # [N-1, 2]
    norms = np.linalg.norm(vectors, axis=1)             # [N-1]
    STOP_EPS = 1e-6                                     # below this = "no motion"

    # Compute yaw safely: reuse previous valid yaw at stop steps.
    yaw = np.zeros(len(vectors), dtype=vectors.dtype)
    last_yaw = 0.0  # default for the very first step if it's a stop
    for i in range(len(vectors)):
        if norms[i] < STOP_EPS:
            yaw[i] = last_yaw
        else:
            yaw[i] = np.arctan2(vectors[i, 1], vectors[i, 0])
            last_yaw = yaw[i]

    delta_yaw = np.diff(yaw)
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    delta_yaw = np.concatenate([[yaw[0]], delta_yaw])

    delta_xyt = np.concatenate([vectors, delta_yaw[:, None]], axis=1)
    return delta_xyt


# =============================================================================
# NEW (2026-05-20) action-label pipeline
#
# Replaces the old `interpolate_and_resample_trajectory` + `xy_to_delta_xyt`
# pipeline. Key differences:
#   1. Does NOT mask out static (in-place rotation) frames. They participate
#      in resampling via the progress parameter `s`.
#   2. Uses REAL robot yaw from pose matrices (the third column of
#      `get_trajectory_relative_to_frame`'s output), not yaw inferred from
#      xy displacement direction.
#   3. Resamples on a synthetic progress parameter
#          s_i = ||Δxy_i||  +  alpha * |Δyaw_i|
#      so 1 frame of FORWARD (0.25 m) and 1 frame of TURN_15° (0.262 rad)
#      contribute equally to progress when alpha=0.95.
#   4. Outputs are normalized to roughly [-1, 1] using empirical 99th
#      percentile scaling (computed via scripts/compute_action_stats.py).
#      Inference must reverse this with the same constants.
# =============================================================================

# Empirical 99% percentile scaling (computed on debugdata, alpha=0.95).
# scripts/compute_action_stats.py reproduces these.
ACTION_PROGRESS_ALPHA = 0.95
ACTION_SCALE = np.array([0.2504, 0.2165, 0.2625], dtype=np.float32)
# layout: [forward_per_step (m), left_per_step (m), dyaw_per_step (rad)]


def normalize_action(actions_unnorm: np.ndarray) -> np.ndarray:
    """Divide each dim by its scale and clip to [-1, 1].

    Args:
        actions_unnorm: shape (T, 3) or (..., 3), in physical units
                        (forward m, left m, dyaw rad).

    Returns:
        actions_norm in [-1, 1].
    """
    return np.clip(actions_unnorm / ACTION_SCALE, -1.0, 1.0)


def denormalize_action(actions_norm: np.ndarray) -> np.ndarray:
    """Inverse of `normalize_action` (no clipping, since clipping was at
    train time and would be incorrect at inference)."""
    return actions_norm * ACTION_SCALE


def compute_spline_actions_v2(
    poses: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    predict_step_num: int,
    camera_deg: float,
    alpha: float = ACTION_PROGRESS_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """New action-label pipeline. See module-level docstring above for context.

    Args:
        poses:            (N, 4, 4) camera extrinsics for the whole episode.
        start_idx, end_idx: segment is poses[start_idx:end_idx].
        predict_step_num: number of output action steps (T).
        camera_deg:       pitch correction (typically 30).
        alpha:            yaw-vs-translation weighting (m / rad).

    Returns:
        actions_normalized: (T, 3) in [-1, 1], dims = (forward, left, dyaw).
        is_pad: (T,) True where the step has effectively zero motion (used as
                moving_flag inverse).
    """
    seg_poses = poses[start_idx:end_idx]
    seg_len = len(seg_poses)
    if seg_len < 2:
        return (np.zeros((predict_step_num, 3), dtype=np.float32),
                np.ones(predict_step_num, dtype=bool))

    # 1. Extract robot-frame (x, y, yaw)
    rel = get_trajectory_relative_to_frame(seg_poses, camera_deg=camera_deg)
    xy = rel[:, :2]
    yaw = np.unwrap(rel[:, 2])  # unwrap to avoid ±π discontinuities

    # 2. Build progress parameter `s`
    delta_xy = np.diff(xy, axis=0)
    delta_yaw_raw = np.diff(yaw)
    ds = np.linalg.norm(delta_xy, axis=1) + alpha * np.abs(delta_yaw_raw)
    s = np.concatenate([[0.0], np.cumsum(ds)])  # (seg_len,)

    # If nothing happened in this whole segment, return zeros + all-pad.
    if s[-1] < 1e-6:
        return (np.zeros((predict_step_num, 3), dtype=np.float32),
                np.ones(predict_step_num, dtype=bool))

    # 3. Resample at evenly-spaced s
    s_target = np.linspace(0.0, s[-1], predict_step_num + 1)

    if seg_len >= 4:
        cs_x = CubicSpline(s, xy[:, 0])
        cs_y = CubicSpline(s, xy[:, 1])
        x_resampled = cs_x(s_target)
        y_resampled = cs_y(s_target)
    else:
        x_resampled = np.interp(s_target, s, xy[:, 0])
        y_resampled = np.interp(s_target, s, xy[:, 1])
    yaw_resampled = np.interp(s_target, s, yaw)  # linear is enough for angle

    # 4. Per-step deltas
    dx = np.diff(x_resampled)
    dy = np.diff(y_resampled)
    dyaw = np.diff(yaw_resampled)
    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi  # wrap each step

    actions_unnorm = np.stack([dx, dy, dyaw], axis=1).astype(np.float32)

    # 5. Moving flag: a step "moved" if either xy or yaw changed appreciably.
    EPS_XY = 0.01      # 1 cm
    EPS_YAW = np.deg2rad(0.5)
    moved = (np.linalg.norm(actions_unnorm[:, :2], axis=1) > EPS_XY) | \
            (np.abs(actions_unnorm[:, 2]) > EPS_YAW)
    is_pad = ~moved

    # 6. Normalize
    actions_norm = normalize_action(actions_unnorm).astype(np.float32)

    return actions_norm, is_pad


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
        class_oversample: Optional[dict] = None,
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
        # Class-balanced oversampling. Default duplicates BIG_TURN ×4,
        # SMALL_TURN ×2, TERMINAL ×3 so the rare-but-important categories get
        # adequate gradient signal vs. the dominant FWD majority.
        self.class_oversample = class_oversample or {
            "FWD": 1,
            "SMALL_TURN": 2,
            "BIG_TURN": 4,
            "TERMINAL": 3,
            "OTHER": 1,
        }

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
        # Categorize each sample (FWD / SMALL_TURN / BIG_TURN / TERMINAL) so
        # we can balance class proportions via physical oversampling.
        self.sample_categories = self._classify_samples()
        # Class-balanced oversampling: physically duplicate under-represented
        # categories (BIG_TURN, SMALL_TURN, TERMINAL) so that
        # ResumableEpochSampler naturally sees a balanced stream.
        self._apply_class_oversampling(class_oversample)
        logger.info(
            f"NavVideoDataset: {len(self.samples)} samples, "
            f"action_horizon={self.action_horizon}, future_stride={self.future_frame_stride}, "
            f"predict_step_num={predict_step_num}, n_future_video={n_future_video_frames}"
        )
        # Log class distribution for visibility
        from collections import Counter
        _cnt = Counter(self.sample_categories)
        logger.info(
            f"NavVideoDataset class distribution (post-oversampling): "
            f"FWD={_cnt.get('FWD', 0)} "
            f"SMALL_TURN={_cnt.get('SMALL_TURN', 0)} "
            f"BIG_TURN={_cnt.get('BIG_TURN', 0)} "
            f"TERMINAL={_cnt.get('TERMINAL', 0)} "
            f"OTHER={_cnt.get('OTHER', 0)}"
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

    def _classify_samples(self) -> List[str]:
        """Assign one of {FWD, SMALL_TURN, BIG_TURN, TERMINAL, OTHER} to each
        sample, based on the magnitude of motion within its action segment.

        TERMINAL takes precedence: any sample whose segment ends at or past the
        episode end gets labelled TERMINAL (so STOP/near-goal supervision is
        always counted as TERMINAL, regardless of motion).

        Used by `get_sampler_weights()` for class-balanced training. We cache
        per-episode pose arrays to avoid re-reading the same parquet for
        different samples on the same episode.

        Results are cached to /tmp for fast reload on subsequent launches.
        """
        # --- Disk cache: compute a hash key from dataset identity ---
        cache_key_str = json.dumps({
            "n_samples": len(self.samples),
            "action_horizon": self.action_horizon,
            "overhead_camera": self.overhead_camera,
            "camera_deg": self._camera_deg,
            "first_sample": str(self.samples[0]) if self.samples else "",
            "last_sample": str(self.samples[-1]) if self.samples else "",
        }, sort_keys=True)
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:12]
        _shared_cache_dir = "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/.cache"
        os.makedirs(_shared_cache_dir, exist_ok=True)
        cache_path = f"{_shared_cache_dir}/nav_classify_cache_{cache_hash}.pkl"

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if len(cached) == len(self.samples):
                    logger.info(f"_classify_samples: loaded from disk cache {cache_path}")
                    return cached
                else:
                    logger.warning(f"_classify_samples: cache size mismatch "
                                   f"({len(cached)} vs {len(self.samples)}), recomputing.")
            except Exception as e:
                logger.warning(f"_classify_samples: failed to load cache: {e}")
        else:
            logger.warning(f"_classify_samples: cache not found at {cache_path} "
                           f"(hash={cache_hash})")

        logger.info(f"_classify_samples: computing classifications for "
                    f"{len(self.samples)} samples (this may take a while on NFS)...")
        t0 = time.time()

        TURN_THRESHOLD_SMALL = np.deg2rad(15)   # > 15° within horizon
        TURN_THRESHOLD_BIG = np.deg2rad(45)    # > 45°  → BIG_TURN
        FWD_THRESHOLD = 0.30                    # > 30 cm xy displacement
        TERMINAL_FRAMES = 5                     # within last 5 frames of ep

        cats: List[str] = []
        pose_cache: dict = {}
        for s_idx, info in enumerate(self.samples):
            sfid = info["start_frame_id"]
            ep_len = info["episode_length"]
            end = min(sfid + self.action_horizon + 1, ep_len)

            # TERMINAL takes precedence: segment that hits or passes the very
            # end of the episode is a "must learn STOP" sample.
            if end >= ep_len - 0:  # i.e. segment includes the last frame
                cats.append("TERMINAL")
                continue
            # Also cover: original segment was clamped because horizon ran out.
            if (end - sfid) < self.action_horizon + 1:
                cats.append("TERMINAL")
                continue

            cache_key = (info["scene_path"], info["episode_idx"])
            if cache_key not in pose_cache:
                try:
                    pose_cache[cache_key] = self._load_poses(
                        info["scene_path"], info["episode_idx"], self.overhead_camera
                    )
                except Exception as e:
                    logger.warning(f"_classify_samples: failed to load poses "
                                   f"for {cache_key}: {e}")
                    cats.append("OTHER")
                    continue
            poses = pose_cache[cache_key]
            if end > len(poses):
                cats.append("OTHER")
                continue

            try:
                rel = get_trajectory_relative_to_frame(
                    poses[sfid:end], camera_deg=self._camera_deg
                )
            except Exception:
                cats.append("OTHER")
                continue
            xy = rel[:, :2]
            yaw = np.unwrap(rel[:, 2])
            d_xy_total = float(np.linalg.norm(xy[-1] - xy[0]))
            d_yaw_total = float(abs(yaw[-1] - yaw[0]))

            if d_yaw_total >= TURN_THRESHOLD_BIG:
                cats.append("BIG_TURN")
            elif d_yaw_total >= TURN_THRESHOLD_SMALL:
                cats.append("SMALL_TURN")
            elif d_xy_total >= FWD_THRESHOLD:
                cats.append("FWD")
            else:
                cats.append("OTHER")

        # --- Save to disk cache for fast reload next time ---
        elapsed = time.time() - t0
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cats, f)
            logger.info(f"_classify_samples: done in {elapsed:.1f}s, "
                        f"saved cache to {cache_path}")
        except Exception as e:
            logger.warning(f"_classify_samples: done in {elapsed:.1f}s, "
                           f"but failed to save cache: {e}")
        return cats

    def _apply_class_oversampling(self, class_oversample_override: Optional[dict] = None):
        """Physically duplicate under-represented samples in `self.samples`.

        Reads `self.class_oversample` (or override) which maps category →
        integer multiplier. A sample of category C with multiplier K appears
        K times in the final list (K=1 means no change).

        Updates both `self.samples` and `self.sample_categories` in place.
        """
        cfg = class_oversample_override if class_oversample_override is not None \
            else self.class_oversample
        new_samples = []
        new_cats = []
        for s, c in zip(self.samples, self.sample_categories):
            mult = int(cfg.get(c, 1))
            mult = max(1, mult)
            for _ in range(mult):
                new_samples.append(s)
                new_cats.append(c)
        self.samples = new_samples
        self.sample_categories = new_cats

    def _compute_spline_actions(
        self, poses: np.ndarray, start_idx: int, end_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute action labels using the v2 (yaw-aware) pipeline.

        Output is normalized to [-1, 1] using ACTION_SCALE constants.
        Output dims: (Δforward, Δleft, Δyaw) per step.
        See `compute_spline_actions_v2` for details.
        """
        return compute_spline_actions_v2(
            poses, start_idx, end_idx,
            predict_step_num=self.predict_step_num,
            camera_deg=self._camera_deg,
            alpha=ACTION_PROGRESS_ALPHA,
        )

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

        # Action end frame: start + 8, clamped to episode end
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
