"""
Sanity-check `_compute_spline_actions` on real debug data.

What this script does:
  1. Load one episode's poses from a parquet file.
  2. Run `get_trajectory_relative_to_frame` → `interpolate_and_resample_trajectory`.
  3. Compare:
       (a) raw episode trajectory (world)
       (b) relative trajectory (after frame-0 reset)
       (c) spline-resampled waypoints
       (d) reconstructed positions from action deltas (sanity check the inverse)
  4. Save 2 figures:
       - traj_world_vs_relative.png  ← shows ego-centric reset works
       - spline_vs_raw_relative.png  ← shows spline + resample matches raw
  5. Also dumps a few numeric diagnostics:
       - first/last positions
       - max distance of resampled traj from raw relative traj (interpolation error)
       - check that integrating action deltas reproduces resampled xy

Usage:
  python scripts/verify_spline_actions.py
"""
import os
import sys
from pathlib import Path

# Make imports work without installing the package
HERE = Path(__file__).resolve()
PROJ_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))

import numpy as np
import pandas as pd

from fastwam.datasets.lerobot.nav_video_dataset import (
    get_trajectory_relative_to_frame,
    interpolate_and_resample_trajectory,
    xy_to_delta_xyt,
    smooth_and_resample_trajectory,
)


# -----------------------------------------------------------------------------
# Config (matches configs/data/nav_vln.yaml)
# -----------------------------------------------------------------------------
SCENE_ROOT = "/apdcephfs/wx_feature/home/xxd/debugdata/debug_data/r2r/17DRP5sb8fy"
EPISODE_IDX = 0
OVERHEAD_CAMERA = "125cm_30deg"   # nav_vln uses this for poses
PRIMARY_CAMERA = "125cm_0deg"
ACTION_HORIZON = 8                 # num_frames in yaml
PREDICT_STEP_NUM = 8

# Try several start frames spread across the episode
START_FRACS = [0.0, 0.25, 0.5, 0.75]

OUT_DIR = Path("/apdcephfs/wx_feature/home/xxd/FastWAM/scripts/_verify_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_poses(scene_root: str, episode_idx: int, camera_key: str) -> np.ndarray:
    parquet_path = os.path.join(
        scene_root, "data", "chunk-000", f"episode_{episode_idx:06d}.parquet"
    )
    df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
    poses_raw = df[f"pose.{camera_key}"].tolist()
    poses = np.array([np.vstack(p) for p in poses_raw])
    return poses


def parse_camera_deg(camera_key: str) -> float:
    parts = camera_key.replace("deg", "").split("_")
    for part in parts:
        try:
            return float(part)
        except ValueError:
            continue
    return 0.0


def integrate_actions(actions_xy: np.ndarray) -> np.ndarray:
    """Reproduce xy positions by accumulating dx/dy. actions_xy shape: [N, 2]."""
    return np.concatenate([[[0.0, 0.0]], np.cumsum(actions_xy, axis=0)], axis=0)


def dist_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    """Min Euclidean distance from `point` (2,) to polyline (M, 2)."""
    seg_starts = polyline[:-1]
    seg_ends = polyline[1:]
    seg = seg_ends - seg_starts
    seg_len_sq = (seg ** 2).sum(axis=1) + 1e-12
    t = ((point - seg_starts) * seg).sum(axis=1) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj = seg_starts + t[:, None] * seg
    d = np.sqrt(((proj - point) ** 2).sum(axis=1))
    return float(d.min())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print(f"[setup] scene = {SCENE_ROOT}")
    print(f"[setup] camera_overhead = {OVERHEAD_CAMERA}")

    poses = load_poses(SCENE_ROOT, EPISODE_IDX, OVERHEAD_CAMERA)
    episode_length = len(poses)
    camera_deg = parse_camera_deg(OVERHEAD_CAMERA)

    print(f"\n[episode] length      = {episode_length}")
    print(f"[episode] camera_deg  = {camera_deg}")
    print(f"[episode] pose[0] xyz = {poses[0, :3, 3]}")
    print(f"[episode] pose[-1] xyz = {poses[-1, :3, 3]}")
    print(f"[episode] full trajectory xyz range:")
    print(f"   x: [{poses[:, 0, 3].min():.3f}, {poses[:, 0, 3].max():.3f}]")
    print(f"   y: [{poses[:, 1, 3].min():.3f}, {poses[:, 1, 3].max():.3f}]")
    print(f"   z: [{poses[:, 2, 3].min():.3f}, {poses[:, 2, 3].max():.3f}]")

    # =========================================================================
    # Test 1: full-episode relative trajectory
    # Verify: relative_to_ref[0] should be (0, 0, 0)
    # Verify: relative trajectory has the SAME shape as world trajectory,
    #         just rigidly transformed.
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: full-episode relative-to-frame[0] coordinate transform")
    print("=" * 70)

    full_relative = get_trajectory_relative_to_frame(poses, camera_deg=camera_deg)
    print(f"  full_relative.shape   = {full_relative.shape}")
    print(f"  full_relative[0]      = {full_relative[0]}   (should be ~[0,0,0])")
    print(f"  full_relative[-1]     = {full_relative[-1]}")

    # Sanity: shape preservation. Check pairwise distances are preserved
    # (rigid transform should not change inter-point distances).
    raw_xyz = poses[:, :3, 3]                     # world translations [N, 3]
    raw_xy = raw_xyz[:, :2]
    rel_xy = full_relative[:, :2]
    n_check = min(50, episode_length)
    idx_check = np.linspace(0, episode_length - 1, n_check).astype(int)
    raw_ds = np.linalg.norm(raw_xy[idx_check][1:] - raw_xy[idx_check][:-1], axis=1)
    rel_ds = np.linalg.norm(rel_xy[idx_check][1:] - rel_xy[idx_check][:-1], axis=1)
    max_diff = float(np.abs(raw_ds - rel_ds).max())
    print(f"  pairwise-distance preservation max diff = {max_diff:.6f} (should be ~0)")
    if max_diff > 1e-3:
        print("  WARNING: distance distortion detected — coord transform is NOT rigid!")
    else:
        print("  OK: rigid transform preserved pairwise distances.")

    # =========================================================================
    # Test 2: at several start frames, run the full _compute_spline_actions
    # equivalent and check:
    #   - resampled trajectory starts at (0, 0)
    #   - resampled trajectory roughly tracks the raw relative trajectory
    #   - cumsum(actions[:, :2] / 4) matches resampled xy
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: per-window spline + resample, compared against raw relative")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(START_FRACS), figsize=(5 * len(START_FRACS), 5))
    if len(START_FRACS) == 1:
        axes = [axes]

    for ax, frac in zip(axes, START_FRACS):
        start_idx = int(frac * (episode_length - ACTION_HORIZON - 1))
        start_idx = max(0, min(start_idx, episode_length - 2))
        end_idx = min(start_idx + ACTION_HORIZON + 1, episode_length)

        segment_poses = poses[start_idx:end_idx]
        segment_len = len(segment_poses)
        print(f"\n[window] start={start_idx}, end={end_idx}, segment_len={segment_len}")

        # Relative trajectory (raw, ego-centric, at this window's frame 0)
        discrete_traj = get_trajectory_relative_to_frame(
            segment_poses, camera_deg=camera_deg
        )
        raw_rel_xy = discrete_traj[:, :2]

        # Run spline + resample
        resampled_traj, resampled_actions = interpolate_and_resample_trajectory(
            discrete_traj, predict_step_num=PREDICT_STEP_NUM
        )
        # resampled_traj: [predict_step_num + 1, 2]
        # resampled_actions: [predict_step_num, 3], xy ×4

        print(f"  raw_rel_xy first/last  = {raw_rel_xy[0]}, {raw_rel_xy[-1]}")
        print(f"  resampled_traj first/last = {resampled_traj[0]}, {resampled_traj[-1]}")
        print(f"  resampled_actions[0:3] =\n{resampled_actions[:3]}")
        print(f"  resampled_actions[-3:] =\n{resampled_actions[-3:]}")

        # Sanity check 1: resampled[0] should be ~(0, 0)
        d0 = float(np.linalg.norm(resampled_traj[0]))
        print(f"  resampled_traj[0] dist from origin = {d0:.6f} (should be ~0)")

        # Sanity check 2: integrating actions should reproduce resampled xy
        actions_xy_meters = resampled_actions[:, :2] / 4.0  # undo ×4 normalization
        integrated_xy = integrate_actions(actions_xy_meters)
        recon_err = float(np.abs(integrated_xy - resampled_traj).max())
        print(f"  cumsum(actions/4) vs resampled xy max diff = {recon_err:.6f} (should be ~0)")

        # Sanity check 3: how far is each resampled point from the raw traj polyline?
        max_d = max(
            dist_point_to_polyline(p, raw_rel_xy) for p in resampled_traj
        )
        print(f"  max distance of resampled point to raw polyline = {max_d:.4f} m")

        # ---- plot ----
        ax.plot(raw_rel_xy[:, 0], raw_rel_xy[:, 1], "-o", color="tab:blue",
                label=f"raw relative (n={segment_len})", markersize=4, alpha=0.7)
        ax.plot(resampled_traj[:, 0], resampled_traj[:, 1], "-x",
                color="tab:orange",
                label=f"spline resampled ({PREDICT_STEP_NUM + 1})", markersize=8)
        ax.plot(integrated_xy[:, 0], integrated_xy[:, 1], ":", color="tab:green",
                label="cumsum(actions/4)", linewidth=2)
        ax.scatter([0], [0], c="red", s=80, zorder=5, label="start (0,0)")
        ax.set_title(f"start_idx={start_idx}, len={segment_len}\n"
                     f"recon_err={recon_err:.2e}  max_dev={max_d:.3f}m")
        ax.set_xlabel("x (m, ego frame)")
        ax.set_ylabel("y (m, ego frame)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"_compute_spline_actions verification — episode {EPISODE_IDX}, "
        f"overhead camera = {OVERHEAD_CAMERA} (pitch={camera_deg}°)",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "spline_vs_raw_relative.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\n[saved] {fig_path}")

    # =========================================================================
    # Test 3: world vs frame-0-relative, full episode
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: visualize full-episode world vs ego-centric trajectory")
    print("=" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # World coords
    ax1.plot(raw_xy[:, 0], raw_xy[:, 1], "-o", color="tab:blue", markersize=2)
    ax1.scatter([raw_xy[0, 0]], [raw_xy[0, 1]], c="green", s=100, label="start", zorder=5)
    ax1.scatter([raw_xy[-1, 0]], [raw_xy[-1, 1]], c="red", s=100, label="end", zorder=5)
    ax1.set_title(f"world coords (raw)\n{episode_length} frames")
    ax1.set_xlabel("x_world (m)")
    ax1.set_ylabel("y_world (m)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Ego-centric coords
    ax2.plot(rel_xy[:, 0], rel_xy[:, 1], "-o", color="tab:orange", markersize=2)
    ax2.scatter([0], [0], c="green", s=100, label="start (0,0)", zorder=5)
    ax2.scatter([rel_xy[-1, 0]], [rel_xy[-1, 1]], c="red", s=100, label="end", zorder=5)
    ax2.set_title(
        f"ego-centric (relative to frame 0)\n"
        f"end pos = ({rel_xy[-1, 0]:.2f}, {rel_xy[-1, 1]:.2f}) m"
    )
    ax2.set_xlabel("x_ego (m)")
    ax2.set_ylabel("y_ego (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(
        f"World vs ego-centric trajectory — episode {EPISODE_IDX}",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path2 = OUT_DIR / "traj_world_vs_relative.png"
    fig.savefig(fig_path2, dpi=120)
    plt.close(fig)
    print(f"[saved] {fig_path2}")

    print("\n" + "=" * 70)
    print("DONE. Open the two PNGs to inspect:")
    print(f"  1) {OUT_DIR / 'traj_world_vs_relative.png'}")
    print(f"  2) {OUT_DIR / 'spline_vs_raw_relative.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
