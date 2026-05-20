"""
Run the NEW _compute_spline_actions algorithm on debugdata to compute
per-dimension action statistics for normalization.

Algorithm:
  1. For each segment [start, end] in every sample:
       a. Get robot-frame (x, y, yaw) for each pose using
          get_trajectory_relative_to_frame(camera_deg=30).
       b. Build progress parameter s_i = ||Δxy_i|| + α·|Δyaw_i|
          (α=0.95 so that 1-frame FORWARD ≈ 1-frame TURN_15°).
       c. Cubic-spline xy and linear-interp yaw at evenly-spaced s_target.
       d. Compute per-step (Δx, Δy, Δyaw_wrapped).
  2. Aggregate across all samples; report per-dim stats:
       - 1% / 99% percentiles → min-max normalization range
       - mean / std            → for reference
       - histogram             → sanity check distribution shape

Output:
  Prints recommended ACTION_MIN / ACTION_MAX constants to put into
  nav_video_dataset.py.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from fastwam.datasets.lerobot.nav_video_dataset import (  # noqa: E402
    get_trajectory_relative_to_frame,
)
from visualize_nav_samples import build_dataset  # noqa: E402

# Hyperparameters
ALPHA = 0.95  # 1 rad of yaw ≈ 0.95 m of xy travel
              # → 1 frame FORWARD (0.25m) ≈ 1 frame TURN (0.262 rad × 0.95 = 0.249m)
PREDICT_STEP_NUM = 8


def compute_step_actions_new(poses, start_idx, end_idx, alpha=ALPHA, predict_step_num=PREDICT_STEP_NUM):
    """New version: keep every frame, progress = ||Δxy|| + α·|Δyaw|."""
    seg = poses[start_idx:end_idx]
    if len(seg) < 2:
        return None
    rel = get_trajectory_relative_to_frame(seg, camera_deg=30.0)
    xy = rel[:, :2]
    yaw = np.unwrap(rel[:, 2])

    delta_xy = np.diff(xy, axis=0)
    delta_yaw_raw = np.diff(yaw)
    ds = np.linalg.norm(delta_xy, axis=1) + alpha * np.abs(delta_yaw_raw)
    s = np.concatenate([[0.0], np.cumsum(ds)])

    if s[-1] < 1e-6:
        return None

    s_target = np.linspace(0, s[-1], predict_step_num + 1)

    # xy: cubic spline if we have enough points, else linear
    if len(seg) >= 4:
        cs_x = CubicSpline(s, xy[:, 0])
        cs_y = CubicSpline(s, xy[:, 1])
        x_resampled = cs_x(s_target)
        y_resampled = cs_y(s_target)
    else:
        x_resampled = np.interp(s_target, s, xy[:, 0])
        y_resampled = np.interp(s_target, s, xy[:, 1])
    yaw_resampled = np.interp(s_target, s, yaw)

    dx = np.diff(x_resampled)
    dy = np.diff(y_resampled)
    dyaw = np.diff(yaw_resampled)
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi

    return np.stack([dx, dy, dyaw], axis=1)


def main():
    ds = build_dataset("/apdcephfs/wx_feature/home/xxd/debugdata/debug_data")
    print(f"#samples = {len(ds)}")

    all_steps = []  # collect per-step (dx, dy, dyaw)
    n_processed = 0

    for s_idx in range(len(ds)):
        info = ds.samples[s_idx]
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        end = min(sfid + ds.action_horizon + 1, ep_len)

        # Load poses
        pq = os.path.join(info["scene_path"], "data", "chunk-000",
                          f"episode_{info['episode_idx']:06d}.parquet")
        df = pd.read_parquet(pq, columns=[f"pose.{ds.overhead_camera}"])
        poses_raw = df[f"pose.{ds.overhead_camera}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])

        actions = compute_step_actions_new(poses, sfid, end)
        if actions is not None:
            all_steps.append(actions)
            n_processed += 1
        if (s_idx + 1) % 200 == 0:
            print(f"  processed {s_idx+1}/{len(ds)}", flush=True)

    arr = np.concatenate(all_steps, axis=0)  # (N*T, 3)
    print(f"\nProcessed {n_processed}/{len(ds)} samples → {arr.shape[0]} step-level actions\n")

    DIM_NAMES = ["dx (forward, m)", "dy (left, m)", "dyaw (rad)"]
    print("=" * 80)
    print(f"{'dim':<22} {'min':>9} {'max':>9} {'p01':>9} {'p99':>9} "
          f"{'mean':>9} {'std':>9}")
    print("-" * 80)
    p01s, p99s = [], []
    for d in range(3):
        vals = arr[:, d]
        p01, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
        p01s.append(p01)
        p99s.append(p99)
        print(f"{DIM_NAMES[d]:<22} {vals.min():>+9.4f} {vals.max():>+9.4f} "
              f"{p01:>+9.4f} {p99:>+9.4f} {vals.mean():>+9.4f} {vals.std():>9.4f}")
    print("=" * 80)

    # Suggested normalization constants (symmetric around 0 for sign-preserving)
    # Use max(|p01|, |p99|) per dim so that the normalization is symmetric in [-1, 1]
    print("\nRecommended normalization (symmetric, scale = max(|p01|, |p99|)):")
    print("  ACTION_SCALE = np.array([")
    scale = []
    for d in range(3):
        s_d = max(abs(p01s[d]), abs(p99s[d]))
        scale.append(s_d)
        print(f"      {s_d:.4f},   # {DIM_NAMES[d]}  → divides into [-1, 1]")
    print("  ])")
    print()
    print("Normalized samples (after dividing by scale, clipped to [-1,1]):")
    arr_norm = np.clip(arr / np.array(scale), -1.0, 1.0)
    for d in range(3):
        print(f"  {DIM_NAMES[d]:<22}  norm-min={arr_norm[:,d].min():+.3f}  "
              f"norm-max={arr_norm[:,d].max():+.3f}  "
              f"norm-mean={arr_norm[:,d].mean():+.3f}  "
              f"norm-std={arr_norm[:,d].std():.3f}")

    # Save stats to JSON
    stats = {
        "alpha": ALPHA,
        "predict_step_num": PREDICT_STEP_NUM,
        "action_scale": [float(x) for x in scale],
        "action_p01": [float(x) for x in p01s],
        "action_p99": [float(x) for x in p99s],
        "action_min": [float(x) for x in arr.min(axis=0)],
        "action_max": [float(x) for x in arr.max(axis=0)],
        "action_mean": [float(x) for x in arr.mean(axis=0)],
        "action_std": [float(x) for x in arr.std(axis=0)],
        "n_step_actions": int(arr.shape[0]),
        "n_samples": int(n_processed),
    }
    out = os.path.join(PROJECT_ROOT, "scripts", "_action_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {out}")


if __name__ == "__main__":
    main()
