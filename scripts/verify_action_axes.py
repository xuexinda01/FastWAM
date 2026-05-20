"""
Empirically verify the (dim0, dim1) semantics of the action labels produced by
the current NavVideoDataset._compute_spline_actions.

Strategy:
  1. Build dataset.
  2. For each sample, take poses[start_idx], poses[start_idx+1] (1-frame ahead).
  3. Compute the relative pose in the robot frame (same transform as
     get_trajectory_relative_to_frame), and compare:
          - hypothesis A: dim0=forward, dim1=left      (current code)
          - hypothesis B: dim0=lateral(=right), dim1=forward  (legacy / docs)
     for which hypothesis the sign of the action label is consistent with the
     camera-extrinsic-derived "true" forward / lateral direction.
  4. Print summary statistics to settle the question.

Usage:
    /usr/bin/python3 /apdcephfs/wx_feature/home/xxd/FastWAM/scripts/verify_action_axes.py
"""
import os
import sys
import numpy as np

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from fastwam.datasets.lerobot.nav_video_dataset import (  # noqa: E402
    NavVideoDataset,
    get_trajectory_relative_to_frame,
)
from visualize_nav_samples import build_dataset  # noqa: E402


def main():
    ds = build_dataset("/apdcephfs/wx_feature/home/xxd/debugdata/debug_data")
    print(f"#samples = {len(ds)}")

    n_check = 0
    n_dim0_dominant_when_forward = 0
    n_dim1_dominant_when_forward = 0
    abs_dim0_sum = 0.0
    abs_dim1_sum = 0.0
    sign_dim0_when_left = []
    sign_dim1_when_left = []

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(ds), size=min(200, len(ds)), replace=False)

    for s_idx in chosen:
        info = ds.samples[int(s_idx)]
        scene = info["scene_path"]
        ep = info["episode_idx"]
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        if sfid + 2 >= ep_len:
            continue

        # --- (a) ground-truth motion in robot frame for the next 1 step ---
        import pandas as pd
        pq = os.path.join(scene, "data", "chunk-000", f"episode_{ep:06d}.parquet")
        df = pd.read_parquet(pq, columns=[f"pose.{ds.overhead_camera}"])
        poses_raw = df[f"pose.{ds.overhead_camera}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])

        # Compute relative xy in robot frame between sfid and sfid+1
        seg = poses[sfid:sfid + 2]
        if len(seg) < 2:
            continue
        rel = get_trajectory_relative_to_frame(seg, camera_deg=30.0)
        # rel[1, :2] = (x, y) of frame sfid+1 in the frame of sfid, robot-frame
        # Under T_camera2robot (ROS-style):
        #   robot x = +forward
        #   robot y = +left
        true_dx_robot = rel[1, 0]  # +forward
        true_dy_robot = rel[1, 1]  # +left

        # only use samples with substantial motion
        disp = np.hypot(true_dx_robot, true_dy_robot)
        if disp < 0.05:
            continue

        # --- (b) action label produced by the dataset for this same sample ---
        actions, _ = ds._compute_spline_actions(poses, sfid,
                                                min(sfid + ds.action_horizon + 1, ep_len))
        # actions: [T, 3] = (a0, a1, a2), a0/a1 already *=4
        a0 = actions[0, 0] / 4.0  # un-do *=4 for direct comparison
        a1 = actions[0, 1] / 4.0

        n_check += 1

        # If true motion is mostly forward (very small lateral), then whichever
        # of (a0, a1) has the larger magnitude is the "forward dim".
        if abs(true_dx_robot) > 3 * abs(true_dy_robot):
            # forward-dominant ground-truth
            if abs(a0) > abs(a1):
                n_dim0_dominant_when_forward += 1
            else:
                n_dim1_dominant_when_forward += 1
            abs_dim0_sum += abs(a0)
            abs_dim1_sum += abs(a1)

        # If true motion is mostly leftward (large +y_robot, small x_robot),
        # check which of (a0, a1) takes a positive value.
        if true_dy_robot > 3 * abs(true_dx_robot) and true_dy_robot > 0.1:
            sign_dim0_when_left.append(np.sign(a0))
            sign_dim1_when_left.append(np.sign(a1))

    print()
    print("=" * 70)
    print(f"Samples checked (with motion>0.05m): {n_check}")
    print()
    print("Forward-dominant ground-truth motion (|dx_robot| > 3·|dy_robot|):")
    print(f"  → dim0 dominates: {n_dim0_dominant_when_forward}")
    print(f"  → dim1 dominates: {n_dim1_dominant_when_forward}")
    print(f"  mean |a0| (when fwd-dominant) = {abs_dim0_sum / max(1, n_dim0_dominant_when_forward + n_dim1_dominant_when_forward):.4f}")
    print(f"  mean |a1| (when fwd-dominant) = {abs_dim1_sum / max(1, n_dim0_dominant_when_forward + n_dim1_dominant_when_forward):.4f}")
    print()
    print("Left-dominant ground-truth motion (dy_robot > 3·|dx_robot|, dy>0.1m):")
    print(f"  N = {len(sign_dim0_when_left)}")
    if sign_dim0_when_left:
        print(f"  sign(a0): +{sum(s>0 for s in sign_dim0_when_left)}  "
              f"-{sum(s<0 for s in sign_dim0_when_left)}  "
              f"0 {sum(s==0 for s in sign_dim0_when_left)}")
        print(f"  sign(a1): +{sum(s>0 for s in sign_dim1_when_left)}  "
              f"-{sum(s<0 for s in sign_dim1_when_left)}  "
              f"0 {sum(s==0 for s in sign_dim1_when_left)}")
    print()
    print("INTERPRETATION:")
    print("  H_A (current code):  dim0 = forward(+fwd), dim1 = left(+left)")
    print("       → fwd-dominant: a0 dominates;   left-dominant: a1 > 0")
    print("  H_B (traj_utils):    dim0 = lateral(+right), dim1 = forward(+fwd)")
    print("       → fwd-dominant: a1 dominates;   left-dominant: a0 < 0")
    print("=" * 70)


if __name__ == "__main__":
    main()
