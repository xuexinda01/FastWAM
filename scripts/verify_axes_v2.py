"""
INDEPENDENT verification of action dim0/dim1 semantics.

Strategy: rather than trusting `get_trajectory_relative_to_frame` (which itself
applies T_camera2robot), use the *raw* extrinsic matrices to find a
ground-truth notion of "forward".

For each pose, the camera's optical axis in world-frame is:
    forward_world = -R[:, 2]    (camera convention: -Z = looking direction)
Or for OpenCV/Habitat convention this could differ — but what we *can* do
without any convention assumption is:

    Take two consecutive poses where the agent clearly moved.
    Look at the world-frame translation: delta_t_world = pose[i+1, :3, 3] - pose[i, :3, 3]
    Compare its magnitude with the action label's dim0 vs dim1 magnitudes.
    If "forward" motion in world (where the agent's optical axis points)
    correlates with dim0 → dim0 is forward; otherwise dim1 is forward.

We use a different definition of "agent moved forward":
    project delta_t_world onto the agent's optical axis direction.
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from visualize_nav_samples import build_dataset  # noqa: E402


def main():
    ds = build_dataset("/apdcephfs/wx_feature/home/xxd/debugdata/debug_data")
    print(f"#samples = {len(ds)}")

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(ds), size=min(300, len(ds)), replace=False)

    # Several conventions for "agent forward in camera frame"
    #   col 0 = +X column of pose
    #   col 1 = +Y column
    #   col 2 = +Z column
    # Whichever column points along the direction of robot translation IS the
    # camera's "forward" axis.
    n = 0
    fwd_consistency = {0: 0, 1: 0, 2: 0, -0: 0, -1: 0, -2: 0}
    abs_a0_when_fwd = []
    abs_a1_when_fwd = []
    sign_records = []  # for left-vs-right disambiguation

    for s_idx in chosen:
        info = ds.samples[int(s_idx)]
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        if sfid + 2 >= ep_len:
            continue
        pq = os.path.join(info["scene_path"], "data", "chunk-000",
                          f"episode_{info['episode_idx']:06d}.parquet")
        df = pd.read_parquet(pq, columns=[f"pose.{ds.overhead_camera}"])
        poses_raw = df[f"pose.{ds.overhead_camera}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])

        # World-frame translation between sfid and sfid+1
        t0 = poses[sfid, :3, 3]
        t1 = poses[sfid + 1, :3, 3]
        dt = t1 - t0
        if np.linalg.norm(dt) < 0.05:
            continue

        # Project dt onto each axis of pose[sfid]
        R = poses[sfid, :3, :3]   # axes are columns
        proj = {
            "+X": dt @ R[:, 0],
            "+Y": dt @ R[:, 1],
            "+Z": dt @ R[:, 2],
        }
        # which axis is the "forward" axis (largest |projection|)
        fwd_axis = max(proj, key=lambda k: abs(proj[k]))
        # whether the agent moved along positive or negative direction of that axis
        fwd_sign = np.sign(proj[fwd_axis])
        n += 1
        if n <= 5:
            print(f"  sample {s_idx}: |dt|={np.linalg.norm(dt):.3f}m  "
                  f"projections: +X={proj['+X']:+.3f} +Y={proj['+Y']:+.3f} "
                  f"+Z={proj['+Z']:+.3f}  → forward axis: {fwd_axis}({fwd_sign:+.0f})")

        # Now action label — use _compute_spline_actions
        end = min(sfid + ds.action_horizon + 1, ep_len)
        actions, _ = ds._compute_spline_actions(poses, sfid, end)
        a0 = actions[0, 0] / 4.0
        a1 = actions[0, 1] / 4.0

        # If the world-projection says "forward direction is mostly along axis X
        # of pose[sfid], with magnitude m", we should see |a0| or |a1| ≈ m
        # in the action label. Whichever matches is the "forward dim".
        m = abs(proj[fwd_axis])
        # which of (a0, a1) is closer to m?
        if abs(abs(a0) - m) < abs(abs(a1) - m):
            abs_a0_when_fwd.append(m)
            sign_records.append(("dim0_is_fwd", np.sign(a0), fwd_sign))
        else:
            abs_a1_when_fwd.append(m)
            sign_records.append(("dim1_is_fwd", np.sign(a1), fwd_sign))

    print()
    print("=" * 78)
    print(f"Total samples with motion: {n}")
    print(f"#samples where dim0 magnitude ≈ |forward translation|: {len(abs_a0_when_fwd)}")
    print(f"#samples where dim1 magnitude ≈ |forward translation|: {len(abs_a1_when_fwd)}")
    print()
    if abs_a0_when_fwd:
        print(f"  (dim0=forward) avg |fwd| = {np.mean(abs_a0_when_fwd):.3f} m")
    if abs_a1_when_fwd:
        print(f"  (dim1=forward) avg |fwd| = {np.mean(abs_a1_when_fwd):.3f} m")

    # Sign analysis
    dim0_signs = [(a, b) for tag, a, b in sign_records if tag == "dim0_is_fwd"]
    dim1_signs = [(a, b) for tag, a, b in sign_records if tag == "dim1_is_fwd"]
    print()
    if dim0_signs:
        same = sum(1 for a, b in dim0_signs if a * b > 0)
        diff = sum(1 for a, b in dim0_signs if a * b < 0)
        print(f"  dim0=forward case: sign(a0) matches sign(world-fwd-proj): "
              f"same={same}  opposite={diff}")
    if dim1_signs:
        same = sum(1 for a, b in dim1_signs if a * b > 0)
        diff = sum(1 for a, b in dim1_signs if a * b < 0)
        print(f"  dim1=forward case: sign(a1) matches sign(world-fwd-proj): "
              f"same={same}  opposite={diff}")
    print("=" * 78)


if __name__ == "__main__":
    main()
