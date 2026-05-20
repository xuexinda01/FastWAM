"""
End-to-end sanity check: feed REAL training-set actions into traj_utils,
both BEFORE and AFTER the agent-side adapter, and see what discrete actions
come out. The training samples we know to be "agent moves forward" should
yield FORWARD actions; if the adapter makes them yield LEFT/RIGHT only, the
adapter is wrong.

Pipeline:
  for sample in dataset:
    actions = _compute_spline_actions(...)   # (T, 3): (forward, left, dyaw_from_atan2)
    actions /= 4                              # un-do training normalization
    # Pick a sample with strong forward and small lateral world motion

    # Variant A: NO adapter (legacy traj_utils belief)
    actions_A = actions.copy()                # dim0=fwd, dim1=left
    # add a moving_flag so traj_utils accepts it
    flag_col = np.ones((actions_A.shape[0], 1))
    traj_A = np.concatenate([actions_A, flag_col], axis=1)
    a_list_A = fastwam_traj_to_actions(traj_A, step_size=0.25, turn_angle_deg=15, lookahead=4)

    # Variant B: WITH adapter (what fastwam_agent now does)
    actions_B = actions.copy()
    fwd  = actions_B[:, 0].copy()
    left = actions_B[:, 1].copy()
    actions_B[:, 0] = -left
    actions_B[:, 1] =  fwd
    traj_B = np.concatenate([actions_B, flag_col], axis=1)
    a_list_B = fastwam_traj_to_actions(traj_B, step_size=0.25, turn_angle_deg=15, lookahead=4)

We expect:
  - For pure-forward training samples (real robot walked straight forward):
      Variant A should output mostly FORWARD with maybe some L/R
      Variant B should output the SAME (because both are equivalent up to
        coordinate naming, IF our analysis is correct)
  - If Variant B starts producing only LEFT/RIGHT (no FORWARD), our adapter
    is geometrically wrong.
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, "/apdcephfs/wx_feature/home/xxd/fast-eval")

from visualize_nav_samples import build_dataset  # noqa: E402
from traj_utils import fastwam_traj_to_actions  # noqa: E402

ACTION_NAMES = {0: "STOP", 1: "FWD ", 2: "LEFT", 3: "RGT "}


def summarize(a_list):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for a in a_list:
        counts[a] += 1
    return f"FWD={counts[1]:>2} L={counts[2]:>2} R={counts[3]:>2} STOP={counts[0]:>2} (total={len(a_list)})"


def main():
    ds = build_dataset("/apdcephfs/wx_feature/home/xxd/debugdata/debug_data")
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(ds), size=300, replace=False)

    n_strict_fwd = 0
    a_only_left_or_right_count = 0
    b_only_left_or_right_count = 0
    a_has_fwd_count = 0
    b_has_fwd_count = 0

    print("Pure-forward training samples:")
    print(f"{'idx':>5} {'sfid':>4} {'a0t':>7} {'a1t':>7}  variant_A (no adapter)              variant_B (with adapter)")
    print("-" * 110)

    for s_idx in chosen:
        info = ds.samples[int(s_idx)]
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        end = min(sfid + ds.action_horizon + 1, ep_len)
        if end - sfid < 5:
            continue
        pq = os.path.join(info["scene_path"], "data", "chunk-000",
                          f"episode_{info['episode_idx']:06d}.parquet")
        df = pd.read_parquet(pq, columns=[f"pose.{ds.overhead_camera}"])
        poses_raw = df[f"pose.{ds.overhead_camera}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])

        actions, _ = ds._compute_spline_actions(poses, sfid, end)
        actions = actions.astype(np.float64).copy()
        actions[:, 0:2] /= 4.0  # un-do *=4

        a0t, a1t = float(actions[:, 0].sum()), float(actions[:, 1].sum())

        # Filter: STRONG forward (a0t > 0.4) and small |left| (|a1t| < 0.1)
        if not (a0t > 0.4 and abs(a1t) < 0.1):
            continue
        n_strict_fwd += 1

        flag_col = np.ones((actions.shape[0], 1))

        # Variant A: legacy assumption (no adapter)
        traj_A = np.concatenate([actions, flag_col], axis=1)
        a_list_A = fastwam_traj_to_actions(traj_A, step_size=0.25, turn_angle_deg=15, lookahead=4)

        # Variant B: with adapter (OLD WRONG)
        actions_B = actions.copy()
        fwd = actions_B[:, 0].copy()
        left = actions_B[:, 1].copy()
        actions_B[:, 0] = -left
        actions_B[:, 1] = fwd
        traj_B = np.concatenate([actions_B, flag_col], axis=1)
        a_list_B = fastwam_traj_to_actions(traj_B, step_size=0.25, turn_angle_deg=15, lookahead=4)

        # Variant C: flip-only (NEW CORRECT)
        actions_C = actions.copy()
        actions_C[:, 1] *= -1.0   # left → right
        traj_C = np.concatenate([actions_C, flag_col], axis=1)
        a_list_C = fastwam_traj_to_actions(traj_C, step_size=0.25, turn_angle_deg=15, lookahead=4)

        a_has_fwd = any(a == 1 for a in a_list_A)
        b_has_fwd = any(a == 1 for a in a_list_B)
        c_has_fwd = any(a == 1 for a in a_list_C)
        a_only_lr = (not a_has_fwd) and any(a in (2, 3) for a in a_list_A)
        b_only_lr = (not b_has_fwd) and any(a in (2, 3) for a in a_list_B)
        c_only_lr = (not c_has_fwd) and any(a in (2, 3) for a in a_list_C)
        a_has_fwd_count += a_has_fwd
        b_has_fwd_count += b_has_fwd
        c_has_fwd_count = locals().get('c_has_fwd_count', 0) + c_has_fwd
        a_only_left_or_right_count += a_only_lr
        b_only_left_or_right_count += b_only_lr

        if n_strict_fwd <= 15:
            print(f"{s_idx:>5} {sfid:>4} {a0t:+.3f} {a1t:+.3f}  "
                  f"A:{summarize(a_list_A)}  "
                  f"B:{summarize(a_list_B)}  "
                  f"C:{summarize(a_list_C)}")

    print()
    print("=" * 78)
    print(f"Total strict-forward samples checked: {n_strict_fwd}")
    print(f"  Variant A (no adapter):  has FORWARD = {a_has_fwd_count}/{n_strict_fwd}  "
          f"only-L/R (no FWD) = {a_only_left_or_right_count}")
    print(f"  Variant B (with adapter): has FORWARD = {b_has_fwd_count}/{n_strict_fwd}  "
          f"only-L/R (no FWD) = {b_only_left_or_right_count}")
    print("=" * 78)


if __name__ == "__main__":
    main()
