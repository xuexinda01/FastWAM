"""
Quantify how much real-robot heading change is lost in the action labels
produced by `_compute_spline_actions`.

For each sample we compare:

    delta_yaw_TRUE :  the *actual* robot yaw change between the segment's
                      first and last frame (extracted from the rotation
                      matrices of poses themselves).

    delta_yaw_ACT  :  the SUM of action[:, 2] returned by
                      `_compute_spline_actions` (i.e. how much the spline
                      "pretends" the robot turned).

If the pipeline preserves heading correctly, |TRUE| ≈ |ACT|.
If turns are lost (mask + spline kills them), |ACT| ≪ |TRUE|.
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from fastwam.datasets.lerobot.nav_video_dataset import (  # noqa: E402
    get_trajectory_relative_to_frame,
)
from visualize_nav_samples import build_dataset  # noqa: E402


def main():
    ds = build_dataset("/apdcephfs/wx_feature/home/xxd/debugdata/debug_data")
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(ds), size=min(400, len(ds)), replace=False)

    rows = []
    for s_idx in chosen:
        info = ds.samples[int(s_idx)]
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        end = min(sfid + ds.action_horizon + 1, ep_len)
        if end - sfid < 3:
            continue

        # Load poses
        pq = os.path.join(info["scene_path"], "data", "chunk-000",
                          f"episode_{info['episode_idx']:06d}.parquet")
        df = pd.read_parquet(pq, columns=[f"pose.{ds.overhead_camera}"])
        poses_raw = df[f"pose.{ds.overhead_camera}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])

        # ---- TRUE robot yaw change across the segment ----
        rel = get_trajectory_relative_to_frame(poses[sfid:end], camera_deg=30.0)
        # rel[:, 2] = yaw of each frame in the seg-start frame
        # TRUE total heading change = unwrap(yaw_last) - yaw_first  (yaw_first ≡ 0)
        yaw_unw = np.unwrap(rel[:, 2])
        true_dyaw = yaw_unw[-1] - yaw_unw[0]
        # also the maximum cumulative deviation (better captures "turn then turn back")
        true_dyaw_abs_total = np.sum(np.abs(np.diff(yaw_unw)))

        # ---- ACTION-LABEL-implied yaw change (sum of dim2) ----
        actions, _ = ds._compute_spline_actions(poses, sfid, end)
        # actions[:, 2] is per-step delta_yaw (no *=4 applied to dim2)
        act_dyaw = float(np.sum(actions[:, 2]))
        act_dyaw_abs_total = float(np.sum(np.abs(actions[:, 2])))

        # ---- xy displacement (for context) ----
        xy0 = rel[0, :2]
        xy1 = rel[-1, :2]
        xy_disp = float(np.linalg.norm(xy1 - xy0))
        path_len = float(np.sum(np.linalg.norm(np.diff(rel[:, :2], axis=0), axis=1)))

        rows.append(dict(
            sample_idx=int(s_idx),
            sfid=sfid,
            seg_len=end - sfid,
            true_dyaw_deg=np.degrees(true_dyaw),
            true_dyaw_abs_total_deg=np.degrees(true_dyaw_abs_total),
            act_dyaw_deg=np.degrees(act_dyaw),
            act_dyaw_abs_total_deg=np.degrees(act_dyaw_abs_total),
            xy_disp_m=xy_disp,
            path_len_m=path_len,
        ))

    df = pd.DataFrame(rows)
    print(f"\nChecked {len(df)} samples.\n")

    # buckets by true_dyaw magnitude
    abs_true = df["true_dyaw_abs_total_deg"].abs()
    abs_act = df["act_dyaw_abs_total_deg"].abs()

    print("=" * 78)
    print(f"{'true |Δyaw|':>14} {'#samples':>10} {'mean act |Δyaw|':>18} "
          f"{'mean ratio act/true':>22}")
    print("=" * 78)
    for lo, hi in [(0, 5), (5, 30), (30, 90), (90, 180), (180, 9999)]:
        m = (abs_true >= lo) & (abs_true < hi)
        if m.sum() == 0:
            continue
        sub = df[m]
        ratio = (abs_act[m] / abs_true[m]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  [{lo:>3},{hi:>4})°  "
              f"{m.sum():>10}  "
              f"{sub['act_dyaw_abs_total_deg'].mean():>18.2f}°  "
              f"{ratio.mean():>22.3f}")
    print("=" * 78)
    print()
    print("Interpretation:")
    print("  ratio ≈ 1.0  → action labels faithfully preserve heading change")
    print("  ratio < 0.3  → most heading change LOST in mask+spline pipeline")
    print()

    # show a few worst offenders
    df["loss_deg"] = abs_true - abs_act
    worst = df.sort_values("loss_deg", ascending=False).head(10)
    print("Top-10 samples with largest 'lost heading change':")
    print(worst[["sample_idx", "sfid", "seg_len",
                 "true_dyaw_abs_total_deg", "act_dyaw_abs_total_deg",
                 "xy_disp_m", "path_len_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
