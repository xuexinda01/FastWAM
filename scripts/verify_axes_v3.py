"""
Independently determine the SIGN convention of dim1 in action labels.

Method:
  - For each pose, extract the camera's 3 axis vectors in world frame
    (the 3 columns of pose[:3, :3]).
  - The "forward" axis is +Z (verified in v2: 151/151 samples agreed).
  - The "right" axis is +X (camera convention: +X=right, +Y=down, +Z=forward).
  - The "left" axis is therefore -X.

  - Find sample segments (over the whole spline horizon) where the agent moved
    SIDEWAYS in world (i.e. world translation projects significantly onto
    pose[sfid]'s +X column → moved RIGHT, or onto -X → moved LEFT).
  - Then look at the sign of dim1 in the action label.

  - If dim1 > 0 when world projection on +X > 0 → dim1 = RIGHT (lateral_right)
  - If dim1 > 0 when world projection on -X > 0 → dim1 = LEFT
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
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(ds), size=min(500, len(ds)), replace=False)

    # Take the WHOLE segment displacement so any net lateral motion shows up.
    records = []
    for s_idx in chosen:
        info = ds.samples[int(s_idx)]
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        end = min(sfid + ds.action_horizon + 1, ep_len)
        if end - sfid < 4:
            continue
        pq = os.path.join(info["scene_path"], "data", "chunk-000",
                          f"episode_{info['episode_idx']:06d}.parquet")
        df = pd.read_parquet(pq, columns=[f"pose.{ds.overhead_camera}"])
        poses_raw = df[f"pose.{ds.overhead_camera}"].tolist()
        poses = np.array([np.vstack(p) for p in poses_raw])

        # World-frame displacement from sfid to end of action horizon
        seg_disp_world = poses[end - 1, :3, 3] - poses[sfid, :3, 3]

        # Project onto pose[sfid] axes
        R = poses[sfid, :3, :3]
        proj_X = float(seg_disp_world @ R[:, 0])  # camera +X = right (assume)
        proj_Y = float(seg_disp_world @ R[:, 1])
        proj_Z = float(seg_disp_world @ R[:, 2])  # camera +Z = forward

        # Need substantial lateral component
        if abs(proj_X) < 0.2:
            continue

        # Action label — TOTAL of dim0 and dim1 over the whole horizon
        actions, _ = ds._compute_spline_actions(poses, sfid, end)
        a0_total = float(actions[:, 0].sum()) / 4.0  # un-do *=4
        a1_total = float(actions[:, 1].sum()) / 4.0

        records.append(dict(
            s_idx=int(s_idx), sfid=sfid, end=end,
            world_right_proj=proj_X, world_fwd_proj=proj_Z,
            a0_total=a0_total, a1_total=a1_total,
        ))

    df = pd.DataFrame(records)
    if df.empty:
        print("No samples with significant lateral motion found.")
        return

    print(f"#samples with |lateral world disp| > 0.2m: {len(df)}\n")
    print(df.head(15).to_string(index=False))

    # Analysis
    print("\n" + "=" * 78)
    # dim0: should reflect FORWARD according to v2 ⇒ a0_total ≈ proj_Z
    corr_a0_fwd = np.corrcoef(df["a0_total"], df["world_fwd_proj"])[0, 1]
    corr_a1_right = np.corrcoef(df["a1_total"], df["world_right_proj"])[0, 1]
    corr_a1_left = np.corrcoef(df["a1_total"], -df["world_right_proj"])[0, 1]
    print(f"corr(dim0, world_fwd_proj)   = {corr_a0_fwd:+.3f}")
    print(f"corr(dim1, world_RIGHT_proj) = {corr_a1_right:+.3f}  ← if ≈+1, dim1=right")
    print(f"corr(dim1, world_LEFT_proj)  = {corr_a1_left:+.3f}  ← if ≈+1, dim1=left")
    print()

    # Direct sign agreement (more robust than correlation)
    same_sign_right = ((df["a1_total"] * df["world_right_proj"]) > 0).sum()
    same_sign_left = ((df["a1_total"] * (-df["world_right_proj"])) > 0).sum()
    print(f"Sign(dim1) matches sign(+RIGHT) on {same_sign_right}/{len(df)} samples")
    print(f"Sign(dim1) matches sign(+LEFT)  on {same_sign_left}/{len(df)} samples")
    print("=" * 78)


if __name__ == "__main__":
    main()
