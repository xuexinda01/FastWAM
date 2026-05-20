"""
Visualize training samples produced by NavVideoDataset (NEW v2 pipeline).

Layout (3 panels, left → right):
  LEFT  : Top-down trajectory.
          - Grey thin line       : full episode trajectory (raw xy)
          - Green star           : episode start
          - Red star             : episode goal
          - Blue thick line      : sub-segment [sfid : sfid+horizon+1] raw xy
          - Blue circle          : current frame (sample start)
          - Purple square        : sub-segment end (clamped to ep end)
          - Orange filled circle : v2 resampled anchor (moving_flag=1)
          - Orange hollow circle : v2 anchor (moving_flag=0, padded)
          - "i:1/0" label        : "step_idx : moving_flag" next to each anchor
          - Orange dashed line   : connects anchors in order
          The anchors come from `compute_spline_actions_v2` (yaw-aware
          arc length s = ||Δxy|| + α·|Δyaw|), NOT the legacy spline.

  CENTER: Action label table (the actual training target tensor).
          Three stacked bar plots over the 8 v2 steps:
            - dim0: Δforward (m, normalized in [-1, 1])
            - dim1: Δleft    (m, normalized in [-1, 1])
            - dim2: Δyaw     (rad, normalized in [-1, 1])
          Each bar is annotated with both the normalized value and the
          un-normalized physical value (m / deg) for easier mental check.
          Bars where moving_flag=0 (= action_is_pad=True) are greyed out.

  RIGHT : 17 condition+future frames as a 3×6 grid.
          - 8 history frames (uniformly sampled from [0, sfid-1])
          - 1 current frame   (sfid)
          - 8 future frames   (sfid + i*future_stride, min-clamped to ep end)
          Frames whose index was clamped (= same frame as ep last) get a red
          border to indicate "padded by replication".

This script bypasses Hydra/PyTorch DataLoader and reuses the dataset class
directly; it patches dataset_dirs to the real on-disk path
(/apdcephfs/wx_feature/home/xxd/debugdata/debug_data/{r2r,rxr,scalevln}).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline

# Make src/ importable
PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from fastwam.datasets.lerobot.nav_video_dataset import (  # noqa: E402
    NavVideoDataset,
    get_trajectory_relative_to_frame,
    compute_spline_actions_v2,
    denormalize_action,
    ACTION_PROGRESS_ALPHA,
    ACTION_SCALE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_dataset(real_root: str) -> NavVideoDataset:
    """Build NavVideoDataset using real on-disk paths and yaml-equivalent kwargs.

    Mirrors `configs/data/nav_vln.yaml` (num_frames=8, predict_step_num=8,
    sample_stride=4, terminal_oversample_ratio=0.5).
    """
    return NavVideoDataset(
        dataset_dirs=[
            os.path.join(real_root, "r2r"),
            os.path.join(real_root, "rxr"),
            os.path.join(real_root, "scalevln"),
        ],
        camera_keys=["125cm_0deg", "125cm_30deg"],
        num_frames=8,
        n_history_frames=8,
        n_future_video_frames=8,
        action_video_freq_ratio=1,
        video_size=[224, 224],
        concat_multi_camera="none",
        text_embedding_cache_dir=None,
        context_len=256,
        sample_stride=4,
        terminal_oversample_ratio=0.5,
        predict_step_num=8,
        min_goal_len=3,
    )


def episode_full_traj_xy(scene_path: str, episode_idx: int, camera_key: str) -> np.ndarray:
    """Load full episode trajectory in robot frame relative to frame 0.

    Returns (xy [N, 2], poses [N, 4, 4]).
    """
    import pandas as pd
    parquet_path = os.path.join(
        scene_path, "data", "chunk-000", f"episode_{episode_idx:06d}.parquet"
    )
    df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
    poses_raw = df[f"pose.{camera_key}"].tolist()
    poses = np.array([np.vstack(p) for p in poses_raw])
    rel = get_trajectory_relative_to_frame(poses, camera_deg=30.0)
    return rel[:, :2], poses


def v2_resampled_xy(seg_rel: np.ndarray, predict_step_num: int,
                    alpha: float = ACTION_PROGRESS_ALPHA) -> np.ndarray:
    """Replicate the v2 resampling **but return the absolute (x, y) anchors**
    instead of just the deltas.

    The dataset's `compute_spline_actions_v2` only returns dxyt because that's
    all that's needed for training. For visualization we want the actual
    anchor positions in the segment-local frame. This function mirrors the
    relevant lines of v2 verbatim.

    Args:
        seg_rel: (seg_len, 3) (x, y, yaw) in segment-local robot frame.
        predict_step_num: number of action steps T (so T+1 anchors).
        alpha: yaw-vs-translation weighting (m/rad).

    Returns:
        anchor_xy: (T+1, 2) in segment-local frame, anchor[0] is (0, 0).
                   Returns zeros if segment too short / static.
    """
    if len(seg_rel) < 2:
        return np.zeros((predict_step_num + 1, 2), dtype=np.float64)

    xy = seg_rel[:, :2]
    yaw = np.unwrap(seg_rel[:, 2])

    delta_xy = np.diff(xy, axis=0)
    delta_yaw_raw = np.diff(yaw)
    ds_step = np.linalg.norm(delta_xy, axis=1) + alpha * np.abs(delta_yaw_raw)
    s = np.concatenate([[0.0], np.cumsum(ds_step)])

    if s[-1] < 1e-6:
        return np.zeros((predict_step_num + 1, 2), dtype=np.float64)

    s_target = np.linspace(0.0, s[-1], predict_step_num + 1)
    seg_len = len(seg_rel)
    if seg_len >= 4:
        x_resampled = CubicSpline(s, xy[:, 0])(s_target)
        y_resampled = CubicSpline(s, xy[:, 1])(s_target)
    else:
        x_resampled = np.interp(s_target, s, xy[:, 0])
        y_resampled = np.interp(s_target, s, xy[:, 1])
    return np.stack([x_resampled, y_resampled], axis=1)


def pick_samples(ds: NavVideoDataset):
    """Pick a few representative sample indices from the dataset.

    Categories we want to cover:
      A. start_frame_id = 0  (degenerate: 9 condition frames are the same)
      B. mid-episode (far from end, all moving_flag=1 expected)
      C. near terminal (some moving_flag=0 expected)
      D. terminal-oversampled extra copy
    """
    picks = {}
    for idx, s in enumerate(ds.samples):
        sfid = s["start_frame_id"]
        ep_len = s["episode_length"]
        # A
        if "A" not in picks and sfid == 0:
            picks["A"] = idx
        # B: middle, far from end
        if "B" not in picks and ep_len > 30 and 12 <= sfid <= ep_len - 15:
            picks["B"] = idx
        # C: near terminal, has reasonable history
        if "C" not in picks and sfid >= 5 and ep_len - sfid <= 4 and ep_len >= 20:
            picks["C"] = idx
        if all(k in picks for k in ["A", "B", "C"]):
            break

    # D: terminal oversample = a sample whose start_frame_id appears more than once
    seen, dupes = {}, set()
    for s in ds.samples:
        key = (s["scene_path"], s["episode_idx"], s["start_frame_id"])
        if key in seen:
            dupes.add(key)
        else:
            seen[key] = True
    used = set(picks.values())
    for idx, s in enumerate(ds.samples):
        if idx in used:
            continue
        key = (s["scene_path"], s["episode_idx"], s["start_frame_id"])
        if key in dupes and s["episode_length"] - s["start_frame_id"] <= 5:
            picks["D"] = idx
            break

    return picks


# ---------------------------------------------------------------------------
# Plot a single sample
# ---------------------------------------------------------------------------

def plot_sample(ds: NavVideoDataset, idx: int, tag: str, out_path: str):
    """Render one training sample under the new (v2) pipeline.

    Layout: trajectory (left, ~3 cols) | action label table (mid, ~2 cols) |
            17 frame grid (right, ~5 cols).
    """
    info = ds.samples[idx]
    scene_path = info["scene_path"]
    ep_idx = info["episode_idx"]
    sfid = info["start_frame_id"]
    ep_len = info["episode_length"]
    instruction = info["instruction"]

    # Reproduce internal logic for video frame indices
    end_frame_id = min(sfid + ds.action_horizon + 1, ep_len)
    history_indices = ds._get_history_indices(sfid)
    future_indices = ds._get_future_indices(sfid, end_frame_id, ep_len)
    all_indices = history_indices + [sfid] + future_indices  # 17

    full_xy, full_poses = episode_full_traj_xy(scene_path, ep_idx, ds.overhead_camera)

    # ----- v2 action label (training target) -----------------------------
    # `compute_spline_actions_v2` returns:
    #   actions_norm: (T, 3) normalized to [-1, 1] — dims = (forward, left, dyaw)
    #   action_is_pad: (T,) bool — True where the step has no real motion
    actions_norm, action_is_pad = compute_spline_actions_v2(
        full_poses, sfid, end_frame_id,
        predict_step_num=ds.predict_step_num,
        camera_deg=30.0,
        alpha=ACTION_PROGRESS_ALPHA,
    )
    actions_unnorm = denormalize_action(actions_norm)  # physical units

    # Apply the same monotonic near-goal stop logic as in `_get` so the
    # visualization matches what the model actually sees.
    goal_pos = full_poses[-1, :3, 3]
    actual_end = min(sfid + ds.action_horizon, ep_len - 1)
    first_stop_step = ds.predict_step_num
    for i in range(ds.predict_step_num):
        frac = (i + 1) / ds.predict_step_num
        interp_frame = min(int(sfid + frac * (actual_end - sfid)), ep_len - 1)
        pos_i = full_poses[interp_frame, :3, 3]
        if np.linalg.norm(goal_pos - pos_i) < 1.0:
            first_stop_step = i
            break
    action_is_pad = action_is_pad.copy()
    action_is_pad[first_stop_step:] = True
    moving_flag = (~action_is_pad).astype(int)

    # ----- v2 anchor positions (in segment-local frame) -------------------
    seg_poses = full_poses[sfid:end_frame_id]
    if len(seg_poses) >= 2:
        seg_rel = get_trajectory_relative_to_frame(seg_poses, camera_deg=30.0)
    else:
        seg_rel = np.zeros((0, 3))
    anchor_xy_seg = v2_resampled_xy(
        seg_rel, ds.predict_step_num, alpha=ACTION_PROGRESS_ALPHA,
    )  # (T+1, 2)

    # Lift seg-local anchors into episode frame so they overlay correctly
    # on the full trajectory plot.
    if len(seg_poses) >= 1:
        camera_rad = np.radians(30.0)
        T_robot2camera = np.array(
            [[0.0, 0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )
        T_deg = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
            [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        T_robot2camera = T_robot2camera @ T_deg
        T_camera2robot_corr = np.linalg.inv(T_robot2camera)
        ep_robot = full_poses @ T_camera2robot_corr
        T_ref_ep = np.linalg.inv(ep_robot[0])
        seg_start_in_ep = T_ref_ep @ ep_robot[sfid]  # 4x4
        R2 = seg_start_in_ep[:2, :2]
        t2 = seg_start_in_ep[:2, 3]
        anchor_xy_ep = (R2 @ anchor_xy_seg.T).T + t2  # (T+1, 2)
    else:
        anchor_xy_ep = np.zeros_like(anchor_xy_seg)

    # ----- Layout ---------------------------------------------------------
    # 3 super-columns: traj (3) | action table (2) | frame grid (6)
    fig = plt.figure(figsize=(24, 9))
    gs = GridSpec(3, 11, figure=fig, hspace=0.35, wspace=0.25,
                  width_ratios=[1.5, 1.5, 1.5,    # traj (3 cols)
                                0.15,            # spacer
                                1.3, 1.3,        # action table (2 cols)
                                0.15,            # spacer
                                1, 1, 1, 1])     # frame grid (4 wide cols)

    # ===== LEFT: trajectory =================================================
    ax_traj = fig.add_subplot(gs[:, :3])
    ax_traj.plot(full_xy[:, 0], full_xy[:, 1], color="lightgrey", lw=1.2, zorder=1)
    ax_traj.scatter(full_xy[:, 0], full_xy[:, 1], color="lightgrey", s=8, zorder=1)
    ax_traj.plot(full_xy[0, 0], full_xy[0, 1], marker="*", color="green",
                 markersize=22, zorder=5, label="ep start")
    ax_traj.plot(full_xy[-1, 0], full_xy[-1, 1], marker="*", color="red",
                 markersize=22, zorder=5, label="ep goal")

    # Sub-segment (raw 17 frames in episode frame)
    seg_xy_in_ep = full_xy[sfid:end_frame_id]
    if len(seg_xy_in_ep) >= 1:
        ax_traj.plot(seg_xy_in_ep[:, 0], seg_xy_in_ep[:, 1], color="royalblue",
                     lw=3, zorder=3, label=f"raw sub-seg [{sfid}:{end_frame_id}]")
        ax_traj.scatter(seg_xy_in_ep[0, 0], seg_xy_in_ep[0, 1],
                        color="royalblue", s=120, zorder=6, label="seg start")
        ax_traj.scatter(seg_xy_in_ep[-1, 0], seg_xy_in_ep[-1, 1],
                        marker="s", color="purple", s=110, zorder=6, label="seg end")

    # v2 anchors (skip anchor[0] which is the seg start, identical to seg_start)
    anchors_plot = anchor_xy_ep[1:]  # (T, 2)
    for i, ((x, y), mf) in enumerate(zip(anchors_plot, moving_flag)):
        if mf == 1:
            ax_traj.scatter(x, y, marker="o", facecolor="orange",
                            edgecolor="black", s=110, zorder=7)
        else:
            ax_traj.scatter(x, y, marker="o", facecolor="white",
                            edgecolor="orange", lw=2, s=110, zorder=7)
        ax_traj.annotate(f"{i}:{mf}", (x, y), textcoords="offset points",
                         xytext=(7, 5), fontsize=9, color="black", zorder=8)
    if len(anchors_plot) > 0:
        ax_traj.plot(anchors_plot[:, 0], anchors_plot[:, 1], color="orange",
                     lw=1.4, ls="--", alpha=0.85, zorder=4,
                     label=f"v2 anchors (α={ACTION_PROGRESS_ALPHA})")

    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel("x (m, robot frame, ep-rel)")
    ax_traj.set_ylabel("y (m, robot frame, ep-rel)")
    ax_traj.set_title(
        f"[{tag}] {os.path.basename(os.path.dirname(scene_path))}/"
        f"{os.path.basename(scene_path)} ep={ep_idx} "
        f"start={sfid}/{ep_len-1} (action_horizon={ds.action_horizon})\n"
        f"first_stop_step={first_stop_step}  pad={action_is_pad.sum()}/{ds.predict_step_num}  "
        f"goal_dist(end)={np.linalg.norm(goal_pos - full_poses[actual_end, :3, 3]):.2f}m\n"
        f"instruction: {instruction[:90]}{'...' if len(instruction) > 90 else ''}",
        fontsize=10, loc="left",
    )
    ax_traj.legend(loc="best", fontsize=8, framealpha=0.85)

    # ===== CENTER: action label table (the actual training target) =========
    # Three stacked horizontal-bar plots, one per dim. Each has T=8 bars.
    ax_d0 = fig.add_subplot(gs[0, 4:6])
    ax_d1 = fig.add_subplot(gs[1, 4:6])
    ax_d2 = fig.add_subplot(gs[2, 4:6])

    T = ds.predict_step_num
    step_idx = np.arange(T)

    def _bar(ax, vals_norm, vals_phys, title, phys_unit, phys_fmt):
        # Greyed-out bars where moving_flag == 0
        colors = ["#2980b9" if mf == 1 else "#bdc3c7" for mf in moving_flag]
        bars = ax.bar(step_idx, vals_norm, color=colors, edgecolor="black",
                      linewidth=0.6)
        ax.axhline(0, color="black", lw=0.7)
        ax.axhline(1.0, color="grey", lw=0.5, ls=":", alpha=0.6)
        ax.axhline(-1.0, color="grey", lw=0.5, ls=":", alpha=0.6)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks(step_idx)
        ax.set_xticklabels([f"s{i}" for i in step_idx], fontsize=8)
        ax.set_ylabel("normalized [-1, 1]", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate with physical value below each bar's top
        for i, (b, vn, vp) in enumerate(zip(bars, vals_norm, vals_phys)):
            yy = vn + (0.06 if vn >= 0 else -0.06)
            ax.text(
                b.get_x() + b.get_width() / 2, yy,
                phys_fmt.format(vp),
                ha="center", va="bottom" if vn >= 0 else "top",
                fontsize=7, color="black",
            )
            # mark pad with red x
            if moving_flag[i] == 0:
                ax.text(
                    b.get_x() + b.get_width() / 2, 0,
                    "PAD", ha="center", va="center", fontsize=6,
                    color="red", fontweight="bold",
                )

    _bar(ax_d0, actions_norm[:, 0], actions_unnorm[:, 0],
         f"dim0  Δforward  (scale={ACTION_SCALE[0]:.3f}m)", "m", "{:+.3f}m")
    _bar(ax_d1, actions_norm[:, 1], actions_unnorm[:, 1],
         f"dim1  Δleft    (scale={ACTION_SCALE[1]:.3f}m)", "m", "{:+.3f}m")
    # For yaw, show degrees instead of radians (more intuitive)
    _bar(ax_d2, actions_norm[:, 2], np.degrees(actions_unnorm[:, 2]),
         f"dim2  Δyaw     (scale={ACTION_SCALE[2]:.3f}rad ≈ "
         f"{np.degrees(ACTION_SCALE[2]):.1f}°)", "deg", "{:+.1f}°")
    ax_d2.set_xlabel("action step index (0..T-1)", fontsize=8)

    # ===== RIGHT: 17-frame grid =============================================
    sub_gs = gs[:, 7:11].subgridspec(3, 6, hspace=0.35, wspace=0.05)
    labels = (["H"] * ds.n_history_frames) + ["C"] + (["F"] * ds.n_future_video_frames)
    label_colors = ((["#4a90e2"] * ds.n_history_frames)
                    + ["#27ae60"]
                    + (["#e67e22"] * ds.n_future_video_frames))

    # Detect which future indices are "padded by replication" (i.e. clamped to
    # ep_len-1 because future stride ran past the episode end). We mark these
    # with a red border so it's obvious in the visualization.
    last_real_frame = ep_len - 1
    is_padded = []
    for k, fidx in enumerate(all_indices):
        if k <= ds.n_history_frames:  # H or C: never padded
            is_padded.append(False)
            continue
        # F-frames: padded if the *unclamped* index exceeded ep_len-1
        future_pos = k - ds.n_history_frames  # 1..n_future
        unclamped = sfid + future_pos * ds.future_frame_stride
        is_padded.append(unclamped > last_real_frame)

    for k, fidx in enumerate(all_indices):
        if k >= 17:
            break
        r = k // 6
        c = k % 6
        ax = fig.add_subplot(sub_gs[r, c])
        try:
            frame = ds._load_and_resize_frame(scene_path, ds.primary_camera, ep_idx, fidx)
            img = frame.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"missing\n{fidx}\n{e}", ha="center", va="center",
                    fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        pad_tag = " ★PAD" if is_padded[k] else ""
        ax.set_title(f"{labels[k]}{k} f={fidx}{pad_tag}", fontsize=8,
                     color=label_colors[k])
        border_color = "red" if is_padded[k] else label_colors[k]
        border_width = 3 if is_padded[k] else 2
        for spine in ax.spines.values():
            spine.set_color(border_color); spine.set_linewidth(border_width)

    fig.suptitle(
        f"Sample idx={idx} | tag={tag} | NEW v2 pipeline "
        f"(α={ACTION_PROGRESS_ALPHA}, ACTION_SCALE={ACTION_SCALE.tolist()}) | "
        f"H=history ({ds.n_history_frames})  C=current  "
        f"F=future stride={ds.future_frame_stride}, red border = replicated last frame",
        fontsize=11, y=0.995,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return {
        "idx": idx,
        "scene": os.path.basename(scene_path),
        "dataset": os.path.basename(os.path.dirname(scene_path)),
        "ep_idx": ep_idx,
        "start_frame_id": sfid,
        "ep_len": ep_len,
        "first_stop_step": first_stop_step,
        "moving_flag": moving_flag.tolist(),
        "action_is_pad": action_is_pad.tolist(),
        "frames_to_end": ep_len - 1 - sfid,
        "n_replicated_frames": int(sum(is_padded)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",
                        default="/apdcephfs/wx_feature/home/xxd/debugdata/debug_data")
    parser.add_argument("--out-dir",
                        default="/apdcephfs/wx_feature/home/xxd/FastWAM/visualizations/nav_vln")
    parser.add_argument("--n", type=int, default=100,
                        help="number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-typed", action="store_true",
                        help="skip the A/B/C/D typed picks (only do random)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Building dataset from: {args.data_root}")
    ds = build_dataset(args.data_root)
    print(f"Total samples: {len(ds)}")

    stats = []

    # typed picks
    if not args.no_typed:
        picks = pick_samples(ds)
        print(f"Picked categories: {picks}")
        for tag, idx in picks.items():
            out = os.path.join(args.out_dir, f"sample_{tag}_idx{idx}.png")
            stats.append(plot_sample(ds, idx, tag, out))

    # ---- Stratified random sampling on start_frame_id / ep_len ratio --------
    # Divide [0, 1] into 5 buckets so that we cover open/early/mid/late/terminal
    # samples roughly evenly even if the dataset is skewed.
    rng = np.random.default_rng(args.seed)
    ratios = np.array([
        s["start_frame_id"] / max(1, s["episode_length"] - 1)
        for s in ds.samples
    ])
    n_buckets = 5
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    per_bucket = max(1, args.n // n_buckets)
    chosen = []
    for b in range(n_buckets):
        lo, hi = bucket_edges[b], bucket_edges[b + 1]
        if b == n_buckets - 1:
            mask = (ratios >= lo) & (ratios <= hi)
        else:
            mask = (ratios >= lo) & (ratios < hi)
        idx_pool = np.flatnonzero(mask)
        if len(idx_pool) == 0:
            continue
        take = min(per_bucket, len(idx_pool))
        picked = rng.choice(idx_pool, size=take, replace=False)
        chosen.extend(picked.tolist())

    # If we still need more, fill randomly from the remainder
    chosen_set = set(chosen)
    if len(chosen) < args.n:
        remaining = [i for i in range(len(ds)) if i not in chosen_set]
        extra = rng.choice(remaining, size=min(args.n - len(chosen), len(remaining)),
                           replace=False)
        chosen.extend(extra.tolist())
    chosen = sorted(chosen[: args.n],
                    key=lambda i: ds.samples[i]["start_frame_id"]
                    / max(1, ds.samples[i]["episode_length"] - 1))

    print(f"\nVisualizing {len(chosen)} stratified samples...")
    for j, idx in enumerate(chosen):
        s = ds.samples[idx]
        ratio = s["start_frame_id"] / max(1, s["episode_length"] - 1)
        # filename encodes the bucket so the file list shows progression
        tag = f"R{j:03d}_r{int(ratio * 100):03d}_ep{s['episode_idx']}_s{s['start_frame_id']}of{s['episode_length']-1}"
        out = os.path.join(args.out_dir, f"sample_{tag}_idx{idx}.png")
        stats.append(plot_sample(ds, int(idx), tag, out))

    # ---------- Statistics summary ------------------------------------------
    print("\n" + "=" * 60)
    print(f"Summary over {len(stats)} visualized samples")
    print("=" * 60)
    fss = np.array([s["first_stop_step"] for s in stats])
    fte = np.array([s["frames_to_end"] for s in stats])
    print(f"first_stop_step distribution (predict_step_num={ds.predict_step_num}):")
    for v in range(ds.predict_step_num + 1):
        n = int((fss == v).sum())
        bar = "#" * n
        meaning = "no stop (all moving)" if v == ds.predict_step_num else f"stop from step {v}"
        print(f"  {v:>2}: {n:>3}  {bar}   ({meaning})")
    print(f"\nframes_to_end distribution (ep_len-1 - start_frame_id):")
    for cutoff in [0, 1, 2, 3, 4, 5, 8, 16, 32]:
        n = int((fte <= cutoff).sum())
        print(f"  <= {cutoff:>2} frames remain: {n:>3} samples")

    # Persist stats
    import json
    stats_path = os.path.join(args.out_dir, "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print(f"All figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
