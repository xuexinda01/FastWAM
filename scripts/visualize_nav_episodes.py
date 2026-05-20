"""
Per-episode visualization of NavVideoDataset training samples.

Layout:
    <out-dir>/
        ep_000_<dataset>_<scene>_ep<ep_idx>_len<ep_len>_n<num_samples>/
            sample_00_sfid000.png
            sample_01_sfid004.png
            ...
            _episode_overview.png    # one whole-episode summary (trajectory + sampling)
        ep_001_.../
            ...
        _episode_stats.json

Each per-sample PNG reuses `plot_sample` from `visualize_nav_samples.py`, so the
content (left trajectory panel + right 17-frame grid) is identical to the existing
visualizations — but here every sample of every episode is dumped into its own
file inside that episode's folder, instead of being mixed together.

The `_episode_overview.png` per folder summarizes WHERE on the trajectory all the
samples were taken from (good for understanding sample_stride and terminal
oversampling at a glance).

Usage:
    python visualize_nav_episodes.py \
        --data-root /apdcephfs/wx_feature/home/xxd/debugdata/debug_data \
        --out-dir   /apdcephfs/wx_feature/home/xxd/FastWAM/visualizations/nav_vln_per_ep \
        --n-episodes 8

    # render a specific episode
    python visualize_nav_episodes.py --episode "<scene_basename>:3"
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

PROJECT_ROOT = "/apdcephfs/wx_feature/home/xxd/FastWAM"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from fastwam.datasets.lerobot.nav_video_dataset import (  # noqa: E402
    NavVideoDataset,
    get_trajectory_relative_to_frame,
)
# Reuse the existing per-sample renderer (left trajectory + right 17-frame grid)
from visualize_nav_samples import build_dataset, plot_sample  # noqa: E402


# ---------------------------------------------------------------------------
# Per-episode grouping
# ---------------------------------------------------------------------------

def group_samples_by_episode(ds: NavVideoDataset):
    """Return dict {(scene_path, ep_idx): list of (sample_idx, sample_info)}."""
    groups = defaultdict(list)
    for s_idx, info in enumerate(ds.samples):
        key = (info["scene_path"], info["episode_idx"])
        groups[key].append((s_idx, info))
    # Sort each episode's samples by start_frame_id, then sample idx (stable)
    for k in groups:
        groups[k].sort(key=lambda x: (x[1]["start_frame_id"], x[0]))
    return groups


def pick_episodes(groups: dict, n_episodes: int, seed: int = 42) -> List[tuple]:
    """Stratify episodes by length so we cover short/medium/long."""
    keys = list(groups.keys())
    lengths = np.array([groups[k][0][1]["episode_length"] for k in keys])

    rng = np.random.default_rng(seed)
    n_buckets = 4
    edges = np.quantile(lengths, np.linspace(0, 1, n_buckets + 1))
    per_bucket = max(1, n_episodes // n_buckets)
    chosen = []
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        if b == n_buckets - 1:
            mask = (lengths >= lo) & (lengths <= hi)
        else:
            mask = (lengths >= lo) & (lengths < hi)
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            continue
        take = min(per_bucket, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())

    if len(chosen) < n_episodes:
        remaining = [i for i in range(len(keys)) if i not in set(chosen)]
        if remaining:
            extra = rng.choice(
                remaining, size=min(n_episodes - len(chosen), len(remaining)),
                replace=False,
            )
            chosen.extend(extra.tolist())

    chosen = chosen[:n_episodes]
    chosen.sort(key=lambda i: lengths[i])  # short → long for nicer file order
    return [keys[i] for i in chosen]


# ---------------------------------------------------------------------------
# Trajectory loading (with yaw)
# ---------------------------------------------------------------------------

def load_episode_xyyaw(scene_path: str, episode_idx: int, camera_key: str):
    """Return (xy [N,2], yaw [N]) in episode-relative robot frame."""
    parquet_path = os.path.join(
        scene_path, "data", "chunk-000", f"episode_{episode_idx:06d}.parquet",
    )
    df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
    poses_raw = df[f"pose.{camera_key}"].tolist()
    poses = np.array([np.vstack(p) for p in poses_raw])
    rel = get_trajectory_relative_to_frame(poses, camera_deg=30.0)
    return rel[:, :2], rel[:, 2]


# ---------------------------------------------------------------------------
# Per-episode "overview" image (one image per episode, summarizes sampling)
# ---------------------------------------------------------------------------

def plot_episode_overview(ds: NavVideoDataset, scene_path: str, ep_idx: int,
                          sample_entries: List[tuple], out_path: str):
    ep_len = sample_entries[0][1]["episode_length"]
    instruction = sample_entries[0][1]["instruction"]
    xy, yaw = load_episode_xyyaw(scene_path, ep_idx, ds.overhead_camera)

    sfid_count = defaultdict(int)
    for _, info in sample_entries:
        sfid_count[info["start_frame_id"]] += 1

    n_samples = len(sample_entries)
    cmap = plt.get_cmap("viridis")
    terminal_start = max(0, ep_len - 5)

    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.6, 1.0],
                  hspace=0.45, wspace=0.18)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_mult = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, 1])
    ax_yaw = fig.add_subplot(gs[2, 1])

    # ---- trajectory ----
    ax_traj.plot(xy[:, 0], xy[:, 1], color="lightgrey", lw=1.2, zorder=1)
    ax_traj.scatter(xy[:, 0], xy[:, 1], color="lightgrey", s=10, zorder=1)

    sel = np.linspace(0, len(xy) - 1, num=min(40, len(xy)), dtype=int)
    arrow_len = max(0.05, 0.04 * (xy.max(0) - xy.min(0)).max())
    ax_traj.quiver(
        xy[sel, 0], xy[sel, 1],
        np.cos(yaw[sel]) * arrow_len, np.sin(yaw[sel]) * arrow_len,
        angles="xy", scale_units="xy", scale=1,
        color="grey", width=0.003, alpha=0.6, zorder=2,
    )

    ax_traj.plot(xy[0, 0], xy[0, 1], marker="*", color="green",
                 markersize=22, zorder=6, label="ep start")
    ax_traj.plot(xy[-1, 0], xy[-1, 1], marker="*", color="red",
                 markersize=22, zorder=6, label="ep goal")

    if terminal_start < ep_len:
        ax_traj.plot(
            xy[terminal_start:, 0], xy[terminal_start:, 1],
            color="crimson", lw=4, alpha=0.25, zorder=2,
            label=f"terminal window [{terminal_start}:{ep_len-1}]",
        )

    seen_sfid = set()
    for j, (_, info) in enumerate(sample_entries):
        sfid = info["start_frame_id"]
        color = cmap(j / max(1, n_samples - 1))
        if sfid in seen_sfid:
            continue
        seen_sfid.add(sfid)
        is_terminal = sfid >= terminal_start
        n_dup = sfid_count[sfid]
        size = 60 + 25 * (n_dup - 1)
        marker = "X" if is_terminal else "o"
        ax_traj.scatter(
            xy[sfid, 0], xy[sfid, 1],
            color=color, s=size, marker=marker,
            edgecolor="black", linewidth=0.6, zorder=5,
        )
        ax_traj.annotate(f"{j}", (xy[sfid, 0], xy[sfid, 1]),
                         textcoords="offset points", xytext=(5, 5),
                         fontsize=7, color="black", zorder=8)

    disp = np.linalg.norm(xy[-1] - xy[0])
    path_len = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    yaw_range_deg = float(np.degrees(yaw.max() - yaw.min()))

    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel("x (m, robot frame, ep-rel)")
    ax_traj.set_ylabel("y (m, robot frame, ep-rel)")
    ax_traj.set_title(
        f"{os.path.basename(os.path.dirname(scene_path))}/"
        f"{os.path.basename(scene_path)} ep={ep_idx} "
        f"len={ep_len} | #samples={n_samples} | sample_stride={ds.sample_stride} "
        f"terminal_oversample={ds.terminal_oversample_ratio}\n"
        f"|goal-start|={disp:.2f}m  path_len={path_len:.2f}m  yaw_range={yaw_range_deg:.1f}°\n"
        f"instruction: {instruction[:120]}{'...' if len(instruction) > 120 else ''}",
        fontsize=10, loc="left",
    )
    ax_traj.legend(loc="best", fontsize=8, framealpha=0.85)

    # ---- multiplicity bar ----
    sfids_sorted = sorted(sfid_count.keys())
    counts = [sfid_count[s] for s in sfids_sorted]
    bar_colors = ["crimson" if s >= terminal_start else "steelblue" for s in sfids_sorted]
    ax_mult.bar(sfids_sorted, counts, color=bar_colors, width=0.8)
    ax_mult.axvspan(terminal_start - 0.5, ep_len - 0.5, color="crimson", alpha=0.1)
    ax_mult.set_xlim(-1, ep_len)
    ax_mult.set_ylim(0, max(counts) + 1)
    ax_mult.set_xlabel("start_frame_id")
    ax_mult.set_ylabel("# samples")
    ax_mult.set_title("Sampling multiplicity per start_frame_id (red = terminal window)",
                      fontsize=9)
    ax_mult.grid(True, alpha=0.3, axis="y")

    # ---- timeline ----
    for j, (_, info) in enumerate(sample_entries):
        sfid = info["start_frame_id"]
        end = min(sfid + ds.action_horizon + 1, ep_len)
        color = cmap(j / max(1, n_samples - 1))
        ax_time.plot([sfid, end - 1], [j, j], color=color, lw=2.5, alpha=0.85)
        ax_time.scatter(sfid, j, color=color, s=18, zorder=4,
                        edgecolor="black", linewidth=0.4)
    ax_time.axvspan(terminal_start - 0.5, ep_len - 0.5, color="crimson", alpha=0.12)
    ax_time.set_xlim(-1, ep_len)
    ax_time.set_ylim(-1, n_samples)
    ax_time.set_xlabel("frame index")
    ax_time.set_ylabel("sample # (within episode)")
    ax_time.set_title(
        f"Sample coverage timeline (each bar = [sfid, sfid+{ds.action_horizon}])",
        fontsize=9,
    )
    ax_time.grid(True, alpha=0.3)

    # ---- yaw curve ----
    yaw_deg = np.degrees(yaw)
    yaw_unw = np.degrees(np.unwrap(yaw))
    ax_yaw.plot(np.arange(ep_len), yaw_deg, color="grey", lw=1.0, alpha=0.6,
                label="yaw (wrapped)")
    ax_yaw.plot(np.arange(ep_len), yaw_unw, color="navy", lw=1.5, label="yaw (unwrapped)")
    ax_yaw.axvspan(terminal_start - 0.5, ep_len - 0.5, color="crimson", alpha=0.12)
    for s in sorted(sfid_count.keys()):
        ax_yaw.axvline(s, color="orange", lw=0.5, alpha=0.4)
    ax_yaw.set_xlim(-1, ep_len)
    ax_yaw.set_xlabel("frame index")
    ax_yaw.set_ylabel("yaw (deg)")
    ax_yaw.set_title("Per-frame yaw (orange ticks = sampled start frames)", fontsize=9)
    ax_yaw.legend(loc="best", fontsize=8)
    ax_yaw.grid(True, alpha=0.3)

    fig.suptitle(
        f"Episode overview  |  action_horizon={ds.action_horizon}  "
        f"predict_step_num={ds.predict_step_num}",
        fontsize=11, y=0.995,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "disp_m": float(disp),
        "path_len_m": path_len,
        "yaw_range_deg": yaw_range_deg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/apdcephfs/wx_feature/home/xxd/debugdata/debug_data",
    )
    parser.add_argument(
        "--out-dir",
        default="/apdcephfs/wx_feature/home/xxd/FastWAM/visualizations/nav_vln_per_ep",
    )
    parser.add_argument("--n-episodes", type=int, default=8,
                        help="number of episodes to visualize (stratified by length)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--episode", type=str, default=None,
        help='Render one specific episode, format "<scene_basename>:<ep_idx>". '
             'Overrides --n-episodes.',
    )
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip the per-episode overview summary image.")
    parser.add_argument("--max-samples-per-ep", type=int, default=None,
                        help="Cap samples rendered per episode (default: all).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Building dataset from: {args.data_root}")
    ds = build_dataset(args.data_root)
    print(f"Total samples: {len(ds)}")

    groups = group_samples_by_episode(ds)
    print(f"Total episodes: {len(groups)}")

    if args.episode is not None:
        scene_name, ep_str = args.episode.rsplit(":", 1)
        ep_idx_target = int(ep_str)
        keys = [k for k in groups
                if os.path.basename(k[0]) == scene_name and k[1] == ep_idx_target]
        if not keys:
            raise ValueError(f"Episode '{args.episode}' not found.")
        chosen_keys = keys
    else:
        chosen_keys = pick_episodes(groups, args.n_episodes, args.seed)

    stats = []
    total_imgs = 0
    for k_idx, key in enumerate(chosen_keys):
        scene_path, ep_idx = key
        sample_entries = groups[key]
        ep_len = sample_entries[0][1]["episode_length"]
        scene_short = os.path.basename(scene_path)
        ds_short = os.path.basename(os.path.dirname(scene_path))

        ep_dir_name = (
            f"ep_{k_idx:03d}_{ds_short}_{scene_short}_ep{ep_idx}"
            f"_len{ep_len}_n{len(sample_entries)}"
        )
        ep_dir = os.path.join(args.out_dir, ep_dir_name)
        os.makedirs(ep_dir, exist_ok=True)

        print(f"\n[ep {k_idx+1}/{len(chosen_keys)}] {ds_short}/{scene_short} "
              f"ep={ep_idx} len={ep_len} #samples={len(sample_entries)} → {ep_dir}")

        # Per-sample images (each call writes one PNG)
        n_render = (
            len(sample_entries) if args.max_samples_per_ep is None
            else min(args.max_samples_per_ep, len(sample_entries))
        )
        for j, (s_idx, info) in enumerate(sample_entries[:n_render]):
            sfid = info["start_frame_id"]
            tag = f"j{j:02d}_sfid{sfid:03d}"
            out_name = f"sample_{j:02d}_sfid{sfid:03d}_idx{s_idx}.png"
            out = os.path.join(ep_dir, out_name)
            try:
                plot_sample(ds, s_idx, tag, out)
                total_imgs += 1
            except Exception as e:
                print(f"  [skip sample {j} sfid={sfid} idx={s_idx}] {e}")

        # Per-episode overview
        ep_stats = {
            "scene": scene_short,
            "dataset": ds_short,
            "ep_idx": ep_idx,
            "ep_len": ep_len,
            "n_samples": len(sample_entries),
            "n_unique_sfid": len({info["start_frame_id"] for _, info in sample_entries}),
            "n_rendered": n_render,
            "out_dir": ep_dir,
        }
        if not args.no_overview:
            try:
                overview_path = os.path.join(ep_dir, "_episode_overview.png")
                ep_stats.update(
                    plot_episode_overview(ds, scene_path, ep_idx,
                                          sample_entries, overview_path)
                )
                print(f"  [overview] {overview_path}")
            except Exception as e:
                print(f"  [overview skipped] {e}")
        stats.append(ep_stats)

    # ----- Summary -----
    if stats:
        print("\n" + "=" * 60)
        print(f"Rendered {len(stats)} episodes, {total_imgs} per-sample images total")
        print("=" * 60)
        ns = np.array([s["n_samples"] for s in stats])
        lens = np.array([s["ep_len"] for s in stats])
        print(f"ep_len      :  min={lens.min()}  max={lens.max()}  mean={lens.mean():.1f}")
        print(f"#samples/ep :  min={ns.min()}  max={ns.max()}  mean={ns.mean():.1f}")
        if "disp_m" in stats[0]:
            disps = np.array([s["disp_m"] for s in stats])
            yaw_r = np.array([s["yaw_range_deg"] for s in stats])
            print(f"|goal-start|:  min={disps.min():.2f}m  max={disps.max():.2f}m  "
                  f"mean={disps.mean():.2f}m")
            print(f"yaw_range   :  min={yaw_r.min():.1f}°  max={yaw_r.max():.1f}°  "
                  f"mean={yaw_r.mean():.1f}°")

        with open(os.path.join(args.out_dir, "_episode_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\nAll figures saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
