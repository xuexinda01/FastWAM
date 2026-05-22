"""
Pre-compute _classify_samples cache with multi-threaded parallel parquet reads.

Usage:
    python scripts/precompute_classify_fast.py

Uses ThreadPoolExecutor to parallelize NFS reads (IO-bound), dramatically
faster than the serial version.
"""
import sys
sys.path.insert(0, '/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/src')

import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---- Minimal imports from the dataset module ----
from fastwam.datasets.lerobot.nav_video_dataset import (
    NavVideoDataset,
    get_trajectory_relative_to_frame,
)

NUM_WORKERS = 64  # threads for parallel parquet reads


def load_poses_for_episode(scene_path: str, episode_idx: int, camera_key: str) -> Tuple[str, int, np.ndarray]:
    """Load poses for a single episode. Returns (scene_path, episode_idx, poses)."""
    parquet_path = os.path.join(
        scene_path, "data", "chunk-000",
        f"episode_{episode_idx:06d}.parquet"
    )
    df = pd.read_parquet(parquet_path, columns=[f"pose.{camera_key}"])
    poses_raw = df[f"pose.{camera_key}"].tolist()
    poses = np.array([np.vstack(p) for p in poses_raw])
    return (scene_path, episode_idx, poses)


def main():
    print("=" * 60)
    print("Pre-computing _classify_samples cache (multi-threaded)")
    print(f"Using {NUM_WORKERS} threads for parallel NFS reads")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Build dataset index (no parquet reading yet)
    print("\n[1/4] Building dataset index...")
    ds = NavVideoDataset.__new__(NavVideoDataset)
    # Minimal init to get samples list
    ds.camera_keys = ['125cm_0deg', '125cm_30deg']
    ds.num_frames = 8
    ds.n_history_frames = 8
    ds.n_future_video_frames = 8
    ds.action_video_freq_ratio = 1
    ds.video_size = [224, 224]
    ds.concat_multi_camera = 'none'
    ds.text_embedding_cache_dir = '/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/text_embeds_cache/nav_vln'
    ds.context_len = 256
    ds.sample_stride = 4
    ds.terminal_oversample_ratio = 0.5
    ds.predict_step_num = 8
    ds.min_goal_len = 3
    ds.overhead_camera = '125cm_30deg'  # camera_keys[1] per dataset code
    ds._camera_deg = 30
    ds.action_horizon = ds.predict_step_num  # 8
    ds.samples = []

    dataset_dirs = [
        '/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r',
        '/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/rxr',
        '/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/scalevln',
    ]
    ds._build_index(dataset_dirs)
    print(f"    Index built: {len(ds.samples)} samples")

    # Step 2: Identify unique episodes that need pose loading
    print("\n[2/4] Identifying episodes that need pose data...")
    action_horizon = ds.action_horizon
    episodes_needed = set()
    terminal_indices = set()

    for s_idx, info in enumerate(ds.samples):
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        end = min(sfid + action_horizon + 1, ep_len)

        if end >= ep_len - 0:
            terminal_indices.add(s_idx)
            continue
        if (end - sfid) < action_horizon + 1:
            terminal_indices.add(s_idx)
            continue
        episodes_needed.add((info["scene_path"], info["episode_idx"]))

    print(f"    TERMINAL samples (no pose needed): {len(terminal_indices)}")
    print(f"    Unique episodes to load: {len(episodes_needed)}")

    # Step 3: Parallel load all episode poses
    print(f"\n[3/4] Loading {len(episodes_needed)} episode poses ({NUM_WORKERS} threads)...")
    pose_cache: Dict[Tuple[str, int], np.ndarray] = {}
    camera_key = ds.overhead_camera
    episodes_list = list(episodes_needed)
    total_eps = len(episodes_list)

    loaded = 0
    failed = 0
    last_report = time.time()
    report_interval = 5  # seconds

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {
            pool.submit(load_poses_for_episode, sp, ei, camera_key): (sp, ei)
            for sp, ei in episodes_list
        }

        for future in as_completed(futures):
            try:
                scene_path, episode_idx, poses = future.result()
                pose_cache[(scene_path, episode_idx)] = poses
                loaded += 1
            except Exception as e:
                failed += 1
                loaded += 1

            # Real-time progress
            now = time.time()
            if now - last_report >= report_interval:
                elapsed = now - t0
                eps_per_sec = loaded / (now - t0 + 0.001)
                remaining = (total_eps - loaded) / max(eps_per_sec, 0.001)
                pct = loaded * 100.0 / total_eps
                print(f"    [{loaded}/{total_eps}] {pct:.1f}%  "
                      f"speed: {eps_per_sec:.1f} eps/s  "
                      f"ETA: {remaining:.0f}s ({remaining/60:.1f}min)  "
                      f"failed: {failed}", flush=True)
                last_report = now

    print(f"    Done loading poses: {loaded} episodes, {failed} failed, "
          f"{time.time() - t0:.1f}s elapsed")

    # Step 4: Classify all samples using loaded poses
    print(f"\n[4/4] Classifying {len(ds.samples)} samples...")
    TURN_THRESHOLD_SMALL = np.deg2rad(15)
    TURN_THRESHOLD_BIG = np.deg2rad(45)
    FWD_THRESHOLD = 0.30
    camera_deg = ds._camera_deg

    cats: List[str] = []
    for s_idx, info in enumerate(ds.samples):
        sfid = info["start_frame_id"]
        ep_len = info["episode_length"]
        end = min(sfid + action_horizon + 1, ep_len)

        if s_idx in terminal_indices:
            cats.append("TERMINAL")
            continue

        cache_key = (info["scene_path"], info["episode_idx"])
        poses = pose_cache.get(cache_key)
        if poses is None or end > len(poses):
            cats.append("OTHER")
            continue

        try:
            rel = get_trajectory_relative_to_frame(poses[sfid:end], camera_deg=camera_deg)
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

    # Save cache
    cache_key_str = json.dumps({
        "n_samples": len(ds.samples),
        "action_horizon": action_horizon,
        "overhead_camera": ds.overhead_camera,
        "camera_deg": ds._camera_deg,
        "first_sample": str(ds.samples[0]) if ds.samples else "",
        "last_sample": str(ds.samples[-1]) if ds.samples else "",
    }, sort_keys=True)
    cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:12]
    _shared_cache_dir = "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/.cache"
    os.makedirs(_shared_cache_dir, exist_ok=True)
    cache_path = f"{_shared_cache_dir}/nav_classify_cache_{cache_hash}.pkl"

    with open(cache_path, "wb") as f:
        pickle.dump(cats, f)

    total_time = time.time() - t0
    from collections import Counter
    dist = Counter(cats)
    print(f"\n{'=' * 60}")
    print(f"DONE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Cache saved to: {cache_path}")
    print(f"Distribution: {dict(dist)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
