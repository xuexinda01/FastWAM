"""
Pre-compute _classify_samples cache with a single process.

Usage:
    python scripts/precompute_classify.py

This avoids 64 processes competing for NFS bandwidth during training startup.
After this script completes, training will load the cached result in seconds.
"""
import sys
sys.path.insert(0, '/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/src')

import time
from fastwam.datasets.lerobot.nav_video_dataset import NavVideoDataset

print("=" * 60)
print("Pre-computing _classify_samples cache (single process)")
print("This reads parquet files from NFS — expect ~30-40 minutes.")
print("=" * 60)

t0 = time.time()

ds = NavVideoDataset(
    dataset_dirs=[
        '/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r',
        '/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/rxr',
        '/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/scalevln',
    ],
    camera_keys=['125cm_0deg', '125cm_30deg'],
    num_frames=8,
    n_history_frames=8,
    n_future_video_frames=8,
    action_video_freq_ratio=1,
    video_size=[224, 224],
    concat_multi_camera='none',
    text_embedding_cache_dir='/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/text_embeds_cache/nav_vln',
    context_len=256,
    sample_stride=4,
    terminal_oversample_ratio=0.5,
    predict_step_num=8,
    min_goal_len=3,
)

elapsed = time.time() - t0
print(f"\nDone! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Dataset samples: {len(ds)}")
print("Cache file saved — next training launch will load it in seconds.")
