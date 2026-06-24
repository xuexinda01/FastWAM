"""Diagnostic: how often does InternNav's NavPixelGoalDataset.__getitem__
return a dict missing 'input_ids' (or other required keys)?

Run a sweep over N random samples through the *inner* InternNav
__getitem__ (bypassing our retry wrapper) and classify results:
  OK              — dict with all 6 required ChatML keys
  MISSING_KEYS    — dict missing one or more required keys
  WRONG_TYPE      — not a dict
  EXCEPTION       — __getitem__ raised

This reveals whether the 8x8 crash was:
(a) rare (1/100k) → retry handles it fine
(b) common  (>1%) → retry will burn its 8-attempt budget too fast,
    and we'd need to investigate the InternNav code path directly.

Also tests N samples *concurrently* via a thread pool to mimic
64-rank dataloader stress.

Usage:
    python scripts/diagnose_bad_samples.py --n 500 --datasets r2r_125cm_0_30
    python scripts/diagnose_bad_samples.py --n 200 --datasets r2r_125cm_0_30,scalevln_125cm_0_30
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
_FASTWAM_ROOT = '/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM'
_INTERNNAV_ROOT = '/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav'
sys.path.insert(0, os.path.join(_FASTWAM_ROOT, 'src'))
sys.path.insert(0, _INTERNNAV_ROOT)
os.environ.setdefault('FASTWAM_ANNOTATION_CACHE',
                      f'{_FASTWAM_ROOT}/data/internnav_annotations_cache')

REQUIRED = ('input_ids', 'labels', 'position_ids', 'attention_mask',
            'pixel_values', 'image_grid_thw')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200,
                   help='Number of samples to test')
    p.add_argument('--datasets', default='r2r_125cm_0_30',
                   help='Comma-separated dataset keys')
    p.add_argument('--workers', type=int, default=8,
                   help='Concurrent threads (mimic dataloader workers)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    # Build minimal data_args
    import argparse as _ap
    data_args = _ap.Namespace()
    data_args.vln_dataset_use = args.datasets
    data_args.data_path = ''
    data_args.data_flatten = False
    data_args.data_packing = False
    data_args.video_max_total_pixels = 1664 * 28 * 28
    data_args.video_min_total_pixels = 256 * 28 * 28
    data_args.model_type = 'qwen2.5vl'
    data_args.sample_step = 4
    data_args.predict_step_num = 32
    data_args.pixel_goal_only = False
    data_args.num_future_steps = 4
    data_args.num_history = 8
    data_args.data_augmentation = True
    data_args.resize_h = 384
    data_args.resize_w = 384

    from torchvision.transforms import v2
    data_args.transform_train = v2.Resize((384, 384))

    from transformers import AutoProcessor, AutoTokenizer
    qwen = '/apdcephfs_qy2/share_303214315/hunyuan/xxd/ckpts/Qwen3-VL-4B-Instruct'
    print(f'[diag] Loading processor + tokenizer from {qwen}', flush=True)
    processor = AutoProcessor.from_pretrained(
        qwen, trust_remote_code=True, local_files_only=True)
    data_args.image_processor = processor.image_processor
    tokenizer = AutoTokenizer.from_pretrained(
        qwen, model_max_length=8192, padding_side='right',
        use_fast=False, trust_remote_code=True, local_files_only=True)
    data_args.fastwam_video_size = 224
    data_args.fastwam_n_history_frames = 9
    data_args.fastwam_n_future_frames = 8
    data_args.fastwam_predict_step_num = 8

    from fastwam.datasets.lerobot.internvla_n1_hfastwam_dataset import (
        InternVLAN1HFastWAMDataset,
    )
    print(f'[diag] Building wrapper dataset (datasets={args.datasets})', flush=True)
    t0 = time.time()
    ds = InternVLAN1HFastWAMDataset(tokenizer=tokenizer, data_args=data_args)
    print(f'[diag] Built in {time.time() - t0:.1f}s, len={len(ds)}', flush=True)

    inner = ds._inner

    random.seed(args.seed)
    indices = random.sample(range(len(inner)), min(args.n, len(inner)))

    # ---- Probe each idx ---- #
    def probe(idx):
        try:
            out = inner[idx]
        except Exception as exc:
            return ('EXCEPTION', idx, type(exc).__name__,
                    str(exc)[:200], traceback.format_exc())
        if not isinstance(out, dict):
            return ('WRONG_TYPE', idx, type(out).__name__, '', '')
        missing = [k for k in REQUIRED if k not in out]
        if missing:
            return ('MISSING_KEYS', idx, ','.join(missing),
                    ','.join(sorted(out.keys())), '')
        return ('OK', idx, '', '', '')

    print(f'[diag] Probing {len(indices)} samples with {args.workers} threads', flush=True)
    categories = Counter()
    missing_keys = Counter()
    exceptions = Counter()
    examples: dict[str, list] = {}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, i): i for i in indices}
        for k, fut in enumerate(as_completed(futures)):
            cat, idx, info, extra, tb = fut.result()
            categories[cat] += 1
            if cat == 'MISSING_KEYS':
                for m in info.split(','):
                    missing_keys[m] += 1
            elif cat == 'EXCEPTION':
                exceptions[info] += 1
            if cat != 'OK':
                examples.setdefault(cat, []).append((idx, info, extra, tb))
            if (k + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (k + 1) / max(elapsed, 0.1)
                print(f'  progress: {k + 1}/{len(indices)} ({rate:.1f}/s)', flush=True)

    elapsed = time.time() - t_start

    # ---- Summary ---- #
    n = sum(categories.values())
    print()
    print('=' * 60)
    print(f'Total samples tested: {n}  ({elapsed:.1f}s, {n/elapsed:.1f}/s)')
    for cat, count in categories.most_common():
        print(f'  {cat:15s}  {count:5d}  ({100 * count / n:.2f}%)')

    if missing_keys:
        print()
        print('Missing keys breakdown:')
        for k, c in missing_keys.most_common():
            print(f'  {k:20s}  {c}')

    if exceptions:
        print()
        print('Exception types:')
        for k, c in exceptions.most_common():
            print(f'  {k:30s}  {c}')

    print()
    if 'MISSING_KEYS' in examples:
        print('First 3 MISSING_KEYS examples:')
        for idx, missing, got, _ in examples['MISSING_KEYS'][:3]:
            print(f'  idx={idx} missing={missing} got_keys={got}')

    if 'EXCEPTION' in examples:
        print('First 3 EXCEPTION examples:')
        for idx, exc_type, msg, tb in examples['EXCEPTION'][:3]:
            print(f'  idx={idx} {exc_type}: {msg}')
            tb_lines = tb.strip().splitlines()
            for line in tb_lines[-6:]:
                print(f'    {line}')
            print()

    # ---- Verdict ---- #
    print()
    print('=' * 60)
    fail = n - categories.get('OK', 0)
    fail_rate = fail / n if n else 0.0
    if fail_rate == 0.0:
        print('VERDICT: 0 failures. Retry pattern is overkill but harmless.')
    elif fail_rate < 0.01:
        print(f'VERDICT: {100 * fail_rate:.2f}% fail rate — RARE. Retry handles it (8 attempts gives p={1 - (1 - fail_rate) ** 8:.5f} of cascading failure).')
    elif fail_rate < 0.1:
        print(f'VERDICT: {100 * fail_rate:.2f}% fail rate — COMMON. Retry still works but training silently skips many samples; investigate root cause.')
    else:
        print(f'VERDICT: {100 * fail_rate:.2f}% fail rate — SYSTEMATIC. Retry will eat its 8-attempt budget. Need to fix InternNav code path or filter dataset.')

    return 0 if fail_rate < 0.5 else 1


if __name__ == '__main__':
    sys.exit(main())
