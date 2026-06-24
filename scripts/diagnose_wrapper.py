"""Minimum repro: just call wrapper's __getitem__ and dump keys.

If the wrapper.__getitem__ returns a dict with 'input_ids', the bug
is somewhere between wrapper.__getitem__ and the dataloader/collator
boundary. If it doesn't, the bug is in our wrapper.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, '/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/src')
sys.path.insert(0, '/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav')
os.environ['FASTWAM_ANNOTATION_CACHE'] = '/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache'

import argparse
data_args = argparse.Namespace()
data_args.vln_dataset_use = 'r2r_125cm_0_30'
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
print(f'Loading processor + tokenizer...', flush=True)
processor = AutoProcessor.from_pretrained(qwen, trust_remote_code=True, local_files_only=True)
data_args.image_processor = processor.image_processor
tokenizer = AutoTokenizer.from_pretrained(
    qwen, model_max_length=8192, padding_side='right',
    use_fast=False, trust_remote_code=True, local_files_only=True)
data_args.fastwam_video_size = 224
data_args.fastwam_n_history_frames = 9
data_args.fastwam_n_future_frames = 8
data_args.fastwam_predict_step_num = 8

from fastwam.datasets.lerobot.internvla_n1_hfastwam_dataset import InternVLAN1HFastWAMDataset

print('Building wrapper...', flush=True)
ds = InternVLAN1HFastWAMDataset(tokenizer=tokenizer, data_args=data_args)
print(f'len={len(ds)}', flush=True)

# Call wrapper.__getitem__ on N indices
import random
random.seed(42)
indices = random.sample(range(len(ds)), 50)

bad = 0
good = 0
for k, i in enumerate(indices):
    try:
        out = ds[i]
    except Exception as exc:
        print(f'  idx={i} EXCEPTION {type(exc).__name__}: {exc}', flush=True)
        bad += 1
        continue
    if not isinstance(out, dict):
        print(f'  idx={i} WRONG_TYPE: {type(out).__name__}', flush=True)
        bad += 1
        continue
    if 'input_ids' not in out:
        print(f'  idx={i} NO_INPUT_IDS keys={sorted(out.keys())}', flush=True)
        bad += 1
        continue
    good += 1
    if k < 3:
        # Detailed dump of first 3 OK samples
        ks = sorted(out.keys())
        types = {kk: type(out[kk]).__name__ for kk in ks}
        shapes = {}
        for kk in ks:
            v = out[kk]
            if hasattr(v, 'shape'):
                shapes[kk] = tuple(v.shape)
            elif hasattr(v, '__len__') and not isinstance(v, str):
                shapes[kk] = f'len={len(v)}'
            else:
                shapes[kk] = repr(v)[:30]
        print(f'  idx={i} OK keys={ks}', flush=True)
        for kk in ks:
            print(f'         {kk:20s} type={types[kk]:15s} shape={shapes[kk]}', flush=True)

print(f'\nResult: {good} OK, {bad} bad out of {len(indices)}', flush=True)
