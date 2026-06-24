#!/usr/bin/env python3
"""Pre-cache preprocess_qwen_2_visual results for all training samples.

KEY INSIGHT: preprocess_qwen_2_visual result is fully determined by
(instruction_text, action_label, n_images). NOT by ep_id or frame_id.
This reduces ~6M samples to ~1M unique combinations — a 6× reduction.

Cache key : sha256(instruction + "|" + action_str + "|" + str(n_images))[:16]
Cache file: <output_dir>/<key>.pt
            Each .pt contains: input_ids, labels, n_images, grid_thw

At training time, the wrapper computes the same hash and replaces the
3.8s preprocess_qwen_2_visual call with a <1ms torch.load (or dict lookup).

Usage (32 workers, ~1 hour for all 10 datasets):
    python scripts/cache_qwen_tokenization.py \
        --output /apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/qwen_tok_cache \
        --workers 32

Resumable: skips already-cached files.
"""
from __future__ import annotations
import argparse, hashlib, logging, os, pickle, sys, time, types
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cache_qwen_tok")

# ── torchcodec stub ──────────────────────────────────────────────────────────
def _stub_torchcodec():
    pkg = types.ModuleType("torchcodec")
    dm  = types.ModuleType("torchcodec.decoders")
    class _S:
        def __init__(self, *a, **kw): raise RuntimeError("torchcodec stubbed")
    dm.VideoDecoder = _S; pkg.decoders = dm
    sys.modules.setdefault("torchcodec", pkg)
    sys.modules.setdefault("torchcodec.decoders", dm)

_stub_torchcodec()

_INTERNNAV_ROOT = os.environ.get(
    "INTERNNAV_ROOT",
    "/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav",
)
sys.path.insert(0, _INTERNNAV_ROOT)

QWEN_MODEL    = "/tmp/Qwen3-VL-4B-Instruct"
ANNO_CACHE_DIR = (
    "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/"
    "data/internnav_annotations_cache"
)
RESIZE_H = 384
RESIZE_W = 384

DATASETS = [
    ("r2r_125cm_0_30",      125, 0,  30, 4, 8),
    ("r2r_125cm_0_45",      125, 0,  45, 4, 8),
    ("r2r_60cm_15_15",       60, 15, 15, 4, 8),
    ("r2r_60cm_30_30",       60, 30, 30, 4, 8),
    ("rxr_125cm_0_30",      125, 0,  30, 4, 8),
    ("rxr_125cm_0_45",      125, 0,  45, 4, 8),
    ("rxr_60cm_15_15",       60, 15, 15, 4, 8),
    ("rxr_60cm_30_30",       60, 30, 30, 4, 8),
    ("scalevln_125cm_0_30", 125, 0,  30, 4, 8),
    ("scalevln_60cm_30_30",  60, 30, 30, 4, 8),
]

IDX2ACTION = {0:"STOP", 1:"↑", 2:"←", 3:"→", 5:"↓"}
# 注意：与训练数据 InternNav idx2actions 完全一致:
#   {0:'STOP', 1:"↑", 2:"←", 3:"→", 5:"↓"}
# pixel_goal 样本: 第一个 gpt 输出是 "↓"（idx=5），然后第二个 gpt 输出是坐标


def cache_key(instruction: str, action_str: str, n_images: int, has_pose: bool = False) -> str:
    """Stable 16-char hex key — same logic used in the training wrapper."""
    # has_pose 加入 key，区分 pixel_goal(2turn) vs turn/stop(1turn) 样本
    raw = f"{instruction}|{action_str}|{n_images}|{'pose' if has_pose else 'npose'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── worker global state ──────────────────────────────────────────────────────
_W_tok  = None
_W_pqv  = None
_W_conj = None


def _init_worker():
    global _W_tok, _W_pqv, _W_conj
    import copy, transformers
    _stub_torchcodec()
    from transformers import AutoTokenizer
    import internnav.dataset.internvla_n1_lerobot_dataset as _il

    _W_tok  = AutoTokenizer.from_pretrained(QWEN_MODEL)
    _W_pqv  = _il.preprocess_qwen_2_visual
    _W_conj = getattr(_il, "CONJUNCTIONS", None) or ["Your current observation:"]

    # Patch A: apply_chat_template must return list[int], not BatchEncoding
    # (transformers >=5.0 changed the default; InternNav expects list[int])
    _orig_apply = _W_tok.apply_chat_template
    def _apply_compat(*args, **kwargs):
        kwargs.setdefault("return_dict", False)
        out = _orig_apply(*args, **kwargs)
        if hasattr(out, "get") and "input_ids" in out:
            out = out["input_ids"]
        if hasattr(out, "tolist"):
            out = out.tolist()
        if isinstance(out, list) and out and isinstance(out[0], list):
            out = out[0]
        return out
    _W_tok.apply_chat_template = _apply_compat

    # Patch B: skip deepcopy(tokenizer) — saves 0.75s per call
    _orig = copy.deepcopy
    def _fast_dc(obj, memo=None):
        if isinstance(obj, transformers.PreTrainedTokenizerBase):
            return obj
        return _orig(obj, memo)
    copy.deepcopy = _fast_dc
    log.debug("worker ready")


def _build_sources(instruction: str, action_str: str, n_images: int,
                   has_history: bool, has_pose: bool = False,
                   coord_str: str = "320 240") -> list:
    """Replicate InternNav's chat_sources (text portion only).

    has_pose=True  → pixel_goal 样本:
        n_images = n_hist + 1(front) + 1(lookdown) = n_hist+2
        conversation 结构:
          [human]: prompt + hist_imgs + conj + <image(front)>
          [gpt]:   ↓
          [human]: conj + <image(lookdown)>
          [gpt]:   x y  (coord_str 占位, 只用于 tokenize 结构)
    has_pose=False → turn/stop 样本:
        n_images = n_hist + 1(front)
        conversation 结构:
          [human]: prompt + hist_imgs + conj + <image(front)>
          [gpt]:   action_str (← → STOP)
    """
    import random
    conj = random.choice(_W_conj)
    conj2 = random.choice(_W_conj)

    if has_pose:
        # pixel_goal 样本: n_hist = n_images - 2 (front + lookdown)
        n_hist = n_images - 2
    else:
        # turn/stop 样本: n_hist = n_images - 1 (front only)
        n_hist = n_images - 1 - (1 if has_history else 0)

    hist_str = "<image>" * max(0, n_hist)

    if has_history and n_hist > 0:
        user = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint's coordinates in the image. "
            "Please output STOP when you have successfully completed the task. "
            f"These are your historical observations: {hist_str}. "
            f"{conj}<image>."
        ).replace("<instruction>", instruction)
    else:
        user = (
            "You are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint's coordinates in the image. "
            "Please output STOP when you have successfully completed the task. "
            f"{conj}<image>."
        ).replace("<instruction>", instruction)

    if has_pose:
        # 2-turn: ↓ 然后 lookdown image 然后坐标
        return [[
            {"from": "human", "value": user},
            {"from": "gpt",   "value": "↓"},
            {"from": "human", "value": f"{conj2}<image>."},
            {"from": "gpt",   "value": coord_str},
        ]]
    else:
        return [[
            {"from": "human", "value": user},
            {"from": "gpt",   "value": action_str},
        ]]


def _compute_one(args):
    """Compute and save one unique combination."""
    import torch
    instruction, action_str, n_images, has_history, has_pose, out_path = args

    # coord_str 占位符（坐标随机，但 token 结构固定）
    coord_str = "320 240"

    sources    = _build_sources(instruction, action_str, n_images,
                                has_history, has_pose=has_pose, coord_str=coord_str)
    patch_h    = RESIZE_H // 28
    patch_w    = RESIZE_W // 28
    merge_size = 2
    grid_thw   = [torch.tensor([1, patch_h, patch_w])] * n_images
    grid_merged = [int(g.prod().item() // (merge_size ** 2)) for g in grid_thw]

    result = _W_pqv(sources, _W_tok, grid_thw_image=grid_merged)

    torch.save({
        "input_ids": result["input_ids"],
        "labels":    result.get("labels", result.get("targets")),
        "n_images":  n_images,
        "grid_thw":  torch.stack(grid_thw),
    }, out_path)
    return True


def _collect_unique(out_dir: str) -> list:
    """Enumerate all unique (instruction, action, n_images) combos across datasets."""
    import numpy as np
    unique: dict[str, tuple] = {}

    for ds_key, height, pitch_1, pitch_2, sample_step, num_history in DATASETS:
        anno_path = os.path.join(ANNO_CACHE_DIR, f"{ds_key}.pkl")
        if not os.path.exists(anno_path):
            log.warning(f"Missing annotation: {anno_path}")
            continue
        with open(anno_path, "rb") as f:
            data = pickle.load(f)
        eps = data["episodes"] if isinstance(data, dict) else data
        log.info(f"{ds_key}: {len(eps)} episodes")

        for ep in eps:
            instruction = ep["instructions"]
            actions     = ep["actions"][1:] + [0]
            pixel_goals = ep["pixel_goals"]
            n = len(actions)

            for step in range(n // sample_step + 1):
                sfid = step * sample_step
                if sfid >= n or sfid == n - 1:
                    continue
                action     = actions[sfid]
                pixel_goal = pixel_goals[sfid] if sfid < len(pixel_goals) else [-1, -1]
                has_pose   = (pixel_goal[0] != -1)

                # Use raw linspace WITHOUT np.unique — matches InternNav's actual behavior.
                # InternNav does NOT deduplicate history frame ids; it passes the raw
                # linspace (which can have repeats for small sfid) to process_image_unified,
                # so the actual len(grid_thw_image) = num_history (always 8), not len(unique).
                if sfid > 0:
                    hist = list(np.linspace(0, sfid - 1, num_history, dtype=np.int32))
                else:
                    hist = []
                n_imgs = len(hist) + 1 + (1 if has_pose else 0)

                if isinstance(action, (list, tuple)):
                    act_str = str(action)
                else:
                    act_str = IDX2ACTION.get(action, str(action))

                ck = cache_key(instruction, act_str, n_imgs, has_pose=has_pose)
                if ck in unique:
                    continue
                out_path = os.path.join(out_dir, f"{ck}.pt")
                if os.path.exists(out_path):
                    continue
                unique[ck] = (instruction, act_str, n_imgs, len(hist) > 0, has_pose, out_path)

    log.info(f"Unique combinations to compute: {len(unique):,}")
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=(
        "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/qwen_tok_cache"
    ))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="subset of dataset keys, e.g. r2r_60cm_30_30")
    args = parser.parse_args()

    # Filter datasets if --datasets specified
    global DATASETS
    if args.datasets:
        DATASETS = [d for d in DATASETS if d[0] in args.datasets]
        log.info(f"Filtering to datasets: {[d[0] for d in DATASETS]}")

    os.makedirs(args.output, exist_ok=True)
    log.info("Scanning annotations for unique combinations...")
    tasks = _collect_unique(args.output)
    if not tasks:
        log.info("All combinations already cached!")
        return

    log.info(f"Will compute {len(tasks):,} combinations with {args.workers} workers")
    log.info(f"Estimated: {len(tasks)*3/args.workers/3600:.1f}h "
             f"(~3s/sample after deepcopy patch, {args.workers} workers)")

    t0 = time.perf_counter()
    done = errors = 0
    BATCH = args.workers * 4  # submit in small batches to avoid queue deadlock

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for batch_start in range(0, len(tasks), BATCH):
            batch = tasks[batch_start: batch_start + BATCH]
            futs = {pool.submit(_compute_one, t): t for t in batch}
            for fut in as_completed(futs):
                try:
                    fut.result()
                    done += 1
                    if done % 1000 == 0:
                        elapsed = time.perf_counter() - t0
                        rate    = done / elapsed
                        eta     = (len(tasks) - done) / rate / 60 if rate > 0 else 0
                        log.info(f"{done:,}/{len(tasks):,}  {rate:.1f}/s  ETA {eta:.0f}min")
                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        t = futs[fut]
                        log.warning(f"Error [{t[1]}, n={t[2]}]: {e}")

    elapsed = time.perf_counter() - t0
    log.info(f"Done {done:,}/{len(tasks):,} in {elapsed/60:.1f}min  errors={errors}")
    log.info(f"Output: {args.output}  ({done} .pt files)")


if __name__ == "__main__":
    main()
