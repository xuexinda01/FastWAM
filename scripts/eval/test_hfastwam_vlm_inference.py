"""Standalone inference test for HFastWAM-VLM (Stage 1).

Purpose: verify that a Stage 1 checkpoint loads + forwards correctly.
Reports per-loss-component values and (optionally) decodes the
language-head logits on the supervised positions to show what the VLM
is actually predicting on a real validation sample.

Mirrors the model construction in
``scripts/train/qwenvl_train/hfastwam_internnav_trainer.py`` so it loads
the SAME architecture the checkpoint was trained with.

Why this exists (vs. a full VLN harness like
``/apdcephfs_tj5/share_302528826/xxd/fastwam_vln_eval``):
  - Stage 1 has no trained action expert → no real trajectory output yet.
  - The VLN Habitat harness needs ``[forward, left, theta, moving_flag]``
    waypoints which only exist after Stage 2 trains the action DiT.
  - This script focuses on what Stage 1 can answer: "is the language CE
    going down?" and "is the video FM denoising sensible?"

Usage:
    bash scripts/eval/run_hfastwam_vlm_eval.sh /path/to/hfastwam_vlm_final.pt

    # With per-sample dump:
    DUMP_DIR=/tmp/hfastwam_eval_dump \\
        bash scripts/eval/run_hfastwam_vlm_eval.sh /path/to/ckpt.pt
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision.transforms import v2

# Bootstrap sys.path the same way the trainer does.
_FASTWAM_ROOT = os.environ.get(
    "FASTWAM_ROOT", "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM"
)
sys.path.insert(0, os.path.join(_FASTWAM_ROOT, "src"))

_INTERNNAV_ROOT = os.environ.get(
    "INTERNNAV_ROOT",
    "/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav",
)
sys.path.insert(0, _INTERNNAV_ROOT)

# Same compat shims as the trainer — see hfastwam_internnav_trainer.py for context.
import transformers.trainer as _tr  # noqa: E402

if not hasattr(_tr, "ALL_LAYERNORM_LAYERS"):
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS as _ALL_LN
        _tr.ALL_LAYERNORM_LAYERS = _ALL_LN
    except Exception:
        _tr.ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


def _install_torchcodec_stub():
    if "torchcodec" in sys.modules:
        return
    import types

    class _StubVideoDecoder:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("torchcodec stubbed (FFmpeg incompat)")

    pkg = types.ModuleType("torchcodec")
    decoders_mod = types.ModuleType("torchcodec.decoders")
    decoders_mod.VideoDecoder = _StubVideoDecoder
    pkg.decoders = decoders_mod
    sys.modules["torchcodec"] = pkg
    sys.modules["torchcodec.decoders"] = decoders_mod


_install_torchcodec_stub()

# Now safe to import InternNav + FastWAM.
import transformers  # noqa: E402

from internnav.trainer.internvla_n1_argument import (  # noqa: E402
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from internnav.dataset.internvla_n1_lerobot_dataset import (  # noqa: E402
    DataCollatorForSupervisedDataset,
    FlattenedDataCollatorForSupervisedDataset,
)

from fastwam.datasets.lerobot.internvla_n1_hfastwam_dataset import (  # noqa: E402
    InternVLAN1HFastWAMDataset,
)
from fastwam.models.hfastwam_v2.hfastwam_vlm import HFastWAMVLM  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Model loader
# --------------------------------------------------------------------- #
def build_hfastwam_vlm(args, device: str = "cuda:0") -> HFastWAMVLM:
    """Mirror of ``hfastwam_internnav_trainer.train`` model build path.

    Picks the same Wan / Action / Qwen3-VL configs the trainer uses so
    the checkpoint's state_dict keys line up.
    """
    torch_dtype = torch.bfloat16

    model = HFastWAMVLM.from_pretrained_qwen3vl(
        device=device,
        torch_dtype=torch_dtype,
        model_id=args.wan_model_id,
        tokenizer_model_id=args.wan_tokenizer_model_id,
        tokenizer_max_len=256,
        load_text_encoder=False,
        skip_dit_load_from_pretrain=args.skip_dit_load_from_pretrain,
        skip_video_dit_load_from_pretrain=args.skip_dit_load_from_pretrain,
        action_dit_pretrained_path=None,
        qwen3_vl_model_id=args.qwen3_vl_model_id,
        qwen3_vl_max_total_len=args.qwen3_vl_max_total_len,
        qwen3_vl_local_files_only=args.qwen3_vl_local_files_only,
        video_dit_config={
            "has_image_input": False,
            "patch_size": [1, 2, 2],
            "in_dim": 48, "out_dim": 48,
            "hidden_dim": 3072, "ffn_dim": 14336, "freq_dim": 256, "text_dim": 4096,
            "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
            "eps": 1.0e-06, "seperated_timestep": True,
            "require_clip_embedding": False, "require_vae_embedding": False,
            "fuse_vae_embedding_in_latents": True,
            "use_gradient_checkpointing": False,   # off for eval
            "video_attention_mask_mode": "first_frame_causal",
            "action_conditioned": False,
            "action_dim": 4,
            "action_group_causal_mask_mode": "group_diagonal",
        },
        action_dit_config={
            "action_dim": 4,
            "hidden_dim": 1024, "ffn_dim": 4096,
            "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
            "text_dim": 4096, "freq_dim": 256, "eps": 1.0e-06,
            "use_gradient_checkpointing": False,
        },
        loss_config={
            "lambda_language": 1.0,
            "lambda_video": 1.0,
            "lambda_action": 0.0,
        },
        training_phase="language_video",
        knowledge_insulation=True,
        strict_expert_compat=False,
        layer_alignment_mode="tail_overlap",
        shared_attention_expert="video",
        freeze_language_expert=False,
        freeze_video_expert=False,
        freeze_action_expert=True,
        fastwam_checkpoint=None,
        mot_checkpoint_mixed_attn=False,
    )

    # Load Stage 1 weights.
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[load] {args.checkpoint}")
        payload = model.load_checkpoint(args.checkpoint)
        ckpt_step = payload.get("step", "?") if isinstance(payload, dict) else "?"
        print(f"[load] checkpoint step = {ckpt_step}")
    else:
        print(f"[load] WARNING: no checkpoint at {args.checkpoint!r}; "
              "running on randomly initialized weights")

    model.eval()
    return model


# --------------------------------------------------------------------- #
# Dataset / collator (single sample)
# --------------------------------------------------------------------- #
def build_eval_dataset(args, processor, tokenizer):
    """Construct InternVLAN1HFastWAMDataset reusing the trainer's data_args."""
    data_args = DataArguments()
    data_args.vln_dataset_use = args.vln_dataset_use
    data_args.data_flatten = True   # required for FlattenedDataCollator
    data_args.num_history = 8
    data_args.num_future_steps = 4
    data_args.predict_step_num = 32
    data_args.sample_step = 4
    data_args.resize_h = 384
    data_args.resize_w = 384
    data_args.pixel_goal_only = False
    data_args.system1 = "none"
    data_args.data_augmentation = False  # eval = no aug
    data_args.transform_train = v2.Resize((data_args.resize_h, data_args.resize_w))
    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"
    data_args.fastwam_video_size = 224
    data_args.fastwam_n_history_frames = 9
    data_args.fastwam_n_future_frames = 8
    data_args.fastwam_predict_step_num = 8

    dataset = InternVLAN1HFastWAMDataset(tokenizer=tokenizer, data_args=data_args)
    print(f"[dataset] {len(dataset)} samples in vln_dataset_use={args.vln_dataset_use}")
    return dataset, data_args


def fastwam_eval_collator(features, base_collator):
    """Same shape as the trainer's ``fastwam_collator`` so the dict layout
    that ``training_loss`` reads is identical."""
    extras = {}
    for k in ("video", "action", "action_is_pad", "video_valid", "action_valid", "prompt"):
        if k in features[0]:
            vals = [f.pop(k) for f in features]
            if k == "prompt":
                extras[k] = vals
            else:
                extras[k] = torch.stack(vals, dim=0)
    batch = base_collator(features)
    batch.update(extras)
    return batch


# --------------------------------------------------------------------- #
# Sanity check / forward
# --------------------------------------------------------------------- #
@torch.no_grad()
def run_forward(model: HFastWAMVLM, batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Run model.training_loss on a single batch, return loss dict + auxiliary
    info (lang accuracy on supervised positions)."""
    # Move to device
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v

    loss, loss_dict = model.training_loss(moved)

    # Compute next-token accuracy on labelled positions (supervised tokens).
    # We re-run the language path to get logits — cheap because everything is
    # already cached in eager mode.
    info: Dict[str, Any] = {"loss_total": float(loss.item()), **loss_dict}

    # Pull labels and logits via VLMLanguageExpert.post_dit. Keeping this
    # block conservative: skip on any unexpected shape rather than crash the
    # whole eval.
    try:
        input_ids = moved["input_ids"].long()
        labels = moved["labels"].long()
        # Re-run language pre_dit for logits (no MoT needed; we've already
        # computed total_loss above which is the canonical signal — this is
        # just for diagnostic accuracy).
        lang_pre = model.language_expert.pre_dit(
            input_ids=input_ids, labels=labels,
            pixel_values=moved.get("pixel_values"),
            image_grid_thw=moved.get("image_grid_thw"),
            pixel_values_videos=moved.get("pixel_values_videos"),
            video_grid_thw=moved.get("video_grid_thw"),
        )
        # Skip MoT: just decode the un-fused language tokens through post_dit
        # to get logits. This ignores video grounding but is enough for a
        # rough next-token accuracy.
        lang_out = model.language_expert.post_dit(lang_pre["tokens"], lang_pre)
        logits = lang_out.logits  # [1, S, V]

        # Shift: labels[:, 1:] vs logits[:, :-1]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        if int(valid.sum().item()) > 0:
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels) & valid
            acc = float(correct.sum().item()) / float(valid.sum().item())
            info["lang_next_token_acc"] = acc
            info["lang_supervised_positions"] = int(valid.sum().item())
    except Exception as e:
        info["lang_acc_error"] = str(e)

    return info


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="",
                   help="Path to hfastwam_vlm_final.pt or step_*.pt")
    p.add_argument("--qwen3_vl_model_id", type=str,
                   default="/tmp/Qwen3-VL-4B-Instruct")
    p.add_argument("--qwen3_vl_local_files_only", type=lambda s: s.lower() == "true",
                   default=True)
    p.add_argument("--qwen3_vl_max_total_len", type=int, default=8192)
    p.add_argument("--wan_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B")
    p.add_argument("--wan_tokenizer_model_id", type=str,
                   default="Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--skip_dit_load_from_pretrain", type=lambda s: s.lower() == "true",
                   default=False)
    p.add_argument("--vln_dataset_use", type=str, default="r2r_125cm_0_30",
                   help="Comma-separated list, same convention as trainer")
    p.add_argument("--num_samples", type=int, default=4,
                   help="How many validation samples to forward")
    p.add_argument("--sample_indices", type=str, default="",
                   help="Optional comma-separated sample idx list (overrides "
                        "num_samples)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dump_dir", type=str, default="",
                   help="If set, save per-sample loss + decoded text to this dir")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("HFastWAM-VLM Stage 1 Inference Test")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 60)

    # Tokenizer + processor (same as trainer).
    print("[init] loading Qwen3-VL tokenizer + processor ...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.qwen3_vl_model_id,
        trust_remote_code=True,
        local_files_only=args.qwen3_vl_local_files_only,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.qwen3_vl_model_id,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        local_files_only=args.qwen3_vl_local_files_only,
    )

    # Dataset.
    dataset, _data_args = build_eval_dataset(args, processor, tokenizer)
    base_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Model.
    print("[init] building HFastWAM-VLM ...")
    model = build_hfastwam_vlm(args, device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e9:.2f}B parameters")

    # Sample selection.
    if args.sample_indices:
        idxs = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    else:
        idxs = list(range(min(args.num_samples, len(dataset))))
    print(f"[eval] running on {len(idxs)} samples: {idxs[:10]}{'...' if len(idxs) > 10 else ''}")

    # Forward each sample.
    rows = []
    for i, idx in enumerate(idxs):
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"  [skip] idx={idx} dataset error: {e}")
            continue
        if not isinstance(sample, dict):
            print(f"  [skip] idx={idx} non-dict sample type={type(sample).__name__}")
            continue
        batch = fastwam_eval_collator([sample], base_collator)
        info = run_forward(model, batch, args.device)
        info["idx"] = idx
        rows.append(info)
        line = (
            f"  [{i+1:>3d}/{len(idxs)}] idx={idx:>7d} "
            f"loss={info['loss_total']:>7.3f} "
            f"L_lang={info.get('loss_language', 0.0):>6.3f} "
            f"L_video={info.get('loss_video', 0.0):>6.3f} "
            f"acc={info.get('lang_next_token_acc', float('nan')):>5.3f} "
            f"(supervised tokens={info.get('lang_supervised_positions', 0)})"
        )
        print(line)

    # Aggregate.
    if rows:
        import statistics
        avg_loss = statistics.mean(r["loss_total"] for r in rows)
        avg_lang = statistics.mean(r.get("loss_language", 0.0) for r in rows)
        avg_video = statistics.mean(r.get("loss_video", 0.0) for r in rows)
        accs = [r["lang_next_token_acc"] for r in rows if "lang_next_token_acc" in r]
        avg_acc = statistics.mean(accs) if accs else float("nan")
        print("-" * 60)
        print(f"[summary] over {len(rows)} samples:")
        print(f"  loss_total:    {avg_loss:.4f}")
        print(f"  loss_language: {avg_lang:.4f}")
        print(f"  loss_video:    {avg_video:.4f}")
        print(f"  lang_acc:      {avg_acc:.4f}")

    # Optional dump.
    if args.dump_dir and rows:
        import json
        os.makedirs(args.dump_dir, exist_ok=True)
        out = os.path.join(args.dump_dir, "eval_per_sample.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[dump] per-sample metrics → {out}")

    print("[done]")


if __name__ == "__main__":
    main()
