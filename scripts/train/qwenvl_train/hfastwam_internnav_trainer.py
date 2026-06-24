"""HF Trainer entrypoint for HFastWAM-VLM dual-stage InternNav training.

Mirrors ``internnav/trainer/internvla_n1_trainer.py`` but instantiates
:class:`fastwam.models.hfastwam_v2.hfastwam_vlm.HFastWAMVLM` (full Qwen3-VL
language branch + Wan2.2 video DiT + ActionDiT, 3-expert MoT) instead of
``InternVLAN1ForCausalLM``.

Data: the InternNav lerobot dataset (ChatML) wrapped by
``InternVLAN1HFastWAMDataset``, which additionally emits the 17-frame raw RGB
video tensor (9 history + 8 future) and an 8-step action chunk.

Stage selection (1 = train VLM+video, 2 = train action) is driven by
``--stage`` through :func:`configure_stage`.

Single-node smoke::

    python scripts/train/qwenvl_train/hfastwam_internnav_trainer.py \\
        --qwen3_vl_model_id /tmp/Qwen3-VL-4B-Instruct \\
        --vln_dataset_use r2r_125cm_0_30 \\
        --output_dir /tmp/hfastwam_vlm_smoke --stage 1 ...

Multi-node::

    bash scripts/train/qwenvl_train/train_hfastwam_v2_stage1.sh
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import transformers
from torchvision.transforms import v2

# Ensure FastWAM src is importable when launched from FastWAM/ root.
_FASTWAM_ROOT = os.environ.get("FASTWAM_ROOT", "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM")
sys.path.insert(0, os.path.join(_FASTWAM_ROOT, "src"))

# InternNav must be on sys.path so we can reuse its argparse + collator.
_INTERNNAV_ROOT = os.environ.get(
    "INTERNNAV_ROOT",
    "/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav",
)
sys.path.insert(0, _INTERNNAV_ROOT)

# ---- transformers 5.x compat shims (must run BEFORE any internnav.* import) ---- #
import transformers.trainer as _tr  # noqa: E402

if not hasattr(_tr, "ALL_LAYERNORM_LAYERS"):
    try:
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS as _ALL_LN
        _tr.ALL_LAYERNORM_LAYERS = _ALL_LN
    except Exception:
        _tr.ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


# ---- torchcodec stub (must run BEFORE InternNav lerobot dataset import) ---- #
def _install_torchcodec_stub():
    if "torchcodec" in sys.modules:
        return
    import types

    class _StubVideoDecoder:  # pragma: no cover - never instantiated
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(
                "torchcodec.decoders.VideoDecoder is stubbed out by FastWAM's "
                "trainer (env FFmpeg 8.0 vs torchcodec wheel). The dataset only "
                "reads jpg frames."
            )

    pkg = types.ModuleType("torchcodec")
    decoders_mod = types.ModuleType("torchcodec.decoders")
    decoders_mod.VideoDecoder = _StubVideoDecoder
    pkg.decoders = decoders_mod
    sys.modules["torchcodec"] = pkg
    sys.modules["torchcodec.decoders"] = decoders_mod


_install_torchcodec_stub()

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
from fastwam.models.hfastwam_v2 import HFastWAMVLM  # noqa: E402

# Open-loop eval callback（100步一次，rank-0 only）
_TRAINER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TRAINER_DIR)
from hfastwam_openloop_eval import HFastWAMOpenLoopEvalCallback  # noqa: E402

logger = logging.getLogger(__name__)
local_rank: int = 0


# ----------------------------------------------------------------------- #
# Extra argument fields for HFastWAM-VLM
# ----------------------------------------------------------------------- #
@dataclass
class HFastWAMArguments:
    """Extra CLI flags parsed as a separate dataclass so InternNav's schema
    stays untouched."""

    stage: int = field(default=1, metadata={"help": "1 = train VLM+video, 2 = train action only"})
    qwen3_vl_model_id: str = field(default="Qwen/Qwen3-VL-4B-Instruct")
    qwen3_vl_local_files_only: bool = field(default=False)
    qwen3_vl_max_total_len: int = field(default=8192)
    wan_model_id: str = field(default="Wan-AI/Wan2.2-TI2V-5B")
    wan_tokenizer_model_id: str = field(default="Wan-AI/Wan2.1-T2V-1.3B")
    wan_skip_dit_load_from_pretrain: bool = field(default=False)
    action_dit_pretrained_path: str = field(default="")
    fastwam_pretrain_checkpoint: str = field(default="")
    fastwam_video_size: int = field(default=224)
    fastwam_n_history_frames: int = field(default=9)
    fastwam_n_future_frames: int = field(default=8)
    fastwam_predict_step_num: int = field(default=8)
    lambda_language: float = field(default=1.0)
    lambda_video: float = field(default=1.0)
    lambda_action: float = field(default=0.0)
    mot_checkpoint_mixed_attn: bool = field(default=True)
    # Number of leading latent frames treated as the clean condition (the 9
    # history RGB frames collapse to ~3 latent frames under the Wan VAE temporal
    # downsampling). Those latent frames get timestep=0 in the video expert.
    n_cond_latent_frames: int = field(default=3)
    local_smoke_adapter_only: bool = field( default=False, metadata={ "help": ( "Local smoke test only: freeze Qwen/Wan/Action and train only " "non-action MoT projection adapters." ) }, )

# ----------------------------------------------------------------------- #
# HF Trainer ↔ HFastWAM-VLM glue
# ----------------------------------------------------------------------- #
class HFastWAMVLMHFWrapper(nn.Module):
    """Wraps HFastWAMVLM so HF Trainer can call ``model(**batch)``.

    We declare the ChatML/fastwam batch keys as explicit named parameters so
    HF Trainer's ``remove_unused_columns`` signature inspection keeps them
    (the launch script also passes ``--remove_unused_columns False`` as a
    belt-and-suspenders guard).
    """
    accepts_loss_kwargs = False
    
    def __init__(self, model: HFastWAMVLM):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids=None,
        labels=None,
        position_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        video=None,
        action=None,
        action_is_pad=None,
        video_valid=None,
        action_valid=None,
        prompt=None,
        **kwargs,
    ):
        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "video": video,
            "action": action,
            "action_is_pad": action_is_pad,
            "video_valid": video_valid,
            "action_valid": action_valid,
        }
        batch = {k: v for k, v in batch.items() if v is not None}
        loss, metrics = self.model.training_loss(batch)
        _log_raw_loss(loss, metrics)

        from transformers.modeling_outputs import ModelOutput

        @dataclass
        class _Out(ModelOutput):
            loss: Optional[torch.Tensor] = None
            metrics: Optional[Dict[str, float]] = None

        return _Out(loss=loss, metrics=metrics)

    @property
    def config(self):
        from transformers import PretrainedConfig
        if not hasattr(self, "_dummy_config"):
            cfg = PretrainedConfig()
            cfg.use_cache = False
            cfg.model_type = "hfastwam_vlm"
            cfg.architectures = ["HFastWAMVLM"]
            self._dummy_config = cfg
        return self._dummy_config

    # Gradient checkpointing is configured at construction inside each expert;
    # HF Trainer's toggles are no-ops here.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self

    def gradient_checkpointing_disable(self):
        return self

    @property
    def is_gradient_checkpointing(self):
        return True

    def enable_input_require_grads(self):
        if hasattr(self.model, "enable_input_require_grads"):
            try:
                self.model.enable_input_require_grads()
            except Exception:
                pass
        return self

    _keys_to_ignore_on_save = None
    model_tags = None

    def tie_weights(self):
        return None

    def get_base_model(self):
        return self


# ----------------------------------------------------------------------- #
# Per-step timing callback (rank-0 only)
# ----------------------------------------------------------------------- #
class StepTimingCallback(transformers.TrainerCallback):
    def __init__(self):
        import time as _time
        self._time = _time
        self._t_step_begin = 0.0
        self._t_substep_last = 0.0
        self._substep_times: list = []
        self._t_pre_opt = 0.0
        self._alloc_retries_begin = 0

    def _is_rank0(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def on_step_begin(self, args, state, control, **kwargs):
        if not self._is_rank0():
            return
        t = self._time.perf_counter()
        self._t_step_begin = t
        self._t_substep_last = t
        self._substep_times = []
        self._t_pre_opt = t
        try:
            self._alloc_retries_begin = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        except Exception:
            self._alloc_retries_begin = 0

    def on_substep_end(self, args, state, control, **kwargs):
        if not self._is_rank0():
            return
        t = self._time.perf_counter()
        self._substep_times.append(t - self._t_substep_last)
        self._t_substep_last = t

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self._is_rank0():
            return
        self._t_pre_opt = self._time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if not self._is_rank0():
            return
        t_end = self._time.perf_counter()
        t_total = t_end - self._t_step_begin
        t_opt = t_end - self._t_pre_opt
        bwd_str = ", ".join(f"{x:.1f}s" for x in self._substep_times)
        try:
            now = torch.cuda.memory_stats().get("num_alloc_retries", 0)
            delta = now - self._alloc_retries_begin
        except Exception:
            delta = -1

        # Compute correct ETA based on actual step time (not tqdm which resets on resume)
        step = state.global_step
        max_steps = state.max_steps if state.max_steps else 0
        remaining = max(0, max_steps - step)
        eta_h = remaining * t_total / 3600 if t_total > 0 else 0
        logger.warning(
            "[STEP_TIMING step=%d/%d] total=%.1fs | bwd=[%s] | opt=%.1fs | alloc_retries=%d | ETA≈%.1fh",
            step, max_steps, t_total, bwd_str, t_opt, delta, eta_h,
        )


# ----------------------------------------------------------------------- #
# Periodic step_*.pt checkpoint callback (eval-friendly)
# ----------------------------------------------------------------------- #
class HFastWAMCheckpointCallback(transformers.TrainerCallback):
    def __init__(self, hfastwam: HFastWAMVLM, output_dir: str, save_total_limit: int = 5):
        self.hfastwam = hfastwam
        self.output_dir = output_dir
        self.save_total_limit = max(1, save_total_limit)

    def on_save(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        step = state.global_step
        ckpt_path = os.path.join(self.output_dir, f"step_{step}.pt")
        logger.warning("[HFastWAMCheckpointCallback] saving step_%d.pt → %s", step, ckpt_path)
        try:
            self.hfastwam.save_checkpoint(ckpt_path, step=step)
        except Exception as exc:
            logger.error("[HFastWAMCheckpointCallback] save FAILED at step %d: %s", step, exc)
            return
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        import glob
        import re

        pattern = os.path.join(self.output_dir, "step_*.pt")
        files = sorted(
            glob.glob(pattern),
            key=lambda p: int(re.search(r"step_(\d+)\.pt$", os.path.basename(p)).group(1)),
        )
        while len(files) > self.save_total_limit:
            old = files.pop(0)
            try:
                os.remove(old)
                logger.warning("[HFastWAMCheckpointCallback] removed old step ckpt: %s", old)
            except OSError as exc:
                logger.warning("[HFastWAMCheckpointCallback] could not remove %s: %s", old, exc)


# ----------------------------------------------------------------------- #
# Per-stage freeze policy
# ----------------------------------------------------------------------- #
def configure_stage(stage: int, hfastwam: HFastWAMVLM, args: HFastWAMArguments) -> None:
    if stage == 1:
        hfastwam.training_phase = "language_video"
        hfastwam.freeze_language_expert = False
        hfastwam.freeze_action_expert = True
        hfastwam.loss_lambda_language = float(args.lambda_language)
        hfastwam.loss_lambda_video = float(args.lambda_video)
        hfastwam.loss_lambda_action = 0.0
        hfastwam.language_expert.requires_grad_(True)
        hfastwam.action_expert.requires_grad_(False)
        # language_only_mode: lambda_video=0 → video forward 완전 비활성화
        if float(args.lambda_video) == 0.0:
            hfastwam.language_only_mode = True
            hfastwam.freeze_video_expert = True
            hfastwam.video_expert.requires_grad_(False)
            logger.warning("[configure_stage] language_only_mode=True (lambda_video=0)")
        else:
            hfastwam.language_only_mode = False
            hfastwam.freeze_video_expert = False
            hfastwam.video_expert.requires_grad_(True)
    elif stage == 2:
        hfastwam.training_phase = "full"
        hfastwam.freeze_language_expert = True
        hfastwam.freeze_video_expert = True
        hfastwam.freeze_action_expert = False
        hfastwam.loss_lambda_language = 0.0
        hfastwam.loss_lambda_video = 0.0
        hfastwam.loss_lambda_action = float(args.lambda_action) if args.lambda_action > 0 else 1.0
        hfastwam.language_expert.requires_grad_(False)
        hfastwam.video_expert.requires_grad_(False)
        hfastwam.action_expert.requires_grad_(True)
    else:
        raise ValueError(f"Unknown stage={stage}; must be 1 or 2.")

    # MoT projection adapters (q/k/v/o between heterogeneous experts) must train
    # in whichever stage their adjacent experts train. Stage 1: train all the
    # language/video adapters; stage 2: train action adapters.
    for n, p in hfastwam.mot.named_parameters():
        if "_proj_to_shared" in n or "_proj_from_shared" in n:
            if stage == 1:
                p.requires_grad = ("action__" not in n)
            else:
                p.requires_grad = ("action__" in n)

    if local_rank == 0:
        n_train = sum(p.numel() for p in hfastwam.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in hfastwam.parameters())
        logger.warning(
            "[Stage %d] trainable=%.2fM / total=%.2fM (%.1f%%)",
            stage, n_train / 1e6, n_total / 1e6, 100.0 * n_train / max(n_total, 1),
        )


# ----------------------------------------------------------------------- #
# Wan / Action DiT configs (must match eval script for ckpt key parity)
# ----------------------------------------------------------------------- #
def _video_dit_config(grad_ckpt: bool) -> Dict[str, Any]:
    return {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48, "out_dim": 48,
        "hidden_dim": 3072, "ffn_dim": 14336, "freq_dim": 256, "text_dim": 4096,
        "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
        "eps": 1.0e-06, "seperated_timestep": True,
        "require_clip_embedding": False, "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": bool(grad_ckpt),
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": 4,
        "action_group_causal_mask_mode": "group_diagonal",
    }


def _action_dit_config(grad_ckpt: bool) -> Dict[str, Any]:
    return {
        "action_dim": 4,
        "hidden_dim": 1024, "ffn_dim": 4096,
        "num_heads": 24, "attn_head_dim": 128, "num_layers": 30,
        "text_dim": 4096, "freq_dim": 256, "eps": 1.0e-06,
        "use_gradient_checkpointing": bool(grad_ckpt),
    }


# ----------------------------------------------------------------------- #
# Runtime correctness guards
# ----------------------------------------------------------------------- #
def _restore_complex_language_rope(
    hfastwam: HFastWAMVLM,
    *,
    max_total_len: int,
    device: str,
) -> None:
    """Rebuild the language RoPE cache after the model-wide BF16 cast.

    ``precompute_freqs_cis`` returns a complex tensor. A recursive
    ``module.to(dtype=torch.bfloat16)`` also casts buffers and therefore drops
    the imaginary component. Rebuilding here restores the intended complex
    representation. Device is changed, dtype is deliberately not changed.
    """
    from fastwam.models.wan22_v2.wan_video_dit import precompute_freqs_cis

    language_expert = hfastwam.language_expert
    head_dim = int(language_expert.attn_head_dim)
    rope_len = max(int(max_total_len), 1024)

    freqs = precompute_freqs_cis(head_dim, end=rope_len).to(device=device)
    if not torch.is_complex(freqs):
        raise RuntimeError(
            "precompute_freqs_cis must return a complex tensor, "
            f"but got dtype={freqs.dtype}"
        )

    # Keep it registered as a non-persistent buffer so state_dict behaviour is
    # unchanged. Replacing the existing entry avoids another dtype cast here.
    language_expert._buffers["freqs"] = freqs

    if local_rank == 0:
        logger.warning(
            "[ROPE_FIXED] dtype=%s is_complex=%s shape=%s",
            freqs.dtype,
            torch.is_complex(freqs),
            tuple(freqs.shape),
        )


def _log_raw_loss(loss: torch.Tensor, metrics: Dict[str, Any]) -> None:
    """Print the unscaled model loss on rank 0 for early-step diagnosis."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    # Avoid flooding long runs. The counter lives on the helper function.
    count = getattr(_log_raw_loss, "_count", 0)
    if count < 8:
        logger.warning(
            "[RAW_MODEL_LOSS] loss=%.6f metrics=%s",
            float(loss.detach().float().item()),
            metrics,
        )
    _log_raw_loss._count = count + 1


# ----------------------------------------------------------------------- #
# Entry
# ----------------------------------------------------------------------- #
def train() -> None:
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, HFastWAMArguments)
    )
    model_args, data_args, training_args, hf_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if local_rank is None or int(local_rank) < 0:
        env_lr = os.environ.get("LOCAL_RANK")
        local_rank = int(env_lr) if (env_lr and env_lr.isdigit()) else 0
        training_args.local_rank = local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ---- Data augmentation pipeline (same as InternNav) ---- #
    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=0.2, saturation=0.2),
            v2.RandomPosterize(bits=4),
            v2.RandomAdjustSharpness(sharpness_factor=1.5),
            v2.RandomAutocontrast(),
            v2.ToPILImage(),
            v2.Resize((data_args.resize_h, data_args.resize_w)),
        ])
    else:
        data_args.transform_train = v2.Resize((data_args.resize_h, data_args.resize_w))

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        hf_args.qwen3_vl_model_id,
        trust_remote_code=True,
        local_files_only=hf_args.qwen3_vl_local_files_only,
    )
    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"  # Qwen3-VL shares the chat-template family

    data_args.fastwam_video_size = hf_args.fastwam_video_size
    data_args.fastwam_n_history_frames = hf_args.fastwam_n_history_frames
    data_args.fastwam_n_future_frames = hf_args.fastwam_n_future_frames
    data_args.fastwam_predict_step_num = hf_args.fastwam_predict_step_num

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_args.qwen3_vl_model_id,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        local_files_only=hf_args.qwen3_vl_local_files_only,
    )

    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    grad_ckpt = bool(training_args.gradient_checkpointing)

    if local_rank == 0:
        logger.warning("Building HFastWAM-VLM (stage=%d) on %s", hf_args.stage, model_device)

    hfastwam = HFastWAMVLM.from_pretrained_qwen3vl(
        device=model_device,
        torch_dtype=torch_dtype,
        model_id=hf_args.wan_model_id,
        tokenizer_model_id=hf_args.wan_tokenizer_model_id,
        tokenizer_max_len=256,
        load_text_encoder=False,
        skip_dit_load_from_pretrain=hf_args.wan_skip_dit_load_from_pretrain,
        skip_video_dit_load_from_pretrain=hf_args.wan_skip_dit_load_from_pretrain,
        action_dit_pretrained_path=(hf_args.action_dit_pretrained_path or None),
        qwen3_vl_model_id=hf_args.qwen3_vl_model_id,
        qwen3_vl_max_total_len=hf_args.qwen3_vl_max_total_len,
        qwen3_vl_local_files_only=hf_args.qwen3_vl_local_files_only,
        video_dit_config=_video_dit_config(grad_ckpt),
        action_dit_config=_action_dit_config(grad_ckpt),
        loss_config={
            "lambda_language": hf_args.lambda_language,
            "lambda_video": hf_args.lambda_video,
            "lambda_action": hf_args.lambda_action,
        },
        training_phase="language_video" if hf_args.stage == 1 else "full",
        knowledge_insulation=True,
        strict_expert_compat=False,
        layer_alignment_mode="tail_overlap",
        shared_attention_expert="video",
        freeze_language_expert=False,
        freeze_video_expert=False,
        freeze_action_expert=(hf_args.stage == 1),
        fastwam_checkpoint=(hf_args.fastwam_pretrain_checkpoint or None),
        mot_checkpoint_mixed_attn=hf_args.mot_checkpoint_mixed_attn,
        n_cond_latent_frames=hf_args.n_cond_latent_frames,
        use_gradient_checkpointing=grad_ckpt,
    )
    # if local_rank == 0:
    #     cache_freqs = hfastwam.language_expert.freqs
    #     print(
    #         "[ROPE_CACHE_CHECK]",
    #         "dtype=", cache_freqs.dtype,
    #         "is_complex=", torch.is_complex(cache_freqs),
    #         "shape=", tuple(cache_freqs.shape),
    #         flush=True,
    #     )

    #     assert torch.is_complex(cache_freqs)
# 模型整体转换为 BF16 后，先重新生成 complex RoPE。
# 每个 rank 都必须执行。
    _restore_complex_language_rope(
        hfastwam,
        max_total_len=hf_args.qwen3_vl_max_total_len,
        device=model_device,
    )

    # 修复完成后再检查。
    cache_freqs = hfastwam.language_expert.freqs

    if local_rank == 0:
        print(
            "[ROPE_CACHE_CHECK]",
            "dtype=", cache_freqs.dtype,
            "is_complex=", torch.is_complex(cache_freqs),
            "shape=", tuple(cache_freqs.shape),
            flush=True,
        )

    assert torch.is_complex(cache_freqs), (
        "Language RoPE must remain complex, "
        f"but got dtype={cache_freqs.dtype}"
    )
    # The model-wide BF16 cast also touches buffers. Restore the complex RoPE


    # configure_stage(stage=hf_args.stage, hfastwam=hfastwam, args=hf_args)
    # if grad_ckpt and hasattr(hfastwam, "enable_input_require_grads"):
    #     hfastwam.enable_input_require_grads()
    configure_stage(stage=hf_args.stage, hfastwam=hfastwam, args=hf_args)

    # ------------------------------------------------------------------
    # Local smoke-test mode:
    #   - Qwen parameters frozen
    #   - Wan parameters frozen
    #   - Action parameters frozen
    #   - only language/video-side MoT projection adapters remain trainable
    #
    # This runs AFTER configure_stage(), because configure_stage() normally
    # enables the complete language expert in Stage 1.
    # ------------------------------------------------------------------
    if hf_args.local_smoke_adapter_only:
        # Freeze absolutely every real model parameter first.
        hfastwam.requires_grad_(False)

        enabled_adapter_names = []

        for name, param in hfastwam.mot.named_parameters():
            is_projection_adapter = (
                "_proj_to_shared" in name
                or "_proj_from_shared" in name
            )

            # Stage-1 language smoke test does not need Action-side adapters.
            is_action_adapter = "action__" in name

            train_this = is_projection_adapter and not is_action_adapter
            param.requires_grad_(train_this)

            if train_this:
                enabled_adapter_names.append(name)

        if not enabled_adapter_names:
            available_names = [
                name for name, _ in hfastwam.mot.named_parameters()
            ]

            raise RuntimeError(
                "local_smoke_adapter_only=True, but no MoT projection adapters "
                "matched '_proj_to_shared' or '_proj_from_shared'. "
                f"First MoT parameter names: {available_names[:50]}"
            )

        n_train = sum(
            param.numel()
            for param in hfastwam.parameters()
            if param.requires_grad
        )

        qwen_trainable = sum(
            param.numel()
            for param in hfastwam.language_expert.parameters()
            if param.requires_grad
        )

        video_trainable = sum(
            param.numel()
            for param in hfastwam.video_expert.parameters()
            if param.requires_grad
        )

        action_trainable = sum(
            param.numel()
            for param in hfastwam.action_expert.parameters()
            if param.requires_grad
        )

        if local_rank == 0:
            logger.warning(
                "[LOCAL_SMOKE_ADAPTER_ONLY] total_trainable=%.2fM "
                "qwen=%.2fM video=%.2fM action=%.2fM adapters=%d",
                n_train / 1e6,
                qwen_trainable / 1e6,
                video_trainable / 1e6,
                action_trainable / 1e6,
                len(enabled_adapter_names),
            )

            for name in enabled_adapter_names[:30]:
                logger.warning("[SMOKE_TRAINABLE_ADAPTER] %s", name)

        assert qwen_trainable == 0, (
            f"Qwen should be frozen, but {qwen_trainable} parameters are trainable"
        )
        assert video_trainable == 0, (
            f"Wan should be frozen, but {video_trainable} parameters are trainable"
        )
        assert action_trainable == 0, (
            f"Action expert should be frozen, but "
            f"{action_trainable} parameters are trainable"
        )

    if grad_ckpt and hasattr(hfastwam, "enable_input_require_grads"):
        hfastwam.enable_input_require_grads()



    # ---- Dataset + collator ---- #
    train_dataset = InternVLAN1HFastWAMDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        base_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        base_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    def fastwam_collator(features):
        extras = {}
        for k in ("video", "action", "action_is_pad", "video_valid", "action_valid", "prompt"):
            if k in features[0]:
                vals = [f.pop(k) for f in features]
                extras[k] = vals if k == "prompt" else torch.stack(vals, dim=0)
        batch = base_collator(features)
        batch.update(extras)
        if local_rank == 0 and "input_ids" in batch:
            total_tokens = batch["input_ids"].numel()
            if total_tokens > 6000:
                pv = str(batch["pixel_values"].shape) if "pixel_values" in batch else "none"
                logger.warning("[BATCH_TOKENS] total=%d pixel_values=%s", total_tokens, pv)
            if "labels" in batch:
                L=batch["labels"]; n_valid=(L!=-100).sum().item()
                logger.warning("[LABEL_CHECK] valid=%d has_down=%s has_right=%s has_left=%s has_stop=%s",n_valid,(L==79029).any().item(),(L==51018).any().item(),(L==71858).any().item(),(L==50669).any().item())
        return batch

    model = HFastWAMVLMHFWrapper(hfastwam)

    callbacks = [
        HFastWAMCheckpointCallback(
            hfastwam=hfastwam,
            output_dir=training_args.output_dir,
            save_total_limit=training_args.save_total_limit or 5,
        ),
        StepTimingCallback(),
    ]

    # Open-loop eval callback 已禁用：
    # rank=0 单独做 generate_text 时其他 63 rank 挂起等待 → NCCL heartbeat timeout / OOM 风险
    # 请用独立脚本 hfastwam_openloop_eval.py 的 run_openloop_eval() 对 step_*.pt 离线评估
    # if os.path.exists(_eval_pkl):
    #     callbacks.append(HFastWAMOpenLoopEvalCallback(...))

    trainer = transformers.Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=fastwam_collator,
        callbacks=callbacks,
    )

    # This wrapper does not consume ``num_items_in_batch``. Keep both guards
    # disabled so Trainer neither multiplies the loss by world size nor skips
    # gradient-accumulation normalization.
    trainer.model_accepts_loss_kwargs = False
    trainer.args.average_tokens_across_devices = False

    if local_rank == 0:
        logger.warning(
            "[LOSS_CONFIG] model_accepts_loss_kwargs=%s "
            "average_tokens_across_devices=%s grad_accum=%s world_size=%s",
            trainer.model_accepts_loss_kwargs,
            trainer.args.average_tokens_across_devices,
            trainer.args.gradient_accumulation_steps,
            trainer.args.world_size,
        )

    flag=False
    # if flag:
    #     test=train_dataset[0]
    print(training_args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        ckpt_path = os.path.join(training_args.output_dir, "hfastwam_vlm_final.pt")
        hfastwam.save_checkpoint(ckpt_path)
        logger.warning("Saved HFastWAM-VLM checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    train()
