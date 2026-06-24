"""
HFastWAM Open-Loop Action Token Evaluation
==========================================
每 N 步在 rank-0 上触发一次 VLM 开环测试:
  - 从 R2R val_unseen episodes 中随机采 K 条轨迹
  - 每条轨迹取 1 个 step（带历史帧 + 当前帧 + 俯视帧，完全复现训练输入）
  - 用 generate_text 解码 action token
  - 统计 action token accuracy（STOP / ↑ / ← / → / ↓ 各类别 + overall）

集成方式：在 hfastwam_internnav_trainer.py 里注册 HFastWAMOpenLoopEvalCallback
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Action token 映射（与训练数据一致）
# ------------------------------------------------------------------ #
IDX2ACTION = {0: "STOP", 1: "↑", 2: "←", 3: "→", 5: "↓"}
ACTION2IDX = {v: k for k, v in IDX2ACTION.items()}

# R2R val_unseen scenes（标准列表）
R2R_VAL_UNSEEN_SCENES = {
    "QUCTc6BB5sX", "EU6Fwq7SyZv", "pLe4wQe7qrG", "oLBMNvg9in8", "TbHJrupSAjP",
    "5q7pvUzZiYa", "1pXnuDYAj8r", "VVfe2KiqLaN", "ZMojNkEp431", "fzynW3qQPVF",
    "UwV83HsGsw3", "D7N2EKCX4Sj", "17DRP5sb8fy", "sT4fr6TAbpF", "X7HyMhZNoso",
}


# ------------------------------------------------------------------ #
# 数据加载工具
# ------------------------------------------------------------------ #

def _get_scene_from_video_path(video_path: str) -> str:
    """从 video path 解析 scene id"""
    # .../traj_data/r2r/<scene>/videos/chunk-000
    parts = video_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("r2r", "rxr", "scalevln") and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _load_val_episodes(pkl_path: str, n_samples: int, rng: random.Random) -> list:
    """从 pkl annotation 里随机取 val_unseen episodes"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    episodes = data["episodes"]

    # 过滤 val_unseen scenes
    val_eps = [
        ep for ep in episodes
        if _get_scene_from_video_path(ep["video"]) in R2R_VAL_UNSEEN_SCENES
    ]
    if not val_eps:
        # fallback：直接用所有 episodes
        val_eps = episodes

    rng.shuffle(val_eps)
    return val_eps[:n_samples]


def _remap_video_path(path: str) -> str:
    """把 gy6 路径映射到 /tmp，和训练时一致"""
    remaps = [
        ("/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r",
         "/tmp/r2r"),
        ("/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/rxr",
         "/tmp/rxr"),
        ("/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/scalevln",
         "/tmp/scalevln"),
    ]
    for old, new in remaps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def _open_frame(video_dir: str, cam_dir: str, frame_id: int, height_px: int = 384) -> Optional[Image.Image]:
    """打开一帧 JPEG，resize 到 height_px"""
    jpg_path = os.path.join(video_dir, cam_dir, f"{frame_id:06d}.jpg")
    if not os.path.exists(jpg_path):
        return None
    try:
        img = Image.open(jpg_path).convert("RGB")
        w, h = img.size
        new_w = int(w * height_px / h)
        return img.resize((new_w, height_px), Image.BILINEAR)
    except Exception:
        return None


def _build_one_sample(
    ep: dict,
    step_idx: int,
    height: int,
    pitch_1: int,
    pitch_2: int,
    n_history: int = 9,
) -> Optional[dict]:
    """
    构造单步 val 样本（复现训练输入格式）:
      - history front-view frames  (≤ n_history, pitch_1)
      - current front-view frame   (1, pitch_1)
      - current lookdown frame     (1, pitch_2)  ← pixel goal
      - instruction text
      - gt_action: int (IDX2ACTION index)
    返回 None 表示帧文件缺失。
    """
    actions = ep["actions"]  # [-1, 3, 3, 1, ...]  -1 = first step dummy
    pixel_goals = ep.get("pixel_goals", [])  # [[frame_id, [x,y]], ...]

    # 过滤掉 -1（dummy 初始 action）和 STOP 本身
    valid_steps = [i for i, a in enumerate(actions) if a not in (-1,) and i > 0]
    if step_idx >= len(valid_steps):
        return None
    real_step = valid_steps[step_idx]
    gt_action = actions[real_step]

    # pixel goal（俯视帧）
    pg_frame_id = None
    if gt_action == 1 and pixel_goals:  # FORWARD → 需要 lookdown
        if real_step < len(pixel_goals):
            pg_frame_id = pixel_goals[real_step][0] if isinstance(pixel_goals[real_step], (list, tuple)) else None

    video_dir = _remap_video_path(ep["video"])
    cam_front = f"observation.images.rgb.{height}cm_{pitch_1}deg"
    cam_look  = f"observation.images.rgb.{height}cm_{pitch_2}deg"

    # 历史帧 idx（linspace，和训练一致）
    history_ids = np.linspace(0, real_step - 1, n_history, dtype=np.int32).tolist() if real_step > 0 else []

    images = []
    # (a) 历史帧
    for fid in history_ids:
        img = _open_frame(video_dir, cam_front, int(fid))
        if img is None:
            return None
        images.append(img)
    # (b) 当前前视角帧
    cur_front = _open_frame(video_dir, cam_front, real_step)
    if cur_front is None:
        return None
    images.append(cur_front)
    # (c) 当前俯视帧（仅 FORWARD 且有 pixel goal）
    if pg_frame_id is not None:
        cur_look = _open_frame(video_dir, cam_look, pg_frame_id)
        if cur_look is not None:
            images.append(cur_look)

    instruction = ep["instructions"] if isinstance(ep["instructions"], str) else ep["instructions"][0]

    return {
        "images": images,
        "instruction": instruction,
        "gt_action": gt_action,
        "gt_action_str": IDX2ACTION.get(gt_action, str(gt_action)),
        "step": real_step,
        "scene": _get_scene_from_video_path(ep["video"]),
    }


# ------------------------------------------------------------------ #
# 核心评估逻辑
# ------------------------------------------------------------------ #

def _build_conversation_prompt(instruction: str, n_images: int, conjunctions: list) -> list:
    """构造 ChatML conversation（复现训练 prompt）"""
    img_placeholder = "<image>"
    img_tokens = "".join([img_placeholder] * n_images)

    # 和 InternNav internvla_n1_lerobot_dataset.py 保持一致的 prompt
    conj = conjunctions[0] if conjunctions else "What is your next action?"
    user_content = (
        f"You are an autonomous navigation assistant. Your task is to {instruction} "
        f"Where should you go next to stay on track? "
        f"When you want to output a waypoint you need to TILT DOWN (↓) by 30 degrees "
        f"then output the next waypoint's coordinates in the image. "
        f"In case the next waypoint is out of view, utilize the turn actions: "
        f"TURN LEFT (←) or TURN RIGHT (→) by 30 degrees. "
        f"Please output STOP when you have successfully completed the task. "
        f"{conj}{img_tokens}."
    )
    return [{"role": "user", "content": user_content}]


@torch.no_grad()
def run_openloop_eval(
    model,           # HFastWAMVLM
    processor,       # AutoProcessor (Qwen3VL)
    pkl_path: str,   # annotation pkl
    height: int = 125,
    pitch_1: int = 0,
    pitch_2: int = 30,
    n_samples: int = 100,
    n_history: int = 9,
    device: str = "cuda:0",
    step_offset: int = 0,  # 每条 episode 取第几个 valid step
    seed: int = 42,
) -> dict:
    """
    运行开环 val，返回指标 dict:
      overall_acc, per_class_acc, n_samples, confusion (pred×gt count)
    """
    rng = random.Random(seed)
    episodes = _load_val_episodes(pkl_path, n_samples * 3, rng)  # 多取一些以防帧缺失

    # conjunctions（和训练一致的连接词）
    conjunctions = [
        "Look around carefully and ",
        "Observe the surroundings and ",
        "Based on the current view, ",
        "Given the current observation, ",
    ]

    results = []
    skipped = 0

    model.eval()

    for ep in episodes:
        if len(results) >= n_samples:
            break

        # 随机选一个 step
        actions = ep["actions"]
        valid_steps = [i for i, a in enumerate(actions) if a not in (-1,) and i > 0]
        if not valid_steps:
            skipped += 1
            continue

        step_idx = rng.randint(0, min(len(valid_steps) - 1, 20))
        sample = _build_one_sample(ep, step_idx, height, pitch_1, pitch_2, n_history)
        if sample is None:
            skipped += 1
            continue

        images = sample["images"]
        instruction = sample["instruction"]
        gt_action = sample["gt_action"]

        # 构造 conversation
        conversation = _build_conversation_prompt(instruction, len(images), conjunctions)

        try:
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            logger.warning("[OPENLOOP_EVAL] processor error: %s", e)
            skipped += 1
            continue

        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, dtype=model.dtype if hasattr(model, 'dtype') else torch.bfloat16)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        # dummy video (stage1 val 只测 language)
        dummy_video = torch.zeros(1, 3, 1, 16, 16, device=device,
                                  dtype=pixel_values.dtype if pixel_values is not None else torch.bfloat16)

        eos_ids = [processor.tokenizer.eos_token_id]
        if hasattr(processor.tokenizer, "convert_tokens_to_ids"):
            im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end and im_end != processor.tokenizer.unk_token_id:
                eos_ids.append(im_end)

        try:
            gen_ids = model.generate_text(
                input_ids=input_ids,
                video=dummy_video,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=16,
                eos_token_id=eos_ids,
            )
        except Exception as e:
            logger.warning("[OPENLOOP_EVAL] generate_text error: %s", e)
            skipped += 1
            continue

        pred_text = processor.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()

        # 解析 pred action token
        pred_action = None
        for tok, idx in ACTION2IDX.items():
            if pred_text.startswith(tok):
                pred_action = idx
                break
        if pred_action is None:
            # 输出坐标 "x y" → 也算 FORWARD (↑) 因为模型先输出 ↓ 再输出坐标
            import re
            if re.match(r"^\d+", pred_text):
                pred_action = 1  # ↑ (pixel goal)
            else:
                pred_action = -1  # unknown

        results.append({
            "gt": gt_action,
            "pred": pred_action,
            "gt_str": IDX2ACTION.get(gt_action, str(gt_action)),
            "pred_str": pred_text[:20],
            "scene": sample["scene"],
        })

    if not results:
        return {"error": "no valid samples", "n_samples": 0}

    # 统计
    correct = sum(1 for r in results if r["gt"] == r["pred"])
    overall_acc = correct / len(results)

    # per-class accuracy
    per_class = {}
    for idx, name in IDX2ACTION.items():
        cls_results = [r for r in results if r["gt"] == idx]
        if cls_results:
            cls_acc = sum(1 for r in cls_results if r["pred"] == idx) / len(cls_results)
            per_class[name] = {"acc": cls_acc, "n": len(cls_results)}

    logger.warning(
        "[OPENLOOP_EVAL] step=%d  overall_acc=%.3f (%d/%d)  skipped=%d",
        step_offset, overall_acc, correct, len(results), skipped
    )
    for name, v in per_class.items():
        logger.warning(
            "[OPENLOOP_EVAL]   class=%-5s  acc=%.3f  n=%d",
            name, v["acc"], v["n"]
        )

    # 打印几个样本
    for r in results[:5]:
        logger.warning(
            "[OPENLOOP_EVAL]   sample  gt=%-5s  pred=%-20s  correct=%s  scene=%s",
            r["gt_str"], r["pred_str"], r["gt"] == r["pred"], r["scene"]
        )

    return {
        "overall_acc": overall_acc,
        "n_correct": correct,
        "n_samples": len(results),
        "n_skipped": skipped,
        "per_class": per_class,
    }


# ------------------------------------------------------------------ #
# HF Trainer Callback
# ------------------------------------------------------------------ #

class HFastWAMOpenLoopEvalCallback(TrainerCallback):
    """
    每 eval_every_steps 步（仅 rank-0）触发开环 val。
    在 hfastwam_internnav_trainer.py 的 main() 里注册即可：
        callbacks.append(HFastWAMOpenLoopEvalCallback(
            hfastwam=hfastwam,
            processor=processor,
            pkl_path=".../r2r_125cm_0_30.pkl",
            eval_every_steps=100,
        ))
    """

    def __init__(
        self,
        hfastwam,
        processor,
        pkl_path: str,
        eval_every_steps: int = 100,
        n_samples: int = 100,
        height: int = 125,
        pitch_1: int = 0,
        pitch_2: int = 30,
        n_history: int = 9,
        device: str = "cuda:0",
    ):
        self.hfastwam = hfastwam
        self.processor = processor
        self.pkl_path = pkl_path
        self.eval_every_steps = eval_every_steps
        self.n_samples = n_samples
        self.height = height
        self.pitch_1 = pitch_1
        self.pitch_2 = pitch_2
        self.n_history = n_history
        self.device = device
        self._last_eval_step = -1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # 只在 rank-0 跑，避免多卡重复
        if args.local_rank not in (-1, 0):
            return

        step = state.global_step
        if step == self._last_eval_step:
            return
        if step % self.eval_every_steps != 0:
            return

        self._last_eval_step = step

        if not os.path.exists(self.pkl_path):
            logger.warning("[OPENLOOP_EVAL] pkl not found: %s", self.pkl_path)
            return

        logger.warning("[OPENLOOP_EVAL] ===== Starting open-loop eval at step %d =====", step)

        # 暂时切 eval 模式
        was_training = self.hfastwam.training
        try:
            metrics = run_openloop_eval(
                model=self.hfastwam,
                processor=self.processor,
                pkl_path=self.pkl_path,
                height=self.height,
                pitch_1=self.pitch_1,
                pitch_2=self.pitch_2,
                n_samples=self.n_samples,
                n_history=self.n_history,
                device=self.device,
                step_offset=step,
                seed=step,  # 每次用不同 seed，采不同样本
            )
            # 写到 trainer logs
            if state.log_history is not None:
                entry = {"step": step, "eval_openloop_acc": metrics.get("overall_acc", 0.0)}
                for cls_name, v in metrics.get("per_class", {}).items():
                    entry[f"eval_openloop_acc_{cls_name}"] = v["acc"]
                state.log_history.append(entry)
        except Exception as e:
            logger.warning("[OPENLOOP_EVAL] ERROR: %s", e, exc_info=True)
        finally:
            if was_training:
                self.hfastwam.train()
