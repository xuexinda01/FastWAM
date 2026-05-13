"""
Pre-compute T5 text embeddings for VLN navigation instructions.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_nav_text_embeds.py
"""

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer


MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
CONTEXT_LEN = 256
BATCH_SIZE = 16

PROMPT_TEMPLATE = "A video recorded from a navigation agent's point of view executing the following instruction: {task}"

DATASET_DIRS = [
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/r2r",
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/rxr",
    "/apdcephfs_gy6/share_303214315/jishengpeng/vlndata/InternData-N1/vln_ce/traj_data/scalevln",
]
CACHE_DIR = Path("/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/text_embeds_cache/nav_vln")


def collect_instructions():
    """Collect all unique instructions from the dataset."""
    instructions = set()
    for dataset_dir in DATASET_DIRS:
        scenes = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
        for scene in scenes:
            ep_file = os.path.join(dataset_dir, scene, "meta", "episodes.jsonl")
            if not os.path.isfile(ep_file):
                continue
            with open(ep_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ep = json.loads(line)
                    tasks = ep.get("tasks", [])
                    if tasks:
                        instructions.add(tasks[0])
    return sorted(instructions)


def _atomic_torch_save(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Compute enc_id from model name
    import re
    enc_id = re.sub(r"[^a-z0-9]+", "", MODEL_ID.split("/")[-1].lower()) or "textenc"

    # Collect instructions
    print("Collecting instructions...")
    instructions = collect_instructions()
    print(f"Found {len(instructions)} unique instructions")

    # Build prompts and filter already-cached
    prompts_to_encode = []
    for inst in instructions:
        prompt = PROMPT_TEMPLATE.format(task=inst)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = CACHE_DIR / f"{hashed}.t5_len{CONTEXT_LEN}.{enc_id}.pt"
        if not cache_path.exists():
            prompts_to_encode.append(prompt)

    print(f"Need to encode {len(prompts_to_encode)} prompts ({len(instructions) - len(prompts_to_encode)} already cached)")

    if not prompts_to_encode:
        print("All embeddings already cached!")
        return

    # Load text encoder
    print("Loading text encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=MODEL_ID,
        tokenizer_model_id=TOKENIZER_MODEL_ID,
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()

    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=CONTEXT_LEN,
        clean="whitespace",
    )
    print(f"Text encoder loaded on {device}")

    # Encode in batches
    print(f"Encoding {len(prompts_to_encode)} prompts in batches of {BATCH_SIZE}...")
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts_to_encode), BATCH_SIZE)):
            batch_prompts = prompts_to_encode[start : start + BATCH_SIZE]
            ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context = text_encoder(ids, mask)

            for i, prompt in enumerate(batch_prompts):
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                payload = {"context": context_i, "mask": mask_i}
                cache_path = CACHE_DIR / f"{hashed}.t5_len{CONTEXT_LEN}.{enc_id}.pt"
                _atomic_torch_save(payload, cache_path)

    print(f"Done! Cached {len(prompts_to_encode)} embeddings to {CACHE_DIR}")


if __name__ == "__main__":
    main()
