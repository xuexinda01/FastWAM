#!/bin/bash
# =============================================================================
# HFastWAM-VLM Stage 1 Inference Test (single-GPU)
#
# Background — why this is just an "inference test" rather than a full VLN
# evaluation harness like ``run_fastwam_eval.sh``:
#   - Stage 1 trains the VLM language expert (ChatML CE) + video expert
#     (flow-matching FM); the action expert is FROZEN.
#   - VLN evaluation needs trajectory waypoints (forward/left/yaw/moving)
#     which only the action expert produces. Those will arrive in Stage 2.
#   - This script verifies what Stage 1 IS supposed to produce:
#       * per-sample loss values (lang CE + video FM) match training trend
#       * VLM next-token accuracy on supervised positions
#
# Reference structure mirror:
#   /apdcephfs_tj5/share_302528826/xxd/fastwam_vln_eval/run_fastwam_eval.sh
# Differences:
#   - No fastwam_server.py + Habitat client split (single-process, no socket).
#   - No checkpoint stop_head logic (stage 1 doesn't have that head).
#   - No max_burst / waypoint_mode flags (no actions yet).
#   - Reuses InternNav lerobot dataset (validation slice) instead of Habitat sim.
#
# Usage:
#   bash scripts/eval/run_hfastwam_vlm_eval.sh [checkpoint] [num_samples] [vln_datasets]
#
# Examples:
#   # Default: latest ckpt under OUTPUT_DIR, 4 samples from r2r:
#   bash scripts/eval/run_hfastwam_vlm_eval.sh
#
#   # Specific ckpt:
#   bash scripts/eval/run_hfastwam_vlm_eval.sh \\
#       /apdcephfs_tj5/share_302528826/xxd/HFastWAM-VLM-Stage1-VLNonly/checkpoint-1000/hfastwam_vlm_final.pt
#
#   # 32 samples from rxr:
#   bash scripts/eval/run_hfastwam_vlm_eval.sh "" 32 rxr_125cm_0_30
#
#   # Override defaults via env:
#   GPU_ID=2 DUMP_DIR=/tmp/eval_out \\
#       bash scripts/eval/run_hfastwam_vlm_eval.sh /path/to/ckpt.pt 16
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_ROOT:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav}"
CONDA_ROOT="${CONDA_ROOT:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3}"
ENV_NAME="${ENV_NAME:-fastwam}"

# Args (positional, with sensible defaults).
CHECKPOINT="${1:-}"
NUM_SAMPLES="${2:-4}"
VLN_DATASETS="${3:-r2r_125cm_0_30}"

# Optional env-overridable settings.
GPU_ID="${GPU_ID:-0}"
QWEN3_VL_MODEL_ID="${QWEN3_VL_MODEL_ID:-/tmp/Qwen3-VL-4B-Instruct}"
QWEN3_VL_LOCAL_FILES_ONLY="${QWEN3_VL_LOCAL_FILES_ONLY:-True}"
FASTWAM_ANNOTATION_CACHE="${FASTWAM_ANNOTATION_CACHE:-/tmp/internnav_annotations_cache}"
DUMP_DIR="${DUMP_DIR:-}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-/apdcephfs_tj5/share_302528826/xxd/HFastWAM-VLM-Stage1-VLNonly}"

# Auto-detect newest checkpoint when none specified.
if [ -z "${CHECKPOINT}" ]; then
    if [ -d "${CKPT_BASE_DIR}" ]; then
        CHECKPOINT=$(find "${CKPT_BASE_DIR}" \
            \( -name "hfastwam_vlm_final.pt" -o -name "step_*.pt" \) \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    fi
    if [ -z "${CHECKPOINT}" ]; then
        echo "[INFO] No checkpoint found under ${CKPT_BASE_DIR}; will run with random init."
    else
        echo "[INFO] Auto-detected latest checkpoint: ${CHECKPOINT}"
    fi
fi

# Conda env (skip activate if /tmp local cache available, like the train script).
LOCAL_ENV_DIR="${LOCAL_ENV_DIR:-/tmp/fastwam_env}"
if [ -f "${LOCAL_ENV_DIR}/bin/python" ]; then
    PYTHON="${LOCAL_ENV_DIR}/bin/python"
    echo "[env] using /tmp local conda copy: ${PYTHON}"
else
    source "${CONDA_ROOT}/bin/activate" "${ENV_NAME}"
    PYTHON="$(which python)"
    echo "[env] using cephfs conda env: ${PYTHON}"
fi

export FASTWAM_ROOT INTERNNAV_ROOT FASTWAM_ANNOTATION_CACHE
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/tmp/fastwam_checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

echo "============================================="
echo "HFastWAM-VLM Stage 1 Inference Test"
echo "============================================="
echo "  GPU_ID:                 ${GPU_ID}"
echo "  CHECKPOINT:             ${CHECKPOINT:-<random init>}"
echo "  NUM_SAMPLES:            ${NUM_SAMPLES}"
echo "  VLN_DATASETS:           ${VLN_DATASETS}"
echo "  QWEN3_VL_MODEL_ID:      ${QWEN3_VL_MODEL_ID}"
echo "  FASTWAM_ANNOTATION_CACHE: ${FASTWAM_ANNOTATION_CACHE}"
echo "  DUMP_DIR:               ${DUMP_DIR:-<none>}"
echo "============================================="

DUMP_FLAG=""
if [ -n "${DUMP_DIR}" ]; then
    DUMP_FLAG="--dump_dir ${DUMP_DIR}"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} -u \
    "${FASTWAM_ROOT}/scripts/eval/test_hfastwam_vlm_inference.py" \
    --checkpoint "${CHECKPOINT}" \
    --num_samples "${NUM_SAMPLES}" \
    --vln_dataset_use "${VLN_DATASETS}" \
    --qwen3_vl_model_id "${QWEN3_VL_MODEL_ID}" \
    --qwen3_vl_local_files_only "${QWEN3_VL_LOCAL_FILES_ONLY}" \
    --device "cuda:0" \
    ${DUMP_FLAG}

echo ""
echo "[done]"
