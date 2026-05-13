#!/usr/bin/env bash
# =============================================================================
# Stop Head 独立训练脚本
# =============================================================================
#
# 这个脚本完全独立于主训练流程 (run_nav_vln_8x8.sh)。
# 只训练一个轻量级的 stop 预测头 (~1.3M 可训练参数)。
#
# 训练量估计:
#   - 主训练 (6B MoT, 64 GPU): ~数天
#   - Stop Head (1.3M params, 1-4 GPU): ~1-2小时
#
# Usage:
#   # 单GPU (推荐，因为模型很小)
#   bash scripts/run_stop_head_train.sh
#
#   # 多GPU (如果数据量大想加速)
#   NUM_GPUS=4 bash scripts/run_stop_head_train.sh
#
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Conda environment activation
# ─────────────────────────────────────────────────────────────────────────────
CONDA_ROOT="/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3"
CONDA_ENV="fastwam"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
    export PATH="${CONDA_ROOT}/bin:${PATH}"
fi
conda activate "${CONDA_ENV}"
echo "[env] Using python: $(which python) ($(python --version 2>&1))"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# VAE 模型路径
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/tmp/fastwam_checkpoints}"

# 输出目录
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/stop_head/${RUN_ID}}"

# ─────────────────────────────────────────────────────────────────────────────
# Training parameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE=${BATCH_SIZE:-32}              # 9帧视频+下倾, 比单帧大, 建议32
LR=${LR:-1e-3}
NUM_EPOCHS=${NUM_EPOCHS:-20}
STOP_THRESHOLD=${STOP_THRESHOLD:-5}       # 距离终点<=5步为stop
SAMPLE_STRIDE=${SAMPLE_STRIDE:-2}         # 数据采样步长
BALANCE_RATIO=${BALANCE_RATIO:-3.0}       # 正样本过采样
N_HISTORY_FRAMES=${N_HISTORY_FRAMES:-8}   # 历史帧数
VIDEO_FEAT_DIM=${VIDEO_FEAT_DIM:-512}
OVERHEAD_FEAT_DIM=${OVERHEAD_FEAT_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-256}

echo "============================================="
echo " Stop Head Training"
echo " 输入: 9帧0deg(8历史+1当前) + 1帧30deg下倾 + text"
echo "============================================="
echo " GPUs: ${NUM_GPUS}"
echo " Batch size (per GPU): ${BATCH_SIZE}"
echo " History frames: ${N_HISTORY_FRAMES}"
echo " LR: ${LR}"
echo " Epochs: ${NUM_EPOCHS}"
echo " Stop threshold: ${STOP_THRESHOLD}"
echo " Output: ${OUTPUT_DIR}"
echo "============================================="

mkdir -p "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --num_epochs "${NUM_EPOCHS}"
    --stop_threshold "${STOP_THRESHOLD}"
    --sample_stride "${SAMPLE_STRIDE}"
    --balance_ratio "${BALANCE_RATIO}"
    --n_history_frames "${N_HISTORY_FRAMES}"
    --video_feat_dim "${VIDEO_FEAT_DIM}"
    --overhead_feat_dim "${OVERHEAD_FEAT_DIM}"
    --hidden_dim "${HIDDEN_DIM}"
    --vae_path "${DIFFSYNTH_MODEL_BASE_PATH}"
    --output_dir "${OUTPUT_DIR}"
    "$@"
)

if (( NUM_GPUS > 1 )); then
    echo "[launch] Using torchrun with ${NUM_GPUS} GPUs"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        "${PROJECT_ROOT}/scripts/train_stop_head.py" \
        "${COMMON_ARGS[@]}"
else
    echo "[launch] Single GPU training"
    python "${PROJECT_ROOT}/scripts/train_stop_head.py" \
        "${COMMON_ARGS[@]}"
fi

echo ""
echo "============================================="
echo " Training complete!"
echo " Checkpoints saved to: ${OUTPUT_DIR}"
echo "============================================="
