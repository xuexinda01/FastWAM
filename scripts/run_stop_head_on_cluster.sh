#!/usr/bin/env bash
# =============================================================================
# Stop Head Training — 8 Machines x 8 GPUs (64 GPUs total)
# =============================================================================
#
# 与主训练 run_nav_vln_8x8.sh 完全相同的多机启动方式。
# 需要在独立的一组机器上运行（不要和主训练共享机器，避免 NCCL 冲突）。
#
# Usage (on each node):
#   NNODES=8 NODE_RANK=<0-7> MASTER_ADDR=<master_ip> MASTER_PORT=29500 \
#       bash scripts/run_stop_head_on_cluster.sh
#
# Or use the helper launcher (from master node):
#   bash scripts/run_stop_head_on_cluster.sh --launch
#
# Environment variables:
#   NNODES       - Number of machines (default: 8)
#   NODE_RANK    - Rank of this machine (0-based, default: 0)
#   MASTER_ADDR  - IP address of rank-0 node (required for multi-node)
#   MASTER_PORT  - Communication port (default: 29500)
#   NUM_GPUS     - GPUs per node (default: 8)
#   HOSTFILE     - Path to hostfile for --launch mode
#
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Conda env
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
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# VAE 路径 (各节点本地磁盘)
export DIFFSYNTH_MODEL_BASE_PATH="/tmp/fastwam_checkpoints"

# NCCL 通信配置
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=WARN

# Output
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="/apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4/stop_head/${RUN_ID}"

# ─────────────────────────────────────────────────────────────────────────────
# Multi-node launcher mode (--launch)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--launch" ]]; then
    HOSTFILE="${HOSTFILE:-${PROJECT_ROOT}/hostfile_stop_head.txt}"
    if [[ ! -f "${HOSTFILE}" ]]; then
        echo "[ERROR] Hostfile not found: ${HOSTFILE}"
        echo "        Create hostfile_stop_head.txt with one IP per line."
        echo "        Use a SEPARATE set of machines from the main training!"
        exit 1
    fi

    mapfile -t HOSTS < <(grep -v '^\s*#' "${HOSTFILE}" | grep -v '^\s*$')
    NUM_HOSTS=${#HOSTS[@]}
    NNODES=${NUM_HOSTS}

    MASTER_IP="${HOSTS[0]}"
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"

    echo "============================================="
    echo " Stop Head Training — ${NNODES} nodes x ${NUM_GPUS} GPUs"
    echo " Master: ${MASTER_IP}:${MASTER_PORT}"
    echo " Total GPUs: $((NNODES * NUM_GPUS))"
    echo " RUN_ID: ${RUN_ID}"
    echo " Output: /apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4/stop_head/${RUN_ID}"
    echo "============================================="

    # Launch on remote nodes first (rank 1..N-1)
    for ((rank = 1; rank < NNODES; rank++)); do
        host="${HOSTS[$rank]}"
        echo "[launcher] Starting rank ${rank} on ${host}..."
        ssh -f -o StrictHostKeyChecking=no "${host}" \
            "cd ${PROJECT_ROOT} && \
             source ${CONDA_ROOT}/etc/profile.d/conda.sh && \
             conda activate ${CONDA_ENV} && \
             NNODES=${NNODES} NODE_RANK=${rank} \
             MASTER_ADDR=${MASTER_IP} MASTER_PORT=${MASTER_PORT} \
             NUM_GPUS=${NUM_GPUS} RUN_ID=${RUN_ID} \
             bash scripts/run_stop_head_on_cluster.sh > logs/stop_head_node_${rank}.log 2>&1 </dev/null"
    done

    # Launch master node (rank 0) in foreground
    echo "[launcher] Starting rank 0 on ${MASTER_IP} (local)..."
    NNODES=${NNODES} NODE_RANK=0 \
    MASTER_ADDR=${MASTER_IP} MASTER_PORT=${MASTER_PORT} \
    NUM_GPUS=${NUM_GPUS} RUN_ID=${RUN_ID} \
    bash scripts/run_stop_head_on_cluster.sh

    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Per-node training launch
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================="
echo " Stop Head Training — Node ${NODE_RANK}/${NNODES}"
echo " Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo " GPUs on this node: ${NUM_GPUS}"
echo " Total GPUs: $((NNODES * NUM_GPUS))"
echo " Output: ${OUTPUT_DIR}"
echo "============================================="

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NUM_GPUS}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/train_stop_head.py" \
    --batch_size ${BATCH_SIZE:-32} \
    --lr ${LR:-1e-3} \
    --num_epochs ${NUM_EPOCHS:-20} \
    --stop_threshold ${STOP_THRESHOLD:-5} \
    --sample_stride ${SAMPLE_STRIDE:-2} \
    --balance_ratio ${BALANCE_RATIO:-3.0} \
    --n_history_frames 8 \
    --video_feat_dim 512 \
    --overhead_feat_dim 256 \
    --hidden_dim 256 \
    --vae_path "${DIFFSYNTH_MODEL_BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 4 \
    "$@"

echo ""
echo "============================================="
echo " Done! $(date)"
echo " Checkpoints: ${OUTPUT_DIR}"
echo "============================================="
