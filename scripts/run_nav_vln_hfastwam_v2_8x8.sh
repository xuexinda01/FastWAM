#!/usr/bin/env bash
# =============================================================================
# Nav VLN Training (H-FastWAM v2) — 8 Machines x 8 GPUs (64 GPUs total)
# =============================================================================
#

# This is a clone of run_nav_vln_8x8.sh that targets the *new* H-FastWAM
# implementation under `fastwam.models.hfastwam_v2` via the
# `hfastwam_nav_5b_v2` model config. The original
# `run_nav_vln_8x8.sh` (action-only 5B) is left untouched — both
# scripts can coexist on the same hosts.
#
# Usage (on each node):
#   NNODES=8 NODE_RANK=<0-7> MASTER_ADDR=<master_ip> MASTER_PORT=29500 \
#       bash scripts/run_nav_vln_hfastwam_v2_8x8.sh
#
# Or use the helper launcher (from master node, requires passwordless ssh):
#   bash scripts/run_nav_vln_hfastwam_v2_8x8.sh --launch
#
# Environment variables (same as run_nav_vln_8x8.sh):
#   NNODES       - Number of machines (default: 8)
#   NODE_RANK    - Rank of this machine (0-based, default: 0)
#   MASTER_ADDR  - IP address of rank-0 node (required for multi-node)
#   MASTER_PORT  - Communication port (default: 29501; bumped from 29500
#                  so this run does not collide with an action-only run)
#   NUM_GPUS     - GPUs per node (default: 8)
#   HOSTFILE     - Path to hostfile for --launch mode
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
NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Use a different port from run_nav_vln_8x8.sh (29500) so an action-only
# run on the same host does not collide with this 3-expert run.
MASTER_PORT=${MASTER_PORT:-29501}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Task / model / data — point at the *new* configs we just added.
TASK="nav_vln_hfastwam_5b"
MODEL="hfastwam_nav_5b_v2"
DATA="nav_vln_hfastwam"
OUTPUT_BASE="${OUTPUT_DIR:-/apdcephfs_tj5/share_302528826/xxd/nav_vln_hfastwam_v2_5b}"
WANDB_NAME="${WANDB_NAME:-nav_vln_hfastwam_v2_5b_8x8}"

# 从本地磁盘加载模型（避免 cephfs 并发读取瓶颈）
export DIFFSYNTH_MODEL_BASE_PATH="/tmp/fastwam_checkpoints"

# NCCL 通信配置
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=WARN

# DeepSpeed ZeRO-2（5B+ 模型 DDP 在 A100 40G 上 OOM）
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_ZERO_STAGE=2
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${PROJECT_ROOT}/scripts/ds_configs/ds_zero2_config.json"
export ACCELERATE_MIXED_PRECISION=bf16

EXTRA_ARGS=("$@")

# ─────────────────────────────────────────────────────────────────────────────
# Multi-node launcher mode (--launch)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--launch" ]]; then
    HOSTFILE="${HOSTFILE:-${PROJECT_ROOT}/hostfile.txt}"
    if [[ ! -f "${HOSTFILE}" ]]; then
        echo "[ERROR] Hostfile not found: ${HOSTFILE}"
        echo "        Create a hostfile with one IP per line (8 lines for 8 nodes)."
        exit 1
    fi

    mapfile -t HOSTS < <(grep -v '^\s*#' "${HOSTFILE}" | grep -v '^\s*$')
    NUM_HOSTS=${#HOSTS[@]}

    if (( NUM_HOSTS < NNODES )); then
        echo "[ERROR] Hostfile has ${NUM_HOSTS} hosts but NNODES=${NNODES}"
        exit 1
    fi

    MASTER_IP="${HOSTS[0]}"

    echo "[launcher] Stopping GPU occupy processes on all nodes..."
    for host in "${HOSTS[@]}"; do
        ssh -o StrictHostKeyChecking=no "$host" "pkill -f 'gpu_stress_occupy' 2>/dev/null" &
    done
    wait
    echo "[launcher] GPU occupy stopped."

    # Clean prior torchrun processes from this script (and any leftovers
    # from the action-only run that match similar patterns).
    echo "[launcher] Cleaning up previous training processes on all nodes..."
    for host in "${HOSTS[@]}"; do
        ssh -o StrictHostKeyChecking=no "$host" \
            "pkill -f 'torchrun.*train.py' 2>/dev/null; pkill -f 'python.*train.py' 2>/dev/null; true" &
    done
    wait
    sleep 3
    echo "[launcher] Previous training processes cleaned up."

    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
    echo "============================================="
    echo " Launching Nav VLN H-FastWAM v2 training on ${NNODES} nodes"
    echo " Master: ${MASTER_IP}:${MASTER_PORT}"
    echo " GPUs per node: ${NUM_GPUS}"
    echo " Total GPUs: $((NNODES * NUM_GPUS))"
    echo " RUN_ID: ${RUN_ID}"
    echo "============================================="

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
             bash scripts/run_nav_vln_hfastwam_v2_8x8.sh > logs/hfastwam_v2_node_${rank}.log 2>&1 </dev/null"
    done

    echo "[launcher] Starting rank 0 on ${MASTER_IP} (local)..."
    NNODES=${NNODES} NODE_RANK=0 \
    MASTER_ADDR=${MASTER_IP} MASTER_PORT=${MASTER_PORT} \
    NUM_GPUS=${NUM_GPUS} RUN_ID=${RUN_ID} \
    bash scripts/run_nav_vln_hfastwam_v2_8x8.sh

    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Per-node training launch
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================="
echo " Nav VLN H-FastWAM v2 — Node ${NODE_RANK}/${NNODES}"
echo " Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo " GPUs on this node: ${NUM_GPUS}"
echo " Total GPUs: $((NNODES * NUM_GPUS))"
echo "============================================="

mkdir -p "${PROJECT_ROOT}/logs"

export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT

export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

TOTAL_PROCESSES=$((NUM_GPUS * NNODES))
echo "[debug] which accelerate: $(which accelerate)"
echo "[debug] which python: $(which python)"
echo "[debug] which torchrun: $(which torchrun)"
echo "[launch] nproc_per_node=${NUM_GPUS} num_machines=${NNODES} machine_rank=${NODE_RANK} total_processes=${TOTAL_PROCESSES} run_id=${RUN_ID}"

torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NUM_GPUS}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${PROJECT_ROOT}/scripts/train.py" \
  "task=${TASK}" \
  "model=${MODEL}" \
  "data=${DATA}" \
  "output_dir=${OUTPUT_BASE}/${RUN_ID}" \
  "wandb.name=${WANDB_NAME}" \
  "${EXTRA_ARGS[@]}"
