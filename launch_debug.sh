#!/usr/bin/env bash
# =============================================================================
# launch_debug.sh - Single-GPU debug script for VSCode debugger
#
# Usage:
#   1. Set a breakpoint in VSCode
#   2. In VSCode "Run and Debug" panel, select "Python: Remote Attach" (port 5678)
#   3. Run this script in terminal: bash launch_debug.sh
#   4. The script will wait for the debugger to attach before proceeding
#
# Alternatively, use the VSCode launch.json config below to launch directly.
# =============================================================================
set -euo pipefail

# --- Conda environment ---
CONDA_ROOT="/apdcephfs_gy2/share_302533218/shaunxhwang/miniconda3"
CONDA_ENV="fastwam"

# Activate conda
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# --- Project directory ---
cd /apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM

# --- Debug settings ---
export CUDA_VISIBLE_DEVICES=0          # Single GPU
export HYDRA_FULL_ERROR=1              # Full hydra stack traces
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# --- Task configuration (same as your training command) ---
TASK="libero_uncond_2cam224_1e-4"
MODEL="fastwam"
DATA="libero_2cam"
RUN_ID="debug_$(date +%Y-%m-%d_%H-%M-%S)"

echo "============================================"
echo " FastWAM Single-GPU Debug Launch"
echo "============================================"
echo " Conda env : ${CONDA_ENV}"
echo " GPU       : ${CUDA_VISIBLE_DEVICES}"
echo " Task      : ${TASK}"
echo " Model     : ${MODEL}"
echo " Data      : ${DATA}"
echo " Run ID    : ${RUN_ID}"
echo "============================================"
echo ""
echo " Waiting for VSCode debugger to attach on port 5678..."
echo " (Make sure you have 'Python: Remote Attach' configured in launch.json)"
echo ""

# Launch training with debugpy (waits for debugger to attach)
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    scripts/train.py \
    "task=${TASK}" \
    "model=${MODEL}" \
    "data=${DATA}" \
    "output_dir=./runs/${TASK}/${RUN_ID}" \
    "wandb.name=${TASK}_debug"
