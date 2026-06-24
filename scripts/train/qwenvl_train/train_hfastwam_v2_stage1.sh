#!/bin/bash
# ======================================================
# HFastWAM-VLM Stage 1 — 多机多卡训练 (SSH + torchrun)
#
# 训练目标：
#   - VLM (Qwen3-VL-2B + 自带 vision encoder) 学 ChatML 子任务 token (CE)
#   - Wan2.2 video expert 学整段 17 帧 flow-matching
#   - Action expert 冻结，不参与 forward
#
# 数据：完全沿用 InternNav 的 lerobot dataset (沿用其样本采样、ChatML 模板)
#       附加 9 history + 8 future 张 raw RGB 喂 Wan VAE
#
# Usage (master node):
#   bash scripts/train/qwenvl_train/train_hfastwam_v2_stage1.sh
#
#   # Override:
#   BATCH_SIZE=2 INTERNNAV_ROOT=/path/to/InternNav \
#     bash scripts/train/qwenvl_train/train_hfastwam_v2_stage1.sh
# ======================================================

DIR=`pwd`

# --- Configurable ---
DS_CONFIG="${DS_CONFIG:-zero2_overlap.json}"
DATA_FLATTEN="${DATA_FLATTEN:-True}"
NUM_WORKERS="${NUM_WORKERS:-16}"
# Per-sample frame-prefetch threads inside the dataset wrapper. Raised from the
# old hardcoded 4 to削平 rxr 长样本(22-24帧)的 jpg-decode stall that 64-rank
# sync amplifies into 200-470 s slow steps.
FASTWAM_FRAME_PREFETCH_WORKERS="${FASTWAM_FRAME_PREFETCH_WORKERS:-12}"
# Data augmentation (ColorJitter/Posterize/Sharpness/Autocontrast per history
# image) is pure-CPU heavy and the main cause of the dataloader's per-sample
# time variance (long rxr samples starve the prefetch buffer → 55s↔200s sawtooth).
# Default OFF for stage1 (video FM doesn't need photometric aug). Set True to restore.
DATA_AUGMENTATION="${DATA_AUGMENTATION:-False}"
# Deeper prefetch buffer (default HF=2) so smooth steps stockpile batches that
# cover the long-sample spikes. Costs CPU RAM (we have headroom).
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-6}"
GRAD_CKPT="${GRAD_CKPT:-True}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav}"
FASTWAM_ROOT="${FASTWAM_ROOT:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM}"
QWEN3_VL_MODEL_ID="${QWEN3_VL_MODEL_ID:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/ckpts/Qwen3-VL-2B-Instruct}"
QWEN3_VL_LOCAL_FILES_ONLY="${QWEN3_VL_LOCAL_FILES_ONLY:-True}"
# Wan2.2 video expert (DiffSynth lazy-downloads to DIFFSYNTH_MODEL_BASE_PATH).
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
# Pre-computed lerobot annotation cache (built by scripts/cache_internnav_annotations.py).
# When set, the dataset wrapper skips the per-rank cephfs scan of every parquet file
# and instead unpickles per-dataset annotations from this directory. Saves ~3 hr of
# cold-start time on 64-rank launches.
FASTWAM_ANNOTATION_CACHE="${FASTWAM_ANNOTATION_CACHE:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache}"

# --- Conda env ---
CONDA_ROOT="${CONDA_ROOT:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3}"
ENV_NAME="${ENV_NAME:-fastwam}"
TORCHRUN_EXE="${CONDA_ROOT}/envs/${ENV_NAME}/bin/torchrun"

# --- Local /tmp cache (set up via scripts/setup_local_node_cache.sh) ---
# If /tmp on each node has a copy of the conda env, Qwen3-VL weights, and
# annotation cache, we use those — avoids the ~50 min cephfs cold-start.
# Detection: each node will check via ssh below; we use the shared marker
# logic where if master has the marker, we assume all nodes do (setup
# script ensures this).
LOCAL_ENV_DIR="${LOCAL_ENV_DIR:-/tmp/fastwam_env}"
LOCAL_QWEN_DIR="${LOCAL_QWEN_DIR:-/tmp/Qwen3-VL-4B-Instruct}"
LOCAL_ANNO_DIR="${LOCAL_ANNO_DIR:-/tmp/internnav_annotations_cache}"
LOCAL_WAN_DIR="${LOCAL_WAN_DIR:-/tmp/fastwam_checkpoints}"
LOCAL_CACHE_MARKER="${LOCAL_CACHE_MARKER:-/tmp/fastwam_local_cache_ready}"
USE_LOCAL_CACHE="${USE_LOCAL_CACHE:-auto}"   # auto / yes / no

# --- 自动抓节点 ---
if [ -f /etc/taiji/environ ]; then
    source /etc/taiji/environ
    echo "$NODE_IP_LIST" > env.txt 2>&1
    sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" > "hostfile"
    sed "s/:.//g"      env.txt | sed "s/,/\n/g" > "pssh.hosts"
fi
HOST_PATH="${HOST_PATH:-${DIR}/hostfile}"
if [ ! -f "${HOST_PATH}" ]; then
    echo "[ERROR] hostfile not found: ${HOST_PATH}" >&2
    exit 1
fi
PSSH_HOSTS="${DIR}/pssh.hosts"
if [ ! -f "${PSSH_HOSTS}" ]; then
    awk '{print $1}' "${HOST_PATH}" > "${PSSH_HOSTS}"
fi

# --- Cleanup leftover processes ---
if command -v pssh >/dev/null 2>&1; then
    pssh -i -t 0 -h "${PSSH_HOSTS}" \
        "ps -ef | grep -E 'hfastwam_internnav_trainer|torchrun' | grep -v grep | awk '{print \$2}' | xargs -r kill -9" \
        2>/dev/null || true
    sleep 5s
fi

# --- NCCL / env ---
export NCCL_IB_TIMEOUT=22
export NCCL_RETRY_COUNTER=1000
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=1800000
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_TOPO_AFFINITY=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG=WARN
export WANDB_MODE=offline

export INTERNNAV_ROOT
export FASTWAM_ROOT
# Wan2.2 权重本地副本(含 DiT + VAE,经 redirect 命中 DiffSynth-Studio 的 VAE)。
# 指向 cephfs 上已下好的完整副本,避免每个 rank 去 modelscope 抢文件锁/联网下载。
# SKIP_DOWNLOAD=true:本地命中就用,缺文件直接报错而不是死等 modelscope 锁。
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/checkpoints}"
export DIFFSYNTH_MODEL_BASE_PATH
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

# --- 训练参数 ---
MASTER_ADDR=$(head -1 "${HOST_PATH}" | awk '{print $1}')
MASTER_PORT="${MASTER_PORT:-62333}"
NUM_NODES=$(wc -l < "${HOST_PATH}")
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# --- Decide whether to use /tmp local cache ---
# Probe master node for the marker; if present, all 8 should be (setup script
# only writes it when all 3 components are in place on each node).
USE_LOCAL=0
if [ "${USE_LOCAL_CACHE}" = "yes" ]; then
    USE_LOCAL=1
elif [ "${USE_LOCAL_CACHE}" = "auto" ]; then
    if ssh -n -o ConnectTimeout=5 "${MASTER_ADDR}" "test -f ${LOCAL_CACHE_MARKER}" 2>/dev/null; then
        # Verify all nodes have it.
        ALL_READY=1
        while IFS=' ' read -r NODE_IP _REST; do
            if ! ssh -n -o ConnectTimeout=5 "${NODE_IP}" "test -f ${LOCAL_CACHE_MARKER}" 2>/dev/null; then
                ALL_READY=0
                echo "[local-cache] ${NODE_IP} missing marker — falling back to cephfs"
                break
            fi
        done < "${HOST_PATH}"
        if [ ${ALL_READY} -eq 1 ]; then
            USE_LOCAL=1
            echo "[local-cache] all 8 nodes have ${LOCAL_CACHE_MARKER} — using /tmp paths"
        fi
    else
        echo "[local-cache] master missing marker — using cephfs paths"
    fi
fi

if [ ${USE_LOCAL} -eq 1 ]; then
    # Override paths to local /tmp copies. Conda env: skip activate, call
    # python directly (sys.path is computed from binary location).
    TORCHRUN_EXE="${LOCAL_ENV_DIR}/bin/python -m torch.distributed.run"
    QWEN3_VL_MODEL_ID="${LOCAL_QWEN_DIR}"
    FASTWAM_ANNOTATION_CACHE="${LOCAL_ANNO_DIR}"
    # Wan2.2 from local SSD instead of cephfs (the big cold-start win).
    DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_WAN_DIR}"
    export DIFFSYNTH_MODEL_BASE_PATH
    SKIP_CONDA_ACTIVATE=1
else
    SKIP_CONDA_ACTIVATE=0
fi

run_name="HFastWAM-VLM-Stage1-VLNonly"
# scalevln_125cm_0_45 dropped: cephfs scalevln parquets do NOT contain
# pose.125cm_45deg / goal.125cm_45deg columns (only r2r/rxr have 45deg pitch data).
# get_annotations_from_lerobot_data returns 0 episodes for it; keeping it in
# the spec produces an empty pickle and a 0-sample dataset that the InternNav
# loader can occasionally trip over. 10 datasets total now.
# Full 10-dataset spec (use when r2r/rxr are localized to /tmp):
#   r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30,scalevln_125cm_0_30,scalevln_60cm_30_30
# Default below = scalevln-only (both scalevln datasets are localized to /tmp).
# r2r/rxr are temporarily excluded to avoid cephfs straggler stalls until they
# are also rsynced to node-local /tmp. Override via VLN_DATASETS env to restore.
vln_datasets="${VLN_DATASETS:-scalevln_125cm_0_30,scalevln_60cm_30_30}"
OUTPUT_DIR="${OUTPUT_DIR:-/apdcephfs_tj5/share_302528826/xxd/${run_name}}"

# --- Env exports for remote nodes ---
ENV_EXPORTS="${DIR}/.train_env_exports_hfastwam_v2_stage1.sh"
cat > "${ENV_EXPORTS}" << EOFENV
export NCCL_IB_TIMEOUT=22
export NCCL_RETRY_COUNTER=1000
export NCCL_NVLS_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=1800000
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_TOPO_AFFINITY=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG=WARN
export WANDB_MODE=offline
export INTERNNAV_ROOT=${INTERNNAV_ROOT}
export FASTWAM_ROOT=${FASTWAM_ROOT}
export FASTWAM_ANNOTATION_CACHE=${FASTWAM_ANNOTATION_CACHE}
# Prevent stale .pyc on cephfs (mtime precision issue causes old code to be loaded)
export PYTHONDONTWRITEBYTECODE=1
export FASTWAM_FRAME_PREFETCH_WORKERS=${FASTWAM_FRAME_PREFETCH_WORKERS}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD}
# Tokenization disk cache (pre-computed by cache_qwen_tokenization.py)
export FASTWAM_QWEN_TOK_CACHE=""  # 禁用 tok cache，直接原始计算
EOFENV

LOG_DIR="${DIR}/node_logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
rm -f "${LOG_DIR}"/hfastwam_v2_stage1_node_*.log

echo "========================================"
echo "[HFastWAM-VLM Stage1 Multi-Node Config]"
echo "  DS_CONFIG:           ${DS_CONFIG}"
echo "  DATA_FLATTEN:        ${DATA_FLATTEN}"
echo "  NUM_WORKERS:         ${NUM_WORKERS}"
echo "  GRAD_CKPT:           ${GRAD_CKPT}"
echo "  BATCH_SIZE:          ${BATCH_SIZE}"
echo "  NUM_NODES:           ${NUM_NODES}"
echo "  NPROC_PER_NODE:      ${NPROC_PER_NODE}"
echo "  MASTER:              ${MASTER_ADDR}:${MASTER_PORT}"
echo "  QWEN3_VL_MODEL_ID:   ${QWEN3_VL_MODEL_ID}"
echo "  VLN_DATASETS:        ${vln_datasets}"
echo "  OUTPUT_DIR:          ${OUTPUT_DIR}"
echo "  INTERNNAV_ROOT:      ${INTERNNAV_ROOT}"
echo "  FASTWAM_ROOT:        ${FASTWAM_ROOT}"
echo "  FASTWAM_ANNOTATION_CACHE: ${FASTWAM_ANNOTATION_CACHE}"
echo "========================================"

# --- Pre-launch: kill any GPU placeholder processes that occupy CUDA memory.
# These are launched manually by the user before training starts (e.g.
# `python -c '# gpu_stress_occupy ...'` matmul loops). They MUST be killed
# on every node before torchrun, otherwise CUDA allocation will fight or
# OOM during model load.
echo "[pre-launch] killing gpu_stress_occupy placeholders on all nodes..."
while IFS=' ' read -r NODE_IP _REST; do
    # Use a regex that doesn't match the pgrep command line itself (avoids
    # false-positive "remaining" count from the search process matching itself).
    # Also pass -n to ssh so it does NOT consume stdin from the while loop —
    # otherwise the very first ssh call gobbles all remaining lines from
    # the hostfile and the loop only processes one node.
    n_before=$(ssh -n -o ConnectTimeout=5 ${NODE_IP} "pgrep -f '[g]pu_stress_occupy' 2>/dev/null | wc -l" 2>/dev/null)
    ssh -n -o ConnectTimeout=5 ${NODE_IP} "pkill -9 -f '[g]pu_stress_occupy' 2>/dev/null; sleep 1" 2>/dev/null
    n_after=$(ssh -n -o ConnectTimeout=5 ${NODE_IP} "pgrep -f '[g]pu_stress_occupy' 2>/dev/null | wc -l" 2>/dev/null)
    printf "  %-15s killed %s placeholder procs (remaining: %s)\n" "${NODE_IP}" "${n_before:-?}" "${n_after:-?}"
done < "${HOST_PATH}"
echo "[pre-launch] done."

# --- Pre-flight validation (runs on master node only, ~30s) ---
# Checks: (1) correct .py code loaded (no stale .pyc), (2) disk cache hit,
# (3) data completeness. Catches all known failure modes before cold start.
_VALIDATE_PY="/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3/envs/fastwam/bin/python"
_VALIDATE_SCRIPT="${DIR}/scripts/validate_training_setup.py"
echo "[preflight] running end-to-end validation..."
FASTWAM_QWEN_TOK_CACHE="" "${_VALIDATE_PY}" "${_VALIDATE_SCRIPT}" 2>&1
if [ $? -ne 0 ]; then
    echo "[preflight] ABORT: validation failed. Fix issues before restarting."
    exit 1
fi
echo "[preflight] validation passed."

# --- Per-node data integrity check (runs on all nodes in parallel) ---
echo "[preflight] running data integrity check on all nodes..."
PREFLIGHT_SCRIPT="${DIR}/scripts/preflight_check.py"
PREFLIGHT_DIR="${DIR}/.cctmp/preflight"
PY_BIN="${PY_BIN:-python}"
# Use the conda python if available, else system python
if [ -x "/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3/envs/fastwam/bin/python" ]; then
    PY_BIN="/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3/envs/fastwam/bin/python"
fi
rm -f "${PREFLIGHT_DIR}"/*.txt 2>/dev/null
mkdir -p "${PREFLIGHT_DIR}"
while IFS=' ' read -r NODE_IP _REST; do
    ssh -n -o ConnectTimeout=8 "${NODE_IP}" \
        "nohup FASTWAM_QWEN_TOK_CACHE='' ${PY_BIN} ${PREFLIGHT_SCRIPT} --datasets ${vln_datasets} >/dev/null 2>&1 &" 2>/dev/null
done < "${HOST_PATH}"
# Wait up to 300s for all nodes to finish (python import torch takes ~60s on cold start)
for _i in $(seq 1 60); do
    sleep 5
    n_done=$(grep -l "STATUS=" "${PREFLIGHT_DIR}"/*.txt 2>/dev/null | wc -l)
    n_nodes=$(wc -l < "${HOST_PATH}")
    [ "${n_done}" -ge "${n_nodes}" ] && break
done
# Collect results
PREFLIGHT_FAIL=0
while IFS=' ' read -r NODE_IP _REST; do
    result_file="${PREFLIGHT_DIR}/${NODE_IP}.txt"
    if [ ! -f "${result_file}" ]; then
        echo "[preflight] WARNING: ${NODE_IP} did not report (timeout)"
        continue
    fi
    status=$(grep "^STATUS=" "${result_file}" | cut -d= -f2)
    if [ "${status}" != "PASS" ]; then
        PREFLIGHT_FAIL=1
        echo "[preflight] FAIL on ${NODE_IP}:"
        grep "^BAD:" "${result_file}" | sed 's/^/  /'
    else
        echo "[preflight] PASS: ${NODE_IP}"
    fi
done < "${HOST_PATH}"
if [ "${PREFLIGHT_FAIL}" -eq 1 ]; then
    echo "[preflight] ABORT: fix data issues before restarting training."
    echo "[preflight] Run: bash .cctmp/repair_rxr_empty.sh  (for rxr empty dirs)"
    echo "[preflight] Run: rsync from .215 for missing scalevln frames"
    exit 1
fi
echo "[preflight] all nodes passed. Starting training..."
echo "========================================"

# Per-node SSH launch
NODE_RANK=0
while IFS=' ' read -r NODE_IP _REST; do
    LOG_FILE="${LOG_DIR}/hfastwam_v2_stage1_node_${NODE_IP}.log"
    echo "[Launch] node_rank=${NODE_RANK} on ${NODE_IP} → ${LOG_FILE}"
    if [ ${SKIP_CONDA_ACTIVATE:-0} -eq 1 ]; then
        ACTIVATE_CMD="echo '[fast-launch] using /tmp local cache, skipping conda activate'"
    else
        ACTIVATE_CMD="source ${CONDA_ROOT}/bin/activate ${ENV_NAME}"
    fi
    ssh -n ${NODE_IP} "
        ${ACTIVATE_CMD}
        cd ${DIR}
        source ${ENV_EXPORTS}
        ${TORCHRUN_EXE} \
            --nnodes=${NUM_NODES} \
            --nproc_per_node=${NPROC_PER_NODE} \
            --rdzv_backend=static \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            --node_rank=${NODE_RANK} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${FASTWAM_ROOT}/scripts/train/qwenvl_train/hfastwam_internnav_trainer.py \
            --deepspeed ${FASTWAM_ROOT}/scripts/train/qwenvl_train/${DS_CONFIG} \
            --model_name_or_path ${QWEN3_VL_MODEL_ID} \
            --vln_dataset_use ${vln_datasets} \
            --data_flatten ${DATA_FLATTEN} \
            --tune_mm_vision True --tune_mm_mlp True --tune_mm_llm True --bf16 \
            --num_history 8 --data_augmentation ${DATA_AUGMENTATION} --resize_h 384 --resize_w 384 --sample_step 4 \
            --num_future_steps 4 --predict_step_num 32 --pixel_goal_only False --system1 none \
            --output_dir ${OUTPUT_DIR} --num_train_epochs 1.0 \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps 4 --max_pixels 200704 --min_pixels 3136 \
            --learning_rate 2e-5 --vision_tower_lr 5e-6 --weight_decay 0 \
            --warmup_ratio 0.003 --max_grad_norm 1 \
            --eval_strategy no --save_strategy steps --save_steps 50 --save_total_limit 3 \
            --lr_scheduler_type cosine --logging_steps 1 --model_max_length 16384 \
            --gradient_checkpointing ${GRAD_CKPT} \
            --dataloader_num_workers ${NUM_WORKERS} --dataloader_prefetch_factor ${DATALOADER_PREFETCH_FACTOR} --run_name ${run_name} --report_to wandb \
            --dataloader_pin_memory True --ddp_timeout 7200 --torch_compile False \
            --remove_unused_columns False \
            --seed ${TRAIN_SEED:-$(shuf -i 1-9999 -n 1)} \
            --stage 1 \
            --qwen3_vl_model_id ${QWEN3_VL_MODEL_ID} \
            --qwen3_vl_local_files_only ${QWEN3_VL_LOCAL_FILES_ONLY} \
            --qwen3_vl_max_total_len 16384 \
            --wan_model_id ${WAN_MODEL_ID} \
            --wan_tokenizer_model_id ${WAN_TOKENIZER_MODEL_ID} \
            --fastwam_video_size 224 \
            --fastwam_n_history_frames 9 \
            --fastwam_n_future_frames 8 \
            --fastwam_predict_step_num 8 \
            --n_cond_latent_frames 3 \
            --mot_checkpoint_mixed_attn True \
            --lambda_language 1.0 \
            --lambda_video ${LAMBDA_VIDEO:-1.0} \
            --lambda_action 0.0 \
            ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}
    " > ${LOG_FILE} 2>&1 &
    NODE_RANK=$((NODE_RANK + 1))
done < "${HOST_PATH}"

echo "========================================"
echo "All ${NUM_NODES} nodes launched. Following master log..."
echo "========================================"
MASTER_LOG="${LOG_DIR}/hfastwam_v2_stage1_node_${MASTER_ADDR}.log"
sleep 5
tail -f "${MASTER_LOG}" &
TAIL_PID=$!
wait $(jobs -p | grep -v ${TAIL_PID})
kill ${TAIL_PID} 2>/dev/null
echo "Stage1 finished."
