#!/bin/bash
# ======================================================
# HFastWAM-VLM Stage 2 — 多机多卡训练 (SSH + torchrun)
#
# 训练目标：
#   - 冻结 VLM (Qwen3-VL) + Wan2.2 video expert
#   - 只训 Action expert (Wan ActionDiT) 学连续轨迹 flow-matching
#   - 载入 stage1 产出的 .pt 作为初始化(VLM + video + MoT 已训好)
#
# 数据：沿用 InternNav lerobot dataset；stage2 只在 pixel_goal 子集(有连续
#       waypoint 的样本)上算 action FM loss(与 InternNav train_dual_system 一致)。
#
# 前提：stage1 已产出 hfastwam_vlm_final.pt 或 step_*.pt。
#
# Usage (master node):
#   STAGE1_CKPT=/path/to/hfastwam_vlm_final.pt \
#     bash scripts/train/qwenvl_train/train_hfastwam_v2_stage2.sh
#
#   # Override:
#   STAGE1_CKPT=... LEARNING_RATE=1e-4 BATCH_SIZE=2 \
#     bash scripts/train/qwenvl_train/train_hfastwam_v2_stage2.sh
# ======================================================

DIR=`pwd`

# --- Configurable ---
DS_CONFIG="${DS_CONFIG:-zero2_overlap.json}"
DATA_FLATTEN="${DATA_FLATTEN:-True}"
NUM_WORKERS="${NUM_WORKERS:-16}"
GRAD_CKPT="${GRAD_CKPT:-True}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"        # stage2 跟 InternNav dual-system 一致
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-/apdcephfs_gy6/share_303214315/zhenye/code_vln/InternNav}"
FASTWAM_ROOT="${FASTWAM_ROOT:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM}"
QWEN3_VL_MODEL_ID="${QWEN3_VL_MODEL_ID:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/ckpts/Qwen3-VL-4B-Instruct}"
QWEN3_VL_LOCAL_FILES_ONLY="${QWEN3_VL_LOCAL_FILES_ONLY:-True}"
WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
FASTWAM_ANNOTATION_CACHE="${FASTWAM_ANNOTATION_CACHE:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache}"
FASTWAM_FRAME_PREFETCH_WORKERS="${FASTWAM_FRAME_PREFETCH_WORKERS:-12}"

# --- Stage1 checkpoint (REQUIRED) ---
# The .pt produced by stage1 (hfastwam_vlm_final.pt or step_<N>.pt). Loaded via
# HFastWAMVLM.load_checkpoint as initialization (language_expert + mot weights).
STAGE1_CKPT="${STAGE1_CKPT:-}"
if [ -z "${STAGE1_CKPT}" ]; then
    echo "[ERROR] STAGE1_CKPT is required for stage2. Point it at stage1's .pt:" >&2
    echo "        STAGE1_CKPT=/apdcephfs_tj5/.../HFastWAM-VLM-Stage1-VLNonly-v2new/hfastwam_vlm_final.pt \\" >&2
    echo "          bash scripts/train/qwenvl_train/train_hfastwam_v2_stage2.sh" >&2
    exit 1
fi
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "[ERROR] STAGE1_CKPT not found: ${STAGE1_CKPT}" >&2
    exit 1
fi

# --- Conda env ---
CONDA_ROOT="${CONDA_ROOT:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3}"
ENV_NAME="${ENV_NAME:-fastwam}"
TORCHRUN_EXE="${CONDA_ROOT}/envs/${ENV_NAME}/bin/torchrun"

# --- Local /tmp cache ---
LOCAL_ENV_DIR="${LOCAL_ENV_DIR:-/tmp/fastwam_env}"
LOCAL_QWEN_DIR="${LOCAL_QWEN_DIR:-/tmp/Qwen3-VL-4B-Instruct}"
LOCAL_ANNO_DIR="${LOCAL_ANNO_DIR:-/tmp/internnav_annotations_cache}"
LOCAL_WAN_DIR="${LOCAL_WAN_DIR:-/tmp/fastwam_checkpoints}"
LOCAL_CACHE_MARKER="${LOCAL_CACHE_MARKER:-/tmp/fastwam_local_cache_ready}"
USE_LOCAL_CACHE="${USE_LOCAL_CACHE:-auto}"

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
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/checkpoints}"
export DIFFSYNTH_MODEL_BASE_PATH
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

# --- 训练参数 ---
MASTER_ADDR=$(head -1 "${HOST_PATH}" | awk '{print $1}')
MASTER_PORT="${MASTER_PORT:-62334}"   # 与 stage1 (62333) 错开,避免端口冲突
NUM_NODES=$(wc -l < "${HOST_PATH}")
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# --- Decide whether to use /tmp local cache ---
USE_LOCAL=0
if [ "${USE_LOCAL_CACHE}" = "yes" ]; then
    USE_LOCAL=1
elif [ "${USE_LOCAL_CACHE}" = "auto" ]; then
    if ssh -n -o ConnectTimeout=5 "${MASTER_ADDR}" "test -f ${LOCAL_CACHE_MARKER}" 2>/dev/null; then
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
            echo "[local-cache] all nodes have ${LOCAL_CACHE_MARKER} — using /tmp paths"
        fi
    else
        echo "[local-cache] master missing marker — using cephfs paths"
    fi
fi

if [ ${USE_LOCAL} -eq 1 ]; then
    TORCHRUN_EXE="${LOCAL_ENV_DIR}/bin/python -m torch.distributed.run"
    QWEN3_VL_MODEL_ID="${LOCAL_QWEN_DIR}"
    FASTWAM_ANNOTATION_CACHE="${LOCAL_ANNO_DIR}"
    DIFFSYNTH_MODEL_BASE_PATH="${LOCAL_WAN_DIR}"
    export DIFFSYNTH_MODEL_BASE_PATH
    SKIP_CONDA_ACTIVATE=1
else
    SKIP_CONDA_ACTIVATE=0
fi

run_name="HFastWAM-VLM-Stage2-DualVLN"
# Stage2 只对有连续 waypoint 的 pixel_goal 子集算 action FM。仍传全部 10 个
# dataset，dataset wrapper 内部用 action_valid 过滤(stop/turn 样本不算 action loss)。
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30,scalevln_125cm_0_30,scalevln_60cm_30_30
OUTPUT_DIR="${OUTPUT_DIR:-/apdcephfs_tj5/share_302528826/xxd/${run_name}}"

# --- Env exports for remote nodes ---
ENV_EXPORTS="${DIR}/.train_env_exports_hfastwam_v2_stage2.sh"
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
export FASTWAM_FRAME_PREFETCH_WORKERS=${FASTWAM_FRAME_PREFETCH_WORKERS}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD}
EOFENV

LOG_DIR="${DIR}/node_logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
rm -f "${LOG_DIR}"/hfastwam_v2_stage2_node_*.log

echo "========================================"
echo "[HFastWAM-VLM Stage2 Multi-Node Config]"
echo "  DS_CONFIG:           ${DS_CONFIG}"
echo "  NUM_WORKERS:         ${NUM_WORKERS}"
echo "  GRAD_CKPT:           ${GRAD_CKPT}"
echo "  BATCH_SIZE:          ${BATCH_SIZE}"
echo "  LEARNING_RATE:       ${LEARNING_RATE}"
echo "  NUM_TRAIN_EPOCHS:    ${NUM_TRAIN_EPOCHS}"
echo "  LAMBDA_ACTION:       ${LAMBDA_ACTION}"
echo "  NUM_NODES:           ${NUM_NODES}"
echo "  NPROC_PER_NODE:      ${NPROC_PER_NODE}"
echo "  MASTER:              ${MASTER_ADDR}:${MASTER_PORT}"
echo "  QWEN3_VL_MODEL_ID:   ${QWEN3_VL_MODEL_ID}"
echo "  STAGE1_CKPT:         ${STAGE1_CKPT}"
echo "  VLN_DATASETS:        ${vln_datasets}"
echo "  OUTPUT_DIR:          ${OUTPUT_DIR}"
echo "  FASTWAM_ANNOTATION_CACHE: ${FASTWAM_ANNOTATION_CACHE}"
echo "========================================"

# --- Pre-launch: kill gpu_stress_occupy placeholders ---
echo "[pre-launch] killing gpu_stress_occupy placeholders on all nodes..."
while IFS=' ' read -r NODE_IP _REST; do
    n_before=$(ssh -n -o ConnectTimeout=5 ${NODE_IP} "pgrep -f '[g]pu_stress_occupy' 2>/dev/null | wc -l" 2>/dev/null)
    ssh -n -o ConnectTimeout=5 ${NODE_IP} "pkill -9 -f '[g]pu_stress_occupy' 2>/dev/null; sleep 1" 2>/dev/null
    n_after=$(ssh -n -o ConnectTimeout=5 ${NODE_IP} "pgrep -f '[g]pu_stress_occupy' 2>/dev/null | wc -l" 2>/dev/null)
    printf "  %-15s killed %s placeholder procs (remaining: %s)\n" "${NODE_IP}" "${n_before:-?}" "${n_after:-?}"
done < "${HOST_PATH}"
echo "[pre-launch] done."
echo "========================================"

# Per-node SSH launch
NODE_RANK=0
while IFS=' ' read -r NODE_IP _REST; do
    LOG_FILE="${LOG_DIR}/hfastwam_v2_stage2_node_${NODE_IP}.log"
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
            --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False --bf16 \
            --num_history 8 --data_augmentation True --resize_h 384 --resize_w 384 --sample_step 4 \
            --num_future_steps 4 --predict_step_num 32 --pixel_goal_only True --system1 none \
            --output_dir ${OUTPUT_DIR} --num_train_epochs ${NUM_TRAIN_EPOCHS} \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps 1 --max_pixels 200704 --min_pixels 3136 \
            --learning_rate ${LEARNING_RATE} --weight_decay 0 \
            --warmup_ratio 0.003 --max_grad_norm 1 \
            --eval_strategy no --save_strategy steps --save_steps 500 --save_total_limit 5 \
            --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{\"min_lr\": 1e-05}' \
            --logging_steps 1 --model_max_length 8192 \
            --gradient_checkpointing ${GRAD_CKPT} \
            --dataloader_num_workers ${NUM_WORKERS} --dataloader_prefetch_factor ${DATALOADER_PREFETCH_FACTOR:-6} --run_name ${run_name} --report_to wandb \
            --dataloader_pin_memory True --ddp_timeout 7200 --torch_compile False \
            --remove_unused_columns False \
            --stage 2 \
            --fastwam_pretrain_checkpoint ${STAGE1_CKPT} \
            --qwen3_vl_model_id ${QWEN3_VL_MODEL_ID} \
            --qwen3_vl_local_files_only ${QWEN3_VL_LOCAL_FILES_ONLY} \
            --qwen3_vl_max_total_len 8192 \
            --wan_model_id ${WAN_MODEL_ID} \
            --wan_tokenizer_model_id ${WAN_TOKENIZER_MODEL_ID} \
            --fastwam_video_size 224 \
            --fastwam_n_history_frames 9 \
            --fastwam_n_future_frames 8 \
            --fastwam_predict_step_num 8 \
            --n_cond_latent_frames 3 \
            --mot_checkpoint_mixed_attn True \
            --lambda_language 0.0 \
            --lambda_video 0.0 \
            --lambda_action ${LAMBDA_ACTION} \
            ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}
    " > ${LOG_FILE} 2>&1 &
    NODE_RANK=$((NODE_RANK + 1))
done < "${HOST_PATH}"

echo "========================================"
echo "All ${NUM_NODES} nodes launched. Following master log..."
echo "========================================"
MASTER_LOG="${LOG_DIR}/hfastwam_v2_stage2_node_${MASTER_ADDR}.log"
sleep 5
tail -f "${MASTER_LOG}" &
TAIL_PID=$!
wait $(jobs -p | grep -v ${TAIL_PID})
kill ${TAIL_PID} 2>/dev/null
echo "Stage2 finished."
