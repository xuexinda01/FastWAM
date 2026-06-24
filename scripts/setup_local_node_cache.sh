#!/bin/bash
# Setup local /tmp cache on each of 8 nodes for fast restart.
#
# Why: cephfs cold-start of 8x8 is ~70 min. Most of that is reading
# 20 GB of small files (conda env + Qwen3-VL weights + dataset
# pickles) from cephfs. Once these are on each node's local SSD
# under /tmp, restart drops to ~10-15 min.
#
# What it copies:
#   /apdcephfs_qy2/.../miniconda3/envs/fastwam     → /tmp/fastwam_env       (8.4 GB)
#   /apdcephfs_qy2/.../ckpts/Qwen3-VL-4B-Instruct  → /tmp/Qwen3-VL-4B-Instruct  (8.3 GB)
#   /apdcephfs_qy2/.../FastWAM/data/internnav_annotations_cache → /tmp/internnav_annotations_cache  (3.1 GB)
#   /apdcephfs_qy2/.../FastWAM/checkpoints (Wan2.2 DiT+VAE) → /tmp/fastwam_checkpoints  (~28 GB)
#
# The Wan2.2 copy is the biggest win: under 64-rank load, cephfs reads of the
# 19 GB DiT drop to ~3.5 MB/s; local SSD is ~GB/s. The stage1/stage2 launch
# scripts auto-point DIFFSYNTH_MODEL_BASE_PATH at /tmp/fastwam_checkpoints when
# the marker is present.
#
# Idempotent: rsync skips files that match. Re-run safely if cephfs source changes.
#
# Usage:
#   bash scripts/setup_local_node_cache.sh                   # run on all 8 nodes
#   bash scripts/setup_local_node_cache.sh --check           # check status only
#   bash scripts/setup_local_node_cache.sh --hosts file      # use custom hostfile
#   FORCE=1 bash scripts/setup_local_node_cache.sh           # force rsync even if marker exists

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_PATH="${HOST_PATH:-${DIR}/train/qwenvl_train/hostfile}"
FORCE="${FORCE:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
if [ "$1" = "--check" ]; then CHECK_ONLY=1; fi

CONDA_ENV_SRC="/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3/envs/fastwam"
QWEN_SRC="/apdcephfs_qy2/share_303214315/hunyuan/xxd/ckpts/Qwen3-VL-4B-Instruct"
ANNO_SRC="/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/data/internnav_annotations_cache"
# Wan2.2 video DiT + VAE (and the redirect dir DiffSynth-Studio/...). This is
# the ~28 GB that dominates cold-start cephfs reads (3.5 MB/s under 64-rank
# contention). Copying it to local SSD is the single biggest speedup.
WAN_SRC="/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/checkpoints"

CONDA_ENV_DST=/tmp/fastwam_env
QWEN_DST=/tmp/Qwen3-VL-4B-Instruct
ANNO_DST=/tmp/internnav_annotations_cache
WAN_DST=/tmp/fastwam_checkpoints

MARKER=/tmp/fastwam_local_cache_ready

if [ ! -f "${HOST_PATH}" ]; then
    echo "[ERROR] hostfile not found: ${HOST_PATH}" >&2
    exit 1
fi

# Read hostnames into array (avoid the ssh-eats-stdin pitfall).
mapfile -t HOSTS < <(awk '{print $1}' "${HOST_PATH}")
NUM_HOSTS=${#HOSTS[@]}

echo "========================================"
echo "Setup local /tmp cache on ${NUM_HOSTS} nodes"
echo "  conda env:   ${CONDA_ENV_SRC} → ${CONDA_ENV_DST}  (8.4 GB)"
echo "  Qwen3-VL:    ${QWEN_SRC} → ${QWEN_DST}  (8.3 GB)"
echo "  Anno cache:  ${ANNO_SRC} → ${ANNO_DST}  (3.1 GB)"
echo "  Wan2.2:      ${WAN_SRC} → ${WAN_DST}  (~28 GB)"
echo "  Total:       ~48 GB / node"
echo "  Marker:      ${MARKER}"
echo "========================================"

# Wan2.2 DiT presence probe (3 sharded safetensors under Wan-AI/Wan2.2-TI2V-5B).
WAN_PROBE="${WAN_DST}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors"

# ---- Check status of all nodes first ---- #
echo "[check] current status:"
for ip in "${HOSTS[@]}"; do
    ssh -n -o ConnectTimeout=5 "${ip}" "
        printf '  %-15s ' '${ip}'
        ce=\$(test -f ${CONDA_ENV_DST}/bin/python && echo OK || echo MISSING)
        qw=\$(test -f ${QWEN_DST}/config.json && echo OK || echo MISSING)
        ac=\$(test -d ${ANNO_DST} && ls ${ANNO_DST}/*.pkl 2>/dev/null | wc -l || echo 0)
        wn=\$(test -f ${WAN_PROBE} && echo OK || echo MISSING)
        mk=\$(test -f ${MARKER} && echo READY || echo NO)
        printf 'env=%s  qwen=%s  anno=%s/10  wan=%s  marker=%s\n' \"\$ce\" \"\$qw\" \"\$ac\" \"\$wn\" \"\$mk\"
    " 2>/dev/null
done

if [ "${CHECK_ONLY}" = "1" ]; then
    exit 0
fi

# ---- rsync each node in parallel ---- #
echo
echo "[rsync] starting parallel rsync on all nodes..."
LOG_DIR=/tmp/setup_local_cache_logs
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}"/*.log

PIDS=()
for ip in "${HOSTS[@]}"; do
    LOG="${LOG_DIR}/${ip}.log"
    (
        ssh -n -o ConnectTimeout=10 "${ip}" "
            set -e
            mkdir -p ${CONDA_ENV_DST} ${QWEN_DST} ${ANNO_DST} ${WAN_DST}

            # --- conda env ---
            if [ '${FORCE}' = '1' ] || ! [ -f ${CONDA_ENV_DST}/bin/python ]; then
                echo '[$(date +%H:%M:%S)] [${ip}] rsync conda env...'
                rsync -a --partial ${CONDA_ENV_SRC}/ ${CONDA_ENV_DST}/
                echo '[$(date +%H:%M:%S)] [${ip}] conda env done'
            else
                echo '[$(date +%H:%M:%S)] [${ip}] conda env already present, skipping'
            fi

            # --- Qwen3-VL ---
            if [ '${FORCE}' = '1' ] || ! [ -f ${QWEN_DST}/config.json ]; then
                echo '[$(date +%H:%M:%S)] [${ip}] rsync Qwen3-VL...'
                rsync -a --partial ${QWEN_SRC}/ ${QWEN_DST}/
                echo '[$(date +%H:%M:%S)] [${ip}] Qwen3-VL done'
            else
                echo '[$(date +%H:%M:%S)] [${ip}] Qwen3-VL already present, skipping'
            fi

            # --- Annotation cache ---
            if [ '${FORCE}' = '1' ] || [ \$(ls ${ANNO_DST}/*.pkl 2>/dev/null | wc -l) -lt 10 ]; then
                echo '[$(date +%H:%M:%S)] [${ip}] rsync annotation cache...'
                rsync -a --partial ${ANNO_SRC}/ ${ANNO_DST}/
                echo '[$(date +%H:%M:%S)] [${ip}] annotation cache done'
            else
                echo '[$(date +%H:%M:%S)] [${ip}] annotation cache already present, skipping'
            fi

            # --- Wan2.2 weights (DiT + VAE + redirect dir) ~28 GB ---
            if [ '${FORCE}' = '1' ] || ! [ -f ${WAN_PROBE} ]; then
                echo '[$(date +%H:%M:%S)] [${ip}] rsync Wan2.2 (~28 GB, slowest)...'
                rsync -a --partial ${WAN_SRC}/ ${WAN_DST}/
                echo '[$(date +%H:%M:%S)] [${ip}] Wan2.2 done'
            else
                echo '[$(date +%H:%M:%S)] [${ip}] Wan2.2 already present, skipping'
            fi

            touch ${MARKER}
            echo '[$(date +%H:%M:%S)] [${ip}] all done, marker created'
        "
    ) > "${LOG}" 2>&1 &
    PIDS+=($!)
    echo "  [${ip}] rsync started in background (pid $!, log ${LOG})"
done

echo
echo "[wait] waiting for all ${#PIDS[@]} rsync tasks..."

# Wait, but show progress every 30 sec
START=$(date +%s)
while true; do
    DONE=0
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            DONE=$((DONE + 1))
        fi
    done
    if [ ${DONE} -eq ${#PIDS[@]} ]; then
        break
    fi
    elapsed=$(( $(date +%s) - START ))
    echo "  [progress] ${DONE}/${#PIDS[@]} nodes done, elapsed ${elapsed}s"
    sleep 30
done

echo
echo "[done] all rsync tasks finished"

# Final status
echo
echo "[check] final status:"
for ip in "${HOSTS[@]}"; do
    ssh -n -o ConnectTimeout=5 "${ip}" "
        printf '  %-15s ' '${ip}'
        ce=\$(test -f ${CONDA_ENV_DST}/bin/python && echo OK || echo MISSING)
        qw=\$(test -f ${QWEN_DST}/config.json && echo OK || echo MISSING)
        ac=\$(test -d ${ANNO_DST} && ls ${ANNO_DST}/*.pkl 2>/dev/null | wc -l || echo 0)
        wn=\$(test -f ${WAN_PROBE} && echo OK || echo MISSING)
        mk=\$(test -f ${MARKER} && echo READY || echo NO)
        printf 'env=%s  qwen=%s  anno=%s/10  wan=%s  marker=%s\n' \"\$ce\" \"\$qw\" \"\$ac\" \"\$wn\" \"\$mk\"
    " 2>/dev/null
done

echo
echo "[done] Local cache setup complete. Next launch will use /tmp paths."
echo "  Run \`bash scripts/train/qwenvl_train/train_hfastwam_v2_stage1.sh\` (same as before; auto-detects /tmp)."
