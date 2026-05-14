#!/usr/bin/env bash
# =============================================================================
# 初始化所有 worker 节点：安装 ceph-fuse、配置 taiji_client、挂载共享磁盘
# =============================================================================
#
# Usage:
#   bash scripts/setup_nodes.sh
#
# 前提：当前节点可以免密 SSH 到所有 worker 节点
# =============================================================================

set -euo pipefail

WORKER_IPS=(
    28.216.18.163
    28.216.19.161
    28.216.19.91
    28.216.18.202
    28.216.19.83
    28.216.19.80
    28.216.19.208
)

TAIJI_CONFIG="/root/.taijiconfig"
BUSINESS_FLAG="TaiJi_HYAide_OS3_Extra_GY_H20"
CHECKPOINT_SRC="/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/checkpoints"
CHECKPOINT_LOCAL="/tmp/fastwam_checkpoints"

# 包含 master 在内的所有节点
ALL_IPS=(
    28.216.18.215
    "${WORKER_IPS[@]}"
)

echo "============================================="
echo " Worker 节点初始化脚本"
echo " 节点数: ${#WORKER_IPS[@]}"
echo "============================================="

# 检查本机 taiji 配置文件
if [[ ! -f "${TAIJI_CONFIG}" ]]; then
    echo "[ERROR] 本机 ${TAIJI_CONFIG} 不存在，无法分发 token"
    exit 1
fi

for ip in "${WORKER_IPS[@]}"; do
    echo ""
    echo "=== [$ip] 开始初始化 ==="

    # Step 1: 安装 ceph-fuse
    echo "  [1/4] 安装 ceph-fuse..."
    ssh -o StrictHostKeyChecking=no "$ip" \
        "if ! command -v ceph-fuse &>/dev/null; then \
            wget -q -O /etc/yum.repos.d/ceph_el7_1.repo http://gaia.repo.oa.com/ceph_el7.repo && \
            yum install -y ceph-fuse >/dev/null 2>&1 && \
            echo '    ceph-fuse 安装成功'; \
         else \
            echo '    ceph-fuse 已存在，跳过'; \
         fi"

    # Step 2: 分发 taiji_client 二进制
    echo "  [2/4] 分发 taiji_client..."
    ssh -o StrictHostKeyChecking=no "$ip" \
        "if ! command -v taiji_client &>/dev/null; then echo 'need_copy'; else echo 'exists'; fi" \
        | grep -q "need_copy" && \
        scp -o StrictHostKeyChecking=no /usr/bin/taiji_client "$ip":/usr/bin/taiji_client >/dev/null 2>&1 && \
        echo "    taiji_client 已复制" || echo "    taiji_client 已存在，跳过"

    # Step 3: 分发 taiji 配置 (token)
    echo "  [3/4] 分发 taiji 配置..."
    scp -o StrictHostKeyChecking=no "${TAIJI_CONFIG}" "$ip":"${TAIJI_CONFIG}" >/dev/null 2>&1
    echo "    配置已同步"

    # Step 4: 挂载磁盘
    echo "  [4/4] 挂载共享磁盘..."
    ssh -o StrictHostKeyChecking=no "$ip" \
        "taiji_client mount -bf ${BUSINESS_FLAG} -l qy 2>&1 | grep -E '\[info\]|\[error\]'; \
         taiji_client mount -bf ${BUSINESS_FLAG} -l gy 2>&1 | grep -E '\[info\]|\[error\]'"

    echo "  === [$ip] 初始化完成 ==="
done

echo ""
echo "============================================="
echo " 所有节点初始化完成！"
echo "============================================="

# 验证：检查项目目录是否可访问
echo ""
echo "验证各节点项目目录访问..."
for ip in "${WORKER_IPS[@]}"; do
    result=$(ssh -o StrictHostKeyChecking=no "$ip" \
        "ls /apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/scripts/train.py 2>&1" && echo "OK" || echo "FAILED")
    printf "  %-16s %s\n" "$ip" "$result"
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 并行预拷贝模型 checkpoints 到所有节点的本地磁盘
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " 并行预拷贝模型到所有节点本地磁盘 (${CHECKPOINT_LOCAL})"
echo " 源路径: ${CHECKPOINT_SRC}"
echo " 节点数: ${#ALL_IPS[@]}"
echo "============================================="

# 记录每个后台任务的 PID
declare -A COPY_PIDS

for ip in "${ALL_IPS[@]}"; do
    if [[ "$ip" == "28.216.18.215" ]]; then
        # master 本地拷贝
        (
            mkdir -p "${CHECKPOINT_LOCAL}" && \
            cp -r "${CHECKPOINT_SRC}"/* "${CHECKPOINT_LOCAL}/" && \
            echo "done" > "${CHECKPOINT_LOCAL}/.done"
        ) &
        COPY_PIDS["$ip"]=$!
    else
        # worker 远程拷贝，通过子 shell + ssh 在后台并行执行
        (
            ssh -o StrictHostKeyChecking=no "$ip" \
                "mkdir -p ${CHECKPOINT_LOCAL} && \
                 cp -r ${CHECKPOINT_SRC}/* ${CHECKPOINT_LOCAL}/ && \
                 echo 'done' > ${CHECKPOINT_LOCAL}/.done"
        ) &
        COPY_PIDS["$ip"]=$!
    fi
    echo "  [${ip}] 拷贝已启动 (PID: ${COPY_PIDS[$ip]})"
done

echo ""
echo "所有 ${#ALL_IPS[@]} 个节点已并行启动拷贝，等待完成..."
echo ""

# 等待所有后台拷贝完成，逐个收集结果
FAILED_NODES=()
for ip in "${ALL_IPS[@]}"; do
    if wait "${COPY_PIDS[$ip]}"; then
        echo "  [${ip}] 拷贝完成 ✓"
    else
        echo "  [${ip}] 拷贝失败 ✗"
        FAILED_NODES+=("$ip")
    fi
done

echo ""
if [[ ${#FAILED_NODES[@]} -eq 0 ]]; then
    echo "============================================="
    echo " 所有节点模型拷贝完成！"
    echo "============================================="
else
    echo "============================================="
    echo " [WARNING] ${#FAILED_NODES[@]} 个节点拷贝失败:"
    for ip in "${FAILED_NODES[@]}"; do
        echo "   - $ip"
    done
    echo "============================================="
    exit 1
fi
