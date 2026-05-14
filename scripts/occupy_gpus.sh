#!/usr/bin/env bash
# =============================================================================
# 临时占用所有节点的 GPU，维持指定利用率（默认 40%）
# =============================================================================
#
# Usage:
#   bash scripts/occupy_gpus.sh          # 启动占用（默认 40% 利用率）
#   bash scripts/occupy_gpus.sh --stop   # 停止占用
#
# 环境变量:
#   GPU_UTIL=40    目标 GPU 利用率百分比（默认 40）
#   NUM_GPUS=8     每节点 GPU 数量（默认 8）
#
# 默认每个节点 8 张 GPU，共 64 张
# =============================================================================

set -euo pipefail

ALL_IPS=(
28.216.18.215
28.216.18.163
28.216.19.161
28.216.19.91
28.216.18.202
28.216.19.83
28.216.19.80
28.216.19.208
)

NUM_GPUS=${NUM_GPUS:-8}
GPU_UTIL=${GPU_UTIL:-40}

# ─────────────────────────────────────────────────────────────────────────────
# 停止模式
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
    echo "正在停止所有节点的 GPU 占用进程..."
    for ip in "${ALL_IPS[@]}"; do
        echo "  [$ip] 停止中..."
        ssh -o StrictHostKeyChecking=no "$ip" \
            "pkill -f 'gpu_stress_occupy' 2>/dev/null; echo '    已停止'" 2>&1 || true
    done
    echo "全部停止完成。"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 启动模式
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================="
echo " GPU 占用脚本 - ${#ALL_IPS[@]} 节点 x ${NUM_GPUS} GPU = $((${#ALL_IPS[@]} * NUM_GPUS)) GPU"
echo " 目标利用率: ~${GPU_UTIL}%"
echo "============================================="
echo ""
echo "停止命令: bash scripts/occupy_gpus.sh --stop"
echo ""

for ip in "${ALL_IPS[@]}"; do
    echo "=== [$ip] 启动 ${NUM_GPUS} 个 GPU 占用进程 (${GPU_UTIL}% util) ==="

    # 先清理可能残留的旧进程
    ssh -o StrictHostKeyChecking=no "$ip" "pkill -f 'gpu_stress_occupy' 2>/dev/null" || true

    # 对每个 GPU 启动一个占用进程
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        ssh -o StrictHostKeyChecking=no "$ip" \
            "CUDA_VISIBLE_DEVICES=${gpu} nohup python -c '# gpu_stress_occupy
import torch, os, signal, sys, time

def handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

gpu_id = int(os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"0\"))
device = torch.device(\"cuda:0\")
util = int(os.environ.get(\"GPU_UTIL\", \"40\"))

# 用大矩阵占用更多显存（~16GB/卡）
size = 16384
a = torch.randn(size, size, device=device, dtype=torch.float16)
b = torch.randn(size, size, device=device, dtype=torch.float16)

# 通过 duty cycle 控制利用率:
# 计算一段时间，然后 sleep 一段时间
# ratio = util/100, 即 work/(work+sleep) = ratio
ratio = util / 100.0
# 每轮工作时间 ~50ms，然后按比例 sleep
work_ms = 50
sleep_s = (work_ms / 1000.0) * (1.0 - ratio) / ratio

print(f\"[occupy] GPU {gpu_id} - target ~{util}% util, sleep={sleep_s:.4f}s\", flush=True)

while True:
    t0 = time.time()
    while (time.time() - t0) < (work_ms / 1000.0):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(sleep_s)
' > /dev/null 2>&1 &" GPU_UTIL=${GPU_UTIL}
    done

    echo "  [$ip] ${NUM_GPUS} 个进程已启动"
done

echo ""
echo "============================================="
echo " 所有 $((${#ALL_IPS[@]} * NUM_GPUS)) 个 GPU 占用已启动 (~${GPU_UTIL}% util)"
echo " 停止: bash scripts/occupy_gpus.sh --stop"
echo "============================================="
