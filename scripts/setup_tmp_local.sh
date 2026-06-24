#!/usr/bin/env bash
# ============================================================
# setup_tmp_local.sh
# 将 conda 环境和模型从 ceph-fuse 迁移到本地 /tmp
# 解决 VSCode 单卡调试启动极慢的问题（根因：ceph-fuse 小文件 I/O）
#
# 用法：bash scripts/setup_tmp_local.sh
# 预计耗时：conda env ~5-15min，模型 ~2-5min（取决于网络带宽）
# ============================================================
set -euo pipefail

SRC_CONDA="/apdcephfs_qy2/share_303214315/hunyuan/xxd/miniconda3/envs/fastwam"
DST_CONDA="/tmp/fastwam_env"

SRC_MODEL="/apdcephfs_qy2/share_303214315/hunyuan/xxd/ckpts/Qwen3-VL-4B-Instruct"
DST_MODEL="/tmp/ckpts/Qwen3-VL-4B-Instruct"

# ---- 颜色 ----
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

# ---- 检查源路径 ----
[[ -d "$SRC_CONDA" ]] || { echo "ERROR: conda env not found: $SRC_CONDA"; exit 1; }
[[ -d "$SRC_MODEL" ]] || { echo "ERROR: model not found: $SRC_MODEL"; exit 1; }

# ---- 同步 conda 环境 ----
if [[ -x "$DST_CONDA/bin/python3.10" ]]; then
    warn "conda env already exists at $DST_CONDA, running incremental rsync..."
else
    log "Creating $DST_CONDA ..."
    mkdir -p "$DST_CONDA"
fi

log "Syncing conda env (~9.4 GB): $SRC_CONDA → $DST_CONDA"
log "  (大约需要 5-15 分钟，取决于 ceph 带宽)"
rsync -a --info=progress2 --no-inc-recursive \
    "$SRC_CONDA/" "$DST_CONDA/"
log "✓ conda env 同步完成"

# ---- 验证 Python 可执行 ----
if "$DST_CONDA/bin/python3.10" -c "import sys; print('Python', sys.version)" 2>/dev/null; then
    log "✓ Python 可正常执行（前缀自动重定位到 $DST_CONDA）"
else
    warn "Python 无法直接执行，可能需要手动设置 PYTHONHOME"
    warn "尝试: PYTHONHOME=$DST_CONDA $DST_CONDA/bin/python3.10 -c 'import sys; print(sys.prefix)'"
fi

# ---- 同步模型 ----
mkdir -p "$(dirname "$DST_MODEL")"

if [[ -d "$DST_MODEL" ]]; then
    warn "模型目录已存在 $DST_MODEL，运行增量 rsync..."
else
    log "Creating $DST_MODEL ..."
fi

log "Syncing Qwen3-VL-4B-Instruct (~8.3 GB): $SRC_MODEL → $DST_MODEL"
log "  (大约需要 2-5 分钟)"
rsync -a --info=progress2 --no-inc-recursive \
    "$SRC_MODEL/" "$DST_MODEL/"
log "✓ 模型同步完成"

# ---- 汇总 ----
echo ""
log "===== 迁移完成 ====="
log "conda env : $DST_CONDA/bin/python"
log "模型      : $DST_MODEL"
echo ""
log "launch.json 已更新指向这些路径，直接 F5 调试即可。"
log "如需保持 /tmp 最新（ceph 上有更新），重新运行此脚本（rsync 会增量同步）。"
