#!/bin/bash
# FastWAM项目更新脚本
# 从GitHub拉取最新更新

echo "=== FastWAM项目更新 ==="

# 1. 检查当前状态
echo "1. 检查当前状态..."
git status

# 2. 检查远程更新
echo "2. 检查远程更新..."
git fetch origin

# 3. 比较本地和远程差异
echo "3. 比较本地和远程差异..."
git log HEAD..origin/main --oneline

# 4. 询问是否继续
read -p "是否拉取更新？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "更新取消"
    exit 1
fi

# 5. 拉取更新
echo "4. 拉取更新..."
git pull origin main

echo "=== 更新完成 ==="
echo "更新内容："
git log --oneline -5
echo ""
echo "当前状态："
git status