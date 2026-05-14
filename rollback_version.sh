#!/bin/bash
# FastWAM版本回退脚本
# 安全回退到指定版本

echo "=== FastWAM版本回退 ==="

# 1. 显示当前提交历史
echo "当前提交历史："
git log --oneline -5
echo ""

# 2. 显示远程状态
echo "远程状态："
git status
echo ""

# 3. 询问目标版本
read -p "请输入要回退到的提交哈希（或输入数字回退几个提交，如1）： " target

# 处理数字输入
if [[ $target =~ ^[0-9]+$ ]]; then
    target="HEAD~$target"
fi

# 4. 验证目标版本
echo "目标版本信息："
git log --oneline $target -1
echo ""

# 5. 确认回退
read -p "确认回退到上述版本？这将删除后续提交。(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "回退取消"
    exit 1
fi

# 6. 创建备份
echo "创建备份..."
git tag "backup-rollback-$(date +%Y%m%d-%H%M)"
git branch "backup-before-rollback-$(date +%Y%m%d-%H%M)"

# 7. 执行回退
echo "执行回退..."
git reset --hard $target

# 8. 显示结果
echo "回退完成！"
echo "当前版本："
git log --oneline -1
echo ""

# 9. 询问是否推送
read -p "是否推送到远程仓库？(谨慎操作！)(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "推送更新..."
    git push --force-with-lease origin main
    echo "推送完成"
else
    echo "未推送，仅本地回退"
fi

echo "=== 回退完成 ==="
echo "备份信息："
git tag -l "backup-rollback-*"
git branch -l "backup-before-rollback-*"