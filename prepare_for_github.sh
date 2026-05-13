#!/bin/bash
# FastWAM GitHub上传准备脚本
# 这个脚本会清理符号链接并创建适合GitHub上传的目录结构

echo "准备FastWAM项目上传到GitHub..."

# 1. 删除符号链接（这些指向外部路径，GitHub无法使用）
echo "删除符号链接..."
rm -f data datasets evaluate_results experiments RoboTwin

# 2. 创建必要的空目录结构（保留目录结构）
echo "创建必要的目录结构..."
mkdir -p data datasets evaluate_results experiments RoboTwin

# 3. 添加.gitkeep文件到空目录（确保Git跟踪这些目录）
echo "添加.gitkeep文件..."
touch data/.gitkeep
touch datasets/.gitkeep
touch evaluate_results/.gitkeep
touch experiments/.gitkeep
touch RoboTwin/.gitkeep

# 4. 检查.gitignore是否包含所有需要排除的内容
echo "检查.gitignore配置..."
if grep -q "checkpoints" .gitignore && grep -q "runs" .gitignore && grep -q "logs" .gitignore; then
    echo "✓ .gitignore配置正确"
else
    echo "⚠ 请检查.gitignore配置"
fi

# 5. 验证关键文件存在
echo "验证关键文件..."
if [ -f "README.md" ] && [ -f "pyproject.toml" ] && [ -d "src/fastwam" ]; then
    echo "✓ 核心文件存在"
else
    echo "⚠ 缺少关键文件"
fi

echo ""
echo "GitHub上传准备完成！"
echo "接下来可以执行："
echo "1. git init"
echo "2. git add ."
echo "3. git commit -m 'Initial FastWAM release'"
echo "4. 在GitHub创建新仓库"
echo "5. git remote add origin <github-repo-url>"
echo "6. git push -u origin main"