# FastWAM GitHub上传指南

## 概述

这个指南帮助你安全地将FastWAM项目上传到GitHub，同时保护敏感数据和训练结果。

## 🎯 上传策略

### ✅ 上传内容
- **源代码**: `src/fastwam/` - 核心模型和训练代码
- **配置文件**: `configs/`, `config/` - 训练和实验配置
- **脚本**: `scripts/` - 训练和预处理脚本
- **实验代码**: `experiments/` - 评估和实验管理
- **文档**: 所有`.md`文档文件
- **项目配置**: `pyproject.toml`, `LICENSE`, `__init__.py`

### ❌ 排除内容（通过.gitignore）
- **数据文件**: `checkpoints/`, `data/`, `datasets/`
- **训练结果**: `runs/`, `logs/`, `evaluate_results/`
- **缓存文件**: `text_embeds_cache/`, `__pycache__/`
- **日志文件**: 所有`.log`文件
- **临时文件**: `.vscode/`, `.claude/`

## 📋 上传步骤

### 步骤1：准备项目
```bash
# 运行准备脚本
chmod +x prepare_for_github.sh
./prepare_for_github.sh
```

### 步骤2：初始化Git仓库
```bash
cd /apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM
git init
git add .
git commit -m "Initial FastWAM release: World Action Model training framework"
```

### 步骤3：创建GitHub仓库
1. 登录GitHub
2. 点击右上角"+" → "New repository"
3. 仓库名: `FastWAM`
4. 描述: "FastWAM: Do World Action Models Need Test-time Future Imagination?"
5. 选择"Public"或"Private"
6. 不勾选"Add a README file"（已有）

### 步骤4：推送到GitHub
```bash
git remote add origin https://github.com/your-username/FastWAM.git
git branch -M main
git push -u origin main
```

## 🔍 上传前检查清单

### 验证.gitignore配置
确保以下内容在.gitignore中：
```
checkpoints/
checkpoints/*
data/*
runs/
runs/*
logs/
logs/*
text_embeds_cache/
text_embeds_cache/*
*.log
*.log.*
__pycache__/
.vscode/
```

### 验证符号链接处理
运行准备脚本后，确保以下符号链接被替换为目录：
- `data/` → 空目录 + .gitkeep
- `datasets/` → 空目录 + .gitkeep  
- `evaluate_results/` → 空目录 + .gitkeep
- `experiments/` → 空目录 + .gitkeep
- `RoboTwin/` → 空目录 + .gitkeep

### 验证核心文件
确保以下关键文件存在：
- `README.md` - 项目文档
- `pyproject.toml` - 依赖配置
- `src/fastwam/` - 核心代码
- `configs/` - 配置文件
- `scripts/` - 训练脚本

## 📁 项目结构（上传后）

```
FastWAM/
├── src/fastwam/              # 核心代码
│   ├── models/               # 模型定义
│   ├── datasets/             # 数据集处理
│   ├── utils/                # 工具函数
│   └── trainer.py            # 训练器
├── configs/                  # 配置文件
│   ├── data/                 # 数据集配置
│   ├── model/                # 模型配置
│   └── task/                 # 任务配置
├── scripts/                  # 训练脚本
├── experiments/              # 实验代码（空目录）
├── data/                     # 数据目录（空目录）
├── datasets/                 # 数据集目录（空目录）
├── README.md                 # 项目文档
├── pyproject.toml            # 依赖配置
└── .gitignore                # Git忽略规则
```

## 🔒 安全注意事项

### 确保不上传的内容
- **模型权重**: 所有`.pt`, `.pth`, `.bin`文件
- **训练数据**: 任何原始数据文件
- **日志文件**: 训练和评估日志
- **缓存文件**: 预计算的嵌入缓存

### 敏感信息检查
上传前运行检查：
```bash
# 检查是否有敏感信息
grep -r "password\|secret\|key\|token" . --include="*.py" --include="*.yaml" --include="*.toml"
```

## 📚 用户使用指南

### 新用户如何开始
1. 克隆仓库
2. 按照README.md中的环境设置步骤
3. 从Hugging Face下载预训练模型和数据
4. 运行训练或评估脚本

### 数据下载说明
在README.md中明确说明：
- 模型权重从Hugging Face下载
- 数据集从指定链接下载
- 提供详细的下载和设置步骤

## 🚀 后续维护

### 版本管理
- 使用语义化版本号
- 为重要功能创建release
- 维护CHANGELOG.md

### 协作开发
- 使用feature分支
- 通过Pull Request合并代码
- 设置CI/CD进行自动化测试

## 📞 技术支持

如果遇到上传问题：
1. 检查.gitignore配置
2. 验证符号链接处理
3. 确保没有大文件（>100MB）
4. 检查网络连接

---

**注意**: 这个指南确保只上传代码和文档，保护训练数据和模型权重。用户需要从外部源下载数据和模型。