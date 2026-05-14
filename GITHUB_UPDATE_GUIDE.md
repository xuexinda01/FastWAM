# FastWAM GitHub更新指南

## 📋 快速更新命令

### 基本更新（无本地修改）
```bash
# 直接拉取最新更新
git pull origin main
```

### 安全更新（有本地修改）
```bash
# 1. 检查本地修改
git status

# 2. 暂存本地修改
git stash

# 3. 拉取更新
git pull origin main

# 4. 恢复本地修改
git stash pop

# 5. 解决冲突（如果有）
# 手动编辑冲突文件后：
git add <冲突文件>
git commit -m "解决合并冲突"
```

## 🛡️ 更新策略

### 场景1：仅查看更新（不修改本地文件）
```bash
# 查看远程有什么更新
git fetch origin
git log HEAD..origin/main --oneline
```

### 场景2：安全更新（推荐）
```bash
# 使用更新脚本
./update_from_github.sh
```

### 场景3：创建功能分支更新
```bash
# 1. 创建功能分支
git checkout -b feature-branch

# 2. 拉取主分支更新
git checkout main
git pull origin main

# 3. 合并到功能分支
git checkout feature-branch
git merge main
```

## 🔍 更新前检查

### 检查远程状态
```bash
# 查看远程分支状态
git remote show origin

# 查看本地与远程差异
git status
git log HEAD..origin/main --oneline
```

### 检查本地修改
```bash
# 查看未提交的修改
git status

# 查看具体的文件修改
git diff
```

## ⚠️ 冲突处理

### 识别冲突
```bash
# 查看冲突文件
git status

# 查看冲突内容
git diff
```

### 解决冲突
1. **手动编辑冲突文件**（搜索`<<<<<<<`标记）
2. **选择保留的代码**
3. **删除冲突标记**
4. **标记为已解决**
```bash
git add <冲突文件>
git commit -m "解决合并冲突"
```

## 🔄 完整更新流程

### 步骤1：检查状态
```bash
git status
git fetch origin
git log HEAD..origin/main --oneline
```

### 步骤2：备份本地修改（可选）
```bash
# 创建备份分支
git checkout -b backup-$(date +%Y%m%d)

# 回到主分支
git checkout main
```

### 步骤3：拉取更新
```bash
git pull origin main
```

### 步骤4：验证更新
```bash
# 查看更新内容
git log --oneline -5

# 检查文件状态
git status

# 运行测试（如果有）
python -m pytest tests/ -v
```

## 🚨 注意事项

### 更新前备份
- **重要文件备份**：configs/, scripts/, src/等关键目录
- **数据文件注意**：checkpoints/, runs/, logs/等不会被Git跟踪

### 冲突预防
- **频繁更新**：定期拉取更新避免大冲突
- **功能分支**：在功能分支开发，定期合并主分支
- **沟通协调**：与团队成员协调修改

### 认证配置
如果提示认证失败：
```bash
# 设置Git凭据
git config --global credential.helper store
# 或使用个人访问令牌
git config --global user.name "xuexinda01"
git config --global user.email "your-email@example.com"
```

## 📞 故障排除

### 常见问题
1. **认证失败**：检查GitHub令牌或密码
2. **网络问题**：检查代理设置
3. **冲突无法解决**：使用`git merge --abort`取消合并

### 紧急回滚
```bash
# 回滚到上一个提交
git reset --hard HEAD~1

# 回滚到特定提交
git reset --hard <commit-hash>

# 强制推送（谨慎使用）
git push --force-with-lease origin main
```

---

**快速开始**：运行 `./update_from_github.sh` 或直接执行 `git pull origin main`