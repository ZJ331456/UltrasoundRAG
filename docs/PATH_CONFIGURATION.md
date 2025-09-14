# 🛣️ 跨平台路径配置指南

## 📋 问题描述

之前的配置文件中硬编码了绝对路径：
```yaml
paths:
  project_root: "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG"  # ❌ 不可移植
```

这会导致在不同设备/环境中无法正常工作。

## ✅ 解决方案

新的智能路径配置支持三种方式，按优先级排序：

### 1. 🌍 环境变量 (最高优先级)
```bash
# Linux/macOS
export ULTRASOUNDRAG_PROJECT_ROOT="/your/custom/path"

# Windows
set ULTRASOUNDRAG_PROJECT_ROOT="C:\your\custom\path"
```

### 2. 🤖 自动检测 (推荐)
```yaml
paths:
  project_root: "auto"  # 智能检测项目根目录
```

### 3. 📁 绝对路径 (手动指定)
```yaml
paths:
  project_root: "/specific/absolute/path"
```

## 🚀 快速设置

### 方法1: 使用设置脚本 (推荐)
```bash
# 进入项目目录
cd UltrasoundRAG

# 运行环境设置脚本
python -m UltrasoundRAG.scripts.setup_environment

# 或指定环境
python -m UltrasoundRAG.scripts.setup_environment --environment production
```

### 方法2: 手动配置
```bash
# 1. 复制配置模板
cp UltrasoundRAG/config/config.template.yaml UltrasoundRAG/config/config.yaml

# 2. 编辑配置文件 (可选)
# 默认 "auto" 即可自动检测

# 3. 验证配置
python -m UltrasoundRAG.scripts.setup_environment --validate-only
```

## 🔍 自动检测机制

系统会按以下顺序查找项目根目录：

1. **环境变量**: `ULTRASOUNDRAG_PROJECT_ROOT`
2. **向上查找**: 从当前位置向父目录查找，直到找到包含以下文件/目录的位置：
   - `requirements.txt`
   - `setup.py`
   - `pyproject.toml`
   - `.git/`
   - `UltrasoundRAG/` 目录
   - `README.md`
3. **兜底方案**: 使用当前工作目录

## 🌐 跨平台兼容性

| 平台 | 支持状态 | 说明 |
|------|----------|------|
| Linux | ✅ 完全支持 | 自动检测和环境变量 |
| macOS | ✅ 完全支持 | 自动检测和环境变量 |
| Windows | ✅ 完全支持 | 路径分隔符自动处理 |
| Docker | ✅ 完全支持 | 容器内路径自动适配 |

## 📝 配置文件层级

支持多层级配置文件：

```
UltrasoundRAG/config/
├── config.template.yaml      # 配置模板
├── config.yaml              # 主配置文件
├── config.development.yaml  # 开发环境覆盖
├── config.testing.yaml      # 测试环境覆盖
└── config.production.yaml   # 生产环境覆盖
```

## 🛠️ 开发者工具

### 查看当前配置
```bash
python -m UltrasoundRAG.scripts.setup_environment --show-config
```

### 验证配置
```bash
python -m UltrasoundRAG.scripts.setup_environment --validate-only
```

### 生成环境变量脚本
```bash
python -m UltrasoundRAG.scripts.setup_environment --permanent-env
# 生成 set_env.sh (Linux/macOS) 和 set_env.bat (Windows)
```

## 🐳 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

# 环境变量会自动被配置系统识别
ENV ULTRASOUNDRAG_PROJECT_ROOT=/app

RUN pip install -r requirements.txt
CMD ["python", "frontend.py"]
```

## 🔧 故障排除

### 问题1: 配置文件找不到
```bash
# 检查配置文件位置
python -c "from UltrasoundRAG.config import config; print(config)"
```

### 问题2: 路径检测失败
```bash
# 手动设置环境变量
export ULTRASOUNDRAG_PROJECT_ROOT="$(pwd)"
```

### 问题3: 权限问题
```bash
# 确保配置目录有写权限
chmod 755 UltrasoundRAG/config/
```

## 📊 最佳实践

1. **开发环境**: 使用 `"auto"` 自动检测
2. **生产环境**: 使用环境变量明确指定
3. **Docker部署**: 使用环境变量 + 挂载卷
4. **团队协作**: 提供 `.env.example` 文件

```bash
# .env.example
ULTRASOUNDRAG_PROJECT_ROOT=/opt/ultrasound-rag
ULTRASOUNDRAG_CONFIG=/opt/ultrasound-rag/config/config.yaml
```

## 🎯 迁移指南

### 从旧版本迁移

1. **备份配置**:
   ```bash
   cp config.yaml config.yaml.backup
   ```

2. **更新配置**:
   ```bash
   python -m UltrasoundRAG.scripts.setup_environment
   ```

3. **验证功能**:
   ```bash
   python -m UltrasoundRAG.scripts.setup_environment --validate-only
   ```

### 团队同步

```bash
# 团队成员执行
git pull
python -m UltrasoundRAG.scripts.setup_environment --environment development
```

现在，你的项目可以在任何设备和环境中无缝运行！🎉
