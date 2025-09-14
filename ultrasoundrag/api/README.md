# UltrasoundRAG API模块

## 概述

本模块提供基于FastAPI的HTTP REST接口（`api.py`）。

## Web API

### 启动方式

#### 方式1: 直接运行模块
```bash
# 在项目根目录下
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG
python -m ultrasoundrag.api
```

#### 方式2: 使用uvicorn
```bash
uvicorn ultrasoundrag.api.api:app --host 0.0.0.0 --port 8000
```

#### 方式3: 直接运行文件
```bash
python ultrasoundrag/api/api.py
```

### API端点

#### 基础端点
- `GET /` - 服务信息
- `GET /health` - 健康检查
- `GET /docs` - API文档 (Swagger)

#### 功能端点
- `GET /databases` - 列出可用数据库
- `POST /search` - 统一搜索接口
- `GET /system/status` - 系统状态

#### 管理端点
- `POST /admin/cache/clear` - 清空缓存
- `POST /admin/models/reload` - 重载模型
- `GET /admin/alerts` - 获取告警

### 搜索请求示例

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "t2t",
    "query": "心脏超声检查方法",
    "top_k": 5,
    "db_name": "default"
  }'
```

### 响应格式

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "doc_id": "doc_123",
        "score": 0.95,
        "content": "心脏超声检查内容...",
        "metadata": {}
      }
    ],
    "total_results": 5
  },
  "metadata": {
    "cached": false,
    "response_time": 0.234
  }
}
```

## 配置

### 环境变量

- `ALLOWED_ORIGINS` - CORS允许的源地址 (默认: http://localhost:8501,http://127.0.0.1:8501)
- `ALLOWED_HOSTS` - 允许的主机 (默认: localhost,127.0.0.1,*.localhost)

### 依赖容错

Web API支持依赖容错模式：
- 如果高级功能模块不可用，将使用基础模式运行
- 核心检索功能保持可用
- 监控和缓存功能可选启用

## 故障排除

### 常见问题

1. **模块导入失败**
   - 检查Python路径和依赖安装
   - 使用基础模式运行

2. **端口占用**
   ```bash
   # 指定其他端口
   python -m ultrasoundrag.api web --port 8001
   ```

3. **依赖缺失**
   ```bash
   pip install fastapi uvicorn
   ```

### 日志查看

Web API启动后会显示访问地址和文档链接，通过浏览器访问 `http://localhost:8000/docs` 查看完整API文档。
