# UltrasoundRAG v3.0 集成系统使用指南

## 🎯 系统概览

UltrasoundRAG v3.0 是一个完全重构的企业级医学超声RAG系统，提供：

- **统一服务接口**: 完整的数据库CRUD和多模态检索功能
- **FastAPI后端**: 企业级REST API服务
- **Vue 3前端**: 现代化管理界面
- **多模态检索**: 支持文本、图像和混合查询

## 🚀 快速启动

### 1. 检查代码状态

所有主要代码已修复和集成：

✅ **main.py**: 统一功能入口，支持所有核心功能
✅ **api.py**: 完整的REST API，支持CRUD和检索
✅ **Vue前端**: 现代化界面，包含所有管理功能

### 2. 启动后端服务

```bash
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG

# 方法1: 使用集成命令
python -m ultrasoundrag serve api

# 方法2: 直接启动API
python -m ultrasoundrag.api

# 方法3: 使用uvicorn
uvicorn ultrasoundrag.api.api:app --host 0.0.0.0 --port 8000
```

### 3. 安装前端依赖

```bash
cd ultrasoundrag/web/rag-vue
npm install
```

### 4. 启动前端开发服务器

```bash
# 开发模式
npm run dev

# 访问: http://localhost:5173
```

### 5. 构建生产版本

```bash
# 构建前端
npm run build

# 构建完成后，访问: http://localhost:8000/app
```

## 📡 API 端点

### 基础信息
- **根路径**: `http://localhost:8000`
- **API文档**: `http://localhost:8000/docs`
- **健康检查**: `http://localhost:8000/health`

### 主要端点

#### 检索功能
```bash
# 统一搜索
POST /search
{
  "mode": "t2t|t2i|i2t|i2i|auto|multimodal",
  "query": "查询文本",
  "db_name": "default",
  "top_k": 10
}

# 多数据库搜索
POST /search/multi-database
{
  "databases": ["db1", "db2"],
  "mode": "t2t",
  "query": "查询内容"
}

# 文件上传搜索
POST /search/upload
Content-Type: multipart/form-data
```

#### 数据库管理
```bash
# 列出数据库
GET /databases

# 创建数据库
POST /databases
{
  "name": "database_name",
  "config": {
    "db_name": "database_name",
    "description": "数据库描述",
    "enabled": true
  }
}

# 更新数据库
PUT /databases/{db_name}

# 删除数据库
DELETE /databases/{db_name}
```

#### 文档管理
```bash
# 添加文档
POST /databases/{db_name}/collections/{collection}/documents

# 获取文档
GET /databases/{db_name}/collections/{collection}/documents/{doc_id}

# 更新文档
PUT /databases/{db_name}/collections/{collection}/documents/{doc_id}

# 删除文档
DELETE /databases/{db_name}/collections/{collection}/documents/{doc_id}

# 列出文档
GET /databases/{db_name}/collections/{collection}/documents
```

#### 系统管理
```bash
# 系统状态
GET /system/status

# 重载配置
POST /system/reload-config

# 构建索引
POST /indexes/build
{
  "target": "all|markdown|image",
  "recreate": false
}

# 运行测试
POST /tests/run
{
  "test_type": "retrieval|benchmark|stress",
  "params": {}
}
```

## 💻 前端功能

### 页面结构
- **仪表盘** (`/app/dashboard`): 系统概览和快速操作
- **智能检索** (`/app/search`): 多模态检索界面
- **数据库管理** (`/app/database`): 数据库和文档管理
- **系统状态** (`/app/status`): 系统监控和配置

### 检索模式说明
1. **T2T (文本→文本)**: 根据文本查询返回相关文本
2. **T2I (文本→图片)**: 根据文本查询返回相关图片
3. **I2T (图片→文本)**: 根据图片查询返回相关文本
4. **I2I (图片→图片)**: 根据图片查询返回相似图片
5. **Auto (自动模式)**: 智能选择最佳检索策略
6. **Multimodal (多模态)**: 融合多种检索策略

## 🔧 统一服务接口

### Python API使用

```python
from ultrasoundrag import get_service

# 获取服务实例
service = get_service()

# 执行搜索
result = service.search(
    query="心脏超声诊断",
    mode="t2t",
    db_name="default",
    top_k=10
)

# 管理数据库
databases = service.list_databases()
service.create_database("new_db", config)

# 构建索引
service.build_indexes("all", recreate=True)

# 运行测试
service.run_tests("retrieval")
```

### 命令行使用

```bash
# 快速开始
python -m ultrasoundrag

# 启动服务
python -m ultrasoundrag serve api     # API服务
python -m ultrasoundrag serve web     # Web界面
python -m ultrasoundrag serve all     # 所有服务

# 管理功能
python -m ultrasoundrag build all --recreate  # 构建索引
python -m ultrasoundrag test retrieval        # 运行测试
python -m ultrasoundrag demo                  # 演示功能
```

## 🛠 开发和部署

### 开发环境
1. 确保Python依赖已安装
2. 启动后端API服务
3. 启动Vue开发服务器
4. 通过代理访问API

### 生产部署
1. 构建Vue前端: `npm run build`
2. 启动FastAPI服务
3. 访问: `http://your-domain:8000/app`

### 依赖要求

#### 后端依赖
```txt
fastapi
uvicorn
pydantic
# ... 其他UltrasoundRAG依赖
```

#### 前端依赖
```json
{
  "vue": "^3.5.18",
  "ant-design-vue": "^4.0.0",
  "vue-router": "^4.2.4",
  "pinia": "^2.1.6",
  "axios": "^1.5.0"
}
```

## 🔍 测试和验证

### 1. 后端API测试
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"mode":"t2t","query":"test","top_k":5}'
```

### 2. 前端功能测试
1. 访问 `http://localhost:5173`
2. 测试各个页面功能
3. 验证API调用正常

### 3. 集成测试
```python
# 运行系统测试
python -c "
from ultrasoundrag import get_service
service = get_service()
result = service.run_tests('retrieval')
print('测试结果:', result)
"
```

## 📚 故障排除

### 常见问题

1. **API服务启动失败**
   - 检查uvicorn是否安装: `pip install uvicorn`
   - 确认端口8000未被占用

2. **前端无法访问API**
   - 检查CORS配置
   - 确认代理配置正确
   - 验证API服务运行状态

3. **检索功能异常**
   - 检查数据库连接
   - 确认索引已构建
   - 查看后端日志

4. **依赖导入失败**
   - 确认所有依赖已安装
   - 检查Python路径配置
   - 验证模块结构

### 日志查看
- API日志: 控制台输出
- 前端日志: 浏览器开发者工具
- 系统状态: `/app/status` 页面

## 🚀 下一步

这个集成系统已经提供了完整的功能框架，您可以：

1. **扩展检索功能**: 添加更多检索模式和算法
2. **增强UI界面**: 自定义样式和组件
3. **优化性能**: 添加缓存和批处理
4. **集成监控**: 添加更详细的系统监控
5. **安全增强**: 添加认证和权限管理

## 📞 支持

如果遇到问题，请：
1. 查看API文档: `http://localhost:8000/docs`
2. 检查系统状态: `http://localhost:8000/app/status`
3. 查看错误日志和控制台输出

---

**UltrasoundRAG v3.0 - 企业级医学超声RAG系统** 🩺
