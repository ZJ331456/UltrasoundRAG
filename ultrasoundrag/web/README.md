# UltrasoundRAG Vue 前端

## 概述

UltrasoundRAG Vue 前端是一个现代化的医学超声RAG系统管理界面，基于 Vue 3 + Ant Design Vue 构建，提供完整的系统管理和多模态检索功能。

## 功能特性

### 🎯 核心功能
- **智能检索**: 支持文本→文本、文本→图片、图片→文本、图片→图片、多模态检索
- **数据库管理**: 完整的数据库CRUD操作和文档管理
- **系统监控**: 实时系统状态监控和性能指标
- **用户友好**: 现代化UI设计，响应式布局

### 🛠 技术栈
- **前端框架**: Vue 3.5.18
- **UI组件库**: Ant Design Vue 4.0.0
- **状态管理**: Pinia 2.1.6
- **路由管理**: Vue Router 4.2.4
- **HTTP客户端**: Axios 1.5.0
- **构建工具**: Vite 7.0.6
- **图表可视化**: Chart.js + Vue-ChartJS

## 快速开始

### 环境要求
- Node.js >= 20.19.0 或 >= 22.12.0
- npm 或 yarn 包管理器

### 安装依赖

```bash
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/ultrasoundrag/web/rag-vue
npm install
```

### 开发模式

```bash
# 启动开发服务器
npm run dev

# 访问地址
http://localhost:5173
```

### 生产构建

```bash
# 构建生产版本
npm run build

# 预览生产版本
npm run preview
```

## 项目结构

```
src/
├── api/                    # API 服务层
│   └── index.js           # 统一 API 接口
├── stores/                 # Pinia 状态管理
│   └── system.js          # 系统状态管理
├── views/                  # 页面组件
│   ├── Dashboard.vue      # 系统仪表盘
│   ├── Search.vue         # 智能检索页面
│   ├── DatabaseManager.vue # 数据库管理
│   └── SystemStatus.vue   # 系统状态监控
├── components/             # 可复用组件
├── assets/                 # 静态资源
├── style.css              # 全局样式
├── App.vue                # 根组件
└── main.js                # 应用入口
```

## 页面功能说明

### 📊 系统仪表盘 (`/app/dashboard`)
- 系统状态概览
- 关键指标展示
- 快速操作入口
- 最近活动时间线

### 🔍 智能检索 (`/app/search`)
- 多种检索模式选择
- 文本查询输入
- 图片拖拽上传
- 高级选项配置
- 结果展示和评分

### 🗄️ 数据库管理 (`/app/database`)
- 数据库列表查看
- 新建数据库
- 编辑数据库配置
- 删除数据库
- 文档CRUD操作

### 📈 系统状态 (`/app/status`)
- 系统健康状态监控
- 服务状态检查
- 系统配置信息
- 实时日志查看

## API 集成

### 配置代理

开发环境下，Vite 自动将 `/api/*` 请求代理到后端服务：

```javascript
// vite.config.js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

### API 服务使用

```javascript
import { searchAPI, databaseAPI, systemAPI } from '@/api'

// 执行搜索
const result = await searchAPI.search({
  mode: 't2t',
  query: '心脏超声诊断',
  top_k: 10
})

// 管理数据库
const databases = await databaseAPI.list()
```

## 状态管理

使用 Pinia 进行状态管理：

```javascript
import { useSystemStore } from '@/stores/system'

const systemStore = useSystemStore()

// 检查系统健康状态
await systemStore.checkSystemHealth()

// 访问状态
const isHealthy = systemStore.isHealthy
const alertCount = systemStore.alertCount
```

## 样式定制

### CSS 变量
```css
:root {
  --primary-color: #1890ff;
  --secondary-color: #722ed1;
  --success-color: #52c41a;
  --warning-color: #faad14;
  --error-color: #f5222d;
}
```

### 响应式设计
- 移动端优先设计
- 断点: 768px (移动端), 1024px (桌面端)
- 弹性布局和网格系统

## 部署

### 构建生产版本
```bash
npm run build
```

### 部署到后端
构建完成后，`dist` 目录将被后端 FastAPI 服务自动识别并提供静态文件服务。

访问地址: `http://your-domain:8000/app`

## 开发指南

### 添加新页面
1. 在 `src/views/` 创建 Vue 组件
2. 在 `src/main.js` 添加路由配置
3. 在 `App.vue` 添加菜单项

### 添加 API 接口
1. 在 `src/api/index.js` 添加 API 方法
2. 更新相应的状态管理
3. 在组件中调用

### 状态管理最佳实践
- 使用 Pinia 进行全局状态管理
- 按模块划分不同的 store
- 使用 computed 进行派生状态

## 常见问题

### 1. 开发服务器无法访问后端 API
确保后端服务运行在 `http://localhost:8000` 并且 CORS 配置正确。

### 2. 构建后静态资源路径错误
检查 `vite.config.js` 中的 `base` 配置是否为 `/app/`。

### 3. 图片上传失败
确认上传的图片格式和大小符合要求（支持 JPG/PNG/JPEG，小于 10MB）。

## 许可证

本项目遵循与 UltrasoundRAG 主项目相同的许可证。