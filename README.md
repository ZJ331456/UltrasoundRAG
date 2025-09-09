# UltrasoundRAG 医学RAG项目说明文档

## 目录

- [项目简介](#项目简介)
- [项目路径说明](#项目路径说明)
- [环境配置](#环境配置)
- [模型下载与准备](#模型下载与准备)
- [启动与使用](#启动与使用)
- [API接口说明](#api接口说明)
- [模型优化特性](#模型优化特性)
- [常见问题](#常见问题)
- [故障排除](#故障排除)

---

## 项目简介

本项目为超声领域的RAG（Retrieval-Augmented Generation）系统，支持文档与图片的混合检索、答案生成、评估等功能。适用于医学知识问答、图文检索等场景。

目前的流程主要是建立索引之后，先去检索文本块，再根据找到的文本块里面去提取对应的图片标题，根据这个图片标题去获取本地对应的图片，根据文本块以及本地的图片传给DolphinUltrasound模型生成对应的答案。

---

## 项目路径说明

```text
UltrasoundRAG/
├── data/                # 数据目录：原始数据、向量库和处理后的数据
│   ├── example_book_vectorstore/      # 文本向量数据库（ChromaDB格式）
│   ├── example_image_vectorstore_clip_taiyi/ # 图像向量数据库（基于CLIP和太一模型）
│   ├── processed/            # 处理后的文本和图像数据
│   │   ├── image/            # 处理后的医学图像
|   |   |——01_超声标准切面图解.md  #文本数据
│   │   └── ...               # 其他处理后的文本文件
├── docs/                # 文档目录==>后续完善
│   └── ragas评估指标说明.md   # RAG评估指标的详细说明
├── MedicalRAG/          # 核心代码目录
│   ├── config/               # 配置文件目录
│   │   ├── config.py         # 配置加载模块
│   │   ├── config.yaml       # 主配置文件
│   │   └── logger.yaml       # 日志配置
│   ├── eval/                 # 评估模块
│   │   └── rag_eval.py       # RAG系统评估工具
│   ├── index/                # 索引创建模块
│   │   ├── custom_document_parser.py  # 自定义文档解析器
│   │   ├── document_loader.py         # 文档加载器
│   │   ├── index_images.py            # 图像索引工具
│   │   └── run_custom_indexer.py      # 自定义索引器运行脚本
│   ├── retrival/             # 检索模块
│   │   ├── hybrid_retrival.py         # 混合检索实现
│   │   └── query_images.py            # 图像查询工具
│   ├── utils/                # 工具函数
│   │   ├── answer_generator.py        # 答案生成器
│   │   ├── embedding_utils.py         # 嵌入工具
│   │   ├── image_utils.py             # 图像处理工具
│   │   ├── llm_utils.py               # 语言模型工具
│   │   ├── logger.py                  # 日志工具
│   │   ├── prompt.py                  # 提示词模板
│   │   ├── rerank_utils.py            # 重排序工具
│   │   └── retrival_utils.py          # 检索工具函数
│   ├── __init__.py           # 包初始化文件
│   ├── medicalrag.py         # 主程序入口
│   └── streamlit_app.py      # Streamlit Web UI 入口
├── models/              # 模型目录（不上传到Git）
│   ├── bge-m3/               # BGE-M3嵌入模型
│   ├── bge-small-zh-v1.5/    # BGE中文小型嵌入模型
│   ├── Bunny-v1_0-3B/        # Bunny语言模型
│   ├── clip-ViT-B-32/        # CLIP视觉-文本模型
│   ├── clip-vit-large-patch14/ # CLIP大型模型
│   ├── MiniCPM-V-2/          # MiniCPM视觉模型V2
│   ├── MiniCPM-V-2_6/        # MiniCPM视觉模型V2.6
│   ├── Qwen3-Embedding-0.6B/ # 通义千问嵌入模型
│   ├── Qwen3-Reranker-0.6B/  # 通义千问重排序模型
│   ├── siglip-so400m-patch14-384/ # SigLIP视觉模型
│   └── Taiyi-CLIP-Roberta-large-326M-Chinese/ # 太一中文CLIP模型
├── results/             # 结果输出目录（不上传到Git）
│   ├── eval/                 # 评估结果
│   └── retrival/             # 检索结果
│   
├── environment.yaml     # Conda环境配置文件
├── README.md            # 项目说明文档
└── requirements.txt     # 依赖包列表
```

### 核心模块说明

1. **streamlit_app.py**: 基于 Streamlit 的 Web UI 界面，提供友好的交互式问答、图片检索和索引管理功能。
2. **MedicalRAG/medicalrag.py**: 系统核心逻辑和命令行入口，包含交互式界面和主要功能实现
3. **MedicalRAG/config/**: 配置管理，控制系统行为和模型路径
4. **MedicalRAG/index/**: 负责文档和图像的索引创建
5. **MedicalRAG/retrival/**: 实现混合检索和图像查询功能
6. **MedicalRAG/utils/**: 提供各类工具函数，如答案生成、嵌入计算等
7. **MedicalRAG/eval/**: 提供系统评估功能

### 数据目录说明

1. **data/book_vectorstore/**: 存储文本向量数据库
2. **data/image_vectorstore_clip_taiyi/**: 存储图像向量数据库
3. **data/processed/**: 存储处理后的文本和图像

---

## 环境配置

1. **建议使用 Python 3.10+，推荐使用虚拟环境（如conda/venv）隔离依赖。**

2. **安装依赖包：**

   方式1：

   ```bash
   # 创建环境
   conda create -n ht-rag python=3.10

   # 激活环境
   conda activate ht-rag

   # 安装依赖
   pip install -r requirements.txt
   ```

   方式2：

   ```bash
   # 用 yaml 文件创建新环境
   conda env create -f environment.yaml

   # 激活环境
   conda activate ultrasoundrag
   ```

---

## 模型下载与准备

1. **本项目的模型文件已放在 `models/` 文件夹下。**  
   若首次运行或模型缺失，请手动下载所需模型并解压到 `models/` 目录，结构如下：

   ```text
   models/
     ├── bge-small-zh-v1.5/
     ├── clip-vit-large-patch14/
     └── Taiyi-CLIP-Roberta-large-326M-Chinese/
   ```

2. **模型下载方式：**
   - 推荐从 [HuggingFace](https://huggingface.co/) 或 [ModelScope](https://modelscope.cn/) 搜索对应模型名称下载。
   - 下载后解压到 `models/` 目录下，保持上述结构。

   ```bash
   git clone https://huggingface.co/BAAI/bge-small-zh-v1.5
   git clone https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese
   git clone https://huggingface.co/openai/clip-vit-large-patch14
   ```

3. **如需自定义模型路径，请在 `MedicalRAG/config/config.yaml` 中修改相关配置。**

---

## 启动与使用

### 1. **Web UI 模式 (推荐)**

通过 Streamlit 启动一个可视化的交互界面，支持文本问答、图片上传检索和在线重建索引等功能。

```bash
# 在项目根目录下运行
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG
python frontend.py
```

然后在浏览器中访问 `http://localhost:8501`

- **功能介绍**:
  - **问答查询**: 在左侧输入框输入问题，点击"生成答案"即可。
  - **图片检索**: 支持上传图片，系统会结合文本问题和图片内容进行联合检索。
  - **索引管理**: 在"高级管理选项"中，可以一键重建知识库索引。

### 2. **前端测试界面**

#### 支持的检索模式

1. **T2T (文本→文本)** - 文本知识检索
2. **T2I (文本→图片)** - 根据描述找图片
3. **I2T (图片→文本)** - 上传图片找相关文本
4. **I2I (图片→图片)** - 上传图片找相似图片
5. **Caption (标题→图片)** - 精确标题匹配
6. **混合检索** - 多种模式同时执行

#### 测试用例

**T2T测试：**

- "心脏超声检查方法"
- "肝脏病变诊断"
- "胎儿发育评估"

**T2I测试：**

- "心脏四腔心切面图"
- "肝脏超声图像"
- "胎儿发育图"

**Caption测试：**

- "图2-3 心脏超声横切面"
- "图1-1 肝脏超声检查"
- "Figure 3.2 胎儿发育图"

### 3. **交互式命令行模式**

```bash
python -m MedicalRAG/medicalrag
```

- 按提示选择是否重建索引、运行模式（交互/文件/默认测试）。
- 交互模式下可直接输入问题进行检索与问答。

### 4. **批量问答/评估**

- 将问题列表写入 json 文件（如 `data/truth_query.json`），选择"文件模式"运行。

---

## API接口说明

### 启动 API 服务

```bash
python start_api.py
```

服务启动后访问 `http://localhost:8000/docs` 查看交互式文档（Swagger UI），`/redoc` 查看 Redoc。

### 基础信息

- 基础URL: `http://localhost:8000/api/v1`
- 文档地址: `/docs`, `/redoc`
- 认证方式: Bearer Token，请在请求头中携带：

```bash
Authorization: Bearer rag_demo_key
```

### 核心接口

- `POST /api/v1/rag/search`：统一检索接口（支持文本/多模态、单条或批量、可控检索与生成）
- `GET  /api/v1/rag/status`：系统状态检查
- `GET  /api/v1/rag/config`：获取当前配置
- `GET  /api/v1/rag/models`：模型列表
- `GET  /health`：健康检查
- `GET  /`：根信息

### 快速示例

- 单个文本查询（Form）

```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Authorization: Bearer rag_demo_key" \
  -H "Content-Type: multipart/form-data" \
  -F "query=胎儿心脏超声检查的正常表现是什么？" \
  -F "top_k=10" \
  -F "enable_generate=true"
```

- 多模态查询（推荐：multipart + payload JSON）

```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Authorization: Bearer rag_demo_key" \
  -H "Content-Type: multipart/form-data" \
  -F "payload={\"query\":\"这张超声图像显示了什么？\",\"top_k\":5,\"enable_generate\":true}" \
  -F "image=@/path/to/ultrasound.jpg"
```

- 批量查询（Form，`queries` 为 JSON 字符串）

```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Authorization: Bearer rag_demo_key" \
  -H "Content-Type: multipart/form-data" \
  -F 'queries=["胎儿心脏超声检查的正常表现是什么？", "如何判断胎儿发育是否正常？"]' \
  -F "top_k=5"
```

更多参数、响应示例与错误说明请参见 `test/API_Documentation.md`。

---

## 模型优化特性

### 性能优化

本项目实现了多项模型加载优化，显著提升系统性能：

#### 1. 单例模式实现

- **FetalCLIPModel单例模式**：防止重复模型加载
- **全局模型管理器**：统一管理所有模型实例
- **智能预加载策略**：按需加载，减少启动时间

#### 2. 性能提升效果

- **模型加载时间**：从15-20秒减少到2-4秒
- **内存使用**：减少约80%的模型内存占用
- **初始化速度**：提升约75%

#### 3. 技术特点

- **单例模式**：确保模型只加载一次，全局共享
- **延迟加载**：按需加载模型，减少资源消耗
- **智能缓存**：模型实例缓存，自动清理机制
- **状态监控**：实时模型状态，性能指标显示

#### 4. 使用方式

```python
from UltrasoundRAG.model.model_manager import get_fetal_clip_model, get_embedding_model

# 获取模型实例（自动单例）
clip_model = get_fetal_clip_model()
embedding_model = get_embedding_model()
```

## 常见问题

- **模型未下载/路径错误**：请检查 `models/` 目录结构和 `config.yaml` 配置。
- **依赖冲突/缺包**：请确保已激活虚拟环境并正确安装 requirements.txt。
- **索引重建慢/内存占用高**：建议在内存充足的环境下运行，或分批处理数据。
- **模型加载慢**：系统已优化模型加载，使用单例模式避免重复加载。
- **前端启动问题**：确保在项目根目录运行 `python frontend.py`。

## 故障排除

如果遇到导入错误，请检查：

1. 项目路径是否正确
2. 依赖包是否已安装
3. 配置文件是否存在
4. Milvus服务是否正在运行

---

> **说明：本项目已实现模型加载优化，具有更快的启动速度、更低的内存占用和更好的用户体验。系统采用单例模式和智能缓存机制，确保高效稳定的性能表现。**
