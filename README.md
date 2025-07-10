# UltrasoundRAG 医学RAG项目说明文档

## 目录

- [项目简介](#项目简介)
- [项目路径说明](#项目路径说明)
- [环境配置](#环境配置)
- [模型下载与准备](#模型下载与准备)
- [启动与使用](#启动与使用)
- [常见问题](#常见问题)
- [说明](#说明)

---

## 项目简介

本项目为超声领域的RAG（Retrieval-Augmented Generation）系统，支持文档与图片的混合检索、答案生成、评估等功能。适用于医学知识问答、图文检索等场景。  
目前的流程主要是建立索引之后，先去检索文本块，再根据找到的文本块里面去提取对应的图片标题，根据这个图片标题去获取本地对应的图片，根据文本块以及本地的图片传给DolphinUltrasound模型生成对应的答案。
---

## 项目路径说明

```
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
│   └── medicalrag.py         # 主程序入口
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

1. **MedicalRAG/medicalrag.py**: 系统主入口，包含交互式界面和主要功能实现
2. **MedicalRAG/config/**: 配置管理，控制系统行为和模型路径
3. **MedicalRAG/index/**: 负责文档和图像的索引创建
4. **MedicalRAG/retrival/**: 实现混合检索和图像查询功能
5. **MedicalRAG/utils/**: 提供各类工具函数，如答案生成、嵌入计算等
6. **MedicalRAG/eval/**: 提供系统评估功能

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

   ```
   models/
     ├── bge-small-zh-v1.5/
     ├── clip-vit-large-patch14/
     └── Taiyi-CLIP-Roberta-large-326M-Chinese/
   ```

2. **模型下载方式：**
   - 推荐从 [HuggingFace](https://huggingface.co/) 或 [ModelScope](https://modelscope.cn/) 搜索对应模型名称下载。
   - 下载后解压到 `models/` 目录下，保持上述结构。
   git clone https://huggingface.co/BAAI/bge-small-zh-v1.5
   git clone https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese
   git clone https://huggingface.co/openai/clip-vit-large-patch14

3. **如需自定义模型路径，请在 `MedicalRAG/config/config.yaml` 中修改相关配置。**


---

## 启动与使用

### 1. **交互式命令行模式**

```bash
python -m MedicalRAG/medicalrag
```

- 按提示选择是否重建索引、运行模式（交互/文件/默认测试）。
- 交互模式下可直接输入问题进行检索与问答。

### 2. **批量问答/评估**

- 将问题列表写入 json 文件（如 `data/truth_query.json`），选择"文件模式"运行。

---

## 常见问题

- **模型未下载/路径错误**：请检查 `models/` 目录结构和 `config.yaml` 配置。
- **依赖冲突/缺包**：请确保已激活虚拟环境并正确安装 requirements.txt。
- **索引重建慢/内存占用高**：建议在内存充足的环境下运行，或分批处理数据。

---

> **说明：本项目目前为初步搭建的框架，后续将持续优化整体代码结构，并根据评估性能有针对性地优化各个模块以提升系统表现。**


