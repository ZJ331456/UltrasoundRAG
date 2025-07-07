下面是为你的项目量身定制的**简明中文说明文档**，涵盖环境配置、模型下载、项目启动等关键步骤。你可以将其保存为 `README.md` 放在项目根目录。

---

# UltrasoundRAG 医学RAG项目说明文档

## 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [模型下载与准备](#模型下载与准备)
- [项目结构说明](#项目结构说明)
- [启动与使用](#启动与使用)
- [常见问题](#常见问题)

---

## 项目简介

本项目为医学领域的RAG（Retrieval-Augmented Generation）系统，支持文档与图片的混合检索、答案生成、评估等功能。适用于医学知识问答、图文检索等场景。

---

## 环境配置

1. **建议使用 Python 3.10+，推荐使用虚拟环境（如conda/venv）隔离依赖。**

2. **安装依赖包：**

   ```bash
   # 创建并激活虚拟环境（可选）
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate

   # 安装依赖
   pip install -r requirements.txt
   ```

3. **如需 Jupyter/Spyder/Notebook 支持，可自行安装相关包。**

---

## 模型下载与准备

1. **本项目的模型文件已放在 `models/` 文件夹下。**  
   若首次运行或模型缺失，请手动下载所需模型并解压到 `models/` 目录，结构如下：

   ```
   models/
     ├── bge-m3/
     ├── bge-small-zh-v1.5/
     ├── Bunny-v1_0-3B/
     ├── clip-ViT-B-32/
     ├── clip-vit-large-patch14/
     ├── MiniCPM-V-2/
     ├── MiniCPM-V-2_6/
     ├── Qwen3-Embedding-0.6B/
     ├── Qwen3-Reranker-0.6B/
     ├── siglip-so400m-patch14-384/
     └── Taiyi-CLIP-Roberta-large-326M-Chinese/
   ```

2. **模型下载方式：**
   - 推荐从 [HuggingFace](https://huggingface.co/) 或 [ModelScope](https://modelscope.cn/) 搜索对应模型名称下载。
   - 下载后解压到 `models/` 目录下，保持上述结构。

3. **如需自定义模型路径，请在 `MedicalRAG/config/config.yaml` 中修改相关配置。**

---

## 项目结构说明

- `MedicalRAG/`：主程序与核心模块
- `data/`：原始数据、向量库、图片等
- `models/`：各类本地模型文件
- `results/`：检索与评估结果
- `scripts/`：测试、工具与评估脚本
- `examples/`：用法示例
- `logs/`：日志文件
- `docs/`：项目文档

---

## 启动与使用

### 1. **交互式命令行模式**

```bash
python MedicalRAG/medicalrag.py
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

