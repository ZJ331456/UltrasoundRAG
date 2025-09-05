# UltrasoundRAG 检索系统

📖 **最新文档**: 请参考以下最新文档获取完整信息：
- [QUICK_START.md](../QUICK_START.md) - 快速开始指南
- [ARCHITECTURE_GUIDE.md](../ARCHITECTURE_GUIDE.md) - 详细架构指南
- [REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md) - 重构总结

## 概述

UltrasoundRAG检索系统已完成模块化重构，现在提供：
- **四种独立检索器**: T2T、T2I、I2T、I2I
- **Caption检索图片**: 独立的标题检索功能
- **多数据库支持**: 灵活的数据库管理
- **配置驱动**: 统一的配置管理

## 🆕 推荐使用方式

```python
# 新的模块化API（推荐）
from UltrasoundRAG.retrival import (
    create_t2t_retriever, create_t2i_retriever,
    create_caption_retriever, create_multi_database_manager
)

# 独立检索器
t2t = create_t2t_retriever()
result = t2t.search("心脏超声检查")

# Caption检索
caption_retriever = create_caption_retriever(search_mode="hybrid_match")
result = caption_retriever.search_single_caption("图2-3 心脏横切面")
```

## 🚀 新架构特点

### 1. 技术升级
- **数据库**: 从ChromaDB升级到Milvus，提供更好的性能和扩展性
- **模块化设计**: 分离基础检索功能和复杂策略，便于维护和扩展
- **智能融合**: 支持多种检索方式的智能融合，提升检索质量
- **高级重排序**: 支持规则和模型两种重排序策略

### 2. 核心功能

#### 🔍 多模态检索类型

- **T2T (Text-to-Text)**: 复杂文本检索策略
  - 文本块检索 + 图片标题精确匹配
  - 图片caption向量匹配
  - 智能结果融合
  
- **T2I (Text-to-Image)**: 文本到图像检索
  - 基于CLIP模型的跨模态检索
  - 支持语义理解的图像匹配
  
- **I2T (Image-to-Text)**: 图像到文本检索
  - 图像特征提取 + caption匹配
  - 获取图像对应的文本描述
  
- **I2I (Image-to-Image)**: 图像到图像检索
  - 视觉相似度计算
  - 相似图像发现和推荐

## 📦 核心组件

### 1. 统一检索器 (MilvusUnifiedRetriever)

新架构的主入口，提供所有检索功能的统一接口。

```python
from UltrasoundRAG.retrival import create_unified_retriever

# 创建统一检索器
retriever = create_unified_retriever(
    top_k=10,
    enable_cache=True,
    text_embedding_provider="default",
    image_model_path="/path/to/fetal_clip_model",
    image_model_config_path="/path/to/config.json"
)

# T2T复杂检索 (推荐使用)
t2t_result = retriever.t2t_search("心脏超声检查异常")
print(f"文本结果: {t2t_result['total_text_results']}")
print(f"相关图片: {t2t_result['total_image_results']}")

# T2I检索
t2i_result = retriever.t2i_search("心脏病变图像")
print(f"图片结果: {t2i_result['total_results']}")

# I2T检索  
i2t_result = retriever.i2t_search("/path/to/heart_image.jpg")
print(f"文本结果: {i2t_result['total_results']}")

# I2I检索
i2i_result = retriever.i2i_search("/path/to/heart_image.jpg")
print(f"相似图片: {i2i_result['total_results']}")

# 多模态融合检索
multimodal_result = retriever.multimodal_search(
    "肝脏囊肿诊断", 
    fusion_strategy="weighted"
)
```

### 2. 融合检索管理器 (FusionRetrievalManager)

负责复杂检索策略的实现和结果融合。

```python
from UltrasoundRAG.retrival import create_fusion_retrieval_manager

# 创建融合检索管理器
fusion_manager = create_fusion_retrieval_manager(
    text_embedding_provider="default",
    image_model_path="/path/to/fetal_clip_model",
    text_search_ratio=0.7,
    image_caption_ratio=0.3,
    enable_exact_title_match=True
)

# 复杂T2T检索
t2t_result = fusion_manager.t2t_search("胆囊结石超声表现")
print(f"策略分解: {t2t_result['strategy_breakdown']}")
```

### 3. 基础检索器

#### 文本检索器 (MilvusTextRetriever)
```python
from UltrasoundRAG.retrival import create_text_retriever

text_retriever = create_text_retriever(
    milvus_uri="http://localhost:19530",
    text_embedding_provider="default"
)

results = text_retriever.search_text_by_text("超声检查方法", top_k=10)
```

#### 图像检索器 (MilvusImageRetriever)  
```python
from UltrasoundRAG.retrival import create_image_retriever

image_retriever = create_image_retriever(
    milvus_uri="http://localhost:19530",
    image_model_path="/path/to/fetal_clip_model",
    image_model_config_path="/path/to/config.json"
)

# 文本到图像
t2i_results = image_retriever.search_images_by_text("心脏超声图", top_k=5)

# 图像到文本
i2t_results = image_retriever.search_text_by_image("/path/to/image.jpg", top_k=5)

# 图像到图像
i2i_results = image_retriever.search_images_by_image("/path/to/image.jpg", top_k=5)

# 精确caption匹配
exact_results = image_retriever.search_images_by_exact_caption("图3-2 心脏超声", top_k=3)
```

### 4. 重排序模块 (Enhanced)

支持规则重排序和模型重排序两种策略。

```python
from UltrasoundRAG.retrival import create_rerank_manager

# 规则重排序
rule_reranker = create_rerank_manager(
    algorithm="rule_based",
    overlap_weight=0.3,
    length_weight=0.1,
    medical_term_weight=0.2
)

# 模型重排序
model_reranker = create_rerank_manager(
    algorithm="model_based",
    model_path="/path/to/cross_encoder_model",
    final_count=20
)

# 执行重排序
rerank_result = rule_reranker.rerank_results("查询文本", search_results)
print(f"重排序完成: {rerank_result.algorithm_used}, 耗时: {rerank_result.processing_time:.3f}s")
```

## 🏗️ 架构设计

### 文件结构
```
retrival/
├── __init__.py                     # 模块导出和便捷函数
├── unified_retriever.py            # 统一检索器 (主入口)
├── fusion_strategy_retrival.py     # 融合检索策略 (高级功能)
├── retrival_util.py               # 基础检索器 (底层功能)
├── data_structures.py             # 数据结构定义
├── enhanced_reranker.py           # 重排序模块 (增强版)
├── example_usage.py               # 使用示例
└── README.md                      # 文档 (本文件)
```

### 设计原则

1. **模块化**: 分离基础功能和高级策略
2. **可扩展**: 易于添加新的检索类型和融合策略  
3. **高性能**: 基于Milvus的高性能向量检索
4. **易用性**: 提供统一接口和便捷函数
5. **兼容性**: 保持向后兼容，平滑迁移

## ⚙️ 配置说明

### 1. UnifiedRetrievalConfig - 统一检索配置

```python
from UltrasoundRAG.retrival import UnifiedRetrievalConfig

config = UnifiedRetrievalConfig(
    # Milvus数据库配置
    milvus_uri="http://localhost:19530",
    milvus_token="your_token",  # 可选
    db_name="ultrasound_rag",
    
    # 基础配置
    top_k=10,
    enable_cache=True,
    
    # 模型配置
    text_embedding_provider="default",
    image_model_path="/path/to/fetal_clip_model",
    image_model_config_path="/path/to/config.json",
    
    # 融合策略配置
    enable_text_image_fusion=True,
    text_search_ratio=0.7,
    image_caption_ratio=0.3,
    enable_exact_title_match=True,
    
    # 结果融合权重
    text_weight=0.6,
    image_weight=0.4,
    
    # 去重配置
    content_similarity_threshold=0.8,
    enable_deduplication=True
)
```

### 2. FusionConfig - 融合策略配置

```python
from UltrasoundRAG.retrival import FusionConfig

fusion_config = FusionConfig(
    top_k=15,
    enable_text_image_fusion=True,
    text_search_ratio=0.8,
    image_caption_ratio=0.2,
    enable_exact_title_match=True,
    text_weight=0.7,
    image_weight=0.3,
    enable_deduplication=True
)
```

### 3. RerankConfig - 重排序配置

```python
from UltrasoundRAG.retrival import RerankConfig

rerank_config = RerankConfig(
    algorithm="rule_based",  # 或 "model_based"
    max_candidates=50,
    final_count=20,
    enable_cache=True,
    
    # 规则重排序权重
    overlap_weight=0.3,
    length_weight=0.1,
    position_weight=0.1,
    medical_term_weight=0.2
)
```

## 🔧 安装和部署

### 1. 依赖要求

```bash
pip install pymilvus
pip install sentence-transformers  # 用于重排序模型
pip install jieba  # 用于中文分词
pip install torch torchvision  # 用于CLIP模型
pip install pillow  # 用于图像处理
```

### 2. Milvus部署

#### Docker部署 (推荐)
```bash
# 下载docker-compose文件
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动Milvus
docker-compose up -d
```

#### 验证连接
```python
from pymilvus import connections
connections.connect("default", host="localhost", port="19530")
print("Milvus连接成功")
```

### 3. 数据索引

使用提供的索引工具创建数据索引：

```python
# 文本数据索引
from UltrasoundRAG.index.markdown_parse import MarkdownParser
from UltrasoundRAG.milvus.milvus_manager import MilvusManager

# 解析Markdown文档
parser = MarkdownParser()
text_chunks = parser.parse_markdowns()

# 索引到Milvus
text_manager = MilvusManager(collection_type="md")
text_manager.insert_data(text_chunks, embeddings)

# 图像数据索引  
from UltrasoundRAG.index.image_parse import ImageParser

# 解析图像数据
image_parser = ImageParser()
image_data = image_parser.parse_images()

# 索引到Milvus
image_manager = MilvusManager(collection_type="image")
image_manager.insert_data(image_data, image_vectors)
```

## 🎯 检索策略详解

### 1. T2T复杂检索策略

T2T检索是系统的核心功能，采用多层策略：

#### 策略1: 文本块检索 + 图片标题匹配
1. 在Milvus文本集合中进行语义检索
2. 从检索结果中提取图片标题信息
3. 根据提取的标题在图像集合中进行精确匹配
4. 返回文本结果和相关图片

#### 策略2: 图片caption向量匹配  
1. 在Milvus图像集合中进行语义检索
2. 获取图片的caption信息
3. 将caption作为文本结果返回
4. 提供图文对应关系

#### 结果融合
- 应用不同权重融合两种策略的结果
- 智能去重，避免重复内容
- 按综合分数重新排序

### 2. 图像标题提取

系统使用智能正则表达式提取图片标题：

```python
title_patterns = [
    r'图\s*(\d+[-~]\d+)\s*([^。\n\r！？；，]{1,100})',  # 图X-X 标题
    r'Figure\s+(\d+\.\d+)\s*([^。\n\r！？；，]{1,100})',  # Figure X.X 标题
    r'图\s*(\d+)\s*([^。\n\r！？；，]{1,100})',  # 图X 标题
    r'图片\s*(\d+)\s*([^。\n\r！？；，]{1,100})',  # 图片X 标题
]
```

### 3. 结果融合策略

#### 加权融合 (Weighted)
```python
# 应用权重
for result in text_results:
    result.score *= text_weight

for result in image_results:
    result.score *= image_weight

# 合并并排序
all_results = text_results + image_results
all_results.sort(key=lambda x: x.score, reverse=True)
```

#### 交替融合 (Interleaved)
```python
# 交替取结果
fused_results = []
max_len = max(len(text_results), len(image_results))
for i in range(max_len):
    if i < len(text_results):
        fused_results.append(text_results[i])
    if i < len(image_results):
        fused_results.append(image_results[i])
```

#### 分数融合 (Score-based)
```python
# 按原始分数排序
all_results = text_results + image_results
all_results.sort(key=lambda x: x.score, reverse=True)
```

## 🚀 性能优化

### 1. 缓存策略
- 文本嵌入缓存：避免重复计算embedding
- 检索结果缓存：缓存常见查询的结果
- 模型实例缓存：避免重复加载模型

### 2. 批处理优化
- 批量embedding计算
- 批量Milvus查询
- 批量重排序处理

### 3. 内存管理
- LRU缓存机制，自动清理过期数据
- 智能内存限制，防止内存溢出
- 延迟加载，按需初始化组件

## 🧪 测试和验证

### 1. 单元测试

```python
# 测试基础检索功能
def test_text_retrieval():
    retriever = create_text_retriever()
    results = retriever.search_text_by_text("测试查询", top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r, RetrievalResult) for r in results)

# 测试融合检索
def test_fusion_retrieval():
    manager = create_fusion_retrieval_manager()
    result = manager.t2t_search("测试查询", top_k=10)
    assert 'total_text_results' in result
    assert 'total_image_results' in result
```

### 2. 性能测试

```python
import time

# 检索性能测试
start_time = time.time()
result = retriever.t2t_search("心脏超声检查", top_k=10)
end_time = time.time()

print(f"检索耗时: {end_time - start_time:.3f}s")
print(f"结果数量: {result['total_text_results']} + {result['total_image_results']}")
```

### 3. 质量评估

```python
# 检索质量评估
def evaluate_retrieval_quality(queries, ground_truth):
    total_score = 0
    for query, expected in zip(queries, ground_truth):
        result = retriever.t2t_search(query, top_k=10)
        score = calculate_relevance_score(result, expected)
        total_score += score
    
    return total_score / len(queries)
```

## 📝 使用示例

详细的使用示例请参考 `example_usage.py` 文件，包含：

1. **基础使用示例**: 展示主要功能的基本用法
2. **高级功能示例**: 展示融合策略和自定义配置
3. **性能测试示例**: 展示性能测试和优化方法
4. **结果分析示例**: 展示如何分析和处理检索结果

## 🔧 故障排除

### 1. 常见问题

**Q: Milvus连接失败**
```bash
# 检查Milvus服务状态
docker ps | grep milvus

# 检查端口占用
netstat -an | grep 19530

# 重启Milvus服务
docker-compose restart
```

**Q: 模型加载失败**
```python
# 检查模型路径
import os
assert os.path.exists(model_path), f"模型路径不存在: {model_path}"

# 检查模型格式
from sentence_transformers import CrossEncoder
model = CrossEncoder(model_path)  # 验证模型可以正常加载
```

**Q: 检索结果为空**
```python
# 检查数据是否已索引
from UltrasoundRAG.milvus.milvus_manager import MilvusManager
manager = MilvusManager(collection_type="md")
manager.get_collection_info()  # 查看集合信息

# 检查embedding是否正确
embedding = text_retriever.text_embedding_model.get_query_embedding("测试")
assert embedding is not None and len(embedding) == 768
```

### 2. 调试建议

1. **启用详细日志**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **检查配置**:
```python
# 打印当前配置
print(retriever.config.__dict__)
print(retriever.get_stats())
```

3. **分步测试**:
```python
# 测试基础组件
text_retriever = create_text_retriever()
image_retriever = create_image_retriever()

# 测试单独功能
text_results = text_retriever.search_text_by_text("测试", top_k=5)
image_results = image_retriever.search_images_by_text("测试", top_k=5)
```

## 🔄 迁移指南

### 从旧版本迁移

1. **数据迁移**: 将ChromaDB数据迁移到Milvus
2. **代码更新**: 更新导入和接口调用
3. **配置更新**: 更新配置文件格式
4. **测试验证**: 验证迁移后的功能正确性

### 迁移示例

```python
# 旧版本代码
from UltrasoundRAG.retrival import create_unified_retriever
retriever = create_unified_retriever(
    text_embedding_provider="default",
    image_embedding_provider="clip"
)
result = retriever.search_text("查询")

# 新版本代码
from UltrasoundRAG.retrival import create_unified_retriever
retriever = create_unified_retriever(
    text_embedding_provider="default",
    image_model_path="/path/to/fetal_clip",
    image_model_config_path="/path/to/config.json"
)
result = retriever.t2t_search("查询")  # 使用新的T2T接口
```

## 📚 相关资源

- [Milvus官方文档](https://milvus.io/docs)
- [Sentence Transformers文档](https://www.sbert.net/)
- [UltrasoundRAG项目文档](../README.md)

## 🤝 贡献指南

欢迎贡献代码和改进建议！请遵循以下步骤：

1. Fork项目仓库
2. 创建特性分支
3. 提交代码更改  
4. 创建Pull Request
5. 等待代码审查

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

*最后更新时间: 2024年*