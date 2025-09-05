# UltrasoundRAG 模块化架构设计指南

## 概述

本文档详细说明了UltrasoundRAG系统的重构后架构，包括模块化设计、多种检索方式、多数据库支持等核心功能。

## 系统架构概览

```
UltrasoundRAG System
├── 配置管理层 (config/)
│   └── config.yaml - 统一配置文件
├── 索引构建层 (index/)
│   ├── markdown_parse.py - Markdown文档解析
│   └── image_parse.py - 图片解析
├── 向量数据库层 (milvus/)
│   └── milvus_manager.py - Milvus数据库管理
├── 检索层 (retrival/)
│   ├── modular_retrievers.py - 四种独立检索器
│   ├── caption_to_image_retriever.py - Caption检索图片
│   ├── multi_database_manager.py - 多数据库管理
│   ├── unified_retriever.py - 统一检索接口
│   └── fusion_strategy_retrival.py - 融合检索策略
├── 工具层 (utils/)
│   ├── embedding_utils.py - 嵌入模型管理
│   ├── llm_utils.py - 大语言模型工具
│   └── logger.py - 日志管理
└── 主入口
    └── ultrasoundrag.py - 系统主入口
```

## 核心特性

### 1. 模块化检索系统

#### 四种独立检索方式

1. **T2T (Text-to-Text)** - 文本到文本检索
   - 基础文本检索
   - 融合策略（文本块 + 图片caption检索）
   - 可配置的权重分配

2. **T2I (Text-to-Image)** - 文本到图片检索
   - 使用CLIP文本编码器
   - 语义相似度匹配
   - 支持不同的文本嵌入模型

3. **I2T (Image-to-Text)** - 图片到文本检索
   - CLIP图片编码器
   - 跨模态检索
   - 返回相关文本内容

4. **I2I (Image-to-Image)** - 图片到图片检索
   - 相似图片检索
   - 图片特征匹配
   - 包含丰富的元数据

#### 使用示例

```python
# 创建独立检索器
from UltrasoundRAG.retrival.modular_retrievers import (
    create_t2t_retriever, create_t2i_retriever,
    create_i2t_retriever, create_i2i_retriever
)

# T2T检索
t2t_retriever = create_t2t_retriever(db_name="default", top_k=10)
result = t2t_retriever.search("心脏超声检查方法", strategy="fusion")

# T2I检索
t2i_retriever = create_t2i_retriever(db_name="default", top_k=5)
result = t2i_retriever.search("心脏病变图像")

# I2T检索
i2t_retriever = create_i2t_retriever(db_name="default", top_k=10)
result = i2t_retriever.search("/path/to/ultrasound_image.jpg")

# I2I检索
i2i_retriever = create_i2i_retriever(db_name="default", top_k=5)
result = i2i_retriever.search("/path/to/query_image.jpg")
```

### 2. Caption检索图片模块

独立的caption到图片检索功能，支持多种匹配策略：

#### 四种匹配模式

1. **精确匹配 (exact_match)**
   - 完全匹配caption文本
   - 最高准确性

2. **模糊匹配 (fuzzy_match)**
   - 基于关键词的模糊匹配
   - 可配置相似度阈值

3. **语义匹配 (semantic_match)**
   - 基于CLIP向量的语义相似度
   - 理解图片内容语义

4. **混合匹配 (hybrid_match)**
   - 结合上述三种方式
   - 最佳的匹配效果

#### 使用示例

```python
from UltrasoundRAG.retrival.caption_to_image_retriever import create_caption_retriever

# 创建caption检索器
caption_retriever = create_caption_retriever(
    db_name="default", 
    search_mode="hybrid_match"
)

# 单个caption检索
result = caption_retriever.search_single_caption(
    "图2-3 心脏超声横切面", 
    top_k=3
)

# 批量caption检索
captions = ["图1-1 肝脏超声", "图2-2 心脏切面"]
batch_result = caption_retriever.search_multiple_captions(captions, top_k=2)

# 从文本块自动提取caption并检索
text_chunk = "图2-1显示心脏四腔心切面，图2-2为心脏短轴切面"
text_result = caption_retriever.search_from_text_chunk(text_chunk, top_k=2)
```

### 3. 多数据库支持

支持多个数据库和集合的灵活配置和管理：

#### 数据库配置

在`config.yaml`中配置多个数据库：

```yaml
retriever:
  databases:
    default:
      db_name: "ultrasound_vector"
      collections:
        text: "md_documents"
        image: "images"
    
    cardiac_specialist:
      db_name: "cardiac_specialist_vector"
      type: "specialized"
      priority: 2
      collections:
        text: "cardiac_docs"
        image: "cardiac_images"
      tags: ["cardiac", "heart"]
      description: "心脏超声专门数据库"
    
    experimental:
      db_name: "experimental_vector"
      type: "experimental"
      priority: 1
      collections:
        text: "exp_docs"
        image: "exp_images"
      tags: ["experimental", "research"]
      description: "实验性数据库"
```

#### 使用多数据库管理器

```python
from UltrasoundRAG.retrival.multi_database_manager import (
    create_multi_database_manager, create_retrieval_strategy
)

# 创建管理器
manager = create_multi_database_manager()

# 获取可用数据库
available_dbs = manager.get_available_databases()

# 单数据库检索
result = manager.search_single_database(
    "cardiac_specialist", "t2t", "心房颤动诊断", top_k=5
)

# 多数据库检索
strategy = create_retrieval_strategy(
    databases=["default", "cardiac_specialist"],
    aggregation_method="weighted",
    max_results=10
)
multi_result = manager.search_multiple_databases(
    strategy, "t2t", "心脏超声检查", top_k=5
)

# 自动选择数据库
auto_selected = manager.auto_select_databases("心脏病变", "t2i", num_databases=2)

# 按标签搜索
tag_result = manager.search_by_tags(["cardiac"], "t2t", "心房异常", top_k=5)
```

### 4. 配置驱动的系统

所有参数都可以通过`config.yaml`配置：

```yaml
# 检索器配置
retriever:
  # 基础检索配置
  top_k: 10
  similarity_threshold: 0.5
  enable_cache: true
  
  # 检索方式配置
  retrieval_modes:
    t2t:
      enabled: true
      strategy: "fusion"
      fusion_config:
        text_search_ratio: 0.7
        image_caption_ratio: 0.3
        enable_exact_title_match: true
    
    t2i:
      enabled: true
      use_clip_text_encoder: true
    
    i2t:
      enabled: true
      return_caption_as_text: true
    
    i2i:
      enabled: true
      include_metadata: true
  
  # Caption检索配置
  caption_to_image:
    enabled: true
    search_mode: "hybrid_match"
    fuzzy_threshold: 0.8
    max_results_per_caption: 3
  
  # 融合检索配置
  fusion:
    default_strategy: "weighted"
    weights:
      text: 0.6
      image: 0.4
```

## 主入口功能

重构后的`ultrasoundrag.py`支持丰富的命令行选项：

### 索引构建

```bash
# 构建Markdown索引
python ultrasoundrag.py --build-md

# 构建图片索引
python ultrasoundrag.py --build-image

# 重建所有索引
python ultrasoundrag.py --build-md --build-image --recreate

# 构建特定数据集
python ultrasoundrag.py --build-md --md-datasets ultrasound_book
```

### 检索测试

```bash
# 基础检索测试
python ultrasoundrag.py --test

# 四种检索模式测试
python ultrasoundrag.py --test-modes

# 多数据库检索测试
python ultrasoundrag.py --test-multidb --query "心脏超声诊断"

# Caption检索测试
python ultrasoundrag.py --test-caption

# 性能基准测试
python ultrasoundrag.py --benchmark --benchmark-count 5

# 图片相关检索测试
python ultrasoundrag.py --test-modes --image-path "/path/to/image.jpg"
```

### 组合使用

```bash
# 重建索引并执行全面测试
python ultrasoundrag.py --build-md --build-image --recreate --test-modes --test-multidb --benchmark

# 指定数据库和参数
python ultrasoundrag.py --test-modes --db-name cardiac_specialist --top-k 5 --query "心房颤动"
```

## 最佳实践

### 1. 数据库规划

- **按专业领域分离**：心脏、肝脏、妇科等专门数据库
- **按数据质量分层**：生产、测试、实验数据库
- **合理设置优先级**：高质量数据库优先级更高

### 2. 检索策略选择

- **T2T融合模式**：适合复杂的文本检索场景
- **T2I语义匹配**：适合根据描述找图片
- **Caption混合匹配**：适合精确的图片标题匹配
- **多数据库加权**：适合跨领域检索

### 3. 性能优化

- **缓存策略**：启用检索器缓存
- **批量处理**：使用批量检索接口
- **索引优化**：根据查询模式优化向量索引
- **阈值调优**：根据实际效果调整相似度阈值

### 4. 监控和维护

- **统计信息**：定期查看数据库和检索统计
- **性能基准**：定期执行基准测试
- **错误处理**：完善的异常处理和日志记录
- **配置管理**：版本化配置文件管理

## 扩展指南

### 添加新的数据库

1. 在`config.yaml`中添加数据库配置
2. 创建对应的Milvus集合
3. 构建索引数据
4. 测试检索功能

### 添加新的检索方式

1. 继承`BaseRetriever`类
2. 实现`search`方法
3. 在`multi_database_manager.py`中注册
4. 添加配置选项

### 自定义匹配策略

1. 继承`CaptionToImageRetriever`
2. 实现自定义匹配逻辑
3. 在配置中添加新的模式
4. 更新工厂函数

## 故障排除

### 常见问题

1. **连接问题**
   - 检查Milvus服务状态
   - 验证连接配置
   - 检查网络连通性

2. **检索结果为空**
   - 确认索引已构建
   - 检查查询向量生成
   - 验证集合数据

3. **性能问题**
   - 调整批处理大小
   - 优化向量维度
   - 启用缓存功能

4. **配置错误**
   - 验证YAML语法
   - 检查路径配置
   - 确认模型文件存在

### 调试技巧

1. **启用详细日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **使用统计信息**
```python
stats = manager.get_database_stats()
print(stats)
```

3. **基准测试诊断**
```bash
python ultrasoundrag.py --benchmark --benchmark-count 1 --top-k 1
```

## 总结

重构后的UltrasoundRAG系统具有以下优势：

1. **模块化设计**：各功能模块独立，易于维护和扩展
2. **灵活配置**：所有参数可配置，适应不同使用场景
3. **多样化检索**：支持四种检索方式，满足不同需求
4. **多数据库支持**：支持复杂的数据管理和检索策略
5. **性能优化**：缓存、批处理、并行等优化机制
6. **易于使用**：丰富的命令行选项和便捷函数

这个架构为未来的功能扩展和性能优化提供了坚实的基础。
