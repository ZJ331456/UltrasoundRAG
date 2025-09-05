# UltrasoundRAG 快速开始指南

## 系统概述

UltrasoundRAG是一个模块化的多模态检索系统，支持四种检索方式、Caption检索图片功能和多数据库管理。

## 核心功能

✅ **四种独立检索方式**：T2T、T2I、I2T、I2I
✅ **Caption检索图片**：从文本标题精确找到对应图片
✅ **多数据库支持**：支持多个数据库和集合的管理
✅ **配置驱动**：所有参数可通过配置文件调整
✅ **高性能**：缓存、批处理、并行优化

## 快速使用

### 1. 基本索引构建

```bash
# 构建所有索引（默认行为）
python ultrasoundrag.py

# 只构建Markdown索引
python ultrasoundrag.py --build-md

# 只构建图片索引
python ultrasoundrag.py --build-image

# 重建所有索引（删除后重新创建）
python ultrasoundrag.py --build-md --build-image --recreate
```

### 2. 四种检索方式测试

```bash
# 测试所有检索模式
python ultrasoundrag.py --test-modes

# 测试多数据库功能
python ultrasoundrag.py --test-multidb

# 测试Caption检索图片
python ultrasoundrag.py --test-caption

# 性能基准测试
python ultrasoundrag.py --benchmark
```

### 3. 编程接口使用

#### 模块化检索器（推荐）

```python
from UltrasoundRAG.retrival import (
    create_t2t_retriever, create_t2i_retriever,
    create_i2t_retriever, create_i2i_retriever,
    create_caption_retriever, create_multi_database_manager
)

# T2T：文本到文本检索
t2t = create_t2t_retriever(db_name="default", top_k=10)
result = t2t.search("心脏超声检查方法", strategy="fusion")
print(f"找到 {result['total_results']} 个文本结果")

# T2I：文本到图片检索
t2i = create_t2i_retriever(db_name="default", top_k=5)
result = t2i.search("心脏病变图像")
print(f"找到 {result['total_results']} 个相关图片")

# I2T：图片到文本检索
i2t = create_i2t_retriever(db_name="default", top_k=10)
result = i2t.search("/path/to/ultrasound_image.jpg")
print(f"找到 {result['total_results']} 个相关文本")

# I2I：图片到图片检索
i2i = create_i2i_retriever(db_name="default", top_k=5)
result = i2i.search("/path/to/query_image.jpg")
print(f"找到 {result['total_results']} 个相似图片")
```

#### Caption检索图片

```python
# 创建Caption检索器（支持混合匹配）
caption_retriever = create_caption_retriever(
    db_name="default", 
    search_mode="hybrid_match"
)

# 单个caption检索
result = caption_retriever.search_single_caption(
    "图2-3 心脏超声横切面", 
    top_k=3
)
print(f"找到 {result['total_results']} 个匹配图片")

# 批量caption检索
captions = ["图1-1 肝脏超声", "图2-2 心脏切面"]
batch_result = caption_retriever.search_multiple_captions(captions, top_k=2)
print(f"批量检索找到 {batch_result['total_results']} 个图片")

# 从文本块自动提取caption并检索
text_chunk = """
心脏超声检查包括多个切面。图2-1显示四腔心切面，
图2-2为心脏短轴切面，图2-3展示了心尖四腔心切面。
"""
text_result = caption_retriever.search_from_text_chunk(text_chunk, top_k=2)
print(f"从文本提取 {len(text_result['extracted_captions'])} 个caption")
print(f"找到 {text_result['total_results']} 个相关图片")
```

#### 多数据库管理

```python
# 创建多数据库管理器
manager = create_multi_database_manager()

# 获取可用数据库
available_dbs = manager.get_available_databases()
print(f"可用数据库: {[db.name for db in available_dbs]}")

# 单数据库检索
result = manager.search_single_database(
    "default", "t2t", "心脏超声诊断", top_k=5
)
print(f"在数据库 'default' 中找到 {result['total_results']} 个结果")

# 多数据库检索策略
from UltrasoundRAG.retrival import create_retrieval_strategy
strategy = create_retrieval_strategy(
    databases=["default", "cardiac_specialist"],
    aggregation_method="weighted",
    max_results=10
)
multi_result = manager.search_multiple_databases(
    strategy, "t2t", "心脏超声检查", top_k=5
)
print(f"多数据库检索找到 {multi_result['total_results']} 个结果")

# 自动选择数据库
auto_selected = manager.auto_select_databases("心脏病变", "t2i", num_databases=2)
print(f"自动选择的数据库: {auto_selected}")
```

## 配置管理

### 数据库配置

在 `config.yaml` 中配置多个数据库：

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
```

### 检索模式配置

```yaml
retriever:
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
    
    caption_to_image:
      enabled: true
      search_mode: "hybrid_match"
      fuzzy_threshold: 0.8
      max_results_per_caption: 3
```

## 实际使用场景

### 场景1：医生查找相关超声图像

```python
# 医生想找心脏相关的超声图像
t2i = create_t2i_retriever()
result = t2i.search("心房间隔缺损超声图像", top_k=5)

for img in result['results']:
    print(f"图片: {img.metadata['relative_path']}")
    print(f"描述: {img.content}")
    print(f"相似度: {img.score:.3f}")
    print("---")
```

### 场景2：根据图片标题精确找图

```python
# 医生记得图片标题，想找到对应的图片
caption_retriever = create_caption_retriever(search_mode="exact_match")
result = caption_retriever.search_single_caption("图3-15 胎儿脊柱侧弯声像图")

if result['total_results'] > 0:
    img = result['results'][0]
    print(f"找到图片: {img.metadata['relative_path']}")
else:
    print("未找到精确匹配的图片")
```

### 场景3：多数据库联合检索

```python
# 在心脏专科和通用数据库中同时搜索
manager = create_multi_database_manager()
strategy = create_retrieval_strategy(
    databases=["default", "cardiac_specialist"],
    aggregation_method="weighted"
)

result = manager.search_multiple_databases(
    strategy, "t2t", "房间隔缺损诊断标准", top_k=10
)

print(f"在 {result['total_databases_searched']} 个数据库中找到 {result['total_results']} 个结果")
```

### 场景4：从报告中提取图片引用

```python
# 从超声报告中自动提取图片引用
report_text = """
患者心脏超声检查显示：图2-1心脏四腔心切面可见左心房轻度扩大，
图2-2心脏短轴切面显示左心室收缩功能正常，图2-3二尖瓣血流频谱
显示轻度返流信号。
"""

caption_retriever = create_caption_retriever(search_mode="fuzzy_match")
result = caption_retriever.search_from_text_chunk(report_text)

print(f"从报告中提取到图片引用: {result['extracted_captions']}")
print(f"找到对应图片: {result['total_results']} 张")
```

## 性能优化建议

### 1. 缓存配置
```yaml
retriever:
  enable_cache: true
```

### 2. 批量处理
```python
# 批量检索多个查询
queries = ["心脏超声", "肝脏检查", "胎儿评估"]
for query in queries:
    result = t2t.search(query)
    # 处理结果...
```

### 3. 数据库选择
```python
# 根据查询内容自动选择最合适的数据库
auto_dbs = manager.auto_select_databases("心脏超声", "t2t")
```

## 常见问题

### Q: 如何添加新的数据库？
A: 在`config.yaml`中添加数据库配置，然后构建对应的索引。

### Q: 检索结果为空怎么办？
A: 检查索引是否已构建、查询向量是否正确生成、集合中是否有数据。

### Q: 如何提高检索精度？
A: 调整相似度阈值、使用融合策略、选择合适的检索模式。

### Q: 支持哪些图片格式？
A: 支持常见的图片格式如JPG、PNG、BMP等。

## 下一步

- 查看 `ARCHITECTURE_GUIDE.md` 了解详细架构
- 运行 `python ultrasoundrag.py --benchmark` 测试性能
- 根据实际需求调整配置参数
- 探索更多高级功能和自定义选项
