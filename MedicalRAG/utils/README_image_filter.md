# 图像过滤工具使用说明

## 概述

图像过滤工具 (`image_filter_utils.py`) 是一个集成了 `filter.py` 中过滤机制的图像检索结果重排序工具。它可以基于文本相似度、图像相似度或多模态相似度对图像检索结果进行过滤和重排序，从而提高检索结果的质量和相关性。

## 主要功能

### 1. 文本相似度过滤
- 基于 TF-IDF、关键词重叠、SimHash、BM25 的文本相似度计算
- 适用于基于文本描述的图像检索结果过滤

### 2. 图像相似度过滤
- 基于 LPIPS、MS-SSIM、HaarPSI 的图像相似度计算
- 适用于以图搜图的检索结果过滤

### 3. 多模态过滤
- 结合文本和图像相似度的综合过滤
- 提供更全面的相似度评估

## 使用方法

### 1. 基本使用

```python
from MedicalRAG.utils.image_filter_utils import create_image_filter

# 创建图像过滤器
image_filter = create_image_filter(
    text_weight=0.3,        # 文本相似度权重
    image_weight=0.7,       # 图像相似度权重
    enable_text_filter=True,  # 启用文本过滤
    enable_image_filter=True  # 启用图像过滤
)

# 对检索结果进行重排序
filtered_results = image_filter.rerank_results(
    query_text="超声心动图显示什么？",
    query_image_path="/path/to/query/image.jpg",  # 可选
    results=retrieval_results,
    filter_type="multimodal",  # "text", "image", "multimodal"
    similarity_threshold=0.1
)
```

### 2. 在 MedicalRAG 系统中使用

图像过滤功能已经集成到 `MedicalRAGSystem` 中，可以通过以下参数控制：

```python
from MedicalRAG.medicalrag import MedicalRAGSystem

rag_system = MedicalRAGSystem()
rag_system.initialize_components()

# 使用图像过滤功能
result = rag_system.search_and_generate(
    query="超声心动图的正常表现",
    image_path="/path/to/query/image.jpg",  # 可选
    enable_image_filtering=True,           # 启用图像过滤
    image_filter_type="multimodal",        # 过滤类型
    similarity_threshold=0.1               # 相似度阈值
)
```

### 3. 交互式模式

在交互式模式中，系统会询问是否启用图像过滤功能：

```bash
python -m MedicalRAG.medicalrag
```

系统会提示：
- 是否启用图像过滤功能？(y/n, 默认y)
- 选择过滤类型 (text/image/multimodal, 默认multimodal)

## 配置参数

### ImageRetrievalFilter 参数

- `text_weight` (float): 文本相似度权重，默认 0.3
- `image_weight` (float): 图像相似度权重，默认 0.7
- `enable_text_filter` (bool): 是否启用文本过滤，默认 True
- `enable_image_filter` (bool): 是否启用图像过滤，默认 True

### 过滤方法参数

- `similarity_threshold` (float): 相似度阈值，低于此值的结果将被过滤，默认 0.1
- `filter_type` (str): 过滤类型
  - `"text"`: 仅使用文本相似度过滤
  - `"image"`: 仅使用图像相似度过滤
  - `"multimodal"`: 使用多模态相似度过滤

## 依赖要求

图像过滤工具依赖于 `filter.py` 模块中的以下组件：

- `TextSimilarityFilter`: 文本相似度过滤器
- `ImageSimilarityFilter`: 图像相似度过滤器
- `MultiModalFilter`: 多模态过滤器

确保 `filter.py` 文件位于正确的路径：`e:\Dolphin\ht-rag\ultrasound_data_traing\filter.py`

## 测试

运行测试脚本来验证图像过滤功能：

```bash
python MedicalRAG/test_image_filter.py
```

测试脚本会：
1. 测试过滤器组件的初始化
2. 测试不同过滤类型的功能
3. 显示过滤统计信息
4. 验证结果质量

## 输出信息

过滤后的结果会在元数据中包含以下信息：

```python
result.metadata.update({
    'text_similarity_score': 0.85,      # 文本相似度分数
    'image_similarity_score': 0.72,     # 图像相似度分数
    'multimodal_similarity_score': 0.78, # 多模态相似度分数
    'filtered_by': 'multimodal'         # 过滤方式
})
```

## 性能优化建议

1. **相似度阈值调整**: 根据实际需求调整 `similarity_threshold`，较高的阈值会过滤更多结果
2. **权重平衡**: 根据应用场景调整 `text_weight` 和 `image_weight`
3. **选择性启用**: 根据查询类型选择合适的过滤类型，避免不必要的计算
4. **批量处理**: 对于大量查询，考虑批量处理以提高效率

## 故障排除

### 常见问题

1. **过滤器初始化失败**
   - 检查 `filter.py` 文件路径是否正确
   - 确保相关依赖库已安装

2. **图像路径错误**
   - 确保图像文件路径存在且可访问
   - 检查图像格式是否支持

3. **过滤结果为空**
   - 降低 `similarity_threshold` 值
   - 检查查询文本或图像是否合适

### 日志信息

系统会记录详细的日志信息，包括：
- 过滤器初始化状态
- 过滤过程的统计信息
- 错误和警告信息

查看日志以获取更多调试信息。

## 扩展功能

可以通过以下方式扩展图像过滤功能：

1. **自定义相似度计算**: 在 `filter.py` 中添加新的相似度计算方法
2. **动态权重调整**: 根据查询类型动态调整文本和图像权重
3. **结果缓存**: 对相似度计算结果进行缓存以提高性能
4. **个性化过滤**: 根据用户偏好调整过滤策略

## 版本信息

- 创建日期: 2024年
- 版本: 1.0.0
- 兼容性: MedicalRAG 系统