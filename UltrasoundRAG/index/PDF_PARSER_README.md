# PDF解析器使用指南

## 概述

PDF解析器是UltrasoundRAG系统的一部分，用于处理PDF文档，提取文本内容并进行智能分块，支持将处理后的数据存储到Milvus向量数据库中。

## 功能特性

- **PDF文本提取**: 使用pdfplumber库提取PDF文档中的文本内容
- **智能分块**: 按章节和段落进行智能分块，保持语义完整性
- **重叠窗口**: 支持块间重叠，确保语义连续性
- **页面信息**: 保留页面范围信息，便于定位
- **Milvus集成**: 支持将处理后的数据存储到Milvus向量数据库

## 安装依赖

```bash
pip install pdfplumber
```

## 配置说明

### 1. 配置文件 (config.yaml)

```yaml
milvus:
  pdf_collection_name: "thesis_papers"

indexing:
  # PDF解析配置
  pdf_parse:
    max_chunk_size: 1500  # 单个chunk的大小
    overlap_ratio: 0.15   # 重叠窗口比例
    min_chunk_size: 100   # 最小chunk大小

  # PDF数据集配置
  pdf:
    datasets:
      thesis_papers:
        enabled: true
        base_pdf_path: "/path/to/pdf/files"
        collection_name: "thesis_papers"
        description: "学术论文PDF数据集"
```

### 2. 配置参数说明

- `max_chunk_size`: 单个文本块的最大字符数
- `overlap_ratio`: 块间重叠比例，用于保持语义连续性
- `min_chunk_size`: 最小文本块大小，避免过短的无意义片段
- `base_pdf_path`: PDF文件的基础路径

## 使用方法

### 1. 基本使用

```python
from UltrasoundRAG.index.pdf_parse import PDFParser

# 初始化PDF解析器
parser = PDFParser(dataset_name="thesis_papers")

# 解析所有PDF文件
chunks = parser.parse_pdfs()

# 处理单个PDF文件
chunks = parser.process_document("/path/to/file.pdf")
```

### 2. 与Milvus集成

```python
from UltrasoundRAG.index.pdf_parse import PDFParser
from UltrasoundRAG.milvus.milvus_manager import MilvusManager

# 初始化解析器和Milvus管理器
pdf_parser = PDFParser(dataset_name="thesis_papers")
milvus_manager = MilvusManager(collection_type="pdf")

# 解析PDF文档
chunks = pdf_parser.parse_pdfs()

# 生成向量（需要调用embedding模型）
embeddings_qwen = generate_qwen_embeddings(chunks)
embeddings_clip = generate_clip_embeddings(chunks)

# 插入数据到Milvus
success = milvus_manager.insert_data(
    data_list=chunks,
    embeddings_qwen=embeddings_qwen,
    embeddings_clip=embeddings_clip
)
```

### 3. 数据结构

每个文本块包含以下字段：

```python
{
    'id': int,                    # 唯一标识符
    'title': str,                 # 章节标题
    'content': str,               # 文本内容
    'pdf_file': str,              # PDF文件路径
    'document_name': str,         # 文档名称
    'chunk_index': int,           # 块索引
    'page_start': int,            # 起始页面
    'page_end': int,              # 结束页面
    'image_paths': list,          # 图片路径（PDF中通常为空）
    'image_captions': list        # 图片标题（PDF中通常为空）
}
```

## 分块策略

### 1. 章节识别

解析器会自动识别PDF中的章节结构，包括：
- 数字编号标题（如1.1, 2.3等）
- 中文章节标题（如第1章、第1节等）
- 英文标题关键词（如Abstract, Introduction等）

### 2. 智能分割

- **优先按段落分割**: 保持段落完整性
- **其次按句子分割**: 在段落过长时按句子分割
- **重叠窗口**: 块间保持15%的重叠，确保语义连续性
- **最小块大小**: 避免生成过短的无意义片段

### 3. 文本清理

- 移除多余的空白字符
- 规范化换行符
- 保留中英文、数字和基本标点符号

## 测试和示例

### 1. 运行测试脚本

```bash
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/UltrasoundRAG/index
python test_pdf_parser.py
```

### 2. 运行使用示例

```bash
python pdf_usage_example.py
```

## 注意事项

1. **依赖安装**: 确保已安装pdfplumber库
2. **文件路径**: 检查配置文件中的PDF文件路径是否正确
3. **内存使用**: 大型PDF文件可能消耗较多内存
4. **文本质量**: PDF文本提取质量取决于PDF的格式和结构
5. **向量生成**: 实际使用时需要调用embedding模型生成向量

## 故障排除

### 1. 导入错误

```
ImportError: No module named 'pdfplumber'
```
**解决方案**: 安装pdfplumber库
```bash
pip install pdfplumber
```

### 2. 文件路径错误

```
FileNotFoundError: [Errno 2] No such file or directory
```
**解决方案**: 检查配置文件中的base_pdf_path路径是否正确

### 3. PDF解析失败

```
pdfplumber.exceptions.PDFSyntaxError
```
**解决方案**: 检查PDF文件是否损坏或格式不支持

## 扩展功能

### 1. 自定义分块策略

可以继承PDFParser类并重写相关方法：

```python
class CustomPDFParser(PDFParser):
    def _is_title(self, line: str) -> bool:
        # 自定义标题识别逻辑
        pass
    
    def _split_long_content(self, content: str, title: str, base_index: int):
        # 自定义分块逻辑
        pass
```

### 2. 添加图片提取

可以扩展解析器以提取PDF中的图片：

```python
def extract_images_from_pdf(self, pdf_path: str):
    # 提取PDF中的图片
    pass
```

## 性能优化

1. **批量处理**: 使用parse_pdfs()方法批量处理多个文件
2. **内存管理**: 对于大型PDF文件，考虑分页处理
3. **并行处理**: 可以扩展为多线程处理多个PDF文件
4. **缓存机制**: 对已处理的PDF文件进行缓存

## 更新日志

- v1.0.0: 初始版本，支持基本PDF文本提取和分块
- 支持智能章节识别
- 支持重叠窗口分块
- 集成Milvus数据库支持
