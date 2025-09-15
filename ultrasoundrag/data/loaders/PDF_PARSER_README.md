# PDF解析器使用说明

## 功能概述

PDF解析器是一个专门用于处理PDF文档的工具，具备以下核心功能：

1. **文本提取**：从PDF文档中提取纯文本内容
2. **图片提取**：提取PDF中的图片并保存为临时文件
3. **OCR识别**：使用PaddleOCR识别图片中的文字内容
4. **智能分块**：按照标题层级和语义完整性对内容进行分块
5. **结构化输出**：生成包含文本、图片、OCR结果的结构化数据

## 技术特点

- **基于PyMuPDF**：高效的PDF文档处理
- **PaddleOCR集成**：支持中英文文字识别
- **智能分块策略**：保持语义完整性的内容分割
- **配置化管理**：通过YAML配置文件灵活管理参数
- **错误容错**：处理缺失文件和处理失败的情况

## 安装依赖

```bash
pip install -r requirements_pdf.txt
```

主要依赖包：
- PyMuPDF >= 1.23.0 (PDF处理)
- paddleocr >= 2.7.0 (OCR识别)
- paddlepaddle >= 2.5.0 (深度学习框架)
- opencv-python >= 4.8.0 (图像处理)
- Pillow >= 9.0.0 (图像处理)
- numpy >= 1.21.0 (数值计算)

## 配置文件

PDF解析器的配置在 `ultrasoundrag/config/config.yaml` 中：

```yaml
# PDF 解析配置
pdf_parse:
  max_chunk_size: 1500      # 单个chunk的大小
  overlap_ratio: 0.15       # 重叠窗口比例
  min_chunk_size: 100       # 最小chunk大小

# PDF 数据集配置
pdf:
  datasets:
    thesis_papers:
      enabled: true
      base_path: "data/thesis"           # PDF文件目录
      file_extensions: [".pdf"]
      collection_name: "thesis_papers"   # 数据库集合名
      description: "学术论文PDF数据集"
```

## 使用方法

### 基本使用

```python
from ultrasoundrag.data.loaders.pdf_parser import PDFParser

# 初始化解析器
parser = PDFParser(dataset_name="thesis_papers")

# 解析所有PDF文件
parsed_chunks = parser.parse_pdfs()

# 查看结果
for chunk in parsed_chunks:
    print(f"标题: {chunk['title']}")
    print(f"内容: {chunk['content'][:100]}...")
    print(f"图片数量: {len(chunk['image_paths'])}")
```

### 运行示例

```bash
python pdf_parser_example.py
```

## 输出格式

每个解析的文档块包含以下字段：

```python
{
    'id': int,                    # 文档块唯一ID
    'title': str,                 # 标题
    'content': str,               # 文本内容
    'image_paths': List[str],     # 相关图片路径列表
    'image_captions': List[str],  # 图片OCR识别文本列表
    'chunk_index': int,           # 块索引
    'pdf_file': str,              # 源PDF文件路径
    'document_name': str          # 文档名称
}
```

## 处理流程

1. **文件扫描**：扫描指定目录下的所有PDF文件
2. **文本提取**：使用PyMuPDF提取每页的文本内容
3. **图片提取**：提取PDF中的图片并保存为临时文件
4. **OCR识别**：使用PaddleOCR识别图片中的文字
5. **内容分块**：按照标题层级和语义完整性分块
6. **关联处理**：将图片OCR结果与相关文本块关联
7. **结果输出**：生成结构化的文档块列表

## 分块策略

PDF解析器采用智能分块策略：

1. **标题识别**：自动识别各种标题格式（数字标题、中文数字标题、英文标题等）
2. **段落分割**：优先按段落（双换行符）分割内容
3. **重叠窗口**：支持配置重叠比例，保持语义连续性
4. **最小长度**：过滤过短的无效片段
5. **语义完整性**：避免在句子中间截断内容

## 图片处理

- **图片提取**：自动提取PDF中的所有图片
- **OCR识别**：使用PaddleOCR进行中英文文字识别
- **置信度过滤**：只保留置信度>0.5的识别结果
- **关联匹配**：通过关键词匹配将图片与相关文本块关联

## 错误处理

- **文件不存在**：跳过不存在的PDF文件
- **图片提取失败**：记录错误并继续处理其他图片
- **OCR识别失败**：返回空字符串，不影响整体流程
- **配置错误**：提供清晰的错误信息

## 性能优化

- **批量处理**：支持批量处理多个PDF文件
- **内存管理**：及时释放PDF文档和图片资源
- **进度显示**：使用tqdm显示处理进度
- **临时文件清理**：建议定期清理OCR产生的临时图片文件

## 注意事项

1. **中文支持**：PaddleOCR默认支持中文识别
2. **内存使用**：处理大文件时注意内存使用情况
3. **临时文件**：OCR处理会产生临时图片文件，需要定期清理
4. **配置路径**：确保配置文件中的PDF目录路径正确
5. **依赖版本**：建议使用指定版本的依赖包以确保兼容性

## 扩展功能

可以根据需要扩展以下功能：

1. **多语言OCR**：支持更多语言的文字识别
2. **表格识别**：专门处理PDF中的表格内容
3. **公式识别**：识别数学公式和特殊符号
4. **布局分析**：更精确的文档布局分析
5. **向量化**：集成文本向量化功能
