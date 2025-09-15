"""
数据加载器模块

负责从各种来源加载和解析数据：
- markdown_parser: Markdown文档解析器
- image_parser: 图像文件解析器
- pdf_parser: PDF文档解析器
- document_loader: 通用文档加载器
"""

from .markdown_parser import MarkdownParser
from .image_parser import ImageParser
from .pdf_parser import PDFParser

# 创建通用文档加载器类
class DocumentLoader:
    """通用文档加载器"""
    
    def __init__(self):
        self.markdown_parser = MarkdownParser()
        self.image_parser = ImageParser()
        self.pdf_parser = PDFParser()
    
    def load_markdown(self, file_path: str):
        """加载Markdown文档"""
        return self.markdown_parser.parse_file(file_path)
    
    def load_image(self, file_path: str):
        """加载图像文件"""
        return self.image_parser.parse_image(file_path)
    
    def load_pdf(self, file_path: str):
        """加载PDF文档"""
        return self.pdf_parser.process_document(file_path)

__all__ = [
    'MarkdownParser',
    'ImageParser',
    'PDFParser',
    'DocumentLoader'
]
