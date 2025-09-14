"""
数据访问层

负责数据的加载、处理和存储：
- loaders: 数据加载器（文档、图像、PDF等）
- processors: 数据处理器（文本处理、图像处理、嵌入生成）
- stores: 存储适配器（Milvus、文件系统、缓存）
"""

from .loaders import (
    MarkdownParser,
    ImageParser,
    DocumentLoader
)

from .stores import (
    MilvusManager,
    FileStore,
    CacheStore
)

__all__ = [
    'MarkdownParser',
    'ImageParser', 
    'DocumentLoader',
    'MilvusManager',
    'FileStore',
    'CacheStore'
]
