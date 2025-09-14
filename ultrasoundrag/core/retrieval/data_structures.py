"""数据结构定义模块
定义系统中共用的数据结构，避免循环导入问题

主要数据结构：
- RetrievalResult: 检索结果统一数据结构
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """检索结果统一数据结构"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_type: str  # 'text_to_text', 'text_to_image', 'image_to_text', 'image_to_image', 'exact_caption_match', 'image_caption_to_text'
    resource_collection: str = ""  # 数据来源的集合名称
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保metadata是字典类型
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # 添加默认字段
        if 'result_type' not in self.metadata:
            self.metadata['result_type'] = 'unknown'
        
        # 根据retrieval_type推断result_type
        if self.retrieval_type in ['text_to_image', 'image_to_image', 'exact_caption_match']:
            self.metadata['result_type'] = 'image'
        elif self.retrieval_type in ['text_to_text', 'image_to_text', 'image_caption_to_text']:
            self.metadata['result_type'] = 'text'
        
        # 为新的检索类型添加额外的元数据标记
        if self.retrieval_type.startswith('image_'):
            self.metadata['query_type'] = 'image_query'
        elif self.retrieval_type.startswith('text_'):
            self.metadata['query_type'] = 'text_query'
