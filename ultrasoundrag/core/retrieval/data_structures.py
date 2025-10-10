"""数据结构定义模块
定义系统中共用的数据结构，避免循环导入问题

主要数据结构：
- RetrievalResult: 检索结果统一数据结构
"""

from typing import Dict, Any
from dataclasses import dataclass
import numpy as np


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
        # 确保metadata是字典类型并清理NumPy类型
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        else:
            self.metadata = self._clean_numpy_types(self.metadata)
        
        # 确保score是Python原生float类型（避免NumPy类型序列化问题）
        if hasattr(self.score, 'item'):  # NumPy标量类型有item()方法
            object.__setattr__(self, 'score', float(self.score.item()))
        elif isinstance(self.score, np.ndarray):  # NumPy数组
            if self.score.size == 1:  # 单元素数组
                object.__setattr__(self, 'score', float(self.score.item()))
            else:  # 多元素数组，取第一个元素
                object.__setattr__(self, 'score', float(self.score.flat[0]))
        else:
            object.__setattr__(self, 'score', float(self.score))
        
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
    
    def __setattr__(self, name: str, value: Any):
        """拦截属性赋值，确保所有数值类型都是Python原生类型"""
        if name == 'score':
            # 转换NumPy类型为Python原生float
            if isinstance(value, np.ndarray):  # NumPy数组
                if value.size == 1:  # 单元素数组
                    value = float(value.item())
                else:  # 多元素数组，取第一个元素
                    value = float(value.flat[0])
            elif hasattr(value, 'item'):  # NumPy标量类型有item()方法
                value = float(value.item())
            else:
                value = float(value)
        elif name == 'metadata' and isinstance(value, dict):
            # 递归清理metadata中的NumPy类型
            value = self._clean_numpy_types(value)
        object.__setattr__(self, name, value)
    
    def _clean_numpy_types(self, obj: Any) -> Any:
        """递归清理对象中的NumPy类型"""
        if hasattr(obj, 'item'):  # NumPy标量类型
            return float(obj.item())
        elif isinstance(obj, np.ndarray):  # NumPy数组
            if obj.size == 1:  # 单元素数组
                return float(obj.item())
            else:  # 多元素数组
                return obj.tolist()  # 转换为Python列表
        elif isinstance(obj, dict):
            return {k: self._clean_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._clean_numpy_types(item) for item in obj)
        else:
            return obj
