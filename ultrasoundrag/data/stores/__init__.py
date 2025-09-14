"""
存储适配器模块

提供统一的存储接口，支持多种存储后端：
- milvus_store: Milvus向量数据库
- file_store: 文件系统存储
- cache_store: 缓存存储
"""

from .milvus_store import MilvusManager

# 创建文件存储和缓存存储的基础实现
class FileStore:
    """文件系统存储适配器"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def save(self, key: str, data: any):
        """保存数据到文件"""
        # TODO: 实现文件保存逻辑
        pass
    
    def load(self, key: str):
        """从文件加载数据"""
        # TODO: 实现文件加载逻辑
        pass

class CacheStore:
    """缓存存储适配器"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
    
    def get(self, key: str):
        """获取缓存数据"""
        return self._cache.get(key)
    
    def set(self, key: str, value: any):
        """设置缓存数据"""
        if len(self._cache) >= self.max_size:
            # 简单的LRU逻辑：删除第一个元素
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value

__all__ = [
    'MilvusManager',
    'FileStore',
    'CacheStore'
]
