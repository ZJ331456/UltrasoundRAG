"""
存储适配器模块

提供统一的存储接口，支持多种存储后端：
- milvus_store: Milvus向量数据库
- file_store: 文件系统存储
- cache_store: 缓存存储
"""

import os
from .milvus_store import MilvusManager

# 创建文件存储和缓存存储的基础实现
class FileStore:
    """文件系统存储适配器"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save(self, key: str, data: any, format: str = "json"):
        """保存数据到文件"""
        try:
            import json
            import pickle
            
            file_path = os.path.join(self.base_path, f"{key}.{format}")
            
            if format == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif format == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            return True
        except Exception as e:
            print(f"保存文件失败 {key}: {e}")
            return False
    
    def load(self, key: str, format: str = "json"):
        """从文件加载数据"""
        try:
            import json
            import pickle
            
            file_path = os.path.join(self.base_path, f"{key}.{format}")
            
            if not os.path.exists(file_path):
                return None
            
            if format == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif format == "pickle":
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"不支持的格式: {format}")
                
        except Exception as e:
            print(f"加载文件失败 {key}: {e}")
            return None
    
    def exists(self, key: str, format: str = "json") -> bool:
        """检查文件是否存在"""
        file_path = os.path.join(self.base_path, f"{key}.{format}")
        return os.path.exists(file_path)
    
    def delete(self, key: str, format: str = "json") -> bool:
        """删除文件"""
        try:
            file_path = os.path.join(self.base_path, f"{key}.{format}")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"删除文件失败 {key}: {e}")
            return False

class CacheStore:
    """缓存存储适配器"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
    
    def get(self, key: str):
        """获取缓存数据"""
        if key in self._cache:
            # 更新访问顺序
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: any):
        """设置缓存数据"""
        if key in self._cache:
            # 更新现有键
            self._cache[key] = value
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # 添加新键
            if len(self._cache) >= self.max_size:
                # LRU: 删除最久未使用的键
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = value
            self._access_order.append(key)
    
    def delete(self, key: str):
        """删除缓存数据"""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
            return True
        return False
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    def keys(self):
        """获取所有键"""
        return list(self._cache.keys())

__all__ = [
    'MilvusManager',
    'FileStore',
    'CacheStore'
]
