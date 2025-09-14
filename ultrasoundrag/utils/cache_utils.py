"""智能缓存工具模块
提供多层级缓存系统，提升检索系统性能

主要功能：
1. 向量嵌入缓存 - 缓存文本和图像的向量表示
2. 检索结果缓存 - 缓存检索结果避免重复计算
3. 模型缓存 - 缓存已加载的模型实例
4. LRU缓存管理 - 内存管理和过期策略
"""

import hashlib
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

from ultrasoundrag.utils.logger import setup_logger


@dataclass
class CacheEntry:
    """缓存条目"""
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.size_bytes == 0:
            try:
                self.size_bytes = len(pickle.dumps(self.data))
            except:
                self.size_bytes = 1024  # 默认估算


class LRUCache:
    """线程安全的LRU缓存"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512, ttl_seconds: int = 3600):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大条目数
            max_memory_mb: 最大内存使用(MB)
            ttl_seconds: 生存时间(秒)
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.logger = setup_logger(self.__class__.__name__)
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size': 0,
            'current_memory_bytes': 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查条目是否过期"""
        return time.time() - entry.timestamp > self.ttl_seconds
    
    def _evict_expired(self):
        """清理过期条目"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            entry = self.cache.pop(key)
            self.stats['current_size'] -= 1
            self.stats['current_memory_bytes'] -= entry.size_bytes
            self.stats['evictions'] += 1
    
    def _evict_lru(self):
        """清理最久未使用的条目"""
        while (len(self.cache) >= self.max_size or 
               self.stats['current_memory_bytes'] > self.max_memory_bytes) and self.cache:
            
            # 找到最久未访问的条目
            lru_key = min(self.cache.keys(), 
                         key=lambda k: self.cache[k].last_access)
            
            entry = self.cache.pop(lru_key)
            self.stats['current_size'] -= 1
            self.stats['current_memory_bytes'] -= entry.size_bytes
            self.stats['evictions'] += 1
    
    def get(self, key: str = None, *args, **kwargs) -> Optional[Any]:
        """获取缓存值"""
        if key is None:
            key = self._generate_key(*args, **kwargs)
        
        with self.lock:
            self._evict_expired()
            
            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry):
                    # 更新访问信息
                    entry.access_count += 1
                    entry.last_access = time.time()
                    # 移到末尾（最近使用）
                    self.cache.move_to_end(key)
                    self.stats['hits'] += 1
                    return entry.data
                else:
                    # 过期删除
                    del self.cache[key]
                    self.stats['current_size'] -= 1
                    self.stats['current_memory_bytes'] -= entry.size_bytes
            
            self.stats['misses'] += 1
            return None
    
    def put(self, data: Any, key: str = None, *args, **kwargs):
        """存储缓存值"""
        if key is None:
            key = self._generate_key(*args, **kwargs)
        
        with self.lock:
            # 清理过期和LRU
            self._evict_expired()
            self._evict_lru()
            
            # 创建新条目
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                last_access=time.time()
            )
            
            # 如果键已存在，更新统计
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats['current_memory_bytes'] -= old_entry.size_bytes
            else:
                self.stats['current_size'] += 1
            
            self.cache[key] = entry
            self.stats['current_memory_bytes'] += entry.size_bytes
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats.update({
                'current_size': 0,
                'current_memory_bytes': 0
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'memory_usage_mb': self.stats['current_memory_bytes'] / (1024 * 1024)
            }


class EmbeddingCache:
    """向量嵌入缓存"""
    
    def __init__(self, max_size: int = 5000, max_memory_mb: int = 256):
        self.cache = LRUCache(max_size=max_size, max_memory_mb=max_memory_mb, ttl_seconds=7200)
        self.logger = setup_logger(self.__class__.__name__)
    
    def get_text_embedding(self, text: str, model_name: str = "default") -> Optional[np.ndarray]:
        """获取文本嵌入"""
        key = f"text_{model_name}_{hashlib.md5(text.encode()).hexdigest()}"
        embedding = self.cache.get(key)
        if embedding is not None:
            self.logger.debug(f"文本嵌入缓存命中: {text[:50]}...")
            return embedding
        return None
    
    def put_text_embedding(self, text: str, embedding: np.ndarray, model_name: str = "default"):
        """存储文本嵌入"""
        key = f"text_{model_name}_{hashlib.md5(text.encode()).hexdigest()}"
        self.cache.put(embedding, key)
        self.logger.debug(f"文本嵌入已缓存: {text[:50]}...")
    
    def get_image_embedding(self, image_path: str, model_name: str = "default") -> Optional[np.ndarray]:
        """获取图像嵌入"""
        key = f"image_{model_name}_{hashlib.md5(image_path.encode()).hexdigest()}"
        embedding = self.cache.get(key)
        if embedding is not None:
            self.logger.debug(f"图像嵌入缓存命中: {image_path}")
            return embedding
        return None
    
    def put_image_embedding(self, image_path: str, embedding: np.ndarray, model_name: str = "default"):
        """存储图像嵌入"""
        key = f"image_{model_name}_{hashlib.md5(image_path.encode()).hexdigest()}"
        self.cache.put(embedding, key)
        self.logger.debug(f"图像嵌入已缓存: {image_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()


class ModelCache:
    """模型实例缓存"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.lock = threading.RLock()
        self.logger = setup_logger(self.__class__.__name__)
        self.load_times: Dict[str, float] = {}
    
    def get_model(self, model_name: str, model_path: str = None) -> Optional[Any]:
        """获取缓存的模型"""
        with self.lock:
            key = f"{model_name}_{model_path}" if model_path else model_name
            if key in self.models:
                self.logger.debug(f"模型缓存命中: {model_name}")
                return self.models[key]
            return None
    
    def put_model(self, model_name: str, model: Any, model_path: str = None):
        """缓存模型实例"""
        with self.lock:
            key = f"{model_name}_{model_path}" if model_path else model_name
            self.models[key] = model
            self.load_times[key] = time.time()
            self.logger.info(f"模型已缓存: {model_name}")
    
    def clear_model(self, model_name: str, model_path: str = None):
        """清除特定模型"""
        with self.lock:
            key = f"{model_name}_{model_path}" if model_path else model_name
            if key in self.models:
                del self.models[key]
                if key in self.load_times:
                    del self.load_times[key]
                self.logger.info(f"模型已清除: {model_name}")
    
    def clear_all(self):
        """清除所有模型"""
        with self.lock:
            self.models.clear()
            self.load_times.clear()
            self.logger.info("所有模型缓存已清除")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取模型缓存统计"""
        with self.lock:
            return {
                'cached_models': list(self.models.keys()),
                'model_count': len(self.models),
                'load_times': self.load_times.copy()
            }


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 0.1):
        """
        初始化批量处理器
        
        Args:
            batch_size: 批量大小
            max_wait_time: 最大等待时间(秒)
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.logger = setup_logger(self.__class__.__name__)
    
    def process_embeddings_batch(self, texts: List[str], embedding_func) -> List[np.ndarray]:
        """批量处理文本嵌入"""
        if not texts:
            return []
        
        embeddings = []
        start_time = time.time()
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = embedding_func(batch)
            
            if isinstance(batch_embeddings, np.ndarray):
                if batch_embeddings.ndim == 1:
                    embeddings.append(batch_embeddings)
                else:
                    embeddings.extend(batch_embeddings)
            else:
                embeddings.extend(batch_embeddings)
        
        process_time = time.time() - start_time
        self.logger.info(f"批量处理完成 - 文本数: {len(texts)}, 耗时: {process_time:.3f}s")
        
        return embeddings
    
    def process_queries_batch(self, queries: List[str], search_func) -> List[Any]:
        """批量处理查询"""
        if not queries:
            return []
        
        results = []
        start_time = time.time()
        
        # 分批处理
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_results = []
            
            for query in batch:
                try:
                    result = search_func(query)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"查询处理失败: {query[:50]}... - {e}")
                    batch_results.append([])
            
            results.extend(batch_results)
        
        process_time = time.time() - start_time
        self.logger.info(f"批量查询完成 - 查询数: {len(queries)}, 耗时: {process_time:.3f}s")
        
        return results


# 全局缓存实例
embedding_cache = EmbeddingCache()
model_cache = ModelCache()
batch_processor = BatchProcessor()


def get_embedding_cache() -> EmbeddingCache:
    """获取向量嵌入缓存实例"""
    return embedding_cache


def get_model_cache() -> ModelCache:
    """获取模型缓存实例"""
    return model_cache


def get_batch_processor() -> BatchProcessor:
    """获取批量处理器实例"""
    return batch_processor


def clear_all_caches():
    """清空所有缓存"""
    embedding_cache.clear()
    model_cache.clear_all()
    logger = setup_logger(__name__)
    logger.info("所有缓存已清空")
