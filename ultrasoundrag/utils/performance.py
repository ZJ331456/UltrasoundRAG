"""
UltrasoundRAG 性能优化模块
提供缓存、批处理、模型管理和性能监控功能
"""

import time
import hashlib
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from functools import wraps, lru_cache
import pickle
import json
from concurrent.futures import ThreadPoolExecutor, Future
import weakref
import gc
import psutil
import numpy as np

from .exceptions import ModelError, UltrasoundRAGException


# ==================== 缓存系统 ====================

class LRUCache:
    """线程安全的LRU缓存实现"""
    
    def __init__(self, maxsize: int = 1000, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl  # 存活时间（秒）
        self.cache = OrderedDict()
        self.timestamps = {} if ttl else None
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get(self, key: str, default=None):
        """获取缓存值"""
        with self.lock:
            # 检查是否过期
            if self.ttl and key in self.timestamps:
                if time.time() - self.timestamps[key] > self.ttl:
                    self._remove_key(key)
                    self.stats["misses"] += 1
                    return default
            
            if key in self.cache:
                # 移动到末尾（最近使用）
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats["hits"] += 1
                return value
            
            self.stats["misses"] += 1
            return default
    
    def put(self, key: str, value: Any):
        """设置缓存值"""
        with self.lock:
            # 如果key已存在，更新
            if key in self.cache:
                self.cache.pop(key)
            # 如果超过最大大小，移除最旧的
            elif len(self.cache) >= self.maxsize:
                oldest_key = next(iter(self.cache))
                self._remove_key(oldest_key)
                self.stats["evictions"] += 1
            
            self.cache[key] = value
            if self.ttl:
                self.timestamps[key] = time.time()
    
    def _remove_key(self, key: str):
        """移除key"""
        self.cache.pop(key, None)
        if self.timestamps:
            self.timestamps.pop(key, None)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            if self.timestamps:
                self.timestamps.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "maxsize": self.maxsize
        }


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}
    
    def get_cache(
        self, 
        name: str, 
        maxsize: int = 1000, 
        ttl: Optional[float] = None
    ) -> LRUCache:
        """获取或创建命名缓存"""
        if name not in self.caches:
            self.caches[name] = LRUCache(maxsize, ttl)
        return self.caches[name]
    
    def clear_all(self):
        """清空所有缓存"""
        for cache in self.caches.values():
            cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有缓存统计"""
        return {name: cache.get_stats() for name, cache in self.caches.items()}


# 全局缓存管理器
cache_manager = CacheManager()


def cached(
    cache_name: str = "default",
    maxsize: int = 128,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """缓存装饰器"""
    def decorator(func):
        cache = cache_manager.get_cache(cache_name, maxsize, ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认key生成策略
                key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # 计算结果并缓存
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        return wrapper
    return decorator


# ==================== 批处理系统 ====================

@dataclass
class BatchJob:
    """批处理任务"""
    job_id: str
    data: Any
    callback: Optional[Callable] = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)


class BatchProcessor:
    """批处理器"""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 1.0,
        max_workers: int = 4,
        processor_func: Optional[Callable] = None
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_workers = max_workers
        self.processor_func = processor_func
        
        self.pending_jobs: List[BatchJob] = []
        self.results: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_batches": 0,
            "avg_batch_size": 0,
            "avg_processing_time": 0
        }
        
        # 启动批处理线程
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
    
    def submit(self, job_id: str, data: Any, callback: Optional[Callable] = None) -> str:
        """提交批处理任务"""
        job = BatchJob(job_id=job_id, data=data, callback=callback)
        
        with self.condition:
            self.pending_jobs.append(job)
            self.stats["total_jobs"] += 1
            self.condition.notify()
        
        return job_id
    
    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        start_time = time.time()
        while True:
            if job_id in self.results:
                return self.results.pop(job_id)
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"任务 {job_id} 超时")
            
            time.sleep(0.01)
    
    def _process_loop(self):
        """批处理循环"""
        while True:
            batch = self._get_batch()
            if batch:
                self._process_batch(batch)
    
    def _get_batch(self) -> List[BatchJob]:
        """获取一批任务"""
        with self.condition:
            # 等待任务或超时
            while not self.pending_jobs:
                self.condition.wait(timeout=self.max_wait_time)
                if not self.pending_jobs:
                    return []
            
            # 获取批次
            batch_size = min(self.batch_size, len(self.pending_jobs))
            batch = self.pending_jobs[:batch_size]
            self.pending_jobs = self.pending_jobs[batch_size:]
            
            return batch
    
    def _process_batch(self, batch: List[BatchJob]):
        """处理一批任务"""
        if not self.processor_func:
            return
        
        start_time = time.time()
        
        try:
            # 提取数据
            batch_data = [job.data for job in batch]
            
            # 批量处理
            batch_results = self.processor_func(batch_data)
            
            # 存储结果
            for job, result in zip(batch, batch_results):
                self.results[job.job_id] = result
                if job.callback:
                    job.callback(result)
                self.stats["completed_jobs"] += 1
            
        except Exception as e:
            # 处理失败
            for job in batch:
                self.results[job.job_id] = UltrasoundRAGException(
                    f"批处理失败: {str(e)}",
                    original_exception=e
                )
                self.stats["failed_jobs"] += 1
        
        # 更新统计
        processing_time = time.time() - start_time
        self.stats["total_batches"] += 1
        self.stats["avg_batch_size"] = (
            self.stats["avg_batch_size"] * (self.stats["total_batches"] - 1) + len(batch)
        ) / self.stats["total_batches"]
        self.stats["avg_processing_time"] = (
            self.stats["avg_processing_time"] * (self.stats["total_batches"] - 1) + processing_time
        ) / self.stats["total_batches"]


# ==================== 智能模型管理 ====================

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    model_instance: Any
    memory_usage: float  # MB
    load_time: float
    last_used: float
    usage_count: int = 0
    is_persistent: bool = False


class SmartModelManager:
    """智能模型管理器"""
    
    def __init__(
        self,
        max_memory_mb: float = 4096,  # 4GB
        cleanup_interval: float = 300,  # 5分钟
        min_usage_for_persistence: int = 10
    ):
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = cleanup_interval
        self.min_usage_for_persistence = min_usage_for_persistence
        
        self.models: Dict[str, ModelInfo] = {}
        self.model_factories: Dict[str, Callable] = {}
        self.lock = threading.RLock()
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def register_model_factory(self, name: str, factory_func: Callable):
        """注册模型工厂函数"""
        self.model_factories[name] = factory_func
    
    def get_model(self, name: str, **kwargs) -> Any:
        """获取模型实例"""
        with self.lock:
            # 如果模型已加载，直接返回
            if name in self.models:
                model_info = self.models[name]
                model_info.last_used = time.time()
                model_info.usage_count += 1
                return model_info.model_instance
            
            # 加载新模型
            return self._load_model(name, **kwargs)
    
    def _load_model(self, name: str, **kwargs) -> Any:
        """加载模型"""
        if name not in self.model_factories:
            raise ModelError(f"未注册的模型: {name}")
        
        # 检查内存使用
        current_memory = self._get_total_memory_usage()
        if current_memory > self.max_memory_mb * 0.8:  # 80%阈值
            self._cleanup_models(force=True)
        
        # 加载模型
        start_time = time.time()
        try:
            model_instance = self.model_factories[name](**kwargs)
            load_time = time.time() - start_time
            
            # 估算内存使用
            memory_usage = self._estimate_model_memory(model_instance)
            
            # 创建模型信息
            model_info = ModelInfo(
                name=name,
                model_instance=model_instance,
                memory_usage=memory_usage,
                load_time=load_time,
                last_used=time.time(),
                usage_count=1
            )
            
            self.models[name] = model_info
            return model_instance
            
        except Exception as e:
            raise ModelError(f"模型加载失败: {name}", original_exception=e)
    
    def _estimate_model_memory(self, model_instance: Any) -> float:
        """估算模型内存使用（MB）"""
        try:
            # 尝试获取模型参数大小
            if hasattr(model_instance, 'parameters'):
                # PyTorch模型
                total_params = sum(p.numel() for p in model_instance.parameters())
                return total_params * 4 / (1024 * 1024)  # 假设float32
            elif hasattr(model_instance, 'get_weights'):
                # TensorFlow模型
                weights = model_instance.get_weights()
                total_size = sum(w.nbytes for w in weights)
                return total_size / (1024 * 1024)
            else:
                # 默认估算
                return 500  # 500MB默认值
        except:
            return 500
    
    def _get_total_memory_usage(self) -> float:
        """获取总内存使用（MB）"""
        return sum(model.memory_usage for model in self.models.values())
    
    def _cleanup_models(self, force: bool = False):
        """清理模型"""
        current_time = time.time()
        models_to_remove = []
        
        for name, model_info in self.models.items():
            # 持久化模型不清理
            if model_info.is_persistent:
                continue
            
            # 检查是否应该持久化
            if model_info.usage_count >= self.min_usage_for_persistence:
                model_info.is_persistent = True
                continue
            
            # 检查是否长时间未使用
            if force or (current_time - model_info.last_used) > self.cleanup_interval:
                models_to_remove.append(name)
        
        # 移除模型
        for name in models_to_remove:
            self._unload_model(name)
    
    def _unload_model(self, name: str):
        """卸载模型"""
        if name in self.models:
            # 清理GPU内存（如果适用）
            model_instance = self.models[name].model_instance
            if hasattr(model_instance, 'cpu'):
                model_instance.cpu()
            
            del self.models[name]
            gc.collect()  # 强制垃圾回收
    
    def _cleanup_loop(self):
        """清理循环"""
        while True:
            time.sleep(self.cleanup_interval)
            with self.lock:
                self._cleanup_models()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            total_memory = self._get_total_memory_usage()
            system_memory = psutil.virtual_memory()
            
            return {
                "loaded_models": len(self.models),
                "total_memory_mb": total_memory,
                "max_memory_mb": self.max_memory_mb,
                "memory_usage_ratio": total_memory / self.max_memory_mb,
                "system_memory_gb": system_memory.total / (1024**3),
                "system_memory_available_gb": system_memory.available / (1024**3),
                "models": {
                    name: {
                        "memory_mb": info.memory_usage,
                        "usage_count": info.usage_count,
                        "last_used": info.last_used,
                        "is_persistent": info.is_persistent
                    }
                    for name, info in self.models.items()
                }
            }


# 全局模型管理器
smart_model_manager = SmartModelManager()


# ==================== 性能监控 ====================

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_time(self, metric_name: str, duration: float):
        """记录时间指标"""
        with self.lock:
            self.metrics[metric_name].append(duration)
            # 保持最近1000条记录
            if len(self.metrics[metric_name]) > 1000:
                self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """增加计数器"""
        with self.lock:
            self.counters[counter_name] += value
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            stats = {
                "counters": dict(self.counters),
                "metrics": {}
            }
            
            for metric_name, values in self.metrics.items():
                if values:
                    stats["metrics"][metric_name] = {
                        "count": len(values),
                        "avg": np.mean(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "p50": np.percentile(values, 50),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99)
                    }
            
            return stats
    
    def clear_metrics(self):
        """清空指标"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()


# 全局性能监控器
performance_monitor = PerformanceMonitor()


def monitor_performance(metric_name: str):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                performance_monitor.increment_counter(f"{metric_name}_success")
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{metric_name}_error")
                raise
            finally:
                duration = time.time() - start_time
                performance_monitor.record_time(metric_name, duration)
        return wrapper
    return decorator


# ==================== 异步批处理 ====================

class AsyncBatchProcessor:
    """异步批处理器"""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 1.0,
        processor_func: Optional[Callable] = None
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor_func = processor_func
        
        self.pending_jobs: List[Tuple[str, Any, asyncio.Future]] = []
        self.lock = asyncio.Lock()
        
        # 启动处理任务
        self.processing_task = None
        self.should_stop = False
    
    async def submit(self, job_id: str, data: Any) -> Any:
        """提交异步批处理任务"""
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_jobs.append((job_id, data, future))
            
            # 启动处理任务（如果还没有）
            if not self.processing_task:
                self.processing_task = asyncio.create_task(self._process_loop())
        
        return await future
    
    async def _process_loop(self):
        """异步处理循环"""
        while not self.should_stop:
            batch = await self._get_batch()
            if batch:
                await self._process_batch(batch)
            else:
                await asyncio.sleep(0.1)
    
    async def _get_batch(self) -> List[Tuple[str, Any, asyncio.Future]]:
        """获取一批任务"""
        async with self.lock:
            if not self.pending_jobs:
                return []
            
            batch_size = min(self.batch_size, len(self.pending_jobs))
            batch = self.pending_jobs[:batch_size]
            self.pending_jobs = self.pending_jobs[batch_size:]
            
            return batch
    
    async def _process_batch(self, batch: List[Tuple[str, Any, asyncio.Future]]):
        """处理一批任务"""
        if not self.processor_func:
            return
        
        try:
            # 提取数据
            batch_data = [data for _, data, _ in batch]
            
            # 批量处理
            if asyncio.iscoroutinefunction(self.processor_func):
                batch_results = await self.processor_func(batch_data)
            else:
                batch_results = self.processor_func(batch_data)
            
            # 设置结果
            for (_, _, future), result in zip(batch, batch_results):
                if not future.done():
                    future.set_result(result)
            
        except Exception as e:
            # 设置异常
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(e)
    
    async def stop(self):
        """停止处理器"""
        self.should_stop = True
        if self.processing_task:
            await self.processing_task
