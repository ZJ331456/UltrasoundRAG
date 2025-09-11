"""
全局模型单例管理器 - 避免重复加载
解决当前项目中模型重复初始化的性能问题
"""

from typing import Dict, Optional, Any
import threading
from UltrasoundRAG.utils.logger import setup_logger


class GlobalModelSingleton:
    """全局模型单例管理器"""
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_once()
        return cls._instance
    
    def _init_once(self):
        """单次初始化"""
        if self._initialized:
            return
        self.logger = setup_logger("GlobalModelSingleton")
        self._initialized = True
        self.logger.info("全局模型单例管理器初始化完成")
    
    def get_or_create_model(self, model_key: str, creator_func, *args, **kwargs):
        """获取或创建模型（线程安全）"""
        if model_key in self._models:
            self.logger.debug(f"复用模型: {model_key}")
            return self._models[model_key]
        
        with self._lock:
            # 双重检查锁定
            if model_key in self._models:
                return self._models[model_key]
            
            self.logger.info(f"首次创建模型: {model_key}")
            model = creator_func(*args, **kwargs)
            self._models[model_key] = model
            return model
    
    def get_model_count(self) -> int:
        """获取已缓存的模型数量"""
        return len(self._models)
    
    def clear_models(self):
        """清空所有模型缓存"""
        with self._lock:
            self._models.clear()
            self.logger.info("已清空所有模型缓存")


# 全局单例实例
global_models = GlobalModelSingleton()


def get_shared_reranker():
    """获取共享的重排序模型"""
    def create_reranker():
        import torch
        from sentence_transformers import CrossEncoder
        model_path = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/models/jina-reranker-v2-base-multilingual"
        return CrossEncoder(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    return global_models.get_or_create_model("jina_reranker", create_reranker)


def get_shared_fetal_clip():
    """获取共享的FetalCLIP模型"""
    def create_fetal_clip():
        from UltrasoundRAG.model.model_manager import model_manager
        return model_manager.get_fetal_clip_model()
    
    return global_models.get_or_create_model("fetal_clip", create_fetal_clip)


def get_shared_embedding_model():
    """获取共享的嵌入模型"""
    def create_embedding():
        from UltrasoundRAG.utils.embedding_utils import embedding_provider
        from UltrasoundRAG.config import config
        provider_name = config['embedding']['provider']
        return embedding_provider[provider_name]
    
    return global_models.get_or_create_model("embedding_model", create_embedding)
