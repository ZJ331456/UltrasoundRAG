#!/usr/bin/env python3
"""
模型管理器 - 统一管理所有模型实例
避免重复加载，提升初始化速度
"""

import os
import torch
from typing import Optional, Dict, Any
from ultrasoundrag.model.fetal_clip_model import FetalCLIPModel
from ultrasoundrag.config import config
from ultrasoundrag.utils.logger import setup_logger


class ModelManager:
    """
    全局模型管理器 - 单例模式
    统一管理所有模型实例，避免重复加载
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = setup_logger("ModelManager")
        self.models = {}
        self._initialized = True
        self.logger.info("模型管理器初始化完成")
    
    def get_fetal_clip_model(self, model_path: str = None, config_path: str = None, device: Optional[torch.device] = None, lazy: bool = True) -> FetalCLIPModel:
        """
        获取FetalCLIP模型实例（单例，支持延迟加载）
        
        Args:
            model_path: 模型权重文件路径
            config_path: 模型配置文件路径
            device: 计算设备
            lazy: 是否延迟加载（默认True）
            
        Returns:
            FetalCLIPModel实例
        """
        # 如果未提供路径，从配置中获取
        if model_path is None or config_path is None:
            image_parse_config = config['indexing']['image_parse']
            model_path = model_path or image_parse_config['model_path']
            config_path = config_path or image_parse_config['model_config_path']
        
        # 如果延迟加载且模型已存在，直接返回
        if lazy and FetalCLIPModel._initialized:
            return FetalCLIPModel._instance
        
        # 使用单例模式获取模型
        return FetalCLIPModel.get_instance(model_path, config_path, device)
    
    def get_embedding_model(self, provider_name: str = None):
        """
        获取嵌入模型实例
        
        Args:
            provider_name: 嵌入模型提供者名称
            
        Returns:
            嵌入模型实例
        """
        if provider_name is None:
            provider_name = config['embedding']['provider']
        
        # 如果已缓存，直接返回
        cache_key = f"embedding_{provider_name}"
        if cache_key in self.models:
            return self.models[cache_key]
        
        # 动态导入并创建嵌入模型
        from ultrasoundrag.utils.embedding_utils import embedding_provider
        model = embedding_provider[provider_name]
        self.models[cache_key] = model
        
        self.logger.info(f"嵌入模型 {provider_name} 已缓存")
        return model
    
    def clear_cache(self):
        """
        清空模型缓存
        """
        self.models.clear()
        FetalCLIPModel.reset_instance()
        self.logger.info("模型缓存已清空")
    
    def preload_models(self, models_to_load: list = None):
        """
        预加载指定模型
        
        Args:
            models_to_load: 要预加载的模型列表，None表示加载所有
        """
        if models_to_load is None:
            models_to_load = ['fetal_clip', 'embedding']
        
        self.logger.info(f"开始预加载模型: {models_to_load}")
        
        for model_name in models_to_load:
            try:
                if model_name == 'fetal_clip':
                    self.get_fetal_clip_model(lazy=False)
                    self.logger.info("✓ FetalCLIP模型预加载完成")
                elif model_name == 'embedding':
                    self.get_embedding_model()
                    self.logger.info("✓ 嵌入模型预加载完成")
            except Exception as e:
                self.logger.error(f"✗ {model_name}模型预加载失败: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            "fetal_clip_initialized": FetalCLIPModel._initialized,
            "cached_models": list(self.models.keys()),
            "total_cached": len(self.models)
        }
        return info


# 全局模型管理器实例
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """
    获取全局模型管理器实例
    
    Returns:
        ModelManager实例
    """
    return model_manager


def get_fetal_clip_model(model_path: str = None, config_path: str = None, device: Optional[torch.device] = None) -> FetalCLIPModel:
    """
    便捷函数：获取FetalCLIP模型实例
    
    Args:
        model_path: 模型权重文件路径
        config_path: 模型配置文件路径
        device: 计算设备
        
    Returns:
        FetalCLIPModel实例
    """
    return model_manager.get_fetal_clip_model(model_path, config_path, device)


def get_embedding_model(provider_name: str = None):
    """
    便捷函数：获取嵌入模型实例
    
    Args:
        provider_name: 嵌入模型提供者名称
        
    Returns:
        嵌入模型实例
    """
    return model_manager.get_embedding_model(provider_name)


if __name__ == "__main__":
    # 测试模型管理器
    print("=== 模型管理器测试 ===")
    
    # 获取模型管理器
    manager = get_model_manager()
    print(f"模型管理器信息: {manager.get_model_info()}")
    
    # 测试FetalCLIP模型获取
    try:
        clip_model = get_fetal_clip_model()
        print("✓ FetalCLIP模型获取成功")
    except Exception as e:
        print(f"✗ FetalCLIP模型获取失败: {e}")
    
    # 测试嵌入模型获取
    try:
        embedding_model = get_embedding_model()
        print("✓ 嵌入模型获取成功")
    except Exception as e:
        print(f"✗ 嵌入模型获取失败: {e}")
    
    print(f"最终模型信息: {manager.get_model_info()}")
