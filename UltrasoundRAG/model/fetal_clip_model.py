#!/usr/bin/env python3
"""
FetalCLIP模型加载器
用于加载预训练的FetalCLIP模型
"""

import os
import json
import torch
import torch.nn.functional as F
import open_clip
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from UltrasoundRAG.config.config import config


class FetalCLIPModel:
    """
    FetalCLIP模型加载和推理类 - 单例模式
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, model_path: str = None, config_path: str = None, device: Optional[torch.device] = None):
        """
        单例模式实现
        """
        if cls._instance is None:
            cls._instance = super(FetalCLIPModel, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: Optional[torch.device] = None):
        """
        初始化FetalCLIP模型
        
        Args:
            model_path: 模型权重文件路径
            config_path: 模型配置文件路径
            device: 计算设备
        """
        # 如果已经初始化过，直接返回
        if self._initialized:
            return
            
        self.model_path = model_path
        
        # 处理config_path，如果为None或不存在，使用默认路径
        if config_path is None:
            default_config_path = config_path
            self.config_path = default_config_path
            print(f"使用默认配置文件路径: {self.config_path}")
        else:
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(config_path):
                config_path = os.path.abspath(config_path)
            self.config_path = config_path
            
        # 检查配置文件是否存在
        if not os.path.exists(self.config_path):
            default_config_path = config_path
            if os.path.exists(default_config_path):
                print(f"配置文件 {self.config_path} 不存在，使用默认路径: {default_config_path}")
                self.config_path = default_config_path
            else:
                raise FileNotFoundError(f"配置文件不存在: {self.config_path} 且默认路径 {default_config_path} 也不存在")
            
        self.device = device or self._get_device()
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化模型
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        
        self._load_model()
        
        # 标记为已初始化
        self._initialized = True
        print(f"FetalCLIP模型加载完成，设备: {self.device}")
    
    @classmethod
    def get_instance(cls, model_path: str = None, config_path: str = None, device: Optional[torch.device] = None):
        """
        获取单例实例
        """
        if cls._instance is None:
            if model_path is None or config_path is None:
                raise ValueError("首次创建实例时必须提供model_path和config_path")
            cls._instance = cls(model_path, config_path, device)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        重置单例实例（主要用于测试）
        """
        cls._instance = None
        cls._initialized = False
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用GPU设备: {device}")
        else:
            device = torch.device('cpu')
            print("使用CPU设备")
        return device
    
    def _load_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def _load_model(self):
        """加载FetalCLIP模型"""
        # 获取架构名称
        arch_name = "ViT-B-16"  # 默认架构
        if 'arch_name' in self.config:
            arch_name = self.config['arch_name']
        elif 'model' in self.config and 'arch_name' in self.config['model']:
            arch_name = self.config['model']['arch_name']
        
        # 注册模型配置
        open_clip.factory._MODEL_CONFIGS[arch_name] = self.config
        
        # 创建模型和变换
        self.model, _, self.image_processor = open_clip.create_model_and_transforms(
            arch_name,
            pretrained=None  # 不使用预训练权重，稍后手动加载
        )
        
        # 获取tokenizer
        self.tokenizer = open_clip.get_tokenizer(arch_name)
        
        # 加载权重
        self._load_weights()
        
        # 移动到指定设备
        self.model.to(self.device)
        self.model.eval()
        
        print(f"FetalCLIP模型架构: {arch_name}")
    
    def _load_weights(self):
        """加载模型权重"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_path}")
        
        print(f"加载模型权重: {self.model_path}")
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理不同的权重格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                # 尝试自动识别state_dict
                for k, v in checkpoint.items():
                    if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
                        state_dict = v
                        print(f"自动识别到state_dict key: {k}")
                        break
                else:
                    raise RuntimeError(f"无法识别的模型权重格式: {checkpoint.keys()}")
        else:
            state_dict = checkpoint
        
        # 处理module.前缀
        if any(key.startswith("module.") for key in state_dict.keys()):
            print("检测到权重key带有module.前缀，自动去除...")
            state_dict = {key.replace("module.", "", 1): val for key, val in state_dict.items()}
        
        # 加载权重到模型
        self.model.load_state_dict(state_dict, strict=False)
        print("✓ 模型权重加载完成")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 图像张量 [B, C, H, W]
            
        Returns:
            image_features: 归一化的图像特征 [B, D]
        """
        with torch.no_grad():
            if images.device != self.device:
                images = images.to(self.device)
            
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
        return image_features
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        编码文本
        
        Args:
            text_tokens: 文本token [B, L]
            
        Returns:
            text_features: 归一化的文本特征 [B, D]
        """
        with torch.no_grad():
            if text_tokens.device != self.device:
                text_tokens = text_tokens.to(self.device)
            
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            
        return text_features
    
    def compute_similarity(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        计算图像和文本的相似度
        
        Args:
            images: 图像张量 [B, C, H, W]
            text_tokens: 文本token [B, L]
            
        Returns:
            similarity: 相似度矩阵 [B, B]
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        
        # 计算相似度矩阵
        similarity = image_features @ text_features.T
        
        return similarity
    
    def tokenize_text(self, texts: list) -> torch.Tensor:
        """
        文本分词
        
        Args:
            texts: 文本列表
            
        Returns:
            text_tokens: 文本token张量
        """
        return self.tokenizer(texts)
    
    def preprocess_image(self, images):
        """
        图像预处理
        
        Args:
            images: PIL图像或图像列表
            
        Returns:
            processed_images: 预处理后的图像张量
        """
        return self.image_processor(images)
    
    @classmethod
    def load_model(cls, 
                   model_path: str,
                   config_path: str,
                   device: Optional[torch.device] = None):
        """
        类方法：加载FetalCLIP模型
        
        Args:
            model_path: 模型权重文件路径
            config_path: 模型配置文件路径
            device: 计算设备
            
        Returns:
            FetalCLIPModel实例
        """
        return cls(model_path=model_path, config_path=config_path, device=device)


def load_fetal_clip_model(model_path: Optional[str] = None,
                          config_path: Optional[str] = None,
                          device: Optional[torch.device] = None) -> FetalCLIPModel:
    """
    便捷函数：加载FetalCLIP模型
    
    Args:
        model_path: 模型权重文件路径，如果为None则从配置文件获取
        config_path: 模型配置文件路径，如果为None则从配置文件获取
        device: 计算设备
        
    Returns:
        FetalCLIPModel实例
    """
    if model_path is None:
        model_path = config['image_search']['image_to_image']['embedding_model']
    if config_path is None:
        config_path = config['image_search']['image_to_image']['config_path']
    return FetalCLIPModel.load_model(model_path, config_path, device=device)


if __name__ == "__main__":
    # 测试模型加载
    try:
        model = load_fetal_clip_model()
        print("模型加载测试成功！")
        
        # 测试文本分词
        texts = ["这是一个测试文本"]
        tokens = model.tokenize_text(texts)
        print(f"文本分词结果形状: {tokens.shape}")
        
    except Exception as e:
        print(f"模型加载测试失败: {e}")