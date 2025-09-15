"""
嵌入处理器

负责文本和图像的向量化处理：
- 文本嵌入生成
- 图像嵌入生成
- 嵌入质量评估
- 批量嵌入处理
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class EmbeddingResult:
    """嵌入结果数据结构"""
    content: str
    embedding: np.ndarray
    model_name: str
    dimension: int
    quality_score: float = 0.0
    metadata: Dict = None

class BaseEmbeddingModel(ABC):
    """嵌入模型基类"""
    
    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        pass
    
    @abstractmethod
    def encode_image(self, images: List[torch.Tensor]) -> np.ndarray:
        """编码图像"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        pass

class FetalCLIPEmbeddingModel(BaseEmbeddingModel):
    """FetalCLIP嵌入模型"""
    
    def __init__(self, model_path: str, config_path: str):
        """
        初始化FetalCLIP模型
        
        Args:
            model_path: 模型路径
            config_path: 配置文件路径
        """
        try:
            from ultrasoundrag.model.fetal_clip_model import FetalCLIPModel
            self.model = FetalCLIPModel(model_path=model_path, config_path=config_path)
            self.dimension = 512  # FetalCLIP的嵌入维度
        except Exception as e:
            print(f"FetalCLIP模型加载失败: {e}")
            raise
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        try:
            embeddings = []
            for text in texts:
                embedding = self.model.encode_text(text)
                embeddings.append(embedding)
            return np.array(embeddings)
        except Exception as e:
            print(f"文本编码失败: {e}")
            return np.array([])
    
    def encode_image(self, images: List[torch.Tensor]) -> np.ndarray:
        """编码图像"""
        try:
            embeddings = []
            for image in images:
                embedding = self.model.encode_image(image)
                embeddings.append(embedding)
            return np.array(embeddings)
        except Exception as e:
            print(f"图像编码失败: {e}")
            return np.array([])
    
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        return self.dimension

class EmbeddingProcessor:
    """嵌入处理器"""
    
    def __init__(self, 
                 text_model: Optional[BaseEmbeddingModel] = None,
                 image_model: Optional[BaseEmbeddingModel] = None,
                 batch_size: int = 32):
        """
        初始化嵌入处理器
        
        Args:
            text_model: 文本嵌入模型
            image_model: 图像嵌入模型
            batch_size: 批处理大小
        """
        self.text_model = text_model
        self.image_model = image_model
        self.batch_size = batch_size
    
    def set_text_model(self, model: BaseEmbeddingModel):
        """设置文本模型"""
        self.text_model = model
    
    def set_image_model(self, model: BaseEmbeddingModel):
        """设置图像模型"""
        self.image_model = model
    
    def encode_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        编码文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入结果列表
        """
        if not self.text_model:
            raise ValueError("文本模型未设置")
        
        results = []
        
        # 批量处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # 编码文本
                embeddings = self.text_model.encode_text(batch_texts)
                
                # 创建结果
                for j, (text, embedding) in enumerate(zip(batch_texts, embeddings)):
                    quality_score = self._calculate_embedding_quality(embedding)
                    
                    result = EmbeddingResult(
                        content=text,
                        embedding=embedding,
                        model_name=self.text_model.__class__.__name__,
                        dimension=len(embedding),
                        quality_score=quality_score,
                        metadata={'batch_index': i + j}
                    )
                    results.append(result)
                    
            except Exception as e:
                print(f"批量文本编码失败: {e}")
                continue
        
        return results
    
    def encode_images(self, images: List[torch.Tensor]) -> List[EmbeddingResult]:
        """
        编码图像列表
        
        Args:
            images: 图像张量列表
            
        Returns:
            嵌入结果列表
        """
        if not self.image_model:
            raise ValueError("图像模型未设置")
        
        results = []
        
        # 批量处理
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            try:
                # 编码图像
                embeddings = self.image_model.encode_image(batch_images)
                
                # 创建结果
                for j, (image, embedding) in enumerate(zip(batch_images, embeddings)):
                    quality_score = self._calculate_embedding_quality(embedding)
                    
                    result = EmbeddingResult(
                        content=f"image_{i + j}",
                        embedding=embedding,
                        model_name=self.image_model.__class__.__name__,
                        dimension=len(embedding),
                        quality_score=quality_score,
                        metadata={'batch_index': i + j}
                    )
                    results.append(result)
                    
            except Exception as e:
                print(f"批量图像编码失败: {e}")
                continue
        
        return results
    
    def _calculate_embedding_quality(self, embedding: np.ndarray) -> float:
        """
        计算嵌入质量分数
        
        Args:
            embedding: 嵌入向量
            
        Returns:
            质量分数 (0-1)
        """
        try:
            # 检查向量是否为零向量
            if np.allclose(embedding, 0):
                return 0.0
            
            # 计算向量的模长
            norm = np.linalg.norm(embedding)
            
            # 计算向量的方差（衡量分布的均匀性）
            variance = np.var(embedding)
            
            # 计算向量的稀疏性
            sparsity = np.count_nonzero(embedding) / len(embedding)
            
            # 综合质量分数
            quality_score = (
                min(norm / 10.0, 1.0) * 0.4 +      # 模长分数
                min(variance, 1.0) * 0.3 +          # 方差分数
                sparsity * 0.3                      # 稀疏性分数
            )
            
            return min(quality_score, 1.0)
        except Exception as e:
            print(f"计算嵌入质量分数失败: {e}")
            return 0.0
    
    def filter_quality_embeddings(self, 
                                 embeddings: List[EmbeddingResult],
                                 quality_threshold: float = 0.5) -> List[EmbeddingResult]:
        """
        过滤高质量嵌入
        
        Args:
            embeddings: 嵌入结果列表
            quality_threshold: 质量阈值
            
        Returns:
            高质量嵌入结果列表
        """
        return [
            emb for emb in embeddings 
            if emb.quality_score >= quality_threshold
        ]
    
    def normalize_embeddings(self, embeddings: List[EmbeddingResult]) -> List[EmbeddingResult]:
        """
        标准化嵌入向量
        
        Args:
            embeddings: 嵌入结果列表
            
        Returns:
            标准化后的嵌入结果列表
        """
        normalized = []
        
        for emb in embeddings:
            try:
                # L2标准化
                normalized_embedding = emb.embedding / np.linalg.norm(emb.embedding)
                
                # 创建新的结果对象
                normalized_emb = EmbeddingResult(
                    content=emb.content,
                    embedding=normalized_embedding,
                    model_name=emb.model_name,
                    dimension=emb.dimension,
                    quality_score=emb.quality_score,
                    metadata=emb.metadata
                )
                normalized.append(normalized_emb)
                
            except Exception as e:
                print(f"嵌入标准化失败: {e}")
                continue
        
        return normalized
    
    def batch_process(self, 
                     texts: Optional[List[str]] = None,
                     images: Optional[List[torch.Tensor]] = None) -> Dict[str, List[EmbeddingResult]]:
        """
        批量处理文本和图像
        
        Args:
            texts: 文本列表
            images: 图像张量列表
            
        Returns:
            处理结果字典
        """
        results = {}
        
        if texts and self.text_model:
            results['texts'] = self.encode_texts(texts)
        
        if images and self.image_model:
            results['images'] = self.encode_images(images)
        
        return results
