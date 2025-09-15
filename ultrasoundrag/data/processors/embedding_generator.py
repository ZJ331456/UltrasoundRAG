"""
嵌入向量生成器
专门负责各种类型的嵌入向量生成
"""

import os
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

from ...utils import setup_logger
from ...config import config
from .database_operations import DatabaseOperations


class EmbeddingGenerator:
    """嵌入向量生成器"""
    
    def __init__(self):
        self.logger = setup_logger("EmbeddingGenerator")
    
    def generate_text_embeddings(self, parsed_data: List[Dict]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        为解析后的文本数据生成嵌入向量
        
        Args:
            parsed_data: 解析后的数据列表
            
        Returns:
            Tuple[Qwen向量, CLIP向量]
        """
        texts = [chunk['content'] for chunk in parsed_data]
        
        return DatabaseOperations.generate_text_embeddings_batch(
            texts, 
            batch_size=64, 
            use_clip=True
        )
    
    def generate_image_embeddings(self, parsed_data: List[Dict]) -> List[List[float]]:
        """
        为图片数据生成嵌入向量
        
        Args:
            parsed_data: 解析后的图片数据列表
            
        Returns:
            图片嵌入向量列表
        """
        embeddings = []
        
        for item in parsed_data:
            if 'image_vector' in item:
                embeddings.append(item['image_vector'])
            else:
                # 如果没有预生成的向量，生成零向量
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def generate_caption_embeddings(self, parsed_data: List[Dict]) -> List[List[float]]:
        """
        为图片标题生成嵌入向量
        
        Args:
            parsed_data: 解析后的图片数据列表
            
        Returns:
            标题嵌入向量列表
        """
        captions = [item.get('caption', '') for item in parsed_data]
        
        if not captions:
            return []
        
        try:
            from ...model.fetal_clip_model import FetalCLIPModel
            clip_model = FetalCLIPModel(
                model_path=config['indexing']['image_parse']['model_path'],
                config_path=config['indexing']['image_parse']['model_config_path']
            )
            
            embeddings = []
            batch_size = 64
            
            for i in tqdm(range(0, len(captions), batch_size), desc="CLIP 标题嵌入", unit="batch"):
                batch_captions = captions[i:i + batch_size]
                tokens = clip_model.tokenize_text(batch_captions)
                feats = clip_model.encode_text(tokens).cpu().numpy()
                
                for row in feats:
                    vec = row.tolist()
                    if len(vec) != 768:
                        vec = [0.0] * 768
                    embeddings.append(vec)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"生成CLIP标题向量失败: {e}")
            return [[0.0] * 768 for _ in captions]
    
    def validate_embedding_dimensions(self, embeddings: List[List[float]], 
                                    expected_dims: List[int]) -> bool:
        """
        验证嵌入向量维度
        
        Args:
            embeddings: 嵌入向量列表
            expected_dims: 期望的维度列表
            
        Returns:
            验证是否通过
        """
        return DatabaseOperations.validate_embeddings(embeddings, expected_dims)
    
    def generate_embeddings_for_dataset(self, dataset_type: str, parsed_data: List[Dict]) -> Dict[str, List[List[float]]]:
        """
        根据数据集类型生成相应的嵌入向量
        
        Args:
            dataset_type: 数据集类型 ("markdown", "image", "pdf")
            parsed_data: 解析后的数据列表
            
        Returns:
            包含各种嵌入向量的字典
        """
        result = {}
        
        if dataset_type in ["markdown", "pdf"]:
            # 文本数据：生成Qwen和CLIP向量
            qwen_emb, clip_emb = self.generate_text_embeddings(parsed_data)
            result['embeddings_qwen'] = qwen_emb
            result['embeddings_clip'] = clip_emb
            
        elif dataset_type == "image":
            # 图片数据：生成图片向量和标题向量
            img_emb = self.generate_image_embeddings(parsed_data)
            cap_emb = self.generate_caption_embeddings(parsed_data)
            result['image_embeddings'] = img_emb
            result['caption_embeddings'] = cap_emb
        
        return result
