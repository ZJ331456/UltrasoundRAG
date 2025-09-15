"""
数据库操作通用工具类
提取通用的数据库操作逻辑，减少代码重复
"""

import os
from typing import List, Optional, Dict, Tuple, Any
from tqdm import tqdm

from ...utils import setup_logger
from ...config import config


class DatabaseOperations:
    """数据库操作通用工具类"""
    
    def __init__(self):
        self.logger = setup_logger("DatabaseOperations")
    
    @staticmethod
    def validate_embeddings(embeddings: List[List[float]], expected_dims: List[int]) -> bool:
        """
        验证嵌入向量的维度
        
        Args:
            embeddings: 嵌入向量列表
            expected_dims: 期望的维度列表
            
        Returns:
            bool: 验证是否通过
        """
        if not embeddings:
            return False
        
        for i, emb in enumerate(embeddings):
            if len(emb) not in expected_dims:
                print(f"警告：第 {i + 1} 个向量维度为 {len(emb)}，期望 {expected_dims}")
                return False
        
        return True
    
    @staticmethod
    def batch_insert_with_progress(manager, data_list: List[Dict], 
                                 embeddings_qwen: Optional[List[List[float]]] = None,
                                 embeddings_clip: Optional[List[List[float]]] = None,
                                 chunk_size: int = 1000) -> bool:
        """
        带进度显示的批量插入数据
        
        Args:
            manager: MilvusManager实例
            data_list: 数据列表
            embeddings_qwen: Qwen嵌入向量
            embeddings_clip: CLIP嵌入向量
            chunk_size: 批处理大小
            
        Returns:
            bool: 插入是否成功
        """
        if not data_list:
            return True
        
        insert_ok = True
        
        for i in tqdm(range(0, len(data_list), chunk_size), desc="写入数据库", unit="chunk"):
            part_data = data_list[i:i + chunk_size]
            part_qwen = embeddings_qwen[i:i + chunk_size] if embeddings_qwen else None
            part_clip = embeddings_clip[i:i + chunk_size] if embeddings_clip else None
            
            ok = manager.insert_data(
                part_data, 
                embeddings_qwen=part_qwen, 
                embeddings_clip=part_clip
            )
            insert_ok = insert_ok and ok
        
        return insert_ok
    
    @staticmethod
    def batch_insert_image_data(manager, data_list: List[Dict], 
                              embeddings: List[List[float]], 
                              chunk_size: int = 1000) -> bool:
        """
        批量插入图片数据
        
        Args:
            manager: MilvusManager实例
            data_list: 图片数据列表
            embeddings: 图片嵌入向量
            chunk_size: 批处理大小
            
        Returns:
            bool: 插入是否成功
        """
        if not data_list:
            return True
        
        insert_ok = True
        
        for i in tqdm(range(0, len(data_list), chunk_size), desc="写入数据库(图片)", unit="chunk"):
            part_data = data_list[i:i + chunk_size]
            part_emb = embeddings[i:i + chunk_size]
            ok = manager.insert_data(part_data, part_emb)
            insert_ok = insert_ok and ok
        
        return insert_ok
    
    @staticmethod
    def generate_text_embeddings_batch(texts: List[str], 
                                     batch_size: int = 64,
                                     use_clip: bool = True) -> Tuple[List[List[float]], List[List[float]]]:
        """
        批量生成文本嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            use_clip: 是否生成CLIP向量
            
        Returns:
            Tuple[Qwen向量, CLIP向量]
        """
        from ...utils.embedding_utils import embedding_provider
        
        # 生成Qwen向量
        embedder_qwen = embedding_provider[config['embedding']['provider']]
        embeddings_qwen = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Qwen 文本嵌入", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_emb = embedder_qwen.embed_documents(batch_texts)
            except Exception as e:
                print(f"Qwen 嵌入失败批次 {i // batch_size}: {e}")
                batch_emb = [[0.0] * 1024 for _ in batch_texts]
            embeddings_qwen.extend(batch_emb)
        
        # 生成CLIP向量（可选）
        embeddings_clip = []
        if use_clip:
            try:
                from ...model.fetal_clip_model import FetalCLIPModel
                clip_model = FetalCLIPModel(
                    model_path=config['indexing']['image_parse']['model_path'],
                    config_path=config['indexing']['image_parse']['model_config_path']
                )
                
                for i in tqdm(range(0, len(texts), batch_size), desc="CLIP 文本嵌入", unit="batch"):
                    batch_texts = texts[i:i + batch_size]
                    tokens = clip_model.tokenize_text(batch_texts)
                    feats = clip_model.encode_text(tokens).cpu().numpy()
                    for row in feats:
                        vec = row.tolist()
                        if len(vec) != 768:
                            vec = [0.0] * 768
                        embeddings_clip.append(vec)
            except Exception as e:
                print(f"生成CLIP文本向量失败: {e}")
                embeddings_clip = [[0.0] * 768 for _ in texts]
        else:
            embeddings_clip = [[0.0] * 768 for _ in texts]
        
        return embeddings_qwen, embeddings_clip
    
    @staticmethod
    def prepare_collection_manager(collection_type: str, collection_name: str, 
                                 recreate: bool = False) -> Any:
        """
        准备集合管理器
        
        Args:
            collection_type: 集合类型 ("md", "image", "pdf")
            collection_name: 集合名称
            recreate: 是否重建集合
            
        Returns:
            MilvusManager实例
        """
        from ...data.stores.milvus_store import MilvusManager
        
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name)
        
        if recreate:
            print(f"重建 {collection_type} 集合: {collection_name}")
            manager.drop_collection()
            manager._setup_collection()
        
        return manager
    
    @staticmethod
    def get_dataset_config(dataset_type: str, dataset_name: str) -> Dict:
        """
        获取数据集配置
        
        Args:
            dataset_type: 数据集类型 ("markdown", "image")
            dataset_name: 数据集名称
            
        Returns:
            数据集配置字典
        """
        cfg = config['indexing'][dataset_type]['datasets'][dataset_name]
        return cfg
    
    @staticmethod
    def get_collection_name_from_config(dataset_type: str, dataset_config: Dict) -> str:
        """
        从配置中获取集合名称
        
        Args:
            dataset_type: 数据集类型
            dataset_config: 数据集配置
            
        Returns:
            集合名称
        """
        # 优先使用数据集配置中的集合名
        if 'collection_name' in dataset_config:
            return dataset_config['collection_name']
        
        # 回退到默认集合名
        default_collections = config['indexing'][dataset_type].get('collections', {})
        if dataset_type == 'markdown':
            return default_collections.get('md_documents', 'md_documents')
        elif dataset_type == 'image':
            return default_collections.get('images', 'images')
        elif dataset_type == 'pdf':
            return default_collections.get('pdf_documents', 'pdf_documents')
        
        return f"{dataset_type}_documents"
    
    @staticmethod
    def assign_global_ids(data_list: List[Dict], start_id: int) -> List[Dict]:
        """
        为数据分配全局ID
        
        Args:
            data_list: 数据列表
            start_id: 起始ID
            
        Returns:
            分配ID后的数据列表
        """
        for i, item in enumerate(data_list):
            item['id'] = int(start_id + i)
        return data_list
    
    @staticmethod
    def log_operation_result(operation_name: str, success_count: int, 
                           total_count: int, errors: List[str] = None):
        """
        记录操作结果
        
        Args:
            operation_name: 操作名称
            success_count: 成功数量
            total_count: 总数量
            errors: 错误列表
        """
        logger = setup_logger("OperationLogger")
        
        logger.info(f"{operation_name} 完成：成功 {success_count}/{total_count}")
        
        if errors:
            logger.warning(f"遇到 {len(errors)} 个错误")
            for error in errors[:5]:  # 只显示前5个错误
                logger.error(f"错误: {error}")
            if len(errors) > 5:
                logger.warning(f"... 还有 {len(errors) - 5} 个错误未显示")
