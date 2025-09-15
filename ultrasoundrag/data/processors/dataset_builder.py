"""
数据集构建器
负责单个数据集的构建逻辑
"""

import os
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from ...utils import setup_logger
from ...config import config
from ...data.loaders.markdown_parser import MarkdownParser
from ...data.loaders.image_parser import ImageParser
from .database_operations import DatabaseOperations
from .embedding_generator import EmbeddingGenerator


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self):
        self.logger = setup_logger("DatasetBuilder")
        self.embedding_generator = EmbeddingGenerator()
    
    def build_markdown_dataset(self, dataset_name: str, dataset_cfg: Dict, 
                             recreate: bool, start_id: int) -> bool:
        """
        构建单个Markdown数据集
        
        Args:
            dataset_name: 数据集名称
            dataset_cfg: 数据集配置
            recreate: 是否重建集合
            start_id: 起始ID
            
        Returns:
            bool: 构建是否成功
        """
        try:
            # 获取集合名称
            target_collection = DatabaseOperations.get_collection_name_from_config('markdown', dataset_cfg)
            
            # 创建管理器
            manager = DatabaseOperations.prepare_collection_manager(
                collection_type="md", 
                collection_name=target_collection, 
                recreate=recreate
            )
            
            # 解析数据
            parser = MarkdownParser(dataset_name=dataset_name)
            parsed_data = parser.parse_markdowns()
            if not parsed_data:
                self.logger.warning(f"数据集 {dataset_name} 没有找到数据")
                return False
            
            # 分配ID
            parsed_data = DatabaseOperations.assign_global_ids(parsed_data, start_id)
            
            # 生成嵌入向量
            embeddings_qwen, embeddings_clip = self.embedding_generator.generate_text_embeddings(parsed_data)
            
            # 批量插入
            success = DatabaseOperations.batch_insert_with_progress(
                manager, parsed_data, embeddings_qwen, embeddings_clip
            )
            
            if success:
                self.logger.info(f"Markdown数据集 {dataset_name} 构建成功")
            else:
                self.logger.error(f"Markdown数据集 {dataset_name} 构建失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"构建Markdown数据集失败: {e}")
            return False
    
    def build_image_dataset(self, dataset_name: str, dataset_cfg: Dict, 
                          recreate: bool, start_id: int) -> bool:
        """
        构建单个图片数据集
        
        Args:
            dataset_name: 数据集名称
            dataset_cfg: 数据集配置
            recreate: 是否重建集合
            start_id: 起始ID
            
        Returns:
            bool: 构建是否成功
        """
        try:
            # 获取集合名称
            target_collection = DatabaseOperations.get_collection_name_from_config('image', dataset_cfg)
            
            # 创建管理器
            manager = DatabaseOperations.prepare_collection_manager(
                collection_type="image", 
                collection_name=target_collection, 
                recreate=recreate
            )
            
            # 解析数据
            parser = ImageParser(dataset_name=dataset_name)
            parsed_data = parser.parse_images()
            if not parsed_data:
                self.logger.warning(f"数据集 {dataset_name} 没有找到数据")
                return False
            
            # 分配ID
            parsed_data = DatabaseOperations.assign_global_ids(parsed_data, start_id)
            
            # 生成嵌入向量
            embeddings = self.embedding_generator.generate_image_embeddings(parsed_data)
            
            # 批量插入
            success = DatabaseOperations.batch_insert_image_data(manager, parsed_data, embeddings)
            
            if success:
                self.logger.info(f"图片数据集 {dataset_name} 构建成功")
            else:
                self.logger.error(f"图片数据集 {dataset_name} 构建失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"构建图片数据集失败: {e}")
            return False
    
    def build_dataset_by_type(self, dataset_type: str, dataset_name: str, 
                            dataset_cfg: Dict, recreate: bool, start_id: int) -> bool:
        """
        根据类型构建数据集
        
        Args:
            dataset_type: 数据集类型 ("markdown", "image")
            dataset_name: 数据集名称
            dataset_cfg: 数据集配置
            recreate: 是否重建集合
            start_id: 起始ID
            
        Returns:
            bool: 构建是否成功
        """
        if dataset_type == "markdown":
            return self.build_markdown_dataset(dataset_name, dataset_cfg, recreate, start_id)
        elif dataset_type == "image":
            return self.build_image_dataset(dataset_name, dataset_cfg, recreate, start_id)
        else:
            self.logger.error(f"不支持的数据集类型: {dataset_type}")
            return False
    
    def get_enabled_datasets(self, dataset_type: str) -> List[tuple]:
        """
        获取启用的数据集列表
        
        Args:
            dataset_type: 数据集类型
            
        Returns:
            启用的数据集列表 [(name, config), ...]
        """
        try:
            cfg = config['indexing'][dataset_type]
            datasets = cfg.get('datasets', {})
            
            enabled_datasets = []
            for name, dataset_cfg in datasets.items():
                if dataset_cfg.get('enabled', False):
                    enabled_datasets.append((name, dataset_cfg))
            
            return enabled_datasets
            
        except Exception as e:
            self.logger.error(f"获取启用数据集失败: {e}")
            return []
    
    def build_all_datasets_of_type(self, dataset_type: str, recreate: bool = False, 
                                 only_datasets: Optional[List[str]] = None) -> Dict:
        """
        构建指定类型的所有数据集
        
        Args:
            dataset_type: 数据集类型
            recreate: 是否重建集合
            only_datasets: 只构建指定的数据集
            
        Returns:
            构建结果字典
        """
        self.logger.info("=" * 50)
        self.logger.info(f"开始构建 {dataset_type} 索引")
        
        # 获取启用的数据集
        enabled_datasets = self.get_enabled_datasets(dataset_type)
        
        if only_datasets:
            enabled_datasets = [(name, cfg) for name, cfg in enabled_datasets 
                              if name in only_datasets]
        
        built_count = 0
        skipped_count = 0
        global_id = 1
        
        for dataset_name, dataset_cfg in tqdm(enabled_datasets, desc=f"{dataset_type} 数据集", unit="ds"):
            self.logger.info(f"处理 {dataset_type} 数据集: {dataset_name}")
            
            try:
                success = self.build_dataset_by_type(
                    dataset_type, dataset_name, dataset_cfg, recreate, global_id
                )
                
                if success:
                    built_count += 1
                else:
                    skipped_count += 1
                
                # 更新global_id
                global_id += 10000
                
            except Exception as e:
                self.logger.error(f"构建数据集 {dataset_name} 时发生错误: {type(e).__name__}: {e}")
                skipped_count += 1
        
        result = {
            'type': dataset_type,
            'built_count': built_count,
            'skipped_count': skipped_count,
            'total_datasets': len(enabled_datasets)
        }
        
        DatabaseOperations.log_operation_result(
            f"{dataset_type} 构建", built_count, len(enabled_datasets)
        )
        
        return result
