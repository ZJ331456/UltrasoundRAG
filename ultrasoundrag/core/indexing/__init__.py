"""
索引构建和管理模块 - 重构版本

负责各种类型索引的构建、更新和管理：
- builders: 索引构建器（重构版本，使用模块化工具类）
- database_operations: 通用数据库操作工具
- embedding_generator: 嵌入向量生成器
- dataset_builder: 数据集构建器
"""

from .builders import (
    IndexBuilder,
    build_markdown_index,
    build_image_index,
    build_all_indexes,
    update_documents_incremental,
    update_single_document,
    delete_single_document,
    delete_collection,
    delete_database,
    create_collection,
    add_data_to_collection,
    list_collections,
    list_databases,
    get_collection_info
)

# 新增的工具类
from ...data.processors.database_operations import DatabaseOperations
from ...data.processors.embedding_generator import EmbeddingGenerator
from ...data.processors.dataset_builder import DatasetBuilder

__all__ = [
    # 原有接口（保持兼容性）
    'IndexBuilder',
    'build_markdown_index',
    'build_image_index', 
    'build_all_indexes',
    'update_documents_incremental',
    'update_single_document',
    'delete_single_document',
    'delete_collection',
    'delete_database',
    'create_collection',
    'add_data_to_collection',
    'list_collections',
    'list_databases',
    'get_collection_info',
    
    # 新增工具类
    'DatabaseOperations',
    'EmbeddingGenerator',
    'DatasetBuilder'
]
