"""
UltrasoundRAG 索引构建核心模块
重构版本：使用模块化的工具类，职责更清晰
"""

import os
from typing import List, Optional, Dict

from ...config import config
from ...utils import setup_logger
from ...data.processors.database_operations import DatabaseOperations
from ...data.processors.dataset_builder import DatasetBuilder
from ...data.processors.document_updater import DocumentUpdateManager, UpdateStats


class IndexBuilder:
    """统一的索引构建器 - 重构版本"""
    
    def __init__(self):
        self.logger = setup_logger("IndexBuilder")
        self.dataset_builder = DatasetBuilder()
    
    def build_markdown_index(self, recreate: bool = False, only_datasets: Optional[List[str]] = None) -> dict:
        """构建Markdown索引 - 重构版本"""
        return self.dataset_builder.build_all_datasets_of_type(
            dataset_type="markdown", 
            recreate=recreate, 
            only_datasets=only_datasets
        )
    
    def build_image_index(self, recreate: bool = False, only_datasets: Optional[List[str]] = None) -> dict:
        """构建图片索引 - 重构版本"""
        return self.dataset_builder.build_all_datasets_of_type(
            dataset_type="image", 
            recreate=recreate, 
            only_datasets=only_datasets
        )
    
    
    def update_documents_incremental(self, target_datasets: Optional[List[str]] = None, 
                                   max_documents_per_dataset: Optional[int] = None) -> dict:
        """增量更新文档索引"""
        self.logger.info("=" * 60)
        self.logger.info("开始增量更新文档索引")
        
        results = {}
        total_stats = UpdateStats()
        
        # 更新 Markdown 数据集
        markdown_cfg = config['indexing']['markdown']
        for dataset_name, dataset_cfg in markdown_cfg['datasets'].items():
            if target_datasets and dataset_name not in target_datasets:
                continue
                
            if dataset_cfg.get('enabled', False):
                self.logger.info(f"更新 Markdown 数据集: {dataset_name}")
                
                try:
                    # 设置数据集类型
                    dataset_cfg['type'] = 'markdown'
                    updater = DocumentUpdateManager(dataset_name, dataset_cfg)
                    
                    # 执行增量更新
                    stats = updater.batch_update_documents(max_documents_per_dataset)
                    results[f"markdown_{dataset_name}"] = stats
                    
                    # 累计统计
                    total_stats.total_documents += stats.total_documents
                    total_stats.added_documents += stats.added_documents
                    total_stats.modified_documents += stats.modified_documents
                    total_stats.deleted_documents += stats.deleted_documents
                    total_stats.failed_documents += stats.failed_documents
                    total_stats.total_chunks_added += stats.total_chunks_added
                    total_stats.total_chunks_deleted += stats.total_chunks_deleted
                    total_stats.processing_time += stats.processing_time
                    total_stats.errors.extend(stats.errors)
                    
                except Exception as e:
                    self.logger.error(f"更新数据集 {dataset_name} 失败: {e}")
                    total_stats.failed_documents += 1
                    total_stats.errors.append(f"{dataset_name}: {e}")
        
        # 更新图片数据集
        image_cfg = config['indexing']['image']
        for dataset_name, dataset_cfg in image_cfg['datasets'].items():
            if target_datasets and dataset_name not in target_datasets:
                continue
                
            if dataset_cfg.get('enabled', False):
                self.logger.info(f"更新图片数据集: {dataset_name}")
                
                try:
                    # 设置数据集类型
                    dataset_cfg['type'] = 'image'
                    updater = DocumentUpdateManager(dataset_name, dataset_cfg)
                    
                    # 执行增量更新
                    stats = updater.batch_update_documents(max_documents_per_dataset)
                    results[f"image_{dataset_name}"] = stats
                    
                    # 累计统计
                    total_stats.total_documents += stats.total_documents
                    total_stats.added_documents += stats.added_documents
                    total_stats.modified_documents += stats.modified_documents
                    total_stats.deleted_documents += stats.deleted_documents
                    total_stats.failed_documents += stats.failed_documents
                    total_stats.total_chunks_added += stats.total_chunks_added
                    total_stats.total_chunks_deleted += stats.total_chunks_deleted
                    total_stats.processing_time += stats.processing_time
                    total_stats.errors.extend(stats.errors)
                    
                except Exception as e:
                    self.logger.error(f"更新数据集 {dataset_name} 失败: {e}")
                    total_stats.failed_documents += 1
                    total_stats.errors.append(f"{dataset_name}: {e}")
        
        # 汇总结果
        results['total_stats'] = total_stats
        
        self.logger.info("=" * 60)
        self.logger.info("增量更新完成统计:")
        self.logger.info(f"总文档数: {total_stats.total_documents}")
        self.logger.info(f"新增: {total_stats.added_documents}, 修改: {total_stats.modified_documents}, 删除: {total_stats.deleted_documents}")
        self.logger.info(f"失败: {total_stats.failed_documents}")
        self.logger.info(f"新增片段: {total_stats.total_chunks_added}, 删除片段: {total_stats.total_chunks_deleted}")
        self.logger.info(f"总处理时间: {total_stats.processing_time:.2f}秒")
        if total_stats.errors:
            self.logger.warning(f"遇到 {len(total_stats.errors)} 个错误")
        self.logger.info("=" * 60)
        
        return results


# 便捷函数
def build_markdown_index(recreate: bool = False, only_datasets: Optional[List[str]] = None) -> dict:
    """构建Markdown索引的便捷函数"""
    builder = IndexBuilder()
    return builder.build_markdown_index(recreate, only_datasets)


def build_image_index(recreate: bool = False, only_datasets: Optional[List[str]] = None) -> dict:
    """构建图片索引的便捷函数"""
    builder = IndexBuilder()
    return builder.build_image_index(recreate, only_datasets)


def build_all_indexes(recreate: bool = False, md_datasets: Optional[List[str]] = None,
                     image_datasets: Optional[List[str]] = None) -> dict:
    """构建所有索引的便捷函数"""
    builder = IndexBuilder()
    
    results = {
        'markdown': builder.build_markdown_index(recreate, md_datasets),
        'image': builder.build_image_index(recreate, image_datasets)
    }
    
    return results


def update_documents_incremental(target_datasets: Optional[List[str]] = None, 
                               max_documents_per_dataset: Optional[int] = None) -> dict:
    """增量更新文档的便捷函数"""
    builder = IndexBuilder()
    return builder.update_documents_incremental(target_datasets, max_documents_per_dataset)


def update_single_document(document_name: str, dataset_name: Optional[str] = None) -> bool:
    """更新单个文档的便捷函数 - 重构版本"""
    from ...utils import setup_logger
    from ...data.processors.database_operations import DatabaseOperations
    from ...data.processors.embedding_generator import EmbeddingGenerator
    
    logger = setup_logger("SingleDocumentUpdater")
    
    try:
        from ...config import config
        from ...data.stores.milvus_store import MilvusManager
        from ...data.loaders.markdown_parser import MarkdownParser
        
        # 如果没有指定数据集名称，使用默认值
        if dataset_name is None:
            dataset_name = "ultrasound_book"
        
        # 获取数据集配置
        datasets = config.get('indexing', {}).get('markdown', {}).get('datasets', {})
        if dataset_name not in datasets:
            logger.error(f"未找到数据集配置: {dataset_name}")
            return False
        
        dataset_config = datasets[dataset_name]
        collection_name = dataset_config.get('collection_name', 'normal_book_md')
        base_path = dataset_config.get('base_path', 'data/book/markdown')
        
        # 构建文件完整路径
        file_path = os.path.join(base_path, document_name)
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        
        logger.info(f"开始更新文档: {document_name}")
        
        # 1. 连接到Milvus并删除旧内容
        manager = MilvusManager(collection_type="md", collection_name=collection_name)
        success, deleted_count = manager.delete_document_chunks(document_name)
        
        if success:
            logger.info(f"成功删除 {deleted_count} 个旧chunk")
        else:
            logger.error("删除旧内容失败")
            return False
        
        # 2. 重新解析文档
        parser = MarkdownParser(dataset_name=dataset_name)
        parsed_data = parser.parse_single_file(file_path)
        
        if not parsed_data:
            logger.error("文档解析失败")
            return False
        
        # 3. 生成嵌入向量
        embedding_generator = EmbeddingGenerator()
        embeddings_qwen, embeddings_clip = embedding_generator.generate_text_embeddings(parsed_data)
        
        # 4. 插入新数据
        success = DatabaseOperations.batch_insert_with_progress(
            manager, parsed_data, embeddings_qwen, embeddings_clip
        )
        
        if success:
            logger.info(f"成功添加 {len(parsed_data)} 个新chunk")
            return True
        else:
            logger.error("添加新内容失败")
            return False
            
    except Exception as e:
        logger.error(f"更新文档时发生错误: {e}")
        return False


def delete_single_document(document_name: str, dataset_name: Optional[str] = None) -> bool:
    """删除单个文档的便捷函数"""
    from ...utils import setup_logger
    logger = setup_logger("SingleDocumentDeleter")
    
    try:
        from ...config import config
        from ...data.stores.milvus_store import MilvusManager
        
        # 如果没有指定数据集名称，使用默认值
        if dataset_name is None:
            dataset_name = "ultrasound_book"
        
        # 获取数据集配置
        datasets = config.get('indexing', {}).get('markdown', {}).get('datasets', {})
        if dataset_name not in datasets:
            logger.error(f"未找到数据集配置: {dataset_name}")
            return False
        
        dataset_config = datasets[dataset_name]
        collection_name = dataset_config.get('collection_name', 'normal_book_md')
        
        logger.info(f"开始删除文档: {document_name}")
        
        # 连接到Milvus并删除文档内容
        manager = MilvusManager(collection_type="md", collection_name=collection_name)
        success, deleted_count = manager.delete_document_chunks(document_name)
        
        if success:
            logger.info(f"成功删除 {deleted_count} 个chunk")
            return True
        else:
            logger.error("删除文档失败")
            return False
            
    except Exception as e:
        logger.error(f"删除文档时发生错误: {e}")
        return False


def delete_collection(collection_name: str, collection_type: str = None) -> bool:
    """删除整个集合的便捷函数
    
    Args:
        collection_name: 集合名称
        collection_type: 集合类型（可选，仅用于日志记录）
    """
    from ...utils import setup_logger
    logger = setup_logger("CollectionDeleter")
    
    try:
        from ...data.stores.milvus_store import MilvusManager
        
        logger.info(f"开始删除集合: {collection_name}")
        
        # 推断集合类型（如果未提供）
        if not collection_type:
            if 'image' in collection_name.lower():
                collection_type = 'image'
            elif 'pdf' in collection_name.lower():
                collection_type = 'pdf'
            else:
                collection_type = 'md'
        
        # 连接到Milvus并删除集合
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name)
        success = manager.drop_collection()
        
        if success:
            logger.info(f"成功删除集合: {collection_name}")
        else:
            logger.error(f"删除集合失败: {collection_name}")
        
        return success
            
    except Exception as e:
        logger.error(f"删除集合时发生错误: {e}")
        return False


def delete_database(db_name: str) -> bool:
    """删除整个数据库的便捷函数（极度危险操作）
    
    Args:
        db_name: 数据库名称
    """
    from ...utils import setup_logger
    logger = setup_logger("DatabaseDeleter")
    
    try:
        from ...data.stores.milvus_store import MilvusManager
        from ...config import config
        
        logger.warning(f"开始删除整个数据库: {db_name}")
        logger.warning(f"这将删除数据库中的所有集合和数据！")
        
        # 获取Milvus配置
        milvus_cfg = config.get('milvus', {})
        
        # 创建临时管理器来获取客户端
        temp_manager = MilvusManager(collection_type="md", collection_name="temp", db_name=db_name)
        client = temp_manager.client
        
        # 列出数据库中的所有集合
        collections = client.list_collections()
        logger.info(f"数据库 '{db_name}' 中的集合: {collections}")
        
        # 二次确认机制
        print(f"\n⚠️  警告：您即将删除数据库 '{db_name}'")
        print(f"   该数据库包含以下集合: {collections}")
        print(f"   此操作将永久删除所有数据，无法恢复！")
        print(f"\n   请输入 'DELETE {db_name}' 来确认删除操作：")
        
        confirmation = input("确认输入: ").strip()
        
        if confirmation != f"DELETE {db_name}":
            logger.info(f"用户取消了数据库删除操作")
            print("删除操作已取消")
            return False
        
        print("确认通过，开始删除数据库...")
        logger.warning(f"用户确认删除数据库: {db_name}")
        
        # 删除所有集合
        for collection_name in collections:
            logger.info(f"删除集合: {collection_name}")
            client.drop_collection(collection_name)
        
        # 删除数据库
        client.drop_database(db_name)
        
        logger.warning(f"成功删除数据库: {db_name}")
        print(f"✅ 数据库 '{db_name}' 已成功删除")
        return True
            
    except Exception as e:
        logger.error(f"删除数据库时发生错误: {e}")
        print(f"❌ 删除数据库时发生错误: {e}")
        return False


def create_collection(collection_name: str, collection_type: str = "md", db_name: str = None) -> bool:
    """创建新集合的便捷函数"""
    from ...utils import setup_logger
    logger = setup_logger("CollectionCreator")
    
    try:
        from ...data.stores.milvus_store import MilvusManager
        
        logger.info(f"开始创建集合: {collection_name}")
        
        # 创建集合
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name, db_name=db_name)
        
        logger.info(f"成功创建集合: {collection_name}")
        return True
            
    except Exception as e:
        logger.error(f"创建集合时发生错误: {e}")
        return False


def add_data_to_collection(data_path: str, collection_name: str, collection_type: str = "md", db_name: str = None, original_filename: str = None) -> bool:
    """将数据添加到指定集合的便捷函数 - 重构版本"""
    from ...utils import setup_logger
    from ...data.processors.database_operations import DatabaseOperations
    from ...data.processors.embedding_generator import EmbeddingGenerator
    
    logger = setup_logger("DataAdder")
    
    try:
        from ...data.stores.milvus_store import MilvusManager
        from ...data.loaders.markdown_parser import MarkdownParser
        from ...data.loaders.pdf_parser import PDFParser
        from ...data.loaders.image_parser import ImageParser
        
        logger.info(f"开始将数据添加到集合: {collection_name}")
        
        # 创建管理器
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name, db_name=db_name)
        
        # 根据集合类型选择解析器
        if collection_type == "md":
            parser = MarkdownParser()
            parsed_data = parser.parse_single_file(data_path, original_filename=original_filename)
        elif collection_type == "pdf":
            parser = PDFParser()
            parsed_data = parser.parse_single_file(data_path, original_filename=original_filename)
        elif collection_type == "image":
            parser = ImageParser()
            parsed_data = parser.parse_single_file(data_path, original_filename=original_filename)
        else:
            logger.error(f"不支持的集合类型: {collection_type}")
            return False
        
        if not parsed_data:
            logger.error("数据解析失败")
            return False
        
        # 生成嵌入向量并插入数据
        embedding_generator = EmbeddingGenerator()
        if collection_type in ["md", "pdf"]:
            embeddings_qwen, embeddings_clip = embedding_generator.generate_text_embeddings(parsed_data)
            success = DatabaseOperations.batch_insert_with_progress(
                manager, parsed_data, embeddings_qwen, embeddings_clip
            )
        else:  # image
            embeddings = embedding_generator.generate_image_embeddings(parsed_data)
            success = DatabaseOperations.batch_insert_image_data(manager, parsed_data, embeddings)
        
        if success:
            logger.info(f"成功添加 {len(parsed_data)} 个数据到集合: {collection_name}")
            return True
        else:
            logger.error("添加数据失败")
            return False
            
    except Exception as e:
        logger.error(f"添加数据时发生错误: {e}")
        return False


def list_collections(db_name: str = None) -> List[str]:
    """列出所有集合的便捷函数"""
    from ...utils import setup_logger
    logger = setup_logger("CollectionLister")
    
    try:
        from ...data.stores.milvus_store import MilvusClient
        from ...config import config
        
        # 直接创建客户端，不创建集合
        milvus_cfg = config.get('milvus', {})
        client = MilvusClient(uri=milvus_cfg['milvus_uri'], token=milvus_cfg['milvus_token'])
        
        # 切换到指定数据库
        if db_name:
            client.use_database(db_name)
        else:
            client.use_database(milvus_cfg['db_name'])
        
        collections = client.list_collections()
        
        logger.info(f"数据库中的集合: {collections}")
        return collections
            
    except Exception as e:
        logger.error(f"列出集合时发生错误: {e}")
        return []


def list_databases() -> List[str]:
    """列出所有数据库的便捷函数"""
    from ...utils import setup_logger
    logger = setup_logger("DatabaseLister")
    
    try:
        from ...data.stores.milvus_store import MilvusClient
        from ...config import config
        
        # 直接创建客户端，不创建集合
        milvus_cfg = config.get('milvus', {})
        client = MilvusClient(uri=milvus_cfg['milvus_uri'], token=milvus_cfg['milvus_token'])
        
        databases = client.list_databases()
        
        logger.info(f"所有数据库: {databases}")
        return databases
            
    except Exception as e:
        logger.error(f"列出数据库时发生错误: {e}")
        return []


def get_collection_info(collection_name: str, db_name: str = None) -> Dict:
    """获取集合信息的便捷函数"""
    from ...utils import setup_logger
    logger = setup_logger("CollectionInfoGetter")
    
    try:
        from ...data.stores.milvus_store import MilvusManager
        
        # 根据集合名称推断类型
        def _infer_collection_type_local(name: str) -> str:
            """根据集合名称推断类型"""
            name_lower = name.lower()
            if 'image' in name_lower or 'img' in name_lower:
                return 'image'
            elif 'pdf' in name_lower:
                return 'pdf'
            else:
                return 'md'
        
        collection_type = _infer_collection_type_local(collection_name)
        
        # 创建管理器
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name, db_name=db_name)
        # 使用修复后的快速方法
        info = manager.get_collection_info_fast()
        
        return info
            
    except Exception as e:
        logger.error(f"获取集合信息时发生错误: {e}")
        return {}
