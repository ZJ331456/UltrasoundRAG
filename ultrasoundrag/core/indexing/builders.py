"""
UltrasoundRAG 索引构建核心模块
集中处理 Markdown 和图片索引的构建逻辑
"""

import os
from typing import List, Optional
from tqdm import tqdm

from ...config import config
from ...utils.embedding_utils import embedding_provider
from ...data.loaders.markdown_parser import MarkdownParser
from ...data.loaders.image_parser import ImageParser
from ...data.stores.milvus_store import MilvusManager
from ...utils import setup_logger
from ..document_updater import DocumentUpdateManager, UpdateStats


class IndexBuilder:
    """统一的索引构建器"""
    
    def __init__(self):
        self.logger = setup_logger("IndexBuilder")
    
    def build_markdown_index(self, recreate: bool = False, only_datasets: Optional[List[str]] = None) -> dict:
        """构建Markdown索引"""
        self.logger.info("=" * 50)
        self.logger.info("开始构建 Markdown 索引")
        
        markdown_cfg = config['indexing']['markdown']
        built_count = 0
        skipped_count = 0
        global_id = 1
        
        datasets_items = list(markdown_cfg['datasets'].items())
        for dataset_name, dataset_cfg in tqdm(datasets_items, desc="Markdown 数据集", unit="ds"):
            if only_datasets and dataset_name not in only_datasets:
                continue
            
            if dataset_cfg.get('enabled', False):
                self.logger.info(f"处理 Markdown 数据集: {dataset_name}")
                
                try:
                    # 构建单个数据集
                    success = self._build_single_markdown_dataset(
                        dataset_name, dataset_cfg, recreate, global_id
                    )
                    if success:
                        built_count += 1
                        self.logger.info(f"数据集 {dataset_name} 索引构建成功")
                    else:
                        self.logger.error(f"数据集 {dataset_name} 索引构建失败")
                        
                    # 更新global_id（简化处理）
                    global_id += 10000  # 为每个数据集预留ID空间
                    
                except FileNotFoundError as e:
                    self.logger.error(f"数据集 {dataset_name} 文件不存在: {e}")
                    skipped_count += 1
                except PermissionError as e:
                    self.logger.error(f"数据集 {dataset_name} 访问权限不足: {e}")
                    skipped_count += 1
                except ConnectionError as e:
                    self.logger.error(f"数据集 {dataset_name} 数据库连接失败: {e}")
                    skipped_count += 1
                except Exception as e:
                    self.logger.error(f"构建数据集 {dataset_name} 时发生错误: {type(e).__name__}: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
        
        result = {
            'type': 'markdown',
            'built_count': built_count,
            'skipped_count': skipped_count,
            'total_datasets': len(datasets_items)
        }
        
        self.logger.info(f"Markdown 构建完成：成功 {built_count}，跳过 {skipped_count}")
        return result
    
    def build_image_index(self, recreate: bool = False, only_datasets: Optional[List[str]] = None) -> dict:
        """构建图片索引"""
        self.logger.info("=" * 50)
        self.logger.info("开始构建图片索引")
        
        image_cfg = config['indexing']['image']
        built_count = 0
        skipped_count = 0
        global_id = 1
        
        datasets_items = list(image_cfg['datasets'].items())
        for dataset_name, dataset_cfg in tqdm(datasets_items, desc="图片 数据集", unit="ds"):
            if only_datasets and dataset_name not in only_datasets:
                continue
            
            if dataset_cfg.get('enabled', False):
                self.logger.info(f"处理图片数据集: {dataset_name}")
                
                try:
                    # 构建单个数据集
                    success = self._build_single_image_dataset(
                        dataset_name, dataset_cfg, recreate, global_id
                    )
                    if success:
                        built_count += 1
                        self.logger.info(f"数据集 {dataset_name} 索引构建成功")
                    else:
                        self.logger.error(f"数据集 {dataset_name} 索引构建失败")
                        
                    # 更新global_id
                    global_id += 10000
                    
                except Exception as e:
                    self.logger.error(f"构建数据集 {dataset_name} 时发生错误: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
        
        result = {
            'type': 'image',
            'built_count': built_count,
            'skipped_count': skipped_count,
            'total_datasets': len(datasets_items)
        }
        
        self.logger.info(f"图片 构建完成：成功 {built_count}，跳过 {skipped_count}")
        return result
    
    def _build_single_markdown_dataset(self, dataset_name: str, dataset_cfg: dict, 
                                     recreate: bool, start_id: int) -> bool:
        """构建单个Markdown数据集"""
        try:
            # 确定集合名
            markdown_cfg = config['indexing']['markdown']
            target_collection = dataset_cfg.get('collection_name', markdown_cfg.get('collections', 'md_documents'))
            
            # 创建管理器
            manager = MilvusManager(collection_type="md", collection_name=target_collection)
            if recreate:
                self.logger.info(f"重建 Markdown 集合: {target_collection}")
                manager.drop_collection()
                manager._setup_collection()
            
            # 解析数据
            parser = MarkdownParser(dataset_name=dataset_name)
            parsed_data = parser.parse_markdowns()
            if not parsed_data:
                self.logger.warning(f"数据集 {dataset_name} 没有找到数据")
                return False
            
            # 重写ID
            for i, item in enumerate(parsed_data):
                item['id'] = int(start_id + i)
            
            # 生成嵌入向量
            embeddings_qwen, embeddings_clip = self._generate_text_embeddings(parsed_data)
            
            # 批量插入
            return self._batch_insert_data(manager, parsed_data, embeddings_qwen, embeddings_clip)
            
        except Exception as e:
            self.logger.error(f"构建Markdown数据集失败: {e}")
            return False
    
    def _build_single_image_dataset(self, dataset_name: str, dataset_cfg: dict, 
                                  recreate: bool, start_id: int) -> bool:
        """构建单个图片数据集"""
        try:
            # 确定集合名
            image_cfg = config['indexing']['image']
            target_collection = dataset_cfg.get('collection_name', image_cfg.get('collections', 'images'))
            
            # 创建管理器
            manager = MilvusManager(collection_type="image", collection_name=target_collection)
            if recreate:
                self.logger.info(f"重建图片集合: {target_collection}")
                manager.drop_collection()
                manager._setup_collection()
            
            # 解析数据
            parser = ImageParser(dataset_name=dataset_name)
            parsed_data = parser.parse_images()
            if not parsed_data:
                self.logger.warning(f"数据集 {dataset_name} 没有找到数据")
                return False
            
            # 重写ID
            for i, item in enumerate(parsed_data):
                item['id'] = int(start_id + i)
            
            # 批量插入
            embeddings = [item['image_vector'] for item in parsed_data]
            return self._batch_insert_image_data(manager, parsed_data, embeddings)
            
        except Exception as e:
            self.logger.error(f"构建图片数据集失败: {e}")
            return False
    
    def _generate_text_embeddings(self, parsed_data: List[dict]) -> tuple:
        """生成文本嵌入向量"""
        texts = [chunk['content'] for chunk in parsed_data]
        
        # Qwen向量
        embedder_qwen = embedding_provider[config['embedding']['provider']]
        embeddings_qwen = []
        batch_size = 64
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Qwen 文本嵌入", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_emb = embedder_qwen.embed_documents(batch_texts)
            except Exception as e:
                self.logger.error(f"Qwen 嵌入失败批次 {i // batch_size}: {e}")
                batch_emb = [[0.0] * 1024 for _ in batch_texts]
            embeddings_qwen.extend(batch_emb)
        
        # CLIP向量
        try:
            from ultrasoundrag.model.fetal_clip_model import FetalCLIPModel
            clip_model = FetalCLIPModel(
                model_path=config['indexing']['image_parse']['model_path'],
                config_path=config['indexing']['image_parse']['model_config_path']
            )
            embeddings_clip = []
            
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
            self.logger.error(f"生成CLIP文本向量失败: {e}")
            embeddings_clip = [[0.0]*768 for _ in texts]
        
        return embeddings_qwen, embeddings_clip
    
    def _batch_insert_data(self, manager: MilvusManager, parsed_data: List[dict], 
                          embeddings_qwen: List, embeddings_clip: List) -> bool:
        """批量插入文本数据"""
        insert_chunk = 1000
        insert_ok = True
        
        for i in tqdm(range(0, len(parsed_data), insert_chunk), desc="写入 Milvus", unit="chunk"):
            part_data = parsed_data[i:i + insert_chunk]
            part_qwen = embeddings_qwen[i:i + insert_chunk]
            part_clip = embeddings_clip[i:i + insert_chunk]
            ok = manager.insert_data(part_data, embeddings_qwen=part_qwen, embeddings_clip=part_clip)
            insert_ok = insert_ok and ok
        
        return insert_ok
    
    def _batch_insert_image_data(self, manager: MilvusManager, parsed_data: List[dict], 
                               embeddings: List) -> bool:
        """批量插入图片数据"""
        insert_chunk = 1000
        insert_ok = True
        
        for i in tqdm(range(0, len(parsed_data), insert_chunk), desc="写入 Milvus(图片)", unit="chunk"):
            part_data = parsed_data[i:i + insert_chunk]
            part_emb = embeddings[i:i + insert_chunk]
            ok = manager.insert_data(part_data, part_emb)
            insert_ok = insert_ok and ok
        
        return insert_ok
    
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
    """更新单个文档的便捷函数"""
    from ultrasoundrag.core.document_updater import update_document_by_name
    return update_document_by_name(document_name, dataset_name)
