"""
模块化检索器 - 分离的四种检索方式
提供独立的T2T、T2I、I2T、I2I检索功能，可以单独使用

主要特点：
1. 每种检索方式完全独立
2. 基于配置的灵活参数管理
3. 支持多数据库、多集合
4. 清晰的结果结构
"""

import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from UltrasoundRAG.utils.logger import setup_logger
from UltrasoundRAG.retrival.data_structures import RetrievalResult
from UltrasoundRAG.milvus.milvus_manager import MilvusManager
from UltrasoundRAG.utils.embedding_utils import embedding_provider
from UltrasoundRAG.model.model_manager import get_fetal_clip_model, get_embedding_model
from UltrasoundRAG.config import config
from UltrasoundRAG.retrival.caption_to_image_retriever import create_caption_retriever
from PIL import Image


@dataclass
class RetrievalContext:
    """检索上下文配置"""
    db_name: str
    text_collection: Optional[str]
    image_collection: Optional[str]
    top_k: int = 10
    milvus_uri: Optional[str] = None
    milvus_token: Optional[str] = None
    # 新增：领域过滤与分区路由配置
    domain: Optional[str] = None
    use_partition: bool = False


class BaseRetriever(ABC):
    """基础检索器抽象类"""
    
    def __init__(self, context: RetrievalContext):
        self.context = context
        self.logger = setup_logger(self.__class__.__name__)
        
        # 从全局配置获取Milvus连接信息
        milvus_config = config['milvus']
        self.milvus_uri = context.milvus_uri or milvus_config['milvus_uri']
        self.milvus_token = context.milvus_token or milvus_config['milvus_token']
    
    @abstractmethod
    def search(self, query: Any, **kwargs) -> Dict[str, Any]:
        """抽象检索方法"""
        pass


class T2TRetriever(BaseRetriever):
    """文本到文本检索器 - 独立模块"""
    
    def __init__(self, context: RetrievalContext):
        super().__init__(context)
        
        # 初始化文本集合管理器
        if not context.text_collection:
            raise ValueError("当前数据库未配置文本集合（collections.text 为空），无法进行 T2T 检索")
        self.text_manager = MilvusManager(
            milvus_uri=self.milvus_uri,
            milvus_token=self.milvus_token,
            db_name=context.db_name,
            collection_type="md",
            collection_name=context.text_collection,
            enable_domain_partition=context.use_partition
        )
        
        # 初始化文本嵌入模型（使用共享实例）
        self.text_embedder = get_embedding_model()
        
        # 获取T2T配置
        self.t2t_config = config['retriever']['retrieval_modes']['t2t']
        
        # 获取CLIP模型（使用共享实例，避免重复加载）
        self.clip_model = get_fetal_clip_model()

        self.logger.info("T2T检索器初始化完成")
    
    def search(self, query: str, top_k: Optional[int] = None, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        执行文本到文本检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            strategy: 检索策略 ("basic" 或 "fusion")
            
        Returns:
            T2T检索结果
        """
        top_k = top_k or self.context.top_k
        strategy = strategy or self.t2t_config['strategy']
        
        if strategy == "fusion":
            return self._fusion_search(query, top_k)
        else:
            return self._basic_search(query, top_k)
    
    def _basic_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """基础文本检索"""
        try:
            # 生成查询向量
            query_vector = self.text_embedder.get_query_embedding(query)
            if not query_vector:
                return self._empty_result(query, "查询向量生成失败")
            
            # 构造领域过滤与分区
            filter_expr = ""
            partitions = None
            if self.context.domain:
                filter_expr = f'{self.text_manager.domain_field_name} == "{self.context.domain}"'
                if self.context.use_partition:
                    partitions = [self.context.domain.replace(' ', '_')[:64]]
            # 执行检索
            milvus_results = self.text_manager.search(query_vector, top_k, filter_expr=filter_expr, partition_names=partitions)
            
            # 转换结果
            results = []
            for result in milvus_results:
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=result.get('content', ''),
                    metadata={
                        'title': result.get('title', ''),
                        'md_file': result.get('md_file', ''),
                        'document_name': result.get('document_name', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'image_links': result.get('image_links', []),
                        'image_captions': result.get('image_captions', []),
                        'search_strategy': 'basic_t2t'
                    },
                    score=result.get('score', 0.0),
                    retrieval_type='text_to_text',
                    resource_collection=self.text_manager.collection_name
                )
                results.append(retrieval_result)
            
            return {
                'query': query,
                'retrieval_type': 't2t',
                'strategy': 'basic',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"T2T基础检索失败: {e}")
            return self._empty_result(query, str(e))
    
    def _fusion_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """融合文本检索（结合图片caption检索）"""
        try:
            # 多向量混合检索（Qwen + CLIP），强制两路同时可用
            # 1) 生成两路查询向量
            qwen_vec = self.text_embedder.get_query_embedding(query)
            tokens = self.clip_model.tokenize_text([query])
            clip_vec = self.clip_model.encode_text(tokens).cpu().numpy().tolist()[0]

            # 2) 严格校验两路维度
            if not (isinstance(qwen_vec, list) and len(qwen_vec) == 1024 and isinstance(clip_vec, list) and len(clip_vec) == 768):
                raise ValueError("T2T融合检索需要两路向量(qwen-1024 与 clip-768)同时可用且维度匹配")

            # 3) 构造过滤/分区
            filter_expr = ""
            partitions = None
            if self.context.domain:
                filter_expr = f'{self.text_manager.domain_field_name} == "{self.context.domain}"'
                if self.context.use_partition:
                    partitions = [self.context.domain.replace(' ', '_')[:64]]

            # 4) 并行检索并融合
            field_to_embedding = {
                "text_vector_qwen_1024": qwen_vec,
                "text_vector_clip_768": clip_vec,
            }
            weights = {
                "text_vector_qwen_1024": 0.6,
                "text_vector_clip_768": 0.4,
            }
            raw = self.text_manager.search_multi_vectors(
                field_to_embedding,
                top_k=top_k,
                weights=weights,
                filter_expr=filter_expr,
                partition_names=partitions,
            )

            # 5) 统一结果结构
            text_results = []
            for result in raw:
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=result.get('content', ''),
                    metadata={
                        'title': result.get('title', ''),
                        'md_file': result.get('md_file', ''),
                        'document_name': result.get('document_name', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'image_links': result.get('image_links', []),
                        'image_captions': result.get('image_captions', []),
                        'search_strategy': 'fusion_t2t_multivector',
                        'domain': result.get(self.text_manager.domain_field_name, '')
                    },
                    score=result.get('score', 0.0),
                    retrieval_type='text_to_text',
                    resource_collection=self.text_manager.collection_name
                )
                text_results.append(retrieval_result)
            
            # 如果启用了融合策略，还需要进行图片caption检索
            fusion_config = self.t2t_config['fusion_config']
            if fusion_config['image_caption_ratio'] > 0:
                caption_results = self._search_by_image_captions(query, top_k)
                
                # 应用权重并合并结果
                text_ratio = fusion_config['text_search_ratio']
                caption_ratio = fusion_config['image_caption_ratio']
                
                for result in text_results:
                    result.score *= text_ratio
                    result.metadata['fusion_component'] = 'text_search'
                
                for result in caption_results:
                    result.score *= caption_ratio
                    result.metadata['fusion_component'] = 'caption_search'
                
                # 合并并排序
                all_results = text_results + caption_results
                all_results.sort(key=lambda x: x.score, reverse=True)
                
                # 去重（如果启用）
                if fusion_config['enable_deduplication']:
                    all_results = self._deduplicate_results(all_results, 
                                                          fusion_config['content_similarity_threshold'])
                
                return {
                    'query': query,
                    'retrieval_type': 't2t',
                    'strategy': 'fusion',
                    'results': all_results[:top_k],
                    'total_results': len(all_results[:top_k]),
                    'fusion_breakdown': {
                        'text_results': len(text_results),
                        'caption_results': len(caption_results)
                    }
                }
            else:
                # 仅返回文本检索结果
                return {
                    'query': query,
                    'retrieval_type': 't2t',
                    'strategy': 'fusion_text_only',
                    'results': text_results,
                    'total_results': len(text_results)
                }
                
        except Exception as e:
            self.logger.error(f"T2T融合检索失败: {e}")
            return self._empty_result(query, str(e))
    
    def _search_by_image_captions(self, query: str, top_k: int) -> List[RetrievalResult]:
        """通过图片caption检索文本结果"""
        try:
            # 这里需要使用图片检索器来搜索相关图片，然后返回其caption作为文本结果
            from UltrasoundRAG.retrival.modular_retrievers import T2IRetriever
            
            # 创建临时T2I检索器
            t2i_retriever = T2IRetriever(self.context)
            t2i_result = t2i_retriever.search(query, top_k)
            
            # 将图片结果转换为文本结果（使用caption）
            caption_results = []
            for img_result in t2i_result.get('results', []):
                caption = img_result.metadata.get('caption', '')
                if caption:
                    text_result = RetrievalResult(
                        doc_id=f"caption_{img_result.doc_id}",
                        content=caption,
                        metadata={
                            **img_result.metadata,
                            'source_type': 'image_caption',
                            'original_image_path': img_result.metadata.get('relative_path', ''),
                            'search_strategy': 'caption_to_text'
                        },
                        score=img_result.score,
                        retrieval_type='caption_to_text',
                        resource_collection=img_result.resource_collection
                    )
                    caption_results.append(text_result)
            
            return caption_results
            
        except Exception as e:
            self.logger.error(f"Caption检索失败: {e}")
            return []
    
    def _deduplicate_results(self, results: List[RetrievalResult], threshold: float) -> List[RetrievalResult]:
        """结果去重"""
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # 使用内容的前200个字符进行去重
            content_key = result.content[:200].strip()
            content_hash = hash(content_key)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _empty_result(self, query: str, error_msg: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'query': query,
            'retrieval_type': 't2t',
            'results': [],
            'total_results': 0,
            'error': error_msg
        }


class T2IRetriever(BaseRetriever):
    """文本到图片检索器 - 独立模块"""
    
    def __init__(self, context: RetrievalContext):
        super().__init__(context)
        
        # 初始化图片集合管理器
        if not context.image_collection:
            raise ValueError("当前数据库未配置图片集合（collections.image 为空），无法进行 T2I 检索")
        self.image_manager = MilvusManager(
            milvus_uri=self.milvus_uri,
            milvus_token=self.milvus_token,
            db_name=context.db_name,
            collection_type="image",
            collection_name=context.image_collection,
            enable_domain_partition=context.use_partition
        )
        
        # 获取CLIP模型（使用共享实例，避免重复加载）
        self.clip_model = get_fetal_clip_model()
        
        # 获取T2I配置
        self.t2i_config = config['retriever']['retrieval_modes']['t2i']
        
        # 初始化图片标题匹配器
        self.title_matcher = create_caption_retriever(
            db_name=context.db_name,
            search_mode="hybrid_match",
            match_mode="variant"
        )
        
        self.logger.info("T2I检索器初始化完成")
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        执行文本到图片检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            T2I检索结果
        """
        top_k = top_k or self.context.top_k
        
        try:
            # 使用CLIP文本编码器生成查询向量
            if self.t2i_config['use_clip_text_encoder']:
                tokens = self.clip_model.tokenize_text([query])
                query_vector = self.clip_model.encode_text(tokens).cpu().numpy().tolist()[0]
            else:
                # 回退到普通文本嵌入
                embedding_provider_name = config['embedding']['provider']
                text_embedder = embedding_provider[embedding_provider_name]
                query_vector = text_embedder.get_query_embedding(query)
            
            if not query_vector:
                return self._empty_result(query, "查询向量生成失败")
            
            # 构造领域过滤与分区
            filter_expr = ""
            partitions = None
            if self.context.domain:
                filter_expr = f'{self.image_manager.domain_field_name} == "{self.context.domain}"'
                if self.context.use_partition:
                    partitions = [self.context.domain.replace(' ', '_')[:64]]
            # 执行检索
            milvus_results = self.image_manager.search(query_vector, top_k, filter_expr=filter_expr, partition_names=partitions)
            
            # 转换结果
            results = []
            for result in milvus_results:
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=result.get('caption', ''),  # 图片的caption作为content
                    metadata={
                        'relative_path': result.get('relative_path', ''),
                        'caption': result.get('caption', ''),
                        'folder_name': result.get('folder_name', ''),
                        'file_size': result.get('file_size', 0),
                        'search_strategy': 'text_to_image'
                    },
                    score=result.get('score', 0.0),
                    retrieval_type='text_to_image',
                    resource_collection=self.image_manager.collection_name
                )
                results.append(retrieval_result)
            
            return {
                'query': query,
                'retrieval_type': 't2i',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"T2I检索失败: {e}")
            return self._empty_result(query, str(e))
    
    def search_by_exact_caption(self, caption: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        根据精确caption匹配图像
        
        Args:
            caption: 图像caption
            top_k: 返回结果数量
            
        Returns:
            T2I检索结果
        """
        top_k = top_k or self.context.top_k
        
        try:
            # 使用统一的图片标题匹配器
            match_result = self.title_matcher.match_single_title(caption, top_k)
            
            self.logger.info(f"精确caption匹配完成，返回 {len(match_result)} 个结果")
            return {
                'query': caption,
                'retrieval_type': 'exact_caption_match',
                'results': match_result,
                'total_results': len(match_result)
            }
            
        except Exception as e:
            self.logger.error(f"精确caption匹配失败: {e}")
            return self._empty_result(caption, str(e))
    
    def _empty_result(self, query: str, error_msg: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'query': query,
            'retrieval_type': 't2i',
            'results': [],
            'total_results': 0,
            'error': error_msg
        }


class I2TRetriever(BaseRetriever):
    """图片到文本检索器 - 独立模块"""
    
    def __init__(self, context: RetrievalContext):
        super().__init__(context)
        
        # 初始化文本集合管理器（用于检索文本）
        self.text_manager = MilvusManager(
            milvus_uri=self.milvus_uri,
            milvus_token=self.milvus_token,
            db_name=context.db_name,
            collection_type="md"
        )
        
        # 获取CLIP模型（使用共享实例，避免重复加载）
        self.clip_model = get_fetal_clip_model()
        
        # 获取I2T配置
        self.i2t_config = config['retriever']['retrieval_modes']['i2t']
        
        self.logger.info("I2T检索器初始化完成")
    
    def search(self, image_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        执行图片到文本检索
        
        Args:
            image_path: 图片路径
            top_k: 返回结果数量
            
        Returns:
            I2T检索结果
        """
        top_k = top_k or self.context.top_k
        
        try:
            # 验证图片路径
            if not os.path.exists(image_path):
                return self._empty_result(image_path, f"图片文件不存在: {image_path}")
            
            # 使用CLIP图片编码器生成查询向量
            image = Image.open(image_path).convert('RGB')
            # 将PIL Image转换为tensor
            image_tensor = self.clip_model.preprocess_image(image).unsqueeze(0)  # 添加batch维度
            query_vector = self.clip_model.encode_image(image_tensor).cpu().numpy().tolist()[0]
            
            if not query_vector:
                return self._empty_result(image_path, "图片向量生成失败")
            
            # 在文本集合中搜索（使用CLIP向量）
            filter_expr = ""
            partitions = None
            if self.context.domain:
                filter_expr = f'{self.text_manager.domain_field_name} == "{self.context.domain}"'
                if self.context.use_partition:
                    partitions = [self.context.domain.replace(' ', '_')[:64]]
            milvus_results = self.text_manager.search(query_vector, top_k, filter_expr=filter_expr, partition_names=partitions)
            
            # 转换结果
            results = []
            for result in milvus_results:
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=result.get('content', ''),
                    metadata={
                        'title': result.get('title', ''),
                        'md_file': result.get('md_file', ''),
                        'document_name': result.get('document_name', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'image_links': result.get('image_links', []),
                        'image_captions': result.get('image_captions', []),
                        'query_image_path': image_path,
                        'search_strategy': 'image_to_text'
                    },
                    score=result.get('score', 0.0),
                    retrieval_type='image_to_text',
                    resource_collection=self.text_manager.collection_name
                )
                results.append(retrieval_result)
            
            return {
                'query_image': image_path,
                'retrieval_type': 'i2t',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"I2T检索失败: {e}")
            return self._empty_result(image_path, str(e))
    
    def _empty_result(self, image_path: str, error_msg: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'query_image': image_path,
            'retrieval_type': 'i2t',
            'results': [],
            'total_results': 0,
            'error': error_msg
        }


class I2IRetriever(BaseRetriever):
    """图片到图片检索器 - 独立模块"""
    
    def __init__(self, context: RetrievalContext):
        super().__init__(context)
        
        # 初始化图片集合管理器
        self.image_manager = MilvusManager(
            milvus_uri=self.milvus_uri,
            milvus_token=self.milvus_token,
            db_name=context.db_name,
            collection_type="image"
        )
        
        # 获取CLIP模型（使用共享实例，避免重复加载）
        self.clip_model = get_fetal_clip_model()
        
        # 获取I2I配置
        self.i2i_config = config['retriever']['retrieval_modes']['i2i']
        
        self.logger.info("I2I检索器初始化完成")
    
    def search(self, image_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        执行图片到图片检索
        
        Args:
            image_path: 查询图片路径
            top_k: 返回结果数量
            
        Returns:
            I2I检索结果
        """
        top_k = top_k or self.context.top_k
        
        try:
            # 验证图片路径
            if not os.path.exists(image_path):
                return self._empty_result(image_path, f"图片文件不存在: {image_path}")
            
            # 使用CLIP图片编码器生成查询向量
            image = Image.open(image_path).convert('RGB')
            # 将PIL Image转换为tensor
            image_tensor = self.clip_model.preprocess_image(image).unsqueeze(0)  # 添加batch维度
            query_vector = self.clip_model.encode_image(image_tensor).cpu().numpy().tolist()[0]
            
            if not query_vector:
                return self._empty_result(image_path, "图片向量生成失败")
            
            # 执行检索
            milvus_results = self.image_manager.search(query_vector, top_k)
            
            # 转换结果
            results = []
            for result in milvus_results:
                metadata = {
                    'relative_path': result.get('relative_path', ''),
                    'caption': result.get('caption', ''),
                    'folder_name': result.get('folder_name', ''),
                    'file_size': result.get('file_size', 0),
                    'query_image_path': image_path,
                    'search_strategy': 'image_to_image'
                }
                
                # 如果配置要求包含更多元数据
                if self.i2i_config['include_metadata']:
                    metadata.update({
                        'image_width': result.get('image_width'),
                        'image_height': result.get('image_height'),
                        'image_format': result.get('image_format'),
                        'created_time': result.get('created_time')
                    })
                
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=result.get('caption', ''),  # 使用caption作为content
                    metadata=metadata,
                    score=result.get('score', 0.0),
                    retrieval_type='image_to_image',
                    resource_collection=self.image_manager.collection_name
                )
                results.append(retrieval_result)
            
            return {
                'query_image': image_path,
                'retrieval_type': 'i2i',
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"I2I检索失败: {e}")
            return self._empty_result(image_path, str(e))
    
    def _empty_result(self, image_path: str, error_msg: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'query_image': image_path,
            'retrieval_type': 'i2i',
            'results': [],
            'total_results': 0,
            'error': error_msg
        }


# 便捷函数
def create_retrieval_context(db_name: str = "default", top_k: int = 10) -> RetrievalContext:
    """
    创建检索上下文
    
    Args:
        db_name: 数据库配置名称（对应config.yaml中的databases配置）
        top_k: 默认返回结果数量
        
    Returns:
        检索上下文对象
    """
    retriever_config = config['retriever']
    databases_config = retriever_config['databases']
    
    if db_name not in databases_config:
        db_name = "default"
    
    db_config = databases_config[db_name]
    cols = db_config.get('collections', {}) or {}
    text_col = cols.get('text', "")
    image_col = cols.get('image', "")
    
    return RetrievalContext(
        db_name=db_config['db_name'],
        text_collection=db_config['collections']['text'],
        image_collection=db_config['collections']['image'],
        top_k=top_k
    )


def create_t2t_retriever(db_name: str = "default", top_k: int = 10) -> T2TRetriever:
    """创建T2T检索器"""
    context = create_retrieval_context(db_name, top_k)
    return T2TRetriever(context)


def create_t2i_retriever(db_name: str = "default", top_k: int = 10) -> T2IRetriever:
    """创建T2I检索器"""
    context = create_retrieval_context(db_name, top_k)
    return T2IRetriever(context)


def create_i2t_retriever(db_name: str = "default", top_k: int = 10) -> I2TRetriever:
    """创建I2T检索器"""
    context = create_retrieval_context(db_name, top_k)
    return I2TRetriever(context)


def create_i2i_retriever(db_name: str = "default", top_k: int = 10) -> I2IRetriever:
    """创建I2I检索器"""
    context = create_retrieval_context(db_name, top_k)
    return I2IRetriever(context)


if __name__ == "__main__":
    # 使用示例
    print("=== 模块化检索器使用示例 ===")
    
    # 创建不同的检索器
    t2t = create_t2t_retriever()
    t2i = create_t2i_retriever()
    i2t = create_i2t_retriever()
    i2i = create_i2i_retriever()
    
    # T2T检索示例
    print("\n1. T2T检索:")
    t2t_result = t2t.search("心脏超声检查", top_k=3)
    print(f"找到 {t2t_result['total_results']} 个文本结果")
    
    # T2I检索示例
    print("\n2. T2I检索:")
    t2i_result = t2i.search("心脏病变", top_k=3)
    print(f"找到 {t2i_result['total_results']} 个图片结果")
    
    # I2T检索示例（需要提供实际图片路径）
    print("\n3. I2T检索:")
    # i2t_result = i2t.search("/path/to/image.jpg", top_k=3)
    # print(f"找到 {i2t_result['total_results']} 个文本结果")
    
    # I2I检索示例（需要提供实际图片路径）
    print("\n4. I2I检索:")
    # i2i_result = i2i.search("/path/to/image.jpg", top_k=3)
    # print(f"找到 {i2i_result['total_results']} 个相似图片")
