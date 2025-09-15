"""
模块化检索器 - 增强版本，集成所有优化功能
提供独立的T2T、T2I、I2T、I2I检索功能，现在包含：

主要特点：
1. 每种检索方式完全独立
2. 基于配置的灵活参数管理
3. 支持多数据库、多集合
4. 清晰的结果结构

增强功能：
5. 查询自适应动态权重调整
6. 领域分区和智能路由
7. Caption增强处理
8. 多模态融合策略
9. 智能重排序

"""
# TODO：我要添加BM25检索，就是在构建milvus集合的时候就新加一个关于bm25索引的字段，这样就不用后面加载了，更快！

import os
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ultrasoundrag.utils.logger import setup_logger
from ultrasoundrag.core.retrieval.data_structures import RetrievalResult
from ultrasoundrag.data.stores.milvus_store import MilvusManager
from ultrasoundrag.utils.embedding_utils import embedding_provider
from ultrasoundrag.model.model_manager import get_fetal_clip_model, get_embedding_model
from ultrasoundrag.model.singleton_models import get_shared_fetal_clip
from ultrasoundrag.config import config
from ultrasoundrag.core.retrieval.caption_to_image_retriever import create_caption_retriever
from .content_processor import process_content_with_images

# 导入优化模块
from ultrasoundrag.utils.query_adaptive_weights import (
    calculate_query_adaptive_weights, get_fusion_weights, get_weight_calculator
)
from ultrasoundrag.utils.domain_partition import get_partition_manager, classify_query_domain
from ultrasoundrag.utils.caption_enhancement import create_caption_enhancer
from ultrasoundrag.core.retrieval.fusion_strategy_retrieval import create_enhanced_fusion_manager
from ultrasoundrag.core.retrieval.enhanced_reranker import create_multimodal_reranker

from PIL import Image


class GlobalComponentManager:
    """全局组件管理器 - 避免重复初始化"""
    _instance = None
    _components = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_component(self, component_name: str, factory_func, *args, **kwargs):
        """获取或创建组件"""
        if component_name not in self._components:
            self._components[component_name] = factory_func(*args, **kwargs)
        return self._components[component_name]
    
    def clear_cache(self):
        """清理组件缓存"""
        self._components.clear()


# 全局组件管理器实例
global_component_manager = GlobalComponentManager()


@dataclass
class RetrievalContext:
    """增强的检索上下文配置"""
    db_name: str
    text_collection: Optional[str]
    image_collection: Optional[str]
    top_k: int = 10
    milvus_uri: Optional[str] = None
    milvus_token: Optional[str] = None
    
    # 领域过滤与分区路由配置
    domain: Optional[str] = None
    use_partition: bool = False
    
    # 优化功能开关
    enable_dynamic_weights: bool = True
    enable_domain_partition: bool = True  
    enable_caption_enhancement: bool = True
    enable_fusion_optimization: bool = True
    enable_multimodal_rerank: bool = True
    
    # 性能配置
    enable_caching: bool = True
    similarity_threshold: float = 0.3


class BaseRetriever(ABC):
    """增强的基础检索器抽象类"""
    
    def __init__(self, context: RetrievalContext):
        self.context = context
        self.logger = setup_logger(self.__class__.__name__)
        
        # 从全局配置获取Milvus连接信息
        milvus_config = config['milvus']
        self.milvus_uri = context.milvus_uri or milvus_config['milvus_uri']
        self.milvus_token = context.milvus_token or milvus_config['milvus_token']
        
        # 初始化优化组件
        self._init_enhancement_components()
        
        # 性能统计
        self.stats = {
            'total_searches': 0,
            'avg_response_time': 0.0,
            'cache_hits': 0
        }
    
    def _init_enhancement_components(self):
        """初始化增强组件 - 使用全局管理器避免重复初始化"""
        if self.context.enable_domain_partition:
            self.partition_manager = global_component_manager.get_component(
                'partition_manager', get_partition_manager
            )
        
        if self.context.enable_caption_enhancement:
            self.caption_enhancer = global_component_manager.get_component(
                'caption_enhancer', create_caption_enhancer
            )
            
        if self.context.enable_fusion_optimization:
            self.fusion_manager = global_component_manager.get_component(
                'fusion_manager', create_enhanced_fusion_manager
            )
            
        if self.context.enable_multimodal_rerank:
            self.reranker = global_component_manager.get_component(
                'multimodal_reranker', create_multimodal_reranker
            )
    
    def _apply_domain_filtering(self, query: str) -> tuple[Optional[str], Optional[List[str]]]:
        """应用领域过滤"""
        if not self.context.enable_domain_partition or not hasattr(self, 'partition_manager'):
            return None, None
        
        filter_expr = self.partition_manager.get_domain_filter_expression(query)
        partitions, _ = self.partition_manager.get_search_partitions(query)
        return filter_expr, partitions
    
    def _get_adaptive_weights(self, query: str) -> tuple[float, float]:
        """获取自适应权重"""
        if not self.context.enable_dynamic_weights:
            return 0.6, 0.4  # 默认权重
        return calculate_query_adaptive_weights(query)
    
    def _update_stats(self, response_time: float):
        """更新性能统计"""
        self.stats['total_searches'] += 1
        total = self.stats['total_searches']
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = (current_avg * (total - 1) + response_time) / total
    
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
        
        # 获取CLIP模型（使用单例管理器，避免重复加载）
        self.clip_model = get_shared_fetal_clip()

        self.logger.info("T2T检索器初始化完成")
    
    def search(self, query: str, top_k: Optional[int] = None, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        执行增强的文本到文本检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            strategy: 检索策略 ("basic", "fusion", "enhanced")
            
        Returns:
            T2T检索结果
        """
        start_time = time.time()
        top_k = top_k or self.context.top_k
        strategy = strategy or self.t2t_config.get('strategy', 'enhanced')
        
        try:
            if strategy == "enhanced":
                result = self._enhanced_search(query, top_k)
            elif strategy == "fusion":
                result = self._fusion_search(query, top_k)
            else:
                result = self._basic_search(query, top_k)
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(response_time)
            result['response_time'] = response_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"T2T检索失败: {e}")
            return self._empty_result(query, str(e))
    
    def _enhanced_search(self, query: str, top_k: int) -> Dict[str, Any]:
        """增强的文本检索 - 集成所有优化功能"""
        try:
            # 1. 获取自适应权重
            qwen_weight, clip_weight = self._get_adaptive_weights(query)
            
            # 2. 生成查询向量
            qwen_vector = self.text_embedder.get_query_embedding(query)
            tokens = self.clip_model.tokenize_text([query])
            # 由 FetalCLIP 模型内部负责将 token 迁移到正确设备
            clip_vector = self.clip_model.encode_text(tokens).cpu().numpy().tolist()[0]
            
            if not (qwen_vector and len(qwen_vector) == 1024 and clip_vector and len(clip_vector) == 768):
                return self._empty_result(query, "向量生成失败")
            
            # 3. 应用领域过滤
            filter_expr, partitions = self._apply_domain_filtering(query)
            
            # 4. 多向量融合检索
            field_to_embedding = {
                "text_vector_qwen_1024": qwen_vector,
                "text_vector_clip_768": clip_vector,
            }
            weights = {
                "text_vector_qwen_1024": qwen_weight,
                "text_vector_clip_768": clip_weight,
            }
            
            raw_results = self.text_manager.search_multi_vectors(
                field_to_embedding,
                top_k=top_k * 2,  # 获取更多候选用于重排序
                weights=weights,
                filter_expr=filter_expr,
                partition_names=partitions,
            )
            
            # 5. 转换为标准结果格式
            results = self._convert_to_retrieval_results(raw_results, 'text_to_text_enhanced')
            
            # 5.5. 应用相似度阈值过滤
            similarity_threshold = self.t2t_config.get('similarity_threshold', 0.2)
            filtered_results = []
            for result in results:
                if result.score >= similarity_threshold:
                    filtered_results.append(result)
            results = filtered_results
            self.logger.debug(f"T2T相似度过滤: {len(raw_results)} -> {len(results)}, 阈值: {similarity_threshold}")
            
            # 6. 应用多模态重排序
            if self.context.enable_multimodal_rerank and hasattr(self, 'reranker') and results:
                results = self.reranker.rerank_results(query, results)
            
            # 7. 最终结果
            final_results = results[:top_k]
            
            return {
                'query': query,
                'retrieval_type': 't2t_enhanced',
                'strategy': 'enhanced',
                'results': final_results,
                'total_results': len(final_results),
                'weights_used': {'qwen': qwen_weight, 'clip': clip_weight},
                'domain_filter': filter_expr,
                'partitions': partitions
            }
            
        except Exception as e:
            self.logger.error(f"增强T2T检索失败: {e}")
            return self._empty_result(query, str(e))
    
    def _convert_to_retrieval_results(self, raw_results: List[Dict], retrieval_type: str) -> List[RetrievalResult]:
        """转换原始结果为标准格式"""
        
        results = []
        for result in raw_results:
            content = result.get('content', '')
            image_paths = result.get('image_paths', [])
            image_captions = result.get('image_captions', [])
            
            # 处理图片链接转换
            processed_content, original_content = process_content_with_images(
                content, image_paths, image_captions, self.context.db_name
            )
            
            retrieval_result = RetrievalResult(
                doc_id=str(result.get('id', '')),
                content=processed_content,  # 使用处理后的内容
                metadata={
                    'title': result.get('title', ''),
                    'md_file': result.get('md_file', ''),
                    'document_name': result.get('document_name', ''),
                    'chunk_index': result.get('chunk_index', 0),
                    'image_paths': image_paths,
                    'image_captions': image_captions,
                    'original_content': original_content,  # 添加原始内容
                    'domain': result.get('domain', ''),
                    'search_strategy': 'enhanced_t2t'
                },
                score=result.get('score', 0.0),
                retrieval_type=retrieval_type,
                resource_collection=self.text_manager.collection_name
            )
            results.append(retrieval_result)
        return results
    
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
                content = result.get('content', '')
                image_paths = result.get('image_paths', [])
                image_captions = result.get('image_captions', [])
                
                # 处理图片链接转换
                processed_content, original_content = process_content_with_images(
                    content, image_paths, image_captions, self.context.db_name
                )
                
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=processed_content,  # 使用处理后的内容
                    metadata={
                        'title': result.get('title', ''),
                        'md_file': result.get('md_file', ''),
                        'document_name': result.get('document_name', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'image_paths': image_paths,
                        'image_captions': image_captions,
                        'original_content': original_content,  # 添加原始内容
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
                content = result.get('content', '')
                image_paths = result.get('image_paths', [])
                image_captions = result.get('image_captions', [])
                
                # 处理图片链接转换
                processed_content, original_content = process_content_with_images(
                    content, image_paths, image_captions, self.context.db_name
                )
                
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=processed_content,  # 使用处理后的内容
                    metadata={
                        'title': result.get('title', ''),
                        'md_file': result.get('md_file', ''),
                        'document_name': result.get('document_name', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'image_paths': image_paths,
                        'image_captions': image_captions,
                        'original_content': original_content,  # 添加原始内容
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
            from ultrasoundrag.core.retrieval.modular_retrievers import T2IRetriever
            
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
                            'original_image_path': img_result.metadata.get('image_path', ''),
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
        
        # 说明：T2I 检索器依赖图片向量集合（image collection）。
        # 在初始化阶段完成三件事：
        # 1) 准备 Milvus 管理器（连接到图片集合）
        # 2) 获取共享的 FetalCLIP 模型实例（单例，避免重复加载）
        # 3) 读取 T2I 专属配置（如相似度阈值等）
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
        
        # 获取CLIP模型（使用单例管理器，避免重复加载）
        self.clip_model = get_shared_fetal_clip()
        
        # 获取T2I配置
        self.t2i_config = config['retriever']['retrieval_modes']['t2i']
        
        self.logger.info("T2I检索器初始化完成")
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        执行增强的文本到图片检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            T2I检索结果
        """
        start_time = time.time()
        top_k = top_k or self.context.top_k
        
        try:
            # 步骤1：将文本查询编码为 CLIP 文本向量
            # 原因：图片集合中的主向量字段是 image_vector（来自CLIP图像编码器），
            #       CLIP 的文本编码子空间与图像编码子空间是对齐的，
            #       因此使用文本向量去检索图像向量可实现跨模态匹配。
            # 使用CLIP文本编码器生成查询向量（仅用于匹配 image_vector）
            tokens = self.clip_model.tokenize_text([query])
            query_vector = self.clip_model.encode_text(tokens).cpu().numpy().tolist()[0]

            if not query_vector:
                return self._empty_result(query, "查询向量生成失败")
            
            # 步骤2：可选的领域过滤/分区路由
            # 作用：将搜索限定在指定的领域或分区，提高查准并降低搜索成本。
            # 应用领域过滤
            filter_expr, partitions = self._apply_domain_filtering(query)
            
            # 步骤3：向量检索（先取 2x top_k 作为候选）
            # 取更多候选的原因：后续会做阈值过滤与重排序，
            # 通过扩大候选集可以提升最终召回率与排序质量。
            # 仅基于 image_vector 进行检索（不融合 caption 向量）
            milvus_results = self.image_manager.search(
                query_vector,
                top_k * 2,
                filter_expr=filter_expr,
                partition_names=partitions
            )
            
            # 步骤4：统一结果为标准结构（RetrievalResult），便于后续处理
            # 转换结果格式
            results = self._convert_to_image_results(milvus_results, 'text_to_image_enhanced')
            
            # 步骤5：相似度阈值过滤（提升召回稳定性与可控性）
            # 配置来自 config['retriever']['retrieval_modes']['t2i']['similarity_threshold']
            # 默认较低（0.1），以保证跨模态召回的覆盖度。
            # 应用相似度阈值过滤，提升T2I检索质量
            similarity_threshold = self.t2i_config.get('similarity_threshold', 0.1)
            filtered_results = []
            for result in results:
                if result.score >= similarity_threshold:
                    filtered_results.append(result)
            results = filtered_results
            self.logger.debug(f"T2I相似度过滤: {len(milvus_results)} -> {len(results)}, 阈值: {similarity_threshold}")
            
            # 步骤6（可选）：Caption 语义增强
            # 目的：当查询更像标题/术语时，利用图片自带 caption 与查询做二次打分融合，
            #       可在低分场景提升排序质量与相关性稳定性。
            # Caption增强处理
            if self.context.enable_caption_enhancement and hasattr(self, 'caption_enhancer') and results:
                results = self._enhance_with_captions(query, results)
            
            # 步骤7（可选）：多模态重排序
            # 目的：结合 CrossEncoder / 多模态特征对候选做细粒度重排，
            #       一般用于进一步提升 Top-K 的相关性质量。
            # 多模态重排序
            if self.context.enable_multimodal_rerank and hasattr(self, 'reranker') and results:
                results = self.reranker.rerank_results(query, results)
            
            # 步骤8：裁剪为最终 Top-K 并返回
            # 最终结果
            final_results = results[:top_k]
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(response_time)
            
            return {
                'query': query,
                'retrieval_type': 't2i_enhanced',
                'results': final_results,
                'total_results': len(final_results),
                'domain_filter': filter_expr,
                'partitions': partitions,
                'response_time': response_time
            }
            
        except Exception as e:
            self.logger.error(f"T2I检索失败: {e}")
            return self._empty_result(query, str(e))
    
    def _convert_to_image_results(self, raw_results: List[Dict], retrieval_type: str) -> List[RetrievalResult]:
        """转换图像结果为标准格式"""
        results = []
        for result in raw_results:
            # 构建图片的API访问路径
            image_path = result.get('image_path', '')
            caption = result.get('caption', '')
            
            # 从图片路径中提取hash（假设路径格式为 /path/to/images/{hash}.jpg）
            image_hash = ""
            if image_path:
                filename = os.path.basename(image_path)
                image_hash = os.path.splitext(filename)[0]  # 去掉扩展名
            
            # 构建HTML格式的content
            if image_hash and caption:
                content = f'<image src="/api/images/{image_hash}.jpg" caption="{caption}"/>'
            elif image_hash:
                content = f'<image src="/api/images/{image_hash}.jpg" caption=""/>'
            else:
                content = caption  # 降级到原始caption
            
            retrieval_result = RetrievalResult(
                doc_id=str(result.get('id', '')),
                content=content,  # 使用HTML格式的content
                metadata={
                    'image_path': image_path,
                    'caption': caption,
                    'image_hash': image_hash,
                    'source': result.get('source', ''),
                    'file_size': result.get('file_size', 0),
                    'domain': result.get('domain', ''),
                    'search_strategy': 'enhanced_t2i'
                },
                score=result.get('score', 0.0),
                retrieval_type=retrieval_type,
                resource_collection=self.image_manager.collection_name
            )
            results.append(retrieval_result)
        return results
    
    def _enhance_with_captions(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """使用Caption增强处理结果"""
        if not self.context.enable_caption_enhancement or not hasattr(self, 'caption_enhancer'):
            return results
        
        try:
            # 准备caption候选
            caption_candidates = []
            for result in results:
                caption = result.content or result.metadata.get('caption', '')
                if caption:
                    caption_candidates.append({
                        'original_caption': caption,
                        'cleaned_caption': caption,
                        'result': result
                    })
            
            if not caption_candidates:
                return results
            
            # 使用Caption增强匹配
            scored_captions = self.caption_enhancer.search_captions_by_query(query, caption_candidates)
            
            # 更新结果分数
            enhanced_results = []
            for caption_data, caption_score in scored_captions:
                result = caption_data['result']
                # 结合原始分数和caption匹配分数
                result.score = result.score * 0.7 + caption_score * 0.3
                result.metadata['caption_match_score'] = caption_score
                result.metadata['caption_enhanced'] = True
                enhanced_results.append(result)
            
            # 按新分数排序
            enhanced_results.sort(key=lambda x: x.score, reverse=True)
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Caption增强处理失败: {e}")
            return results
    
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
            collection_type="md",
            collection_name=context.text_collection,
            enable_domain_partition=context.use_partition
        )
        
        # 获取CLIP模型（使用单例管理器，避免重复加载）
        self.clip_model = get_shared_fetal_clip()
        
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
            milvus_results = self.text_manager.search(query_vector, top_k * 2, filter_expr=filter_expr, partition_names=partitions)

            # 应用相似度阈值过滤
            similarity_threshold = self.i2t_config.get('similarity_threshold', 0.15)
            filtered_milvus_results = []
            for result in milvus_results:
                if result.get('score', 0) >= similarity_threshold:
                    filtered_milvus_results.append(result)
                if len(filtered_milvus_results) >= top_k:
                    break
            
            self.logger.debug(f"I2T相似度过滤: {len(milvus_results)} -> {len(filtered_milvus_results)}, 阈值: {similarity_threshold}")

            # 转换文本结果
            results = []
            for result in filtered_milvus_results:
                content = result.get('content', '')
                image_paths = result.get('image_paths', [])
                image_captions = result.get('image_captions', [])
                
                # 处理图片链接转换
                processed_content, original_content = process_content_with_images(
                    content, image_paths, image_captions, self.context.db_name
                )
                
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=processed_content,  # 使用处理后的内容
                    metadata={
                        'title': result.get('title', ''),
                        'md_file': result.get('md_file', ''),
                        'document_name': result.get('document_name', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'image_links': result.get('image_links', []),
                        'image_paths': image_paths,
                        'image_captions': image_captions,
                        'original_content': original_content,  # 添加原始内容
                        'query_image_path': image_path,
                        'search_strategy': 'image_to_text'
                    },
                    score=result.get('score', 0.0),
                    retrieval_type='image_to_text',
                    resource_collection=self.text_manager.collection_name
                )
                results.append(retrieval_result)

            # 从文本结果中提取caption并到图片集合匹配（仅标量/模糊匹配，不做向量匹配）
            matched_images: List[RetrievalResult] = []
            try:
                # 汇总所有文本块的图片标题
                caption_set = set()
                for r in results:
                    caps = r.metadata.get('image_captions', []) or []
                    for c in caps:
                        c = str(c).strip()
                        if c:
                            caption_set.add(c)

                if caption_set:
                    cap_retriever = create_caption_retriever(db_name=self.context.db_name)
                    # 使用精简接口：直接把 caption 列表交给 MD->Image 匹配
                    cap_result = cap_retriever.search_from_md_image_captions(list(caption_set)[:max(1, top_k*2)], top_k_per_caption=max(1, top_k//2), use_like=True)
                    matched_images = cap_result.get('results', [])[:top_k]
            except Exception as e:
                self.logger.warning(f"从文本结果匹配图片失败: {e}")

            return {
                'query_image': image_path,
                'retrieval_type': 'i2t',
                'results': results,
                'total_results': len(results),
                'matched_images': matched_images,
                'matched_image_count': len(matched_images)
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
            collection_type="image",
            collection_name=context.image_collection,
            enable_domain_partition=context.use_partition
        )
        
        # 获取CLIP模型（使用单例管理器，避免重复加载）
        self.clip_model = get_shared_fetal_clip()
        
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
            milvus_results = self.image_manager.search(query_vector, top_k * 2)
            
            # 应用相似度阈值过滤
            similarity_threshold = self.i2i_config.get('similarity_threshold', 0.2)
            filtered_milvus_results = []
            for result in milvus_results:
                if result.get('score', 0) >= similarity_threshold:
                    filtered_milvus_results.append(result)
                if len(filtered_milvus_results) >= top_k:
                    break
            
            self.logger.debug(f"I2I相似度过滤: {len(milvus_results)} -> {len(filtered_milvus_results)}, 阈值: {similarity_threshold}")
            
            # 转换结果
            results = []
            for result in filtered_milvus_results:
                # 构建图片的API访问路径
                result_image_path = result.get('image_path', '')
                caption = result.get('caption', '')
                
                # 从图片路径中提取hash
                image_hash = ""
                if result_image_path:
                    filename = os.path.basename(result_image_path)
                    image_hash = os.path.splitext(filename)[0]  # 去掉扩展名
                
                # 构建HTML格式的content
                if image_hash and caption:
                    content = f'<image src="/api/images/{image_hash}.jpg" caption="{caption}"/>'
                elif image_hash:
                    content = f'<image src="/api/images/{image_hash}.jpg" caption=""/>'
                else:
                    content = caption  # 降级到原始caption
                
                metadata = {
                    'image_path': result_image_path,
                    'caption': caption,
                    'image_hash': image_hash,
                    'source': result.get('source', ''),
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
                    content=content,  # 使用HTML格式的content
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


class EnhancedMultimodalRetriever:
    """增强的统一多模态检索器 - 智能策略选择"""
    
    def __init__(self, context: RetrievalContext):
        self.context = context
        self.logger = setup_logger(self.__class__.__name__)
        
        # 初始化子检索器
        self.t2t_retriever = T2TRetriever(context) if context.text_collection else None
        self.t2i_retriever = T2IRetriever(context) if context.image_collection else None
        self.i2t_retriever = I2TRetriever(context) if context.text_collection else None
        self.i2i_retriever = I2IRetriever(context) if context.image_collection else None
        
        # 初始化融合管理器 - 使用全局管理器
        if context.enable_fusion_optimization:
            self.fusion_manager = global_component_manager.get_component(
                'fusion_manager', create_enhanced_fusion_manager
            )
        
        # 性能统计
        self.stats = {
            'total_searches': 0,
            'mode_usage': {'t2t': 0, 't2i': 0, 'i2t': 0, 'i2i': 0, 'multimodal': 0},
            'avg_response_time': 0.0
        }
    
    def search(self, query: Union[str, Dict], mode: str = "auto", top_k: Optional[int] = None, 
              text_query: Optional[str] = None, image_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        统一的增强多模态检索接口
        
        Args:
            query: 主查询（文本字符串、图片路径或包含路径的字典）
            mode: 检索模式 ("t2t", "t2i", "i2t", "i2i", "multimodal", "auto")
            top_k: 返回结果数量
            text_query: 明确的文本查询（用于图文混合）
            image_path: 明确的图片路径（用于图文混合）
            
        Returns:
            统一的检索结果
        """
        start_time = time.time()
        top_k = top_k or self.context.top_k
        
        # 解析输入参数
        parsed_input = self._parse_input(query, text_query, image_path)
        
        # 自动模式选择
        if mode == "auto":
            mode = self._auto_select_mode(parsed_input)
        
        try:
            if mode == "t2t" and self.t2t_retriever:
                if not parsed_input['text_query']:
                    raise ValueError("T2T模式需要文本查询")
                result = self.t2t_retriever.search(parsed_input['text_query'], top_k, **kwargs)
                
            elif mode == "t2i" and self.t2i_retriever:
                if not parsed_input['text_query']:
                    raise ValueError("T2I模式需要文本查询")
                result = self.t2i_retriever.search(parsed_input['text_query'], top_k, **kwargs)
                
            elif mode == "i2t" and self.i2t_retriever:
                if not parsed_input['image_path']:
                    raise ValueError("I2T模式需要图片路径")
                result = self.i2t_retriever.search(parsed_input['image_path'], top_k, **kwargs)
                
            elif mode == "i2i" and self.i2i_retriever:
                if not parsed_input['image_path']:
                    raise ValueError("I2I模式需要图片路径")
                result = self.i2i_retriever.search(parsed_input['image_path'], top_k, **kwargs)
                
            elif mode == "multimodal":
                result = self._multimodal_search(parsed_input, top_k, **kwargs)
                
            else:
                raise ValueError(f"不支持的检索模式: {mode} 或缺少对应的检索器")
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(mode, response_time)
            result['unified_response_time'] = response_time
            result['parsed_input'] = parsed_input
            result['selected_mode'] = mode
            
            return result
            
        except Exception as e:
            self.logger.error(f"统一检索失败: {e}")
            return {
                'query': str(query),
                'retrieval_type': f'unified_{mode}',
                'results': [],
                'total_results': 0,
                'error': str(e),
                'response_time': time.time() - start_time,
                'parsed_input': parsed_input,
                'selected_mode': mode
            }
    
    def _parse_input(self, query: Union[str, Dict], text_query: Optional[str] = None, 
                    image_path: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        解析输入参数，提取文本查询和图片路径
        
        Args:
            query: 主查询
            text_query: 明确的文本查询
            image_path: 明确的图片路径
            
        Returns:
            包含text_query和image_path的字典
        """
        parsed = {'text_query': None, 'image_path': None}
        
        # 优先使用明确指定的参数
        if text_query:
            parsed['text_query'] = text_query
        if image_path:
            parsed['image_path'] = image_path
        
        # 如果是字典格式的查询
        if isinstance(query, dict):
            if not parsed['text_query'] and 'query' in query:
                parsed['text_query'] = query['query']
            if not parsed['text_query'] and 'text_query' in query:
                parsed['text_query'] = query['text_query']
            if not parsed['image_path'] and 'image_path' in query:
                parsed['image_path'] = query['image_path']
        
        # 如果是字符串查询
        elif isinstance(query, str):
            # 判断是否为图片路径
            if query.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                if not parsed['image_path']:
                    parsed['image_path'] = query
            else:
                if not parsed['text_query']:
                    parsed['text_query'] = query
        
        return parsed
    
    def _auto_select_mode(self, parsed_input: Dict[str, Optional[str]]) -> str:
        """
        根据输入自动选择最佳的单一检索模式
        
        Auto模式的策略：
        - 只有文本输入：根据文本特征智能选择 t2t 或 t2i
        - 只有图片输入：根据应用场景选择 i2t 或 i2i
        - 图文混合输入：选择 multimodal 进行融合检索
        
        Args:
            parsed_input: 解析后的输入，包含text_query和image_path
            
        Returns:
            选择的检索模式
        """
        text_query = parsed_input['text_query']
        image_path = parsed_input['image_path']
        
        # 图文混合输入 -> 使用multimodal融合
        if text_query and image_path:
            return "multimodal"
        
        # 只有文本输入 -> 智能选择t2t或t2i
        elif text_query and not image_path:
            return self._analyze_text_for_mode_selection(text_query)
        
        # 只有图片输入 -> 默认选择i2t（图片理解文档）
        elif image_path and not text_query:
            return "i2t"  # 图片输入通常是想理解图片内容，所以选择i2t
        
        # 都没有输入
        else:
            raise ValueError("需要至少提供文本查询或图片路径")
    
    # 这里会只能选择输入的问题是想去选择找图片还是对应的文本的一个简单分析逻辑
    def _analyze_text_for_mode_selection(self, text_query: str) -> str:
        """
        分析文本查询特征，智能选择t2t或t2i模式
        
        Args:
            text_query: 文本查询
            
        Returns:
            选择的模式 ("t2t" 或 "t2i")
        """
        query_lower = text_query.lower().strip()
        
        # 图片相关关键词 -> t2i
        image_keywords = [
            '图', '图片', '图像', '照片', '显示', '看到', '截面', '切面', 
            '影像', '表现', '征象', '形态', '外观', '样子', '长什么样',
            'image', 'picture', 'show', 'appearance', 'look like'
        ]
        
        # 文档相关关键词 -> t2t  
        text_keywords = [
            '方法', '步骤', '流程', '原理', '机制', '原因', '如何', '怎么',
            '诊断', '治疗', '检查', '操作', '技术', '指南', '标准',
            'method', 'procedure', 'how to', 'diagnosis', 'treatment'
        ]
        
        # 计算关键词匹配分数
        image_score = sum(1 for keyword in image_keywords if keyword in query_lower)
        text_score = sum(1 for keyword in text_keywords if keyword in query_lower)
        
        # 如果包含明确的图片关键词，选择t2i
        if image_score > text_score:
            return "t2i"
        # 如果包含明确的文档关键词，选择t2t
        elif text_score > image_score:
            return "t2t"
        # 默认情况下选择t2t（文档检索通常更通用）
        else:
            return "t2t"
    
    def _multimodal_search(self, parsed_input: Dict[str, Optional[str]], top_k: int, **kwargs) -> Dict[str, Any]:
        """
        执行多模态融合检索
        
        根据输入情况选择检索策略：
        - 只有文本：执行 t2t + t2i
        - 只有图片：执行 i2t + i2i  
        - 图文混合：执行 i2t + t2t + t2i
        """
        start_time = time.time()
        
        text_query = parsed_input['text_query']
        image_path = parsed_input['image_path']
        
        # 存储各种检索结果
        t2t_results = []
        t2i_results = []
        i2t_results = []
        i2i_results = []
        
        retrieval_breakdown = {
            't2t_executed': False,
            't2i_executed': False,
            'i2t_executed': False,
            'i2i_executed': False
        }
        
        try:
            # 情况1：只有文本输入 -> 执行 t2t + t2i
            if text_query and not image_path:
                if self.t2t_retriever:
                    t2t_result = self.t2t_retriever.search(text_query, top_k, **kwargs)
                    t2t_results = t2t_result.get('results', [])
                    retrieval_breakdown['t2t_executed'] = True
                
                if self.t2i_retriever:
                    t2i_result = self.t2i_retriever.search(text_query, top_k, **kwargs)
                    t2i_results = t2i_result.get('results', [])
                    retrieval_breakdown['t2i_executed'] = True
            
            # 情况2：只有图片输入 -> 执行 i2t + i2i
            elif image_path and not text_query:
                if self.i2t_retriever:
                    i2t_result = self.i2t_retriever.search(image_path, top_k, **kwargs)
                    i2t_results = i2t_result.get('results', [])
                    retrieval_breakdown['i2t_executed'] = True
                
                if self.i2i_retriever:
                    i2i_result = self.i2i_retriever.search(image_path, top_k, **kwargs)
                    i2i_results = i2i_result.get('results', [])
                    retrieval_breakdown['i2i_executed'] = True
            
            # 情况3：图文混合输入 -> 执行 i2t + t2t + t2i
            elif text_query and image_path:
                # 基于图片理解内容
                if self.i2t_retriever:
                    i2t_result = self.i2t_retriever.search(image_path, top_k, **kwargs)
                    i2t_results = i2t_result.get('results', [])
                    retrieval_breakdown['i2t_executed'] = True
                
                # 基于文本查询找文档
                if self.t2t_retriever:
                    t2t_result = self.t2t_retriever.search(text_query, top_k, **kwargs)
                    t2t_results = t2t_result.get('results', [])
                    retrieval_breakdown['t2t_executed'] = True
                
                # 基于文本查询找相关图片
                if self.t2i_retriever:
                    t2i_result = self.t2i_retriever.search(text_query, top_k, **kwargs)
                    t2i_results = t2i_result.get('results', [])
                    retrieval_breakdown['t2i_executed'] = True
            
            # 合并所有结果并应用权重
            all_results = []
            
            # 为不同类型的结果分配权重
            if text_query and image_path:
                # 图文混合：图片理解40%，文本理解35%，相关图片25%
                weights = {'i2t': 0.4, 't2t': 0.35, 't2i': 0.25}
            elif text_query and not image_path:
                # 只有文本：文本检索60%，图片检索40%
                weights = {'t2t': 0.6, 't2i': 0.4}
            elif image_path and not text_query:
                # 只有图片：图片到文本70%，图片到图片30%
                weights = {'i2t': 0.7, 'i2i': 0.3}
            else:
                weights = {}
            
            # 应用权重并合并结果
            for result in i2t_results:
                result.score = result.score * weights.get('i2t', 1.0)
                result.metadata['multimodal_component'] = 'image_understanding'
                all_results.append(result)
            
            for result in t2t_results:
                result.score = result.score * weights.get('t2t', 1.0)
                result.metadata['multimodal_component'] = 'text_understanding'
                all_results.append(result)
            
            for result in t2i_results:
                result.score = result.score * weights.get('t2i', 1.0)
                result.metadata['multimodal_component'] = 'related_images'
                all_results.append(result)
            
            for result in i2i_results:
                result.score = result.score * weights.get('i2i', 1.0)
                result.metadata['multimodal_component'] = 'similar_images'
                all_results.append(result)
            
            # 使用增强融合策略（如果可用）
            if (self.context.enable_fusion_optimization and hasattr(self, 'fusion_manager') 
                and all_results):
                try:
                    # 将结果按类型分组用于融合
                    text_type_results = [r for r in all_results if r.metadata.get('multimodal_component') in ['image_understanding', 'text_understanding']]
                    image_type_results = [r for r in all_results if r.metadata.get('multimodal_component') in ['related_images', 'similar_images']]
                    
                    fusion_result = self.fusion_manager.fuse_multimodal_results(
                        text_query or "图像查询", text_type_results, image_type_results
                    )
                    final_results = fusion_result.results[:top_k]
                    fusion_strategy = fusion_result.strategy_used
                    fusion_weights = fusion_result.weights_applied
                except Exception as e:
                    self.logger.warning(f"融合策略失败，使用简单排序: {e}")
                    all_results.sort(key=lambda x: x.score, reverse=True)
                    final_results = all_results[:top_k]
                    fusion_strategy = "simple_sort_fallback"
                    fusion_weights = weights
            else:
                # 简单排序
                all_results.sort(key=lambda x: x.score, reverse=True)
                final_results = all_results[:top_k]
                fusion_strategy = "weighted_sort"
                fusion_weights = weights
            
            response_time = time.time() - start_time
            
            return {
                'query': text_query or "图像查询",
                'image_query': image_path,
                'retrieval_type': 'multimodal_enhanced',
                'fusion_strategy': fusion_strategy,
                'fusion_weights': fusion_weights,
                'results': final_results,
                'total_results': len(final_results),
                'retrieval_breakdown': retrieval_breakdown,
                'component_counts': {
                    'i2t_results': len(i2t_results),
                    't2t_results': len(t2t_results),
                    't2i_results': len(t2i_results),
                    'i2i_results': len(i2i_results)
                },
                'response_time': response_time
            }
            
        except Exception as e:
            self.logger.error(f"多模态检索失败: {e}")
            return {
                'query': text_query or "图像查询",
                'image_query': image_path,
                'retrieval_type': 'multimodal_enhanced',
                'results': [],
                'total_results': 0,
                'error': str(e),
                'retrieval_breakdown': retrieval_breakdown,
                'response_time': time.time() - start_time
            }
    
    def _update_stats(self, mode: str, response_time: float):
        """更新性能统计"""
        self.stats['total_searches'] += 1
        self.stats['mode_usage'][mode] = self.stats['mode_usage'].get(mode, 0) + 1
        
        total = self.stats['total_searches']
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = (current_avg * (total - 1) + response_time) / total


def create_enhanced_multimodal_retriever(db_name: str = "default", top_k: int = 10, **context_kwargs) -> EnhancedMultimodalRetriever:
    """创建增强的统一多模态检索器"""
    context = create_retrieval_context(db_name, top_k)
    
    # 更新context配置
    for key, value in context_kwargs.items():
        if hasattr(context, key):
            setattr(context, key, value)
    
    return EnhancedMultimodalRetriever(context)


def clear_global_components():
    """清理全局组件缓存 - 用于内存管理或重新初始化"""
    global_component_manager.clear_cache()


def get_component_stats():
    """获取组件缓存统计信息"""
    return {
        'cached_components': list(global_component_manager._components.keys()),
        'component_count': len(global_component_manager._components)
    }


if __name__ == "__main__":
    # 使用示例
    print("=== 模块化检索器使用示例 ===")
    
    # 显示初始组件状态
    print("初始组件缓存:", get_component_stats())
    
    # 创建不同的检索器
    t2t = create_t2t_retriever()
    print("创建T2T检索器后的组件缓存:", get_component_stats())
    
    t2i = create_t2i_retriever()
    print("创建T2I检索器后的组件缓存:", get_component_stats())
    
    i2t = create_i2t_retriever()
    print("创建I2T检索器后的组件缓存:", get_component_stats())
    
    i2i = create_i2i_retriever()
    print("创建I2I检索器后的组件缓存:", get_component_stats())
    
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
    
    # 最终组件状态
    print("\n=== 组件重用统计 ===")
    stats = get_component_stats()
    print(f"共缓存了 {stats['component_count']} 个组件:")
    for component in stats['cached_components']:
        print(f"  - {component}")
    
    print("\n提示: 所有检索器现在共享相同的增强组件实例，避免了重复初始化！")
