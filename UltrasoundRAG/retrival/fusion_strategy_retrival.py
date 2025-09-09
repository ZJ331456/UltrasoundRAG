"""融合检索策略模块
包含复杂的检索策略、结果融合和数据库管理

主要功能：
1. 复杂的t2t检索策略（文本块检索 + 图片标题匹配）
2. 多种检索方式的结果融合
3. 检索策略管理
4. 数据库配置管理
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from UltrasoundRAG.utils.logger import setup_logger
from UltrasoundRAG.retrival.data_structures import RetrievalResult
from UltrasoundRAG.retrival.modular_retrievers import (
    T2TRetriever, 
    T2IRetriever,
    I2TRetriever,
    I2IRetriever,
    RetrievalContext
)
from UltrasoundRAG.retrival.caption_to_image_retriever import CaptionToImageRetriever, create_caption_retriever


@dataclass
class FusionConfig:
    """融合检索配置"""
    # 基础配置
    top_k: int = 10
    enable_text_image_fusion: bool = True  # 是否启用文本-图片融合
    
    # t2t策略配置
    text_search_ratio: float = 0.7  # 文本检索在t2t中的比重
    image_caption_ratio: float = 0.3  # 图片caption检索的比重
    enable_exact_title_match: bool = True  # 是否启用精确标题匹配
    
    # 结果融合权重
    text_weight: float = 0.6
    image_weight: float = 0.4
    
    # 去重配置
    content_similarity_threshold: float = 0.8  # 内容相似度阈值
    enable_deduplication: bool = True
    # 新增：可配置来源权重与去重粒度
    source_weights: Dict[str, float] = None
    dedup_key: str = "content"


# ImageTitleExtractor 已移至 image_title_matcher.py 模块


class T2TFusionStrategy:
    """T2T融合策略 - 实现复杂的文本到文本检索"""
    
    def __init__(self, 
                 text_retriever: T2TRetriever,
                 image_retriever: T2IRetriever,
                 config: FusionConfig):
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
        # 使用统一的图片标题匹配器
        self.title_matcher = create_caption_retriever(
            db_name="default",
            search_mode="hybrid_match",
            match_mode="variant"
        )
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        执行复杂的T2T检索策略
        
        策略：
        (1) 去文本块里面检索 → 提取图片标题 → 精确匹配图片
        (2) 去图片向量数据库里面匹配caption → 获取图文对
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            包含所有检索结果的字典
        """
        top_k = top_k or self.config.top_k
        
        # 策略1：文本块检索 + 图片标题匹配
        text_results, related_images = self._text_block_with_image_strategy(query, top_k)
        
        # 策略2：图片caption向量匹配
        image_caption_results = self._image_caption_strategy(query, top_k)
        
        # 融合结果
        all_text_results = self._merge_text_results(text_results, image_caption_results)
        all_image_results = related_images
        
        return {
            'query': query,
            'text_results': all_text_results,
            'image_results': all_image_results,
            'strategy_breakdown': {
                'text_block_results': len(text_results),
                'related_images_from_text': len(related_images),
                'image_caption_results': len(image_caption_results)
            },
            'total_text_results': len(all_text_results),
            'total_image_results': len(all_image_results)
        }
    
    def _text_block_with_image_strategy(self, query: str, top_k: int) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        策略1：文本块检索 + 图片标题精确匹配
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            (文本结果列表, 相关图片列表)
        """
        # 1. 执行基础文本检索
        text_result = self.text_retriever.search(query, top_k * 2)
        text_results = text_result.get('results', [])
        
        # 2. 从文本结果中提取图片标题并匹配
        related_images = []
        all_extracted_titles = set()
        
        if self.config.enable_exact_title_match:
            for result in text_results:
                # 使用统一的图片标题匹配器
                match_result = self.title_matcher.extract_and_match(
                    text=result.content,
                    metadata=result.metadata,
                    top_k=3
                )
                
                # 收集提取的标题
                all_extracted_titles.update(match_result['extracted_titles'])
                
                # 收集匹配的图片
                related_images.extend(match_result['matched_images'])
        
        # 3. 为文本结果添加图片匹配信息
        for result in text_results:
            result.metadata['extracted_image_titles'] = list(all_extracted_titles)
            result.metadata['has_related_images'] = len(related_images) > 0
        
        self.logger.info(f"文本块策略：文本结果 {len(text_results)}，相关图片 {len(related_images)}")
        return text_results[:top_k], related_images
    
    def _image_caption_strategy(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        策略2：图片caption向量匹配
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            图片检索结果列表（作为文本结果返回）
        """
        try:
            image_result = self.image_retriever.search(query, top_k)
            image_results = image_result.get('results', [])
            
            # 将图片结果转换为文本结果（包含caption信息）
            caption_results = []
            for img_result in image_results:
                caption = img_result.content or img_result.metadata.get('caption', '')
                if caption:
                    # 创建一个新的文本结果，内容是图片的caption
                    text_result = RetrievalResult(
                        doc_id=f"caption_{img_result.doc_id}",
                        content=caption,
                        metadata={
                            **img_result.metadata,
                            'source_type': 'image_caption',
                            'original_image_path': img_result.metadata.get('image_path', ''),
                            'original_score': img_result.score
                        },
                        score=img_result.score * self.config.image_caption_ratio,
                        retrieval_type='image_caption_to_text',
                        resource_collection=img_result.resource_collection
                    )
                    caption_results.append(text_result)
            
            self.logger.info(f"图片caption策略：返回 {len(caption_results)} 个结果")
            return caption_results
            
        except Exception as e:
            self.logger.error(f"图片caption策略失败: {e}")
            return []
    
    def _merge_text_results(self, text_results: List[RetrievalResult], 
                           caption_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        合并文本检索结果和caption检索结果
        
        Args:
            text_results: 文本检索结果
            caption_results: caption检索结果
            
        Returns:
            合并后的结果列表
        """
        # 应用权重（支持自定义来源权重）
        src_weights = self.config.source_weights or {}
        for result in text_results:
            w = src_weights.get('text_search', self.config.text_search_ratio)
            result.score *= w
            result.metadata['fusion_weight'] = 'text_search'
        
        for result in caption_results:
            w = src_weights.get('image_caption', self.config.image_caption_ratio)
            result.score *= w
            result.metadata['fusion_weight'] = 'image_caption'
        
        # 合并并排序
        all_results = text_results + caption_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # 去重（可选）
        if self.config.enable_deduplication:
            all_results = self._deduplicate_results(all_results)
        
        return all_results
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        结果去重
        
        Args:
            results: 待去重的结果列表
            
        Returns:
            去重后的结果列表
        """
        unique_results = []
        seen_contents = set()
        
        for result in results:
            key = result.content if self.config.dedup_key == 'content' else result.metadata.get(self.config.dedup_key, '')
            key = (key or '')[:200]
            content_hash = hash(key)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results


class ResultFusionManager:
    """结果融合管理器 - 管理不同检索方式的结果融合"""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
    
    def fuse_multimodal_results(self, 
                               text_results: List[RetrievalResult],
                               image_results: List[RetrievalResult],
                               strategy: str = "weighted") -> List[RetrievalResult]:
        """
        融合多模态检索结果
        
        Args:
            text_results: 文本检索结果
            image_results: 图像检索结果
            strategy: 融合策略 ("weighted", "interleaved", "score_based")
            
        Returns:
            融合后的结果列表
        """
        if strategy == "weighted":
            return self._weighted_fusion(text_results, image_results)
        elif strategy == "interleaved":
            return self._interleaved_fusion(text_results, image_results)
        elif strategy == "score_based":
            return self._score_based_fusion(text_results, image_results)
        else:
            self.logger.warning(f"未知的融合策略: {strategy}，使用默认weighted策略")
            return self._weighted_fusion(text_results, image_results)
    
    def _weighted_fusion(self, text_results: List[RetrievalResult], 
                        image_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """加权融合策略"""
        # 应用权重
        for result in text_results:
            result.score *= self.config.text_weight
            result.metadata['fusion_strategy'] = 'weighted_text'
        
        for result in image_results:
            result.score *= self.config.image_weight
            result.metadata['fusion_strategy'] = 'weighted_image'
        
        # 合并并排序
        all_results = text_results + image_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:self.config.top_k]
    
    def _interleaved_fusion(self, text_results: List[RetrievalResult], 
                           image_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """交替融合策略"""
        fused_results = []
        max_len = max(len(text_results), len(image_results))
        
        for i in range(max_len):
            if i < len(text_results):
                text_results[i].metadata['fusion_strategy'] = 'interleaved_text'
                fused_results.append(text_results[i])
            
            if i < len(image_results):
                image_results[i].metadata['fusion_strategy'] = 'interleaved_image'
                fused_results.append(image_results[i])
            
            if len(fused_results) >= self.config.top_k:
                break
        
        return fused_results[:self.config.top_k]
    
    def _score_based_fusion(self, text_results: List[RetrievalResult], 
                           image_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """基于分数的融合策略"""
        all_results = []
        
        for result in text_results:
            result.metadata['fusion_strategy'] = 'score_based_text'
            all_results.append(result)
        
        for result in image_results:
            result.metadata['fusion_strategy'] = 'score_based_image'
            all_results.append(result)
        
        # 按原始分数排序
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:self.config.top_k]


class FusionRetrievalManager:
    """融合检索管理器 - 主要的检索协调器"""
    
    def __init__(self, 
                 milvus_uri: Optional[str] = None,
                 milvus_token: Optional[str] = None,
                 db_name: Optional[str] = None,
                 text_embedding_provider: str = "bge_zh_local_embedding",
                 image_model_path: Optional[str] = None,
                 image_model_config_path: Optional[str] = None,
                 fusion_config: Optional[FusionConfig] = None):
        """
        初始化融合检索管理器
        
        Args:
            milvus_uri: Milvus服务地址
            milvus_token: Milvus认证令牌
            db_name: 数据库名称
            text_embedding_provider: 文本嵌入提供者
            image_model_path: 图像模型路径
            image_model_config_path: 图像模型配置路径
            fusion_config: 融合配置
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.fusion_config = fusion_config or FusionConfig()
        
        # 创建检索上下文
        context = RetrievalContext(
            db_name=db_name or "default",
            text_collection="text_collection",  # 从配置获取
            image_collection="image_collection",  # 从配置获取
            top_k=self.fusion_config.top_k,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token
        )
        
        # 初始化基础检索器
        try:
            self.text_retriever = T2TRetriever(context)
            self.image_retriever = T2IRetriever(context)
            self.i2t_retriever = I2TRetriever(context)
            self.i2i_retriever = I2IRetriever(context)
            self.logger.info("基础检索器初始化成功")
        except Exception as e:
            self.logger.error(f"基础检索器初始化失败: {e}")
            raise
        
        # 初始化策略管理器
        self.t2t_strategy = T2TFusionStrategy(
            self.text_retriever, 
            self.image_retriever, 
            self.fusion_config
        )
        self.fusion_manager = ResultFusionManager(self.fusion_config)
        
        self.logger.info("融合检索管理器初始化完成")
    
    def t2t_search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        复杂的T2T检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            T2T检索结果
        """
        return self.t2t_strategy.search(query, top_k)
    
    def t2i_search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        T2I检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            T2I检索结果
        """
        top_k = top_k or self.fusion_config.top_k
        image_result = self.image_retriever.search(query, top_k)
        results = image_result.get('results', [])
        
        return {
            'query': query,
            'image_results': results,
            'total_results': len(results)
        }
    
    def i2t_search(self, image_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        I2T检索
        
        Args:
            image_path: 图像路径
            top_k: 返回结果数量
            
        Returns:
            I2T检索结果
        """
        top_k = top_k or self.fusion_config.top_k
        i2t_result = self.i2t_retriever.search(image_path, top_k)
        results = i2t_result.get('results', [])
        
        return {
            'query_image': image_path,
            'text_results': results,
            'total_results': len(results)
        }
    
    def i2i_search(self, image_path: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        I2I检索
        
        Args:
            image_path: 图像路径
            top_k: 返回结果数量
            
        Returns:
            I2I检索结果
        """
        top_k = top_k or self.fusion_config.top_k
        i2i_result = self.i2i_retriever.search(image_path, top_k)
        results = i2i_result.get('results', [])
        
        return {
            'query_image': image_path,
            'similar_images': results,
            'total_results': len(results)
        }
    
    def multimodal_search(self, query: str, top_k: Optional[int] = None, 
                         fusion_strategy: str = "weighted") -> Dict[str, Any]:
        """
        多模态融合检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            fusion_strategy: 融合策略
            
        Returns:
            多模态检索结果
        """
        top_k = top_k or self.fusion_config.top_k
        
        # 执行T2T和T2I检索
        t2t_result = self.t2t_search(query, top_k)
        t2i_result = self.t2i_search(query, top_k)
        
        # 融合结果
        fused_results = self.fusion_manager.fuse_multimodal_results(
            t2t_result['text_results'],
            t2i_result['image_results'],
            fusion_strategy
        )
        
        return {
            'query': query,
            'text_results': t2t_result['text_results'],
            'image_results': t2i_result['image_results'],
            'fused_results': fused_results,
            'fusion_strategy': fusion_strategy,
            'total_text_results': len(t2t_result['text_results']),
            'total_image_results': len(t2i_result['image_results']),
            'total_fused_results': len(fused_results)
        }


# 便捷函数
def create_fusion_retrieval_manager(
    milvus_uri: Optional[str] = None,
    milvus_token: Optional[str] = None,
    db_name: Optional[str] = None,
    text_embedding_provider: str = "bge_zh_local_embedding",
    image_model_path: Optional[str] = None,
    image_model_config_path: Optional[str] = None,
    **fusion_kwargs
) -> FusionRetrievalManager:
    """创建融合检索管理器的便捷函数"""
    fusion_config = FusionConfig(**fusion_kwargs)
    
    return FusionRetrievalManager(
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        db_name=db_name,
        text_embedding_provider=text_embedding_provider,
        image_model_path=image_model_path,
        image_model_config_path=image_model_config_path,
        fusion_config=fusion_config
    )
