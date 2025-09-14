"""增强的融合检索策略模块
包含复杂的检索策略、结果融合和数据库管理

主要功能：
1. 复杂的t2t检索策略（文本块检索 + 图片标题匹配）
2. 多种检索方式的结果融合
3. 检索策略管理
4. 数据库配置管理

增强功能：
5. 查询自适应权重调整
6. 动态文本/图像融合权重
7. 智能融合策略选择
8. 领域分区集成
9. 多模态重排序
"""

import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

if TYPE_CHECKING:
    from ultrasoundrag.core.retrieval.modular_retrievers import T2TRetriever, T2IRetriever, I2TRetriever, I2IRetriever, RetrievalContext

from ultrasoundrag.utils.logger import setup_logger
from ultrasoundrag.core.retrieval.data_structures import RetrievalResult
from ultrasoundrag.core.retrieval.caption_to_image_retriever import CaptionToImageRetriever, create_caption_retriever

# 导入增强模块
from ultrasoundrag.utils.query_adaptive_weights import (
    get_fusion_weights, get_weight_calculator, calculate_query_adaptive_weights
)
from ultrasoundrag.utils.domain_partition import get_partition_manager, classify_query_domain
from ultrasoundrag.utils.caption_enhancement import create_caption_enhancer  
from ultrasoundrag.core.retrieval.enhanced_reranker import create_multimodal_reranker


class FusionStrategy(Enum):
    """融合策略枚举"""
    SIMPLE_CONCAT = "simple_concat"  # 简单拼接
    WEIGHTED_MERGE = "weighted_merge"  # 加权合并
    ADAPTIVE_SMART = "adaptive_smart"  # 自适应智能融合
    INTERLEAVED = "interleaved"  # 交错融合
    SCORE_BASED = "score_based"  # 基于分数的融合


@dataclass
class FusionConfig:
    """增强的融合检索配置"""
    # 基础配置
    top_k: int = 10
    enable_text_image_fusion: bool = True  # 是否启用文本-图片融合
    
    # t2t策略配置
    text_search_ratio: float = 0.7  # 文本检索在t2t中的比重
    image_caption_ratio: float = 0.3  # 图片caption检索的比重
    enable_exact_title_match: bool = True  # 是否启用精确标题匹配
    
    # 结果融合权重（基础值，会被动态调整）
    text_weight: float = 0.6
    image_weight: float = 0.4
    
    # 去重配置
    content_similarity_threshold: float = 0.8  # 内容相似度阈值
    enable_deduplication: bool = True
    source_weights: Dict[str, float] = field(default_factory=dict)
    dedup_key: str = "content"
    
    # 增强功能开关
    enable_adaptive_weights: bool = True  # 启用自适应权重
    enable_domain_partition: bool = True  # 启用领域分区
    enable_caption_enhancement: bool = True  # 启用Caption增强
    enable_multimodal_rerank: bool = True  # 启用多模态重排序
    enable_smart_fusion: bool = True  # 启用智能融合
    
    # 融合策略配置
    default_fusion_strategy: FusionStrategy = FusionStrategy.ADAPTIVE_SMART
    fallback_strategy: FusionStrategy = FusionStrategy.WEIGHTED_MERGE
    
    # 性能配置
    enable_caching: bool = True
    max_candidates_for_rerank: int = 50
    
    # 动态权重阈值
    image_hint_boost: float = 0.3  # 检测到图像提示时的权重提升
    technical_text_boost: float = 0.2  # 检测到技术文本时的权重提升


# ImageTitleExtractor 已移至 image_title_matcher.py 模块


class T2TFusionStrategy:
    """T2T融合策略 - 实现复杂的文本到文本检索"""
    
    def __init__(self, 
                 text_retriever: 'T2TRetriever',
                 image_retriever: 'T2IRetriever',
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


@dataclass 
class FusionResult:
    """融合结果数据类"""
    results: List[RetrievalResult]
    strategy_used: str
    weights_applied: Dict[str, float]
    total_candidates: int
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultFusionManager:
    """增强的结果融合管理器 - 管理不同检索方式的结果融合"""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
        # 初始化增强组件
        self._init_enhancement_components()
        
        # 性能统计
        self.stats = {
            'total_fusions': 0,
            'strategy_usage': defaultdict(int),
            'avg_response_time': 0.0
        }
    
    def _init_enhancement_components(self):
        """初始化增强组件"""
        if self.config.enable_adaptive_weights:
            self.weight_calculator = get_weight_calculator()
        
        if self.config.enable_domain_partition:
            self.partition_manager = get_partition_manager()
            
        if self.config.enable_caption_enhancement:
            self.caption_enhancer = create_caption_enhancer()
            
        if self.config.enable_multimodal_rerank:
            self.reranker = create_multimodal_reranker()
    
    def fuse_multimodal_results(self, 
                               query: str,
                               text_results: List[RetrievalResult],
                               image_results: List[RetrievalResult],
                               strategy: Optional[FusionStrategy] = None) -> FusionResult:
        """
        增强的多模态检索结果融合
        
        Args:
            query: 查询文本（用于自适应权重计算）
            text_results: 文本检索结果
            image_results: 图像检索结果
            strategy: 融合策略，如果为None则自动选择
            
        Returns:
            融合结果对象
        """
        start_time = time.time()
        
        # 自动选择融合策略
        if strategy is None:
            strategy = self._select_fusion_strategy(query, text_results, image_results)
        
        # 获取动态权重
        dynamic_weights = self._get_dynamic_weights(query)
        
        # 执行融合
        if strategy == FusionStrategy.ADAPTIVE_SMART:
            fused_results = self._adaptive_smart_fusion(query, text_results, image_results, dynamic_weights)
        elif strategy == FusionStrategy.WEIGHTED_MERGE:
            fused_results = self._weighted_fusion(text_results, image_results, dynamic_weights)
        elif strategy == FusionStrategy.INTERLEAVED:
            fused_results = self._interleaved_fusion(text_results, image_results, dynamic_weights)
        elif strategy == FusionStrategy.SCORE_BASED:
            fused_results = self._score_based_fusion(text_results, image_results, dynamic_weights)
        else:
            # 简单拼接作为兜底
            fused_results = self._simple_concat_fusion(text_results, image_results)
            dynamic_weights = {"text_weight": 0.5, "image_weight": 0.5}
        
        # 应用多模态重排序
        if (self.config.enable_multimodal_rerank and hasattr(self, 'reranker') 
            and fused_results and len(fused_results) > 1):
            
            candidates = fused_results[:self.config.max_candidates_for_rerank]
            fused_results = self.reranker.rerank_results(query, candidates)
        
        # 最终去重和截取
        if self.config.enable_deduplication:
            fused_results = self._deduplicate_results(fused_results)
        
        final_results = fused_results[:self.config.top_k]
        
        # 更新统计
        response_time = time.time() - start_time
        self._update_stats(strategy.value, response_time)
        
        return FusionResult(
            results=final_results,
            strategy_used=strategy.value,
            weights_applied=dynamic_weights,
            total_candidates=len(text_results) + len(image_results),
            response_time=response_time,
            metadata={
                'text_count': len(text_results),
                'image_count': len(image_results),
                'rerank_applied': self.config.enable_multimodal_rerank and hasattr(self, 'reranker'),
                'dedup_applied': self.config.enable_deduplication
            }
        )
    
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
    
    def _weighted_fusion(self, text_results: List[RetrievalResult], 
                        image_results: List[RetrievalResult], 
                        dynamic_weights: Optional[Dict[str, float]] = None) -> List[RetrievalResult]:
        """增强的加权融合策略"""
        # 使用动态权重或默认权重
        if dynamic_weights:
            text_weight = dynamic_weights.get('text_weight', self.config.text_weight)
            image_weight = dynamic_weights.get('image_weight', self.config.image_weight)
        else:
            text_weight = self.config.text_weight
            image_weight = self.config.image_weight
        
        # 应用权重
        weighted_text_results = []
        for result in text_results:
            new_result = result.__class__(
                doc_id=result.doc_id,
                content=result.content,
                metadata=result.metadata.copy(),
                score=result.score * text_weight,
                retrieval_type=result.retrieval_type,
                resource_collection=result.resource_collection
            )
            new_result.metadata['weight_applied'] = text_weight
            new_result.metadata['fusion_source'] = 'text'
            new_result.metadata['fusion_strategy'] = 'weighted_enhanced'
            weighted_text_results.append(new_result)
        
        weighted_image_results = []
        for result in image_results:
            new_result = result.__class__(
                doc_id=result.doc_id,
                content=result.content,
                metadata=result.metadata.copy(),
                score=result.score * image_weight,
                retrieval_type=result.retrieval_type,
                resource_collection=result.resource_collection
            )
            new_result.metadata['weight_applied'] = image_weight
            new_result.metadata['fusion_source'] = 'image'
            new_result.metadata['fusion_strategy'] = 'weighted_enhanced'
            weighted_image_results.append(new_result)
        
        # 合并并排序
        all_results = weighted_text_results + weighted_image_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results  # 不在这里限制top_k，由调用方处理
    
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
    
    # === 新增增强方法 ===
    
    def _get_dynamic_weights(self, query: str) -> Dict[str, float]:
        """获取查询自适应的动态权重"""
        if not self.config.enable_adaptive_weights or not hasattr(self, 'weight_calculator'):
            return {"text_weight": self.config.text_weight, "image_weight": self.config.image_weight}
        
        # 使用查询自适应权重计算器
        fusion_weights = get_fusion_weights(query)
        
        # 应用配置的权重提升
        if 'image_hint' in fusion_weights and fusion_weights['image_hint'] > 0.5:
            fusion_weights['image_weight'] += self.config.image_hint_boost
            fusion_weights['text_weight'] -= self.config.image_hint_boost
        
        # 确保权重在合理范围内
        fusion_weights['text_weight'] = max(0.1, min(0.9, fusion_weights['text_weight']))
        fusion_weights['image_weight'] = max(0.1, min(0.9, fusion_weights['image_weight']))
        
        return fusion_weights
    
    def _select_fusion_strategy(self, query: str, text_results: List[RetrievalResult], 
                               image_results: List[RetrievalResult]) -> FusionStrategy:
        """智能选择融合策略"""
        if not self.config.enable_smart_fusion:
            return self.config.default_fusion_strategy
        
        # 分析查询特征
        if hasattr(self, 'weight_calculator'):
            features = self.weight_calculator.analyze_query(query)
            
            # 基于查询特征选择策略
            if features.is_caption_like and len(image_results) > len(text_results):
                return FusionStrategy.SCORE_BASED  # Caption查询优先使用分数融合
            elif features.is_technical and len(text_results) > len(image_results):
                return FusionStrategy.WEIGHTED_MERGE  # 技术查询优先使用加权融合
            elif len(text_results) > 0 and len(image_results) > 0:
                return FusionStrategy.ADAPTIVE_SMART  # 平衡查询使用自适应融合
        
        # 基于结果数量选择策略
        if len(text_results) == 0:
            return FusionStrategy.SIMPLE_CONCAT
        elif len(image_results) == 0:
            return FusionStrategy.SIMPLE_CONCAT
        else:
            return self.config.default_fusion_strategy
    
    def _adaptive_smart_fusion(self, query: str, text_results: List[RetrievalResult], 
                              image_results: List[RetrievalResult], 
                              dynamic_weights: Dict[str, float]) -> List[RetrievalResult]:
        """自适应智能融合策略"""
        # 分析结果分布
        text_avg_score = sum(r.score for r in text_results) / len(text_results) if text_results else 0
        image_avg_score = sum(r.score for r in image_results) / len(image_results) if image_results else 0
        
        # 如果一种模态的平均分数明显高于另一种，调整权重
        score_diff = abs(text_avg_score - image_avg_score)
        if score_diff > 0.3:  # 阈值可配置
            if text_avg_score > image_avg_score:
                dynamic_weights['text_weight'] += 0.1
                dynamic_weights['image_weight'] -= 0.1
            else:
                dynamic_weights['image_weight'] += 0.1
                dynamic_weights['text_weight'] -= 0.1
        
        # 如果结果数量差异很大，使用交错融合
        count_ratio = len(text_results) / max(len(image_results), 1)
        if count_ratio > 3 or count_ratio < 0.33:
            return self._interleaved_fusion(text_results, image_results, dynamic_weights)
        
        # 否则使用加权融合
        return self._weighted_fusion(text_results, image_results, dynamic_weights)
    
    def _simple_concat_fusion(self, text_results: List[RetrievalResult], 
                             image_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """简单拼接融合"""
        all_results = text_results + image_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # 添加融合元数据
        for result in all_results:
            result.metadata['fusion_strategy'] = 'simple_concat'
        
        return all_results
    
    def _interleaved_fusion(self, text_results: List[RetrievalResult], 
                           image_results: List[RetrievalResult],
                           dynamic_weights: Optional[Dict[str, float]] = None) -> List[RetrievalResult]:
        """增强的交错融合策略"""
        if dynamic_weights:
            # 如果提供了动态权重，先应用权重
            weighted_text = self._apply_weights_to_results(text_results, dynamic_weights.get('text_weight', 0.6))
            weighted_image = self._apply_weights_to_results(image_results, dynamic_weights.get('image_weight', 0.4))
        else:
            weighted_text = text_results
            weighted_image = image_results
        
        # 交错合并
        fused_results = []
        i = j = 0
        while i < len(weighted_text) and j < len(weighted_image):
            if weighted_text[i].score >= weighted_image[j].score:
                weighted_text[i].metadata['fusion_strategy'] = 'interleaved_text'
                fused_results.append(weighted_text[i])
                i += 1
            else:
                weighted_image[j].metadata['fusion_strategy'] = 'interleaved_image'
                fused_results.append(weighted_image[j])
                j += 1
        
        # 添加剩余结果
        while i < len(weighted_text):
            weighted_text[i].metadata['fusion_strategy'] = 'interleaved_text_remaining'
            fused_results.append(weighted_text[i])
            i += 1
        
        while j < len(weighted_image):
            weighted_image[j].metadata['fusion_strategy'] = 'interleaved_image_remaining'
            fused_results.append(weighted_image[j])
            j += 1
        
        return fused_results
    
    def _apply_weights_to_results(self, results: List[RetrievalResult], weight: float) -> List[RetrievalResult]:
        """为结果列表应用权重"""
        weighted_results = []
        for result in results:
            new_result = result.__class__(
                doc_id=result.doc_id,
                content=result.content,
                metadata=result.metadata.copy(),
                score=result.score * weight,
                retrieval_type=result.retrieval_type,
                resource_collection=result.resource_collection
            )
            new_result.metadata['weight_applied'] = weight
            weighted_results.append(new_result)
        return weighted_results
    
    def _update_stats(self, strategy: str, response_time: float):
        """更新性能统计"""
        self.stats['total_fusions'] += 1
        self.stats['strategy_usage'][strategy] += 1
        
        total = self.stats['total_fusions']
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = (current_avg * (total - 1) + response_time) / total


def create_enhanced_fusion_manager(config: Optional[FusionConfig] = None) -> ResultFusionManager:
    """创建增强的融合管理器"""
    if config is None:
        config = FusionConfig()
    return ResultFusionManager(config)


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
        
        # 延迟导入以避免循环依赖
        self._retriever_classes = None
        self._context_params = {
            'db_name': db_name or "default",
            'text_collection': "text_collection",  # 从配置获取
            'image_collection': "image_collection",  # 从配置获取
            'top_k': self.fusion_config.top_k,
            'milvus_uri': milvus_uri,
            'milvus_token': milvus_token
        }
        
        # 检索器将在需要时初始化
        self.text_retriever = None
        self.image_retriever = None
        self.i2t_retriever = None
        self.i2i_retriever = None
        
        # 策略管理器也将延迟初始化
        self.t2t_strategy = None
        self.fusion_manager = ResultFusionManager(self.fusion_config)
        
        self.logger.info("融合检索管理器初始化完成")
    
    def _ensure_retrievers_initialized(self):
        """确保检索器已初始化"""
        if self.text_retriever is None:
            # 延迟导入以避免循环依赖
            from ultrasoundrag.core.retrieval.modular_retrievers import (
                T2TRetriever, T2IRetriever, I2TRetriever, I2IRetriever, RetrievalContext
            )
            
            # 创建检索上下文
            context = RetrievalContext(**self._context_params)
            
            # 初始化检索器
            try:
                self.text_retriever = T2TRetriever(context)
                self.image_retriever = T2IRetriever(context)
                self.i2t_retriever = I2TRetriever(context)
                self.i2i_retriever = I2IRetriever(context)
                
                # 初始化策略管理器
                self.t2t_strategy = T2TFusionStrategy(
                    self.text_retriever, 
                    self.image_retriever, 
                    self.fusion_config
                )
                
                self.logger.info("基础检索器延迟初始化成功")
            except Exception as e:
                self.logger.error(f"基础检索器初始化失败: {e}")
                raise
    
    def t2t_search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        复杂的T2T检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            T2T检索结果
        """
        self._ensure_retrievers_initialized()
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
        self._ensure_retrievers_initialized()
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
        self._ensure_retrievers_initialized()
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
