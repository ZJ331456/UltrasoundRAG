"""增强重排序模块 - 集成多模态功能
提供规则重排序、模型重排序和多模态重排序功能，支持Milvus检索结果

主要功能：
1. 基于规则的重排序（词汇重叠、长度惩罚、医学术语奖励等）
2. 基于本地重排序模型（如CrossEncoder）的重排序
3. 智能重排序策略选择
4. 多模态相关性计算和跨模态语义对齐
5. 混合重排序策略
"""

import time
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ultrasoundrag.utils.logger import setup_logger
from ultrasoundrag.core.retrieval.data_structures import RetrievalResult
from ultrasoundrag.utils.cache_utils import LRUCache

from sentence_transformers import CrossEncoder
import torch
CROSS_ENCODER_AVAILABLE = True

# 多模态模型导入
try:
    from ultrasoundrag.model.model_manager import get_fetal_clip_model
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False


# 尝试导入jieba用于中文分词
try:
    # 统一使用项目根目录设置的缓存目录
    import os as _os
    _cache_dir = _os.environ.get('JIEBA_CACHE_DIR')
    if _cache_dir:
        _os.makedirs(_cache_dir, exist_ok=True)
    import jieba
    jieba.setLogLevel(20)  # 设置为WARNING级别，减少日志输出
except ImportError:
    jieba = None


@dataclass
class RerankConfig:
    """重排序配置"""
    algorithm: str = "rule_based"  # rule_based, model_based
    max_candidates: int = 50  # 最大候选结果数
    final_count: int = 20  # 最终返回结果数
    enable_cache: bool = True  # 是否启用缓存
    
    # 规则重排序权重
    overlap_weight: float = 0.3
    length_weight: float = 0.1
    position_weight: float = 0.1
    medical_term_weight: float = 0.2


@dataclass
class RerankResult:
    """重排序结果"""
    results: List[RetrievalResult]
    algorithm_used: str
    processing_time: float
    cache_hits: int
    quality_improvement: float = 0.0


class BaseReranker:
    """重排序器基类"""
    
    def __init__(self, config: RerankConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
        # 缓存系统
        if config.enable_cache:
            self.cache = LRUCache(max_size=1000, max_memory_mb=128, ttl_seconds=1800)
        else:
            self.cache = None
        
        # 性能统计
        self.stats = {
            'total_reranks': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0
        }
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """重排序结果"""
        raise NotImplementedError
    
    def _get_cache_key(self, query: str, results: List[RetrievalResult]) -> str:
        """生成缓存键"""
        result_ids = [r.doc_id for r in results[:10]]  # 只使用前10个结果ID
        return f"{self.__class__.__name__}_{hash(query)}_{hash(tuple(result_ids))}"

    def _update_stats(self, processing_time: float) -> None:
        """更新统计信息（在基类提供，子类直接复用）"""
        self.stats['total_reranks'] += 1
        total_reranks = self.stats['total_reranks']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (
            current_avg * (total_reranks - 1) + processing_time
        ) / total_reranks


class RuleBasedReranker(BaseReranker):
    """规则重排序器"""
    
    def __init__(self, config: RerankConfig):
        super().__init__(config)
        
        # 医学术语词典
        self.medical_terms = {
            '超声', '检查', '诊断', '病变', '肿瘤', '囊肿', '结节', '炎症',
            '血管', '器官', '组织', '细胞', '病理', '症状', '治疗', '手术',
            '心脏', '肝脏', '肾脏', '脾脏', '胰腺', '胆囊', '子宫', '卵巢'
        }
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """基于规则的重排序"""
        if not results:
            return results
        
        start_time = time.time()
        
        # 检查缓存
        if self.cache:
            cache_key = self._get_cache_key(query, results)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return cached_result
        
        # 计算重排序分数
        reranked_results = []
        for i, result in enumerate(results):
            # 原始分数
            base_score = result.score
            
            # 词汇重叠分数
            overlap_score = self._calculate_overlap_score(query, result.content)
            
            # 长度惩罚
            length_score = self._calculate_length_penalty(result.content)
            
            # 位置奖励
            position_score = self._calculate_position_bonus(i, len(results))
            
            # 医学术语奖励
            medical_score = self._calculate_medical_term_bonus(result.content)
            
            # 综合分数
            final_score = (
                base_score + 
                overlap_score * self.config.overlap_weight +
                length_score * self.config.length_weight +
                position_score * self.config.position_weight +
                medical_score * self.config.medical_term_weight
            )
            
            # 更新结果 - 确保是Python原生float类型
            result.score = float(final_score)
            result.metadata['rerank_components'] = {
                'base': base_score,
                'overlap': overlap_score,
                'length': length_score,
                'position': position_score,
                'medical': medical_score
            }
            reranked_results.append(result)
        
        # 排序
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        final_results = reranked_results[:self.config.final_count]
        
        # 缓存结果
        if self.cache:
            self.cache.put(final_results, key=cache_key)
        
        # 更新统计
        processing_time = time.time() - start_time
        self._update_stats(processing_time)
        
        return final_results
    
    def _tokenize(self, text: str) -> List[str]:
        """智能分词，支持中文和英文"""
        if not text:
            return []
        
        text = text.lower()
        
        if jieba:
            # 使用jieba进行中文分词
            tokens = list(jieba.cut_for_search(text))
            # 过滤掉停用词和短词
            stop_words = {'的', '了', '是', '在', '有', '和', '与', '或', '但', '而', '为', '从', '到', '把', '被'}
            tokens = [token for token in tokens if len(token.strip()) > 1 and token.strip() not in stop_words]
        else:
            # 简单的英文分词
            tokens = re.findall(r'\w+', text)
            tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def _calculate_overlap_score(self, query: str, content: str) -> float:
        """计算词汇重叠分数 - 改进版"""
        query_words = set(self._tokenize(query))
        content_words = set(self._tokenize(content))
        
        if not query_words:
            return 0.0
        
        # 计算交集
        overlap = len(query_words.intersection(content_words))
        
        # 计算基础重叠分数
        basic_score = overlap / len(query_words)
        
        # 考虑部分匹配（子字符串匹配）
        partial_matches = 0
        for query_word in query_words:
            for content_word in content_words:
                if query_word in content_word or content_word in query_word:
                    partial_matches += 1
                    break
        
        partial_score = partial_matches / len(query_words) * 0.5  # 部分匹配权重较低
        
        return min(basic_score + partial_score, 1.0)
    
    def _calculate_length_penalty(self, content: str, optimal_length: int = 200) -> float:
        """计算长度惩罚"""
        content_length = len(content)
        if content_length == 0:
            return 0.0
        
        penalty = abs(content_length - optimal_length) / optimal_length
        return max(0, 1 - penalty)
    
    def _calculate_position_bonus(self, position: int, total_count: int) -> float:
        """计算位置奖励"""
        if total_count <= 1:
            return 1.0
        return 1.0 - (position / total_count) * 0.2
    
    def _calculate_medical_term_bonus(self, content: str) -> float:
        """计算医学术语奖励"""
        content_lower = content.lower()
        medical_count = sum(1 for term in self.medical_terms if term in content_lower)
        content_words = len(content.split())
        
        if content_words == 0:
            return 0.0
        
        return min(medical_count / content_words, 0.5)  # 最多50%的奖励
    
    def _update_stats(self, processing_time: float):
        """更新统计信息"""
        self.stats['total_reranks'] += 1
        
        # 更新平均处理时间
        total_reranks = self.stats['total_reranks']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (current_avg * (total_reranks - 1) + processing_time) / total_reranks


class ModelBasedReranker(BaseReranker):
    """基于CrossEncoder模型的重排序器"""
    
    def __init__(self, config: RerankConfig, model_path: Optional[str] = None):
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        self.fallback_reranker = None
        
        if CROSS_ENCODER_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                self._load_model()
            except Exception as e:
                self.logger.error(f"加载重排序模型失败: {e}")
                self._init_fallback()
        else:
            self.logger.warning("CrossEncoder不可用或模型路径无效，使用规则重排序替代")
            self._init_fallback()
    
    def _load_model(self):
        """加载CrossEncoder模型"""
        try:
            from ultrasoundrag.model.singleton_models import global_models
            
            def create_cross_encoder():
                device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
                return CrossEncoder(
                    self.model_path,
                    device=device,
                    trust_remote_code=True,
                    model_kwargs={"torch_dtype": "auto"}
                )
            
            self.model = global_models.get_or_create_model("cross_encoder", create_cross_encoder)
            self.logger.info(f"成功加载重排序模型: {self.model_path}")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            self._init_fallback()
    
    def _init_fallback(self):
        """初始化回退方案"""
        self.fallback_reranker = RuleBasedReranker(self.config)
        self.logger.info("已初始化规则重排序作为回退方案")
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """基于模型的重排序"""
        if not results:
            return results
        
        # 如果模型不可用，使用回退方案
        if self.model is None:
            if self.fallback_reranker:
                return self.fallback_reranker.rerank(query, results)
            else:
                return results
        
        start_time = time.time()
        
        try:
            # 准备查询-文档对
            query_doc_pairs = [(query, result.content) for result in results]
            
            # 使用模型预测相关性分数
            if len(query_doc_pairs) == 1:
                scores = [self.model.predict([query_doc_pairs[0]])[0]]
            else:
                scores = self.model.predict(query_doc_pairs, batch_size=min(8, len(query_doc_pairs)))
            
            # 更新结果分数
            for result, score in zip(results, scores):
                result.metadata['original_score'] = result.score
                result.metadata['model_rerank_score'] = float(score)
                result.score = float(score)
            
            # 重新排序
            results.sort(key=lambda x: x.score, reverse=True)
            
            # 更新统计
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            self.logger.info(f"模型重排序完成，处理了 {len(results)} 个结果，耗时 {processing_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"模型重排序失败: {e}")
            # 回退到规则重排序
            if self.fallback_reranker:
                return self.fallback_reranker.rerank(query, results)
            else:
                return results


class SimpleRerankManager:
    """简化的重排序管理器 - 支持多种重排序策略"""
    
    def __init__(self, config: RerankConfig = None, model_path: Optional[str] = None):
        self.config = config or RerankConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        # 初始化重排序器
        if self.config.algorithm == "model_based":
            self.reranker = ModelBasedReranker(self.config, model_path)
        else:
            self.reranker = RuleBasedReranker(self.config)
        
        self.logger.info(f"重排序管理器初始化完成 - 算法: {self.config.algorithm}")
    
    def rerank_results(self, query: str, results: List[RetrievalResult]) -> RerankResult:
        """
        重排序结果
        
        Args:
            query: 查询文本
            results: 检索结果列表
            
        Returns:
            重排序结果对象
        """
        if not results:
            return RerankResult([], "none", 0.0, 0)
        
        start_time = time.time()
        
        # 执行重排序
        reranked_results = self.reranker.rerank(query, results.copy())
        
        processing_time = time.time() - start_time
        
        # 获取缓存命中数
        cache_hits = self.reranker.stats.get('cache_hits', 0)
        
        # 计算质量改进（简单估算）
        quality_improvement = self._calculate_quality_improvement(results, reranked_results)
        
        return RerankResult(
            results=reranked_results,
            algorithm_used=self.config.algorithm,
            processing_time=processing_time,
            cache_hits=cache_hits,
            quality_improvement=quality_improvement
        )
    
    def _calculate_quality_improvement(self, original: List[RetrievalResult], 
                                     reranked: List[RetrievalResult]) -> float:
        """计算重排序质量改进度"""
        if not original or not reranked:
            return 0.0
        
        try:
            # 简单计算：比较前3个结果的平均分数变化
            top_k = min(3, len(original), len(reranked))
            
            original_avg = sum(r.score for r in original[:top_k]) / top_k
            reranked_avg = sum(r.score for r in reranked[:top_k]) / top_k
            
            improvement = (reranked_avg - original_avg) / max(original_avg, 0.01)
            return max(0.0, improvement)  # 只返回正向改进
        except Exception:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.reranker.stats.copy()
        stats['algorithm'] = self.config.algorithm
        stats['config'] = {
            'max_candidates': self.config.max_candidates,
            'final_count': self.config.final_count,
            'enable_cache': self.config.enable_cache
        }
        return stats


# 便捷函数
def create_rerank_manager(algorithm: str = "rule_based", 
                         max_candidates: int = 50,
                         final_count: int = 20,
                         model_path: Optional[str] = None,
                         **kwargs) -> SimpleRerankManager:
    """创建重排序管理器的便捷函数
    
    Args:
        algorithm: 重排序算法 ("rule_based" 或 "model_based")
        max_candidates: 最大候选结果数
        final_count: 最终返回结果数
        model_path: 重排序模型路径（仅在algorithm="model_based"时使用）
        **kwargs: 其他配置参数
        
    Returns:
        重排序管理器实例
    """
    config = RerankConfig(
        algorithm=algorithm,
        max_candidates=max_candidates,
        final_count=final_count,
        **kwargs
    )
    return SimpleRerankManager(config, model_path)


# === 多模态重排序功能 ===

class RerankMode(Enum):
    """重排序模式"""
    TEXT_ONLY = "text_only"           # 仅文本重排序
    IMAGE_ONLY = "image_only"         # 仅图像重排序
    MULTIMODAL = "multimodal"         # 多模态重排序
    HYBRID = "hybrid"                 # 混合重排序


@dataclass
class MultimodalRerankConfig:
    """多模态重排序配置"""
    mode: RerankMode = RerankMode.MULTIMODAL
    top_k: int = 20
    
    # 权重配置
    semantic_weight: float = 0.7      # 语义相关性权重
    visual_weight: float = 0.2        # 视觉相关性权重
    textual_weight: float = 0.5       # 文本相关性权重
    cross_modal_weight: float = 0.3   # 跨模态相关性权重
    
    # 质量控制
    min_score_threshold: float = 0.1
    diversity_penalty: float = 0.05   # 多样性惩罚
    
    # 性能配置
    enable_caching: bool = True
    batch_size: int = 8
    max_candidates: int = 50
    
    # 融合旧的重排序配置
    enable_rule_based: bool = True
    enable_model_based: bool = True
    rule_weight: float = 0.4
    model_weight: float = 0.5


@dataclass
class MultimodalRerankResult:
    """多模态重排序结果"""
    results: List[RetrievalResult]
    strategy_used: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalReranker:
    """多模态重排序器 - 集成传统和多模态重排序"""
    
    def __init__(self, config: Optional[MultimodalRerankConfig] = None):
        self.config = config or MultimodalRerankConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        # 初始化传统重排序组件
        self._init_traditional_rerankers()
        
        # 初始化多模态组件
        self._init_multimodal_components()
        
        # 缓存
        if self.config.enable_caching:
            self.cache = LRUCache(max_size=1000, max_memory_mb=256, ttl_seconds=3600)
        else:
            self.cache = None
        
        # 性能统计
        self.stats = {
            'total_reranks': 0,
            'mode_usage': {mode.value: 0 for mode in RerankMode},
            'avg_processing_time': 0.0,
            'cache_hits': 0
        }
    
    def _init_traditional_rerankers(self):
        """初始化传统重排序器"""
        # 规则重排序器
        if self.config.enable_rule_based:
            rule_config = RerankConfig(
                algorithm="rule_based",
                max_candidates=self.config.max_candidates,
                final_count=self.config.top_k
            )
            self.rule_reranker = RuleBasedReranker(rule_config)
        
        # 模型重排序器
        if self.config.enable_model_based and CROSS_ENCODER_AVAILABLE:
            # 从全局配置获取模型路径
            from ultrasoundrag.config import config
            model_path = config.get('rerank', {}).get('model_path', '/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/models/jina-reranker-v2-base-multilingual')
            
            model_config = RerankConfig(
                algorithm="model_based",
                max_candidates=self.config.max_candidates,
                final_count=self.config.top_k
            )
            self.model_reranker = ModelBasedReranker(model_config, model_path=model_path)
    
    def _init_multimodal_components(self):
        """初始化多模态组件"""
        if MULTIMODAL_AVAILABLE:
            try:
                from ultrasoundrag.model.singleton_models import get_shared_fetal_clip
                self.clip_model = get_shared_fetal_clip()
                self.multimodal_enabled = True
                self.logger.info("多模态CLIP模型初始化成功")
            except Exception as e:
                self.logger.warning(f"多模态CLIP模型初始化失败: {e}")
                self.multimodal_enabled = False
        else:
            self.multimodal_enabled = False
            self.logger.warning("多模态功能不可用")
    
    def rerank_results(self, query: str, results: List[RetrievalResult], 
                      mode: Optional[RerankMode] = None) -> List[RetrievalResult]:
        """
        多模态重排序主接口
        
        Args:
            query: 查询文本
            results: 检索结果列表
            mode: 重排序模式，如果为None则自动选择
            
        Returns:
            重排序后的结果列表
        """
        start_time = time.time()
        
        if not results:
            return results
        
        # 缓存检查
        if self.cache:
            cache_key = f"{hash(query)}_{hash(tuple(r.doc_id for r in results))}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                return cached_result
        
        # 自动选择模式
        if mode is None:
            mode = self._auto_select_mode(query, results)
        
        # 执行重排序
        if mode == RerankMode.MULTIMODAL and self.multimodal_enabled:
            reranked_results = self._multimodal_rerank(query, results)
        elif mode == RerankMode.HYBRID:
            reranked_results = self._hybrid_rerank(query, results)
        elif mode == RerankMode.TEXT_ONLY:
            reranked_results = self._text_only_rerank(query, results)
        else:
            # 默认使用混合模式
            reranked_results = self._hybrid_rerank(query, results)
        
        # 最终结果限制
        final_results = reranked_results[:self.config.top_k]
        
        # 更新统计
        processing_time = time.time() - start_time
        self._update_stats(mode, processing_time)
        
        # 缓存结果
        if self.cache:
            self.cache.put(final_results, key=cache_key)
        
        return final_results
    
    def _auto_select_mode(self, query: str, results: List[RetrievalResult]) -> RerankMode:
        """自动选择重排序模式"""
        # 分析结果类型
        has_images = any('image_path' in r.metadata or 'image' in r.retrieval_type for r in results)
        has_text = any('text' in r.retrieval_type or r.content for r in results)
        
        # 分析查询类型
        query_has_image_hints = any(hint in query.lower() for hint in ['图', 'figure', '图像', '影像'])
        
        if has_images and has_text and self.multimodal_enabled:
            if query_has_image_hints:
                return RerankMode.MULTIMODAL
            else:
                return RerankMode.HYBRID
        elif has_images and not has_text:
            return RerankMode.IMAGE_ONLY
        else:
            return RerankMode.TEXT_ONLY
    
    def _multimodal_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """多模态重排序"""
        if not self.multimodal_enabled:
            return self._hybrid_rerank(query, results)
        
        try:
            # 计算查询向量
            query_tokens = self.clip_model.tokenize_text([query])
            query_text_vector = self.clip_model.encode_text(query_tokens).cpu().numpy()[0]
            
            # 为每个结果计算多模态相关性分数
            enhanced_results = []
            for result in results:
                multimodal_score = self._calculate_multimodal_score(
                    query, query_text_vector, result
                )
                
                # 创建新的结果对象，包含多模态分数
                new_result = RetrievalResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    metadata=result.metadata.copy(),
                    score=multimodal_score,
                    retrieval_type=result.retrieval_type,
                    resource_collection=result.resource_collection
                )
                new_result.metadata['original_score'] = result.score
                new_result.metadata['multimodal_score'] = multimodal_score
                new_result.metadata['rerank_strategy'] = 'multimodal'
                enhanced_results.append(new_result)
            
            # 按多模态分数排序
            enhanced_results.sort(key=lambda x: x.score, reverse=True)
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"多模态重排序失败: {e}")
            return self._hybrid_rerank(query, results)
    
    def _calculate_multimodal_score(self, query: str, query_vector: np.ndarray, 
                                   result: RetrievalResult) -> float:
        """计算多模态相关性分数"""
        # 基础分数
        base_score = result.score
        
        # 文本相关性分数
        text_score = self._calculate_text_similarity(query, result.content)
        
        # 图像相关性分数（如果有图像）
        image_score = 0.0
        if 'image_path' in result.metadata and self.multimodal_enabled:
            image_score = self._calculate_image_similarity(query_vector, result)
        
        # 跨模态分数
        cross_modal_score = self._calculate_cross_modal_score(query, result)
        
        # 加权组合
        final_score = (
            base_score * 0.2 +
            text_score * self.config.textual_weight +
            image_score * self.config.visual_weight +
            cross_modal_score * self.config.cross_modal_weight
        )
        
        return max(final_score, self.config.min_score_threshold)
    
    def _calculate_text_similarity(self, query: str, content: str) -> float:
        """计算文本相似性"""
        if not content:
            return 0.0
        
        # 简单的词汇重叠相似性
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_image_similarity(self, query_vector: np.ndarray, result: RetrievalResult) -> float:
        """计算图像相似性"""
        # 这里可以扩展为实际的图像向量计算
        # 暂时使用caption相似性作为代理
        caption = result.metadata.get('caption', '')
        if not caption:
            return 0.0
        
        try:
            caption_tokens = self.clip_model.tokenize_text([caption])
            caption_vector = self.clip_model.encode_text(caption_tokens).cpu().numpy()[0]
            
            # 计算余弦相似性
            similarity = np.dot(query_vector, caption_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(caption_vector)
            )
            # 确保返回Python原生float类型
            return max(0.0, float(similarity))
        except:
            return 0.0
    
    def _calculate_cross_modal_score(self, query: str, result: RetrievalResult) -> float:
        """计算跨模态分数"""
        # 检查文本中是否有图像引用
        query_has_image_ref = any(ref in query.lower() for ref in ['图', 'figure', '图像', '如图'])
        result_has_image = 'relative_path' in result.metadata
        
        if query_has_image_ref and result_has_image:
            return 0.8  # 高跨模态相关性
        elif query_has_image_ref and not result_has_image:
            return 0.2  # 低跨模态相关性
        else:
            return 0.5  # 中等相关性
    
    def _hybrid_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """混合重排序策略"""
        enhanced_results = []
        
        # 应用传统重排序
        rule_scores = {}
        model_scores = {}
        
        if hasattr(self, 'rule_reranker') and self.config.enable_rule_based:
            rule_result = self.rule_reranker.rerank(query, results)
            for i, result in enumerate(rule_result):
                rule_scores[result.doc_id] = result.score
        
        if hasattr(self, 'model_reranker') and self.config.enable_model_based:
            model_result = self.model_reranker.rerank(query, results)
            for i, result in enumerate(model_result):
                model_scores[result.doc_id] = result.score
        
        # 融合分数
        for result in results:
            doc_id = result.doc_id
            
            # 组合分数
            combined_score = result.score * 0.3  # 原始分数权重
            
            if doc_id in rule_scores:
                combined_score += rule_scores[doc_id] * self.config.rule_weight
            
            if doc_id in model_scores:
                combined_score += model_scores[doc_id] * self.config.model_weight
            
            # 创建新结果
            new_result = RetrievalResult(
                doc_id=result.doc_id,
                content=result.content,
                metadata=result.metadata.copy(),
                score=combined_score,
                retrieval_type=result.retrieval_type,
                resource_collection=result.resource_collection
            )
            new_result.metadata['original_score'] = result.score
            new_result.metadata['rule_score'] = rule_scores.get(doc_id, 0.0)
            new_result.metadata['model_score'] = model_scores.get(doc_id, 0.0)
            new_result.metadata['rerank_strategy'] = 'hybrid'
            enhanced_results.append(new_result)
        
        # 排序
        enhanced_results.sort(key=lambda x: x.score, reverse=True)
        return enhanced_results
    
    def _text_only_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """仅文本重排序"""
        if hasattr(self, 'model_reranker') and self.config.enable_model_based:
            return self.model_reranker.rerank(query, results)
        elif hasattr(self, 'rule_reranker') and self.config.enable_rule_based:
            return self.rule_reranker.rerank(query, results)
        else:
            return results
    
    def _update_stats(self, mode: RerankMode, processing_time: float):
        """更新性能统计"""
        self.stats['total_reranks'] += 1
        self.stats['mode_usage'][mode.value] += 1
        
        total = self.stats['total_reranks']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total


def create_multimodal_reranker(config: Optional[MultimodalRerankConfig] = None) -> MultimodalReranker:
    """创建多模态重排序器"""
    return MultimodalReranker(config)