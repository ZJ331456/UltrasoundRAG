"""重排序模块 - 新架构版本
提供规则重排序和模型重排序功能，支持Milvus检索结果

主要功能：
1. 基于规则的重排序（词汇重叠、长度惩罚、医学术语奖励等）
2. 基于本地重排序模型（如CrossEncoder）的重排序
3. 智能重排序策略选择
"""

import time
import re
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from UltrasoundRAG.utils.logger import setup_logger
from UltrasoundRAG.retrival.data_structures import RetrievalResult
from UltrasoundRAG.utils.cache_utils import LRUCache

from sentence_transformers import CrossEncoder
import torch
CROSS_ENCODER_AVAILABLE = True


# 尝试导入jieba用于中文分词
try:
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
            
            # 更新结果
            result.score = final_score
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
            self.cache.put(final_results, cache_key)
        
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
            device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
            self.model = CrossEncoder(
                self.model_path,
                device=device,
                trust_remote_code=True,
                automodel_args={"torch_dtype": "auto"}
            )
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