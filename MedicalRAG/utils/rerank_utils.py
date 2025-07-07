"""
重排序工具模块
提供两种核心重排序策略：基于规则的重排序和基于本地重排序模型的重排序

主要功能：
1. 基于规则的重排序（词汇重叠、长度惩罚等）
2. 基于本地重排序模型（如Qwen3-Reranker-0.6B）的重排序
"""

import os
import sys
import torch
from typing import List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import re

from MedicalRAG.utils.logger import setup_logger

# 避免循环导入，使用TYPE_CHECKING
if TYPE_CHECKING:
    from MedicalRAG.utils.retrival_utils import RetrievalResult

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("警告: sentence-transformers 未安装，重排序模型功能不可用")

try:
    import jieba
except ImportError:
    jieba = None



class RuleBasedReranker:
    """基于规则的重排序器"""
    
    def __init__(
        self,
        overlap_weight: float = 0.3,
        length_weight: float = 0.1,
        position_weight: float = 0.1,
    ):
        """
        初始化规则重排序器
        
        Args:
            overlap_weight: 词汇重叠权重
            length_weight: 长度惩罚权重
            position_weight: 位置权重
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.overlap_weight = overlap_weight
        self.length_weight = length_weight
        self.position_weight = position_weight
    
    def tokenize(self, text: str) -> List[str]:
        """文本分词"""
        if jieba:
            return list(jieba.cut(text.lower()))
        else:
            return re.findall(r'\w+', text.lower())
    
    def calculate_overlap_score(self, query: str, content: str) -> float:
        """计算词汇重叠分数"""
        query_words = set(self.tokenize(query))
        content_words = set(self.tokenize(content))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    def calculate_length_penalty(self, content: str, optimal_length: int = 200) -> float:
        """计算长度惩罚（避免过长或过短的文档）"""
        content_length = len(content)
        penalty = abs(content_length - optimal_length) / optimal_length
        return max(0, 1 - penalty)
    
    def calculate_position_bonus(self, position: int, total_count: int) -> float:
        """计算位置奖励（早期检索到的文档可能更重要）"""
        if total_count <= 1:
            return 1.0
        return 1.0 - (position / total_count) * 0.2  # 最多20%的位置惩罚
    
    def rerank(self, query: str, results: List['RetrievalResult']) -> List['RetrievalResult']:
        """基于规则的重排序"""
        if not results:
            return results
        
        for i, result in enumerate(results):
            # 计算各种分数
            overlap_score = self.calculate_overlap_score(query, result.content)
            length_score = self.calculate_length_penalty(result.content)
            position_score = self.calculate_position_bonus(i, len(results))
            
            # 组合分数
            new_score = (
                result.score * (1 - self.overlap_weight - self.length_weight - self.position_weight) +
                overlap_score * self.overlap_weight +
                length_score * self.length_weight +
                position_score * self.position_weight
            )
            
            result.score = new_score
            
            # 记录各个分数到元数据
            result.metadata.update({
                'rerank_overlap_score': overlap_score,
                'rerank_length_score': length_score,
                'rerank_position_score': position_score,
                'rerank_final_score': new_score
            })
        
        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results

class ModelBasedReranker:
    """基于本地重排序模型的重排序器"""
    
    def __init__(
        self,
        model_path: str = r"C:\Users\zero\Desktop\ht-rag\models\Qwen3-Reranker-0.6B",
        device: str = "auto",
        batch_size: int = 1,  # 默认设置为 1 避免 padding 问题
    ):
        """
        初始化模型重排序器
        
        Args:
            model_path: 本地模型路径
            device: 计算设备
            batch_size: 批处理大小
        """
        self.logger = setup_logger(self.__class__.__name__)
        
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("需要安装 sentence-transformers 才能使用重排序模型")
        
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """加载本地重排序模型"""
        try:
            self.logger.info(f"加载重排序模型: {self.model_path}")
            
            # 检查本地模型路径
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 设置离线模式，避免网络请求
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            # 加载模型并配置 tokenizer
            self.model = CrossEncoder(
                self.model_path, 
                device=self.device,
                trust_remote_code=True
            )
            
            # 修复 padding token 问题
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                if self.model.tokenizer.pad_token is None:
                    # 使用 eos_token 作为 pad_token
                    if self.model.tokenizer.eos_token is not None:
                        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
                        self.logger.info(f"设置 pad_token 为 eos_token: {self.model.tokenizer.eos_token}")
                    else:
                        # 如果没有 eos_token，使用第一个特殊 token
                        special_tokens = self.model.tokenizer.special_tokens_map
                        for token_type, token in special_tokens.items():
                            if token:
                                self.model.tokenizer.pad_token = token
                                self.logger.info(f"设置 pad_token 为 {token_type}: {token}")
                                break
                        else:
                            # 如果没有特殊 token，添加一个新的
                            self.model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                            self.logger.info("添加新的 pad_token: [PAD]")
            
            self.logger.info(f"重排序模型加载成功，设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"加载重排序模型失败: {e}")
            # 清除环境变量
            if 'TRANSFORMERS_OFFLINE' in os.environ:
                del os.environ['TRANSFORMERS_OFFLINE']
            if 'HF_HUB_OFFLINE' in os.environ:
                del os.environ['HF_HUB_OFFLINE']
            raise
    
    def rerank(self, query: str, results: List['RetrievalResult']) -> List['RetrievalResult']:
        """使用模型重排序"""
        if not results or self.model is None:
            return results
        
        try:
            # 准备输入对
            query_doc_pairs = [(query, result.content) for result in results]
            
            # 为了避免 padding token 问题，使用 batch_size=1 或者较小的批处理
            safe_batch_size = min(self.batch_size, 1)  # 暂时使用 batch_size=1 避免问题
            
            # 如果只有少量结果，逐个处理
            if len(results) <= 3:
                relevance_scores = []
                for query_doc_pair in query_doc_pairs:
                    score = self.model.predict([query_doc_pair], show_progress_bar=False)[0]
                    relevance_scores.append(score)
            else:
                # 对于更多结果，尝试批量处理
                try:
                    relevance_scores = self.model.predict(
                        query_doc_pairs,
                        batch_size=safe_batch_size,
                        show_progress_bar=False
                    )
                except Exception as batch_error:
                    self.logger.warning(f"批量处理失败，转为逐个处理: {batch_error}")
                    # 回退到逐个处理
                    relevance_scores = []
                    for query_doc_pair in query_doc_pairs:
                        score = self.model.predict([query_doc_pair], show_progress_bar=False)[0]
                        relevance_scores.append(score)
            
            # 更新分数
            for result, score in zip(results, relevance_scores):
                result.metadata['original_score'] = result.score
                result.metadata['model_rerank_score'] = float(score)
                result.score = float(score)  # 使用模型分数
            
            # 重新排序
            results.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"模型重排序完成，处理了 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"模型重排序失败: {e}")
            return results

class ResultProcessor:
    """结果处理器 - 负责合并、归一化和重排序检索结果"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        初始化结果处理器
        
        Args:
            config_dict: 配置字典
        """
        self.logger = setup_logger(__name__)
        self.config = config_dict or {}
        
        # 初始化千问重排序模型（如果配置了）
        self.qwen_reranker = None
        if self.config.get('rerank', {}).get('enabled', False):
            try:
                rerank_config = self.config.get('rerank', {})
                model_path = rerank_config.get('model_path', r"C:\Users\zero\Desktop\ht-rag\models\Qwen3-Reranker-0.6B")
                device = rerank_config.get('device', 'auto')
                batch_size = rerank_config.get('batch_size', 1)
                
                self.qwen_reranker = ModelBasedReranker(
                    model_path=model_path,
                    device=device,
                    batch_size=batch_size
                )
                self.logger.info("千问重排序模型初始化成功")
            except Exception as e:
                self.logger.warning(f"千问重排序模型初始化失败: {e}")
    
    def combine_results(
        self, 
        dense_results: List['RetrievalResult'],
        sparse_results: List['RetrievalResult'],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List['RetrievalResult']:
        """
        合并密集检索和稀疏检索结果
        
        Args:
            dense_results: 密集检索结果
            sparse_results: 稀疏检索结果
            dense_weight: 密集检索权重
            sparse_weight: 稀疏检索权重
            
        Returns:
            合并后的结果列表
        """
        # 归一化分数
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)
        
        # 使用字典合并结果，避免重复
        result_dict = {}
        
        # 添加密集检索结果
        for result in dense_results:
            doc_id = result.doc_id
            result.score = result.score * dense_weight
            result.retrieval_type = 'hybrid'
            result_dict[doc_id] = result
        
        # 添加稀疏检索结果
        for result in sparse_results:
            doc_id = result.doc_id
            if doc_id in result_dict:
                # 合并分数
                result_dict[doc_id].score += result.score * sparse_weight
            else:
                result.score = result.score * sparse_weight
                result.retrieval_type = 'hybrid'
                result_dict[doc_id] = result
        
        # 按分数排序
        combined_results = list(result_dict.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"合并了密集检索({len(dense_results)})和稀疏检索({len(sparse_results)})的结果，最终结果数: {len(combined_results)}")
        return combined_results
    
    def _normalize_scores(self, results: List['RetrievalResult']) -> List['RetrievalResult']:
        """归一化分数到[0,1]范围"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        if len(scores) == 1:
            results[0].score = 1.0
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for result in results:
                result.score = 1.0
        else:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def qwen_rerank_results(self, query: str, results: List['RetrievalResult']) -> List['RetrievalResult']:
        """使用千问重排序模型重新排序结果"""
        if not self.qwen_reranker or not results:
            return results
        
        try:
            # 直接使用原始结果，不需要类型转换
            reranked = self.qwen_reranker.rerank(query, results)
            
            self.logger.info(f"千问重排序完成，处理了{len(results)}个结果")
            return reranked
            
        except Exception as e:
            self.logger.error(f"千问重排序失败: {e}")
            return results
    
    def rerank_results(self, query: str, results: List['RetrievalResult'], use_qwen: bool = True) -> List['RetrievalResult']:
        """
        重排序结果（优先使用千问重排序，回退到简单重排序）
        
        Args:
            query: 原始查询
            results: 待重排序的结果
            use_qwen: 是否使用千问重排序
            
        Returns:
            重排序后的结果
        """
        if use_qwen and self.qwen_reranker:
            return self.qwen_rerank_results(query, results)
        else:
            # 使用规则重排序作为备选
            from MedicalRAG.utils.retrival_utils import RetrievalResult
            rule_reranker = RuleBasedReranker()
            return rule_reranker.rerank(query, results)


# 注释掉原来的别名定义以避免循环导入
# from MedicalRAG.utils.retrival_utils import RetrievalResult as RerankRetrievalResult