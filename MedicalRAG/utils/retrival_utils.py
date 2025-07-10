"""
检索工具模块
提供密集检索、稀疏检索和混合检索的核心功能

主要功能：
1. BM25稀疏检索器 - 基于关键词的传统检索
2. 密集检索器 - 基于语义向量的检索
3. 检索结果统一数据结构
"""

import os
import sys
import torch
from MedicalRAG.config.config import get_document_collection
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from MedicalRAG.config.config import config
from MedicalRAG.utils.embedding_utils import embedding_provider
from MedicalRAG.utils.logger import setup_logger

# 尝试导入jieba用于中文分词
try:
    import jieba
    jieba.setLogLevel(jieba.logging.INFO)
except ImportError:
    print("警告: jieba 未安装，将使用简单的字符分词")
    jieba = None

@dataclass
class RetrievalResult:
    """检索结果统一数据结构"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_type: str  # 'dense', 'sparse', 'hybrid'


class BM25SparseRetriever:
    """BM25稀疏检索器 - 基于关键词匹配的传统检索方法"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            k1: 控制词频饱和度的参数
            b: 控制文档长度归一化的参数
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_lens = []
        self.avgdl = 0
        self.logger = setup_logger(__name__)
        
    def tokenize(self, text: str) -> List[str]:
        """文本分词处理"""
        if jieba:
            tokens = list(jieba.cut_for_search(text.lower()))
            self.logger.debug(f"jieba分词结果: {tokens}")
        else:
            # 简单的字符级分词作为备选
            tokens = [char for char in text.lower() if char.strip()]
        
        # 过滤短词，但对中文要宽松一些
        if jieba:
            tokens = [token for token in tokens if len(token.strip()) > 0 and token.strip() not in {'的', '了', '是', '在', '有', '和', '与', '或', '但', '而'}]
        else:
            tokens = [token for token in tokens if len(token.strip()) > 1]
        
        return tokens
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """构建BM25索引"""
        self.logger.info(f"构建BM25索引，文档数量: {len(documents)}")
        
        self.documents = documents
        self.doc_lens = []
        
        # 构建词频统计
        for doc in documents:
            content = str(doc.get('content', ''))
            tokens = self.tokenize(content)
            self.doc_lens.append(len(tokens))
            
            # 统计每个词在多少个文档中出现
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # 计算平均文档长度
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        
        # 计算IDF值
        num_docs = len(documents)
        for token, freq in self.doc_freqs.items():
            self.idf[token] = np.log((num_docs - freq + 0.5) / (freq + 0.5))
        
        self.logger.info(f"索引构建完成，词汇表大小: {len(self.idf)}")
    
    def get_scores(self, query: str) -> List[float]:
        """计算查询与所有文档的BM25分数"""
        query_tokens = self.tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.documents):
            content = str(doc.get('content', ''))
            doc_tokens = self.tokenize(content)
            
            # 计算文档中每个词的频率
            term_freqs = defaultdict(int)
            for token in doc_tokens:
                term_freqs[token] += 1
            
            score = 0.0
            for token in query_tokens:
                if token in self.idf:
                    tf = term_freqs[token]
                    idf = self.idf[token]
                    
                    # BM25公式
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl))
            
            scores.append(score)
        
        return scores
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """执行BM25搜索"""
        scores = self.get_scores(query)
        
        # 获取top_k结果
        doc_scores = [(i, score) for i, score in enumerate(scores)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in doc_scores[:top_k]:
            doc = self.documents[i]
            result = RetrievalResult(
                doc_id=str(doc.get('id', i)),
                content=str(doc.get('content', '')),
                metadata=doc.get('metadata', {}),
                score=score,
                retrieval_type='sparse'
            )
            results.append(result)
        
        return results


class DenseRetriever:
    """密集检索器 - 基于语义向量搜索"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化密集检索器
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict
        self.logger = setup_logger(__name__)
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_collection = None
        
        self._setup_components()
    
    def _setup_components(self):
        """设置检索组件"""
        # 初始化嵌入模型
        # 从配置中获取要使用的 embedding provider 的名称
        provider_name = self.config['embedding'].get('provider')
        if not provider_name:
            raise ValueError("配置文件 'embedding' 部分缺少 'provider' 键，请指定要使用的嵌入模型提供者")

        try:
            # 直接通过名称从 embedding_provider 获取模型实例
            self.embedding_model = embedding_provider[provider_name]
            self.logger.info(f"成功从 provider 获取嵌入模型: {provider_name}")
        except ValueError as e:
            self.logger.error(f"无法获取嵌入模型 '{provider_name}': {e}")
            raise  # 重新抛出异常
        
        # 使用统一的ChromaDB管理器
        self.chroma_collection = get_document_collection()
        self.logger.info("成功连接到ChromaDB集合")
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        执行密集检索（语义向量搜索）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.get_query_embedding(query)
            
            # 在ChromaDB中搜索
            chroma_results = self.chroma_collection.query(
                # query_embeddings=[query_embedding.tolist()]
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            results = []
            if chroma_results['documents'] and chroma_results['documents'][0]:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    chroma_results['ids'][0],
                    chroma_results['documents'][0],
                    chroma_results['metadatas'][0],
                    chroma_results['distances'][0]
                )):
                    # 将距离转换为相似度分数
                    similarity_score = 1 - distance if distance is not None else 0.0
                    
                    result = RetrievalResult(
                        doc_id=doc_id,
                        content=document,
                        metadata=metadata or {},
                        score=similarity_score,
                        retrieval_type='dense'
                    )
                    results.append(result)
            
            self.logger.info(f"密集检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"密集检索失败: {e}")
            return []


class HybridSearchEngine:
    """混合搜索引擎 - 核心检索逻辑"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化混合搜索引擎
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict
        self.logger = setup_logger(__name__)
        
        # 初始化检索组件
        self.dense_retriever = DenseRetriever(self.config)
        self.bm25_retriever = BM25SparseRetriever()
        
        # 为BM25检索器加载文档
        self._load_documents_for_bm25()
        
        self.logger.info("混合搜索引擎初始化完成")
    
    def _load_documents_for_bm25(self):
        """从ChromaDB加载文档并为BM25构建索引"""
        try:
            # 从ChromaDB获取所有文档
            collection = self.dense_retriever.chroma_collection
            
            # 分批获取所有文档 (ChromaDB的limit限制)
            batch_size = 1000
            offset = 0
            all_documents = []
            
            while True:
                # 获取一批文档
                results = collection.get(
                    offset=offset,
                    limit=batch_size,
                    include=['documents', 'metadatas']
                )
                
                if not results['documents']:
                    break
                
                # 转换为BM25需要的格式
                for doc_id, content, metadata in zip(
                    results['ids'], 
                    results['documents'], 
                    results['metadatas']
                ):
                    document = {
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata or {}
                    }
                    all_documents.append(document)
                
                offset += batch_size
                
                # 如果返回的文档少于batch_size，说明已经到达末尾
                if len(results['documents']) < batch_size:
                    break
            
            self.logger.info(f"从ChromaDB加载了 {len(all_documents)} 个文档用于BM25索引")
            
            # 为BM25构建索引
            if all_documents:
                self.bm25_retriever.build_index(all_documents)
                self.logger.info("BM25索引构建完成")
            else:
                self.logger.warning("没有找到文档来构建BM25索引")
                
        except Exception as e:
            self.logger.error(f"加载文档构建BM25索引失败: {e}")
            # 如果失败，至少确保BM25有一个空的文档列表
            self.bm25_retriever.build_index([])

    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库基本信息"""
        try:
            collection = self.dense_retriever.chroma_collection
            return {
                "document_count": collection.count()
            }
        except Exception as e:
            self.logger.error(f"获取数据库信息失败: {e}")
            return {"error": str(e)}

    def dense_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """执行密集检索"""
        return self.dense_retriever.search(query, top_k)
    
    def sparse_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """执行稀疏检索"""
        return self.bm25_retriever.search(query, top_k)
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        enable_reranking: bool = True,
        use_qwen_rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        混合检索 - 核心方法
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            dense_weight: 密集检索权重
            sparse_weight: 稀疏检索权重
            enable_reranking: 是否启用重排序
            use_qwen_rerank: 是否使用千问重排序
            
        Returns:
            检索结果列表
        """
        self.logger.info(f"开始混合检索: '{query}'")
        
        # 1. 分别执行密集检索和稀疏检索
        dense_results = self.dense_search(query, top_k * 2)  # 获取更多候选
        sparse_results = self.sparse_search(query, top_k * 2)
        
        self.logger.info(f"密集检索返回 {len(dense_results)} 个结果，稀疏检索返回 {len(sparse_results)} 个结果")
        
        # 2. 合并结果 (需要导入ResultProcessor)
        from MedicalRAG.utils.rerank_utils import ResultProcessor
        result_processor = ResultProcessor(self.config)
        
        combined_results = result_processor.combine_results(
            dense_results, 
            sparse_results, 
            dense_weight, 
            sparse_weight
        )
        
        # 3. 重排序（可选）
        if enable_reranking and combined_results:
            self.logger.info("开始重排序...")
            combined_results = result_processor.rerank_results(
                query, 
                combined_results, 
                use_qwen=use_qwen_rerank
            )
        
        # 4. 返回top_k结果
        final_results = combined_results[:top_k]
        self.logger.info(f"混合检索完成，返回 {len(final_results)} 个结果")
        
        return final_results