"""混合检索系统主入口

主要功能:
整合了基于文本的密集检索（如Vector Search）、稀疏检索（如BM25）以及后续的重排序（Reranking）
功能，形成一个完整的、端到端的文本检索流程。它作为应用层的协调器，调用底层utils中的
各个组件（如AnswerGenerator），专注于业务逻辑的编排，
接收用户查询，返回最终生成的答案。

核心组件:
- **DenseRetriever**: 密集检索器 - 基于语义向量搜索
- **BM25SparseRetriever**: BM25稀疏检索器 - 基于关键词匹配的传统检索方法
- **HybridSearchEngine**: 混合搜索引擎 - 核心检索逻辑
- **HybridRetriever**: 核心类，封装了检索和生成的完整逻辑。
- **main函数**: 提供了一个命令行使用的示例，用于批量处理问题并评估结果。
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from MedicalRAG.config.config import config, get_document_collection
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.answer_generator import AnswerGenerator
from MedicalRAG.utils.embedding_utils import embedding_provider

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
    retrieval_type: str  # 'dense', 'sparse', 'hybrid', 'image'


# 直接从utils模块导入实现
from MedicalRAG.utils.retrival_utils import BM25SparseRetriever as _BM25SparseRetriever
from MedicalRAG.utils.retrival_utils import DenseRetriever as _DenseRetriever



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
        self.dense_retriever = _DenseRetriever(self.config)
        self.bm25_retriever = _BM25SparseRetriever()
        
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


class HybridRetriever:
    """混合检索器 - 应用层协调器，专注于业务流程编排"""
    
    def __init__(self, config_path: str = None):
        """
        初始化混合检索器
        
        Args:
            config_path: 配置文件路径（可选，默认使用全局config）
        """
        self.logger = setup_logger(__name__)
        self.config = config  # 使用全局配置
        
        # 初始化核心组件
        self.search_engine = HybridSearchEngine(self.config)
        self.answer_generator = AnswerGenerator(self.config)
        
        # 会话相关
        self.current_session_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info("混合检索器初始化完成")
    
    def search_and_generate(
        self, 
        query: str, 
        top_k: int = 10,
        **search_kwargs
    ) -> Dict[str, Any]:
        """
        一站式检索并生成答案
        
        Args:
            query: 用户查询
            top_k: 检索结果数量
            **search_kwargs: 传递给hybrid_search的其他参数
            
        Returns:
            包含检索结果和生成答案的字典
        """
        # 执行混合检索
        results = self.search_engine.hybrid_search(query, top_k, **search_kwargs)
        
        # 生成答案
        answer = self.answer_generator.generate_answer(query, results)
        
        # 构建返回结果
        result_dict = {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "retrieval": {}
        }
        
        # 添加检索结果详情
        for i, result in enumerate(results):
            result_dict["retrieval"][f"doc_{i+1}"] = {
                "doc_id": result.doc_id,
                "content": result.content,
                "score": result.score,
                "retrieval_type": result.retrieval_type,
                "metadata": result.metadata
            }
        
        return result_dict
    
    def save_query_result(self, query_result: Dict[str, Any]):
        """保存单个查询结果到会话"""
        if not hasattr(self, 'current_session_results'):
            self.current_session_results = []
        self.current_session_results.append(query_result)
    
    def save_all_results(self) -> str:
        """保存所有会话结果到文件"""
        if not hasattr(self, 'current_session_results') or not self.current_session_results:
            self.logger.warning("没有查询结果需要保存")
            return ""
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"query_results_{timestamp}.json"
        
        # 确保results目录存在
        os.makedirs("results/retrival", exist_ok=True)
        full_path = os.path.join("results/retrival", output_file)

        # 保存到文件
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_session_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"保存了 {len(self.current_session_results)} 个查询结果到: {full_path}")
        return full_path
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        return self.search_engine.get_database_info()


# def main():
#     """主函数 - 演示混合检索系统的使用"""
#     # 初始化检索器
#     retriever = HybridRetriever()

#     # 显示数据库信息
#     db_info = retriever.get_database_info()
#     print(f"\n=== ChromaDB数据库信息 ===")
#     print(f"集合名称: {db_info['collection_name']}")
#     print(f"文档数量: {db_info['document_count']}")
#     print(f"存储路径: {db_info['vectorstore_path']}")
    
#     if db_info['document_count'] == 0:
#         print("警告: 数据库中没有文档，请先运行 document_loader.py 创建索引")
#         return

#     # 读取测试问题
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(os.path.dirname(current_dir))
#     truth_query_path = os.path.join(project_root, "data", "truth_query.json")
#     if not os.path.exists(truth_query_path):
#         print(f"错误: 找不到测试问题文件 {truth_query_path}")
#         return
    
#     with open(truth_query_path, 'r', encoding='utf-8') as f:
#         questions = json.load(f)

#     # 处理所有问题
#     print(f"\n=== 开始处理 {len(questions)} 个问题 ===")
#     for i, item in enumerate(questions, 1):
#         query = item.get("question", "")
#         if not query.strip():
#             continue
            
#         print(f"\n[{i}/{len(questions)}] 处理问题: {query}")
        
#         # --- 从配置中获取参数 ---
#         top_k = retriever.config.get('retriever', {}).get('top_k', 10)

#         # 执行检索和答案生成, 参数由config控制
#         result = retriever.search_and_generate(
#             query, 
#             top_k=top_k
#         )
        
#         # 保存结果
#         retriever.save_query_result(result)
        
#         print(f"✓ 完成，检索到 {result['total_results']} 个相关文档")

#     # 保存所有结果
#     final_file = retriever.save_all_results()
#     print(f"\n=== 处理完成 ===")
#     print(f"所有结果已保存到: {final_file}")


# if __name__ == "__main__":
#     main()