"""
混合检索系统主入口
整合密集检索、稀疏检索和重排序的完整RAG流程

重构后的简化架构：
1. 使用utils模块中的组件
2. 简化主要逻辑流程  
3. 专注于业务逻辑协调
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.retrival_utils import HybridSearchEngine, RetrievalResult
from MedicalRAG.utils.answer_generator import AnswerGenerator


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


def main():
    """主函数 - 演示混合检索系统的使用"""
    # --- 优化点 ---
    # 此处强制指定使用我们定制化索引流程创建的集合
    # 这是为了确保检索系统能利用图文块优化后的高质量数据
    # config['document']['collection_name'] = "medical_rag_custom"
    
    # 初始化检索器
    retriever = HybridRetriever()

    # 显示数据库信息
    db_info = retriever.get_database_info()
    print(f"\n=== ChromaDB数据库信息 ===")
    print(f"集合名称: {db_info['collection_name']}")
    print(f"文档数量: {db_info['document_count']}")
    print(f"存储路径: {db_info['vectorstore_path']}")
    
    if db_info['document_count'] == 0:
        print("警告: 数据库中没有文档，请先运行 document_loader_bge.py 创建索引")
        return

    # 读取测试问题
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    truth_query_path = os.path.join(project_root, "data", "truth_query.json")
    if not os.path.exists(truth_query_path):
        print(f"错误: 找不到测试问题文件 {truth_query_path}")
        return
    
    with open(truth_query_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # 处理所有问题
    print(f"\n=== 开始处理 {len(questions)} 个问题 ===")
    for i, item in enumerate(questions, 1):
        query = item.get("question", "")
        if not query.strip():
            continue
            
        print(f"\n[{i}/{len(questions)}] 处理问题: {query}")
        
        # 执行检索和答案生成
        result = retriever.search_and_generate(
            query, 
            top_k=5,
            enable_reranking=True,
            use_qwen_rerank=True
        )
        
        # 保存结果
        retriever.save_query_result(result)
        
        print(f"✓ 完成，检索到 {result['total_results']} 个相关文档")

    # 保存所有结果
    final_file = retriever.save_all_results()
    print(f"\n=== 处理完成 ===")
    print(f"所有结果已保存到: {final_file}")


if __name__ == "__main__":
    main()
