"""
医疗RAG系统 - 完整解决方案
集成文档索引创建和混合检索功能

功能特性：
1. 支持重建索引（可选）
2. 支持单个或多个查询
3. 混合检索（密集+稀疏+重排序）
4. 结果保存为JSON格式
5. 交互式界面

使用方式：
1. 设置是否重建索引
2. 输入查询问题
3. 自动保存结果
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.retrival_utils import HybridSearchEngine, RetrievalResult
from MedicalRAG.utils.answer_generator import AnswerGenerator
from MedicalRAG.index.document_loader import load_and_index_documents
from MedicalRAG.retrival.query_images import CaptionImageMatcher


class MedicalRAGSystem:
    """医疗RAG系统主类"""
    
    def __init__(self):
        """初始化系统"""
        self.logger = setup_logger(__name__)
        self.config = config
        
        # 初始化核心组件
        self.search_engine = None
        self.answer_generator = None
        self.image_matcher = None
        
        # 会话相关
        self.current_session_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info("医疗RAG系统初始化完成")
    
    def initialize_components(self):
        """初始化检索和生成组件"""
        try:
            self.search_engine = HybridSearchEngine(self.config)
            self.answer_generator = AnswerGenerator(self.config)

            # 初始化图像匹配器
            if 'image_retrieval' in self.config:
                self.logger.info("正在初始化图像检索组件...")
                image_config = self.config['image_retrieval']
                self.image_matcher = CaptionImageMatcher(
                    text_embedding_model_path=image_config['text_embedding_model'],
                    chroma_persist_dir=image_config['chroma_persist_dir'],
                    collection_name=image_config['collection_name']
                )
                self.logger.info("图像检索组件初始化成功")
            else:
                self.logger.warning("配置文件中未找到 'image_retrieval' 配置, 图像检索功能将不可用")

            self.logger.info("系统组件初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}", exc_info=True)
            return False
    
    def rebuild_index(self) -> bool:
        """重建文档索引"""
        try:
            self.logger.info("开始重建文档索引...")
            index = load_and_index_documents()
            self.logger.info("文档索引重建完成")
            return True
        except Exception as e:
            self.logger.error(f"索引重建失败: {e}")
            print(f"索引重建失败: {e}")
            return False
    
    def check_database_status(self) -> Dict[str, Any]:
        """检查数据库状态"""
        try:
            if not self.search_engine:
                self.initialize_components()
            
            db_info = self.search_engine.get_database_info()
            return db_info
        except Exception as e:
            self.logger.error(f"检查数据库状态失败: {e}")
            return {"error": str(e)}
    
    def search_and_generate(
        self, 
        query: str, 
        top_k: int = 2,
        enable_reranking: bool = True,
        use_qwen_rerank: bool = True,
        enable_image_search: bool = True
    ) -> Dict[str, Any]:
        """
        执行检索并生成答案
        
        Args:
            query: 用户查询
            top_k: 文本检索结果数量
            enable_reranking: 是否启用重排序
            use_qwen_rerank: 是否使用Qwen重排序
            enable_image_search: 是否启用图像检索
            
        Returns:
            包含检索结果和生成答案的字典
        """
        try:
            if not self.search_engine:
                if not self.initialize_components():
                    return {"error": "系统组件初始化失败"}
            
            # --- 阶段1: 混合文本检索 ---
            text_results = self.search_engine.hybrid_search(
                query, 
                top_k, 
                enable_reranking=enable_reranking,
                use_qwen_rerank=use_qwen_rerank
            )
            
            # --- 阶段2: 从文本中提取描述并关联图像 ---
            image_results: List[RetrievalResult] = []
            found_image_paths = set() # 用于避免重复添加相同的图片
            if enable_image_search and self.image_matcher:
                self.logger.info("开始从检索到的文本中关联图像...")
                # 预编译正则表达式以提取图片标题
                caption_pattern = re.compile(r"(图\d+-\d+\s+[\w\s（）(),]+)")
                
                for text_result in text_results:
                    # 从文本内容中查找所有匹配的图片标题
                    captions = caption_pattern.findall(text_result.content)
                    if not captions:
                        continue

                    for caption in captions:
                        caption = caption.strip()
                        self.logger.info(f"从文本块中提取到图片标题进行搜索: '{caption}'")
                        
                        # 使用提取到的标题精确搜索图片，只取最相关的那一张
                        raw_image_results = self.image_matcher.search(caption, top_n=1)
                        
                        # 如果找到了图片，则处理并添加到结果列表
                        if raw_image_results and raw_image_results.get('ids') and raw_image_results['ids'][0]:
                            image_base_path = self.config.get('image_retrieval', {}).get('image_base_path', 'data/processed/image')
                            meta = raw_image_results['metadatas'][0][0]
                            image_path_suffix = meta['image_path']

                            # 如果该图片尚未添加，则进行处理
                            if image_path_suffix not in found_image_paths:
                                found_image_paths.add(image_path_suffix)
                                full_image_path = os.path.join(image_base_path, image_path_suffix)
                                
                                # 将图像检索结果包装成统一的 RetrievalResult 格式
                                image_results.append(RetrievalResult(
                                    doc_id=image_path_suffix,
                                    content=f"图片描述: {raw_image_results['documents'][0][0]}\n图片路径: {full_image_path}",
                                    score=1 - raw_image_results['distances'][0][0], # 使用本次精确搜索的分数
                                    retrieval_type='image_from_text', # 新的检索类型，方便调试
                                    metadata=meta
                                ))
                                self.logger.info(f"成功找到关联图片: {full_image_path}")
                                # 新增：以更易读的格式打印图片的详细元数据
                                self.logger.info(f"图片详细元数据:\n{json.dumps(meta, ensure_ascii=False, indent=2)}")

            # 3. 合并和排序结果
            all_results = sorted(
                text_results + image_results, 
                key=lambda x: x.score, 
                reverse=True
            )
            
            # 4. 生成答案
            answer = self.answer_generator.generate_answer(query, all_results)
            
            # 5. 构建返回结果
            result_dict = {
                "query": query,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "total_results": len(all_results),
                "retrieval": {}
            }
            
            # 添加检索结果详情
            for i, result in enumerate(all_results):
                result_dict["retrieval"][f"doc_{i+1}"] = {
                    "doc_id": result.doc_id,
                    "content": result.content,
                    "score": result.score,
                    "retrieval_type": result.retrieval_type,
                    "metadata": result.metadata
                }
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"检索生成失败: {e}", exc_info=True)
            return {"error": str(e)}
    
    def process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表
            
        Returns:
            结果列表
        """
        results = []
        
        print(f"\n开始处理 {len(queries)} 个查询...")
        
        for i, query in enumerate(queries, 1):
            if not query.strip():
                continue
                
            print(f"\n[{i}/{len(queries)}] 处理查询: {query}")
            
            # 执行检索和答案生成
            result = self.search_and_generate(query)
            
            if "error" in result:
                print(f"❌ 处理失败: {result['error']}")
            else:
                print(f"✓ 完成，检索到 {result['total_results']} 个相关文档")
                
            results.append(result)
            self.current_session_results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]] = None) -> str:
        """保存结果到文件"""
        if results is None:
            results = self.current_session_results
        
        if not results:
            self.logger.warning("没有结果需要保存")
            return ""
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"medical_rag_results_{timestamp}.json"
        
        # 确保results目录存在
        os.makedirs("results/retrival", exist_ok=True)
        full_path = os.path.join("results/retrival", output_file)
        
        # 保存到文件
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"保存了 {len(results)} 个查询结果到: {full_path}")
        print(f"结果已保存到: {full_path}")
        return full_path
    
    def interactive_mode(self):
        """交互式模式"""
        print("\n=== 医疗RAG系统交互式模式 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'save' 保存当前会话结果")
        print("输入 'status' 查看数据库状态")
        
        while True:
            try:
                query = input("\n请输入查询问题: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    break
                elif query.lower() == 'save':
                    self.save_results()
                    continue
                elif query.lower() == 'status':
                    db_info = self.check_database_status()
                    print(f"数据库状态: {json.dumps(db_info, indent=2, ensure_ascii=False)}")
                    continue
                elif not query:
                    continue
                
                # 处理查询
                result = self.search_and_generate(query)
                
                if "error" in result:
                    print(f"❌ 处理失败: {result['error']}")
                else:
                    print(f"\n📝 答案: {result['answer']}")
                    print(f"📊 检索到 {result['total_results']} 个相关文档")
                    
                    # 显示前3个检索结果
                    if result['total_results'] > 0:
                        print("\n📋 主要检索结果:")
                        for i in range(min(3, result['total_results'])):
                            doc_info = result['retrieval'][f'doc_{i+1}']
                            print(f"  {i+1}. 评分: {doc_info['score']:.3f}")
                            print(f"     内容: {doc_info['content'][:100]}...")
                    
                    self.current_session_results.append(result)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"处理过程中发生错误: {e}")
        
        # 退出时保存结果
        if self.current_session_results:
            self.save_results()
    
    def run_with_file(self, file_path: str):
        """从文件读取查询并处理"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取查询
            queries = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'question' in item:
                        queries.append(item['question'])
                    elif isinstance(item, str):
                        queries.append(item)
            elif isinstance(data, dict) and 'questions' in data:
                queries = data['questions']
            
            if not queries:
                print("文件中没有找到有效的查询问题")
                return
            
            # 处理查询
            results = self.process_queries(queries)
            
            # 保存结果
            self.save_results(results)
            
        except Exception as e:
            print(f"从文件处理查询失败: {e}")
            self.logger.error(f"从文件处理查询失败: {e}")


def main():
    """主函数"""
    print("=== 医疗RAG系统 ===")
    
    # 初始化系统
    rag_system = MedicalRAGSystem()
    
    # 询问是否重建索引
    rebuild_choice = input("\n是否需要重建索引? (y/n): ").lower().strip()
    if rebuild_choice == 'y':
        print("\n正在重建索引...")
        if rag_system.rebuild_index():
            print("✓ 索引重建成功")
        else:
            print("❌ 索引重建失败，但可以尝试使用现有索引")
    
    # 检查数据库状态
    print("\n检查数据库状态...")
    db_info = rag_system.check_database_status()
    
    if "error" in db_info:
        print(f"❌ 数据库检查失败: {db_info['error']}")
        return
    
    print(f"✓ 数据库状态正常")
    print(f"  集合名称: {db_info.get('collection_name', 'N/A')}")
    print(f"  文档数量: {db_info.get('document_count', 'N/A')}")
    
    if db_info.get('document_count', 0) == 0:
        print("警告: 数据库中没有文档，建议重建索引")
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 交互式模式 (手动输入查询)")
    print("2. 文件模式 (从文件读取查询)")
    print("3. 使用默认测试问题")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == '1':
        rag_system.interactive_mode()
    elif choice == '2':
        file_path = input("请输入查询文件路径: ").strip()
        if os.path.exists(file_path):
            rag_system.run_with_file(file_path)
        else:
            print("文件不存在")
    elif choice == '3':
        # 使用默认测试问题
        truth_query_path = "./data/truth_query.json"
        if os.path.exists(truth_query_path):
            rag_system.run_with_file(truth_query_path)
        else:
            print("默认测试文件不存在")
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
