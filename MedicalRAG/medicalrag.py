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

from MedicalRAG.config.config import config, config_manager
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.retrival.text_retrival import HybridSearchEngine, RetrievalResult
from MedicalRAG.utils.answer_generator import AnswerGenerator
from MedicalRAG.index.run_custom_indexer import run_all_indexers
from MedicalRAG.retrival.image_retrival import ImageRetriever
from MedicalRAG.config.config import get_chroma_client
from MedicalRAG.utils.image_filter_utils import create_image_filter


class MedicalRAGSystem:
    """超声RAG系统主类"""
    
    def __init__(self):
        """初始化系统"""
        self.logger = setup_logger(__name__)
        self.config = config
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 初始化核心组件
        self.search_engine = None
        self.answer_generator = None
        self.image_retriever = None
        self.image_filter = None
        
        # 会话相关
        self.current_session_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info("医疗超声RAG系统初始化完成")
    
    def initialize_components(self):
        """初始化检索和生成组件"""
        try:
            self._init_core_components()
            self._init_image_components()
            self.logger.info("系统组件初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}", exc_info=True)
            return False
    
    def _init_core_components(self):
        """初始化核心组件"""
        self.search_engine = HybridSearchEngine(self.config)
        self.answer_generator = AnswerGenerator(self.config)
    
    def _init_image_components(self):
        """初始化图像相关组件"""
        image_search_config = self.config.get('image_search', {})
        if not image_search_config.get('enabled', False):
            self.logger.warning("图像检索功能未启用")
            return
        
        print(f"图像检索功能已启用: True")
        self.logger.info("正在初始化图像检索组件...")
        
        # 确保图像集合存在
        collection_name = image_search_config['collection_name']
        chroma_client = get_chroma_client()
        chroma_client.get_or_create_collection(name=collection_name)
        self.logger.info(f"图像集合 '{collection_name}' 已确认存在或已创建")
        
        # 初始化图像检索器和过滤器
        self.image_retriever = ImageRetriever()
        self.logger.info("图像检索组件初始化成功")
        
        try:
            self.image_filter = create_image_filter(
                text_weight=0.3, image_weight=0.7,
                enable_text_filter=True, enable_image_filter=True
            )
            self.logger.info("图像过滤器初始化成功")
        except Exception as e:
            self.logger.warning(f"图像过滤器初始化失败: {e}")
            self.image_filter = None
    
    def rebuild_index(self) -> bool:
        """重建文档索引"""
        try:
            self.logger.info("开始重建文档索引...")
            # index = load_and_index_documents()
            index = run_all_indexers()
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
    
    def _get_full_image_path(self, image_path_suffix: str) -> str:
        """根据相对路径获取图片的完整绝对路径"""
        image_base_path_rel = self.config.get('image_search', {}).get('image_base_path', 'data/processed/image')
        image_base_path_abs = os.path.join(self.project_root, image_base_path_rel)
        return os.path.normpath(os.path.join(image_base_path_abs, image_path_suffix))
    
    def _is_valid_result(self, result_dict: dict) -> bool:
        """检查检索结果是否有效"""
        return result_dict and "error" not in result_dict and result_dict.get('results')
    
    def _convert_to_retrieval_results(self, results: list, retrieval_type: str) -> List[RetrievalResult]:
        """将检索结果转换为RetrievalResult对象"""
        converted_results = []
        for result_item in results:
            image_path_suffix = result_item['metadata'].get('image_path')
            full_image_path = self._get_full_image_path(image_path_suffix)
            
            converted_results.append(RetrievalResult(
                doc_id=result_item['image_id'],
                content=f"图片描述: {result_item['caption']}\n图片路径: {full_image_path}",
                score=result_item['score'],
                retrieval_type=retrieval_type,
                metadata=result_item['metadata']
            ))
        return converted_results

    def _perform_image_to_image_search(self, image_path: str, top_k: int) -> List[RetrievalResult]:
        """执行以图搜图"""
        if not (image_path and self.image_retriever):
            return []

        self.logger.info(f"开始以图搜图, 图片路径: {image_path}")
        result_dict = self.image_retriever.search_by_image(image_path, top_k=top_k)
        
        if not self._is_valid_result(result_dict):
            return []

        results = self._convert_to_retrieval_results(result_dict['results'], 'image_to_image')
        self.logger.info(f"以图搜图完成, 找到 {len(results)} 张相似图片")
        return results

    def _perform_text_to_image_caption_search(self, text_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """从文本结果中提取标题并通过图片标题搜索关联图片"""
        if not self.image_retriever:
            self.logger.warning("图像检索器未初始化")
            return []

        self.logger.info("开始从检索到的文本中关联图像...")
        caption_pattern = re.compile(r"(图\d+-\d+\s+[\w\s（）(),]+)")
        found_image_paths = set()
        image_results = []

        for text_result in text_results:
            captions = caption_pattern.findall(text_result.content)
            for caption in captions:
                caption = caption.strip()
                self.logger.info(f"从文本块中提取到图片标题进行搜索: '{caption}'")
                
                result_dict = self.image_retriever.search_by_text(caption, top_k=1, search_type="caption")
                
                if not self._is_valid_result(result_dict):
                    continue

                result_item = result_dict['results'][0]
                image_path_suffix = result_item['metadata']['image_path']

                if image_path_suffix not in found_image_paths:
                    found_image_paths.add(image_path_suffix)
                    full_image_path = self._get_full_image_path(image_path_suffix)
                    
                    image_results.append(RetrievalResult(
                        doc_id=result_item['image_id'],
                        content=f"图片描述: {result_item['caption']}\n图片路径: {full_image_path}",
                        score=result_item['score'],
                        retrieval_type='image_from_text',
                        metadata=result_item['metadata']
                    ))
                    self.logger.info(f"成功找到关联图片: {full_image_path}")
                    self.logger.info(f"图片详细元数据:\n{json.dumps(result_item['metadata'], ensure_ascii=False, indent=2)}")
        return image_results

    def _perform_text_to_image_clip_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """使用CLIP模型直接根据查询文本检索图像内容"""
        if not self.image_retriever:
            self.logger.warning("图像检索器未初始化，无法执行CLIP文本检索")
            return []

        self.logger.info(f"开始使用CLIP模型检索图像内容: {query}")
        
        try:
            result_dict = self.image_retriever.search_by_text(query, top_k=top_k, search_type="clip")
            self.logger.info(f"CLIP搜索原始结果: {result_dict}")
            
            if not self._is_valid_result(result_dict):
                self.logger.warning("CLIP文本检索未返回有效结果")
                return []

            results = self._convert_to_retrieval_results(result_dict['results'], 'clip_text_to_image')
            self.logger.info(f"CLIP文本检索完成，找到 {len(results)} 张相关图片")
            return results
            
        except Exception as e:
            self.logger.error(f"CLIP文本检索失败: {e}", exc_info=True)
            return []

    def _perform_all_image_searches(
        self, query: str, image_path: Optional[str], text_results: List[RetrievalResult], 
        top_k: int, enable_image_search: bool, enable_clip_text_search: bool
    ) -> List[RetrievalResult]:
        """执行所有类型的图像检索"""
        image_results = []
        
        # 以图搜图
        image_results.extend(self._perform_image_to_image_search(image_path, top_k))
        
        # 文本标题关联图片
        if enable_image_search:
            image_results.extend(self._perform_text_to_image_caption_search(text_results))
        
        # CLIP文本搜图
        if enable_clip_text_search:
            image_results.extend(self._perform_text_to_image_clip_search(query, top_k))
        
        return image_results
    
    def _apply_image_filtering(
        self, query: str, image_path: Optional[str], image_results: List[RetrievalResult],
        enable_filtering: bool, filter_type: str, similarity_threshold: float
    ) -> List[RetrievalResult]:
        """应用图像过滤器"""
        if not (enable_filtering and self.image_filter and image_results):
            return image_results
        
        try:
            self.logger.info(f"开始应用图像过滤器，过滤类型: {filter_type}")
            filtered_results = self.image_filter.rerank_results(
                query_text=query, query_image_path=image_path,
                results=image_results, filter_type=filter_type,
                similarity_threshold=similarity_threshold
            )
            
            filter_stats = self.image_filter.get_filter_stats(filtered_results)
            self.logger.info(f"图像过滤完成，统计信息: {filter_stats}")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"图像过滤失败: {e}")
            return image_results
    
    def _generate_answer_safely(self, query: str, all_results: List[RetrievalResult], image_path: Optional[str]) -> str:
        """安全地生成答案"""
        try:
            return self.answer_generator.generate_answer(query, all_results, image_path)
        except Exception as e:
            self.logger.error(f"生成答案失败: {e}", exc_info=True)
            return f"生成答案失败: {e}"
    
    def _build_final_response(self, query: str, answer: str, all_results: List[RetrievalResult]) -> Dict[str, Any]:
        """构建最终的API响应字典"""
        return {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "total_results": len(all_results),
            "retrieval": {
                f"doc_{i+1}": {
                    "doc_id": result.doc_id,
                    "content": result.content,
                    "score": result.score,
                    "retrieval_type": result.retrieval_type,
                    "metadata": result.metadata
                }
                for i, result in enumerate(all_results)
            }
        }

    def search_and_generate(
        self, 
        query: str, 
        image_path: Optional[str] = None,
        top_k: Optional[int] = None,
        enable_reranking: bool = True,
        use_qwen_rerank: bool = True,
        enable_image_search: bool = True,
        enable_clip_text_search: bool = True,
        enable_image_filtering: bool = True,
        image_filter_type: str = "multimodal",
        similarity_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        执行检索并生成答案的核心流程
        """
        try:
            if not self.search_engine and not self.initialize_components():
                return {"error": "系统组件初始化失败"}
            
            top_k = top_k or self.config.get('retriever', {}).get('top_k', 10)

            # 1. 执行各类检索
            text_results = self.search_engine.hybrid_search(
                query, top_k, enable_reranking=enable_reranking, use_qwen_rerank=use_qwen_rerank
            )
            
            image_results = self._perform_all_image_searches(
                query, image_path, text_results, top_k, 
                enable_image_search, enable_clip_text_search
            )

            # 2. 应用图像过滤器
            filtered_image_results = self._apply_image_filtering(
                query, image_path, image_results, 
                enable_image_filtering, image_filter_type, similarity_threshold
            )
            
            # 3. 合并和排序结果
            all_results = sorted(
                text_results + filtered_image_results,
                key=lambda x: x.score, reverse=True
            )
            
            # 4. 生成答案
            answer = self._generate_answer_safely(query, all_results, image_path)
            
            return self._build_final_response(query, answer, all_results)
            
        except Exception as e:
            self.logger.error(f"检索生成过程发生严重错误: {e}", exc_info=True)
            return {"error": f"检索生成过程发生严重错误: {e}"}
    
    def process_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表
            
        Returns:
            结果列表
        """
        results = []
        valid_queries = [q for q in queries if q.strip()]
        
        print(f"\n开始处理 {len(valid_queries)} 个查询...")
        
        for i, query in enumerate(valid_queries, 1):
            print(f"\n[{i}/{len(valid_queries)}] 处理查询: {query}")
            
            result = self._process_single_query(query)
            results.append(result)
            self.current_session_results.append(result)
        
        return results
    
    def _process_single_query(self, query: str) -> Dict[str, Any]:
        """处理单个查询"""
        start_time = datetime.now()
        result = self.search_and_generate(query)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result['processing_time'] = processing_time
        
        if "error" in result:
            print(f"处理失败: {result['error']} (耗时: {processing_time:.2f}秒)")
        else:
            print(f"完成，检索到 {result['total_results']} 个相关文档 (耗时: {processing_time:.2f}秒)")
        
        return result
    
    def save_results(self, results: List[Dict[str, Any]] = None) -> str:
        """保存结果到文件"""
        results = results or self.current_session_results
        
        if not results:
            self.logger.warning("没有结果需要保存")
            return ""
        
        # 生成文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"medical_rag_results_{timestamp}.json"
        os.makedirs("results/retrival", exist_ok=True)
        full_path = os.path.join("results/retrival", output_file)
        
        # 计算统计信息并保存
        summary = self._calculate_summary_stats(results)
        final_output = {"summary": summary, "results": results}
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        self._log_save_results(full_path, len(results), summary)
        return full_path
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算结果统计信息"""
        total_questions = len(results)
        successful_queries = sum(1 for r in results if "error" not in r)
        failed_queries = total_questions - successful_queries
        total_time = sum(r.get('processing_time', 0) for r in results if "error" not in r)
        avg_time = total_time / successful_queries if successful_queries > 0 else 0
        
        return {
            "total_questions": total_questions,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": f"{successful_queries/total_questions*100:.2f}%" if total_questions > 0 else "0%",
            "total_processing_time": f"{total_time:.2f}秒",
            "average_processing_time": f"{avg_time:.2f}秒",
            "generated_at": datetime.now().isoformat()
        }
    
    def _log_save_results(self, full_path: str, result_count: int, summary: Dict[str, Any]):
        """记录保存结果的日志"""
        self.logger.info(f"保存了 {result_count} 个查询结果到: {full_path}")
        print(f"结果已保存到: {full_path}")
        print(f"统计信息: 成功 {summary['successful_queries']}/{summary['total_questions']}, 平均耗时 {summary['average_processing_time']}")
    
    def interactive_mode(self):
        """交互式模式"""
        self._print_interactive_help()
        
        while True:
            try:
                query = input("\n请输入查询问题: ").strip()
                if not query:
                    continue

                # 处理特殊命令
                if self._handle_special_commands(query):
                    continue
                
                # 获取用户输入参数
                image_path = self._get_image_path_input()
                enable_image_filtering, filter_type = self._get_filter_settings()
                
                # 处理查询并显示结果
                result = self.search_and_generate(
                    query, image_path=image_path,
                    enable_image_filtering=enable_image_filtering,
                    image_filter_type=filter_type
                )
                
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"处理过程中发生错误: {e}")
        
        # 退出时保存结果
        if self.current_session_results:
            self.save_results()
    
    def _print_interactive_help(self):
        """打印交互式模式帮助信息"""
        print("\n=== 医疗RAG系统交互式模式 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'save' 保存当前会话结果")
        print("输入 'status' 查看数据库状态")
    
    def _handle_special_commands(self, query: str) -> bool:
        """处理特殊命令，返回True表示已处理"""
        query_lower = query.lower()
        
        if query_lower in ['quit', 'exit']:
            return False  # 退出循环
        elif query_lower == 'save':
            self.save_results()
            return True
        elif query_lower == 'status':
            db_info = self.check_database_status()
            print(f"数据库状态: {json.dumps(db_info, indent=2, ensure_ascii=False)}")
            return True
        
        return False  # 不是特殊命令
    
    def _get_image_path_input(self) -> Optional[str]:
        """获取图像路径输入"""
        image_path = input("请输入查询图片的路径 (可选,直接回车跳过): ").strip()
        if image_path and not os.path.exists(image_path):
            self.logger.warning(f"提供的图片路径不存在: {image_path}")
            print("警告: 图片路径不存在，将只进行文本搜索。")
            return None
        return image_path or None
    
    def _get_filter_settings(self) -> tuple[bool, str]:
        """获取过滤器设置"""
        enable_filtering = input("是否启用图像过滤功能? (y/n, 默认y): ").strip().lower()
        enable_image_filtering = enable_filtering != 'n'
        
        filter_type = "multimodal"
        if enable_image_filtering:
            filter_type_input = input("选择过滤类型 (text/image/multimodal, 默认multimodal): ").strip().lower()
            if filter_type_input in ['text', 'image', 'multimodal']:
                filter_type = filter_type_input
        
        return enable_image_filtering, filter_type
    
    def _display_result(self, result: Dict[str, Any]):
        """显示查询结果"""
        if "error" in result:
            print(f"处理失败: {result['error']}")
            return
        
        print(f"\n答案: {result['answer']}")
        print(f"检索到 {result['total_results']} 个相关文档")
        
        if result.get('total_results', 0) > 0:
            self._display_top_results(result)
        
        self.current_session_results.append(result)
    
    def _display_top_results(self, result: Dict[str, Any]):
        """显示前几个检索结果"""
        print("\n主要检索结果:")
        retrieval_data = result.get('retrieval', {})
        for i in range(min(3, result['total_results'])):
            doc_key = f'doc_{i+1}'
            if doc_key in retrieval_data:
                doc_info = retrieval_data[doc_key]
                print(f"  {i+1}. 类型: {doc_info.get('retrieval_type', 'N/A')}, 评分: {doc_info.get('score', 0):.3f}")
                print(f"     内容: {doc_info.get('content', '')[:100]}...")
    
    def _parse_queries_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """从JSON文件中解析查询"""
        queries_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'question' in item:
                    queries_data.append(item)
                elif isinstance(item, str):
                    queries_data.append({'question': item})
        elif isinstance(data, dict) and 'questions' in data:
            for q in data['questions']:
                queries_data.append({'question': q})
        
        return queries_data

    def run_with_file(self, file_path: str):
        """从文件读取查询并处理"""
        try:
            original_data = self._parse_queries_from_file(file_path)
            queries = [item['question'] for item in original_data]
            
            if not queries:
                print("文件中没有找到有效的查询问题")
                return
            
            print(f"从文件中读取到 {len(queries)} 个问题")
            
            results = self.process_queries(queries)
            
            final_results = []
            for i, (original, result) in enumerate(zip(original_data, results)):
                if "error" not in result:
                    final_result = {
                        "id": i + 1,
                        "original_question": original.get('question', ''),
                        "original_answer": original.get('answer', ''),
                        "generated_answer": result.get('answer', ''),
                        "retrieval_results": result.get('retrieval', {}),
                        "total_retrieved": result.get('total_results', 0),
                        "timestamp": result.get('timestamp', ''),
                        "processing_time": result.get('processing_time', '')
                    }
                else:
                    final_result = {
                        "id": i + 1,
                        "original_question": original.get('question', ''),
                        "original_answer": original.get('answer', ''),
                        "error": result.get('error', ''),
                        "timestamp": datetime.now().isoformat()
                    }
                final_results.append(final_result)
            
            self.save_results(final_results)
            
        except Exception as e:
            self.logger.error(f"从文件处理查询失败: {e}", exc_info=True)
            print(f"从文件处理查询失败: {e}")


def main():
    """主函数，用于启动和管理RAG系统"""
    print("=== 医疗RAG系统 ===")
    rag_system = MedicalRAGSystem()

    # 1. 重建索引（可选）
    if input("\n是否需要重建索引? (y/n): ").lower().strip() == 'y':
        _handle_index_rebuild(rag_system)

    # 2. 检查数据库状态
    if not _check_database_status(rag_system):
        return

    # 3. 选择运行模式
    _run_selected_mode(rag_system)


def _handle_index_rebuild(rag_system):
    """处理索引重建"""
    print("\n正在重建索引...")
    if rag_system.rebuild_index():
        print("索引重建成功")
    else:
        print("索引重建失败，程序将继续尝试使用现有索引。")


def _check_database_status(rag_system) -> bool:
    """检查数据库状态，返回是否成功"""
    print("\n检查数据库状态...")
    db_info = rag_system.check_database_status()
    if "error" in db_info:
        print(f"数据库检查失败: {db_info['error']}")
        return False
    
    print(f"  数据库状态正常，集合 '{db_info.get('collection_name', 'N/A')}' 中有 {db_info.get('document_count', 'N/A')} 个文档。")
    if db_info.get('document_count', 0) == 0:
        print("警告: 数据库为空，建议重建索引。")
    return True


def _run_selected_mode(rag_system):
    """运行用户选择的模式"""
    menu = {
        '1': ('交互式模式', rag_system.interactive_mode),
        '2': ('文件模式', lambda: _run_file_mode(rag_system)),
        '3': ('使用默认测试问题', lambda: _run_default_test(rag_system))
    }
    
    print("\n请选择运行模式:")
    for key, (desc, _) in menu.items():
        print(f"{key}. {desc}")

    choice = input("请选择: ").strip()
    action = menu.get(choice)

    if action:
        action[1]()
    else:
        print("无效选择")


def _run_file_mode(rag_system):
    """运行文件模式"""
    file_path = input("请输入查询文件路径: ").strip()
    if os.path.exists(file_path):
        rag_system.run_with_file(file_path)
    else:
        print(f"文件不存在: {file_path}")


def _run_default_test(rag_system):
    """运行默认测试"""
    default_file = "./data/truth_query.json"
    if os.path.exists(default_file):
        rag_system.run_with_file(default_file)
    else:
        print(f"默认测试文件不存在: {default_file}")


if __name__ == "__main__":
    main()