"""
UltrasoundRAG 命令行接口模块
统一处理命令行参数和操作调度
"""

import argparse
import random
import torch
import numpy as np
from typing import List, Optional

from ..utils import setup_logger
from .indexing import build_markdown_index, build_image_index
from .evaluation import (
    run_retrieval_test, run_multi_database_test, run_caption_test,
    run_retrieval_benchmark, run_enhanced_comparison, run_stress_test
)


def set_seed(seed: int) -> None:
    """设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CLIRunner:
    """命令行运行器"""
    
    def __init__(self):
        self.logger = setup_logger("CLIRunner")
    
    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            description="UltrasoundRAG 索引构建与多种检索系统",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  # 构建索引
  python ultrasoundrag.py --build-md --build-image --recreate
  
  # 运行测试
  python ultrasoundrag.py --test-modes --test-multidb --test-caption
  
  # 性能基准测试
  python ultrasoundrag.py --benchmark --benchmark-count 20
  
  # 压力测试
  python ultrasoundrag.py --stress-test --concurrent-queries 50
            """
        )
        
        # 索引构建相关参数
        indexing_group = parser.add_argument_group('索引构建选项')
        indexing_group.add_argument("--build-md", action="store_true", help="构建 Markdown 索引")
        indexing_group.add_argument("--build-image", action="store_true", help="构建图片索引")
        indexing_group.add_argument("--recreate", action="store_true", help="重建集合（先删除后创建）")
        indexing_group.add_argument("--md-datasets", type=str, nargs='*', default=None, 
                                   help="仅构建指定的 Markdown 数据集名（空则按配置 enabled）")
        indexing_group.add_argument("--image-datasets", type=str, nargs='*', default=None, 
                                   help="仅构建指定的图片数据集名（空则按配置 enabled）")
        
        # 检索测试相关参数
        retrieval_group = parser.add_argument_group('检索测试选项')
        retrieval_group.add_argument("--test", action="store_true", help="执行基础检索测试")
        retrieval_group.add_argument("--test-modes", action="store_true", help="测试四种检索模式 (T2T, T2I, I2T, I2I)")
        retrieval_group.add_argument("--test-multidb", action="store_true", help="测试多数据库检索功能")
        retrieval_group.add_argument("--multi-db-keys", type=str, nargs='*', help="多数据库键列表")
        retrieval_group.add_argument("--test-caption", action="store_true", help="测试Caption检索图片功能")
        
        # 基准测试相关参数
        benchmark_group = parser.add_argument_group('基准测试选项')
        benchmark_group.add_argument("--benchmark", action="store_true", help="执行检索性能基准测试")
        benchmark_group.add_argument("--enhanced-comparison", action="store_true", help="增强功能性能对比测试")
        benchmark_group.add_argument("--stress-test", action="store_true", help="执行压力测试")
        benchmark_group.add_argument("--benchmark-queries", type=str, nargs='*', help="自定义基准测试查询列表")
        benchmark_group.add_argument("--benchmark-count", type=int, default=10, help="基准测试查询数量，默认 10")
        benchmark_group.add_argument("--concurrent-queries", type=int, default=10, help="压力测试并发查询数，默认 10")
        
        # 通用参数
        general_group = parser.add_argument_group('通用选项')
        general_group.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
        general_group.add_argument("--top-k", type=int, default=3, help="测试返回 TopK，默认 3")
        general_group.add_argument("--db-name", type=str, default="default", help="使用的数据库名称，默认 'default'")
        general_group.add_argument("--query", type=str, help="自定义查询文本（用于测试）")
        general_group.add_argument("--image-path", type=str, help="图片路径（用于I2T和I2I测试）")
        
        return parser
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """运行CLI程序"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            set_seed(parsed_args.seed)
            
            # 确定是否需要构建索引
            has_build_flags = parsed_args.build_md or parsed_args.build_image
            has_test_flags = (parsed_args.test or parsed_args.test_modes or parsed_args.test_multidb or 
                             parsed_args.test_caption)
            has_benchmark_flags = (parsed_args.benchmark or parsed_args.enhanced_comparison or 
                                 parsed_args.stress_test)
            
            # 如果没有任何标志，默认执行构建
            if not has_build_flags and not has_test_flags and not has_benchmark_flags:
                has_build_flags = True
                parsed_args.build_md = True
                parsed_args.build_image = True
            
            # 执行索引构建
            if has_build_flags:
                self.logger.info("开始索引构建...")
                self._run_index_building(parsed_args)
            
            # 执行检索测试
            if has_test_flags:
                self.logger.info("开始检索测试...")
                self._run_retrieval_tests(parsed_args)
            
            # 执行基准测试
            if has_benchmark_flags:
                self.logger.info("开始基准测试...")
                self._run_benchmark_tests(parsed_args)
            
            self.logger.info("=" * 60)
            self.logger.info("UltrasoundRAG 系统运行完成！")
            self.logger.info("=" * 60)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"系统运行出错: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _run_index_building(self, args) -> None:
        """运行索引构建"""
        if args.build_md or (not args.build_image):
            self.logger.info("构建 Markdown 索引...")
            md_result = build_markdown_index(recreate=args.recreate, only_datasets=args.md_datasets)
            self._log_build_result("Markdown", md_result)
        
        if args.build_image or (not args.build_md):
            self.logger.info("构建图片索引...")
            img_result = build_image_index(recreate=args.recreate, only_datasets=args.image_datasets)
            self._log_build_result("图片", img_result)
    
    def _run_retrieval_tests(self, args) -> None:
        """运行检索测试"""
        # 基础检索测试
        if args.test or args.test_modes:
            self.logger.info("运行检索模式测试...")
            test_result = run_retrieval_test(db_name=args.db_name, top_k=args.top_k)
            self._log_test_result("检索模式测试", test_result)
        
        # 多数据库检索测试
        if args.test_multidb:
            query = args.query or "心脏超声诊断"
            self.logger.info("运行多数据库检索测试...")
            
            if args.multi_db_keys and len(args.multi_db_keys) >= 2:
                # 多数据库融合示例
                self.logger.info(f"使用多数据库融合: {args.multi_db_keys}")
                try:
                    from ultrasoundrag.core.retrieval.multi_database_manager import create_multi_database_manager, create_retrieval_strategy
                    manager = create_multi_database_manager()
                    strategy = create_retrieval_strategy(args.multi_db_keys, aggregation_method="weighted", max_results=args.top_k)
                    result = manager.search_multiple_databases(strategy, "t2t", query, top_k=args.top_k)
                    self.logger.info(f"融合结果数: {result.get('total_results', 0)}")
                except Exception as e:
                    self.logger.error(f"多数据库融合失败: {e}")
            else:
                multidb_result = run_multi_database_test(query=query, top_k=args.top_k)
                self._log_test_result("多数据库检索测试", multidb_result)
        
        # Caption检索测试
        if args.test_caption:
            self.logger.info("运行Caption检索测试...")
            caption_result = run_caption_test(db_name=args.db_name, top_k=args.top_k)
            self._log_test_result("Caption检索测试", caption_result)
    
    def _run_benchmark_tests(self, args) -> None:
        """运行基准测试"""
        # 基础性能基准测试
        if args.benchmark:
            queries = args.benchmark_queries
            if not queries:
                queries = [
                    "心脏超声检查方法", "肝脏病变诊断", "胎儿发育评估", "血管多普勒检查", "肾脏结石诊断",
                    "超声引导穿刺", "腹部超声检查", "妇科超声诊断", "甲状腺超声检查", "乳腺超声检查"
                ]
            
            # 限制查询数量
            queries = queries[:args.benchmark_count]
            self.logger.info(f"运行基准测试 ({len(queries)} 个查询)...")
            
            benchmark_result = run_retrieval_benchmark(
                queries=queries, 
                top_k=args.top_k, 
                db_name=args.db_name
            )
            self._log_benchmark_result("基准测试", benchmark_result)
        
        # 增强功能对比测试
        if args.enhanced_comparison:
            queries = args.benchmark_queries or [
                "心脏超声检查方法", "肝脏病变诊断", "胎儿发育评估", "血管多普勒检查", "肾脏结石诊断"
            ]
            queries = queries[:5]  # 限制5个查询
            
            self.logger.info("运行增强功能对比测试...")
            comparison_result = run_enhanced_comparison(
                queries=queries,
                db_name=args.db_name,
                top_k=args.top_k
            )
            self._log_benchmark_result("增强对比测试", comparison_result)
        
        # 压力测试
        if args.stress_test:
            query = args.query or "心脏超声检查"
            self.logger.info(f"运行压力测试 (并发: {args.concurrent_queries})...")
            
            stress_result = run_stress_test(
                concurrent_queries=args.concurrent_queries,
                query=query,
                db_name=args.db_name,
                top_k=args.top_k
            )
            self._log_stress_result("压力测试", stress_result)
    
    def _log_build_result(self, build_type: str, result: dict) -> None:
        """记录构建结果"""
        if result:
            self.logger.info(f"{build_type}构建完成: 成功 {result.get('built_count', 0)}, 跳过 {result.get('skipped_count', 0)}")
        else:
            self.logger.error(f"{build_type}构建失败")
    
    def _log_test_result(self, test_type: str, result: dict) -> None:
        """记录测试结果"""
        if result.get('success', False):
            summary = result.get('summary', {})
            total_tests = summary.get('total_tests', 0)
            successful_tests = summary.get('successful_tests', 0)
            self.logger.info(f"{test_type}完成: 成功 {successful_tests}/{total_tests}")
        else:
            self.logger.error(f"{test_type}失败: {result.get('error', '未知错误')}")
    
    def _log_benchmark_result(self, benchmark_type: str, result: dict) -> None:
        """记录基准测试结果"""
        if result:
            summary = result.get('summary', {})
            self.logger.info(f"{benchmark_type}完成:")
            self.logger.info(f"  总查询数: {result.get('total_queries', 0)}")
            self.logger.info(f"  最快检索器: {summary.get('fastest_retriever', 'N/A')}")
            self.logger.info(f"  平均响应时间: {summary.get('overall_avg_time', 0):.3f}s")
        else:
            self.logger.error(f"{benchmark_type}失败")
    
    def _log_stress_result(self, test_type: str, result: dict) -> None:
        """记录压力测试结果"""
        if result:
            self.logger.info(f"{test_type}完成:")
            self.logger.info(f"  成功率: {result.get('success_rate', 0):.1%}")
            self.logger.info(f"  吞吐量: {result.get('queries_per_second', 0):.1f} 查询/秒")
            self.logger.info(f"  平均响应时间: {result.get('avg_response_time', 0):.3f}s")
        else:
            self.logger.error(f"{test_type}失败")


def main(args: Optional[List[str]] = None) -> int:
    """主函数"""
    runner = CLIRunner()
    return runner.run(args)


if __name__ == "__main__":
    exit(main())
