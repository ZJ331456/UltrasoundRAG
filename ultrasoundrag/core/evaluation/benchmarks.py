"""
UltrasoundRAG 基准测试模块
集中处理性能测试和基准对比功能
"""

import time
from typing import List, Dict, Any, Optional

from ultrasoundrag.core.retrieval.modular_retrievers import (
    create_t2t_retriever, create_t2i_retriever,
    create_enhanced_multimodal_retriever
)
from ultrasoundrag.utils.logger import setup_logger


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.logger = setup_logger("BenchmarkRunner")
    
    def run_retrieval_performance_benchmark(self, queries: Optional[List[str]] = None, 
                                          top_k: int = 10, db_name: str = "default") -> dict:
        """检索性能基准测试"""
        self.logger.info("=" * 60)
        self.logger.info("开始检索性能基准测试")
        self.logger.info("=" * 60)
        
        if not queries:
            queries = self._get_default_test_queries()
        
        retriever_types = ["t2t", "t2i"]
        results = {}
        
        for retriever_type in retriever_types:
            self.logger.info(f"测试 {retriever_type.upper()} 检索性能:")
            self.logger.info("-" * 40)
            
            # 创建检索器
            retriever = self._create_retriever(retriever_type, db_name, top_k)
            if not retriever:
                continue
            
            # 执行基准测试
            type_results = self._benchmark_retriever(retriever, queries, top_k, retriever_type)
            results[retriever_type] = type_results
            
            # 打印统计信息
            self._print_retriever_stats(retriever_type, type_results)
        
        # 性能对比
        self._print_performance_comparison(results)
        
        return {
            'benchmark_time': time.time(),
            'total_queries': len(queries),
            'db_name': db_name,
            'top_k': top_k,
            'results': results,
            'summary': self._generate_summary(results)
        }
    
    def run_enhanced_performance_comparison(self, queries: Optional[List[str]] = None,
                                          db_name: str = "default", top_k: int = 10) -> dict:
        """对比增强前后的性能"""
        self.logger.info("=" * 80)
        self.logger.info("⚡ UltrasoundRAG 增强性能对比测试")
        self.logger.info("=" * 80)
        
        if not queries:
            queries = self._get_default_test_queries()[:5]  # 只测试前5个查询
        
        results = {
            'traditional': {'total_time': 0, 'avg_time': 0, 'results_count': 0, 'tests': []},
            'enhanced': {'total_time': 0, 'avg_time': 0, 'results_count': 0, 'tests': []}
        }
        
        for i, query in enumerate(queries):
            self.logger.info(f"🔍 查询 {i+1}: '{query[:30]}...'")
            
            # 传统T2T检索
            traditional_result = self._test_traditional_retrieval(query, db_name, top_k)
            if traditional_result['success']:
                results['traditional']['total_time'] += traditional_result['duration']
                results['traditional']['results_count'] += traditional_result['result_count']
                results['traditional']['tests'].append(traditional_result)
                self.logger.info(f"   传统检索: {traditional_result['duration']:.3f}s, {traditional_result['result_count']} 结果")
            
            # 增强多模态检索
            enhanced_result = self._test_enhanced_retrieval(query, db_name, top_k)
            if enhanced_result['success']:
                results['enhanced']['total_time'] += enhanced_result['duration']
                results['enhanced']['results_count'] += enhanced_result['result_count']
                results['enhanced']['tests'].append(enhanced_result)
                self.logger.info(f"   增强检索: {enhanced_result['duration']:.3f}s, {enhanced_result['result_count']} 结果")
                self.logger.info(f"   使用模式: {enhanced_result.get('retrieval_type', 'unknown')}")
        
        # 计算平均性能
        num_queries = len(queries)
        if num_queries > 0:
            results['traditional']['avg_time'] = results['traditional']['total_time'] / num_queries
            results['enhanced']['avg_time'] = results['enhanced']['total_time'] / num_queries
            
            self._print_enhanced_comparison_summary(results)
        
        return {
            'comparison_time': time.time(),
            'total_queries': num_queries,
            'results': results,
            'performance_improvement': self._calculate_improvement(results)
        }
    
    def run_stress_test(self, concurrent_queries: int = 10, query: str = "心脏超声检查",
                       db_name: str = "default", top_k: int = 5) -> dict:
        """压力测试"""
        self.logger.info("=" * 60)
        self.logger.info(f"开始压力测试 - 并发查询数: {concurrent_queries}")
        self.logger.info("=" * 60)
        
        retriever = create_t2t_retriever(db_name, top_k)
        
        start_time = time.time()
        successful_queries = 0
        failed_queries = 0
        response_times = []
        
        for i in range(concurrent_queries):
            query_start = time.time()
            try:
                result = retriever.search(f"{query} {i}", top_k=top_k)
                query_duration = time.time() - query_start
                response_times.append(query_duration)
                successful_queries += 1
                
                if i % 10 == 0:
                    self.logger.info(f"完成查询 {i+1}/{concurrent_queries}")
                    
            except Exception as e:
                self.logger.error(f"查询 {i} 失败: {e}")
                failed_queries += 1
        
        total_duration = time.time() - start_time
        
        stress_results = {
            'total_duration': total_duration,
            'concurrent_queries': concurrent_queries,
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'queries_per_second': successful_queries / total_duration if total_duration > 0 else 0,
            'success_rate': successful_queries / concurrent_queries if concurrent_queries > 0 else 0
        }
        
        self._print_stress_test_results(stress_results)
        
        return stress_results
    
    def _create_retriever(self, retriever_type: str, db_name: str, top_k: int):
        """创建检索器"""
        try:
            if retriever_type == "t2t":
                return create_t2t_retriever(db_name, top_k)
            elif retriever_type == "t2i":
                return create_t2i_retriever(db_name, top_k)
            else:
                self.logger.error(f"不支持的检索器类型: {retriever_type}")
                return None
        except Exception as e:
            self.logger.error(f"创建检索器失败 {retriever_type}: {e}")
            return None
    
    def _benchmark_retriever(self, retriever, queries: List[str], top_k: int, 
                           retriever_type: str) -> dict:
        """基准测试单个检索器"""
        times = []
        total_results = []
        successful_queries = 0
        
        for i, query in enumerate(queries):
            start_time = time.time()
            
            try:
                result = retriever.search(query, top_k=top_k)
                response_time = time.time() - start_time
                
                times.append(response_time)
                total_results.append(result.get('total_results', 0))
                successful_queries += 1
                
                self.logger.info(f"  查询 {i+1}: {response_time:.3f}s, 结果数: {result.get('total_results', 0)}")
                
            except Exception as e:
                self.logger.error(f"查询 '{query}' 失败: {e}")
                times.append(0)
                total_results.append(0)
        
        # 计算统计信息
        valid_times = [t for t in times if t > 0]
        return {
            'avg_time': sum(valid_times) / len(valid_times) if valid_times else 0,
            'max_time': max(valid_times) if valid_times else 0,
            'min_time': min(valid_times) if valid_times else 0,
            'avg_results': sum(total_results) / len(total_results) if total_results else 0,
            'total_queries': len(queries),
            'successful_queries': successful_queries,
            'success_rate': successful_queries / len(queries) if queries else 0,
            'all_times': times,
            'all_results': total_results
        }
    
    def _test_traditional_retrieval(self, query: str, db_name: str, top_k: int) -> dict:
        """测试传统检索"""
        try:
            start_time = time.time()
            traditional_retriever = create_t2t_retriever(db_name, top_k)
            result = traditional_retriever.search(query, top_k=top_k)
            duration = time.time() - start_time
            
            return {
                'success': True,
                'duration': duration,
                'result_count': result.get('total_results', 0),
                'query': query
            }
        except Exception as e:
            self.logger.error(f"传统检索失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_enhanced_retrieval(self, query: str, db_name: str, top_k: int) -> dict:
        """测试增强检索"""
        try:
            start_time = time.time()
            enhanced_retriever = create_enhanced_multimodal_retriever(
                db_name=db_name, 
                top_k=top_k,
                enable_dynamic_weights=True,
                enable_domain_partition=True,
                enable_multimodal_rerank=True
            )
            result = enhanced_retriever.search(query, mode="auto", top_k=top_k)
            duration = time.time() - start_time
            
            return {
                'success': True,
                'duration': duration,
                'result_count': result.get('total_results', 0),
                'retrieval_type': result.get('retrieval_type', 'unknown'),
                'fusion_strategy': result.get('fusion_strategy', 'unknown'),
                'query': query
            }
        except Exception as e:
            self.logger.error(f"增强检索失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_default_test_queries(self) -> List[str]:
        """获取默认测试查询"""
        return [
            "心脏超声检查方法",
            "肝脏病变诊断",
            "胎儿发育评估", 
            "血管多普勒检查",
            "肾脏结石诊断",
            "超声引导穿刺",
            "腹部超声检查",
            "妇科超声诊断",
            "甲状腺超声检查",
            "乳腺超声检查"
        ]
    
    def _print_retriever_stats(self, retriever_type: str, stats: dict) -> None:
        """打印检索器统计信息"""
        self.logger.info(f"  平均响应时间: {stats['avg_time']:.3f}s")
        self.logger.info(f"  最长响应时间: {stats['max_time']:.3f}s")
        self.logger.info(f"  最短响应时间: {stats['min_time']:.3f}s")
        self.logger.info(f"  平均结果数: {stats['avg_results']:.1f}")
        self.logger.info(f"  成功率: {stats['success_rate']:.1%}")
    
    def _print_performance_comparison(self, results: dict) -> None:
        """打印性能对比"""
        self.logger.info("性能对比总结:")
        self.logger.info("-" * 40)
        for retriever_type, stats in results.items():
            self.logger.info(f"{retriever_type.upper()}检索:")
            self.logger.info(f"  平均响应时间: {stats['avg_time']:.3f}s")
            self.logger.info(f"  平均结果数: {stats['avg_results']:.1f}")
            self.logger.info(f"  查询吞吐量: {1/stats['avg_time']:.1f} 查询/秒")
            self.logger.info("")
    
    def _print_enhanced_comparison_summary(self, results: dict) -> None:
        """打印增强对比总结"""
        traditional = results['traditional']
        enhanced = results['enhanced']
        
        self.logger.info("📊 性能对比总结:")
        self.logger.info(f"   传统检索: 平均 {traditional['avg_time']:.3f}s/查询")
        self.logger.info(f"   增强检索: 平均 {enhanced['avg_time']:.3f}s/查询")
        
        if traditional['avg_time'] > 0:
            improvement = (traditional['avg_time'] - enhanced['avg_time']) / traditional['avg_time'] * 100
            if improvement > 0:
                self.logger.info(f"   ⬆️  性能提升: {improvement:.1f}%")
            else:
                self.logger.info(f"   ⬇️  性能开销: {abs(improvement):.1f}%")
    
    def _print_stress_test_results(self, results: dict) -> None:
        """打印压力测试结果"""
        self.logger.info("压力测试结果:")
        self.logger.info("-" * 40)
        self.logger.info(f"总耗时: {results['total_duration']:.3f}s")
        self.logger.info(f"成功查询: {results['successful_queries']}")
        self.logger.info(f"失败查询: {results['failed_queries']}")
        self.logger.info(f"成功率: {results['success_rate']:.1%}")
        self.logger.info(f"吞吐量: {results['queries_per_second']:.1f} 查询/秒")
        self.logger.info(f"平均响应时间: {results['avg_response_time']:.3f}s")
        self.logger.info(f"最大响应时间: {results['max_response_time']:.3f}s")
        self.logger.info(f"最小响应时间: {results['min_response_time']:.3f}s")
    
    def _generate_summary(self, results: dict) -> dict:
        """生成测试总结"""
        summary = {
            'total_retrievers_tested': len(results),
            'fastest_retriever': None,
            'slowest_retriever': None,
            'most_accurate_retriever': None,
            'overall_avg_time': 0
        }
        
        if results:
            # 找出最快和最慢的检索器
            avg_times = {k: v['avg_time'] for k, v in results.items() if v['avg_time'] > 0}
            if avg_times:
                summary['fastest_retriever'] = min(avg_times, key=avg_times.get)
                summary['slowest_retriever'] = max(avg_times, key=avg_times.get)
                summary['overall_avg_time'] = sum(avg_times.values()) / len(avg_times)
            
            # 找出最准确的检索器（平均结果数最多）
            avg_results = {k: v['avg_results'] for k, v in results.items()}
            if avg_results:
                summary['most_accurate_retriever'] = max(avg_results, key=avg_results.get)
        
        return summary
    
    def _calculate_improvement(self, results: dict) -> dict:
        """计算性能提升"""
        traditional = results['traditional']
        enhanced = results['enhanced']
        
        improvement = {}
        
        if traditional['avg_time'] > 0:
            time_improvement = (traditional['avg_time'] - enhanced['avg_time']) / traditional['avg_time'] * 100
            improvement['time_improvement_percent'] = time_improvement
        
        improvement['traditional_avg_time'] = traditional['avg_time']
        improvement['enhanced_avg_time'] = enhanced['avg_time']
        improvement['traditional_avg_results'] = traditional['results_count'] / len(traditional['tests']) if traditional['tests'] else 0
        improvement['enhanced_avg_results'] = enhanced['results_count'] / len(enhanced['tests']) if enhanced['tests'] else 0
        
        return improvement


# 便捷函数
def run_retrieval_benchmark(queries: Optional[List[str]] = None, top_k: int = 10, 
                          db_name: str = "default") -> dict:
    """运行检索基准测试的便捷函数"""
    runner = BenchmarkRunner()
    return runner.run_retrieval_performance_benchmark(queries, top_k, db_name)


def run_enhanced_comparison(queries: Optional[List[str]] = None, db_name: str = "default", 
                          top_k: int = 10) -> dict:
    """运行增强对比测试的便捷函数"""
    runner = BenchmarkRunner()
    return runner.run_enhanced_performance_comparison(queries, db_name, top_k)


def run_stress_test(concurrent_queries: int = 10, query: str = "心脏超声检查",
                   db_name: str = "default", top_k: int = 5) -> dict:
    """运行压力测试的便捷函数"""
    runner = BenchmarkRunner()
    return runner.run_stress_test(concurrent_queries, query, db_name, top_k)
