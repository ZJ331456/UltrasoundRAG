"""
评估和测试模块

负责系统的性能评估、基准测试和功能测试：
- runners: 测试执行器
- benchmarks: 基准测试  
- metrics: 评估指标
"""

from .runners import (
    TestRunner,
    run_retrieval_test,
    run_multi_database_test,
    run_caption_test
)

from .benchmarks import (
    BenchmarkRunner,
    run_retrieval_benchmark,
    run_enhanced_comparison,
    run_stress_test
)

__all__ = [
    'TestRunner',
    'run_retrieval_test',
    'run_multi_database_test',
    'run_caption_test',
    'BenchmarkRunner',
    'run_retrieval_benchmark',
    'run_enhanced_comparison',
    'run_stress_test'
]
