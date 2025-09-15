#!/usr/bin/env python3
"""
UltrasoundRAG 主入口文件 - 重构版本

这是重构后的简洁入口点，提供：
1. 命令行接口
2. 向后兼容的函数
3. 快速开始功能

v3.0 重构特性：
- 标准化的Python包结构
- 清晰的模块职责分离
- 统一的API接口
- 增强的配置管理
- 完整的测试框架
"""

import sys
from typing import List, Optional

# 主要API导入
from .core import (
    build_markdown_index, build_image_index, build_all_indexes,
    run_retrieval_test, run_multi_database_test, run_caption_test,
    run_retrieval_benchmark, run_enhanced_comparison, run_stress_test
)
from .cli import main as cli_main

# 版本信息
__version__ = "3.0.0"
__description__ = "UltrasoundRAG - 企业级医学超声RAG系统"
__author__ = "UltrasoundRAG Team"


# === 向后兼容的便捷接口 ===

def test_retrieval_modes(db_name: str = "default", top_k: int = 3):
    """测试检索模式 - 兼容接口"""
    return run_retrieval_test(db_name, top_k)


def test_multi_database_retrieval(query: str = "心脏超声诊断", top_k: int = 5):
    """测试多数据库检索 - 兼容接口"""
    return run_multi_database_test(query, top_k)


def benchmark_retrieval_performance(queries: list = None, top_k: int = 10, db_name: str = "default"):
    """检索性能基准测试 - 兼容接口"""
    return run_retrieval_benchmark(queries, top_k, db_name)


def test_search(top_k: int = 3):
    """兼容性测试搜索功能"""
    print("=" * 50)
    print("开始基础搜索功能测试")
    
    # 调用新的检索测试
    result = test_retrieval_modes("default", top_k)
    
    print("\n" + "=" * 50)
    print("基础搜索功能测试完成")
    return result


def test_enhanced_features(query: str = "心脏超声图像", db_name: str = "default", top_k: int = 5):
    """测试所有增强功能 - 兼容接口"""
    try:
        # 使用检索测试功能
        result = run_retrieval_test(db_name=db_name, top_k=top_k)
        
        print(f"\n✅ 增强功能测试完成!")
        print(f"   查询: '{query}'")
        print(f"   测试结果: {result}")
        
        return {'success': True, 'result': result}
        
    except Exception as e:
        print(f"❌ 增强功能测试失败: {e}")
        return {'success': False, 'error': str(e)}


def benchmark_enhanced_performance(queries: list = None, db_name: str = "default", top_k: int = 10):
    """对比增强前后的性能 - 兼容接口"""
    return run_enhanced_comparison(queries, db_name, top_k)


def quick_demo():
    """快速演示所有增强功能 - 兼容接口"""
    print("\n" + "🌟" * 20)
    print("UltrasoundRAG 增强功能快速演示")
    print("🌟" * 20)
    
    demo_queries = [
        "心脏超声图像分析",
        "胎儿发育异常检查", 
        "肝脏超声诊断要点",
        "图2-3显示的病变特征"
    ]
    
    import time
    for query in demo_queries:
        print(f"\n🔍 演示查询: '{query}'")
        test_enhanced_features(query, top_k=3)
        time.sleep(1)  # 短暂停顿以便观察


# === 快速开始功能 ===

def quick_start():
    """快速开始功能"""
    print(f"\n🚀 {__description__} v{__version__}")
    print("=" * 60)
    
    # 演示基本功能
    try:
        # 运行检索测试
        result = run_retrieval_test(db_name="default", top_k=3)
        
        print(f"✅ 快速搜索成功，测试结果: {result}")
        
    except Exception as e:
        print(f"❌ 快速搜索失败: {e}")
        print("💡 请先运行重建命令构建索引")
        print("💡 重建命令: python -c \"from ultrasoundrag.core.indexing import build_all_indexes; build_all_indexes(recreate=True)\"")


def main(args: Optional[List[str]] = None):
    """主入口函数"""
    if args is None:
        args = sys.argv[1:]
    
    # 如果没有参数，显示快速开始
    if not args:
        quick_start()
        return 0
    
    # 否则使用CLI处理
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
