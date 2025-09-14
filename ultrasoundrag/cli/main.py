"""
UltrasoundRAG 命令行接口主入口

提供标准化的CLI接口来访问系统所有功能。
"""

import sys
import argparse
from typing import List, Optional

from ..core.cli_runner import main as core_main, CLIRunner


def main(args: Optional[List[str]] = None):
    """主CLI入口函数"""
    if args is None:
        args = sys.argv[1:]
    
    return core_main(args)


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog='ultrasoundrag',
        description='UltrasoundRAG - 企业级医学超声RAG系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  ultrasoundrag index --build-all                    # 构建所有索引
  ultrasoundrag search "心脏超声检查"                 # 文本搜索
  ultrasoundrag test --mode all                      # 运行所有测试
  ultrasoundrag benchmark --queries "心脏,肝脏"      # 基准测试
        """
    )
    
    # 版本信息
    parser.add_argument(
        '--version', 
        action='version', 
        version='UltrasoundRAG 3.0.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 索引构建命令
    index_parser = subparsers.add_parser('index', help='索引管理')
    index_parser.add_argument('--build-md', action='store_true', help='构建Markdown索引')
    index_parser.add_argument('--build-img', action='store_true', help='构建图像索引')
    index_parser.add_argument('--build-all', action='store_true', help='构建所有索引')
    index_parser.add_argument('--recreate', action='store_true', help='重新创建索引')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='搜索功能')
    search_parser.add_argument('query', help='搜索查询')
    search_parser.add_argument('--mode', default='auto', help='搜索模式')
    search_parser.add_argument('--top-k', type=int, default=5, help='返回结果数量')
    search_parser.add_argument('--db-name', default='default', help='数据库名称')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='测试功能')
    test_parser.add_argument('--mode', default='all', help='测试模式')
    test_parser.add_argument('--db-name', default='default', help='数据库名称')
    test_parser.add_argument('--top-k', type=int, default=5, help='测试结果数量')
    
    # 基准测试命令
    benchmark_parser = subparsers.add_parser('benchmark', help='基准测试')
    benchmark_parser.add_argument('--queries', help='测试查询列表，逗号分隔')
    benchmark_parser.add_argument('--db-name', default='default', help='数据库名称')
    benchmark_parser.add_argument('--top-k', type=int, default=10, help='测试结果数量')
    
    return parser


if __name__ == '__main__':
    sys.exit(main())
