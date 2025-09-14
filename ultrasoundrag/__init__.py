#!/usr/bin/env python3
"""
UltrasoundRAG - 企业级医学超声RAG系统

这是重构后的v3.0版本，采用标准化的Python包结构，提供：
- 清晰的模块化架构
- 统一的API接口
- 增强的配置管理
- 完整的测试框架

主要特性：
1. 多模态检索（T2T、T2I、I2T、I2I）
2. 智能融合策略
3. 领域分区和路由
4. 动态权重调整
5. Caption增强检索
6. 多数据库支持
"""

# 统一设置 jieba 缓存目录到项目根目录，避免 /tmp 权限问题
import os
from pathlib import Path
import tempfile

try:
    # 将缓存统一放到项目根目录
    ROOT_DIR = Path(__file__).resolve().parent.parent
    default_tmp_dir = ROOT_DIR / ".tmp"

    # 1) 重定向临时目录，避免写入系统 /tmp
    os.makedirs(default_tmp_dir, exist_ok=True)
    for var in ("TMPDIR", "TMP", "TEMP"):
        if not os.environ.get(var):
            os.environ[var] = str(default_tmp_dir)

    # 触发tempfile模块重新读取环境变量
    tempfile.tempdir = None
    tempfile.gettempdir()

    # 2) 统一Jieba缓存目录到 .tmp 下
    if not os.environ.get("JIEBA_CACHE_DIR"):
        os.environ["JIEBA_CACHE_DIR"] = str(default_tmp_dir / "jieba_cache")
    os.makedirs(os.environ["JIEBA_CACHE_DIR"], exist_ok=True)
except (OSError, PermissionError) as e:
    # 目录创建权限问题时忽略，使用系统默认
    import warnings
    warnings.warn(f"无法设置自定义缓存目录: {e}，将使用系统默认目录")
except ImportError as e:
    # 某些依赖模块缺失时忽略
    import warnings  
    warnings.warn(f"依赖模块导入失败: {e}，某些功能可能受限")
except Exception as e:
    # 其他环境异常时忽略，不影响主流程
    import warnings
    warnings.warn(f"环境设置异常: {e}，将使用默认配置")

# === 版本信息 ===
__version__ = "3.0.0"
__description__ = "UltrasoundRAG - 企业级医学超声RAG系统"
__author__ = "UltrasoundRAG Team"

# === 主要API导入（精简为Web API为主） ===

# 核心功能模块
from .core import (
    # 索引构建
    IndexBuilder,
    build_markdown_index, 
    build_image_index, 
    build_all_indexes,
    # 文档更新
    update_documents_incremental, 
    update_single_document,
    # 检索相关
    EnhancedMultimodalRetriever,
    create_enhanced_multimodal_retriever,
    # 测试运行
    TestRunner,
    run_retrieval_test, 
    run_multi_database_test, 
    run_caption_test,
    # 基准测试
    BenchmarkRunner,
    run_retrieval_benchmark, 
    run_enhanced_comparison, 
    run_stress_test
)

# 配置管理
from .config import (
    get_config,
    get_config_value, 
    set_config_value,
    reload_config,
    validate_config
)

# 检索引擎 - 已在核心模块中导入
# from .core.retrieval import (
#     create_enhanced_multimodal_retriever
# )

# 工具函数
from .utils import (
    setup_logger
)

__all__ = [
    # 版本信息
    '__version__', 
    '__description__', 
    '__author__',
    
    # 核心功能类
    'IndexBuilder',
    'TestRunner', 
    'BenchmarkRunner',
    'EnhancedMultimodalRetriever',
    
    # 核心功能函数
    'build_markdown_index', 
    'build_image_index', 
    'build_all_indexes',
    'update_documents_incremental', 
    'update_single_document',
    'run_retrieval_test', 
    'run_multi_database_test', 
    'run_caption_test',
    'run_retrieval_benchmark', 
    'run_enhanced_comparison', 
    'run_stress_test',
    'create_enhanced_multimodal_retriever',
    
    # 配置管理
    'get_config',
    'get_config_value', 
    'set_config_value',
    'reload_config',
    'validate_config',
    
    # 工具函数
    'setup_logger',
]

# === 向后兼容的快捷导入 ===

# 为了保持向后兼容性，提供原有的导入路径
def __getattr__(name):
    """动态导入，支持向后兼容"""
    if name == 'config':
        from .config import get_config
        return get_config()
    
    # 如果找不到属性，抛出标准错误
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 移除依赖已废弃程序化API的演示代码
