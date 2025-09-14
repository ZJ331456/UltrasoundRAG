#!/usr/bin/env python3
"""
UltrasoundRAG检索模块 - 重构后的模块化架构
提供完整的多模态检索功能和多数据库支持

新架构特点：
- 模块化设计，四种检索方式独立
- Caption检索图片独立模块
- 多数据库、多集合支持
- 配置驱动的灵活系统
- 完整的融合检索策略

主要组件：
- ModularRetrievers: 四种独立检索器 (T2T, T2I, I2T, I2I)
- CaptionToImageRetriever: Caption检索图片
- MultiDatabaseManager: 多数据库管理
- FusionStrategies: 融合检索策略
"""

# === 新的模块化检索器 (推荐使用) ===
from .modular_retrievers import (
    T2TRetriever,
    T2IRetriever, 
    I2TRetriever,
    I2IRetriever,
    EnhancedMultimodalRetriever,
    RetrievalContext,
    create_t2t_retriever,
    create_t2i_retriever,
    create_i2t_retriever,
    create_i2i_retriever,
    create_enhanced_multimodal_retriever,
    create_retrieval_context
)

# Caption检索图片模块
from .caption_to_image_retriever import (
    CaptionToImageRetriever,
    create_caption_retriever
)

# 图片标题匹配功能已整合到caption_to_image_retriever中

# 多数据库管理
from .multi_database_manager import (
    MultiDatabaseManager,
    DatabaseConfig,
    DatabaseType,
    RetrievalStrategy,
    create_multi_database_manager,
    create_retrieval_strategy
)

# === 统一检索器 (已删除) ===
# 注意：unified_retriever.py 已删除，功能已整合到 fusion_strategy_retrieval.py

# 融合检索策略
from .fusion_strategy_retrieval import (
    FusionRetrievalManager,
    FusionConfig,
    T2TFusionStrategy,
    ResultFusionManager,
    create_fusion_retrieval_manager
)


# 数据结构
from .data_structures import RetrievalResult

# 重排序
from .enhanced_reranker import (
    SimpleRerankManager,
    RerankConfig,
    RerankResult,
    create_rerank_manager
)

__all__ = [
    # === 新的模块化检索器 (推荐使用) ===
    'T2TRetriever',
    'T2IRetriever',
    'I2TRetriever', 
    'I2IRetriever',
    'EnhancedMultimodalRetriever',
    'RetrievalContext',
    'create_t2t_retriever',
    'create_t2i_retriever',
    'create_i2t_retriever',
    'create_i2i_retriever',
    'create_enhanced_multimodal_retriever',
    'create_retrieval_context',
    
    # Caption检索图片
    'CaptionToImageRetriever',
    'create_caption_retriever',
    
    # 多数据库管理
    'MultiDatabaseManager',
    'DatabaseConfig',
    'DatabaseType', 
    'RetrievalStrategy',
    'create_multi_database_manager',
    'create_retrieval_strategy',
    
    # === 向后兼容接口 (已删除) ===
    # 注意：unified_retriever.py 已删除，请使用 fusion_strategy_retrieval.py
    
    # 融合检索策略 (高级功能)
    'FusionRetrievalManager',
    'FusionConfig',
    'T2TFusionStrategy',
    'ResultFusionManager',
    'create_fusion_retrieval_manager',
    
    # 数据结构
    'RetrievalResult',
    
    # 重排序
    'SimpleRerankManager',
    'RerankConfig',
    'RerankResult',
    'create_rerank_manager',
    
    # === 已弃用 (向后兼容) ===
    # 注意：retrival_util.py 中的类已迁移到 modular_retrievers.py
]

# 版本信息
__version__ = "3.0.0"
__author__ = "UltrasoundRAG Team"
__description__ = "模块化多模态检索系统，支持多数据库"

# 使用指南
USAGE_GUIDE = """
UltrasoundRAG检索系统使用指南 v3.0

=== 推荐的模块化使用方式 ===

1. 独立检索器（最推荐）：
   # 四种检索方式完全独立
   t2t = create_t2t_retriever(db_name="default", top_k=10)
   t2i = create_t2i_retriever(db_name="default", top_k=5)
   i2t = create_i2t_retriever(db_name="default", top_k=10)
   i2i = create_i2i_retriever(db_name="default", top_k=5)
   
   result = t2t.search("心脏超声检查", strategy="fusion")

2. Caption检索图片（独立功能）：
   caption_retriever = create_caption_retriever(search_mode="hybrid_match")
   result = caption_retriever.search_single_caption("图2-3 心脏横切面")
   text_result = caption_retriever.search_from_text_chunk(text_chunk)

3. 多数据库管理（高级功能）：
   manager = create_multi_database_manager()
   result = manager.search_single_database("cardiac_db", "t2t", query)
   
   strategy = create_retrieval_strategy(["db1", "db2"], "weighted")
   result = manager.search_multiple_databases(strategy, "t2t", query)

=== 高级融合功能 ===

4. 融合检索策略（推荐用于复杂场景）：
   fusion_manager = create_fusion_retrieval_manager()
   result = fusion_manager.multimodal_search("心脏超声检查", fusion_strategy="weighted")

=== 命令行工具 ===

5. 主入口功能：
   python ultrasoundrag.py --test-modes --test-multidb --test-caption --benchmark

详细文档：ARCHITECTURE_GUIDE.md | QUICK_START.md
"""

def print_usage():
    """打印使用指南"""
    print(USAGE_GUIDE)

# 便捷函数
def quick_start_modular():
    """快速开始函数，返回模块化检索器"""
    return {
        't2t': create_t2t_retriever(),
        't2i': create_t2i_retriever(), 
        'i2t': create_i2t_retriever(),
        'i2i': create_i2i_retriever(),
        'caption': create_caption_retriever(),
        'manager': create_multi_database_manager()
    }

def quick_start():
    """快速开始函数，返回融合检索管理器（推荐）"""
    return create_fusion_retrieval_manager()