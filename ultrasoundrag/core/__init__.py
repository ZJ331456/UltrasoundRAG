"""
UltrasoundRAG 核心功能模块

包含系统的核心业务逻辑：
- indexing: 索引构建和管理
- retrieval: 检索引擎和算法
- evaluation: 测试和评估
- generation: 内容生成

这些模块提供了系统的主要功能，同时保持了清晰的职责分离。
"""

# 索引相关
from .indexing import (
    IndexBuilder,
    build_markdown_index,
    build_image_index,
    build_all_indexes,
    update_documents_incremental,
    update_single_document
)

# 检索相关
from .retrieval import (
    EnhancedMultimodalRetriever,
    create_enhanced_multimodal_retriever,
    create_t2t_retriever,
    create_t2i_retriever,
    create_i2t_retriever,
    create_i2i_retriever,
    create_caption_retriever,
    create_multi_database_manager
)

# 查询处理
from .query import (
    QueryProcessor,
    QueryTextProcessor,
    TermWeightCalculator,
    SynonymLookup,
    MatchTextExpr,
    create_query_processor,
    create_text_processor,
    create_term_weight_calculator,
    create_synonym_lookup,
    create_fulltext_queryer  # 向后兼容
)

# 评估测试相关
from .evaluation import (
    TestRunner,
    BenchmarkRunner,
    run_retrieval_test,
    run_multi_database_test,
    run_caption_test,
    run_retrieval_benchmark,
    run_enhanced_comparison,
    run_stress_test
)

# 生成相关 - 暂时移除，模块结构需要重构
# from .generation import (
#     ContentGenerator,
#     generate_response,
#     generate_summary
# )

# CLI接口
from .cli_runner import (
    CLIRunner,
    main
)

__all__ = [
    # 索引构建
    'IndexBuilder',
    'build_markdown_index', 
    'build_image_index',
    'build_all_indexes',
    'update_documents_incremental',
    'update_single_document',
    
    # 检索相关
    'EnhancedMultimodalRetriever',
    'create_enhanced_multimodal_retriever',
    'create_t2t_retriever',
    'create_t2i_retriever',
    'create_i2t_retriever',
    'create_i2i_retriever',
    'create_caption_retriever',
    'create_multi_database_manager',
    
    # 查询处理
    'QueryProcessor',
    'QueryTextProcessor',
    'TermWeightCalculator',
    'SynonymLookup',
    'MatchTextExpr',
    'create_query_processor',
    'create_text_processor',
    'create_term_weight_calculator',
    'create_synonym_lookup',
    'create_fulltext_queryer',  # 向后兼容
    
    # 测试评估
    'TestRunner',
    'BenchmarkRunner',
    'run_retrieval_test',
    'run_multi_database_test',
    'run_caption_test',
    'run_retrieval_benchmark',
    'run_enhanced_comparison',
    'run_stress_test',
    
    # 内容生成 - 暂时移除
    # 'ContentGenerator',
    # 'generate_response', 
    # 'generate_summary',
    
    # CLI
    'CLIRunner',
    'main'
]
