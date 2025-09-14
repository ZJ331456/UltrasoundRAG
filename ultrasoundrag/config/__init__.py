#!/usr/bin/env python3
"""
UltrasoundRAG配置模块
包含配置管理功能
"""

# 兼容性导入 - 保持原有接口
from .config import (
    config,
    config_manager,
    ConfigDict,
    # 便捷的配置访问函数
    get_milvus_config,
    get_embedding_config,
    get_llm_providers,
    get_embedding_providers,
    get_indexing_config,
    get_retriever_config,
    get_generator_config,
    get_rerank_config,
    get_evaluator_config,
    get_prompt_engineer_config,
)

# 新的增强配置管理器
from .config_manager import (
    enhanced_config_manager,
    ConfigEnvironment,
    ConfigValidationRule,
    get_config,
    get_config_value,
    set_config_value,
    reload_config,
    validate_config,
    set_environment,
)

__all__ = [
    # 原有兼容接口
    'config',
    'config_manager',
    'ConfigDict',
    'get_milvus_config',
    'get_embedding_config',
    'get_llm_providers',
    'get_embedding_providers',
    'get_indexing_config',
    'get_retriever_config',
    'get_generator_config',
    'get_rerank_config',
    'get_evaluator_config',
    'get_prompt_engineer_config',
    
    # 增强配置管理
    'enhanced_config_manager',
    'ConfigEnvironment',
    'ConfigValidationRule',
    'get_config',
    'get_config_value',
    'set_config_value',
    'reload_config',
    'validate_config',
    'set_environment',
]