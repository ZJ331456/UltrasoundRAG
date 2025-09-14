#!/usr/bin/env python3
"""
MedicalRAG工具模块
包含各种实用工具和功能组件
"""

# 基础工具 - 避免循环导入
from .logger import setup_logger

# 其他工具将按需导入，避免模块初始化时的循环依赖问题

__all__ = [
    # 基础工具
    'setup_logger'
]

# 动态导入函数，避免循环依赖
def __getattr__(name):
    """动态导入模块属性，避免循环导入问题"""
    
    if name == 'embedding_provider':
        from .embedding_utils import embedding_provider
        return embedding_provider
    elif name in ['LocalEmbedding', 'APIEmbedding', 'EmbeddingProvider', 'EmbeddingError', 'APIEmbeddingError']:
        from .embedding_utils import LocalEmbedding, APIEmbedding, EmbeddingProvider, EmbeddingError, APIEmbeddingError
        mapping = {
            'LocalEmbedding': LocalEmbedding,
            'APIEmbedding': APIEmbedding,
            'EmbeddingProvider': EmbeddingProvider,
            'EmbeddingError': EmbeddingError,
            'APIEmbeddingError': APIEmbeddingError
        }
        return mapping[name]
    elif name == 'llm_provider':
        from .llm_utils import llm_provider
        return llm_provider
    elif name in ['LLMProvider', 'APILLMProvider', 'LocalLLMProvider', 'LLMError', 'APIError']:
        from .llm_utils import LLMProvider, APILLMProvider, LocalLLMProvider, LLMError, APIError
        mapping = {
            'LLMProvider': LLMProvider,
            'APILLMProvider': APILLMProvider,
            'LocalLLMProvider': LocalLLMProvider,
            'LLMError': LLMError,
            'APIError': APIError
        }
        return mapping[name]
    elif name in ['RetrievalResult', 'DenseRetriever', 'BM25SparseRetriever']:
        from ultrasoundrag.core.retrieval.data_structures import RetrievalResult
        # 注意：DenseRetriever 和 BM25SparseRetriever 需要从其他模块导入
        mapping = {
            'RetrievalResult': RetrievalResult,
            'DenseRetriever': DenseRetriever,
            'BM25SparseRetriever': BM25SparseRetriever
        }
        return mapping[name]
    elif name == 'HybridSearchEngine':
        # HybridSearchEngine 定义在检索模块中
        from ultrasoundrag.core.retrieval import HybridSearchEngine
        return HybridSearchEngine
    elif name == 'AnswerGenerator':
        from .answer_generator import AnswerGenerator
        return AnswerGenerator
    elif name in ['get_image_embedding', 'load_clip_model', 'preprocess_image']:
        from .image_utils import get_image_embedding, load_clip_model, preprocess_image
        mapping = {
            'get_image_embedding': get_image_embedding,
            'load_clip_model': load_clip_model,
            'preprocess_image': preprocess_image
        }
        return mapping[name]
    elif name == 'force_remove_directory':
        from .file_util import force_remove_directory
        return force_remove_directory
    elif name == 'ImageIDService':
        from .image_id_service import ImageIDService
        return ImageIDService
    elif name == 'ImageMappingGenerator':
        from .image_mapping_generator import ImageMappingGenerator
        return ImageMappingGenerator
    elif name == 'generate_image_mapping':
        from .image_mapping_generator import generate_image_mapping
        return generate_image_mapping
    elif name == 'SmartRetrievalStrategy':
        # 注意：SmartRetrievalStrategy 需要从其他模块导入
        raise AttributeError(f"SmartRetrievalStrategy 已迁移到其他模块")
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
