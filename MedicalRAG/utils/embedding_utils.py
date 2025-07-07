"""
嵌入生成工具模块，提供基于本地模型或API的统一嵌入功能。

主要功能：
- 支持本地模型加载（如 BGE-small-zh-v1.5）和API调用（如 OpenAI）。
- 统一的接口设计，通过配置文件方便地切换嵌入模型。
- 兼容 LlamaIndex 和 ChromaDB。
- 健壮的错误处理和重试机制。
"""

import os
import torch
import numpy as np
import requests
import json
import time
from transformers import AutoTokenizer, AutoModel
from typing import List, Any, Optional
from llama_index.core.embeddings import BaseEmbedding
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.config.config import config
# 添加Chroma兼容性导入
try:
    from chromadb.api.types import EmbeddingFunction
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
from sentence_transformers import SentenceTransformer


# --- Custom Exceptions ---

class EmbeddingError(Exception):
    """自定义嵌入模块异常基类"""
    pass

class APIEmbeddingError(EmbeddingError):
    """API相关异常"""
    pass

class ModelLoadError(EmbeddingError):
    """本地模型加载异常"""
    pass

# --- API Embedding Class ---

class APIEmbedding(BaseEmbedding):
    """通过API调用获取嵌入向量。"""
    
    def __init__(
        self,
        api_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        batch_size: int = 32,
        timeout: int = 30,
        **kwargs: Any,
    ):
        """
        初始化API嵌入客户端。

        Args:
            api_url (str): API服务地址 (e.g., "https://api.openai.com/v1").
            model_name (str): 模型名称。
            api_key (str, optional): API密钥。
            batch_size (int): 批量处理大小。
            timeout (int): 请求超时时间。
        """
        super().__init__(**kwargs)
        
        object.__setattr__(self, 'logger', setup_logger(__name__))
        
        if not api_url or not model_name:
            raise ValueError("APIEmbedding 缺少必要的参数: api_url 和 model_name")
            
        object.__setattr__(self, 'api_url', api_url.rstrip('/'))
        object.__setattr__(self, 'api_key', api_key)
        object.__setattr__(self, 'model_name', model_name)
        object.__setattr__(self, 'batch_size', batch_size)
        object.__setattr__(self, 'timeout', timeout)
        
        self.logger.info(f"初始化 API Embedding - URL: {self.api_url}, Model: {self.model_name}")

    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """发送API请求以获取嵌入。"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        if not self.api_key:
            headers.pop('Authorization')
            
        data = {
            'input': texts,
            'model': self.model_name
        }
        
        try:
            # 常见的 embedding 端点是 /embeddings
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if not result.get('data') or not isinstance(result['data'], list):
                raise APIEmbeddingError(f"API返回无效数据: {result}")
            
            # 按索引排序以确保顺序正确
            sorted_data = sorted(result['data'], key=lambda item: item['index'])
            embeddings = [item['embedding'] for item in sorted_data]
            return embeddings
            
        except requests.exceptions.Timeout as e:
            self.logger.error(f"API请求超时: {self.api_url}")
            raise APIEmbeddingError(f"API请求超时: {self.api_url}") from e
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API请求失败: {e}")
            raise APIEmbeddingError(f"API请求失败: {e}") from e
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"解析API响应失败: {e}")
            raise APIEmbeddingError(f"解析API响应失败: {e}") from e

    def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        """批量生成文本嵌入向量。"""
        self.logger.debug(f"Generating embeddings for {len(texts)} texts via API")
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self._make_request(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Error processing batch {i//self.batch_size + 1} via API: {str(e)}")
                raise
        
        self.logger.debug(f"Generated {len(all_embeddings)} embeddings via API")
        return all_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.get_text_embedding_batch([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.get_text_embedding_batch([text])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def embed_query(self, query: str) -> List[float]:
        """兼容LangChain的查询嵌入方法。"""
        return self._get_query_embedding(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """兼容LangChain的文档嵌入方法。"""
        return self.get_text_embedding_batch(texts)
    
    def __call__(self, input: Any) -> List[List[float]]:
        """Chroma兼容性方法"""
        if isinstance(input, str):
            return [self.get_text_embedding_batch([input])[0]]
        elif isinstance(input, list):
            return self.get_text_embedding_batch(input)
        else:
            raise ValueError(f"Unsupported input type for embedding: {type(input)}")

    def embed_with_retries(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """带重试机制的嵌入生成"""
        for attempt in range(max_retries):
            try:
                return self.get_text_embedding_batch(texts)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"API嵌入生成失败，正在重试 ({attempt + 1}/{max_retries}): {e}")
                time.sleep(1)
        return []

class LocalEmbedding(BaseEmbedding):
    """自定义本地嵌入类，使用指定模型生成文本嵌入向量。"""
    def __init__(
        self,
        model_name: str,
        device: str = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        """
        初始化嵌入模型。

        Args:
            model_name (str): HuggingFace 模型名称（默认 BGE-small-zh-v1.5）。
            device (str, optional): 计算设备（'cuda', 'cpu' 或 None 自动检测）。
            batch_size (int): 批量处理大小。
        """
        super().__init__(**kwargs)
        
        # 使用 object.__setattr__ 来设置属性，避免 Pydantic 限制
        object.__setattr__(self, 'logger', setup_logger(__name__))
        object.__setattr__(self, 'batch_size', batch_size)
        object.__setattr__(self, 'model_name', model_name)
        
        self.logger.info(f"Initializing CustomEmbedding with model: {model_name}")

        # 设置设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        object.__setattr__(self, 'device', device)

        # 加载模型和分词器
        try:
            # 检查是否为本地路径
            if os.path.exists(model_name):
                self.logger.info(f"Using local model from: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    local_files_only=True,
                    # trust_remote_code=True
                )
                model = AutoModel.from_pretrained(
                    model_name, 
                    local_files_only=True,
                    # trust_remote_code=True
                ).to(device).eval()
            else:
                self.logger.info(f"Downloading model from HuggingFace: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    # trust_remote_code=True
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    # trust_remote_code=True
                ).to(device).eval()
            
            object.__setattr__(self, 'tokenizer', tokenizer)
            object.__setattr__(self, 'model', model)
            self.logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}") from e

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        生成查询的嵌入向量。
        
        Args:
            query (str): 查询文本。
            
        Returns:
            List[float]: 嵌入向量。
        """
        return self.get_text_embedding_batch([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        生成文本的嵌入向量。
        
        Args:
            text (str): 输入文本。
            
        Returns:
            List[float]: 嵌入向量。
        """
        return self.get_text_embedding_batch([text])[0]

    def embed_query(self, query: str) -> List[float]:
        """兼容LangChain的查询嵌入方法。"""
        return self._get_query_embedding(query)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """兼容LangChain的文档嵌入方法。"""
        return self.get_text_embedding_batch(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        异步生成查询的嵌入向量。
        
        Args:
            query (str): 查询文本。
            
        Returns:
            List[float]: 嵌入向量。
        """
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """
        异步生成文本的嵌入向量。
        
        Args:
            text (str): 输入文本。
            
        Returns:
            List[float]: 嵌入向量。
        """
        return self._get_text_embedding(text)

    def __call__(self, input) -> List[List[float]]:
        """
        Chroma兼容性方法，用于ChromaDB的EmbeddingFunction接口
        
        Args:
            input: 输入文本列表或单个文本
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if isinstance(input, str):
            return [self.get_text_embedding_batch([input])[0]]
        elif isinstance(input, list):
            return self.get_text_embedding_batch(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    def get_text_embedding_batch(self, texts: List[str], max_length: int = 512, show_progress: bool = False, **kwargs) -> List[List[float]]:
        """
        批量生成文本嵌入向量，包含标准化。

        Args:
            texts (List[str]): 输入文本列表。
            max_length (int): 最大输入长度。

        Returns:
            List[List[float]]: 批量嵌入向量列表。
        """
        self.logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            try:
                # 预处理文本
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)

                # 生成嵌入
                with torch.no_grad():
                    outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                # 标准化嵌入向量
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # 防止除零
                batch_embeddings = batch_embeddings / norms
                embeddings.extend([emb.tolist() for emb in batch_embeddings])
            except Exception as e:
                self.logger.error(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                raise

        self.logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def get_agg_embedding_from_queries(self, queries: List[str], max_length: int = 512) -> List[float]:
        """
        生成查询的聚合嵌入向量。

        Args:
            queries (List[str]): 查询文本列表。
            max_length (int): 最大输入长度。

        Returns:
            List[float]: 聚合后的嵌入向量。
        """
        self.logger.debug(f"Aggregating embeddings for {len(queries)} queries")
        embeddings = self.get_text_embedding_batch(queries, max_length)
        agg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(agg_embedding)
        if norm == 0:
            norm = 1
        return (agg_embedding / norm).tolist()

    def __getstate__(self) -> dict:
        """支持序列化，保存配置但忽略模型和分词器。"""
        return {"model_name": self.model_name, "device": self.device, "batch_size": self.batch_size}

    def __setstate__(self, state: dict):
        """支持序列化，恢复配置并重新加载模型。"""
        self.__init__(state["model_name"], state["device"], state["batch_size"])

    def get_single_text_embedding(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入向量。

        Args:
            text (str): 输入文本。

        Returns:
            List[float]: 嵌入向量。
        """
        return self.get_text_embedding_batch([text])[0]

    # 添加Chroma兼容性方法
    def __call__(self, input) -> List[List[float]]:
        """
        Chroma兼容性方法，用于ChromaDB的EmbeddingFunction接口
        
        Args:
            input: 输入文本列表或单个文本
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if isinstance(input, str):
            return [self.get_text_embedding_batch([input])[0]]
        elif isinstance(input, list):
            return self.get_text_embedding_batch(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    
    def embed_with_retries(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        带重试机制的嵌入生成
        
        Args:
            texts: 输入文本列表
            max_retries: 最大重试次数
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        for attempt in range(max_retries):
            try:
                return self.get_text_embedding_batch(texts)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"嵌入生成失败，正在重试 ({attempt + 1}/{max_retries}): {e}")
                import time
                time.sleep(1)  # 等待1秒后重试
        
        return []  # 不会执行到这里，但为了类型检查


class EmbeddingProvider:
    """
    嵌入模型提供者（工厂类）
    根据配置创建并缓存Embedding实例
    """
    _instances = {}
    _logger = None

    @classmethod
    def _get_logger(cls):
        if cls._logger is None:
            cls._logger = setup_logger(__name__)
        return cls._logger

    def __getitem__(self, name: str) -> BaseEmbedding:
        logger = self._get_logger()
        
        if name not in self._instances:
            logger.info(f"首次创建Embedding实例: {name}")

            if not isinstance(name, str) or not name:
                logger.error(f"无效的 embedding provider 名称: '{name}' (类型: {type(name)})")
                raise ValueError(f"无效的 embedding provider 名称: '{name}'")

            provider_config = config.get('embedding_providers', {}).get(name)
            if not provider_config:
                available_providers = list(config.get('embedding_providers', {}).keys())
                logger.error(f"在config.yaml中未找到名为 '{name}' 的 embedding provider 配置。可用的 providers: {available_providers}")
                raise ValueError(f"在config.yaml中未找到名为 '{name}' 的 embedding provider 配置")

            embed_type = provider_config.get('type', 'local').lower()
            params = provider_config.get('params', {})

            if embed_type == 'local':
                self._instances[name] = LocalEmbedding(**params)
            elif embed_type == 'api':
                self._instances[name] = APIEmbedding(**params)
            else:
                logger.error(f"不支持的Embedding类型: {embed_type} (在 '{name}' provider中配置)")
                raise ValueError(f"不支持的Embedding类型: {embed_type} (在 '{name}' provider中配置)")
        
        return self._instances[name]

# 创建全局Embedding提供者实例
embedding_provider = EmbeddingProvider()
