"""检索工具模块
提供文本检索和图像检索的核心功能

主要功能：
文本检索：
1. BM25稀疏检索器 - 基于关键词的传统检索
2. 密集检索器 - 基于语义向量的检索
3. 混合检索引擎 - 结合密集和稀疏检索

图像检索：
1. 以图搜图 - 基于图像相似性的检索
2. 以文搜图(标题匹配) - 通过文本在图像标题中进行语义搜索
3. 以文搜图(CLIP内容) - 通过CLIP模型直接检索图像内容
4. 统一图像检索接口 - 整合多种图像检索方式

数据结构：
- RetrievalResult: 检索结果统一数据结构
"""

import os
import sys
import logging
import torch
from MedicalRAG.config.config import get_document_collection, get_image_collection, config_manager
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod

# 图像处理相关导入
import chromadb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPImageProcessor
from chromadb.config import Settings

from MedicalRAG.config.config import config
from MedicalRAG.utils.embedding_utils import embedding_provider
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.image_utils import get_image_embedding, load_clip_model

# 尝试导入jieba用于中文分词
try:
    import jieba
    jieba.setLogLevel(jieba.logging.INFO)
except ImportError:
    print("警告: jieba 未安装，将使用简单的字符分词")
    jieba = None


def find_chinese_font() -> Optional[str]:
    """找到一个可用的中文字体"""
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    font_check_order = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'Heiti', 'Arial Unicode MS']
    
    for font_name_check in font_check_order:
        for font_path in font_paths:
            try:
                font_name = fm.FontProperties(fname=font_path).get_name()
                if font_name_check in font_name:
                    return font_path
            except Exception:
                continue
    
    return None

@dataclass
class RetrievalResult:
    """检索结果统一数据结构"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_type: str  # 'dense', 'sparse', 'hybrid', 'image'


class BaseImageSearcher(ABC):
    """图像搜索基类，提供公共功能"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.initialized = False
        self.config = config_manager.config.get('image_search', {})
    
    def _validate_config(self, required_keys: List[str]) -> bool:
        """验证配置是否包含必需的键"""
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"配置中缺少必需的键: {key}")
                return False
        return True
    
    def _handle_error(self, operation: str, error: Exception) -> None:
        """统一的错误处理"""
        self.logger.error(f"{operation}失败: {error}", exc_info=True)
        self.initialized = False
    
    @abstractmethod
    def search(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """抽象搜索方法"""
        pass


class BM25SparseRetriever:
    """BM25稀疏检索器 - 基于关键词匹配的传统检索方法"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            k1: 控制词频饱和度的参数
            b: 控制文档长度归一化的参数
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_lens = []
        self.avgdl = 0
        self.logger = setup_logger(__name__)
        
    def tokenize(self, text: str) -> List[str]:
        """文本分词处理"""
        if jieba:
            tokens = list(jieba.cut_for_search(text.lower()))
            self.logger.debug(f"jieba分词结果: {tokens}")
        else:
            # 简单的字符级分词作为备选
            tokens = [char for char in text.lower() if char.strip()]
        
        # 过滤短词，但对中文要宽松一些
        if jieba:
            tokens = [token for token in tokens if len(token.strip()) > 0 and token.strip() not in {'的', '了', '是', '在', '有', '和', '与', '或', '但', '而'}]
        else:
            tokens = [token for token in tokens if len(token.strip()) > 1]
        
        return tokens
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """构建BM25索引"""
        self.logger.info(f"构建BM25索引，文档数量: {len(documents)}")
        
        self.documents = documents
        self.doc_lens = []
        
        # 构建词频统计
        for doc in documents:
            content = str(doc.get('content', ''))
            tokens = self.tokenize(content)
            self.doc_lens.append(len(tokens))
            
            # 统计每个词在多少个文档中出现
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1
        
        # 计算平均文档长度
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        
        # 计算IDF值
        num_docs = len(documents)
        for token, freq in self.doc_freqs.items():
            self.idf[token] = np.log((num_docs - freq + 0.5) / (freq + 0.5))
        
        self.logger.info(f"索引构建完成，词汇表大小: {len(self.idf)}")
    
    def get_scores(self, query: str) -> List[float]:
        """计算查询与所有文档的BM25分数"""
        query_tokens = self.tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.documents):
            content = str(doc.get('content', ''))
            doc_tokens = self.tokenize(content)
            
            # 计算文档中每个词的频率
            term_freqs = defaultdict(int)
            for token in doc_tokens:
                term_freqs[token] += 1
            
            score = 0.0
            for token in query_tokens:
                if token in self.idf:
                    tf = term_freqs[token]
                    idf = self.idf[token]
                    
                    # BM25公式
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl))
            
            scores.append(score)
        
        return scores
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """执行BM25搜索"""
        scores = self.get_scores(query)
        
        # 获取top_k结果
        doc_scores = [(i, score) for i, score in enumerate(scores)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in doc_scores[:top_k]:
            doc = self.documents[i]
            result = RetrievalResult(
                doc_id=str(doc.get('id', i)),
                content=str(doc.get('content', '')),
                metadata=doc.get('metadata', {}),
                score=score,
                retrieval_type='sparse'
            )
            results.append(result)
        
        return results


class DenseRetriever:
    """密集检索器 - 基于语义向量搜索"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化密集检索器
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict
        self.logger = setup_logger(__name__)
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_collection = None
        
        self._setup_components()
    
    def _setup_components(self):
        """设置检索组件"""
        # 初始化嵌入模型
        # 从配置中获取要使用的 embedding provider 的名称
        provider_name = self.config['embedding'].get('provider')
        if not provider_name:
            raise ValueError("配置文件 'embedding' 部分缺少 'provider' 键，请指定要使用的嵌入模型提供者")

        try:
            # 直接通过名称从 embedding_provider 获取模型实例
            self.embedding_model = embedding_provider[provider_name]
            self.logger.info(f"成功从 provider 获取嵌入模型: {provider_name}")
        except ValueError as e:
            self.logger.error(f"无法获取嵌入模型 '{provider_name}': {e}")
            raise  # 重新抛出异常
        
        # 使用统一的ChromaDB管理器
        self.chroma_collection = get_document_collection()
        self.logger.info("成功连接到ChromaDB集合")
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        执行密集检索（语义向量搜索）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.get_query_embedding(query)
            
            # 在ChromaDB中搜索
            chroma_results = self.chroma_collection.query(
                # query_embeddings=[query_embedding.tolist()]
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            results = []
            if chroma_results['documents'] and chroma_results['documents'][0]:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    chroma_results['ids'][0],
                    chroma_results['documents'][0],
                    chroma_results['metadatas'][0],
                    chroma_results['distances'][0]
                )):
                    # 将距离转换为相似度分数
                    similarity_score = 1 - distance if distance is not None else 0.0
                    
                    result = RetrievalResult(
                        doc_id=doc_id,
                        content=document,
                        metadata=metadata or {},
                        score=similarity_score,
                        retrieval_type='dense'
                    )
                    results.append(result)
            
            self.logger.info(f"密集检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"密集检索失败: {e}")
            return []


class ImageSearcher(BaseImageSearcher):
    """图像到图像的相似性搜索器"""
    
    def __init__(self, image_model_path: str, config_path: str, chroma_persist_dir: str = None, collection_name: str = None):
        super().__init__()
        self.image_model_path = image_model_path
        self.config_path = config_path
        self.image_model = None
        self.image_processor = None
        self.collection = None
        
        try:
            self._load_models()
            self._connect_to_db()
            self.initialized = True
            self.logger.info("ImageSearcher 初始化成功")
        except Exception as e:
            self._handle_error("ImageSearcher 初始化", e)
    
    def _load_models(self) -> None:
        """加载图像模型"""
        self.image_model, self.image_processor = load_clip_model(self.image_model_path, self.config_path)
        if not self.image_model:
            raise RuntimeError(f"无法加载模型: {self.image_model_path}")
    
    def _connect_to_db(self) -> None:
        """连接到数据库"""
        self.logger.info("正在连接到 ChromaDB")
        self.collection = get_image_collection()
        self.logger.info("成功连接到图像集合")
    
    def search(self, image_path: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据图像路径搜索相似图像"""
        return self.search_by_image(image_path, top_n)
    
    def search_by_image(self, image_path: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据给定的图片路径，在索引中搜索最相似的图片"""
        if not self.initialized:
            self.logger.warning("ImageSearcher 未正确初始化")
            return None
        
        if not os.path.exists(image_path):
            self.logger.error(f"图像文件不存在: {image_path}")
            return None
        
        try:
            self.logger.info(f"正在编码查询图像: {image_path}")
            query_embedding = get_image_embedding(image_path, self.image_model, self.image_processor)
            if query_embedding is None:
                return None
            
            self.logger.info("正在数据库中搜索相似图像")
            results = self.collection.query(
                query_embeddings=[query_embedding.flatten().tolist()],
                n_results=top_n + 1  # 请求 N+1 个结果，以防查询本身被过滤
            )
            
            # 截断结果到指定数量
            if results and results.get('ids') and len(results['ids'][0]) > top_n:
                for key in results:
                    if results[key] and isinstance(results[key], list) and len(results[key]) > 0:
                        results[key] = [lst[:top_n] for lst in results[key]]
            
            return results
            
        except Exception as e:
            self._handle_error("图像搜索", e)
            return None


class CaptionImageMatcher(BaseImageSearcher):
    """文本到图像的语义匹配搜索器"""
    
    def __init__(self, text_embedding_model_path: str, chroma_persist_dir: str = None, collection_name: str = None):
        super().__init__()
        self.text_embedding_model_path = text_embedding_model_path
        self.text_model = None
        self.caption_collection = None
        self.captions_with_metadata = []
        
        try:
            self._load_model()
            self._build_caption_index()
            self.logger.info("CaptionImageMatcher 初始化成功")
        except Exception as e:
            self._handle_error("CaptionImageMatcher 初始化", e)
    
    def _load_model(self) -> None:
        """加载文本嵌入模型"""
        self.logger.info(f"正在加载文本嵌入模型: {self.text_embedding_model_path}")
        self.text_model = SentenceTransformer(self.text_embedding_model_path)
    
    def _build_caption_index(self) -> None:
        """构建图片标题索引"""
        self.logger.info("正在从 ChromaDB 加载图像元数据")
        
        image_collection = get_image_collection()
        all_data = image_collection.get(include=["metadatas", "documents"])
        
        # 提取标题和元数据
        if all_data and all_data.get('metadatas'):
            for meta, doc in zip(all_data['metadatas'], all_data['documents']):
                caption = meta.get('full_caption', doc)
                image_path = meta.get('image_path')
                if caption and image_path:
                    self.captions_with_metadata.append({
                        'caption': caption,
                        'image_path': image_path
                    })
        
        if not self.captions_with_metadata:
            self.logger.warning("数据库中未找到有效的图片标题或图片路径，文本搜图功能将不可用")
            self.initialized = False
            return
        
        # 创建内存中的标题索引
        all_captions = [item['caption'] for item in self.captions_with_metadata]
        self.logger.info(f"为 {len(all_captions)} 个图片标题创建内存中的语义索引")
        
        in_memory_client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
        self.caption_collection = in_memory_client.create_collection(
            name="image_captions_temp",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 编码标题并添加到集合
        caption_embeddings = self.text_model.encode(
            all_captions, 
            convert_to_tensor=False, 
            show_progress_bar=True
        )
        
        self.caption_collection.add(
            embeddings=caption_embeddings.tolist(),
            documents=all_captions,
            metadatas=[{'image_path': item['image_path']} for item in self.captions_with_metadata],
            ids=[str(i) for i in range(len(all_captions))]
        )
        
        self.logger.info("图片标题索引构建成功")
        self.initialized = True
    
    def search(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据文本查询搜索相关图像"""
        if not self.initialized or not self.caption_collection:
            self.logger.warning("CaptionImageMatcher 未成功初始化，无法执行搜索")
            return None
        
        try:
            self.logger.info(f"正在编码查询: {query_text}")
            query_embedding = self.text_model.encode(query_text, convert_to_tensor=False)
            
            self.logger.info("正在搜索相似的图片标题")
            results = self.caption_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_n
            )
            return results
            
        except Exception as e:
            self._handle_error("文本搜索图像", e)
            return None


class CLIPTextImageSearcher(BaseImageSearcher):
    """基于CLIP的文本到图像内容检索器"""
    
    def __init__(self, image_model_path: str, config_path: str, chroma_persist_dir: str = None, collection_name: str = None):
        super().__init__()
        self.image_model_path = image_model_path
        self.config_path = config_path
        self.clip_model = None
        self.clip_processor = None
        self.collection = None
        
        try:
            self._load_clip_model()
            self._connect_to_db()
            self.initialized = True
            self.logger.info("CLIPTextImageSearcher 初始化成功")
        except Exception as e:
            self._handle_error("CLIPTextImageSearcher 初始化", e)
    
    def _load_clip_model(self) -> None:
        """加载CLIP模型"""
        self.logger.info(f"正在加载FetalCLIP模型: {self.image_model_path}")
        from MedicalRAG.model.fetal_clip_model import load_fetal_clip_model
        self.clip_model = load_fetal_clip_model(self.image_model_path, self.config_path)
        if not self.clip_model:
            raise RuntimeError(f"无法加载FetalCLIP模型: {self.image_model_path}")
        self.logger.info("FetalCLIP模型加载成功")
    
    def _connect_to_db(self) -> None:
        """连接到数据库"""
        self.logger.info("正在连接到 ChromaDB 图像集合")
        self.collection = get_image_collection()
        self.logger.info("成功连接到图像集合")
    
    def search(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据文本查询直接检索图像内容"""
        return self.search_by_clip_text(query_text, top_n)
    
    def search_by_clip_text(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """使用CLIP模型根据文本直接检索图像内容"""
        if not self.initialized:
            self.logger.warning("CLIPTextImageSearcher 未正确初始化")
            return None
        
        try:
            self.logger.info(f"正在使用FetalCLIP编码查询文本: {query_text}")
            
            # 使用FetalCLIP模型编码文本
            import torch
            
            # 使用FetalCLIP的tokenizer对文本进行分词
            text_tokens = self.clip_model.tokenize_text([query_text])
            
            # 使用FetalCLIP模型编码文本
            text_features = self.clip_model.encode_text(text_tokens)
            
            # 转换为列表格式用于ChromaDB查询
            query_embedding = text_features.squeeze().cpu().numpy().tolist()
            
            self.logger.info("正在数据库中搜索相似图像")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_n
            )
            
            self.logger.info(f"CLIP文本搜索完成，找到 {len(results.get('ids', [[]])[0])} 个结果")
            return results
            
        except Exception as e:
            self._handle_error("CLIP文本搜索图像", e)
            return None


class ImageDisplayer:
    """图像搜索结果显示器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chinese_font = find_chinese_font()
        
        # 设置matplotlib中文字体
        if self.chinese_font:
            plt.rcParams['font.sans-serif'] = [self.chinese_font]
            plt.rcParams['axes.unicode_minus'] = False
            self.logger.info(f"设置中文字体: {self.chinese_font}")
        else:
            self.logger.warning("未找到中文字体，可能无法正确显示中文")
    
    def display_results(self, results: Dict[str, Any], search_type: str = "图像搜索") -> None:
        """显示搜索结果"""
        if not results or 'ids' not in results:
            self.logger.warning("没有搜索结果可显示")
            return
        
        ids = results['ids'][0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        if not ids:
            self.logger.info("搜索结果为空")
            return
        
        print(f"\n{search_type}结果 (共 {len(ids)} 个):")
        print("=" * 50)
        
        for i, (id_, metadata, distance) in enumerate(zip(ids, metadatas, distances)):
            self._display_single_result(i + 1, id_, metadata, distance)
    
    def _display_single_result(self, index: int, id_: str, metadata: Dict, distance: float) -> None:
        """显示单个搜索结果"""
        print(f"\n结果 {index}:")
        print(f"  ID: {id_}")
        print(f"  相似度距离: {distance:.4f}")
        
        if metadata:
            for key, value in metadata.items():
                if key == 'image_path':
                    print(f"  图片路径: {value}")
                elif key == 'caption':
                    print(f"  图片标题: {value}")
                else:
                    print(f"  {key}: {value}")
    
    def display_images(self, results: Dict[str, Any], max_display: int = 6) -> None:
        """显示图像搜索结果的图片"""
        if not results or 'ids' not in results:
            self.logger.warning("没有搜索结果可显示")
            return
        
        ids = results['ids'][0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        if not ids:
            self.logger.info("搜索结果为空")
            return
        
        # 限制显示数量
        display_count = min(len(ids), max_display)
        
        # 计算子图布局
        cols = min(3, display_count)
        rows = (display_count + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if display_count == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        for i in range(display_count):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0
            
            image_path = metadata.get('image_path', '')
            caption = metadata.get('caption', '无标题')
            
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    axes[i].imshow(img)
                    axes[i].set_title(f"{caption}\n距离: {distance:.4f}", fontsize=10)
                else:
                    axes[i].text(0.5, 0.5, f"图片不存在\n{image_path}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f"{caption}\n距离: {distance:.4f}", fontsize=10)
            except Exception as e:
                self.logger.error(f"显示图片时出错: {e}")
                axes[i].text(0.5, 0.5, f"显示错误\n{str(e)}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{caption}\n距离: {distance:.4f}", fontsize=10)
            
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(display_count, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


# 辅助函数
def get_image_collection():
    """获取图像集合的辅助函数"""
    from MedicalRAG.config.config import get_image_collection as config_get_image_collection
    return config_get_image_collection()


def create_image_retrieval_result(image_path: str, caption: str = None, 
                                 score: float = 0.0, metadata: Dict = None) -> RetrievalResult:
    """创建图像检索结果"""
    return RetrievalResult(
        content=caption or f"图像: {os.path.basename(image_path)}",
        metadata=metadata or {'image_path': image_path, 'caption': caption},
        score=score,
        retrieval_type='image'
    )