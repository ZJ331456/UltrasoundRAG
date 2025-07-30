"""图像检索系统主入口

主要功能:
整合了基于图像的检索功能，包括图像搜索、图像标题匹配以及CLIP文本-图像搜索等功能。
它作为应用层的协调器，调用底层utils中的各个组件，专注于图像检索业务逻辑的编排，
接收用户查询，返回最终的图像检索结果。

核心组件:
- **ImageSearcher**: 图像搜索器 - 基于图像相似度搜索
- **CaptionImageMatcher**: 图像标题匹配器 - 基于图像标题的文本匹配
- **CLIPTextImageSearcher**: CLIP文本-图像搜索器 - 基于CLIP模型的跨模态搜索
- **UnifiedImageSearcher**: 统一图像搜索器 - 整合多种图像检索方法
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from PIL import Image

from MedicalRAG.config.config import config, get_image_collection
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.embedding_utils import embedding_provider

# 尝试导入CLIP相关库
try:
    import clip
except ImportError:
    print("警告: clip 未安装，CLIP功能将不可用")
    clip = None


@dataclass
class ImageRetrievalResult:
    """图像检索结果统一数据结构"""
    image_id: str
    image_path: str
    caption: str
    metadata: Dict[str, Any]
    score: float
    retrieval_type: str  # 'image_similarity', 'caption_match', 'clip_text_image'


# 直接从utils模块导入实现
from MedicalRAG.utils.retrival_utils import ImageSearcher as _ImageSearcher
from MedicalRAG.utils.retrival_utils import CaptionImageMatcher as _CaptionImageMatcher
from MedicalRAG.utils.retrival_utils import CLIPTextImageSearcher as _CLIPTextImageSearcher
from MedicalRAG.utils.retrival_utils import ImageDisplayer


class UnifiedImageSearcher:
    """统一的图像搜索器，整合以图搜图和多种以文搜图功能"""
    
    def __init__(self, config):
        self.logger = setup_logger(__name__)
        self.config = config
        
        # 初始化各个搜索器
        self.image_searcher = None
        self.caption_matcher = None
        self.clip_searcher = None
        self.displayer = ImageDisplayer()
        
        # 初始化状态
        self.image_search_ready = False
        self.caption_search_ready = False
        self.clip_search_ready = False
        
        try:
            # 从配置中获取参数
            # image_model_path = getattr(config, 'image_model_path', None)
            # config_path = getattr(config, 'config_path', None)
            # text_model_name = getattr(config, 'text_model_name', None)
            # chroma_persist_dir = getattr(config, 'chroma_persist_dir', None)
            # collection_name = getattr(config, 'collection_name', None)
            image_model_path = self.config['image_search']['image_to_image']['embedding_model']
            config_path = self.config['image_search']['image_to_image']['config_path']
            text_model_name = self.config['image_search']['text_to_image']['embedding_model']
            chroma_persist_dir = self.config['image_search']['vectorstore_path']
            collection_name = self.config['image_search']['collection_name']
            # 初始化以图搜图功能
            if image_model_path and config_path:
                self.image_searcher = _ImageSearcher(image_model_path, config_path, 
                                                  chroma_persist_dir, collection_name)
                self.image_search_ready = self.image_searcher.initialized
                
            # 初始化标题匹配功能
            if text_model_name:
                self.caption_matcher = _CaptionImageMatcher(text_model_name, chroma_persist_dir, collection_name)
                self.caption_search_ready = self.caption_matcher.initialized
                
            # 初始化CLIP文本搜索功能
            if image_model_path and config_path:
                self.clip_searcher = _CLIPTextImageSearcher(image_model_path, config_path, 
                                                         chroma_persist_dir, collection_name)
                self.clip_search_ready = self.clip_searcher.initialized
                
            self.logger.info(f"UnifiedImageSearcher 初始化完成 - "
                           f"以图搜图: {self.image_search_ready}, "
                           f"标题搜索: {self.caption_search_ready}, "
                           f"CLIP搜索: {self.clip_search_ready}")
                           
        except Exception as e:
            self.logger.error(f"UnifiedImageSearcher 初始化失败: {e}")
    
    def search_by_image(self, image_path: str, top_n: int = 10) -> List[ImageRetrievalResult]:
        """以图搜图"""
        if not self.image_search_ready:
            self.logger.warning("以图搜图功能未初始化")
            return []
            
        results = self.image_searcher.search(image_path, top_n)
        if not results:
            return []
            
        # 转换为统一格式
        retrieval_results = []
        ids = results.get('ids', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for i, (id_, metadata, distance) in enumerate(zip(ids, metadatas, distances)):
            retrieval_results.append(ImageRetrievalResult(
                image_id=id_,
                image_path=metadata.get('image_path', ''),
                caption=metadata.get('caption', ''),
                metadata=metadata,
                score=1.0 - distance,  # 转换距离为相似度分数
                retrieval_type='image_similarity'
            ))
        
        return retrieval_results
    
    def search_by_text(self, query_text: str, top_n: int = 10) -> List[ImageRetrievalResult]:
        """统一的文本搜索，结合多种方法"""
        all_results = []
        
        # CLIP文本搜索
        if self.clip_search_ready:
            clip_results = self.clip_searcher.search(query_text, top_n)
            if clip_results:
                ids = clip_results.get('ids', [[]])[0]
                metadatas = clip_results.get('metadatas', [[]])[0]
                distances = clip_results.get('distances', [[]])[0]
                
                for id_, metadata, distance in zip(ids, metadatas, distances):
                    all_results.append(ImageRetrievalResult(
                        image_id=id_,
                        image_path=metadata.get('image_path', ''),
                        caption=metadata.get('caption', ''),
                        metadata=metadata,
                        score=1.0 - distance,
                        retrieval_type='clip_text_image'
                    ))
        
        # 标题搜索
        if self.caption_search_ready:
            caption_results = self.caption_matcher.search(query_text, top_n)
            if caption_results:
                ids = caption_results.get('ids', [[]])[0]
                metadatas = caption_results.get('metadatas', [[]])[0]
                distances = caption_results.get('distances', [[]])[0]
                
                for id_, metadata, distance in zip(ids, metadatas, distances):
                    all_results.append(ImageRetrievalResult(
                        image_id=id_,
                        image_path=metadata.get('image_path', ''),
                        caption=metadata.get('caption', ''),
                        metadata=metadata,
                        score=1.0 - distance,
                        retrieval_type='caption_match'
                    ))
        
        # 去重并按分数排序
        unique_results = {}
        for result in all_results:
            if result.image_id not in unique_results or result.score > unique_results[result.image_id].score:
                unique_results[result.image_id] = result
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:top_n]
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        info = {
            'image_search_ready': self.image_search_ready,
            'caption_search_ready': self.caption_search_ready,
            'clip_search_ready': self.clip_search_ready,
            'image_count': 0,
            'collection_name': 'unknown'
        }
        
        # 尝试从任一可用的搜索器获取数据库信息
        if self.image_search_ready and hasattr(self.image_searcher, 'collection'):
            try:
                collection = self.image_searcher.collection
                info['image_count'] = collection.count()
                info['collection_name'] = collection.name
            except Exception as e:
                self.logger.warning(f"获取数据库信息失败: {e}")
        
        return info
    
    def get_status(self) -> Dict[str, bool]:
        """获取各组件状态"""
        return {
            'image_search': self.image_search_ready,
            'caption_search': self.caption_search_ready,
            'clip_search': self.clip_search_ready
        }


class ImageRetriever:
    """图像检索器 - 应用层协调器，专注于业务流程编排"""
    
    def __init__(self, config_path: str = None):
        """
        初始化图像检索器
        
        Args:
            config_path: 配置文件路径（可选，默认使用全局config）
        """
        self.logger = setup_logger(__name__)
        self.config = config  # 使用全局配置
        
        # 初始化核心组件
        self.unified_searcher = UnifiedImageSearcher(self.config)
        
        # 会话相关
        self.current_session_results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info("图像检索器初始化完成")
    
    def search_by_image(self, image_path: str, top_k: int = 10) -> Dict[str, Any]:
        """
        以图搜图
        
        Args:
            image_path: 查询图像路径
            top_k: 返回结果数量
            
        Returns:
            包含检索结果的字典
        """
        try:
            # 执行以图搜图 - 现在返回 List[ImageRetrievalResult]
            results = self.unified_searcher.search_by_image(image_path, top_k)
            
            # 构建返回结果
            result_dict = {
                "query_type": "image_to_image",
                "query_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
                "results": []
            }
            
            # 直接使用 ImageRetrievalResult 对象
            for result in results:
                result_dict["results"].append({
                    "image_id": result.image_id,
                    "image_path": result.image_path,
                    "caption": result.caption,
                    "score": result.score,
                    "retrieval_type": result.retrieval_type,
                    "metadata": result.metadata
                })
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"以图搜图失败: {e}")
            return {"error": f"以图搜图失败: {e}"}
    
    def search_by_text(
        self, 
        query_text: str, 
        top_k: int = 10,
        search_type: str = "unified"
    ) -> Dict[str, Any]:
        """
        以文搜图
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            search_type: 搜索类型 ('caption', 'clip', 'unified')
            
        Returns:
            包含检索结果的字典
        """
        try:
            if search_type == "unified":
                # 使用统一搜索方法
                results = self.unified_searcher.search_by_text(query_text, top_k)
            else:
                # 对于特定类型的搜索，需要直接调用底层搜索器并转换结果
                if search_type == "caption" and self.unified_searcher.caption_search_ready:
                    raw_results = self.unified_searcher.caption_matcher.search(query_text, top_k)
                    results = self._convert_raw_results_to_retrieval_results(raw_results, 'caption_match')
                elif search_type == "clip" and self.unified_searcher.clip_search_ready:
                    raw_results = self.unified_searcher.clip_searcher.search(query_text, top_k)
                    results = self._convert_raw_results_to_retrieval_results(raw_results, 'clip_text_image')
                else:
                    results = []
            
            # 构建返回结果
            result_dict = {
                "query_type": f"text_to_image_{search_type}",
                "query_text": query_text,
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
                "results": []
            }
            
            # 转换结果格式
            for result in results:
                result_dict["results"].append({
                    "image_id": result.image_id,
                    "image_path": result.image_path,
                    "caption": result.caption,
                    "score": result.score,
                    "retrieval_type": result.retrieval_type,
                    "metadata": result.metadata
                })
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"以文搜图失败: {e}")
            return {"error": f"以文搜图失败: {e}"}
    
    def _convert_raw_results_to_retrieval_results(self, raw_results: Dict[str, Any], retrieval_type: str) -> List[ImageRetrievalResult]:
        """将原始ChromaDB结果转换为ImageRetrievalResult列表"""
        if not raw_results:
            return []
        
        results = []
        ids = raw_results.get('ids', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        
        for id_, metadata, distance in zip(ids, metadatas, distances):
            results.append(ImageRetrievalResult(
                image_id=id_,
                image_path=metadata.get('image_path', ''),
                caption=metadata.get('caption', ''),
                metadata=metadata,
                score=1.0 - distance,  # 转换距离为相似度分数
                retrieval_type=retrieval_type
            ))
        
        return results
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取图像数据库信息"""
        return self.unified_searcher.get_database_info()


# def main():
#     """主函数 - 演示图像检索系统的使用"""
#     # 初始化图像检索器
#     retriever = ImageRetriever()
    
#     # 显示数据库信息
#     db_info = retriever.get_database_info()
#     print(f"\n=== 图像数据库信息 ===")
#     print(f"图像数量: {db_info.get('image_count', 'N/A')}")
#     print(f"集合名称: {db_info.get('collection_name', 'N/A')}")
    
#     if db_info.get('image_count', 0) == 0:
#         print("警告: 图像数据库中没有图像，请先运行图像索引创建程序")
#         return
    
#     # 示例查询
#     test_queries = [
#         "心脏病",
#         "肺部感染",
#         "骨折",
#         "皮肤病变"
#     ]
    
#     print(f"\n=== 开始处理 {len(test_queries)} 个图像查询 ===")
    
#     for i, query in enumerate(test_queries, 1):
#         print(f"\n[{i}/{len(test_queries)}] 处理查询: {query}")
        
#         # 执行文本到图像的搜索
#         results = retriever.search_by_text(query, top_k=5)
        
#         print(f"✓ 完成，找到 {results.get('total_results', 0)} 个相关图像")
        
#         # 显示前3个结果的详情
#         for j, result in enumerate(results.get('results', [])[:3], 1):
#             print(f"  {j}. 图像ID: {result['image_id']}")
#             print(f"     路径: {result['image_path']}")
#             caption = result['caption']
#             print(f"     标题: {caption[:100]}..." if len(caption) > 100 else f"     标题: {caption}")
#             print(f"     分数: {result['score']:.4f}")
#             print(f"     类型: {result['retrieval_type']}")
    
#     print(f"\n=== 图像检索演示完成 ===")


# if __name__ == "__main__":
#     main()