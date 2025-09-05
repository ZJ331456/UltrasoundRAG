"""图片过滤工具模块
基于filter.py中的过滤机制，为图像检索结果提供重排序功能

主要功能：
1. 文本相似度过滤 - 基于TF-IDF、关键词重叠、SimHash、BM25的文本相似度计算
2. 图像相似度过滤 - 基于LPIPS、MS-SSIM、HaarPSI的图像相似度计算
3. 多模态过滤 - 结合文本和图像相似度的综合过滤
4. 图像检索结果重排序 - 应用过滤机制对检索结果进行重新排序
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from PIL import Image
import hashlib

# 导入依赖库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lpips
import piq
from torchvision import transforms

from UltrasoundRAG.utils.logger import setup_logger
from UltrasoundRAG.retrival.data_structures import RetrievalResult

# 在导入任何模块之前设置jieba缓存目录
import os
cache_dir = os.path.expanduser("~/.cache/jieba")
os.makedirs(cache_dir, exist_ok=True)
os.environ['JIEBA_CACHE_DIR'] = cache_dir

try:
    import jieba
    import jieba.analyse
except ImportError:
    print("警告: jieba未安装，中文分词功能将受限")
    jieba = None
except Exception as e:
    print(f"警告: jieba 初始化失败: {e}，中文分词功能将受限")
    jieba = None




# ==================== 核心过滤功能函数 ====================

def extract_text_features(texts, method='tfidf'):
    """
    提取文本特征用于重排序
    Args:
        texts: 文本列表
        method: 特征提取方法 ('tfidf', 'keyword', 'combined')
    """
    if method == 'tfidf':
        if TfidfVectorizer is None:
            raise ImportError("sklearn未安装，无法使用TF-IDF特征")
        # 使用TF-IDF特征
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # 医学文本不过滤停用词
            ngram_range=(1, 2)  # 使用1-gram和2-gram
        )
        features = vectorizer.fit_transform(texts)
        return features, vectorizer
    
    elif method == 'keyword':
        if jieba is None:
            raise ImportError("jieba未安装，无法使用关键词特征")
        # 使用关键词特征
        keyword_features = []
        for text in texts:
            # 提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True)
            # 构建关键词向量
            keyword_dict = dict(keywords)
            keyword_features.append(keyword_dict)
        return keyword_features, None
    
    elif method == 'combined':
        # 结合TF-IDF和关键词
        tfidf_features, vectorizer = extract_text_features(texts, 'tfidf')
        keyword_features, _ = extract_text_features(texts, 'keyword')
        return (tfidf_features, keyword_features), vectorizer
    
    else:
        raise ValueError(f"不支持的特征提取方法: {method}")


def calculate_text_similarity(query_text, candidate_texts, method='fused'):
    """
    计算查询文本与候选文本的相似度（融合多种文本相似度指标）
    Args:
        query_text: 查询文本
        candidate_texts: 候选文本列表
        method: 相似度计算方法，默认'fused'为融合
    Returns:
        similarities: np.ndarray
    """
    if not candidate_texts:
        return np.array([])
    
    # 1. TF-IDF 余弦相似度
    tfidf_sim = np.zeros(len(candidate_texts))
    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            all_texts = [query_text] + candidate_texts
            tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 2))
            tfidf_features = tfidf_vectorizer.fit_transform(all_texts)
            tfidf_sim = cosine_similarity(tfidf_features[0:1], tfidf_features[1:])[0]
        except Exception as e:
            print(f"TF-IDF计算失败: {e}")
            tfidf_sim = np.zeros(len(candidate_texts))

    # 2. 关键词重叠度
    keyword_sim = np.zeros(len(candidate_texts))
    if jieba is not None:
        try:
            query_keywords = set(jieba.analyse.extract_tags(query_text, topK=10))
            keyword_sim = []
            for candidate_text in candidate_texts:
                candidate_keywords = set(jieba.analyse.extract_tags(candidate_text, topK=10))
                if len(query_keywords) == 0 or len(candidate_keywords) == 0:
                    keyword_sim.append(0.0)
                else:
                    overlap = len(query_keywords & candidate_keywords)
                    similarity = overlap / max(len(query_keywords), len(candidate_keywords))
                    keyword_sim.append(similarity)
            keyword_sim = np.array(keyword_sim)
        except Exception as e:
            print(f"关键词重叠度计算失败: {e}")
            keyword_sim = np.zeros(len(candidate_texts))

    # 3. SimHash/Hamming距离（简单实现，归一化为相似度）
    def simhash(text):
        # 用hashlib快速近似SimHash
        return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    
    try:
        query_hash = simhash(query_text)
        simhash_sim = []
        for candidate_text in candidate_texts:
            cand_hash = simhash(candidate_text)
            hamming_dist = bin(query_hash ^ cand_hash).count('1')
            sim = 1 - hamming_dist / 128  # 128位hash
            simhash_sim.append(sim)
        simhash_sim = np.array(simhash_sim)
    except Exception as e:
        print(f"SimHash计算失败: {e}")
        simhash_sim = np.zeros(len(candidate_texts))

    # 4. BM25分数（可选，若无则跳过）
    bm25_scores = np.zeros(len(candidate_texts))
    try:
        from rank_bm25 import BM25Okapi
        if jieba is not None:
            tokenized_corpus = [list(jieba.cut(t)) for t in candidate_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(list(jieba.cut(query_text))))
            if bm25_scores.max() > bm25_scores.min():
                bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    except ImportError:
        pass
    except Exception as e:
        print(f"BM25计算失败: {e}")

    # 融合（权重可调，默认均分）
    sims = [tfidf_sim, keyword_sim, simhash_sim, bm25_scores]
    sims = [s if np.std(s) > 0 else np.zeros_like(s) for s in sims]  # 防止全0
    
    # 确保所有相似度数组长度一致
    valid_sims = []
    for s in sims:
        if len(s) == len(candidate_texts):
            valid_sims.append(s)
    
    if valid_sims:
        fused_sim = np.mean(np.stack(valid_sims, axis=0), axis=0)
    else:
        fused_sim = np.zeros(len(candidate_texts))
    
    return fused_sim


def rerank_with_text_similarity(query_text, candidate_texts, original_scores, method='fused', alpha=0.5):
    """
    基于文本相似度重排序（融合多指标）
    Args:
        query_text: 查询文本
        candidate_texts: 候选文本列表
        original_scores: 原始CLIP相似度分数
        method: 文本相似度计算方法，默认'fused'
        alpha: 重排序权重 (0-1, 0表示只用原始分数，1表示只用文本相似度)
    Returns:
        final_scores: 重排序后的分数
        text_similarities: 文本相似度分数
    """
    if len(candidate_texts) == 0 or len(original_scores) == 0:
        return original_scores, np.array([])
    
    # 计算文本相似度
    text_similarities = calculate_text_similarity(query_text, candidate_texts, method)
    
    # 确保数组长度一致
    min_len = min(len(original_scores), len(text_similarities))
    original_scores = original_scores[:min_len]
    text_similarities = text_similarities[:min_len]
    
    # 归一化原始分数和文本相似度
    if original_scores.max() > original_scores.min():
        original_scores_norm = (original_scores - original_scores.min()) / (original_scores.max() - original_scores.min() + 1e-8)
    else:
        original_scores_norm = original_scores
    
    if text_similarities.max() > text_similarities.min():
        text_similarities_norm = (text_similarities - text_similarities.min()) / (text_similarities.max() - text_similarities.min() + 1e-8)
    else:
        text_similarities_norm = text_similarities
    
    # 加权组合
    final_scores = (1 - alpha) * original_scores_norm + alpha * text_similarities_norm
    return final_scores, text_similarities


def rerank_with_image_similarity(query_idx, candidate_indices, all_image_paths, original_scores, device, weights=None):
    """
    基于多指标融合（LPIPS, MS-SSIM, HaarPSI）进行图片相似度重排序
    Args:
        query_idx: 查询图片在all_image_paths中的索引
        candidate_indices: 候选图片在all_image_paths中的索引列表
        all_image_paths: 所有图片路径列表
        original_scores: (K,) np.ndarray，原始CLIP分数
        device: torch.device
        weights: 权重列表，默认[0.4, 0.3, 0.3]
    Returns:
        final_scores: (K,) np.ndarray
        fusion_similarities: (K,) np.ndarray
    """
    if lpips is None or piq is None or transforms is None:
        print("警告: 缺少必要的图像处理库，返回原始分数")
        return original_scores, np.zeros_like(original_scores)
    
    try:
        # 初始化模型
        lpips_model = lpips.LPIPS(net='vgg').eval().to(device)
        
        # 图像预处理
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # 加载查询图像
        query_img = tfm(Image.open(all_image_paths[query_idx]).convert('RGB')).unsqueeze(0).to(device)
        
        fusion_similarities = []
        for idx in candidate_indices:
            try:
                # 加载候选图像
                cand_img = tfm(Image.open(all_image_paths[idx]).convert('RGB')).unsqueeze(0).to(device)
                
                # 计算LPIPS
                lpips_score = lpips_model(query_img, cand_img).item()
                
                # 计算MS-SSIM
                ms_ssim_score = piq.multi_scale_ssim(query_img, cand_img, data_range=1.0).item()
                
                # 计算HaarPSI
                haarpsi_score = piq.haarpsi(query_img, cand_img, data_range=1.0).item()
                
                # 融合相似度分数
                w = weights or [0.4, 0.3, 0.3]
                sim = w[0] * (1 - lpips_score) + w[1] * ms_ssim_score + w[2] * haarpsi_score
                fusion_similarities.append(sim)
            except Exception as e:
                print(f"处理图像 {idx} 时出错: {e}")
                fusion_similarities.append(0.0)
        
        fusion_similarities = np.array(fusion_similarities)
        
        # 归一化
        if original_scores.max() > original_scores.min():
            original_scores_norm = (original_scores - original_scores.min()) / (original_scores.max() - original_scores.min() + 1e-8)
        else:
            original_scores_norm = original_scores
        
        if fusion_similarities.max() > fusion_similarities.min():
            fusion_similarities_norm = (fusion_similarities - fusion_similarities.min()) / (fusion_similarities.max() - fusion_similarities.min() + 1e-8)
        else:
            fusion_similarities_norm = fusion_similarities
        
        # 加权组合
        alpha = 0.5  # 可调
        final_scores = (1 - alpha) * original_scores_norm + alpha * fusion_similarities_norm
        
        return final_scores, fusion_similarities
    
    except Exception as e:
        print(f"图像相似度计算失败: {e}")
        return original_scores, np.zeros_like(original_scores)


def calculate_metrics(ranks):
    """
    计算检索指标
    Args:
        ranks: 排名列表
    Returns:
        r1, r5, r10: R@1, R@5, R@10指标
    """
    if not ranks:
        return 0, 0, 0
    
    r1 = sum(1 for r in ranks if r == 1) / len(ranks)
    r5 = sum(1 for r in ranks if r <= 5) / len(ranks)
    r10 = sum(1 for r in ranks if r <= 10) / len(ranks)
    return r1, r5, r10


# ==================== 过滤器类 ====================

class TextSimilarityFilter:
    """
    文本相似度过滤器类
    """
    
    def __init__(self, method='fused', alpha=0.5):
        """
        初始化文本相似度过滤器
        Args:
            method: 相似度计算方法
            alpha: 重排序权重
        """
        self.method = method
        self.alpha = alpha
        self.logger = setup_logger(self.__class__.__name__)
    
    def filter(self, query_text, candidate_texts, original_scores):
        """
        执行文本相似度过滤
        Args:
            query_text: 查询文本
            candidate_texts: 候选文本列表
            original_scores: 原始分数
        Returns:
            filtered_scores: 过滤后的分数
            similarities: 相似度分数
        """
        return rerank_with_text_similarity(
            query_text, candidate_texts, original_scores, self.method, self.alpha
        )
    
    def filter_text_to_image(self, query_text, retrieval_results, similarity_threshold=0.1):
        """
        文本到图像检索的过滤
        Args:
            query_text: 查询文本
            retrieval_results: 检索结果列表
            similarity_threshold: 相似度阈值
        Returns:
            filtered_results: 过滤后的结果列表
        """
        if not retrieval_results:
            return []
        
        try:
            # 提取候选文本和原始分数
            candidate_texts = [result['text'] for result in retrieval_results]
            original_scores = np.array([result['score'] for result in retrieval_results])
            
            # 执行文本相似度过滤
            filtered_scores, text_similarities = self.filter(query_text, candidate_texts, original_scores)
            
            # 应用阈值过滤
            filtered_results = []
            for i, (result, score, sim) in enumerate(zip(retrieval_results, filtered_scores, text_similarities)):
                if sim >= similarity_threshold:
                    result_copy = result.copy()
                    result_copy['score'] = score
                    result_copy['text_similarity_score'] = sim
                    filtered_results.append(result_copy)
            
            # 按分数排序
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            
            return filtered_results
        
        except Exception as e:
            self.logger.error(f"文本过滤失败: {e}")
            return retrieval_results


class ImageSimilarityFilter:
    """
    图像相似度过滤器类
    """
    
    def __init__(self, device=None, weights=None):
        """
        初始化图像相似度过滤器
        Args:
            device: 计算设备
            weights: 融合权重
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights = weights or [0.4, 0.3, 0.3]
        self.logger = setup_logger(self.__class__.__name__)
    
    def filter(self, query_idx, candidate_indices, all_image_paths, original_scores):
        """
        执行图像相似度过滤
        Args:
            query_idx: 查询图像索引
            candidate_indices: 候选图像索引列表
            all_image_paths: 所有图像路径
            original_scores: 原始分数
        Returns:
            filtered_scores: 过滤后的分数
            similarities: 相似度分数
        """
        return rerank_with_image_similarity(
            query_idx, candidate_indices, all_image_paths, original_scores, self.device, self.weights
        )
    
    def filter_image_to_image(self, query_image_path, retrieval_results, similarity_threshold=0.1):
        """
        图像到图像检索的过滤
        Args:
            query_image_path: 查询图像路径
            retrieval_results: 检索结果列表
            similarity_threshold: 相似度阈值
        Returns:
            filtered_results: 过滤后的结果列表
        """
        if not retrieval_results or not os.path.exists(query_image_path):
            return retrieval_results
        
        try:
            # 构建图像路径列表
            all_image_paths = [query_image_path]
            candidate_indices = []
            valid_results = []
            
            for i, result in enumerate(retrieval_results):
                image_path = result.get('image_path')
                if image_path and os.path.exists(image_path):
                    all_image_paths.append(image_path)
                    candidate_indices.append(len(all_image_paths) - 1)
                    valid_results.append(result)
            
            if not valid_results:
                return []
            
            # 提取原始分数
            original_scores = np.array([result['score'] for result in valid_results])
            
            # 执行图像相似度过滤
            filtered_scores, image_similarities = self.filter(
                0, candidate_indices, all_image_paths, original_scores
            )
            
            # 应用阈值过滤
            filtered_results = []
            for i, (result, score, sim) in enumerate(zip(valid_results, filtered_scores, image_similarities)):
                if sim >= similarity_threshold:
                    result_copy = result.copy()
                    result_copy['score'] = score
                    result_copy['image_similarity_score'] = sim
                    filtered_results.append(result_copy)
            
            # 按分数排序
            filtered_results.sort(key=lambda x: x['score'], reverse=True)
            
            return filtered_results
        
        except Exception as e:
            self.logger.error(f"图像过滤失败: {e}")
            return retrieval_results


class MultiModalFilter:
    """
    多模态过滤器类，结合文本和图像相似度
    """
    
    def __init__(self, text_weight=0.3, image_weight=0.7, text_filter=None, image_filter=None):
        """
        初始化多模态过滤器
        Args:
            text_weight: 文本权重
            image_weight: 图像权重
            text_filter: 文本相似度过滤器
            image_filter: 图像相似度过滤器
        """
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.text_filter = text_filter or TextSimilarityFilter()
        self.image_filter = image_filter or ImageSimilarityFilter()
        self.logger = setup_logger(self.__class__.__name__)
    
    def filter_text_to_image(self, query_text, retrieval_results, similarity_threshold=0.1):
        """
        文本到图像检索的过滤
        Args:
            query_text: 查询文本
            retrieval_results: 检索结果列表
            similarity_threshold: 相似度阈值
        Returns:
            filtered_results: 过滤后的结果列表
        """
        if not retrieval_results:
            return []
        
        try:
            # 使用文本过滤器进行过滤
            filtered_results = self.text_filter.filter_text_to_image(
                query_text, retrieval_results, similarity_threshold
            )
            
            # 添加多模态标记
            for result in filtered_results:
                result['multimodal_similarity_score'] = (
                    self.text_weight * result.get('text_similarity_score', 0)
                )
            
            return filtered_results
        
        except Exception as e:
            self.logger.error(f"多模态文本到图像过滤失败: {e}")
            return retrieval_results
    
    def filter_image_to_text(self, query_image_path, retrieval_results, similarity_threshold=0.1):
        """
        图像到文本检索的过滤
        Args:
            query_image_path: 查询图像路径
            retrieval_results: 检索结果列表
            similarity_threshold: 相似度阈值
        Returns:
            filtered_results: 过滤后的结果列表
        """
        if not retrieval_results:
            return []
        
        try:
            # 分别进行文本和图像过滤
            text_results = []
            image_results = []
            
            # 提取查询文本（如果有的话）
            query_text = ""  # 可以从图像描述中提取
            
            if query_text:
                text_results = self.text_filter.filter_text_to_image(
                    query_text, retrieval_results, similarity_threshold
                )
            
            if query_image_path and os.path.exists(query_image_path):
                image_results = self.image_filter.filter_image_to_image(
                    query_image_path, retrieval_results, similarity_threshold
                )
            
            # 融合结果
            if text_results and image_results:
                # 创建结果字典用于快速查找
                text_dict = {r['id']: r for r in text_results}
                image_dict = {r['id']: r for r in image_results}
                
                filtered_results = []
                for result_id in set(text_dict.keys()) | set(image_dict.keys()):
                    text_result = text_dict.get(result_id)
                    image_result = image_dict.get(result_id)
                    
                    if text_result and image_result:
                        # 融合分数
                        combined_score = (
                            self.text_weight * text_result['score'] +
                            self.image_weight * image_result['score']
                        )
                        
                        result = text_result.copy()
                        result['score'] = combined_score
                        result['multimodal_similarity_score'] = (
                            self.text_weight * text_result.get('text_similarity_score', 0) +
                            self.image_weight * image_result.get('image_similarity_score', 0)
                        )
                        filtered_results.append(result)
                    elif text_result:
                        result = text_result.copy()
                        result['multimodal_similarity_score'] = (
                            self.text_weight * text_result.get('text_similarity_score', 0)
                        )
                        filtered_results.append(result)
                    elif image_result:
                        result = image_result.copy()
                        result['multimodal_similarity_score'] = (
                            self.image_weight * image_result.get('image_similarity_score', 0)
                        )
                        filtered_results.append(result)
                
                # 按分数排序
                filtered_results.sort(key=lambda x: x['score'], reverse=True)
                return filtered_results
            
            elif text_results:
                return text_results
            elif image_results:
                return image_results
            else:
                return retrieval_results
        
        except Exception as e:
            self.logger.error(f"多模态图像到文本过滤失败: {e}")
            return retrieval_results


# ==================== 图像检索过滤器主类 ====================

class ImageRetrievalFilter:
    """
    图像检索结果过滤器
    
    集成所有过滤机制，为图像检索结果提供重排序功能
    """
    
    def __init__(self, 
                 text_weight: float = 0.3,
                 image_weight: float = 0.7,
                 enable_text_filter: bool = True,
                 enable_image_filter: bool = True):
        """
        初始化图像检索过滤器
        
        Args:
            text_weight: 文本相似度权重
            image_weight: 图像相似度权重
            enable_text_filter: 是否启用文本过滤
            enable_image_filter: 是否启用图像过滤
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.enable_text_filter = enable_text_filter
        self.enable_image_filter = enable_image_filter
        
        # 初始化过滤器
        self.text_filter = None
        self.image_filter = None
        self.multimodal_filter = None
        
        self._initialize_filters()
    
    def _initialize_filters(self):
        """初始化过滤器组件"""
        try:
            if self.enable_text_filter:
                self.text_filter = TextSimilarityFilter()
                self.logger.info("文本相似度过滤器初始化成功")
            
            if self.enable_image_filter:
                self.image_filter = ImageSimilarityFilter()
                self.logger.info("图像相似度过滤器初始化成功")
            
            if self.enable_text_filter and self.enable_image_filter:
                self.multimodal_filter = MultiModalFilter(
                    text_weight=self.text_weight,
                    image_weight=self.image_weight,
                    text_filter=self.text_filter,
                    image_filter=self.image_filter
                )
                self.logger.info("多模态过滤器初始化成功")
                
        except Exception as e:
            self.logger.error(f"过滤器初始化失败: {e}")
    
    def filter_by_text_similarity(self, 
                                 query: str, 
                                 results: List[RetrievalResult],
                                 similarity_threshold: float = 0.1) -> List[RetrievalResult]:
        """
        基于文本相似度过滤图像检索结果
        
        Args:
            query: 查询文本
            results: 检索结果列表
            similarity_threshold: 相似度阈值
            
        Returns:
            过滤后的结果列表
        """
        if not self.text_filter or not results:
            return results
        
        try:
            self.logger.info(f"开始基于文本相似度过滤，原始结果数量: {len(results)}")
            
            # 准备数据格式
            retrieval_results = []
            for result in results:
                # 提取图片描述作为文本内容
                content = result.content
                if "图片描述:" in content:
                    description = content.split("图片描述:")[1].split("\n")[0].strip()
                else:
                    description = content
                
                retrieval_results.append({
                    'id': result.doc_id,
                    'text': description,
                    'score': result.score,
                    'metadata': result.metadata
                })
            
            # 使用文本过滤器进行过滤
            filtered_results = self.text_filter.filter_text_to_image(
                query_text=query,
                retrieval_results=retrieval_results,
                similarity_threshold=similarity_threshold
            )
            
            # 转换回RetrievalResult格式
            filtered_retrieval_results = []
            for filtered_result in filtered_results:
                # 找到原始结果
                original_result = next(
                    (r for r in results if r.doc_id == filtered_result['id']), 
                    None
                )
                if original_result:
                    # 更新分数和元数据
                    original_result.score = filtered_result['score']
                    original_result.metadata.update({
                        'text_similarity_score': filtered_result.get('text_similarity_score', 0),
                        'filtered_by': 'text_similarity'
                    })
                    filtered_retrieval_results.append(original_result)
            
            self.logger.info(f"文本相似度过滤完成，过滤后结果数量: {len(filtered_retrieval_results)}")
            return filtered_retrieval_results
            
        except Exception as e:
            self.logger.error(f"文本相似度过滤失败: {e}")
            return results
    
    def filter_by_image_similarity(self, 
                                  query_image_path: str, 
                                  results: List[RetrievalResult],
                                  similarity_threshold: float = 0.1) -> List[RetrievalResult]:
        """
        基于图像相似度过滤图像检索结果
        
        Args:
            query_image_path: 查询图像路径
            results: 检索结果列表
            similarity_threshold: 相似度阈值
            
        Returns:
            过滤后的结果列表
        """
        if not self.image_filter or not results or not query_image_path:
            return results
        
        try:
            self.logger.info(f"开始基于图像相似度过滤，原始结果数量: {len(results)}")
            
            # 准备数据格式
            retrieval_results = []
            for result in results:
                # 提取图片路径
                image_path = None
                if "图片路径:" in result.content:
                    image_path = result.content.split("图片路径:")[1].strip()
                elif 'image_path' in result.metadata:
                    image_path = result.metadata['image_path']
                
                if image_path and os.path.exists(image_path):
                    retrieval_results.append({
                        'id': result.doc_id,
                        'image_path': image_path,
                        'score': result.score,
                        'metadata': result.metadata
                    })
            
            # 使用图像过滤器进行过滤
            filtered_results = self.image_filter.filter_image_to_image(
                query_image_path=query_image_path,
                retrieval_results=retrieval_results,
                similarity_threshold=similarity_threshold
            )
            
            # 转换回RetrievalResult格式
            filtered_retrieval_results = []
            for filtered_result in filtered_results:
                # 找到原始结果
                original_result = next(
                    (r for r in results if r.doc_id == filtered_result['id']), 
                    None
                )
                if original_result:
                    # 更新分数和元数据
                    original_result.score = filtered_result['score']
                    original_result.metadata.update({
                        'image_similarity_score': filtered_result.get('image_similarity_score', 0),
                        'filtered_by': 'image_similarity'
                    })
                    filtered_retrieval_results.append(original_result)
            
            self.logger.info(f"图像相似度过滤完成，过滤后结果数量: {len(filtered_retrieval_results)}")
            return filtered_retrieval_results
            
        except Exception as e:
            self.logger.error(f"图像相似度过滤失败: {e}")
            return results
    
    def filter_multimodal(self, 
                         query_text: str,
                         query_image_path: Optional[str],
                         results: List[RetrievalResult],
                         similarity_threshold: float = 0.1) -> List[RetrievalResult]:
        """
        基于多模态相似度过滤图像检索结果
        
        Args:
            query_text: 查询文本
            query_image_path: 查询图像路径（可选）
            results: 检索结果列表
            similarity_threshold: 相似度阈值
            
        Returns:
            过滤后的结果列表
        """
        if not self.multimodal_filter or not results:
            return results
        
        try:
            self.logger.info(f"开始基于多模态相似度过滤，原始结果数量: {len(results)}")
            
            # 准备数据格式
            retrieval_results = []
            for result in results:
                # 提取图片描述和路径
                content = result.content
                description = ""
                image_path = None
                
                if "图片描述:" in content:
                    description = content.split("图片描述:")[1].split("\n")[0].strip()
                
                if "图片路径:" in content:
                    image_path = content.split("图片路径:")[1].strip()
                elif 'image_path' in result.metadata:
                    image_path = result.metadata['image_path']
                
                retrieval_results.append({
                    'id': result.doc_id,
                    'text': description,
                    'image_path': image_path if image_path and os.path.exists(image_path) else None,
                    'score': result.score,
                    'metadata': result.metadata
                })
            
            # 使用多模态过滤器进行过滤
            if query_image_path:
                # 图像到图像+文本的检索
                filtered_results = self.multimodal_filter.filter_image_to_text(
                    query_image_path=query_image_path,
                    retrieval_results=retrieval_results,
                    similarity_threshold=similarity_threshold
                )
            else:
                # 文本到图像的检索
                filtered_results = self.multimodal_filter.filter_text_to_image(
                    query_text=query_text,
                    retrieval_results=retrieval_results,
                    similarity_threshold=similarity_threshold
                )
            
            # 转换回RetrievalResult格式
            filtered_retrieval_results = []
            for filtered_result in filtered_results:
                # 找到原始结果
                original_result = next(
                    (r for r in results if r.doc_id == filtered_result['id']), 
                    None
                )
                if original_result:
                    # 更新分数和元数据
                    original_result.score = filtered_result['score']
                    original_result.metadata.update({
                        'multimodal_similarity_score': filtered_result.get('multimodal_similarity_score', 0),
                        'text_similarity_score': filtered_result.get('text_similarity_score', 0),
                        'image_similarity_score': filtered_result.get('image_similarity_score', 0),
                        'filtered_by': 'multimodal'
                    })
                    filtered_retrieval_results.append(original_result)
            
            self.logger.info(f"多模态相似度过滤完成，过滤后结果数量: {len(filtered_retrieval_results)}")
            return filtered_retrieval_results
            
        except Exception as e:
            self.logger.error(f"多模态相似度过滤失败: {e}")
            return results
    
    def rerank_results(self, 
                      query_text: str,
                      query_image_path: Optional[str],
                      results: List[RetrievalResult],
                      filter_type: str = "multimodal",
                      similarity_threshold: float = 0.1) -> List[RetrievalResult]:
        """
        对图像检索结果进行重排序
        
        Args:
            query_text: 查询文本
            query_image_path: 查询图像路径（可选）
            results: 检索结果列表
            filter_type: 过滤类型 ("text", "image", "multimodal")
            similarity_threshold: 相似度阈值
            
        Returns:
            重排序后的结果列表
        """
        if not results:
            return results
        
        self.logger.info(f"开始对图像检索结果进行重排序，过滤类型: {filter_type}")
        
        try:
            if filter_type == "text" and self.enable_text_filter:
                return self.filter_by_text_similarity(query_text, results, similarity_threshold)
            elif filter_type == "image" and self.enable_image_filter and query_image_path:
                return self.filter_by_image_similarity(query_image_path, results, similarity_threshold)
            elif filter_type == "multimodal" and (self.enable_text_filter or self.enable_image_filter):
                return self.filter_multimodal(query_text, query_image_path, results, similarity_threshold)
            else:
                self.logger.warning(f"不支持的过滤类型或过滤器未启用: {filter_type}")
                return results
                
        except Exception as e:
            self.logger.error(f"重排序失败: {e}")
            return results
    
    def get_filter_stats(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        获取过滤统计信息
        
        Args:
            results: 检索结果列表
            
        Returns:
            统计信息字典
        """
        stats = {
            'total_results': len(results),
            'filtered_by_text': 0,
            'filtered_by_image': 0,
            'filtered_by_multimodal': 0,
            'avg_score': 0,
            'score_range': [0, 0]
        }
        
        if not results:
            return stats
        
        scores = [r.score for r in results]
        stats['avg_score'] = np.mean(scores)
        stats['score_range'] = [min(scores), max(scores)]
        
        for result in results:
            filtered_by = result.metadata.get('filtered_by', '')
            if filtered_by == 'text_similarity':
                stats['filtered_by_text'] += 1
            elif filtered_by == 'image_similarity':
                stats['filtered_by_image'] += 1
            elif filtered_by == 'multimodal':
                stats['filtered_by_multimodal'] += 1
        
        return stats


# ==================== 工厂函数 ====================

def create_image_filter(text_weight: float = 0.3,
                       image_weight: float = 0.7,
                       enable_text_filter: bool = True,
                       enable_image_filter: bool = True) -> ImageRetrievalFilter:
    """
    创建图像检索过滤器实例
    
    Args:
        text_weight: 文本相似度权重
        image_weight: 图像相似度权重
        enable_text_filter: 是否启用文本过滤
        enable_image_filter: 是否启用图像过滤
        
    Returns:
        图像检索过滤器实例
    """
    return ImageRetrievalFilter(
        text_weight=text_weight,
        image_weight=image_weight,
        enable_text_filter=enable_text_filter,
        enable_image_filter=enable_image_filter
    )


# ==================== 融合相似度计算函数 ====================

def calculate_text_similarity_fusion(query_text, candidate_texts, method='fused'):
    """
    计算文本相似度融合分数（兼容性函数）
    """
    return calculate_text_similarity(query_text, candidate_texts, method)


def calculate_image_similarity_fusion(query_idx, candidate_indices, all_image_paths, device, weights=None):
    """
    计算图像相似度融合分数（兼容性函数）
    """
    # 创建虚拟原始分数
    original_scores = np.ones(len(candidate_indices))
    _, fusion_similarities = rerank_with_image_similarity(
        query_idx, candidate_indices, all_image_paths, original_scores, device, weights
    )
    return fusion_similarities