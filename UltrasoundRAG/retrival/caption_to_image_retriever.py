"""
Caption检索图片模块 - 统一的图片标题提取和检索功能

主要功能：
1. 从文本中提取图片标题
2. 处理多种caption变体格式
3. 支持多种搜索模式（精确、模糊、语义、混合）
4. 批量检索和文本块检索
5. 精确匹配图片（支持变体匹配）

特点：
- 支持多种caption格式：原样、去注、JSON数组等
- 智能变体匹配，避免过于严格导致匹配失败
- 统一的接口，可在不同检索器中使用
- 精确匹配，避免一找多或找错的情况
"""

import re
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from UltrasoundRAG.utils.logger import setup_logger
from UltrasoundRAG.retrival.data_structures import RetrievalResult
from UltrasoundRAG.milvus.milvus_manager import MilvusManager
from UltrasoundRAG.utils.embedding_utils import embedding_provider
from UltrasoundRAG.model.model_manager import get_fetal_clip_model
from UltrasoundRAG.config import config


class CaptionMatchMode(Enum):
    """Caption匹配模式"""
    EXACT = "exact"  # 精确匹配
    FUZZY = "fuzzy"  # 模糊匹配
    VARIANT = "variant"  # 变体匹配（推荐）


class SearchMode(Enum):
    """搜索模式枚举"""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    HYBRID_MATCH = "hybrid_match"  # 混合匹配


@dataclass
class CaptionMatchConfig:
    """Caption匹配配置"""
    match_mode: CaptionMatchMode = CaptionMatchMode.VARIANT
    enable_note_removal: bool = True  # 是否移除"注："部分
    enable_json_parsing: bool = True  # 是否解析JSON格式
    enable_whitespace_normalization: bool = True  # 是否标准化空白字符
    similarity_threshold: float = 0.8  # 相似度阈值（用于模糊匹配）
    max_candidates: int = 5  # 最大候选数量


@dataclass
class CaptionSearchConfig:
    """Caption搜索配置"""
    search_mode: SearchMode = SearchMode.EXACT_MATCH
    fuzzy_threshold: float = 0.8
    semantic_threshold: float = 0.7
    max_results_per_caption: int = 3
    enable_keyword_boost: bool = True
    keyword_boost_weight: float = 1.5


class ImageTitleExtractor:
    """图片标题提取器"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        
        # 图片标题的正则表达式模式
        self.title_patterns = [
            r'图\s*(\d+[-~]\d+)\s*[：:]\s*([^。\n\r！？；，]{5,100})',  # 图X-X: 标题
            r'Figure\s+(\d+\.\d+)\s*[：:]\s*([^。\n\r！？；，]{5,100})',  # Figure X.X: 标题
            r'图\s*(\d+)\s*[：:]\s*([^。\n\r！？；，]{5,100})',  # 图X: 标题
            r'图片\s*(\d+)\s*[：:]\s*([^。\n\r！？；，]{5,100})',  # 图片X: 标题
            r'图\s*(\d+[-~]\d+)\s+([^。\n\r！？；，]{5,100})',  # 图X-X 标题
            r'图\s*(\d+)\s+([^。\n\r！？；，]{5,100})',  # 图X 标题
        ]
    
    def extract_from_text(self, text: str) -> List[str]:
        """
        从文本中提取图片标题
        
        Args:
            text: 输入文本
            
        Returns:
            提取的图片标题列表
        """
        titles = []
        
        for pattern in self.title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    # 组合编号和标题
                    title = f"图{match[0]} {match[1]}".strip()
                    title = self._clean_title(title)
                    if title and len(title) > 3:
                        titles.append(title)
        
        return list(set(titles))  # 去重
    
    def extract_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        从元数据中提取图片标题
        
        Args:
            metadata: 元数据字典
            
        Returns:
            提取的图片标题列表
        """
        titles = []
        
        # 从image_captions字段提取
        image_captions = metadata.get('image_captions', [])
        if isinstance(image_captions, list):
            for caption in image_captions:
                if caption and isinstance(caption, str):
                    clean_caption = self._clean_title(caption)
                    if clean_caption and len(clean_caption) > 3:
                        titles.append(clean_caption)
        elif isinstance(image_captions, str):
            # 如果是字符串，尝试解析为JSON
            try:
                captions_list = json.loads(image_captions)
                if isinstance(captions_list, list):
                    for caption in captions_list:
                        if caption and isinstance(caption, str):
                            clean_caption = self._clean_title(caption)
                            if clean_caption and len(clean_caption) > 3:
                                titles.append(clean_caption)
            except (json.JSONDecodeError, TypeError):
                # 直接作为字符串处理
                clean_caption = self._clean_title(image_captions)
                if clean_caption and len(clean_caption) > 3:
                    titles.append(clean_caption)
        
        return list(set(titles))  # 去重
    
    def _clean_title(self, title: str) -> str:
        """清理图片标题"""
        if not title:
            return ""
        
        # 移除多余的空格
        title = re.sub(r'\s+', ' ', title)
        
        # 移除末尾的标点符号
        title = re.sub(r'[，。！？；、:：]+$', '', title)
        
        # 移除开头和结尾的空格
        title = title.strip()
        
        return title


class CaptionVariantMatcher:
    """Caption变体匹配器 - 处理多种caption格式"""
    
    def __init__(self, config: CaptionMatchConfig = None):
        self.config = config or CaptionMatchConfig()
        self.logger = setup_logger(self.__class__.__name__)
    
    def generate_variants(self, caption: str) -> List[str]:
        """
        生成caption的变体列表
        
        Args:
            caption: 原始caption
            
        Returns:
            caption变体列表
        """
        variants = []
        
        if not caption:
            return variants
        
        # 1. 原样
        variants.append(caption.strip())
        
        # 2. 处理JSON数组格式
        if self.config.enable_json_parsing:
            json_variants = self._extract_from_json_array(caption)
            variants.extend(json_variants)
        
        # 3. 移除"注："部分
        if self.config.enable_note_removal:
            note_removed = self._remove_note_section(caption)
            if note_removed and note_removed not in variants:
                variants.append(note_removed)
        
        # 4. 标准化空白字符
        if self.config.enable_whitespace_normalization:
            normalized = self._normalize_whitespace(caption)
            if normalized and normalized not in variants:
                variants.append(normalized)
        
        # 5. 组合变体：去注 + 标准化
        if self.config.enable_note_removal and self.config.enable_whitespace_normalization:
            combined = self._normalize_whitespace(self._remove_note_section(caption))
            if combined and combined not in variants:
                variants.append(combined)
        
        # 去重并过滤空值
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant and variant.strip() and variant not in seen:
                seen.add(variant)
                unique_variants.append(variant.strip())
        
        return unique_variants[:self.config.max_candidates]
    
    def _extract_from_json_array(self, caption: str) -> List[str]:
        """从JSON数组格式中提取caption"""
        variants = []
        
        try:
            # 尝试解析JSON
            if caption.strip().startswith('[') and caption.strip().endswith(']'):
                data = json.loads(caption)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str) and item.strip():
                            variants.append(item.strip())
        except (json.JSONDecodeError, TypeError):
            pass
        
        return variants
    
    def _remove_note_section(self, caption: str) -> str:
        """移除"注："部分"""
        for marker in ['注：', '注:', 'Note:', 'Note：']:
            idx = caption.find(marker)
            if idx != -1:
                return caption[:idx].strip()
        return caption
    
    def _normalize_whitespace(self, caption: str) -> str:
        """标准化空白字符"""
        return ' '.join(caption.split())


class CaptionToImageRetriever:
    """Caption到图片检索器 - 统一的图片标题提取和检索功能"""
    
    def __init__(self, 
                 db_name: str = "default",
                 milvus_uri: Optional[str] = None,
                 milvus_token: Optional[str] = None,
                 match_config: CaptionMatchConfig = None):
        """
        初始化Caption检索器
        
        Args:
            db_name: 数据库配置名称
            milvus_uri: Milvus服务地址（可选）
            milvus_token: Milvus认证令牌（可选）
            match_config: 匹配配置（可选）
        """
        self.logger = setup_logger(self.__class__.__name__)
        
        # 获取数据库配置
        retriever_config = config['retriever']
        databases_config = retriever_config['databases']
        
        if db_name not in databases_config:
            db_name = "default"
        
        db_config = databases_config[db_name]
        
        # 获取Milvus连接信息
        milvus_config = config['milvus']
        self.milvus_uri = milvus_uri or milvus_config['milvus_uri']
        self.milvus_token = milvus_token or milvus_config['milvus_token']
        
        # 初始化图片集合管理器
        self.image_manager = MilvusManager(
            milvus_uri=self.milvus_uri,
            milvus_token=self.milvus_token,
            db_name=db_config['db_name'],
            collection_type="image"
        )
        
        # 获取Caption检索配置
        caption_config = retriever_config['caption_to_image']
        self.search_config = CaptionSearchConfig(
            search_mode=SearchMode(caption_config['search_mode']),
            fuzzy_threshold=caption_config['fuzzy_threshold'],
            max_results_per_caption=caption_config['max_results_per_caption']
        )
        
        # 初始化匹配配置
        self.match_config = match_config or CaptionMatchConfig()
        
        # 初始化CLIP模型（用于语义匹配）
        self.clip_model = None
        self._ensure_clip_initialized(self.search_config.search_mode)
        
        # 初始化子组件
        self.extractor = ImageTitleExtractor()
        self.variant_matcher = CaptionVariantMatcher(self.match_config)
        
        self.logger.info(f"Caption检索器初始化完成，搜索模式: {self.search_config.search_mode.value}")

    def _ensure_clip_initialized(self, search_mode: SearchMode) -> None:
        """在需要语义/混合匹配时，确保CLIP已被初始化（惰性初始化）。"""
        if search_mode in [SearchMode.SEMANTIC_MATCH, SearchMode.HYBRID_MATCH] and self.clip_model is None:
            try:
                # 使用共享模型实例，避免重复加载
                self.clip_model = get_fetal_clip_model()
                self.logger.info("CLIP模型初始化成功，支持语义匹配")
            except Exception as e:
                self.logger.warning(f"CLIP模型初始化失败: {e}，将回退到非语义匹配模式")
                self.clip_model = None
    
    def search_single_caption(self, caption: str, 
                            top_k: Optional[int] = None,
                            search_mode: Optional[SearchMode] = None) -> Dict[str, Any]:
        """
        根据单个caption检索图片
        
        Args:
            caption: 图片标题/描述
            top_k: 返回结果数量
            search_mode: 搜索模式（可覆盖默认配置）
            
        Returns:
            检索结果字典
        """
        top_k = top_k or self.search_config.max_results_per_caption
        search_mode = search_mode or self.search_config.search_mode
        
        try:
            if search_mode == SearchMode.EXACT_MATCH:
                results = self._exact_match_search(caption, top_k)
            elif search_mode == SearchMode.FUZZY_MATCH:
                results = self._fuzzy_match_search(caption, top_k)
            elif search_mode == SearchMode.SEMANTIC_MATCH:
                results = self._semantic_match_search(caption, top_k)
            elif search_mode == SearchMode.HYBRID_MATCH:
                results = self._hybrid_match_search(caption, top_k)
            else:
                raise ValueError(f"不支持的搜索模式: {search_mode}")
            
            return {
                'caption': caption,
                'search_mode': search_mode.value,
                'results': results,
                'total_results': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Caption检索失败 '{caption}': {e}")
            return self._empty_result(caption, str(e))
    
    def search_multiple_captions(self, captions: List[str], 
                                top_k: Optional[int] = None,
                                search_mode: Optional[SearchMode] = None) -> Dict[str, Any]:
        """
        根据多个caption批量检索图片
        
        Args:
            captions: 图片标题/描述列表
            top_k: 每个caption返回的结果数量
            search_mode: 搜索模式
            
        Returns:
            批量检索结果字典
        """
        results_by_caption = {}
        all_results = []
        
        for caption in captions:
            single_result = self.search_single_caption(caption, top_k, search_mode)
            results_by_caption[caption] = single_result
            all_results.extend(single_result['results'])
        
        # 对所有结果按分数排序并去重
        all_results = self._deduplicate_results(all_results)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return {
            'captions': captions,
            'search_mode': (search_mode or self.search_config.search_mode).value,
            'results_by_caption': results_by_caption,
            'all_results': all_results,
            'total_captions': len(captions),
            'total_results': len(all_results)
        }
    
    def search_from_text_chunk(self, text_chunk: str, 
                              top_k: Optional[int] = None,
                              search_mode: Optional[SearchMode] = None) -> Dict[str, Any]:
        """
        从文本块中提取caption并检索对应图片
        
        Args:
            text_chunk: 包含图片引用的文本块
            top_k: 每个caption返回的结果数量
            search_mode: 搜索模式
            
        Returns:
            检索结果字典
        """
        # 提取图片标题
        titles = self.extractor.extract_from_text(text_chunk)
        
        if not titles:
            return {
                'text_chunk': text_chunk[:100] + "..." if len(text_chunk) > 100 else text_chunk,
                'extracted_captions': [],
                'results': [],
                'total_results': 0,
                'message': '未提取到图片标题'
            }
        
        # 匹配图片
        all_matches = []
        for title in titles:
            matches = self._match_single_title(title, top_k or self.search_config.max_results_per_caption, search_mode)
            all_matches.extend(matches)
        
        # 去重
        unique_matches = self._deduplicate_matches(all_matches)
        
        return {
            'text_chunk': text_chunk[:100] + "..." if len(text_chunk) > 100 else text_chunk,
            'extracted_captions': titles,
            'results': unique_matches,
            'total_results': len(unique_matches),
            'match_details': {
                'titles_processed': len(titles),
                'total_candidates': len(all_matches),
                'unique_matches': len(unique_matches)
            }
        }
    
    def _exact_match_search(self, caption: str, top_k: int) -> List[RetrievalResult]:
        """精确匹配搜索"""
        try:
            # 使用内部匹配方法进行精确匹配
            matches = self._match_single_title(caption, top_k, SearchMode.EXACT_MATCH)
            return matches
            
        except Exception as e:
            self.logger.error(f"精确匹配搜索失败: {e}")
            return []
    
    def _match_single_title(self, title: str, top_k: int, search_mode: Optional[SearchMode] = None) -> List[RetrievalResult]:
        """
        匹配单个图片标题
        
        Args:
            title: 图片标题
            top_k: 返回结果数量
            search_mode: 搜索模式
            
        Returns:
            匹配的图片结果列表
        """
        if not title:
            return []
        
        search_mode = search_mode or self.search_config.search_mode
        
        if search_mode == SearchMode.EXACT_MATCH:
            return self._exact_title_match(title, top_k)
        elif search_mode == SearchMode.FUZZY_MATCH:
            return self._fuzzy_title_match(title, top_k)
        elif search_mode == SearchMode.SEMANTIC_MATCH:
            return self._semantic_title_match(title, top_k)
        elif search_mode == SearchMode.HYBRID_MATCH:
            return self._hybrid_title_match(title, top_k)
        else:
            return self._exact_title_match(title, top_k)
    
    def _exact_title_match(self, title: str, top_k: int) -> List[RetrievalResult]:
        """精确标题匹配"""
        # 生成变体
        variants = self.variant_matcher.generate_variants(title)
        
        all_matches = []
        for variant in variants:
            try:
                matches = self._search_caption_variant(variant, top_k)
                all_matches.extend(matches)
                
                # 如果找到匹配，优先返回
                if matches:
                    break
                    
            except Exception as e:
                self.logger.warning(f"匹配变体 '{variant}' 失败: {e}")
        
        return all_matches[:top_k]
    
    def _fuzzy_title_match(self, title: str, top_k: int) -> List[RetrievalResult]:
        """模糊标题匹配"""
        # 生成变体
        variants = self.variant_matcher.generate_variants(title)
        
        all_matches = []
        for variant in variants:
            try:
                # 提取关键词进行模糊匹配
                keywords = self._extract_keywords(variant)
                
                # 构建模糊匹配表达式
                filter_expressions = []
                for keyword in keywords:
                    filter_expressions.append(f'caption like "%{keyword}%"')
                
                if not filter_expressions:
                    continue
                
                # 使用OR逻辑连接多个关键词
                filter_expr = " || ".join(filter_expressions)
                
                # 执行过滤搜索
                results = self.image_manager.search_with_filter(
                    filter_expr=filter_expr,
                    limit=top_k * 2  # 获取更多结果用于后续排序
                )
                
                # 计算模糊匹配分数并排序
                scored_results = []
                for result in results:
                    score = self._calculate_fuzzy_score(variant, result.get('caption', ''))
                    if score >= self.search_config.fuzzy_threshold:
                        result['fuzzy_score'] = score
                        scored_results.append(result)
                
                # 按模糊匹配分数排序
                scored_results.sort(key=lambda x: x['fuzzy_score'], reverse=True)
                
                matches = self._convert_milvus_results(scored_results[:top_k], 'fuzzy_match')
                all_matches.extend(matches)
                
                # 如果找到匹配，优先返回
                if matches:
                    break
                    
            except Exception as e:
                self.logger.warning(f"模糊匹配变体 '{variant}' 失败: {e}")
        
        return all_matches[:top_k]
    
    def _semantic_title_match(self, title: str, top_k: int) -> List[RetrievalResult]:
        """语义标题匹配"""
        self._ensure_clip_initialized(SearchMode.SEMANTIC_MATCH)
        if not self.clip_model:
            self.logger.warning("CLIP模型未初始化，无法进行语义匹配")
            return []
        
        try:
            # 使用CLIP文本编码器生成查询向量
            tokens = self.clip_model.tokenize_text([title])
            query_vector = self.clip_model.encode_text(tokens).cpu().numpy().tolist()[0]
            
            # 执行向量检索
            milvus_results = self.image_manager.search(query_vector, top_k)
            
            return self._convert_milvus_results(milvus_results, 'semantic_match')
            
        except Exception as e:
            self.logger.error(f"语义标题匹配失败: {e}")
            return []
    
    def _hybrid_title_match(self, title: str, top_k: int) -> List[RetrievalResult]:
        """混合标题匹配"""
        # 分别执行三种搜索
        exact_results = self._exact_title_match(title, max(1, top_k // 3))
        fuzzy_results = self._fuzzy_title_match(title, max(1, top_k // 3))
        semantic_results = self._semantic_title_match(title, max(1, top_k // 3))
        
        # 合并结果并应用权重
        all_results = []
        
        # 精确匹配权重最高
        for result in exact_results:
            result.score = result.score * 1.5  # 精确匹配加权
            result.metadata['match_type'] = 'exact'
            all_results.append(result)
        
        # 模糊匹配权重中等
        for result in fuzzy_results:
            result.score = result.score * 1.2  # 模糊匹配加权
            result.metadata['match_type'] = 'fuzzy'
            all_results.append(result)
        
        # 语义匹配权重较低
        for result in semantic_results:
            result.score = result.score * 1.0  # 语义匹配不加权
            result.metadata['match_type'] = 'semantic'
            all_results.append(result)
        
        # 去重并排序
        all_results = self._deduplicate_matches(all_results)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:top_k]
    
    def _search_caption_variant(self, caption: str, top_k: int) -> List[RetrievalResult]:
        """搜索caption变体"""
        try:
            # 使用Milvus的标量搜索
            if hasattr(self.image_manager, 'scalar_search_image_by_caption'):
                results = self.image_manager.scalar_search_image_by_caption(caption, limit=top_k)
            else:
                # 回退到标量字段搜索
                results = self.image_manager.search_by_scalar_field(
                    field_name="caption", 
                    field_value=caption, 
                    limit=top_k
                )
            
            # 转换为RetrievalResult格式
            retrieval_results = []
            for result in results:
                retrieval_result = RetrievalResult(
                    doc_id=str(result.get('id', '')),
                    content=result.get('caption', ''),
                    metadata={
                        'relative_path': result.get('relative_path', ''),
                        'caption': result.get('caption', ''),
                        'folder_name': result.get('folder_name', ''),
                        'file_size': result.get('file_size', 0),
                        'match_type': 'exact_caption',
                        'matched_variant': caption
                    },
                    score=1.0,  # 精确匹配给满分
                    retrieval_type='caption_to_image',
                    resource_collection=self.image_manager.collection_name
                )
                retrieval_results.append(retrieval_result)
            
            return retrieval_results
            
        except Exception as e:
            self.logger.error(f"搜索caption变体失败 '{caption}': {e}")
            return []
    
    def _deduplicate_matches(self, matches: List[RetrievalResult]) -> List[RetrievalResult]:
        """去重匹配结果"""
        unique_matches = []
        seen_paths = set()
        
        for match in matches:
            path = match.metadata.get('relative_path', '')
            if path and path not in seen_paths:
                seen_paths.add(path)
                unique_matches.append(match)
        
        return unique_matches
    
    def _fuzzy_match_search(self, caption: str, top_k: int) -> List[RetrievalResult]:
        """模糊匹配搜索"""
        try:
            # 提取关键词
            keywords = self._extract_keywords(caption)
            
            # 构建模糊匹配表达式
            filter_expressions = []
            for keyword in keywords:
                filter_expressions.append(f'caption like "%{keyword}%"')
            
            if not filter_expressions:
                return []
            
            # 使用OR逻辑连接多个关键词
            filter_expr = " || ".join(filter_expressions)
            
            # 执行过滤搜索
            results = self.image_manager.search_with_filter(
                filter_expr=filter_expr,
                limit=top_k * 2  # 获取更多结果用于后续排序
            )
            
            # 计算模糊匹配分数并排序
            scored_results = []
            for result in results:
                score = self._calculate_fuzzy_score(caption, result.get('caption', ''))
                if score >= self.search_config.fuzzy_threshold:
                    result['fuzzy_score'] = score
                    scored_results.append(result)
            
            # 按模糊匹配分数排序
            scored_results.sort(key=lambda x: x['fuzzy_score'], reverse=True)
            
            return self._convert_milvus_results(scored_results[:top_k], 'fuzzy_match')
            
        except Exception as e:
            self.logger.error(f"模糊匹配搜索失败: {e}")
            return []
    
    def _semantic_match_search(self, caption: str, top_k: int) -> List[RetrievalResult]:
        """语义匹配搜索"""
        self._ensure_clip_initialized(SearchMode.SEMANTIC_MATCH)
        if not self.clip_model:
            self.logger.warning("CLIP模型未初始化，无法进行语义匹配")
            return []
        
        try:
            # 使用CLIP文本编码器生成查询向量
            tokens = self.clip_model.tokenize_text([caption])
            query_vector = self.clip_model.encode_text(tokens).cpu().numpy().tolist()[0]
            
            # 执行向量检索
            milvus_results = self.image_manager.search(query_vector, top_k)
            
            return self._convert_milvus_results(milvus_results, 'semantic_match')
            
        except Exception as e:
            self.logger.error(f"语义匹配搜索失败: {e}")
            return []
    
    def _hybrid_match_search(self, caption: str, top_k: int) -> List[RetrievalResult]:
        """混合匹配搜索（结合精确、模糊和语义匹配）"""
        # 分别执行三种搜索
        exact_results = self._exact_match_search(caption, max(1, top_k // 3))
        fuzzy_results = self._fuzzy_match_search(caption, max(1, top_k // 3))
        semantic_results = self._semantic_match_search(caption, max(1, top_k // 3))
        
        # 合并结果并应用权重
        all_results = []
        
        # 精确匹配权重最高
        for result in exact_results:
            result.score = result.score * 1.5  # 精确匹配加权
            result.metadata['match_type'] = 'exact'
            all_results.append(result)
        
        # 模糊匹配权重中等
        for result in fuzzy_results:
            result.score = result.score * 1.2  # 模糊匹配加权
            result.metadata['match_type'] = 'fuzzy'
            all_results.append(result)
        
        # 语义匹配权重较低
        for result in semantic_results:
            result.score = result.score * 1.0  # 语义匹配不加权
            result.metadata['match_type'] = 'semantic'
            all_results.append(result)
        
        # 去重并排序
        all_results = self._deduplicate_results(all_results)
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:top_k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词 - 用于模糊匹配"""
        # 移除标点符号并分词
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        words = cleaned_text.split()
        
        # 过滤短词和常见停用词
        stopwords = {'的', '是', '在', '有', '和', '与', '或', '但', '了', '图', '图片', 'Figure'}
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        
        return keywords[:10]  # 限制关键词数量
    
    def _calculate_fuzzy_score(self, query_caption: str, target_caption: str) -> float:
        """计算模糊匹配分数"""
        if not query_caption or not target_caption:
            return 0.0
        
        # 简单的字符级相似度计算
        query_set = set(query_caption.lower())
        target_set = set(target_caption.lower())
        
        intersection = len(query_set & target_set)
        union = len(query_set | target_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _convert_milvus_results(self, milvus_results: List[Dict], search_type: str) -> List[RetrievalResult]:
        """转换Milvus结果为标准格式"""
        results = []
        
        for result in milvus_results:
            retrieval_result = RetrievalResult(
                doc_id=str(result.get('id', '')),
                content=result.get('caption', ''),
                metadata={
                    'relative_path': result.get('relative_path', ''),
                    'caption': result.get('caption', ''),
                    'folder_name': result.get('folder_name', ''),
                    'file_size': result.get('file_size', 0),
                    'search_type': search_type,
                    'fuzzy_score': result.get('fuzzy_score', None)
                },
                score=result.get('score', 0.0),
                retrieval_type='caption_to_image',
                resource_collection=self.image_manager.collection_name
            )
            results.append(retrieval_result)
        
        return results
    
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """结果去重"""
        unique_results = []
        seen_paths = set()
        
        for result in results:
            path = result.metadata.get('relative_path', '')
            if path and path not in seen_paths:
                seen_paths.add(path)
                unique_results.append(result)
        
        return unique_results
    
    def _empty_result(self, caption: str, error_msg: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'caption': caption,
            'results': [],
            'total_results': 0,
            'error': error_msg
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        try:
            stats = {
                'retriever_type': 'CaptionToImageRetriever',
                'search_mode': self.search_config.search_mode.value,
                'fuzzy_threshold': self.search_config.fuzzy_threshold,
                'max_results_per_caption': self.search_config.max_results_per_caption,
                'clip_model_available': self.clip_model is not None,
                'match_config': {
                    'match_mode': self.match_config.match_mode.value,
                    'enable_note_removal': self.match_config.enable_note_removal,
                    'enable_json_parsing': self.match_config.enable_json_parsing,
                    'max_candidates': self.match_config.max_candidates
                }
            }
            
            # 获取图片集合统计信息
            collection_stats = self.image_manager.get_collection_stats()
            stats.update(collection_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {'error': str(e)}


# 便捷函数
def create_caption_retriever(db_name: str = "default", 
                           search_mode: str = "exact_match",
                           match_mode: str = "variant") -> CaptionToImageRetriever:
    """
    创建Caption检索器的便捷函数
    
    Args:
        db_name: 数据库配置名称
        search_mode: 搜索模式 ("exact_match", "fuzzy_match", "semantic_match", "hybrid_match")
        match_mode: 匹配模式 ("exact", "fuzzy", "variant")
        
    Returns:
        Caption检索器实例
    """
    # 创建匹配配置
    try:
        match_config = CaptionMatchConfig(match_mode=CaptionMatchMode(match_mode))
    except ValueError:
        match_config = CaptionMatchConfig()  # 使用默认配置
    
    retriever = CaptionToImageRetriever(db_name, match_config=match_config)
    
    # 设置搜索模式
    try:
        retriever.search_config.search_mode = SearchMode(search_mode)
    except ValueError:
        retriever.logger.warning(f"无效的搜索模式: {search_mode}，使用默认模式")
    
    return retriever


if __name__ == "__main__":
    # 使用示例
    print("=== Caption检索图片模块使用示例 ===")
    
    # 创建检索器
    retriever = create_caption_retriever(search_mode="hybrid_match")
    
    # 单个caption检索
    print("\n1. 单个Caption检索:")
    result = retriever.search_single_caption("图2-3 心脏超声横切面图", top_k=3)
    print(f"找到 {result['total_results']} 个图片")
    
    # 多个caption批量检索
    print("\n2. 批量Caption检索:")
    captions = ["图1-1 超声探头位置", "图2-3 心脏横切面", "图3-5 肝脏病变"]
    batch_result = retriever.search_multiple_captions(captions, top_k=2)
    print(f"总共找到 {batch_result['total_results']} 个图片")
    
    # 从文本块提取caption并检索
    print("\n3. 从文本块检索:")
    text_chunk = """
    心脏超声检查是临床诊断的重要手段。图2-1显示了正常的心脏四腔心切面，
    可以清楚地观察到左右心房和心室的结构。图2-2展示了心脏短轴切面，
    主要用于评估左心室的收缩功能。
    """
    text_result = retriever.search_from_text_chunk(text_chunk, top_k=2)
    print(f"从文本中提取到 {len(text_result['extracted_captions'])} 个caption")
    print(f"找到 {text_result['total_results']} 个相关图片")
    
    # 获取统计信息
    print("\n4. 检索器统计信息:")
    stats = retriever.get_stats()
    print(f"搜索模式: {stats.get('search_mode')}")
    print(f"CLIP模型可用: {stats.get('clip_model_available')}")
    print(f"匹配模式: {stats.get('match_config', {}).get('match_mode')}")
    
    # 测试变体匹配
    print("\n5. 变体匹配测试:")
    test_caption = '图1-2 眼前段中央区的UBM图像 注：1.角膜：2.前房：3.虹膜：4.瞳孔：5.晶状体'
    variants = retriever.variant_matcher.generate_variants(test_caption)
    print(f"Caption变体: {variants}")
    
    # 测试JSON格式
    print("\n6. JSON格式测试:")
    json_caption = '["图1-2 眼前段中央区的UBM图像；图1-3 眼角及周边结构的UBM图像；图1-4 眼前段测量方法"]'
    json_variants = retriever.variant_matcher.generate_variants(json_caption)
    print(f"JSON Caption变体: {json_variants}")