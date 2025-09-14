"""
Caption信息增强处理模块
为图片caption生成独立向量索引，改进图文对齐

主要功能：
1. Caption文本预处理和标准化
2. Caption独立向量生成（Qwen + CLIP双路）
3. 图文对齐优化
4. Caption检索增强
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ultrasoundrag.utils.logger import setup_logger
from ultrasoundrag.utils.query_adaptive_weights import get_weight_calculator


@dataclass
class CaptionInfo:
    """Caption信息结构"""
    original_text: str
    cleaned_text: str
    figure_number: Optional[str]
    description: str
    domain: Optional[str]
    confidence: float


@dataclass
class CaptionVectors:
    """Caption向量组"""
    qwen_vector: List[float]
    clip_vector: List[float]
    combined_vector: Optional[List[float]] = None


@dataclass
class CaptionEnhanceConfig:
    """Caption增强配置"""
    enable_preprocessing: bool = True
    enable_figure_extraction: bool = True
    enable_domain_tagging: bool = True
    min_caption_length: int = 3
    max_caption_length: int = 500
    figure_number_weight: float = 0.3  # 图号匹配权重
    description_weight: float = 0.7   # 描述匹配权重


class CaptionPreprocessor:
    """Caption预处理器"""
    
    def __init__(self, config: Optional[CaptionEnhanceConfig] = None):
        self.config = config or CaptionEnhanceConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        # 图片编号模式
        self.figure_patterns = [
            r'(?:图|Figure|Fig|插图|示意图)[\s\.\-]*(\d+(?:[\.\-]\d+)*)',
            r'(\d+(?:[\.\-]\d+)*)[\s\.\-]*(?:图|Figure|Fig)',
        ]
        
        # 清理模式
        self.cleanup_patterns = [
            (r'\s+', ' '),  # 多个空格合并
            (r'[^\w\s\.\-\u4e00-\u9fff]', ''),  # 移除特殊字符，保留中文、英文、数字、点、横线
            (r'^\s+|\s+$', ''),  # 去除首尾空格
        ]
        
        # 停用词
        self.stop_words = {
            '的', '了', '在', '是', '有', '和', '与', '或', '但', '而', '为', 
            '从', '到', '把', '被', '可以', '能够', '显示', '表明', '说明'
        }
    
    def extract_figure_info(self, caption: str) -> Tuple[Optional[str], str]:
        """
        从caption中提取图号和描述
        
        Args:
            caption: 原始caption文本
            
        Returns:
            (figure_number, description) 元组
        """
        caption_clean = caption.strip()
        figure_number = None
        description = caption_clean
        
        if self.config.enable_figure_extraction:
            for pattern in self.figure_patterns:
                match = re.search(pattern, caption_clean, re.IGNORECASE)
                if match:
                    figure_number = match.group(1)
                    # 移除图号部分，保留描述
                    description = re.sub(pattern, '', caption_clean, flags=re.IGNORECASE).strip()
                    break
        
        return figure_number, description
    
    def clean_caption_text(self, text: str) -> str:
        """
        清理caption文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not self.config.enable_preprocessing:
            return text.strip()
        
        cleaned = text
        
        # 应用清理模式
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # 长度检查
        if len(cleaned) < self.config.min_caption_length:
            self.logger.warning(f"Caption过短被过滤: '{cleaned}'")
            return ""
        
        if len(cleaned) > self.config.max_caption_length:
            cleaned = cleaned[:self.config.max_caption_length] + "..."
            self.logger.debug(f"Caption被截断: 原长度{len(text)}, 截断后{len(cleaned)}")
        
        return cleaned
    
    def process_caption(self, caption: str, image_path: Optional[str] = None) -> CaptionInfo:
        """
        处理单个caption
        
        Args:
            caption: 原始caption文本
            image_path: 关联的图片路径（可选）
            
        Returns:
            处理后的Caption信息
        """
        # 提取图号和描述
        figure_number, description = self.extract_figure_info(caption)
        
        # 清理文本
        cleaned_description = self.clean_caption_text(description)
        
        # 领域标注（如果启用）
        domain = None
        confidence = 1.0
        if self.config.enable_domain_tagging:
            try:
                from ultrasoundrag.utils.domain_partition import classify_query_domain
                domain_matches = classify_query_domain(cleaned_description)
                if domain_matches:
                    domain = domain_matches[0].domain.value
                    confidence = domain_matches[0].confidence
            except ImportError:
                self.logger.warning("领域分类模块不可用，跳过领域标注")
        
        return CaptionInfo(
            original_text=caption.strip(),
            cleaned_text=cleaned_description,
            figure_number=figure_number,
            description=cleaned_description,
            domain=domain,
            confidence=confidence
        )


class CaptionVectorGenerator:
    """Caption向量生成器"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self._text_embedder = None
        self._clip_model = None
    
    @property
    def text_embedder(self):
        """延迟加载文本嵌入模型"""
        if self._text_embedder is None:
            try:
                from ultrasoundrag.model.model_manager import get_embedding_model
                self._text_embedder = get_embedding_model()
            except Exception as e:
                self.logger.error(f"加载文本嵌入模型失败: {e}")
                raise
        return self._text_embedder
    
    @property
    def clip_model(self):
        """延迟加载CLIP模型"""
        if self._clip_model is None:
            try:
                from ultrasoundrag.model.singleton_models import get_shared_fetal_clip
                self._clip_model = get_shared_fetal_clip()
            except Exception as e:
                self.logger.error(f"加载CLIP模型失败: {e}")
                raise
        return self._clip_model
    
    def generate_vectors(self, caption_info: CaptionInfo) -> CaptionVectors:
        """
        为caption生成双路向量
        
        Args:
            caption_info: Caption信息
            
        Returns:
            Caption向量组
        """
        text = caption_info.cleaned_text
        if not text:
            # 空文本返回零向量
            return CaptionVectors(
                qwen_vector=[0.0] * 1024,
                clip_vector=[0.0] * 768
            )
        
        try:
            # 生成Qwen向量 (1024维)
            qwen_vector = self.text_embedder.get_query_embedding(text)
            if not qwen_vector or len(qwen_vector) != 1024:
                qwen_vector = [0.0] * 1024
            
            # 生成CLIP文本向量 (768维)
            tokens = self.clip_model.tokenize_text([text])
            clip_feat = self.clip_model.encode_text(tokens).cpu().numpy()
            clip_vector = clip_feat[0].tolist() if len(clip_feat) > 0 else [0.0] * 768
            
            if len(clip_vector) != 768:
                clip_vector = [0.0] * 768
            
            return CaptionVectors(
                qwen_vector=qwen_vector,
                clip_vector=clip_vector
            )
            
        except Exception as e:
            self.logger.error(f"生成caption向量失败: {e}")
            return CaptionVectors(
                qwen_vector=[0.0] * 1024,
                clip_vector=[0.0] * 768
            )
    
    def generate_weighted_vector(self, vectors: CaptionVectors, query: str) -> List[float]:
        """
        根据查询生成加权组合向量
        
        Args:
            vectors: Caption向量组
            query: 查询文本（用于确定权重）
            
        Returns:
            加权组合向量
        """
        try:
            # 计算自适应权重
            weight_calc = get_weight_calculator()
            qwen_weight, clip_weight = weight_calc.calculate_weights(query)
            
            # 向量加权组合 (需要统一维度)
            # 这里我们可以选择拼接或者降维处理
            # 简单方案：直接拼接两个向量
            combined = (
                [v * qwen_weight for v in vectors.qwen_vector] +
                [v * clip_weight for v in vectors.clip_vector]
            )
            
            return combined
            
        except Exception as e:
            self.logger.error(f"生成加权向量失败: {e}")
            # 回退：简单拼接
            return vectors.qwen_vector + vectors.clip_vector


class CaptionMatchingEnhancer:
    """Caption匹配增强器"""
    
    def __init__(self, config: Optional[CaptionEnhanceConfig] = None):
        self.config = config or CaptionEnhanceConfig()
        self.logger = setup_logger(self.__class__.__name__)
    
    def enhanced_caption_matching(self, query: str, caption_info: CaptionInfo) -> float:
        """
        增强的caption匹配算法
        
        Args:
            query: 查询文本
            caption_info: Caption信息
            
        Returns:
            匹配分数 (0-1)
        """
        score = 0.0
        
        # 图号精确匹配
        figure_score = self._calculate_figure_match_score(query, caption_info.figure_number)
        
        # 描述语义匹配  
        description_score = self._calculate_description_match_score(query, caption_info.description)
        
        # 领域相关性匹配
        domain_score = self._calculate_domain_match_score(query, caption_info.domain)
        
        # 加权组合
        score = (
            figure_score * self.config.figure_number_weight +
            description_score * self.config.description_weight +
            domain_score * 0.2  # 领域匹配权重
        )
        
        return min(score, 1.0)
    
    def _calculate_figure_match_score(self, query: str, figure_number: Optional[str]) -> float:
        """计算图号匹配分数"""
        if not figure_number:
            return 0.0
        
        # 提取查询中的图号
        query_figures = []
        for pattern in [r'图\s*(\d+(?:[\.\-]\d+)*)', r'Figure\s*(\d+(?:[\.\-]\d+)*)', r'Fig\s*(\d+(?:[\.\-]\d+)*)']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            query_figures.extend(matches)
        
        if not query_figures:
            return 0.0
        
        # 检查是否有匹配的图号
        for query_fig in query_figures:
            if query_fig == figure_number:
                return 1.0  # 精确匹配
            elif query_fig in figure_number or figure_number in query_fig:
                return 0.8  # 部分匹配
        
        return 0.0
    
    def _calculate_description_match_score(self, query: str, description: str) -> float:
        """计算描述匹配分数"""
        if not description:
            return 0.0
        
        # 简单的词汇重叠计算
        query_words = set(re.findall(r'\w+', query.lower()))
        desc_words = set(re.findall(r'\w+', description.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(desc_words))
        return overlap / len(query_words)
    
    def _calculate_domain_match_score(self, query: str, caption_domain: Optional[str]) -> float:
        """计算领域匹配分数"""
        if not caption_domain:
            return 0.0
        
        try:
            from UltrasoundRAG.utils.domain_partition import classify_query_domain
            query_domains = classify_query_domain(query)
            
            if query_domains:
                query_domain = query_domains[0].domain.value
                if query_domain == caption_domain:
                    return 1.0
                # 可以添加相关领域的部分匹配逻辑
        except ImportError:
            pass
        
        return 0.0


class CaptionEnhancementManager:
    """Caption增强管理器"""
    
    def __init__(self, config: Optional[CaptionEnhanceConfig] = None):
        self.config = config or CaptionEnhanceConfig()
        self.preprocessor = CaptionPreprocessor(config)
        self.vector_generator = CaptionVectorGenerator()
        self.matcher = CaptionMatchingEnhancer(config)
        self.logger = setup_logger(self.__class__.__name__)
    
    def process_caption_for_indexing(self, caption: str, image_path: Optional[str] = None) -> Dict:
        """
        为索引处理caption
        
        Args:
            caption: 原始caption文本
            image_path: 关联的图片路径
            
        Returns:
            包含处理结果的字典
        """
        # 预处理caption
        caption_info = self.preprocessor.process_caption(caption, image_path)
        
        if not caption_info.cleaned_text:
            self.logger.warning(f"Caption预处理后为空，跳过: '{caption}'")
            return {}
        
        # 生成向量
        vectors = self.vector_generator.generate_vectors(caption_info)
        
        # 构建索引数据
        index_data = {
            'original_caption': caption_info.original_text,
            'cleaned_caption': caption_info.cleaned_text,
            'figure_number': caption_info.figure_number,
            'description': caption_info.description,
            'domain': caption_info.domain,
            'domain_confidence': caption_info.confidence,
            'caption_qwen_vector': vectors.qwen_vector,
            'caption_clip_vector': vectors.clip_vector,
            'image_path': image_path
        }
        
        return index_data
    
    def search_captions_by_query(self, query: str, caption_candidates: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        根据查询搜索最匹配的captions
        
        Args:
            query: 查询文本
            caption_candidates: Caption候选列表
            
        Returns:
            按匹配分数排序的(caption_data, score)列表
        """
        scored_captions = []
        
        for caption_data in caption_candidates:
            # 构建Caption信息
            caption_info = CaptionInfo(
                original_text=caption_data.get('original_caption', ''),
                cleaned_text=caption_data.get('cleaned_caption', ''),
                figure_number=caption_data.get('figure_number'),
                description=caption_data.get('description', ''),
                domain=caption_data.get('domain'),
                confidence=caption_data.get('domain_confidence', 1.0)
            )
            
            # 计算匹配分数
            score = self.matcher.enhanced_caption_matching(query, caption_info)
            
            if score > 0.1:  # 最低分数阈值
                scored_captions.append((caption_data, score))
        
        # 按分数排序
        scored_captions.sort(key=lambda x: x[1], reverse=True)
        
        return scored_captions


# 便捷函数
def create_caption_enhancer(config: Optional[CaptionEnhanceConfig] = None) -> CaptionEnhancementManager:
    """创建Caption增强管理器"""
    return CaptionEnhancementManager(config)


def process_caption_batch(captions: List[str], image_paths: Optional[List[str]] = None) -> List[Dict]:
    """
    批量处理captions
    
    Args:
        captions: Caption文本列表
        image_paths: 对应的图片路径列表（可选）
        
    Returns:
        处理结果列表
    """
    manager = create_caption_enhancer()
    results = []
    
    for i, caption in enumerate(captions):
        image_path = image_paths[i] if image_paths and i < len(image_paths) else None
        result = manager.process_caption_for_indexing(caption, image_path)
        if result:  # 只添加非空结果
            results.append(result)
    
    return results


if __name__ == "__main__":
    # 测试示例
    manager = create_caption_enhancer()
    
    test_captions = [
        "图2-3 心脏四腔心切面超声图像",
        "Figure 1.2 胎儿发育22周超声检查显示",
        "肝脏横切面显示多发性结节病变",
        "甲状腺左叶低回声结节，边界清晰",
        "血管多普勒显示颈动脉狭窄程度"
    ]
    
    print("=== Caption增强处理测试 ===")
    for caption in test_captions:
        result = manager.process_caption_for_indexing(caption)
        
        if result:
            print(f"\n原始: '{caption}'")
            print(f"  清理后: '{result['cleaned_caption']}'")
            print(f"  图号: {result['figure_number']}")
            print(f"  领域: {result['domain']} ({result['domain_confidence']:.3f})")
            print(f"  向量维度: Qwen={len(result['caption_qwen_vector'])}, CLIP={len(result['caption_clip_vector'])}")
        else:
            print(f"\n原始: '{caption}' -> 处理失败或被过滤")