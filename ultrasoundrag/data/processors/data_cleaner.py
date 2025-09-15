"""
数据清洗器

负责数据的清洗、标准化和质量控制：
- 数据质量检测
- 重复数据去除
- 数据格式标准化
- 异常数据过滤
"""

import re
import hashlib
from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_count: int
    valid_count: int
    invalid_count: int
    duplicate_count: int
    quality_score: float
    issues: List[str]

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, 
                 min_text_length: int = 10,
                 max_text_length: int = 10000,
                 duplicate_threshold: float = 0.9):
        """
        初始化数据清洗器
        
        Args:
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
            duplicate_threshold: 重复阈值
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.duplicate_threshold = duplicate_threshold
        
        # 常见停用词
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'
        }
    
    def clean_text_data(self, texts: List[str]) -> List[str]:
        """
        清洗文本数据
        
        Args:
            texts: 原始文本列表
            
        Returns:
            清洗后的文本列表
        """
        cleaned_texts = []
        
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            
            # 基础清洗
            cleaned = self._basic_text_clean(text)
            
            # 长度检查
            if len(cleaned) < self.min_text_length or len(cleaned) > self.max_text_length:
                continue
            
            # 质量检查
            if self._is_quality_text(cleaned):
                cleaned_texts.append(cleaned)
        
        return cleaned_texts
    
    def _basic_text_clean(self, text: str) -> str:
        """基础文本清洗"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符但保留中文
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        # 移除重复的标点符号
        text = re.sub(r'([。！？；])\1+', r'\1', text)
        
        return text.strip()
    
    def _is_quality_text(self, text: str) -> bool:
        """检查文本质量"""
        if not text:
            return False
        
        # 检查是否主要是标点符号
        punct_ratio = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text)) / len(text)
        if punct_ratio > 0.5:
            return False
        
        # 检查是否包含有意义的内容
        meaningful_chars = len(re.findall(r'[\u4e00-\u9fff\w]', text))
        if meaningful_chars < len(text) * 0.3:
            return False
        
        return True
    
    def remove_duplicates(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """
        去除重复文本
        
        Args:
            texts: 文本列表
            
        Returns:
            (去重后的文本列表, 保留的索引列表)
        """
        if not texts:
            return [], []
        
        # 使用哈希值进行快速去重
        seen_hashes = set()
        unique_texts = []
        kept_indices = []
        
        for i, text in enumerate(texts):
            # 计算文本的标准化哈希值
            normalized_text = self._normalize_for_dedup(text)
            text_hash = hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
                kept_indices.append(i)
        
        return unique_texts, kept_indices
    
    def _normalize_for_dedup(self, text: str) -> str:
        """为去重标准化文本"""
        if not text:
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除所有标点符号和空白字符
        text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        
        return text
    
    def detect_similar_texts(self, texts: List[str]) -> List[List[int]]:
        """
        检测相似文本
        
        Args:
            texts: 文本列表
            
        Returns:
            相似文本组的索引列表
        """
        if not texts:
            return []
        
        # 计算文本的字符级相似度
        similarity_groups = []
        processed = set()
        
        for i, text1 in enumerate(texts):
            if i in processed:
                continue
            
            similar_group = [i]
            processed.add(i)
            
            for j, text2 in enumerate(texts[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_text_similarity(text1, text2)
                if similarity >= self.duplicate_threshold:
                    similar_group.append(j)
                    processed.add(j)
            
            if len(similar_group) > 1:
                similarity_groups.append(similar_group)
        
        return similarity_groups
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 标准化文本
        norm1 = self._normalize_for_dedup(text1)
        norm2 = self._normalize_for_dedup(text2)
        
        # 计算Jaccard相似度
        set1 = set(norm1)
        set2 = set(norm2)
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def clean_metadata(self, metadata_list: List[Dict]) -> List[Dict]:
        """
        清洗元数据
        
        Args:
            metadata_list: 元数据列表
            
        Returns:
            清洗后的元数据列表
        """
        cleaned_metadata = []
        
        for metadata in metadata_list:
            if not isinstance(metadata, dict):
                continue
            
            cleaned = {}
            for key, value in metadata.items():
                if isinstance(key, str) and key.strip():
                    # 清洗键名
                    clean_key = key.strip()
                    
                    # 清洗值
                    if isinstance(value, str):
                        clean_value = value.strip()
                        if clean_value:
                            cleaned[clean_key] = clean_value
                    elif value is not None:
                        cleaned[clean_key] = value
            
            if cleaned:
                cleaned_metadata.append(cleaned)
        
        return cleaned_metadata
    
    def generate_quality_report(self, texts: List[str]) -> DataQualityReport:
        """
        生成数据质量报告
        
        Args:
            texts: 文本列表
            
        Returns:
            数据质量报告
        """
        total_count = len(texts)
        valid_count = 0
        invalid_count = 0
        duplicate_count = 0
        issues = []
        
        # 统计有效文本
        for text in texts:
            if text and isinstance(text, str) and self._is_quality_text(text):
                valid_count += 1
            else:
                invalid_count += 1
        
        # 检测重复
        if texts:
            _, kept_indices = self.remove_duplicates(texts)
            duplicate_count = total_count - len(kept_indices)
        
        # 识别问题
        if invalid_count > 0:
            issues.append(f"发现 {invalid_count} 个无效文本")
        
        if duplicate_count > 0:
            issues.append(f"发现 {duplicate_count} 个重复文本")
        
        # 计算质量分数
        if total_count > 0:
            quality_score = (valid_count - duplicate_count) / total_count
        else:
            quality_score = 0.0
        
        return DataQualityReport(
            total_count=total_count,
            valid_count=valid_count,
            invalid_count=invalid_count,
            duplicate_count=duplicate_count,
            quality_score=quality_score,
            issues=issues
        )
    
    def batch_clean(self, data: Dict[str, List]) -> Dict[str, List]:
        """
        批量清洗数据
        
        Args:
            data: 数据字典
            
        Returns:
            清洗后的数据字典
        """
        cleaned_data = {}
        
        for key, values in data.items():
            if key == 'texts' and isinstance(values, list):
                cleaned_data[key] = self.clean_text_data(values)
            elif key == 'metadata' and isinstance(values, list):
                cleaned_data[key] = self.clean_metadata(values)
            else:
                cleaned_data[key] = values
        
        return cleaned_data
