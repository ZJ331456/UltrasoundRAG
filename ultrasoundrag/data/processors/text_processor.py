"""
文本处理器

负责文本的预处理、清洗和标准化：
- 文本清洗和标准化
- 分块策略优化
- 文本质量评估
- 多语言支持
"""

import re
import string
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import jieba
import unicodedata

@dataclass
class TextChunk:
    """文本块数据结构"""
    content: str
    metadata: Dict
    chunk_id: str
    start_pos: int
    end_pos: int
    quality_score: float = 0.0

class TextProcessor:
    """文本处理器"""
    
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1000,
                 overlap_ratio: float = 0.1):
        """
        初始化文本处理器
        
        Args:
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小  
            overlap_ratio: 重叠比例
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
        
        # 初始化中文分词
        jieba.initialize()
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text)
        
        # 标准化空白字符
        text = ' '.join(text.split())
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """
        标准化文本
        
        Args:
            text: 输入文本
            
        Returns:
            标准化后的文本
        """
        if not text:
            return ""
        
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 转换为小写（保留中文）
        text = text.lower()
        
        # 移除多余标点
        text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)
        
        return self.clean_text(text)
    
    def segment_text(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
        
        # 使用jieba进行中文分词
        segments = jieba.lcut(text)
        
        # 过滤空字符串和单字符
        segments = [seg for seg in segments if len(seg.strip()) > 1]
        
        return segments
    
    def calculate_quality_score(self, text: str) -> float:
        """
        计算文本质量分数
        
        Args:
            text: 输入文本
            
        Returns:
            质量分数 (0-1)
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # 长度分数 (0-0.3)
        length_score = min(len(text) / self.max_chunk_size, 1.0) * 0.3
        score += length_score
        
        # 中文字符比例 (0-0.4)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        chinese_ratio = chinese_chars / len(text) if text else 0
        score += min(chinese_ratio, 1.0) * 0.4
        
        # 词汇丰富度 (0-0.3)
        segments = self.segment_text(text)
        unique_ratio = len(set(segments)) / len(segments) if segments else 0
        score += min(unique_ratio, 1.0) * 0.3
        
        return min(score, 1.0)
    
    def smart_chunk(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        智能分块
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        # 清洗文本
        clean_text = self.clean_text(text)
        
        # 按段落分割
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
        
        chunks = []
        chunk_id = 0
        
        for para in paragraphs:
            if len(para) <= self.max_chunk_size:
                # 段落可以直接作为一个块
                chunk = TextChunk(
                    content=para,
                    metadata=metadata or {},
                    chunk_id=f"chunk_{chunk_id}",
                    start_pos=0,
                    end_pos=len(para),
                    quality_score=self.calculate_quality_score(para)
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 段落需要进一步分割
                sub_chunks = self._split_long_paragraph(para, metadata, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
        
        return chunks
    
    def _split_long_paragraph(self, text: str, metadata: Dict, start_id: int) -> List[TextChunk]:
        """分割长段落"""
        chunks = []
        sentences = re.split(r'[。！？；]', text)
        
        current_chunk = ""
        chunk_id = start_id
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查添加这个句子是否会超过最大长度
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    # 保存当前块
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        metadata=metadata or {},
                        chunk_id=f"chunk_{chunk_id}",
                        start_pos=0,
                        end_pos=len(current_chunk),
                        quality_score=self.calculate_quality_score(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # 开始新块
                current_chunk = sentence
            else:
                current_chunk += sentence + "。"
        
        # 处理最后一个块
        if current_chunk:
            chunk = TextChunk(
                content=current_chunk.strip(),
                metadata=metadata or {},
                chunk_id=f"chunk_{chunk_id}",
                start_pos=0,
                end_pos=len(current_chunk),
                quality_score=self.calculate_quality_score(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def process_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        完整的文本处理流程
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            处理后的文本块列表
        """
        # 标准化文本
        normalized_text = self.normalize_text(text)
        
        # 智能分块
        chunks = self.smart_chunk(normalized_text, metadata)
        
        # 过滤低质量块
        quality_chunks = [
            chunk for chunk in chunks 
            if chunk.quality_score > 0.3 and len(chunk.content) >= self.min_chunk_size
        ]
        
        return quality_chunks
