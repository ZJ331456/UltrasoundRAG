#!/usr/bin/env python3
"""
BM25编码器 - 将文本转换为稀疏向量用于混合检索

核心功能：
1. 中文分词（jieba）
2. 生成BM25稀疏向量
3. 预存储分词结果，避免检索时重复分词
4. 维护全局词汇表和统计信息

使用场景：
- 数据插入时：生成BM25向量和分词结果
- 查询时：将查询文本编码为BM25向量
- 混合检索：与密集向量一起进行检索
"""

import os
import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import json
from ...utils.logger import setup_logger

class BM25Encoder:
    """
    BM25编码器 - 将文本转换为稀疏向量
    
    特点：
    - 使用jieba中文分词
    - 预计算词汇表和统计信息
    - 生成Milvus SPARSE_FLOAT_VECTOR格式
    - 支持持久化和加载
    """
    
    def __init__(self, vocab_size: int = 100000, use_medical_dict: bool = True):
        """
        初始化BM25编码器
        
        Args:
            vocab_size: 词汇表大小（用于稀疏向量索引）
            use_medical_dict: 是否加载医疗领域词典
        """
        self.logger = setup_logger("BM25Encoder")
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.next_id = 0
        
        # BM25参数（针对医疗文档优化）
        self.k1 = 1.2  # 词频饱和参数（医疗术语重复度高，用较小值）
        self.b = 0.8   # 文档长度归一化（医疗文档长度差异大，用较大值）
        
        # 统计信息
        self.doc_count = 0
        self.total_doc_length = 0
        self.avg_doc_length = 0
        
        # IDF缓存
        self.idf_cache: Dict[str, float] = {}
        self.doc_freq: Dict[str, int] = {}  # 词项文档频率
        
        # 医疗领域停用词
        self.stopwords = self._load_stopwords()
        
        # 加载医疗词典
        if use_medical_dict:
            self._load_medical_dict()
        
        self.logger.info(f"BM25编码器初始化完成 (词汇表大小: {vocab_size})")
    
    def _load_stopwords(self) -> set:
        """加载停用词表"""
        # 基础停用词
        basic_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '能', '而',
            '与', '等', '如', '及', '将', '对于', '由于', '或', '但', '可',
        }
        
        # 医疗高频但低信息量的词
        medical_common = {
            '患者', '临床', '表现', '情况', '进行', '可以', '应该',
            '需要', '通过', '采用', '方法', '治疗', '使用', '操作',
        }
        
        return basic_stopwords | medical_common
    
    def _load_medical_dict(self):
        """加载医疗领域词典"""
        try:
            # 医疗专业术语
            medical_terms = [
                # 超声相关
                '超声检查', '超声诊断', '超声影像', '超声探头', '超声图像',
                '甲状腺', '甲状腺结节', '甲状腺肿瘤', '甲状腺囊肿',
                '心脏超声', '腹部超声', '妇科超声', '产科超声',
                
                # 解剖结构
                '横切面', '纵切面', '斜切面', '矢状面', '冠状面',
                '左心室', '右心室', '左心房', '右心房', '二尖瓣', '三尖瓣',
                
                # 病理特征
                '低回声', '高回声', '无回声', '等回声', '混合回声',
                '边界清晰', '边界模糊', '形态规则', '形态不规则',
                '血流信号', '钙化', '囊性变', '实性',
                
                # 诊断术语
                '良性', '恶性', '可疑', '提示', '符合', '考虑',
                '鉴别诊断', '辅助检查', '随访', '复查',
            ]
            
            # 添加到jieba词典
            for term in medical_terms:
                jieba.add_word(term, freq=10000)  # 高频词
            
            self.logger.info(f"已加载 {len(medical_terms)} 个医疗术语")
            
        except Exception as e:
            self.logger.warning(f"加载医疗词典失败: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表（已过滤停用词）
        """
        if not text or not text.strip():
            return []
        
        # 使用jieba分词
        tokens = jieba.lcut(text.lower().strip())
        
        # 过滤停用词和短词
        tokens = [
            t for t in tokens 
            if len(t) > 1 and t not in self.stopwords and not t.isspace()
        ]
        
        return tokens
    
    def get_or_create_token_id(self, token: str) -> int:
        """
        获取或创建词项ID
        
        Args:
            token: 词项
            
        Returns:
            词项ID
        """
        if token not in self.token_to_id:
            if self.next_id >= self.vocab_size:
                # 词汇表满了，使用hash映射
                return hash(token) % self.vocab_size
            
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        
        return self.token_to_id[token]
    
    def encode(self, text: str, use_idf: bool = False) -> Tuple[Dict[int, float], List[str], int]:
        """
        将文本编码为BM25稀疏向量
        
        Args:
            text: 输入文本
            use_idf: 是否使用IDF权重（需要先update_statistics）
            
        Returns:
            (sparse_vector, tokens, doc_length)
            - sparse_vector: {token_id: score} 稀疏向量（Milvus格式）
            - tokens: 分词结果列表
            - doc_length: 文档长度（词数）
        """
        # 1. 分词
        tokens = self.tokenize(text)
        doc_length = len(tokens)
        
        if doc_length == 0:
            return {}, [], 0
        
        # 2. 计算词频
        token_freq = Counter(tokens)
        
        # 3. 生成稀疏向量
        sparse_vector = {}
        
        for token, freq in token_freq.items():
            token_id = self.get_or_create_token_id(token)
            
            # 计算BM25分数
            # TF部分：freq / (freq + k1 * (1 - b + b * doc_length / avg_doc_length))
            if self.avg_doc_length > 0:
                norm_factor = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
            else:
                norm_factor = 1.0
            
            tf_score = freq / (freq + self.k1 * norm_factor)
            
            # IDF部分（可选）
            if use_idf and self.doc_count > 0:
                idf = self.calculate_idf(token)
                score = tf_score * idf
            else:
                score = tf_score
            
            sparse_vector[token_id] = float(score)
        
        return sparse_vector, tokens, doc_length
    
    def encode_to_milvus_format(self, text: str) -> Dict:
        """
        编码为Milvus插入格式
        
        Returns:
            {
                'bm25_sparse_vector': {token_id: score, ...},
                'tokens': 'token1,token2,...',  # 逗号分隔
                'doc_length': int
            }
        """
        sparse_vector, tokens, doc_length = self.encode(text, use_idf=False)
        
        return {
            'bm25_sparse_vector': sparse_vector,
            'tokens': ','.join(tokens) if tokens else '',
            'doc_length': doc_length
        }
    
    def update_statistics(self, tokens: List[str]):
        """
        更新全局统计信息（用于IDF计算）
        
        Args:
            tokens: 文档的分词结果
        """
        self.doc_count += 1
        self.total_doc_length += len(tokens)
        self.avg_doc_length = self.total_doc_length / self.doc_count
        
        # 更新文档频率（每个词在文档中只计数一次）
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        
        # 清空IDF缓存（需要重新计算）
        self.idf_cache.clear()
    
    def calculate_idf(self, token: str) -> float:
        """
        计算词项的IDF值
        
        Args:
            token: 词项
            
        Returns:
            IDF值
        """
        if token in self.idf_cache:
            return self.idf_cache[token]
        
        if self.doc_count == 0:
            return 0.0
        
        # BM25 IDF公式：log((N - df + 0.5) / (df + 0.5) + 1)
        df = self.doc_freq.get(token, 0)
        idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        
        self.idf_cache[token] = idf
        return idf
    
    def get_statistics(self) -> Dict:
        """获取编码器统计信息"""
        return {
            'vocab_size': self.next_id,
            'max_vocab_size': self.vocab_size,
            'doc_count': self.doc_count,
            'avg_doc_length': round(self.avg_doc_length, 2),
            'total_tokens': sum(self.doc_freq.values()),
            'unique_tokens': len(self.doc_freq),
        }
    
    def save(self, filepath: str):
        """
        保存编码器状态到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            state = {
                'vocab_size': self.vocab_size,
                'token_to_id': self.token_to_id,
                'next_id': self.next_id,
                'k1': self.k1,
                'b': self.b,
                'doc_count': self.doc_count,
                'total_doc_length': self.total_doc_length,
                'avg_doc_length': self.avg_doc_length,
                'doc_freq': self.doc_freq,
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"BM25编码器状态已保存到: {filepath}")
            self.logger.info(f"统计信息: {self.get_statistics()}")
            
        except Exception as e:
            self.logger.error(f"保存BM25编码器状态失败: {e}")
            raise
    
    def load(self, filepath: str):
        """
        从文件加载编码器状态
        
        Args:
            filepath: 文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.vocab_size = state.get('vocab_size', self.vocab_size)
            self.token_to_id = state['token_to_id']
            self.next_id = state['next_id']
            self.k1 = state.get('k1', 1.2)
            self.b = state.get('b', 0.8)
            self.doc_count = state['doc_count']
            self.total_doc_length = state['total_doc_length']
            self.avg_doc_length = state['avg_doc_length']
            self.doc_freq = state['doc_freq']
            
            # 重建ID到token映射
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            
            # 清空IDF缓存
            self.idf_cache.clear()
            
            self.logger.info(f"BM25编码器状态已加载: {filepath}")
            self.logger.info(f"统计信息: {self.get_statistics()}")
            
        except Exception as e:
            self.logger.error(f"加载BM25编码器状态失败: {e}")
            raise


# ==================== 全局单例 ====================

_global_bm25_encoder: Optional[BM25Encoder] = None
_encoder_save_path = None

def get_bm25_encoder(
    force_reload: bool = False,
    save_path: Optional[str] = None
) -> BM25Encoder:
    """
    获取全局BM25编码器实例（单例模式）
    
    Args:
        force_reload: 是否强制重新加载
        save_path: 保存/加载路径（默认: data/bm25_encoder.json）
        
    Returns:
        BM25编码器实例
    """
    global _global_bm25_encoder, _encoder_save_path
    
    # 设置默认保存路径
    if save_path is None:
        save_path = '/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/bm25_encoder.json'
    
    _encoder_save_path = save_path
    
    # 如果已存在且不强制重载，直接返回
    if _global_bm25_encoder is not None and not force_reload:
        return _global_bm25_encoder
    
    # 创建新实例
    _global_bm25_encoder = BM25Encoder()
    
    # 尝试加载已保存的状态
    if os.path.exists(save_path):
        try:
            _global_bm25_encoder.load(save_path)
        except Exception as e:
            _global_bm25_encoder.logger.warning(f"无法加载BM25编码器状态，使用新实例: {e}")
    
    return _global_bm25_encoder


def save_global_bm25_encoder():
    """保存全局BM25编码器状态"""
    global _global_bm25_encoder, _encoder_save_path
    
    if _global_bm25_encoder is not None and _encoder_save_path is not None:
        try:
            _global_bm25_encoder.save(_encoder_save_path)
        except Exception as e:
            _global_bm25_encoder.logger.error(f"保存BM25编码器失败: {e}")


# ==================== 便捷函数 ====================

def encode_text_for_milvus(text: str) -> Dict:
    """
    便捷函数：将文本编码为Milvus格式
    
    Args:
        text: 输入文本
        
    Returns:
        包含bm25_sparse_vector, tokens, doc_length的字典
    """
    encoder = get_bm25_encoder()
    return encoder.encode_to_milvus_format(text)


def encode_query(query: str, use_idf: bool = True) -> Dict[int, float]:
    """
    便捷函数：将查询文本编码为BM25向量
    
    Args:
        query: 查询文本
        use_idf: 是否使用IDF权重
        
    Returns:
        稀疏向量 {token_id: score}
    """
    encoder = get_bm25_encoder()
    sparse_vector, _, _ = encoder.encode(query, use_idf=use_idf)
    return sparse_vector


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试BM25编码器
    print("="*50)
    print("BM25编码器测试")
    print("="*50)
    
    # 创建编码器
    encoder = BM25Encoder()
    
    # 测试文本
    texts = [
        "甲状腺超声检查显示低回声结节，边界清晰",
        "心脏超声检查二尖瓣脱垂，左心室扩大",
        "腹部超声检查肝脏回声增粗，脾脏增大",
    ]
    
    print("\n1. 编码文本:")
    for i, text in enumerate(texts):
        result = encoder.encode_to_milvus_format(text)
        print(f"\n文本{i+1}: {text}")
        print(f"  分词数: {result['doc_length']}")
        print(f"  稀疏向量维度: {len(result['bm25_sparse_vector'])}")
        print(f"  分词结果: {result['tokens'][:50]}...")
        
        # 更新统计
        tokens = result['tokens'].split(',')
        encoder.update_statistics(tokens)
    
    print(f"\n2. 编码器统计:")
    stats = encoder.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n3. 查询编码:")
    query = "甲状腺结节"
    query_vector = encode_query(query)
    print(f"  查询: {query}")
    print(f"  稀疏向量: {query_vector}")
    
    print("\n4. 保存编码器:")
    save_path = "/tmp/bm25_encoder_test.json"
    encoder.save(save_path)
    print(f"  已保存到: {save_path}")
    
    print("\n5. 加载编码器:")
    encoder2 = BM25Encoder()
    encoder2.load(save_path)
    print(f"  统计信息: {encoder2.get_statistics()}")
    
    print("\n✅ 测试完成!")

