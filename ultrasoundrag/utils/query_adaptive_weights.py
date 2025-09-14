"""
查询自适应动态权重调整模块
根据查询特征自动调整Qwen和CLIP向量权重

主要功能：
1. 查询特征分析（长度、专业术语、图片提示词等）
2. 动态权重计算
3. 可配置的权重策略
"""

import re
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from ultrasoundrag.utils.logger import setup_logger


@dataclass
class QueryFeatures:
    """查询特征"""
    length: int
    medical_term_ratio: float
    image_hint_ratio: float
    complexity_score: float
    is_technical: bool
    is_caption_like: bool


@dataclass  
class WeightConfig:
    """权重配置"""
    # 基础权重
    base_qwen_weight: float = 0.6
    base_clip_weight: float = 0.4
    
    # 调整因子
    length_factor: float = 0.2  # 长度对权重的影响
    medical_factor: float = 0.3  # 医学术语对权重的影响  
    image_hint_factor: float = 0.4  # 图片提示词对权重的影响
    
    # 权重范围限制
    min_qwen_weight: float = 0.3
    max_qwen_weight: float = 0.8
    min_clip_weight: float = 0.2
    max_clip_weight: float = 0.7


class QueryAdaptiveWeights:
    """查询自适应权重计算器"""
    
    def __init__(self, config: Optional[WeightConfig] = None):
        self.config = config or WeightConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        # 医学术语词典
        self.medical_terms = {
            '超声', '检查', '诊断', '病变', '肿瘤', '囊肿', '结节', '炎症',
            '血管', '器官', '组织', '细胞', '病理', '症状', '治疗', '手术',
            '心脏', '肝脏', '肾脏', '脾脏', '胰腺', '胆囊', '子宫', '卵巢',
            '胎儿', '孕妇', '产科', '妇科', '内科', '外科', '影像', '报告',
            'ultrasound', 'diagnosis', 'pathology', 'examination', 'scan'
        }
        
        # 图片提示词模式
        self.image_hint_patterns = [
            r'图\s*\d+[-\.\s]*\d*',  # 图2-3, 图1.2, 图 1 2
            r'figure\s*\d+[-\.\s]*\d*',  # Figure 2-3, figure 1.2
            r'fig\s*\d+[-\.\s]*\d*',  # Fig 2-3, fig 1.2  
            r'插图\s*\d+',  # 插图1
            r'示意图\s*\d*',  # 示意图1
            r'切面图?\s*\d*',  # 切面图1, 切面1
            r'横[切断]面\s*\d*',  # 横切面1
            r'纵[切断]面\s*\d*',  # 纵切面1
            r'矢状面\s*\d*',  # 矢状面1
            r'冠状面\s*\d*',  # 冠状面1
        ]
        
        # 复杂度指示词
        self.complexity_terms = {
            '鉴别诊断', '综合分析', '详细检查', '深入研究', '病理机制',
            '治疗方案', '手术方式', '预后评估', '随访观察', '多普勒'
        }
    
    def analyze_query(self, query: str) -> QueryFeatures:
        """
        分析查询特征
        
        Args:
            query: 查询文本
            
        Returns:
            查询特征对象
        """
        query_clean = query.strip()
        length = len(query_clean)
        
        # 计算医学术语比例
        medical_count = sum(1 for term in self.medical_terms if term.lower() in query_clean.lower())
        word_count = len(query_clean.split())
        medical_term_ratio = medical_count / max(word_count, 1)
        
        # 计算图片提示词比例
        image_hints = 0
        for pattern in self.image_hint_patterns:
            matches = len(re.findall(pattern, query_clean, re.IGNORECASE))
            image_hints += matches
        image_hint_ratio = image_hints / max(word_count, 1)
        
        # 计算复杂度分数
        complexity_count = sum(1 for term in self.complexity_terms if term in query_clean)
        complexity_score = complexity_count / max(word_count, 1)
        
        # 判断是否技术性查询
        is_technical = medical_term_ratio > 0.3 or complexity_score > 0.2
        
        # 判断是否类似图片caption
        is_caption_like = image_hint_ratio > 0.1 or any(
            keyword in query_clean.lower() 
            for keyword in ['图', 'figure', 'fig', '切面', '显示']
        )
        
        return QueryFeatures(
            length=length,
            medical_term_ratio=medical_term_ratio,
            image_hint_ratio=image_hint_ratio,
            complexity_score=complexity_score,
            is_technical=is_technical,
            is_caption_like=is_caption_like
        )
    
    def calculate_weights(self, query: str) -> Tuple[float, float]:
        """
        计算自适应权重
        
        Args:
            query: 查询文本
            
        Returns:
            (qwen_weight, clip_weight) 权重元组
        """
        features = self.analyze_query(query)
        
        # 从基础权重开始
        qwen_weight = self.config.base_qwen_weight
        clip_weight = self.config.base_clip_weight
        
        # 根据查询长度调整
        if features.length > 50:  # 长查询倾向于Qwen
            qwen_adjustment = min(0.2, (features.length - 50) / 200) * self.config.length_factor
            qwen_weight += qwen_adjustment
            clip_weight -= qwen_adjustment
        elif features.length < 20:  # 短查询倾向于CLIP
            clip_adjustment = min(0.2, (20 - features.length) / 20) * self.config.length_factor
            clip_weight += clip_adjustment
            qwen_weight -= clip_adjustment
        
        # 根据医学术语密度调整
        if features.medical_term_ratio > 0.2:  # 高医学术语密度倾向于Qwen
            medical_adjustment = features.medical_term_ratio * self.config.medical_factor
            qwen_weight += medical_adjustment
            clip_weight -= medical_adjustment
        
        # 根据图片提示词调整
        if features.image_hint_ratio > 0:  # 有图片提示词倾向于CLIP
            image_adjustment = min(features.image_hint_ratio, 0.5) * self.config.image_hint_factor
            clip_weight += image_adjustment
            qwen_weight -= image_adjustment
        
        # 特殊情况处理
        if features.is_caption_like:  # Caption类查询大幅倾向于CLIP
            clip_weight += 0.3
            qwen_weight -= 0.3
        
        # 权重范围限制和归一化
        qwen_weight = max(self.config.min_qwen_weight, 
                         min(self.config.max_qwen_weight, qwen_weight))
        clip_weight = max(self.config.min_clip_weight,
                         min(self.config.max_clip_weight, clip_weight))
        
        # 确保权重和为1
        total_weight = qwen_weight + clip_weight
        qwen_weight /= total_weight
        clip_weight /= total_weight
        
        self.logger.debug(
            f"查询权重计算: '{query[:30]}...' -> "
            f"Qwen: {qwen_weight:.3f}, CLIP: {clip_weight:.3f} "
            f"(特征: 长度={features.length}, 医学={features.medical_term_ratio:.2f}, "
            f"图片={features.image_hint_ratio:.2f})"
        )
        
        return qwen_weight, clip_weight
    
    def get_multimodal_fusion_weights(self, query: str) -> Dict[str, float]:
        """
        获取多模态融合权重（文本vs图像）
        
        Args:
            query: 查询文本
            
        Returns:
            包含text_weight和image_weight的字典
        """
        features = self.analyze_query(query)
        
        # 基础权重：文本0.6，图像0.4
        text_weight = 0.6
        image_weight = 0.4
        
        # 根据查询特征调整
        if features.is_caption_like or features.image_hint_ratio > 0.1:
            # 有图片相关提示时，增加图像权重
            image_weight = min(0.7, image_weight + 0.2)
            text_weight = 1.0 - image_weight
        elif features.is_technical and features.medical_term_ratio > 0.3:
            # 高技术性查询时，增加文本权重
            text_weight = min(0.8, text_weight + 0.2)
            image_weight = 1.0 - text_weight
        
        return {
            'text_weight': text_weight,
            'image_weight': image_weight
        }


# 单例实例，提供全局访问
_global_weight_calculator = None

def get_weight_calculator(config: Optional[WeightConfig] = None) -> QueryAdaptiveWeights:
    """获取全局权重计算器实例"""
    global _global_weight_calculator
    if _global_weight_calculator is None:
        _global_weight_calculator = QueryAdaptiveWeights(config)
    return _global_weight_calculator


def calculate_query_adaptive_weights(query: str) -> Tuple[float, float]:
    """
    便捷函数：计算查询自适应权重
    
    Args:
        query: 查询文本
        
    Returns:
        (qwen_weight, clip_weight) 权重元组
    """
    calculator = get_weight_calculator()
    return calculator.calculate_weights(query)


def get_fusion_weights(query: str) -> Dict[str, float]:
    """
    便捷函数：获取多模态融合权重
    
    Args:
        query: 查询文本
        
    Returns:
        包含text_weight和image_weight的字典
    """
    calculator = get_weight_calculator()
    return calculator.get_multimodal_fusion_weights(query)


if __name__ == "__main__":
    # 测试示例
    calculator = QueryAdaptiveWeights()
    
    test_queries = [
        "心脏超声检查方法和诊断标准",  # 长技术查询
        "图2-3",  # 短图片提示
        "心脏病变",  # 短医学查询
        "Figure 3.2 胎儿发育超声图像显示",  # 图片描述
        "详细的心脏超声鉴别诊断与病理分析方法",  # 复杂技术查询
    ]
    
    print("=== 查询自适应权重测试 ===")
    for query in test_queries:
        qwen_w, clip_w = calculator.calculate_weights(query)
        fusion_w = calculator.get_multimodal_fusion_weights(query)
        features = calculator.analyze_query(query)
        
        print(f"\n查询: '{query}'")
        print(f"  向量权重: Qwen={qwen_w:.3f}, CLIP={clip_w:.3f}")
        print(f"  融合权重: Text={fusion_w['text_weight']:.3f}, Image={fusion_w['image_weight']:.3f}")
        print(f"  特征: 长度={features.length}, 医学={features.medical_term_ratio:.2f}, "
              f"图片={features.image_hint_ratio:.2f}, 技术={features.is_technical}")
