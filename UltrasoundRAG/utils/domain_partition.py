"""
领域分区优化模块
提供细粒度的领域识别和分区路由功能

主要功能:
1. 智能领域识别
2. 分区路由策略  
3. 领域特定的检索优化
4. 跨领域检索支持
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from UltrasoundRAG.utils.logger import setup_logger


class DomainType(Enum):
    """领域类型枚举"""
    HEART = "heart"          # 心脏
    LIVER = "liver"          # 肝脏  
    KIDNEY = "kidney"        # 肾脏
    FETAL = "fetal"          # 胎儿
    THYROID = "thyroid"      # 甲状腺
    BREAST = "breast"        # 乳腺
    GYNECOLOGY = "gynecology"  # 妇科
    VASCULAR = "vascular"    # 血管
    ABDOMEN = "abdomen"      # 腹部
    GENERAL = "general"      # 通用


@dataclass
class DomainMatch:
    """领域匹配结果"""
    domain: DomainType
    confidence: float
    keywords: List[str]
    reasoning: str


@dataclass
class PartitionConfig:
    """分区配置"""
    enable_partitioning: bool = True
    confidence_threshold: float = 0.6
    fallback_to_general: bool = True
    cross_domain_search: bool = False
    max_domains_per_query: int = 2


class DomainClassifier:
    """领域分类器"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        
        # 详细的领域关键词字典
        self.domain_keywords = {
            DomainType.HEART: {
                'primary': ['心脏', '心肌', '心室', '心房', '瓣膜', '心包', '主动脉', '心电', '心律', '心率'],
                'secondary': ['左室', '右室', '左房', '右房', '二尖瓣', '三尖瓣', '主动脉瓣', '肺动脉瓣', '心功能', '射血分数'],
                'clinical': ['心衰', '心梗', '心肌病', '心律失常', '房颤', '室颤', '心绞痛', '冠心病', '心肌梗死', '心包积液'],
                'imaging': ['四腔心', '长轴', '短轴', 'M型', '彩超', '多普勒', 'LVEF', '室壁运动']
            },
            DomainType.LIVER: {
                'primary': ['肝脏', '肝', '肝叶', '肝段', '门静脉', '肝动脉', '胆管', '胆囊'],
                'secondary': ['右肝', '左肝', '肝左叶', '肝右叶', '尾状叶', '方叶', '肝门', '第一肝门'],
                'clinical': ['肝炎', '肝硬化', '肝癌', '肝囊肿', '肝血管瘤', '脂肪肝', '肝纤维化', '肝腹水'],
                'imaging': ['肝实质', '肝回声', '肝边缘', '肝被膜', '肝内血管', '肝内胆管']
            },
            DomainType.KIDNEY: {
                'primary': ['肾脏', '肾', '肾盂', '肾盏', '输尿管', '膀胱'],
                'secondary': ['左肾', '右肾', '肾皮质', '肾髓质', '肾柱', '肾锥体', '肾窦'],
                'clinical': ['肾炎', '肾病', '肾衰', '肾结石', '肾囊肿', '肾肿瘤', '肾积水', '肾萎缩'],
                'imaging': ['肾实质', '肾皮髓质', '肾窦回声', '肾血流', '阻力指数']
            },
            DomainType.FETAL: {
                'primary': ['胎儿', '胎', '孕', '妊娠', '孕妇', '产前', '产检'],
                'secondary': ['胎头', '胎心', '胎动', '羊水', '胎盘', '脐带', '胎位'],
                'clinical': ['胎儿发育', '胎儿畸形', '胎儿异常', '胎儿监护', '胎儿生长', '胎儿窘迫'],
                'imaging': ['双顶径', 'BPD', '头围', 'HC', '腹围', 'AC', '股骨长', 'FL', '羊水量', 'AFI']
            },
            DomainType.THYROID: {
                'primary': ['甲状腺', '甲状腺叶', '甲状腺峡', '颈部'],
                'secondary': ['左叶', '右叶', '甲状腺峡部', '甲状腺包膜'],
                'clinical': ['甲亢', '甲减', '甲状腺炎', '甲状腺结节', '甲状腺癌', '甲状腺肿'],
                'imaging': ['甲状腺回声', '甲状腺血流', '甲状腺大小', '结节性质']
            },
            DomainType.BREAST: {
                'primary': ['乳腺', '乳房', '乳头', '乳晕'],
                'secondary': ['左乳', '右乳', '乳腺组织', '乳腺导管', '乳腺小叶'],
                'clinical': ['乳腺增生', '乳腺炎', '乳腺癌', '乳腺肿块', '乳腺结节', '乳腺纤维瘤'],
                'imaging': ['BI-RADS', '乳腺密度', '乳腺回声', '血流信号', '弹性成像']
            },
            DomainType.GYNECOLOGY: {
                'primary': ['子宫', '卵巢', '输卵管', '盆腔', '妇科'],
                'secondary': ['子宫体', '子宫颈', '子宫内膜', '卵巢囊肿', '附件'],
                'clinical': ['子宫肌瘤', '卵巢囊肿', '盆腔炎', '子宫内膜异位', '多囊卵巢'],
                'imaging': ['子宫大小', '内膜厚度', '卵泡', '黄体', '盆腔积液']
            },
            DomainType.VASCULAR: {
                'primary': ['血管', '动脉', '静脉', '血流', '多普勒'],
                'secondary': ['颈动脉', '椎动脉', '下肢血管', '上肢血管', '腹主动脉'],
                'clinical': ['血管狭窄', '血栓', '动脉硬化', '静脉曲张', '血管瘤'],
                'imaging': ['血流速度', '阻力指数', 'PSV', 'EDV', 'RI', 'PI', '频谱多普勒']
            },
            DomainType.ABDOMEN: {
                'primary': ['腹部', '腹腔', '腹膜', '肠管', '脾脏', '胰腺'],
                'secondary': ['上腹部', '下腹部', '腹膜后', '肠系膜', '网膜'],
                'clinical': ['腹痛', '腹胀', '腹水', '肠梗阻', '脾大', '胰腺炎'],
                'imaging': ['腹部器官', '肠管蠕动', '腹腔积液', '淋巴结']
            },
            DomainType.GENERAL: {
                'primary': ['超声', '检查', '诊断', '图像', '声像', '探头'],
                'secondary': ['B超', '彩超', '实时', '二维', '切面', '声窗'],
                'clinical': ['超声诊断', '超声检查', '影像学', '医学影像'],
                'imaging': ['回声', '强回声', '低回声', '无回声', '混合回声', '声影', '后方回声增强']
            }
        }
        
        # 编译正则表达式模式
        self.compiled_patterns = {}
        for domain, categories in self.domain_keywords.items():
            patterns = []
            for category, keywords in categories.items():
                for keyword in keywords:
                    patterns.append(re.escape(keyword))
            self.compiled_patterns[domain] = re.compile('|'.join(patterns), re.IGNORECASE)
    
    def classify_domain(self, text: str) -> List[DomainMatch]:
        """
        对文本进行领域分类
        
        Args:
            text: 待分类的文本
            
        Returns:
            领域匹配结果列表，按置信度降序排列
        """
        matches = []
        
        for domain in DomainType:
            if domain in self.compiled_patterns:
                pattern = self.compiled_patterns[domain]
                found_keywords = pattern.findall(text)
                
                if found_keywords:
                    confidence = self._calculate_confidence(text, domain, found_keywords)
                    
                    if confidence > 0.1:  # 最小置信度阈值
                        match = DomainMatch(
                            domain=domain,
                            confidence=confidence,
                            keywords=list(set(found_keywords)),  # 去重
                            reasoning=f"检测到{len(found_keywords)}个{domain.value}相关关键词"
                        )
                        matches.append(match)
        
        # 按置信度降序排序
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # 如果没有匹配到任何领域，返回通用领域
        if not matches:
            matches.append(DomainMatch(
                domain=DomainType.GENERAL,
                confidence=0.5,
                keywords=[],
                reasoning="未检测到特定领域关键词，归类为通用领域"
            ))
        
        return matches
    
    def _calculate_confidence(self, text: str, domain: DomainType, keywords: List[str]) -> float:
        """计算领域置信度"""
        # 基础分数：关键词覆盖率
        unique_keywords = list(set(keywords))
        keyword_score = len(unique_keywords) / len(text.split()) if text.split() else 0
        
        # 权重分数：不同类别关键词的权重
        domain_categories = self.domain_keywords[domain]
        category_weights = {'primary': 1.0, 'secondary': 0.8, 'clinical': 0.9, 'imaging': 0.7}
        
        weighted_score = 0
        total_weight = 0
        
        for category, category_keywords in domain_categories.items():
            weight = category_weights.get(category, 0.5)
            category_matches = [kw for kw in unique_keywords if kw in category_keywords]
            
            if category_matches:
                weighted_score += len(category_matches) * weight
                total_weight += len(category_keywords) * weight
        
        category_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # 综合分数
        confidence = min(0.3 * keyword_score + 0.7 * category_score, 1.0)
        
        # 特殊领域的置信度调整
        if domain == DomainType.GENERAL:
            confidence *= 0.5  # 通用领域置信度降低
        
        return confidence


class DomainPartitionManager:
    """领域分区管理器"""
    
    def __init__(self, config: Optional[PartitionConfig] = None):
        self.config = config or PartitionConfig()
        self.classifier = DomainClassifier()
        self.logger = setup_logger(self.__class__.__name__)
        
        # 领域到分区的映射
        self.domain_to_partition = {
            DomainType.HEART: "heart_partition",
            DomainType.LIVER: "liver_partition", 
            DomainType.KIDNEY: "kidney_partition",
            DomainType.FETAL: "fetal_partition",
            DomainType.THYROID: "thyroid_partition",
            DomainType.BREAST: "breast_partition",
            DomainType.GYNECOLOGY: "gynecology_partition",
            DomainType.VASCULAR: "vascular_partition",
            DomainType.ABDOMEN: "abdomen_partition",
            DomainType.GENERAL: "general_partition"
        }
        
        # 相关联的分区（用于跨领域搜索）
        self.related_partitions = {
            "heart_partition": ["vascular_partition", "general_partition"],
            "liver_partition": ["abdomen_partition", "vascular_partition", "general_partition"],
            "kidney_partition": ["abdomen_partition", "vascular_partition", "general_partition"],
            "fetal_partition": ["gynecology_partition", "general_partition"],
            "thyroid_partition": ["general_partition"],
            "breast_partition": ["general_partition"],
            "gynecology_partition": ["fetal_partition", "abdomen_partition", "general_partition"],
            "vascular_partition": ["heart_partition", "general_partition"],
            "abdomen_partition": ["liver_partition", "kidney_partition", "general_partition"],
            "general_partition": []
        }
    
    def get_search_partitions(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """
        获取查询的搜索分区
        
        Args:
            query: 查询文本
            
        Returns:
            (分区列表, 分区权重字典)
        """
        if not self.config.enable_partitioning:
            return [], {}
        
        # 分类领域
        domain_matches = self.classifier.classify_domain(query)
        
        partitions = []
        weights = {}
        
        # 主要领域分区
        for match in domain_matches[:self.config.max_domains_per_query]:
            if match.confidence >= self.config.confidence_threshold:
                partition = self.domain_to_partition[match.domain]
                partitions.append(partition)
                weights[partition] = match.confidence
                
                # 添加相关分区
                if self.config.cross_domain_search:
                    related = self._get_related_partitions(partition)
                    for related_partition in related:
                        if related_partition not in partitions:
                            partitions.append(related_partition)
                            weights[related_partition] = match.confidence * 0.5
        
        # 如果没有找到合适的分区，使用通用分区
        if not partitions and self.config.fallback_to_general:
            general_partition = self.domain_to_partition[DomainType.GENERAL]
            partitions.append(general_partition)
            weights[general_partition] = 0.5
        
        return partitions, weights
    
    def _get_related_partitions(self, primary_partition: str) -> List[str]:
        """获取相关分区"""
        return self.related_partitions.get(primary_partition, [])
    
    def get_domain_filter_expression(self, query: str) -> Optional[str]:
        """
        生成Milvus过滤表达式
        
        Args:
            query: 查询文本
            
        Returns:
            Milvus过滤表达式字符串
        """
        domain_matches = self.classifier.classify_domain(query)
        
        if not domain_matches:
            return None
        
        # 取置信度最高的领域
        top_match = domain_matches[0]
        if top_match.confidence < self.config.confidence_threshold:
            return None
        
        # 生成过滤表达式
        domain_value = top_match.domain.value
        filter_expr = f'domain == "{domain_value}"'
        
        # 如果启用跨领域搜索，添加相关领域
        if self.config.cross_domain_search and len(domain_matches) > 1:
            related_domains = [match.domain.value for match in domain_matches[1:] 
                             if match.confidence >= self.config.confidence_threshold * 0.5]
            if related_domains:
                related_expr = ' or '.join([f'domain == "{domain}"' for domain in related_domains])
                filter_expr = f'({filter_expr}) or ({related_expr})'
        
        return filter_expr
    
    def suggest_domain_for_indexing(self, content: str) -> DomainType:
        """
        为索引内容建议领域
        
        Args:
            content: 内容文本
            
        Returns:
            建议的领域类型
        """
        domain_matches = self.classifier.classify_domain(content)
        
        if domain_matches and domain_matches[0].confidence >= 0.3:
            return domain_matches[0].domain
        else:
            return DomainType.GENERAL


# 全局实例管理
_global_partition_manager = None

def get_partition_manager(config: Optional[PartitionConfig] = None) -> DomainPartitionManager:
    """获取全局分区管理器实例"""
    global _global_partition_manager
    if _global_partition_manager is None:
        _global_partition_manager = DomainPartitionManager(config)
    return _global_partition_manager


def classify_query_domain(query: str) -> List[DomainMatch]:
    """便捷函数：分类查询领域"""
    manager = get_partition_manager()
    return manager.classifier.classify_domain(query)


def get_search_domains(query: str) -> Tuple[List[str], Dict[str, float]]:
    """便捷函数：获取搜索领域"""
    manager = get_partition_manager()
    return manager.get_search_partitions(query)


if __name__ == "__main__":
    # 测试示例
    test_queries = [
        "心脏超声四腔心切面检查",
        "肝脏占位性病变声像图特征", 
        "胎儿双顶径测量方法",
        "甲状腺结节超声特征分析",
        "乳腺BI-RADS分级标准",
        "血管多普勒检查方法",
        "腹部超声系统检查"
    ]
    
    print("=== 领域分区测试 ===")
    
    for query in test_queries:
        print(f"\n🔍 查询: '{query}'")
        
        # 领域分类
        domain_matches = classify_query_domain(query)
        print(f"  识别领域: {[f'{m.domain.value}(置信度:{m.confidence:.3f})' for m in domain_matches[:3]]}")
        
        # 搜索分区
        partitions, weights = get_search_domains(query)
        print(f"  搜索分区: {partitions}")
        print(f"  分区权重: {weights}")
        
        # 过滤表达式
        manager = get_partition_manager()
        filter_expr = manager.get_domain_filter_expression(query)
        print(f"  过滤表达式: {filter_expr}")