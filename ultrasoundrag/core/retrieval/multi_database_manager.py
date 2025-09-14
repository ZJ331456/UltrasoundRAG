"""
多数据库管理器 - 支持多个数据库和集合的灵活配置

主要功能：
1. 多数据库配置管理
2. 动态数据库和集合选择
3. 跨数据库检索
4. 数据库统计和监控
5. 数据库切换和路由

使用场景：
- 不同专业领域的数据分离（如心脏超声、腹部超声等）
- 不同数据质量级别的分层存储
- 实验性数据和生产数据的分离
- 多租户系统支持
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

from ultrasoundrag.utils.logger import setup_logger
from ultrasoundrag.core.retrieval.data_structures import RetrievalResult
from ultrasoundrag.data.stores.milvus_store import MilvusManager
from ultrasoundrag.core.retrieval.modular_retrievers import (
    T2TRetriever, T2IRetriever, I2TRetriever, I2IRetriever,
    RetrievalContext
)
from ultrasoundrag.core.retrieval.caption_to_image_retriever import CaptionToImageRetriever
from ultrasoundrag.config import config


class DatabaseType(Enum):
    """数据库类型枚举"""
    GENERAL = "general"  # 通用数据库
    SPECIALIZED = "specialized"  # 专门数据库
    EXPERIMENTAL = "experimental"  # 实验数据库
    PRODUCTION = "production"  # 生产数据库


@dataclass
class DatabaseConfig:
    """数据库配置"""
    name: str
    db_name: str
    text_collection: str
    image_collection: str
    db_type: DatabaseType = DatabaseType.GENERAL
    priority: int = 1  # 优先级，数值越高优先级越高
    enabled: bool = True
    description: str = ""
    tags: List[str] = None


@dataclass
class RetrievalStrategy:
    """检索策略配置"""
    databases: List[str]  # 要搜索的数据库列表
    aggregation_method: str = "merge"  # "merge", "weighted", "best_only"
    max_results_per_db: int = 10
    total_max_results: int = 20
    enable_cross_db_dedup: bool = True


class MultiDatabaseManager:
    """多数据库管理器"""
    
    def __init__(self, milvus_uri: Optional[str] = None, milvus_token: Optional[str] = None):
        """
        初始化多数据库管理器
        
        Args:
            milvus_uri: Milvus服务地址
            milvus_token: Milvus认证令牌
        """
        self.logger = setup_logger(self.__class__.__name__)
        
        # 获取Milvus连接信息
        milvus_config = config['milvus']
        self.milvus_uri = milvus_uri or milvus_config['milvus_uri']
        self.milvus_token = milvus_token or milvus_config['milvus_token']
        
        # 加载数据库配置
        self.databases: Dict[str, DatabaseConfig] = self._load_database_configs()
        
        # 缓存检索器实例
        self._retriever_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'queries_by_database': defaultdict(int),
            'queries_by_type': defaultdict(int),
            'avg_response_time': 0.0
        }
        
        self.logger.info(f"多数据库管理器初始化完成，加载了 {len(self.databases)} 个数据库配置")
    
    def _load_database_configs(self) -> Dict[str, DatabaseConfig]:
        """从配置文件加载数据库配置"""
        databases = {}
        
        retriever_config = config['retriever']
        databases_config = retriever_config['databases']
        
        for db_name, db_config in databases_config.items():
            # 解析数据库类型和标签
            db_type = DatabaseType(db_config['type'])
            tags = db_config['tags']
            
            database_config = DatabaseConfig(
                name=db_name,
                db_name=db_config['db_name'],
                text_collection=db_config['collections']['text'],
                image_collection=db_config['collections']['image'],
                db_type=db_type,
                priority=db_config['priority'],
                enabled=db_config['enabled'],
                description=db_config['description'],
                tags=tags
            )
            
            databases[db_name] = database_config
        
        return databases
    
    def get_available_databases(self, db_type: Optional[DatabaseType] = None,
                              tags: Optional[List[str]] = None,
                              enabled_only: bool = True) -> List[DatabaseConfig]:
        """
        获取可用的数据库列表
        
        Args:
            db_type: 数据库类型过滤
            tags: 标签过滤
            enabled_only: 是否只返回启用的数据库
            
        Returns:
            符合条件的数据库配置列表
        """
        available_dbs = []
        
        for db_config in self.databases.values():
            # 启用状态过滤
            if enabled_only and not db_config.enabled:
                continue
            
            # 数据库类型过滤
            if db_type and db_config.db_type != db_type:
                continue
            
            # 标签过滤
            if tags:
                if not db_config.tags or not any(tag in db_config.tags for tag in tags):
                    continue
            
            available_dbs.append(db_config)
        
        # 按优先级排序
        available_dbs.sort(key=lambda x: x.priority, reverse=True)
        
        return available_dbs
    
    def create_retriever(self, db_name: str, retriever_type: str, **kwargs) -> Any:
        """
        创建指定数据库的检索器
        
        Args:
            db_name: 数据库名称
            retriever_type: 检索器类型 ("t2t", "t2i", "i2t", "i2i", "caption")
            **kwargs: 额外参数
            
        Returns:
            检索器实例
        """
        if db_name not in self.databases:
            raise ValueError(f"数据库 '{db_name}' 不存在")
        
        db_config = self.databases[db_name]
        
        # 检查缓存
        cache_key = f"{db_name}_{retriever_type}"
        if cache_key in self._retriever_cache:
            return self._retriever_cache[cache_key]
        
        # 创建检索上下文
        context = RetrievalContext(
            db_name=db_config.db_name,
            text_collection=db_config.text_collection,
            image_collection=db_config.image_collection,
            milvus_uri=self.milvus_uri,
            milvus_token=self.milvus_token,
            **kwargs
        )
        
        # 创建对应的检索器
        if retriever_type == "t2t":
            retriever = T2TRetriever(context)
        elif retriever_type == "t2i":
            retriever = T2IRetriever(context)
        elif retriever_type == "i2t":
            retriever = I2TRetriever(context)
        elif retriever_type == "i2i":
            retriever = I2IRetriever(context)
        elif retriever_type == "caption":
            retriever = CaptionToImageRetriever(
                db_name=db_name,
                milvus_uri=self.milvus_uri,
                milvus_token=self.milvus_token
            )
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
        
        # 缓存检索器
        self._retriever_cache[cache_key] = retriever
        
        return retriever
    
    def search_single_database(self, db_name: str, retriever_type: str, 
                             query: Any, **kwargs) -> Dict[str, Any]:
        """
        在单个数据库中执行检索
        
        Args:
            db_name: 数据库名称
            retriever_type: 检索器类型
            query: 查询内容
            **kwargs: 额外参数
            
        Returns:
            检索结果
        """
        start_time = time.time()
        
        try:
            # 创建检索器
            retriever = self.create_retriever(db_name, retriever_type, **kwargs)
            
            # 执行检索
            result = retriever.search(query, **kwargs)
            
            # 添加数据库信息到结果中
            result['database'] = db_name
            result['database_config'] = self.databases[db_name].__dict__
            
            # 更新统计信息
            response_time = time.time() - start_time
            self._update_stats(db_name, retriever_type, response_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"在数据库 '{db_name}' 中检索失败: {e}")
            return {
                'database': db_name,
                'error': str(e),
                'results': [],
                'total_results': 0
            }
    
    def search_multiple_databases(self, strategy: RetrievalStrategy, 
                                retriever_type: str, query: Any, 
                                **kwargs) -> Dict[str, Any]:
        """
        在多个数据库中执行检索
        
        Args:
            strategy: 检索策略
            retriever_type: 检索器类型
            query: 查询内容
            **kwargs: 额外参数
            
        Returns:
            聚合后的检索结果
        """
        start_time = time.time()
        
        # 验证数据库列表
        valid_databases = [db for db in strategy.databases if db in self.databases]
        if not valid_databases:
            return {
                'error': '没有有效的数据库',
                'results': [],
                'total_results': 0
            }
        
        # 并行执行检索（这里简化为串行）
        results_by_db = {}
        all_results = []
        
        for db_name in valid_databases:
            db_result = self.search_single_database(
                db_name, retriever_type, query, 
                top_k=strategy.max_results_per_db, 
                **kwargs
            )
            
            results_by_db[db_name] = db_result
            
            # 为每个结果添加数据库来源信息
            for result in db_result.get('results', []):
                result.metadata['source_database'] = db_name
                result.metadata['database_priority'] = self.databases[db_name].priority
                all_results.append(result)
        
        # 聚合结果
        aggregated_results = self._aggregate_results(all_results, strategy)
        
        # 构建最终结果
        final_result = {
            'query': query,
            'retrieval_type': retriever_type,
            'strategy': strategy.__dict__,
            'results_by_database': results_by_db,
            'aggregated_results': aggregated_results,
            'total_databases_searched': len(valid_databases),
            'total_results': len(aggregated_results),
            'response_time': time.time() - start_time
        }
        
        return final_result
    
    def search_by_tags(self, tags: List[str], retriever_type: str, 
                      query: Any, **kwargs) -> Dict[str, Any]:
        """
        根据标签搜索相关数据库
        
        Args:
            tags: 标签列表
            retriever_type: 检索器类型
            query: 查询内容
            **kwargs: 额外参数
            
        Returns:
            检索结果
        """
        # 获取匹配标签的数据库
        matching_dbs = self.get_available_databases(tags=tags)
        
        if not matching_dbs:
            return {
                'error': f'没有找到标签为 {tags} 的数据库',
                'results': [],
                'total_results': 0
            }
        
        # 构建检索策略
        strategy = RetrievalStrategy(
            databases=[db.name for db in matching_dbs],
            aggregation_method="weighted",
            max_results_per_db=kwargs.get('top_k', 10),
            total_max_results=kwargs.get('top_k', 10) * 2
        )
        
        return self.search_multiple_databases(strategy, retriever_type, query, **kwargs)
    
    def auto_select_databases(self, query: str, retriever_type: str, 
                            num_databases: int = 2) -> List[str]:
        """
        基于查询内容自动选择最合适的数据库
        
        Args:
            query: 查询文本
            retriever_type: 检索器类型
            num_databases: 选择的数据库数量
            
        Returns:
            选择的数据库名称列表
        """
        # 简单的关键词匹配策略
        keyword_db_mapping = {
            '心脏': ['cardiac', 'default'],
            '肝脏': ['liver', 'default'],
            '肾脏': ['kidney', 'default'],
            '妇科': ['gynecology', 'default'],
            '产科': ['obstetrics', 'default'],
            '血管': ['vascular', 'default'],
            '实验': ['experimental'],
            '研究': ['experimental', 'default']
        }
        
        # 计算每个数据库的匹配分数
        db_scores = defaultdict(float)
        
        for keyword, db_list in keyword_db_mapping.items():
            if keyword in query:
                for db_name in db_list:
                    if db_name in self.databases:
                        db_scores[db_name] += 1.0
        
        # 如果没有匹配，使用默认数据库
        if not db_scores:
            available_dbs = self.get_available_databases()
            return [db.name for db in available_dbs[:num_databases]]
        
        # 按分数排序并返回前N个
        sorted_dbs = sorted(db_scores.items(), key=lambda x: x[1], reverse=True)
        return [db_name for db_name, _ in sorted_dbs[:num_databases]]
    
    def _aggregate_results(self, all_results: List[RetrievalResult], 
                         strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """聚合多数据库的检索结果"""
        if strategy.aggregation_method == "merge":
            return self._merge_results(all_results, strategy)
        elif strategy.aggregation_method == "weighted":
            return self._weighted_aggregate(all_results, strategy)
        elif strategy.aggregation_method == "best_only":
            return self._best_only_aggregate(all_results, strategy)
        else:
            self.logger.warning(f"未知的聚合方法: {strategy.aggregation_method}")
            return self._merge_results(all_results, strategy)
    
    def _merge_results(self, all_results: List[RetrievalResult], 
                      strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """简单合并结果"""
        # 去重
        if strategy.enable_cross_db_dedup:
            all_results = self._deduplicate_cross_db(all_results)
        
        # 按分数排序
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:strategy.total_max_results]
    
    def _weighted_aggregate(self, all_results: List[RetrievalResult], 
                          strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """基于数据库优先级的加权聚合"""
        # 应用数据库优先级权重
        for result in all_results:
            db_name = result.metadata.get('source_database')
            if db_name and db_name in self.databases:
                priority = self.databases[db_name].priority
                result.score *= (1.0 + priority * 0.1)  # 优先级加权
        
        return self._merge_results(all_results, strategy)
    
    def _best_only_aggregate(self, all_results: List[RetrievalResult], 
                           strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """只选择最高分数的结果"""
        # 按数据库分组
        results_by_db = defaultdict(list)
        for result in all_results:
            db_name = result.metadata.get('source_database', 'unknown')
            results_by_db[db_name].append(result)
        
        # 从每个数据库选择最佳结果
        best_results = []
        for db_results in results_by_db.values():
            if db_results:
                best_result = max(db_results, key=lambda x: x.score)
                best_results.append(best_result)
        
        # 排序并限制数量
        best_results.sort(key=lambda x: x.score, reverse=True)
        return best_results[:strategy.total_max_results]
    
    def _deduplicate_cross_db(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """跨数据库去重"""
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # 使用内容的哈希值进行去重
            content_hash = hash(result.content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _update_stats(self, db_name: str, retriever_type: str, response_time: float):
        """更新统计信息"""
        self.stats['total_queries'] += 1
        self.stats['queries_by_database'][db_name] += 1
        self.stats['queries_by_type'][retriever_type] += 1
        
        # 更新平均响应时间
        total_time = self.stats['avg_response_time'] * (self.stats['total_queries'] - 1)
        self.stats['avg_response_time'] = (total_time + response_time) / self.stats['total_queries']
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {
            'total_databases': len(self.databases),
            'enabled_databases': len([db for db in self.databases.values() if db.enabled]),
            'databases_by_type': defaultdict(int),
            'usage_stats': dict(self.stats),
            'database_details': {}
        }
        
        # 按类型统计数据库
        for db_config in self.databases.values():
            stats['databases_by_type'][db_config.db_type.value] += 1
        
        # 数据库详细信息
        for db_name, db_config in self.databases.items():
            stats['database_details'][db_name] = {
                'enabled': db_config.enabled,
                'type': db_config.db_type.value,
                'priority': db_config.priority,
                'description': db_config.description,
                'tags': db_config.tags,
                'query_count': self.stats['queries_by_database'][db_name]
            }
        
        return stats
    
    def add_database(self, db_config: DatabaseConfig) -> bool:
        """动态添加数据库配置"""
        try:
            self.databases[db_config.name] = db_config
            self.logger.info(f"成功添加数据库配置: {db_config.name}")
            return True
        except Exception as e:
            self.logger.error(f"添加数据库配置失败: {e}")
            return False
    
    def remove_database(self, db_name: str) -> bool:
        """移除数据库配置"""
        try:
            if db_name in self.databases:
                del self.databases[db_name]
                
                # 清理缓存
                cache_keys_to_remove = [key for key in self._retriever_cache.keys() 
                                      if key.startswith(f"{db_name}_")]
                for key in cache_keys_to_remove:
                    del self._retriever_cache[key]
                
                self.logger.info(f"成功移除数据库配置: {db_name}")
                return True
            else:
                self.logger.warning(f"数据库配置不存在: {db_name}")
                return False
        except Exception as e:
            self.logger.error(f"移除数据库配置失败: {e}")
            return False


# 便捷函数
def create_multi_database_manager() -> MultiDatabaseManager:
    """创建多数据库管理器的便捷函数"""
    return MultiDatabaseManager()


def create_retrieval_strategy(databases: List[str], 
                            aggregation_method: str = "weighted",
                            max_results: int = 20) -> RetrievalStrategy:
    """创建检索策略的便捷函数"""
    return RetrievalStrategy(
        databases=databases,
        aggregation_method=aggregation_method,
        max_results_per_db=max_results // len(databases) if databases else 10,
        total_max_results=max_results
    )


if __name__ == "__main__":
    # 使用示例
    print("=== 多数据库管理器使用示例 ===")
    
    # 创建管理器
    manager = create_multi_database_manager()
    
    # 获取可用数据库
    print("\n1. 可用数据库:")
    available_dbs = manager.get_available_databases()
    for db in available_dbs:
        print(f"  - {db.name}: {db.description} (优先级: {db.priority})")
    
    # 单数据库检索
    print("\n2. 单数据库检索:")
    if available_dbs:
        result = manager.search_single_database(
            available_dbs[0].name, "t2t", "心脏超声检查", top_k=3
        )
        print(f"在数据库 '{available_dbs[0].name}' 中找到 {result.get('total_results', 0)} 个结果")
    
    # 多数据库检索
    print("\n3. 多数据库检索:")
    if len(available_dbs) >= 2:
        strategy = create_retrieval_strategy([db.name for db in available_dbs[:2]])
        multi_result = manager.search_multiple_databases(
            strategy, "t2t", "肝脏病变诊断", top_k=5
        )
        print(f"在 {multi_result.get('total_databases_searched', 0)} 个数据库中找到 {multi_result.get('total_results', 0)} 个结果")
    
    # 自动选择数据库
    print("\n4. 自动选择数据库:")
    auto_selected = manager.auto_select_databases("心脏超声心房检查", "t2i", num_databases=2)
    print(f"为查询自动选择的数据库: {auto_selected}")
    
    # 统计信息
    print("\n5. 数据库统计:")
    stats = manager.get_database_stats()
    print(f"总数据库数: {stats['total_databases']}")
    print(f"启用的数据库数: {stats['enabled_databases']}")
    print(f"总查询次数: {stats['usage_stats']['total_queries']}")
