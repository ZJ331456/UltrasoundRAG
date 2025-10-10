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
        
        self.logger.info(f"多数据库管理器初始化完成，加载了 {len(self.databases)} 个数据库集合配置")
    
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
    
    def _detect_collection_type(self, collection_name: str) -> str:
        """
        检测集合类型（文本集合还是图片集合）
        
        Args:
            collection_name: 集合名称
            
        Returns:
            "image" 如果集合包含image_vector字段，否则返回 "text"
        """
        try:
            # 创建临时的MilvusManager来检查集合字段
            temp_manager = MilvusManager(
                milvus_uri=self.milvus_uri,
                milvus_token=self.milvus_token,
                db_name="ultrasound_vector",  # 默认数据库
                collection_type="md",  # 临时设置，不重要
                collection_name=collection_name
            )
            
            # 检查是否包含image_vector字段
            if temp_manager._has_field("image_vector"):
                return "image"
            else:
                return "text"
                
        except Exception as e:
            self.logger.warning(f"检测集合类型时出错: {e}，默认为文本集合")
            return "text"
    
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
        
        # 检查集合类型兼容性
        if retriever_type in ["t2t"] and not db_config.text_collection:
            raise ValueError(f"检索器类型 {retriever_type} 需要文本集合，但数据库 {db_name} 没有配置文本集合")
        elif retriever_type in ["t2i", "i2t", "i2i", "caption"] and not db_config.image_collection:
            raise ValueError(f"检索器类型 {retriever_type} 需要图片集合，但数据库 {db_name} 没有配置图片集合")
        elif retriever_type in ["auto", "multimodal"] and not (db_config.text_collection or db_config.image_collection):
            raise ValueError(f"检索器类型 {retriever_type} 需要文本集合或图片集合，但数据库 {db_name} 都没有配置")
        
        # 使用通用方法创建检索器
        retriever = self._create_retriever_for_collection(db_config, retriever_type, **kwargs)
        
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
            
            # 添加调试日志
            self.logger.info(f"开始在数据库 '{db_name}' 中执行 {retriever_type} 检索")
            self.logger.info(f"查询参数: query='{query}', kwargs={kwargs}")
            
            # 执行检索（根据检索器类型调整参数）
            if retriever_type in ["i2t", "i2i"]:
                # I2T和I2I检索器期望image_path作为第一个参数
                self.logger.info(f"调用 {retriever_type} 检索器，image_path='{query}'")
                result = retriever.search(query, **kwargs)  # 这里query实际上是image_path
            else:
                # T2T、T2I和其他检索器期望query作为第一个参数
                self.logger.info(f"调用 {retriever_type} 检索器，query='{query}'")
                result = retriever.search(query, **kwargs)
            
            self.logger.info(f"检索完成，结果类型: {type(result)}, 包含键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            # 确保结果格式正确
            if not isinstance(result, dict):
                result = {'results': []}
            
            # 如果result是RetrievalResult对象，需要序列化
            if hasattr(result, 'results'):
                # 处理RetrievalResult对象
                results_list = []
                for item in result.results:
                    if hasattr(item, '__dict__'):
                        # 序列化对象为字典
                        item_dict = item.__dict__.copy()
                        if hasattr(item, 'metadata') and hasattr(item.metadata, '__dict__'):
                            item_dict['metadata'] = item.metadata.__dict__
                        results_list.append(item_dict)
                    else:
                        results_list.append(item)
                result = {'results': results_list}
            elif 'results' not in result:
                result = {'results': []}
            
            # 添加数据库信息到结果中
            result['database'] = db_name
            result['database_config'] = self.databases[db_name].__dict__
            
            # 更新统计信息
            response_time = time.time() - start_time
            self._update_stats(db_name, retriever_type, response_time)
            
            return result
            
        except ValueError as e:
            # 检索器类型不兼容的情况（如文本集合尝试图片检索）
            self.logger.warning(f"在数据库 '{db_name}' 中跳过不兼容的检索器类型 '{retriever_type}': {e}")
            return {
                'database': db_name,
                'skipped': True,
                'reason': str(e),
                'results': [],
                'total_results': 0
            }
        except Exception as e:
            self.logger.error(f"在数据库 '{db_name}' 中检索失败: {e}")
            return {
                'database': db_name,
                'error': str(e),
                'results': [],
                'total_results': 0
            }
    
    def search_by_collections(self, text_collections: List[str], 
                            image_collections: List[str], retriever_type: str, 
                            query: Any = None, text_query: Optional[str] = None, 
                            image_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        基于集合列表执行检索（新的核心方法）
        
        Args:
            text_collections: 文本集合列表
            image_collections: 图片集合列表  
            retriever_type: 检索器类型
            query: 查询内容（向后兼容）
            text_query: 明确的文本查询（用于图文混合）
            image_path: 明确的图片路径（用于图文混合）
            **kwargs: 额外参数
            
        Returns:
            聚合后的检索结果
        """
        start_time = time.time()
        
        # 解析输入参数（使用modular_retrievers中的方法）
        from .modular_retrievers import EnhancedMultimodalRetriever
        temp_retriever = EnhancedMultimodalRetriever(RetrievalContext(
            db_name="temp", text_collection="", image_collection=""
        ))
        parsed_input = temp_retriever._parse_input(query, text_query, image_path)
        self.logger.info(f"MultiDatabaseManager输入解析结果: {parsed_input}")
        
        # 根据检索器类型确定要搜索的集合
        target_collections = []
        
        if retriever_type == "t2t":
            # 文本到文本，只搜索文本集合
            target_collections = [(col, "text") for col in text_collections if col]
        elif retriever_type == "t2i":
            # 文本到图片，只搜索图片集合
            target_collections = [(col, "image") for col in image_collections if col]
        elif retriever_type == "i2t":
            # 图片到文本，只搜索文本集合
            target_collections = [(col, "text") for col in text_collections if col]
        elif retriever_type == "i2i":
            # 图片到图片，只搜索图片集合
            target_collections = [(col, "image") for col in image_collections if col]
        elif retriever_type in ["auto", "multimodal"]:
            # 混合模式，使用EnhancedMultimodalRetriever进行图文混合检索
            return self._handle_multimodal_search(
                text_collections, image_collections, parsed_input, **kwargs
            )
        
        if not target_collections:
            return {
                'error': f'检索器类型 {retriever_type} 没有找到对应的集合',
                'results': [],
                'total_results': 0,
                'debug_info': {
                    'retriever_type': retriever_type,
                    'text_collections': text_collections,
                    'image_collections': image_collections
                }
            }
        
        # 执行检索
        results_by_collection = {}
        all_results = []
        
        for collection_name, collection_type in target_collections:
            # 为每个集合创建临时数据库配置
            temp_db_config = self._create_temp_database_config(collection_name, collection_type)
            collection_key = f"{collection_name}_{collection_type}"
            
            try:
                # 创建检索器并执行检索
                retriever = self._create_retriever_for_collection(
                    temp_db_config, retriever_type, **kwargs
                )
                
                self.logger.info(f"在集合 '{collection_name}' (类型: {collection_type}) 中执行 {retriever_type} 检索")
                
                # 执行检索
                if retriever_type in ["i2t", "i2i"]:
                    # I2T和I2I检索器需要图片路径
                    search_input = parsed_input['image_path']
                    if not search_input:
                        raise ValueError(f"{retriever_type}检索需要图片路径")
                    result = retriever.search(search_input, **kwargs)
                else:
                    # T2T和T2I检索器需要文本查询
                    search_input = parsed_input['text_query']
                    if not search_input:
                        raise ValueError(f"{retriever_type}检索需要文本查询")
                    result = retriever.search(search_input, **kwargs)
                
                # 处理结果
                if isinstance(result, dict) and 'results' in result:
                    collection_results = result['results']
                else:
                    collection_results = []
                
                results_by_collection[collection_key] = {
                    'collection_name': collection_name,
                    'collection_type': collection_type,
                    'results': collection_results,
                    'total_results': len(collection_results)
                }
                
                # 添加到聚合结果中
                for res in collection_results:
                    if hasattr(res, 'metadata'):
                        res.metadata['source_collection'] = collection_name
                        res.metadata['collection_type'] = collection_type
                    all_results.append(res)
                    
            except Exception as e:
                self.logger.warning(f"在集合 '{collection_name}' 中检索失败: {e}")
                results_by_collection[collection_key] = {
                    'collection_name': collection_name,
                    'collection_type': collection_type,
                    'error': str(e),
                    'results': [],
                    'total_results': 0
                }
        
        # 聚合和排序结果
        all_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        # 限制结果数量
        max_results = kwargs.get('top_k', 10)
        final_results = all_results[:max_results]
        
        # 构建最终结果
        final_result = {
            'query': query,
            'retrieval_type': retriever_type,
            'text_collections': text_collections,
            'image_collections': image_collections,
            'results_by_collection': results_by_collection,
            'results': final_results,
            'total_collections_searched': len(target_collections),
            'total_results': len(final_results),
            'response_time': time.time() - start_time
        }
        
        return final_result
    
    def _create_retriever_for_collection(self, db_config: DatabaseConfig, retriever_type: str, **kwargs) -> Any:
        """
        为集合创建检索器（重构后的通用方法）
        
        Args:
            db_config: 数据库配置
            retriever_type: 检索器类型
            **kwargs: 额外参数
            
        Returns:
            检索器实例
        """
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
            return T2TRetriever(context)
        elif retriever_type == "t2i":
            return T2IRetriever(context)
        elif retriever_type == "i2t":
            return I2TRetriever(context)
        elif retriever_type == "i2i":
            return I2IRetriever(context)
        elif retriever_type == "caption":
            return CaptionToImageRetriever(
                db_name=db_config.name,
                milvus_uri=self.milvus_uri,
                milvus_token=self.milvus_token
            )
        elif retriever_type in ["auto", "multimodal"]:
            # 对于auto和multimodal模式，使用EnhancedMultimodalRetriever
            from .modular_retrievers import EnhancedMultimodalRetriever
            return EnhancedMultimodalRetriever(context)
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")

    def _create_temp_database_config(self, collection_name: str, collection_type: str) -> DatabaseConfig:
        """为单个集合创建临时数据库配置"""
        if collection_type == "text":
            return DatabaseConfig(
                name=f"temp_{collection_name}",
                db_name="ultrasound_vector",
                text_collection=collection_name,
                image_collection="",
                db_type=DatabaseType.GENERAL,
                priority=1,
                enabled=True,
                description=f"临时配置（文本集合）：{collection_name}",
                tags=[]
            )
        else:  # image
            return DatabaseConfig(
                name=f"temp_{collection_name}",
                db_name="ultrasound_vector",
                text_collection="",
                image_collection=collection_name,
                db_type=DatabaseType.GENERAL,
                priority=1,
                enabled=True,
                description=f"临时配置（图片集合）：{collection_name}",
                tags=[]
            )

    def search_multiple_databases(self, strategy: RetrievalStrategy, 
                                retriever_type: str, query: Any, 
                                **kwargs) -> Dict[str, Any]:
        """
        在多个数据库中执行检索（保持向后兼容）
        
        Args:
            strategy: 检索策略
            retriever_type: 检索器类型
            query: 查询内容
            **kwargs: 额外参数
            
        Returns:
            聚合后的检索结果
        """
        # 从数据库配置中提取集合列表
        text_collections = []
        image_collections = []
        
        for db_name in strategy.databases:
            if db_name in self.databases:
                config = self.databases[db_name]
                if config.text_collection:
                    text_collections.append(config.text_collection)
                if config.image_collection:
                    image_collections.append(config.image_collection)
            else:
                # 直接当作集合名称处理
                from ...core.indexing import list_collections
                collection_names = list_collections()
                if db_name in collection_names:
                    collection_type = self._detect_collection_type(db_name)
                    if collection_type == "image":
                        image_collections.append(db_name)
                    else:
                        text_collections.append(db_name)
        
        # 使用新的基于集合的搜索方法
        return self.search_by_collections(text_collections, image_collections, retriever_type, query, **kwargs)
    
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
    
    
    def _handle_multimodal_search(self, text_collections: List[str], image_collections: List[str], 
                                 parsed_input: Dict[str, Optional[str]], **kwargs) -> Dict[str, Any]:
        """
        处理多模态检索（auto/multimodal模式）
        
        Args:
            text_collections: 文本集合列表
            image_collections: 图片集合列表
            parsed_input: 解析后的输入参数
            **kwargs: 额外参数
            
        Returns:
            多模态检索结果
        """
        start_time = time.time()
        
        text_query = parsed_input['text_query']
        image_path = parsed_input['image_path']
        
        self.logger.info(f"MultiDatabaseManager多模态检索: text_query='{text_query}', image_path='{image_path}'")
        
        # 创建EnhancedMultimodalRetriever
        from .modular_retrievers import EnhancedMultimodalRetriever, RetrievalContext
        
        # 为每个集合创建检索器并执行检索
        all_results = []
        results_by_collection = {}
        
        # 处理文本集合
        for collection_name in text_collections:
            if not collection_name:
                continue
                
            try:
                # 创建临时数据库配置
                temp_db_config = self._create_temp_database_config(collection_name, "text")
                
                # 创建EnhancedMultimodalRetriever
                multimodal_retriever = self._create_retriever_for_collection(
                    temp_db_config, "multimodal", **kwargs
                )
                
                # 执行多模态检索
                self.logger.info(f"在文本集合 '{collection_name}' 中执行多模态检索")
                result = multimodal_retriever.search(
                    query=text_query or image_path,
                    mode="multimodal",
                    text_query=text_query,
                    image_path=image_path,
                    **kwargs
                )
                
                # 处理结果
                collection_results = result.get('results', [])
                results_by_collection[f"{collection_name}_text"] = {
                    'collection_name': collection_name,
                    'collection_type': 'text',
                    'results': collection_results,
                    'total_results': len(collection_results),
                    'retrieval_breakdown': result.get('retrieval_breakdown', {}),
                    'component_counts': result.get('component_counts', {})
                }
                
                # 添加到聚合结果中
                for res in collection_results:
                    if hasattr(res, 'metadata'):
                        res.metadata['source_collection'] = collection_name
                        res.metadata['collection_type'] = 'text'
                    all_results.append(res)
                    
            except Exception as e:
                self.logger.warning(f"在文本集合 '{collection_name}' 中多模态检索失败: {e}")
                results_by_collection[f"{collection_name}_text"] = {
                    'collection_name': collection_name,
                    'collection_type': 'text',
                    'error': str(e),
                    'results': [],
                    'total_results': 0
                }
        
        # 处理图片集合
        for collection_name in image_collections:
            if not collection_name:
                continue
                
            try:
                # 创建临时数据库配置
                temp_db_config = self._create_temp_database_config(collection_name, "image")
                
                # 创建EnhancedMultimodalRetriever
                multimodal_retriever = self._create_retriever_for_collection(
                    temp_db_config, "multimodal", **kwargs
                )
                
                # 执行多模态检索
                self.logger.info(f"在图片集合 '{collection_name}' 中执行多模态检索")
                result = multimodal_retriever.search(
                    query=text_query or image_path,
                    mode="multimodal",
                    text_query=text_query,
                    image_path=image_path,
                    **kwargs
                )
                
                # 处理结果
                collection_results = result.get('results', [])
                results_by_collection[f"{collection_name}_image"] = {
                    'collection_name': collection_name,
                    'collection_type': 'image',
                    'results': collection_results,
                    'total_results': len(collection_results),
                    'retrieval_breakdown': result.get('retrieval_breakdown', {}),
                    'component_counts': result.get('component_counts', {})
                }
                
                # 添加到聚合结果中
                for res in collection_results:
                    if hasattr(res, 'metadata'):
                        res.metadata['source_collection'] = collection_name
                        res.metadata['collection_type'] = 'image'
                    all_results.append(res)
                    
            except Exception as e:
                self.logger.warning(f"在图片集合 '{collection_name}' 中多模态检索失败: {e}")
                results_by_collection[f"{collection_name}_image"] = {
                    'collection_name': collection_name,
                    'collection_type': 'image',
                    'error': str(e),
                    'results': [],
                    'total_results': 0
                }
        
        # 聚合和排序结果
        all_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        # 限制结果数量
        max_results = kwargs.get('top_k', 10)
        final_results = all_results[:max_results]
        
        # 构建最终结果
        final_result = {
            'query': text_query or image_path,
            'text_query': text_query,
            'image_path': image_path,
            'retrieval_type': 'multimodal',
            'text_collections': text_collections,
            'image_collections': image_collections,
            'results_by_collection': results_by_collection,
            'results': final_results,
            'total_collections_searched': len(text_collections) + len(image_collections),
            'total_results': len(final_results),
            'response_time': time.time() - start_time
        }
        
        return final_result

    def _is_image_path(self, query: Any) -> bool:
        """判断输入是否为图片路径"""
        if not isinstance(query, str):
            return False
        
        # 检查是否是文件路径
        import os
        if os.path.exists(query):
            # 检查文件扩展名
            _, ext = os.path.splitext(query.lower())
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
            return ext in image_extensions
        
        # 如果文件不存在，但路径格式像图片文件
        _, ext = os.path.splitext(query.lower())
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return ext in image_extensions
    
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
