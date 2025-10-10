import os
from typing import List, Dict, Optional, Union, Tuple, Any
from pymilvus import MilvusClient, DataType
from ultrasoundrag.config import config  # 添加导入
from ultrasoundrag.utils.logger import setup_logger

class MilvusManager:
    """
    Milvus向量数据库管理器
    
    负责Milvus数据库的连接、集合管理、数据插入和搜索功能
    支持MD文档和图片两个独立的集合
    """
    
    def __init__(self, 
                 milvus_uri: Optional[str] = None,
                 milvus_token: Optional[str] = None,
                 db_name: Optional[str] = None,
                 collection_type: str = "md",
                 collection_name: Optional[str] = None,
                 enable_domain_partition: bool = False,
                 domain_field_name: str = "domain"):
        """
        初始化Milvus管理器
        
        Args:
            milvus_uri: Milvus服务地址
            milvus_token: Milvus认证令牌
            db_name: 数据库名称
            collection_type: 集合类型 ("md" 或 "image")
        """
        milvus_cfg = config['milvus']  # 修复：milvus_sfg -> milvus_cfg
        self.milvus_uri = milvus_uri or milvus_cfg['milvus_uri']
        self.milvus_token = milvus_token or milvus_cfg['milvus_token']
        self.db_name = db_name or milvus_cfg['db_name']
        self.collection_type = collection_type
        self.enable_domain_partition = enable_domain_partition
        self.domain_field_name = domain_field_name
        
        # 根据集合类型设置集合名称
        if collection_name:
            self.collection_name = collection_name
        elif collection_type == "md":
            self.collection_name = milvus_cfg['md_collection_name']
        elif collection_type == "image":
            self.collection_name = milvus_cfg['image_collection_name']
        elif collection_type == "pdf":
            self.collection_name = milvus_cfg['pdf_collection_name']
        else:
            raise ValueError("collection_type 必须是 'md'、'image' 或 'pdf'")
        
        # 初始化日志记录器
        self.logger = setup_logger(f"MilvusManager-{collection_type}")
        
        # 初始化Milvus客户端
        self.client = MilvusClient(uri=self.milvus_uri, token=self.milvus_token)
        
        # 初始化数据库和集合
        self._setup_database()
        self._setup_collection()
        # 缓存字段名（兼容旧集合缺字段）
        self._cached_fields = None

    def _get_collection_fields(self) -> List[str]:
        """获取并缓存当前集合的字段名列表。"""
        if self._cached_fields is not None:
            return self._cached_fields
        try:
            desc = self.client.describe_collection(collection_name=self.collection_name)
            fields: List[str] = []
            if isinstance(desc, dict):
                schema = desc.get('schema', {})
                if isinstance(schema, dict):
                    fields = [f.get('name') for f in schema.get('fields', []) if isinstance(f, dict) and f.get('name')]
            self._cached_fields = fields
            return self._cached_fields
        except ConnectionError as e:
            self.logger.error(f"数据库连接失败，无法获取集合字段: {e}")
            self._cached_fields = []
            return self._cached_fields
        except (KeyError, AttributeError) as e:
            self.logger.warning(f"集合描述格式异常: {e}")
            self._cached_fields = []
            return self._cached_fields
        except Exception as e:
            self.logger.error(f"获取集合字段失败: {type(e).__name__}: {e}")
            self._cached_fields = []
            return self._cached_fields

    def _has_field(self, field_name: str) -> bool:
        """集合是否包含指定字段。"""
        return field_name in (self._get_collection_fields() or [])
    
    def _get_accurate_count(self, filter_expr: str = "") -> int:
        """获取准确的记录数量 - 突破Milvus查询窗口限制"""
        try:
            # 策略1: 尝试使用Milvus的统计信息（最快）
            try:
                stats = self.client.get_collection_stats(collection_name=self.collection_name)
                if isinstance(stats, dict):
                    cached_count = stats.get("row_count", 0)
                    if cached_count > 0:
                        # 如果缓存统计显示有数据，我们需要验证是否准确
                        # 通过小样本查询来验证
                        try:
                            sample = self.client.query(
                                collection_name=self.collection_name,
                                filter=filter_expr or None,
                                output_fields=['id'],
                                limit=100
                            )
                            if sample and len(sample) > 0:
                                # 如果样本查询成功，说明过滤条件有效
                                # 对于大集合，我们使用缓存统计 + 验证的方式
                                if cached_count > 10000:
                                    self.logger.info(f"使用缓存统计 + 验证: {cached_count} 条记录")
                                    return cached_count
                        except Exception:
                            pass
            except Exception:
                pass
            
            # 策略2: 突破查询窗口限制的分批查询
            total_count = 0
            batch_size = 8000  # 单次查询大小
            offset = 0
            max_iterations = 10  # 增加最大迭代次数
            
            for iteration in range(max_iterations):
                try:
                    # 计算当前批次的offset和limit
                    current_offset = offset
                    current_limit = min(batch_size, 16384 - current_offset)
                    
                    # 如果当前offset已经达到或接近16384限制，重置offset
                    if current_offset >= 16384:
                        # 使用ID范围查询来突破offset限制
                        # 先获取当前批次的最大ID
                        if total_count > 0:
                            # 获取最后一批数据的最大ID
                            last_batch = self.client.query(
                                collection_name=self.collection_name,
                                filter=filter_expr or None,
                                output_fields=['id'],
                                limit=1,
                                offset=current_offset - 1
                            )
                            if last_batch and len(last_batch) > 0:
                                last_id = last_batch[0].get('id') if isinstance(last_batch[0], dict) else getattr(last_batch[0], 'id', None)
                                if last_id:
                                    # 使用ID范围查询继续
                                    id_filter = f"id > {last_id}"
                                    if filter_expr:
                                        id_filter = f"({filter_expr}) && ({id_filter})"
                                    
                                    result = self.client.query(
                                        collection_name=self.collection_name,
                                        filter=id_filter,
                                        output_fields=['id'],
                                        limit=batch_size
                                    )
                                else:
                                    break
                            else:
                                break
                        else:
                            break
                    else:
                        # 正常的分批查询
                        result = self.client.query(
                            collection_name=self.collection_name,
                            filter=filter_expr or None,
                            output_fields=['id'],
                            limit=current_limit,
                            offset=current_offset
                        )
                    
                    if not result or len(result) == 0:
                        break  # 没有更多数据
                    
                    batch_count = len(result)
                    total_count += batch_count
                    
                    # 如果返回的记录数少于limit，说明已经到末尾
                    if batch_count < current_limit:
                        break
                    
                    offset += batch_count
                    
                    # 如果达到查询窗口限制，准备下一轮
                    if current_offset + current_limit >= 16384:
                        self.logger.info(f"达到查询窗口限制，已统计 {total_count} 条记录，准备下一轮查询")
                        # 重置offset，使用ID范围查询继续
                        offset = 0
                    
                except Exception as e:
                    self.logger.warning(f"分批查询失败，iteration={iteration}, offset={offset}: {e}")
                    break
            
            # 策略3: 如果分批查询成功，返回结果
            if total_count > 0:
                self.logger.info(f"通过突破限制的分批查询统计: {total_count} 条记录")
                return total_count
            
            # 策略4: 最后回退到缓存统计
            try:
                stats = self.client.get_collection_stats(collection_name=self.collection_name)
                if isinstance(stats, dict):
                    fallback_count = stats.get("row_count", 0)
                    self.logger.info(f"回退到缓存统计: {fallback_count} 条记录")
                    return fallback_count
                else:
                    import re
                    match = re.search(r'row_count["\']?\s*:\s*(\d+)', str(stats))
                    if match:
                        fallback_count = int(match.group(1))
                        self.logger.info(f"回退到缓存统计(解析): {fallback_count} 条记录")
                        return fallback_count
            except Exception:
                pass
            
            # 策略5: 最后的最后，返回-1表示未知
            self.logger.warning("所有统计方法都失败，返回-1")
            return -1
            
        except Exception as e:
            self.logger.error(f"获取准确记录数失败: {e}")
            return 0
    
    def _setup_database(self):
        """设置Milvus数据库"""
        if self.db_name not in self.client.list_databases():
            self.client.create_database(db_name=self.db_name)
            self.logger.info(f"数据库 '{self.db_name}' 创建成功")
        else:
            self.logger.debug(f"数据库 '{self.db_name}' 已存在")
        
        self.client.use_database(self.db_name)
    
    def _setup_collection(self):
        """设置Milvus集合"""
        collections = self.client.list_collections()
        
        if self.collection_name not in collections:
            # 创建schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            
            # 添加字段
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            
            if self.collection_type == "md":
                # MD集合字段（双向量：Qwen3 1024维 + FetalCLIP 768维）
                schema.add_field(field_name="text_vector_qwen_1024", datatype=DataType.FLOAT_VECTOR, dim=1024)
                schema.add_field(field_name="text_vector_clip_768", datatype=DataType.FLOAT_VECTOR, dim=768)
                # 将 content 提升到 Milvus VARCHar 上限，减少插入失败
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="file", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="image_paths", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="image_captions", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=256)
                schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
                schema.add_field(field_name=self.domain_field_name, datatype=DataType.VARCHAR, max_length=128)
                schema.add_field(field_name="is_deleted", datatype=DataType.BOOL)
                
                # 索引
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="id", index_type="AUTOINDEX")
                index_params.add_index(field_name="text_vector_qwen_1024", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="text_vector_clip_768", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="title", index_type="AUTOINDEX")
                index_params.add_index(field_name="file", index_type="AUTOINDEX")
                index_params.add_index(field_name="document_name", index_type="AUTOINDEX")
                index_params.add_index(field_name="chunk_index", index_type="AUTOINDEX")
                index_params.add_index(field_name=self.domain_field_name, index_type="AUTOINDEX")
                index_params.add_index(field_name="is_deleted", index_type="AUTOINDEX")
                
            elif self.collection_type == "pdf":
                # PDF集合字段（双向量：Qwen3 1024维 + FetalCLIP 768维）
                schema.add_field(field_name="text_vector_qwen_1024", datatype=DataType.FLOAT_VECTOR, dim=1024)
                schema.add_field(field_name="text_vector_clip_768", datatype=DataType.FLOAT_VECTOR, dim=768)
                # 将 content 提升到 Milvus VARCHar 上限，减少插入失败
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="file", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="image_paths", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="image_captions", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=256)
                schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
                schema.add_field(field_name="page_start", datatype=DataType.INT64)
                schema.add_field(field_name="page_end", datatype=DataType.INT64)
                schema.add_field(field_name=self.domain_field_name, datatype=DataType.VARCHAR, max_length=128)
                schema.add_field(field_name="is_deleted", datatype=DataType.BOOL)
                
                # 索引
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="id", index_type="AUTOINDEX")
                index_params.add_index(field_name="text_vector_qwen_1024", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="text_vector_clip_768", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="title", index_type="AUTOINDEX")
                index_params.add_index(field_name="file", index_type="AUTOINDEX")
                index_params.add_index(field_name="document_name", index_type="AUTOINDEX")
                index_params.add_index(field_name="chunk_index", index_type="AUTOINDEX")
                index_params.add_index(field_name="page_start", index_type="AUTOINDEX")
                index_params.add_index(field_name="page_end", index_type="AUTOINDEX")
                index_params.add_index(field_name=self.domain_field_name, index_type="AUTOINDEX")
                index_params.add_index(field_name="is_deleted", index_type="AUTOINDEX")
                
            elif self.collection_type == "image":
                # 图片集合字段
                schema.add_field(field_name="image_vector", datatype=DataType.FLOAT_VECTOR, dim=768)  # 修改为768
                # 图片caption文本向量（仅 CLIP 768）
                schema.add_field(field_name="caption_vector_clip_768", datatype=DataType.FLOAT_VECTOR, dim=768)
                schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="caption", datatype=DataType.VARCHAR, max_length=65535)
                schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=256)
                schema.add_field(field_name="file", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name=self.domain_field_name, datatype=DataType.VARCHAR, max_length=128)
                schema.add_field(field_name="is_deleted", datatype=DataType.BOOL)
                
                # 索引
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="id", index_type="AUTOINDEX")
                index_params.add_index(field_name="image_vector", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="caption_vector_clip_768", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="source", index_type="AUTOINDEX")
                index_params.add_index(field_name="file", index_type="AUTOINDEX")
                index_params.add_index(field_name=self.domain_field_name, index_type="AUTOINDEX")
                index_params.add_index(field_name="is_deleted", index_type="AUTOINDEX")
            
            # 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            self.logger.info(f"集合 '{self.collection_name}' ({self.collection_type}) 创建成功")
        else:
            self.logger.debug(f"集合 '{self.collection_name}' ({self.collection_type}) 已存在")
        
        # 加载集合
        self.client.load_collection(self.collection_name)
        
        # 可选：预创建默认分区（领域未知流向 default 分区）
        if self.enable_domain_partition:
            try:
                parts = {p["name"] if isinstance(p, dict) else getattr(p, "name", "") for p in self.client.list_partitions(self.collection_name)}
                if "default" not in parts:
                    self.client.create_partition(self.collection_name, "default")
            except Exception:
                pass
    
    def _sanitize_partitions(self, partition_names: Optional[List[str]]) -> Optional[List[str]]:
        """过滤不存在的分区名，避免查询时报 partition not found。
        若过滤后为空，则返回 None（不限定分区）。
        """
        if not partition_names:
            return None
        try:
            existing = {p["name"] if isinstance(p, dict) else getattr(p, "name", "") for p in self.client.list_partitions(self.collection_name)}
            valid = [p for p in partition_names if p in existing]
            return valid or None
        except Exception:
            return None
    
    def insert_data(self, data_list: List[Dict], embeddings: Optional[List[List[float]]] = None, embeddings_qwen: Optional[List[List[float]]] = None, embeddings_clip: Optional[List[List[float]]] = None, domain_values: Optional[List[Optional[str]]] = None, embeddings_caption_clip: Optional[List[List[float]]] = None) -> bool:
        """
        插入数据到Milvus（支持MD或图片数据）
        
        Args:
            data_list: 数据列表（MD或图片数据）
            embeddings: 对应的向量列表（MD为文本向量，图片为图片向量）
            
        Returns:
            bool: 插入是否成功
        """
        success_count = 0
        
        for i, data in enumerate(data_list):
            try:
                if self.collection_type == "md":
                    # 处理MD数据
                    content = data['content']
                    
                    # content 字段按 schema 支持的最大长度（65535）
                    if len(content) > 65535:
                        print(f"警告：第 {i + 1} 个块内容长度 {len(content)} 超出限制 65535，截断后写入")
                        content = content[:65535]
                    
                    import json
                    # 规范化 image_paths 为 JSON 字符串（避免二次编码）
                    raw_links = data.get('image_paths', [])
                    if isinstance(raw_links, str):
                        try:
                            parsed = json.loads(raw_links)
                            raw_links = parsed if isinstance(parsed, list) else [str(raw_links)]
                        except Exception:
                            raw_links = [str(raw_links)]
                    image_paths = json.dumps(raw_links, ensure_ascii=False)
                    if len(image_paths) > 65535:
                        image_paths = image_paths[:65535]

                    # 规范化 image_captions 为 JSON 字符串（避免二次编码，保留中文）
                    raw_caps = data.get('image_captions', [])
                    if isinstance(raw_caps, str):
                        try:
                            parsed = json.loads(raw_caps)
                            raw_caps = parsed if isinstance(parsed, list) else [str(raw_caps)]
                        except Exception:
                            raw_caps = [str(raw_caps)]
                    image_captions = json.dumps(raw_caps, ensure_ascii=False)
                    if len(image_captions) > 65535:
                        image_captions = image_captions[:65535]
                    
                    # 读取两路向量（优先使用新参数，其次兼容旧的 embeddings 参数）
                    vec_qwen = None
                    vec_clip = None
                    if embeddings_qwen is not None:
                        vec_qwen = embeddings_qwen[i]
                    elif embeddings is not None and len(embeddings[i]) in (1024, 768):
                        # 兼容旧单向量：如果是1024则填充到qwen；如果是768则填充到clip
                        if len(embeddings[i]) == 1024:
                            vec_qwen = embeddings[i]
                        else:
                            vec_clip = embeddings[i]
                    if embeddings_clip is not None:
                        vec_clip = embeddings_clip[i]
                    
                    # 维度校验
                    if vec_qwen is not None and len(vec_qwen) != 1024:
                        print(f"警告：第 {i + 1} 个块的Qwen向量维度为 {len(vec_qwen)}，期望1024")
                        vec_qwen = None
                    if vec_clip is not None and len(vec_clip) != 768:
                        print(f"警告：第 {i + 1} 个块的CLIP向量维度为 {len(vec_clip)}，期望768")
                        vec_clip = None
                    
                    insert_data = {
                        "id": int(data.get('id', i + 1)),
                        "text_vector_qwen_1024": vec_qwen if vec_qwen is not None else [0.0] * 1024,
                        "text_vector_clip_768": vec_clip if vec_clip is not None else [0.0] * 768,
                        "content": content,
                        "file": data.get('file', data.get('md_file', ''))[:512],
                        "title": data.get('title', '')[:512],
                        "image_paths": image_paths,
                        "image_captions": image_captions,
                        "document_name": data.get('document_name', '')[:256],
                        "chunk_index": int(data.get('chunk_index', 0)),
                        self.domain_field_name: (domain_values[i] if domain_values and i < len(domain_values) else data.get(self.domain_field_name, ''))[:128] if (domain_values or data.get(self.domain_field_name)) else "",
                        "is_deleted": bool(data.get('is_deleted', False))
                    }
                    
                elif self.collection_type == "pdf":
                    # 处理PDF数据
                    content = data['content']
                    
                    # content 字段按 schema 支持的最大长度（65535）
                    if len(content) > 65535:
                        print(f"警告：第 {i + 1} 个块内容长度 {len(content)} 超出限制 65535，截断后写入")
                        content = content[:65535]
                    
                    import json
                    # 规范化 image_paths 为 JSON 字符串（避免二次编码）
                    raw_links = data.get('image_paths', [])
                    if isinstance(raw_links, str):
                        try:
                            parsed = json.loads(raw_links)
                            raw_links = parsed if isinstance(parsed, list) else [str(raw_links)]
                        except Exception:
                            raw_links = [str(raw_links)]
                    image_paths = json.dumps(raw_links, ensure_ascii=False)
                    if len(image_paths) > 65535:
                        image_paths = image_paths[:65535]

                    # 规范化 image_captions 为 JSON 字符串（避免二次编码，保留中文）
                    raw_caps = data.get('image_captions', [])
                    if isinstance(raw_caps, str):
                        try:
                            parsed = json.loads(raw_caps)
                            raw_caps = parsed if isinstance(parsed, list) else [str(raw_caps)]
                        except Exception:
                            raw_caps = [str(raw_caps)]
                    image_captions = json.dumps(raw_caps, ensure_ascii=False)
                    if len(image_captions) > 65535:
                        image_captions = image_captions[:65535]
                    
                    # 读取两路向量（优先使用新参数，其次兼容旧的 embeddings 参数）
                    vec_qwen = None
                    vec_clip = None
                    if embeddings_qwen is not None:
                        vec_qwen = embeddings_qwen[i]
                    elif embeddings is not None and len(embeddings[i]) in (1024, 768):
                        # 兼容旧单向量：如果是1024则填充到qwen；如果是768则填充到clip
                        if len(embeddings[i]) == 1024:
                            vec_qwen = embeddings[i]
                        else:
                            vec_clip = embeddings[i]
                    if embeddings_clip is not None:
                        vec_clip = embeddings_clip[i]
                    
                    # 维度校验
                    if vec_qwen is not None and len(vec_qwen) != 1024:
                        print(f"警告：第 {i + 1} 个块的Qwen向量维度为 {len(vec_qwen)}，期望1024")
                        vec_qwen = None
                    if vec_clip is not None and len(vec_clip) != 768:
                        print(f"警告：第 {i + 1} 个块的CLIP向量维度为 {len(vec_clip)}，期望768")
                        vec_clip = None
                    
                    insert_data = {
                        "id": int(data.get('id', i + 1)),
                        "text_vector_qwen_1024": vec_qwen if vec_qwen is not None else [0.0] * 1024,
                        "text_vector_clip_768": vec_clip if vec_clip is not None else [0.0] * 768,
                        "content": content,
                        "file": data.get('file', data.get('pdf_file', ''))[:512],
                        "title": data.get('title', '')[:512],
                        "image_paths": image_paths,
                        "image_captions": image_captions,
                        "document_name": data.get('document_name', '')[:256],
                        "chunk_index": int(data.get('chunk_index', 0)),
                        "page_start": int(data.get('page_start', 0)),
                        "page_end": int(data.get('page_end', 0)),
                        self.domain_field_name: (domain_values[i] if domain_values and i < len(domain_values) else data.get(self.domain_field_name, ''))[:128] if (domain_values or data.get(self.domain_field_name)) else "",
                        "is_deleted": bool(data.get('is_deleted', False))
                    }
                    
                elif self.collection_type == "image":
                    # 处理图片数据
                    if embeddings and len(embeddings[i]) != 768:  # 修改为768
                        print(f"警告：第 {i + 1} 个图片的向量维度为 {len(embeddings[i])}，期望768")
                        continue
                    
                    # 校验caption向量维度（优先从数据中获取，否则从参数获取）
                    cap_clip_vec = None
                    if 'caption_vector_clip_768' in data:
                        cap_clip_vec = data['caption_vector_clip_768']
                    elif embeddings_caption_clip is not None:
                        cap_clip_vec = embeddings_caption_clip[i]
                    
                    if cap_clip_vec is not None and len(cap_clip_vec) != 768:
                        print(f"警告：第 {i + 1} 个图片caption的CLIP向量维度为 {len(cap_clip_vec)}，期望768")
                        cap_clip_vec = None

                    # 基础字段校验（不向后兼容，必须是新字段且非空）
                    img_path_val = (data.get('image_path') or '').strip()
                    caption_val = (data.get('caption') or '').strip()
                    source_val = (data.get('source') or '').strip()
                    if not img_path_val or not caption_val:
                        print(f"跳过：第 {i + 1} 个图片记录关键字段为空 image_path='{img_path_val}' caption='{caption_val}'")
                        continue

                    # 从image_path中提取file字段（目录部分）
                    file_val = ""
                    if img_path_val:
                        # 提取路径中的目录部分，如 "01_超声标准切面/图片00001.jpg" -> "01_超声标准切面"
                        import os
                        file_val = os.path.dirname(img_path_val)
                        if not file_val:  # 如果没有目录，使用文件名（去掉扩展名）
                            file_val = os.path.splitext(os.path.basename(img_path_val))[0]

                    insert_data = {
                        "id": int(data.get('id', i + 1)),
                        "image_vector": embeddings[i] if embeddings else [0.0] * 768,  # 修改为768
                        "caption_vector_clip_768": cap_clip_vec if cap_clip_vec is not None else [0.0] * 768,
                        "image_path": img_path_val[:512],
                        "caption": caption_val[:65535],
                        "source": source_val[:256],
                        "file": file_val[:512],
                        self.domain_field_name: (domain_values[i] if domain_values and i < len(domain_values) else data.get(self.domain_field_name, ''))[:128] if (domain_values or data.get(self.domain_field_name)) else "",
                        "is_deleted": bool(data.get('is_deleted', False))
                    }
                
                # 可选：基于领域写入分区
                partition_name = None
                domain_value = insert_data.get(self.domain_field_name, "")
                if self.enable_domain_partition and domain_value:
                    partition_name = domain_value.replace(' ', '_')[:64]
                    try:
                        parts = {p["name"] if isinstance(p, dict) else getattr(p, "name", "") for p in self.client.list_partitions(self.collection_name)}
                        if partition_name not in parts:
                            self.client.create_partition(self.collection_name, partition_name)
                    except Exception:
                        partition_name = None

                self.client.insert(collection_name=self.collection_name, data=insert_data, partition_name=partition_name)
                success_count += 1
                
                if (i + 1) % 100 == 0:
                    print(f"已处理 {i + 1}/{len(data_list)} 个数据")
                    
            except ConnectionError as e:
                self.logger.error(f"数据库连接失败，插入第 {i + 1} 个数据时出错: {e}")
                continue
            except (ValueError, TypeError) as e:
                self.logger.error(f"数据格式错误，插入第 {i + 1} 个数据时出错: {e}")
                continue
            except TimeoutError as e:
                self.logger.error(f"插入超时，第 {i + 1} 个数据: {e}")
                continue
            except Exception as e:
                self.logger.error(f"插入第 {i + 1} 个数据时出错: {type(e).__name__}: {e}")
                continue
        
        print(f"成功插入 {success_count}/{len(data_list)} 个数据到集合 '{self.collection_name}'")
        
        return success_count >= len(data_list) * 0.9
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter_expr: str = "", partition_names: Optional[List[str]] = None) -> List[Dict]:
        """
        向量搜索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 维度校验与容错：
        # - md 集合支持两路向量：1024(Qwen) 与 768(CLIP)
        # - image 集合固定 768(CLIP)
        if self.collection_type == "md":
            if len(query_embedding) not in (1024, 768):
                print(f"警告：查询向量维度为 {len(query_embedding)}，期望1024或768")
                return []
        else:
            if len(query_embedding) != 768:
                print(f"警告：查询向量维度为 {len(query_embedding)}，期望768")
                return []
        
        if self.collection_type == "md":
            # 根据查询向量维度自动选择字段
            if len(query_embedding) == 1024:
                field_name = "text_vector_qwen_1024"
            elif len(query_embedding) == 768:
                field_name = "text_vector_clip_768"
            else:
                print(f"警告：查询向量维度为 {len(query_embedding)}，期望1024或768")
                return []
            desired_fields = ["id", "content", "file", "title", "image_paths", "image_captions", "document_name", "chunk_index", self.domain_field_name, "is_deleted"]
        elif self.collection_type == "pdf":
            # 根据查询向量维度自动选择字段
            if len(query_embedding) == 1024:
                field_name = "text_vector_qwen_1024"
            elif len(query_embedding) == 768:
                field_name = "text_vector_clip_768"
            else:
                print(f"警告：查询向量维度为 {len(query_embedding)}，期望1024或768")
                return []
            desired_fields = ["id", "content", "file", "title", "image_paths", "image_captions", "document_name", "chunk_index", "page_start", "page_end", self.domain_field_name, "is_deleted"]
        elif self.collection_type == "image":
            field_name = "image_vector"
            desired_fields = ["id", "image_path", "caption", "source", "file", self.domain_field_name, "is_deleted"]

        # 直接使用期望的字段，因为_get_collection_fields()可能返回空集合
        # 这些字段在直接查询中都能正常工作，说明它们确实存在
        output_fields = desired_fields
        
        try:
            # 自动追加软删除过滤（兼容 None）
            composed_filter = (filter_expr or "").strip()
            if self._has_field("is_deleted"):
                delete_guard = "is_deleted == false"
                composed_filter = delete_guard if not composed_filter else f"({delete_guard}) && ({composed_filter})"

            # 设置搜索参数，包括相似度阈值
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = self.client.search(
                collection_name=self.collection_name, 
                data=[query_embedding], 
                anns_field=field_name, 
                limit=top_k, 
                output_fields=output_fields, 
                filter=composed_filter or None, 
                partition_names=self._sanitize_partitions(partition_names),
                search_params=search_params
            )
            
            formatted_results = []
            if results and len(results) > 0:
                for result in results[0]:
                    # 正确提取Hit对象中的字段数据
                    entity_dict = {}
                    for field in output_fields:
                        try:
                            # 方式1: 通过entity属性访问
                            if hasattr(result, 'entity') and hasattr(result.entity, field):
                                entity_dict[field] = getattr(result.entity, field)
                            # 方式2: 通过fields属性访问
                            elif hasattr(result, 'fields') and isinstance(result.fields, dict) and field in result.fields:
                                entity_dict[field] = result.fields[field]
                            # 方式3: 直接从result访问（某些字段如id）
                            elif hasattr(result, field):
                                entity_dict[field] = getattr(result, field)
                            # 方式4: 从result字典访问
                            elif field in result:
                                entity_dict[field] = result[field]
                            else:
                                entity_dict[field] = None
                        except Exception:
                            entity_dict[field] = None
                    
                    
                    result_dict = {
                        'score': result.score,
                        **entity_dict
                    }
                    
                    # 解析 JSON 字符串字段为可读格式
                    if self.collection_type == "md":
                        import json
                        def _robust_json_loads(val):
                            # 将可能被重复转义或非列表形式的值，尽力解析为列表[str]
                            if val is None:
                                return []
                            obj = val
                            # 如果是 bytes，先转 str
                            if isinstance(obj, bytes):
                                try:
                                    obj = obj.decode('utf-8', errors='ignore')
                                except Exception:
                                    obj = str(obj)
                            # 尝试一次解码
                            if isinstance(obj, str):
                                try:
                                    obj = json.loads(obj)
                                except Exception:
                                    # 可能是被再次转义的 JSON 字符串，如 "[\"中文\"]"
                                    try:
                                        tmp = json.loads(obj.strip('"'))
                                        obj = tmp
                                    except Exception:
                                        obj = [obj]
                            # 如果最终不是列表，则包装为列表
                            if not isinstance(obj, list):
                                obj = [str(obj)]
                            # 确保元素是字符串
                            return [str(x) for x in obj]

                        if 'image_paths' in result_dict:
                            result_dict['image_paths'] = _robust_json_loads(result_dict.get('image_paths'))
                        if 'image_captions' in result_dict:
                            result_dict['image_captions'] = _robust_json_loads(result_dict.get('image_captions'))
                    
                    formatted_results.append(result_dict)
            
            return formatted_results
            
        except ConnectionError as e:
            self.logger.error(f"数据库连接失败，搜索时出错: {e}")
            return []
        except (ValueError, TypeError) as e:
            self.logger.error(f"搜索参数错误: {e}")
            return []
        except TimeoutError as e:
            self.logger.error(f"搜索请求超时: {e}")
            return []
        except Exception as e:
            self.logger.error(f"搜索时出错: {type(e).__name__}: {e}")
            return []

    def search_multi_vectors(self, field_to_embedding: Dict[str, List[float]], top_k: int = 5, weights: Optional[Dict[str, float]] = None, filter_expr: str = "", partition_names: Optional[List[str]] = None) -> List[Dict]:
        """多向量混合检索：分别对多个向量字段检索并按权重融合。
        field_to_embedding: 例如 {"text_vector_qwen_1024": [...], "text_vector_clip_768": [...]}。
        """
        if not field_to_embedding:
            return []
        weights = weights or {}
        merged: Dict[int, Dict] = {}
        for field_name, embedding in field_to_embedding.items():
            if not isinstance(embedding, list) or not embedding:
                continue
            # 输出字段尽量齐全（仅取存在的字段）
            base_fields = ["id", self.domain_field_name]
            if self._has_field("is_deleted"):
                base_fields.append("is_deleted")
            if self.collection_type == "md":
                desired = base_fields + ["content", "file", "title", "image_paths", "image_captions", "document_name", "chunk_index"]
            elif self.collection_type == "pdf":
                desired = base_fields + ["content", "file", "title", "image_paths", "image_captions", "document_name", "chunk_index", "page_start", "page_end"]
            else:
                desired = base_fields + ["image_path", "caption", "source", "file"]
            # 直接使用期望的字段，因为_get_collection_fields()可能返回空集合
            output_fields = desired

            composed_filter = (filter_expr or "").strip()
            if self._has_field("is_deleted"):
                delete_guard = "is_deleted == false"
                composed_filter = delete_guard if not composed_filter else f"({delete_guard}) && ({composed_filter})"

            try:
                res = self.client.search(collection_name=self.collection_name, data=[embedding], anns_field=field_name, limit=top_k, output_fields=output_fields, filter=composed_filter or None, partition_names=self._sanitize_partitions(partition_names))
            except Exception:
                res = []
            if not res:
                continue
            field_weight = float(weights.get(field_name, 1.0))
            for r in res[0]:
                # 正确提取Hit对象中的字段数据
                ent = {}
                for of in output_fields:
                    try:
                        # 方式1: 通过entity属性访问
                        if hasattr(r, 'entity') and hasattr(r.entity, of):
                            ent[of] = getattr(r.entity, of)
                        # 方式2: 通过fields属性访问
                        elif hasattr(r, 'fields') and isinstance(r.fields, dict) and of in r.fields:
                            ent[of] = r.fields[of]
                        # 方式3: 直接从result访问（某些字段如id）
                        elif hasattr(r, of):
                            ent[of] = getattr(r, of)
                        # 方式4: 从result字典访问
                        elif of in r:
                            ent[of] = r[of]
                        else:
                            ent[of] = None
                    except Exception:
                        ent[of] = None
                primary_id = int(ent.get("id", 0))
                if primary_id not in merged:
                    merged[primary_id] = {**ent, "score": r.score * field_weight}
                else:
                    merged[primary_id]["score"] = max(merged[primary_id]["score"], r.score * field_weight)
                    # 合并缺失字段
                    for k, v in ent.items():
                        if k not in merged[primary_id] or merged[primary_id][k] in (None, "", []):
                            merged[primary_id][k] = v

        results = list(merged.values())
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:top_k]

    def search_with_filter(self, filter_expr: str, limit: int = 10, output_fields: Optional[List[str]] = None) -> List[Dict]:
        """仅基于标量过滤的查询（用于caption模糊匹配等）。"""
        if not output_fields:
            # 强制返回所有字段，不依赖字段检测
            output_fields = None
        # 自动追加软删除过滤（兼容 None）
        composed_filter = (filter_expr or "").strip()
        if self._has_field("is_deleted"):
            delete_guard = "is_deleted == false"
            composed_filter = delete_guard if not composed_filter else f"({delete_guard}) && ({composed_filter})"
        try:
            rows = self.client.query(collection_name=self.collection_name, filter=composed_filter, output_fields=output_fields or None, limit=limit)
            normalized: List[Dict] = []
            for r in rows or []:
                if isinstance(r, dict):
                    normalized.append(r)
                else:
                    # 兜底读取
                    try:
                        entry = {f: getattr(r, f, None) for f in (output_fields or [])}
                    except Exception:
                        entry = {}
                    normalized.append(entry)
            return normalized
        except Exception as e:
            print(f"过滤查询出错: {e}")
            return []

    def _search_with_unlimited_filter(self, filter_expr: str, output_fields: Optional[List[str]] = None) -> List[Dict]:
        """突破查询窗口限制的过滤查询"""
        try:
            # 自动追加软删除过滤
            composed_filter = (filter_expr or "").strip()
            if self._has_field("is_deleted"):
                delete_guard = "is_deleted == false"
                composed_filter = delete_guard if not composed_filter else f"({delete_guard}) && ({composed_filter})"
            
            all_results = []
            batch_size = 8000
            offset = 0
            max_iterations = 20
            last_max_id = 0
            
            for iteration in range(max_iterations):
                try:
                    # 计算当前批次的offset和limit
                    current_offset = offset
                    current_limit = min(batch_size, 16384 - current_offset)
                    
                    # 如果当前offset已经达到或接近16384限制，使用ID范围查询
                    if current_offset >= 16384:
                        if last_max_id > 0:
                            # 使用ID范围查询继续
                            id_filter = f"id > {last_max_id}"
                            if composed_filter:
                                id_filter = f"({composed_filter}) && ({id_filter})"
                            
                            rows = self.client.query(
                                collection_name=self.collection_name,
                                filter=id_filter,
                                output_fields=output_fields or None,
                                limit=batch_size
                            )
                        else:
                            break
                    else:
                        # 正常的分批查询
                        rows = self.client.query(
                            collection_name=self.collection_name,
                            filter=composed_filter,
                            output_fields=output_fields or None,
                            limit=current_limit,
                            offset=current_offset
                        )
                    
                    if not rows or len(rows) == 0:
                        break  # 没有更多数据
                    
                    # 处理结果
                    batch_results = []
                    current_max_id = 0
                    for r in rows:
                        if isinstance(r, dict):
                            batch_results.append(r)
                            record_id = r.get('id', 0)
                        else:
                            # 兜底读取
                            try:
                                entry = {f: getattr(r, f, None) for f in (output_fields or [])}
                                batch_results.append(entry)
                                record_id = getattr(r, 'id', 0)
                            except Exception:
                                continue
                        
                        if record_id > current_max_id:
                            current_max_id = record_id
                    
                    all_results.extend(batch_results)
                    last_max_id = current_max_id
                    
                    # 如果返回的记录数少于limit，说明已经到末尾
                    if len(rows) < current_limit:
                        break
                    
                    offset += len(rows)
                    
                    # 如果达到查询窗口限制，准备下一轮
                    if current_offset + current_limit >= 16384:
                        self.logger.info(f"达到查询窗口限制，已获取 {len(all_results)} 条记录，准备下一轮查询")
                        # 重置offset，使用ID范围查询继续
                        offset = 0
                        
                except Exception as e:
                    self.logger.warning(f"分批查询失败，iteration={iteration}: {e}")
                    break
            
            self.logger.info(f"突破限制查询完成，总共获取 {len(all_results)} 条记录")
            return all_results
            
        except Exception as e:
            self.logger.error(f"突破限制查询失败: {e}")
            return []

    def search_by_scalar_field(self, field_name: str, field_value: Union[str, int, float, bool], limit: int = 10, output_fields: Optional[List[str]] = None) -> List[Dict]:
        safe_val = str(field_value).replace('\\', r'\\').replace('"', r'\"')
        expr = f'{field_name} == "{safe_val}"' if isinstance(field_value, str) else f'{field_name} == {field_value}'
        return self.search_with_filter(expr, limit=limit, output_fields=output_fields)

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息 - 通过实际查询获取准确记录数"""
        try:
            desc = self.client.describe_collection(collection_name=self.collection_name)
            
            # 通过实际查询获取准确的记录数（排除已删除的记录）
            try:
                # 检查是否有is_deleted字段
                if self._has_field('is_deleted'):
                    # 查询未删除的记录 - 使用更大的limit或分批查询
                    actual_count = self._get_accurate_count('is_deleted == false')
                else:
                    # 如果没有is_deleted字段，查询所有记录
                    actual_count = self._get_accurate_count('')
                
                self.logger.info(f"通过实际查询获取集合 {self.collection_name} 的记录数: {actual_count}")
            except Exception as query_error:
                self.logger.warning(f"实际查询失败，回退到统计信息: {query_error}")
                # 回退到原来的统计信息方法
                stats = self.client.get_collection_stats(collection_name=self.collection_name)
                if isinstance(stats, dict):
                    actual_count = stats.get("row_count", 0)
                else:
                    import re
                    match = re.search(r'row_count["\']?\s*:\s*(\d+)', str(stats))
                    actual_count = int(match.group(1)) if match else 0
            
            return {
                "description": desc, 
                "collection_name": self.collection_name, 
                "db_name": self.db_name,
                "total_records": actual_count,
                "document_count": actual_count  # 兼容字段
            }
        except Exception as e:
            self.logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}
    
    def get_collection_info_fast(self) -> Dict[str, Any]:
        """快速获取集合信息 - 使用实时查询获取准确记录数"""
        try:
            # 获取集合描述信息
            desc = self.client.describe_collection(collection_name=self.collection_name)
            
            # 获取记录总数 - 使用实时查询而不是缓存统计
            total_count = 0
            try:
                # 使用实际查询获取准确的记录数（排除已删除的记录）
                if self._has_field('is_deleted'):
                    # 查询未删除的记录
                    total_count = self._get_accurate_count('is_deleted == false')
                else:
                    # 如果没有is_deleted字段，查询所有记录
                    total_count = self._get_accurate_count('')
                
                self.logger.info(f"使用实时查询获取集合 {self.collection_name} 的记录数: {total_count}")
                
            except Exception as e1:
                self.logger.warning(f"实时查询失败，回退到缓存统计: {e1}")
                try:
                    # 回退到缓存统计信息
                    stats = self.client.get_collection_stats(collection_name=self.collection_name)
                    if isinstance(stats, dict):
                        total_count = stats.get("row_count", 0)
                    else:
                        # 如果stats是字符串，尝试解析
                        import re
                        match = re.search(r'row_count["\']?\s*:\s*(\d+)', str(stats))
                        if match:
                            total_count = int(match.group(1))
                        else:
                            total_count = -1
                    self.logger.info(f"使用缓存统计获取集合 {self.collection_name} 的记录数: {total_count}")
                    
                except Exception as e2:
                    self.logger.warning(f"缓存统计也失败: {e2}")
                    try:
                        # 最后手段: 使用分页查询估算
                        sample = self.client.query(
                            collection_name=self.collection_name,
                            filter="",
                            output_fields=["id"],
                            limit=100
                        )
                        if sample and len(sample) > 0:
                            if len(sample) == 100:
                                # 可能还有更多数据，返回-1表示未知但非空
                                total_count = -1
                            else:
                                total_count = len(sample)
                        else:
                            total_count = 0
                        self.logger.info(f"使用分页查询估算集合 {self.collection_name} 的记录数: {total_count}")
                    except Exception as e3:
                        self.logger.error(f"所有统计方法都失败: {e3}")
                        total_count = -1  # 表示未知
            
            # 获取集合状态
            try:
                is_loaded = self.client.has_collection(collection_name=self.collection_name)
            except Exception:
                is_loaded = True  # 假设已加载
            
            # 提取字段信息
            fields_info = []
            if desc and hasattr(desc, 'fields'):
                for field in desc.fields:
                    fields_info.append({
                        "name": field.name,
                        "type": str(field.dtype),
                        "is_primary": field.is_primary,
                        "description": getattr(field, 'description', '')
                    })
            
            return {
                "collection_name": self.collection_name,
                "db_name": self.db_name,
                "collection_type": self.collection_type,
                "total_records": total_count,
                "document_count": total_count,  # 兼容字段
                "is_loaded": is_loaded,
                "fields": fields_info,
                "description": getattr(desc, 'description', '') if desc else '',
                "created_at": getattr(desc, 'created_at', '') if desc else '',
                "status": "active" if is_loaded else "inactive"
            }
        except Exception as e:
            self.logger.error(f"获取集合信息失败: {e}")
            return {
                "collection_name": self.collection_name,
                "db_name": self.db_name,
                "collection_type": self.collection_type,
                "error": str(e),
                "status": "error"
            }

    def update_index_params(self, field_name: str, index_type: str = "HNSW", metric_type: str = "COSINE", **kwargs) -> bool:
        """在线创建或更新索引参数（新增二级索引或重建）。"""
        try:
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name=field_name, index_type=index_type, metric_type=metric_type, params=kwargs or None)
            self.client.create_index(collection_name=self.collection_name, index_params=index_params)
            self.client.load_collection(self.collection_name)
            return True
        except Exception as e:
            print(f"更新索引参数失败: {e}")
            return False

    def delete_by_ids(self, ids: List[int]) -> bool:
        try:
            expr = f"id in [{', '.join(str(int(i)) for i in ids)}]"
            self.client.delete(collection_name=self.collection_name, filter=expr)
            
            # 删除后刷新集合统计信息
            try:
                self.client.flush(collection_name=self.collection_name)
                self.logger.info(f"硬删除 {len(ids)} 条记录后已刷新集合统计信息")
            except Exception as flush_error:
                self.logger.warning(f"刷新集合统计信息失败: {flush_error}")
            
            return True
        except Exception as e:
            print(f"删除实体失败: {e}")
            return False

    def mark_deleted(self, ids: List[int]) -> bool:
        """硬删除：直接从数据库中删除记录"""
        # 直接使用硬删除，真正删除数据
        # 软删除是会有一个is_deleted字段，为true表示删除，但是实际上还存在数据库里面，我觉得不太需要这种方式，若需要后面再添加！
        return self.delete_by_ids(ids)
    
    def find_document_chunks(self, document_name: str) -> List[Dict]:
        """查找特定文档的所有chunk片段 - 突破查询限制版本"""
        try:
            if self.collection_type in ["md", "pdf"]:
                # 使用file字段查询
                filter_expr = f'file == "{document_name}"'
            else:
                # 对于image集合，可以使用file字段或source字段
                filter_expr = f'file == "{document_name}"'
            
            # 使用突破限制的查询方法
            return self._search_with_unlimited_filter(filter_expr)
        except Exception as e:
            print(f"查找文档片段失败: {e}")
            return []
    
    def delete_document_chunks(self, document_name: str) -> Tuple[bool, int]:
        """删除特定文档的所有chunk片段（软删除）"""
        try:
            chunks = self.find_document_chunks(document_name)
            if not chunks:
                return True, 0
            
            chunk_ids = [chunk.get("id") for chunk in chunks if chunk.get("id") is not None]
            if chunk_ids:
                success = self.mark_deleted(chunk_ids)
                return success, len(chunk_ids)
            return True, 0
        except Exception as e:
            print(f"删除文档片段失败: {e}")
            return False, 0
    
    def upsert_data(self, data: List[Dict]) -> bool:
        """更新或插入数据（支持文档更新场景）"""
        try:
            if not data:
                return True
            
            # 确保数据格式正确
            processed_data = []
            for item in data:
                processed_item = item.copy()
                # 确保ID是整数
                if "id" in processed_item:
                    processed_item["id"] = int(processed_item["id"])
                # 确保is_deleted字段存在且为False
                processed_item["is_deleted"] = False
                processed_data.append(processed_item)
            
            # 尝试使用upsert，失败则回退到insert
            if hasattr(self.client, "upsert"):
                self.client.upsert(collection_name=self.collection_name, data=processed_data)
            else:
                # 回退：先删除可能存在的ID，再插入
                existing_ids = [item["id"] for item in processed_data if "id" in item]
                if existing_ids:
                    self.delete_by_ids(existing_ids)
                self.client.insert(collection_name=self.collection_name, data=processed_data)
            
            return True
        except Exception as e:
            print(f"数据更新失败: {e}")
            return False
    def drop_collection(self):
        """删除集合"""
        try:
            collections = self.client.list_collections()
            if self.collection_name in collections:
                print(f"删除现有集合 '{self.collection_name}'")
                
                # 确保集合先释放资源再删除
                try:
                    from pymilvus import Collection
                    collection = Collection(self.collection_name)
                    collection.release()  # 释放集合资源
                    print(f"已释放集合 '{self.collection_name}' 的资源")
                except Exception as e:
                    print(f"释放集合资源时警告: {e}")
                
                # 执行删除操作
                self.client.drop_collection(self.collection_name)
                print(f"成功删除集合 '{self.collection_name}'")
                
                # 验证删除是否成功
                updated_collections = self.client.list_collections()
                if self.collection_name not in updated_collections:
                    print(f"集合 '{self.collection_name}' 确认已从数据库中删除")
                    return True
                else:
                    print(f"警告: 集合 '{self.collection_name}' 仍然存在于数据库中")
                    return False
            else:
                print(f"集合 '{self.collection_name}' 不存在，无需删除")
                return True
        except Exception as e:
            print(f"删除集合 '{self.collection_name}' 时发生错误: {e}")
            return False
    
    def get_collection_info(self):
        """获取集合信息"""
        try:
            description = self.client.describe_collection(
                collection_name=self.collection_name
            )
            print(f"集合描述: {description}")
            
            # 获取文档数量 - 修复连接问题
            document_count = 0
            try:
                # 使用现有的客户端连接进行查询
                try:
                    # 尝试简单查询获取数量
                    sample = self.client.query(
                        collection_name=self.collection_name,
                        filter="",
                        output_fields=["id"],
                        limit=10  # 先小批量测试
                    )
                    
                    if sample and len(sample) > 0:
                        # 获取更大样本进行估算
                        try:
                            larger_sample = self.client.query(
                                collection_name=self.collection_name,
                                filter="",
                                output_fields=["id"],
                                limit=1000
                            )
                            document_count = len(larger_sample) if larger_sample else len(sample)
                            print(f"集合文档数量估算: {document_count}")
                        except Exception as e3:
                            # 如果大批量查询失败，使用小样本数量
                            document_count = len(sample)
                            print(f"使用小样本估算文档数量: {document_count}")
                    else:
                        document_count = 0
                        print("集合为空或无法获取数据")
                        
                except Exception as e2:
                    print(f"查询集合数据失败: {e2}")
                    # 如果查询失败，设置为-1表示数量未知但集合存在
                    document_count = -1
                        
                print(f"集合中的文档数量: {document_count}")
                        
            except Exception as e:
                print(f"获取文档数量时出错: {e}")
                document_count = 0
            
            # 返回集合信息
            return {
                'collection_name': self.collection_name,
                'description': description,
                'document_count': document_count,
                'created_timestamp': description.get('created_timestamp', '') if isinstance(description, dict) else '',
                'update_timestamp': description.get('update_timestamp', '') if isinstance(description, dict) else '',
                'status': 'active' if document_count >= 0 else 'unknown'
            }
                        
        except Exception as e:
            print(f"获取集合信息时出错: {e}")
            return {
                'collection_name': self.collection_name,
                'description': '',
                'document_count': 0,
                'created_timestamp': '',
                'update_timestamp': '',
                'status': 'error',
                'error': str(e)
            }

    def scalar_search_image_by_caption(self, caption: str, limit: int = 5) -> List[Dict]:
        """基于标量等值匹配的图片caption查询（仅 image 集合）"""
        if self.collection_type != "image":
            return []
        try:
            # 简单转义，避免过滤表达式语法冲突
            safe = caption.replace('\\', r'\\').replace('"', r'\"')
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f'caption == "{safe}"',
                output_fields=["image_path", "caption", "source", "file"],
                limit=limit
            )
            normalized: List[Dict] = []
            for r in (results or []):
                if isinstance(r, dict):
                    normalized.append({
                        'score': 1.0,
                        'image_path': r.get('image_path', ''),
                        'caption': r.get('caption', ''),
                        'source': r.get('source', '')
                    })
                else:
                    try:
                        normalized.append({
                            'score': 1.0,
                            'image_path': getattr(r, 'image_path', ''),
                            'caption': getattr(r, 'caption', ''),
                            'source': getattr(r, 'source', '')
                        })
                    except Exception:
                        continue
            return normalized
        except Exception as e:
            print(f"标量caption查询出错: {e}")
            return []

    def get_distinct_file_values(self, limit: int = 50000) -> List[str]:
        """获取file字段的所有去重值 - 突破查询窗口限制版本"""
        try:
            # 构建过滤条件，排除软删除的记录
            filter_expr = ""
            if self._has_field("is_deleted"):
                filter_expr = "is_deleted == false"
            
            # 使用突破限制的方法获取去重值
            file_values = set()
            batch_size = 8000  # 单次查询大小
            offset = 0
            max_iterations = 20  # 增加最大迭代次数
            last_max_id = 0  # 用于ID范围查询
            
            for iteration in range(max_iterations):
                try:
                    # 计算当前批次的offset和limit
                    current_offset = offset
                    current_limit = min(batch_size, 16384 - current_offset)
                    
                    # 如果当前offset已经达到或接近16384限制，使用ID范围查询
                    if current_offset >= 16384:
                        if last_max_id > 0:
                            # 使用ID范围查询继续
                            id_filter = f"id > {last_max_id}"
                            if filter_expr:
                                id_filter = f"({filter_expr}) && ({id_filter})"
                            
                            results = self.client.query(
                                collection_name=self.collection_name,
                                filter=id_filter,
                                output_fields=["file", "id"],
                                limit=batch_size
                            )
                        else:
                            break
                    else:
                        # 正常的分批查询
                        results = self.client.query(
                            collection_name=self.collection_name,
                            filter=filter_expr or None,
                            output_fields=["file", "id"],
                            limit=current_limit,
                            offset=current_offset
                        )
                    
                    if not results or len(results) == 0:
                        break  # 没有更多数据
                    
                    # 提取并去重file值，同时记录最大ID
                    batch_files = set()
                    current_max_id = 0
                    for r in results:
                        if isinstance(r, dict):
                            file_val = r.get('file', '')
                            record_id = r.get('id', 0)
                        else:
                            try:
                                file_val = getattr(r, 'file', '')
                                record_id = getattr(r, 'id', 0)
                            except Exception:
                                continue
                        
                        if file_val and file_val.strip():
                            batch_files.add(file_val.strip())
                        
                        if record_id > current_max_id:
                            current_max_id = record_id
                    
                    # 合并到总集合中
                    file_values.update(batch_files)
                    last_max_id = current_max_id
                    
                    # 如果返回的记录数少于limit，说明已经到末尾
                    if len(results) < current_limit:
                        break
                    
                    offset += len(results)
                    
                    # 如果已经达到限制，停止查询
                    if len(file_values) >= limit:
                        break
                    
                    # 如果达到查询窗口限制，准备下一轮
                    if current_offset + current_limit >= 16384:
                        self.logger.info(f"达到查询窗口限制，已获取 {len(file_values)} 个不同文件，准备下一轮查询")
                        # 重置offset，使用ID范围查询继续
                        offset = 0
                        
                except Exception as e:
                    self.logger.warning(f"分批查询file字段失败，iteration={iteration}: {e}")
                    break
            
            # 转换为排序列表
            result_list = sorted(list(file_values))
            self.logger.info(f"集合 {self.collection_name} 获取到 {len(result_list)} 个不同的file值")
            return result_list
            
        except Exception as e:
            self.logger.error(f"获取file字段去重值失败: {e}")
            return []

    def get_file_statistics(self) -> Dict[str, Any]:
        """获取file字段的统计信息 - 使用最新Milvus方法优化"""
        try:
            # 首先获取集合的基本统计信息
            collection_info = self.get_collection_info_fast()
            total_records = collection_info.get("total_records", 0)
            
            # 获取去重的file值
            file_values = self.get_distinct_file_values()
            
            stats = {
                "total_files": len(file_values),
                "file_list": file_values,
                "collection_type": self.collection_type,
                "collection_name": self.collection_name,
                "total_records": total_records,
                "document_count": total_records  # 兼容字段
            }
            
            # 根据文件数量选择不同的统计策略
            if len(file_values) <= 20:
                # 文件数量少，使用精确统计
                file_counts = self._get_file_counts_precise(file_values)
                stats["file_counts"] = file_counts
                self.logger.info(f"使用精确统计方法，统计了 {len(file_values)} 个文件")
                
            elif len(file_values) <= 100:
                # 文件数量中等，使用批量查询
                file_counts = self._get_file_counts_batch(file_values)
                stats["file_counts"] = file_counts
                self.logger.info(f"使用批量查询方法，统计了 {len(file_values)} 个文件")
                
            else:
                # 文件数量多，使用采样方法
                file_counts = self._get_file_counts_sampling(file_values)
                stats["file_counts"] = file_counts
                self.logger.info(f"使用采样统计方法，统计了 {len(file_values)} 个文件")
            
            return stats
        except Exception as e:
            self.logger.error(f"获取file统计信息失败: {e}")
            return {"error": str(e)}
    
    def _get_file_counts_precise(self, file_values: List[str]) -> Dict[str, int]:
        """精确统计每个文件的记录数量 - 突破查询限制版本"""
        file_counts = {}
        for file_val in file_values:
            try:
                # 构建过滤条件
                filter_expr = f'file == "{file_val}"'
                if self._has_field("is_deleted"):
                    filter_expr = f"({filter_expr}) && (is_deleted == false)"
                
                # 使用突破限制的查询方法
                results = self._search_with_unlimited_filter(filter_expr, output_fields=["id"])
                file_counts[file_val] = len(results or [])
            except Exception as e:
                self.logger.warning(f"精确统计文件 {file_val} 的记录数量失败: {e}")
                file_counts[file_val] = 0
        
        return file_counts
    
    def _get_file_counts_batch(self, file_values: List[str]) -> Dict[str, int]:
        """批量统计文件记录数量 - 适用于文件数量中等的情况"""
        file_counts = {}
        
        # 分批处理，每批处理10个文件
        batch_size = 10
        for i in range(0, len(file_values), batch_size):
            batch_files = file_values[i:i + batch_size]
            
            try:
                # 构建批量查询条件
                file_conditions = []
                for file_val in batch_files:
                    file_conditions.append(f'file == "{file_val}"')
                
                # 使用OR条件查询多个文件
                batch_filter = " || ".join(file_conditions)
                if self._has_field("is_deleted"):
                    batch_filter = f"({batch_filter}) && (is_deleted == false)"
                
                # 查询这批文件的所有记录
                results = self.client.query(
                    collection_name=self.collection_name,
                    filter=batch_filter,
                    output_fields=["file"],
                    limit=50000
                )
                
                # 统计每个文件的记录数
                for file_val in batch_files:
                    file_counts[file_val] = 0
                
                for r in (results or []):
                    if isinstance(r, dict):
                        file_val = r.get('file', '')
                    else:
                        try:
                            file_val = getattr(r, 'file', '')
                        except Exception:
                            continue
                    
                    if file_val in file_counts:
                        file_counts[file_val] += 1
                        
            except Exception as e:
                self.logger.warning(f"批量统计文件失败: {e}")
                # 如果批量查询失败，回退到精确统计
                for file_val in batch_files:
                    file_counts[file_val] = 0
        
        return file_counts

    def _get_file_counts_sampling(self, file_values: List[str], sample_size: int = 2000) -> Dict[str, int]:
        """使用采样方法估算文件记录数量 - 优化版本"""
        try:
            file_counts = {}
            
            # 构建过滤条件
            filter_expr = ""
            if self._has_field("is_deleted"):
                filter_expr = "is_deleted == false"
            
            # 使用更大的样本进行更准确的估算
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr or None,
                output_fields=["file"],
                limit=sample_size
            )
            
            # 统计样本中每个file的出现次数
            sample_counts = {}
            for r in (results or []):
                if isinstance(r, dict):
                    file_val = r.get('file', '')
                else:
                    try:
                        file_val = getattr(r, 'file', '')
                    except Exception:
                        continue
                
                if file_val and file_val.strip():
                    file_val = file_val.strip()
                    sample_counts[file_val] = sample_counts.get(file_val, 0) + 1
            
            # 根据样本比例估算总数
            total_sample_records = sum(sample_counts.values())
            if total_sample_records > 0:
                # 获取集合总记录数 - 使用优化后的方法
                collection_info = self.get_collection_info_fast()
                total_records = collection_info.get("total_records", 0)
                
                if total_records > 0:
                    # 按比例估算
                    ratio = total_records / total_sample_records
                    for file_val in file_values:
                        sample_count = sample_counts.get(file_val, 0)
                        estimated_count = int(sample_count * ratio)
                        # 确保估算值不为负数
                        file_counts[file_val] = max(0, estimated_count)
                else:
                    # 如果无法获取总数，使用样本数据
                    for file_val in file_values:
                        file_counts[file_val] = sample_counts.get(file_val, 0)
            else:
                # 如果样本为空，所有文件计数为0
                for file_val in file_values:
                    file_counts[file_val] = 0
            
            self.logger.info(f"采样统计完成，样本大小: {total_sample_records}, 估算了 {len(file_values)} 个文件")
            return file_counts
        except Exception as e:
            self.logger.error(f"采样统计失败: {e}")
            # 降级到简单方法
            return {file_val: 0 for file_val in file_values}

    def delete_by_file(self, file_value: str) -> Tuple[bool, int]:
        """根据file字段值删除所有相关记录 - 突破查询限制版本"""
        try:
            # 构建过滤条件
            filter_expr = f'file == "{file_value}"'
            if self._has_field("is_deleted"):
                filter_expr = f"({filter_expr}) && (is_deleted == false)"
            
            # 使用突破限制的查询方法查找所有相关的记录
            related_records = self._search_with_unlimited_filter(filter_expr, output_fields=["id"])
            
            if not related_records:
                return True, 0
            
            # 提取ID列表
            record_ids = []
            for r in related_records:
                if isinstance(r, dict):
                    record_id = r.get('id')
                else:
                    try:
                        record_id = getattr(r, 'id', None)
                    except Exception:
                        continue
                
                if record_id is not None:
                    record_ids.append(int(record_id))
            
            if not record_ids:
                return True, 0
            
            # 执行硬删除（真正删除数据）
            success = self.mark_deleted(record_ids)
            
            # 删除后刷新集合统计信息
            if success and record_ids:
                try:
                    self.client.flush(collection_name=self.collection_name)
                    self.logger.info(f"硬删除 {len(record_ids)} 条记录后已刷新集合统计信息")
                except Exception as flush_error:
                    self.logger.warning(f"刷新集合统计信息失败: {flush_error}")
            
            return success, len(record_ids)
        except Exception as e:
            self.logger.error(f"根据file字段删除记录失败: {e}")
            return False, 0

    def batch_delete_by_files(self, file_values: List[str]) -> Dict[str, Tuple[bool, int]]:
        """批量根据file字段值删除记录"""
        results = {}
        for file_val in file_values:
            success, count = self.delete_by_file(file_val)
            results[file_val] = (success, count)
        return results