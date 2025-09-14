import os
from typing import List, Dict, Optional, Union, Tuple
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
                schema.add_field(field_name="md_file", datatype=DataType.VARCHAR, max_length=512)
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
                schema.add_field(field_name="pdf_file", datatype=DataType.VARCHAR, max_length=512)
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
                schema.add_field(field_name=self.domain_field_name, datatype=DataType.VARCHAR, max_length=128)
                schema.add_field(field_name="is_deleted", datatype=DataType.BOOL)
                
                # 索引
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="id", index_type="AUTOINDEX")
                index_params.add_index(field_name="image_vector", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="caption_vector_clip_768", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="source", index_type="AUTOINDEX")
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
                        "md_file": data.get('md_file', '')[:512],
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
                        "pdf_file": data.get('pdf_file', '')[:512],
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

                    insert_data = {
                        "id": int(data.get('id', i + 1)),
                        "image_vector": embeddings[i] if embeddings else [0.0] * 768,  # 修改为768
                        "caption_vector_clip_768": cap_clip_vec if cap_clip_vec is not None else [0.0] * 768,
                        "image_path": img_path_val[:512],
                        "caption": caption_val[:65535],
                        "source": source_val[:256],
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
            desired_fields = ["id", "content", "md_file", "title", "image_paths", "image_captions", "document_name", "chunk_index", self.domain_field_name, "is_deleted"]
        elif self.collection_type == "pdf":
            # 根据查询向量维度自动选择字段
            if len(query_embedding) == 1024:
                field_name = "text_vector_qwen_1024"
            elif len(query_embedding) == 768:
                field_name = "text_vector_clip_768"
            else:
                print(f"警告：查询向量维度为 {len(query_embedding)}，期望1024或768")
                return []
            desired_fields = ["id", "content", "pdf_file", "title", "image_paths", "image_captions", "document_name", "chunk_index", "page_start", "page_end", self.domain_field_name, "is_deleted"]
        elif self.collection_type == "image":
            field_name = "image_vector"
            desired_fields = ["id", "image_path", "caption", "source", self.domain_field_name, "is_deleted"]

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
                desired = base_fields + ["content", "md_file", "title", "image_paths", "image_captions", "document_name", "chunk_index"]
            elif self.collection_type == "pdf":
                desired = base_fields + ["content", "pdf_file", "title", "image_paths", "image_captions", "document_name", "chunk_index", "page_start", "page_end"]
            else:
                desired = base_fields + ["image_path", "caption", "source"]
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
            # 默认：返回集合中实际存在的全部字段（更稳健，兼容旧集合字段名不一致的情况）
            existing = list(self._get_collection_fields() or [])
            if self.domain_field_name not in existing:
                existing.append(self.domain_field_name)
            # 如果拿不到字段列表，则传 None 让 Milvus 返回全部字段
            output_fields = existing if existing else None
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

    def search_by_scalar_field(self, field_name: str, field_value: Union[str, int, float, bool], limit: int = 10, output_fields: Optional[List[str]] = None) -> List[Dict]:
        safe_val = str(field_value).replace('\\', r'\\').replace('"', r'\"')
        expr = f'{field_name} == "{safe_val}"' if isinstance(field_value, str) else f'{field_name} == {field_value}'
        return self.search_with_filter(expr, limit=limit, output_fields=output_fields)

    def get_collection_stats(self) -> Dict:
        try:
            desc = self.client.describe_collection(collection_name=self.collection_name)
            try:
                stats = self.client.get_collection_statistics(collection_name=self.collection_name)
            except Exception:
                stats = {"row_count": self.client.num_entities(collection_name=self.collection_name)}
            return {"description": desc, "stats": stats, "collection_name": self.collection_name, "db_name": self.db_name}
        except Exception as e:
            return {"error": str(e)}

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
            return True
        except Exception as e:
            print(f"删除实体失败: {e}")
            return False

    def mark_deleted(self, ids: List[int]) -> bool:
        """软删除：尽量通过upsert将 is_deleted 置为true；失败则回退为硬删除。"""
        try:
            rows = [{"id": int(i), "is_deleted": True} for i in ids]
            # 一些版本的客户端提供 upsert
            if hasattr(self.client, "upsert"):
                self.client.upsert(collection_name=self.collection_name, data=rows)
                return True
            # 回退：硬删除
            return self.delete_by_ids(ids)
        except Exception:
            return self.delete_by_ids(ids)
    
    def find_document_chunks(self, document_name: str) -> List[Dict]:
        """查找特定文档的所有chunk片段"""
        try:
            if self.collection_type in ["md", "pdf"]:
                filter_expr = f'document_name == "{document_name}"'
            else:
                filter_expr = f'source == "{document_name}"'
            
            return self.search_with_filter(filter_expr, limit=10000)
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
        collections = self.client.list_collections()
        if self.collection_name in collections:
            print(f"删除现有集合 '{self.collection_name}'")
            self.client.drop_collection(self.collection_name)
    
    def get_collection_info(self):
        """获取集合信息"""
        try:
            description = self.client.describe_collection(
                collection_name=self.collection_name
            )
            print(f"集合描述: {description}")
            
            try:
                stats = self.client.get_collection_statistics(
                    collection_name=self.collection_name
                )
                print(f"集合统计信息: {stats}")
            except:
                try:
                    row_count = self.client.num_entities(
                        collection_name=self.collection_name
                    )
                    print(f"集合中的实体数量: {row_count}")
                except:
                    try:
                        count_result = self.client.query(
                            collection_name=self.collection_name,
                            filter="",
                            output_fields=["id"],
                            limit=1
                        )
                        print(f"集合查询测试成功，可以正常访问数据")
                    except Exception as e:
                        print(f"无法获取集合统计信息: {e}")
                        
        except Exception as e:
            print(f"获取集合信息时出错: {e}")

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
                output_fields=["image_path", "caption", "source"],
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