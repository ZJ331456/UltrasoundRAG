import os
from typing import List, Dict, Optional, Union, Tuple
from pymilvus import MilvusClient, DataType
from UltrasoundRAG.config import config  # 添加导入

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
        else:
            raise ValueError("collection_type 必须是 'md' 或 'image'")
        
        # 初始化Milvus客户端
        self.client = MilvusClient(uri=self.milvus_uri, token=self.milvus_token)
        
        # 初始化数据库和集合
        self._setup_database()
        self._setup_collection()
    
    def _setup_database(self):
        """设置Milvus数据库"""
        if self.db_name not in self.client.list_databases():
            self.client.create_database(db_name=self.db_name)
            print(f"数据库 '{self.db_name}' 创建成功")
        else:
            print(f"数据库 '{self.db_name}' 已存在")
        
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
                schema.add_field(field_name="image_links", datatype=DataType.VARCHAR, max_length=65535)
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
                
            elif self.collection_type == "image":
                # 图片集合字段
                schema.add_field(field_name="image_vector", datatype=DataType.FLOAT_VECTOR, dim=768)  # 修改为768
                schema.add_field(field_name="relative_path", datatype=DataType.VARCHAR, max_length=512)
                schema.add_field(field_name="caption", datatype=DataType.VARCHAR, max_length=1024)
                schema.add_field(field_name="folder_name", datatype=DataType.VARCHAR, max_length=256)
                schema.add_field(field_name=self.domain_field_name, datatype=DataType.VARCHAR, max_length=128)
                schema.add_field(field_name="is_deleted", datatype=DataType.BOOL)
                
                # 索引
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="id", index_type="AUTOINDEX")
                index_params.add_index(field_name="image_vector", index_type="HNSW", metric_type="COSINE")
                index_params.add_index(field_name="folder_name", index_type="AUTOINDEX")
                index_params.add_index(field_name=self.domain_field_name, index_type="AUTOINDEX")
                index_params.add_index(field_name="is_deleted", index_type="AUTOINDEX")
            
            # 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            print(f"集合 '{self.collection_name}' ({self.collection_type}) 创建成功")
        else:
            print(f"集合 '{self.collection_name}' ({self.collection_type}) 已存在")
        
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
    
    def insert_data(self, data_list: List[Dict], embeddings: Optional[List[List[float]]] = None, embeddings_qwen: Optional[List[List[float]]] = None, embeddings_clip: Optional[List[List[float]]] = None, domain_values: Optional[List[Optional[str]]] = None) -> bool:
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
                    # 规范化 image_links 为 JSON 字符串（避免二次编码）
                    raw_links = data.get('image_links', [])
                    if isinstance(raw_links, str):
                        try:
                            parsed = json.loads(raw_links)
                            raw_links = parsed if isinstance(parsed, list) else [str(raw_links)]
                        except Exception:
                            raw_links = [str(raw_links)]
                    image_links = json.dumps(raw_links, ensure_ascii=False)
                    if len(image_links) > 65535:
                        image_links = image_links[:65535]

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
                        "image_links": image_links,
                        "image_captions": image_captions,
                        "document_name": data.get('document_name', '')[:256],
                        "chunk_index": int(data.get('chunk_index', 0)),
                        self.domain_field_name: (domain_values[i] if domain_values and i < len(domain_values) else data.get(self.domain_field_name, ''))[:128] if (domain_values or data.get(self.domain_field_name)) else "",
                        "is_deleted": bool(data.get('is_deleted', False))
                    }
                    
                elif self.collection_type == "image":
                    # 处理图片数据
                    if embeddings and len(embeddings[i]) != 768:  # 修改为768
                        print(f"警告：第 {i + 1} 个图片的向量维度为 {len(embeddings[i])}，期望768")
                        continue
                    
                    insert_data = {
                        "id": int(data.get('id', i + 1)),
                        "image_vector": embeddings[i] if embeddings else [0.0] * 768,  # 修改为768
                        "relative_path": data.get('relative_path', '')[:512],
                        "caption": data.get('caption', '')[:1024],
                        "folder_name": data.get('folder_name', '')[:256],
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
                    
            except Exception as e:
                print(f"插入第 {i + 1} 个数据时出错: {e}")
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
            output_fields = ["id", "content", "md_file", "title", "image_links", "image_captions", "document_name", "chunk_index", self.domain_field_name, "is_deleted"]
        elif self.collection_type == "image":
            field_name = "image_vector"
            output_fields = ["id", "relative_path", "caption", "folder_name", self.domain_field_name, "is_deleted"]
        
        try:
            # 自动追加软删除过滤
            composed_filter = filter_expr.strip()
            if "is_deleted" in output_fields:
                delete_guard = "is_deleted == false"
                composed_filter = delete_guard if not composed_filter else f"({delete_guard}) && ({composed_filter})"

            results = self.client.search(collection_name=self.collection_name, data=[query_embedding], anns_field=field_name, limit=top_k, output_fields=output_fields, filter=composed_filter or None, partition_names=partition_names)
            
            formatted_results = []
            if results and len(results) > 0:
                for result in results[0]:
                    result_dict = {
                        'score': result.score,
                        **{field: result.entity.get(field) for field in output_fields}
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

                        if 'image_links' in result_dict:
                            result_dict['image_links'] = _robust_json_loads(result_dict.get('image_links'))
                        if 'image_captions' in result_dict:
                            result_dict['image_captions'] = _robust_json_loads(result_dict.get('image_captions'))
                    
                    formatted_results.append(result_dict)
            
            return formatted_results
            
        except Exception as e:
            print(f"搜索时出错: {e}")
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
            # 输出字段尽量齐全
            base_fields = ["id", self.domain_field_name, "is_deleted"]
            if self.collection_type == "md":
                output_fields = base_fields + ["content", "md_file", "title", "image_links", "image_captions", "document_name", "chunk_index"]
            else:
                output_fields = base_fields + ["relative_path", "caption", "folder_name"]

            composed_filter = filter_expr.strip()
            if "is_deleted" in output_fields:
                delete_guard = "is_deleted == false"
                composed_filter = delete_guard if not composed_filter else f"({delete_guard}) && ({composed_filter})"

            try:
                res = self.client.search(collection_name=self.collection_name, data=[embedding], anns_field=field_name, limit=top_k, output_fields=output_fields, filter=composed_filter or None, partition_names=partition_names)
            except Exception:
                res = []
            if not res:
                continue
            field_weight = float(weights.get(field_name, 1.0))
            for r in res[0]:
                ent = {of: r.entity.get(of) for of in output_fields}
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
            if self.collection_type == "md":
                output_fields = ["id", "content", "title", "md_file", "document_name", "chunk_index", self.domain_field_name]
            else:
                output_fields = ["id", "relative_path", "caption", "folder_name", self.domain_field_name]
        # 自动追加软删除过滤
        composed_filter = filter_expr.strip()
        delete_guard = "is_deleted == false"
        composed_filter = delete_guard if not composed_filter else f"({delete_guard}) && ({composed_filter})"
        try:
            rows = self.client.query(collection_name=self.collection_name, filter=composed_filter, output_fields=output_fields, limit=limit)
            normalized: List[Dict] = []
            for r in rows or []:
                if isinstance(r, dict):
                    normalized.append(r)
                else:
                    # 兜底读取
                    entry = {f: getattr(r, f, None) for f in output_fields}
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
                output_fields=["relative_path", "caption", "folder_name"],
                limit=limit
            )
            normalized: List[Dict] = []
            for r in (results or []):
                if isinstance(r, dict):
                    normalized.append({
                        'score': 1.0,
                        'relative_path': r.get('relative_path', ''),
                        'caption': r.get('caption', ''),
                        'folder_name': r.get('folder_name', '')
                    })
                else:
                    try:
                        normalized.append({
                            'score': 1.0,
                            'relative_path': getattr(r, 'relative_path', ''),
                            'caption': getattr(r, 'caption', ''),
                            'folder_name': getattr(r, 'folder_name', '')
                        })
                    except Exception:
                        continue
            return normalized
        except Exception as e:
            print(f"标量caption查询出错: {e}")
            return []