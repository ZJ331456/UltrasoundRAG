"""
UltrasoundRAG API集合管理模块
提供集合的创建、删除、查询、文档管理等接口
"""

from typing import List
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Query

from .models import StandardResponse, TaskResponse
from .utils import (
    _infer_collection_type, _create_standard_response, _handle_collection_operation_error,
    _create_milvus_manager_safely, _find_dataset_name_by_collection
)

# 创建路由器
router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=StandardResponse)
async def get_all_collections():
    """
    列出所有集合（重命名以避免函数名冲突）
    
    Returns:
        包含所有集合信息的响应
    """
    try:
        from ...core.indexing import list_collections, get_collection_info
        
        # 获取集合名称列表
        collection_names = list_collections()
        
        # 为每个集合获取详细信息
        collections = []
        for name in collection_names:
            try:
                info = get_collection_info(name)
                # 构造集合信息对象
                collection_data = {
                    'name': name,
                    'type': _infer_collection_type(name),  # 从名称推断类型
                    'status': 'active' if info else 'unknown',
                    'document_count': info.get('document_count', 0) if info else 0,
                    'description': info.get('description', '') if info else '',
                    'created_at': info.get('created_timestamp', '') if info else '',
                    'updated_at': info.get('update_timestamp', '') if info else ''
                }
                collections.append(collection_data)
            except Exception as e:
                # 如果获取某个集合信息失败，使用默认值
                collection_data = {
                    'name': name,
                    'type': _infer_collection_type(name),
                    'status': 'unknown',
                    'document_count': 0,
                    'description': '',
                    'created_at': '',
                    'updated_at': ''
                }
                collections.append(collection_data)
        
        return _create_standard_response(
            success=True,
            data={
                'collections': collections,
                'total': len(collections)
            }
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"列出集合失败: {str(e)}"
        )


@router.get("/source-files-summary", response_model=StandardResponse)
async def get_all_collections_source_files_summary():
    """获取所有集合的源文件统计摘要"""
    try:
        from ...core.indexing import list_collections
        
        # 获取所有集合名称
        collection_names = list_collections()
        
        summary_data = []
        total_source_files = 0
        total_chunks_processed = 0
        
        for collection_name in collection_names:
            try:
                manager = _create_milvus_manager_safely(collection_name)
                if not manager:
                    continue
                
                # 快速统计：只获取前1000条记录来估算源文件数量
                sample_docs = manager.search_with_filter("", limit=1000)
                
                # 统计源文件
                source_files = set()
                for doc in sample_docs:
                    # 所有集合都使用file字段
                    source_key = doc.get('file', '')
                    if source_key:
                        source_files.add(source_key)
                
                collection_summary = {
                    'collection_name': collection_name,
                    'collection_type': _infer_collection_type(collection_name),
                    'estimated_source_files': len(source_files),
                    'sample_size': len(sample_docs),
                    'note': f'基于前{len(sample_docs)}条记录估算' if len(sample_docs) < 1000 else '基于前1000条记录估算'
                }
                
                summary_data.append(collection_summary)
                total_source_files += len(source_files)
                total_chunks_processed += len(sample_docs)
                
            except Exception as e:
                # 如果某个集合查询失败，记录错误但继续处理其他集合
                collection_summary = {
                    'collection_name': collection_name,
                    'collection_type': _infer_collection_type(collection_name),
                    'estimated_source_files': 0,
                    'sample_size': 0,
                    'note': f'查询失败: {str(e)}'
                }
                summary_data.append(collection_summary)
        
        return _create_standard_response(
            success=True,
            data={
                'collections_summary': summary_data,
                'total_collections': len(collection_names),
                'total_estimated_source_files': total_source_files,
                'total_chunks_processed': total_chunks_processed,
                'note': '这是基于样本数据的估算，实际源文件数量可能更多'
            }
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取集合源文件统计失败: {str(e)}"
        )


@router.post("", response_model=StandardResponse)
async def create_collection(
    name: str = Form(...),
    type: str = Form(..., description="集合类型: md, image, pdf"),
    description: str = Form("", description="集合描述")
):
    """
    创建新集合
    
    Args:
        name: 集合名称
        type: 集合类型（md, image, pdf）
        description: 集合描述
        
    Returns:
        创建操作结果
    """
    try:
        from ...core.indexing import create_collection
        success = create_collection(name, type)
        
        if success:
            return _create_standard_response(
                success=True,
                message=f"集合 {name} 创建成功",
                data={'name': name, 'type': type, 'description': description}
            )
        else:
            return _create_standard_response(
                success=False,
                error=f"集合 {name} 创建失败"
            )
    except Exception as e:
        return _handle_collection_operation_error("创建", name, e)


@router.get("/{collection_name}", response_model=StandardResponse)
async def get_collection_info(collection_name: str):
    """获取集合信息"""
    try:
        from ...core.indexing import get_collection_info
        info = get_collection_info(collection_name)
        
        return _create_standard_response(
            success=True,
            data={
                'collection': collection_name,
                'info': info
            }
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取集合信息失败: {str(e)}"
        )


@router.delete("/{collection_name}", response_model=StandardResponse)
async def delete_collection(collection_name: str):
    """
    删除指定集合
    
    Args:
        collection_name: 要删除的集合名称
        
    Returns:
        删除操作结果
    """
    try:
        from ...core.indexing import delete_collection
        success = delete_collection(collection_name)
        
        if success:
            return _create_standard_response(
                success=True,
                message=f"集合 {collection_name} 删除成功"
            )
        else:
            return _create_standard_response(
                success=False,
                error=f"集合 {collection_name} 删除失败"
            )
    except Exception as e:
        return _handle_collection_operation_error("删除", collection_name, e)


@router.get("/{collection_name}/documents", response_model=StandardResponse)
async def list_collection_documents(collection_name: str, offset: int = 0, limit: int = 20):
    """
    获取集合中的源文件列表（智能分批查询）
    
    Args:
        collection_name: 集合名称
        offset: 偏移量（分页）
        limit: 限制数量（分页）
        
    Returns:
        源文件列表和统计信息
    """
    try:
        # 使用安全创建的管理器（减少重复代码）
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 智能分批查询策略：避免"query results exceed the limit size"错误
        source_files = {}
        unique_sources = set()  # 用于快速去重
        batch_size = 2000  # 只查询source字段，可以增加批次大小
        max_batches = 100  # 增加批次数量
        total_processed = 0
        
        for batch_num in range(max_batches):
            try:
                # 只查询source字段，减少数据传输量
                batch_docs = manager.search_with_filter("", limit=batch_size, output_fields=["file"])
                if not batch_docs:
                    break  # 没有更多数据了
                
                total_processed += len(batch_docs)
                
                # 收集唯一的源文件字段（使用Set快速去重）
                for doc in batch_docs:
                    # 所有集合都使用file字段
                    source_key = doc.get('file', '')
                    if source_key and source_key not in unique_sources:
                        unique_sources.add(source_key)
                        file_type = 'image' if _infer_collection_type(collection_name) == 'image' else 'document'
                        source_files[source_key] = {
                            'source_file': source_key,
                            'file_type': file_type,
                            'chunk_count': 1
                        }
                    elif source_key in source_files:
                        source_files[source_key]['chunk_count'] += 1
                
                # 如果返回的数据少于batch_size，说明已经获取完了
                if len(batch_docs) < batch_size:
                    break
                    
            except Exception as e:
                if "exceed the limit size" in str(e):
                    # 如果遇到查询限制，尝试更小的批次
                    try:
                        smaller_batch = manager.search_with_filter("", limit=500, output_fields=["file"])
                        if smaller_batch:
                            total_processed += len(smaller_batch)
                            # 处理小批次数据...
                            for doc in smaller_batch:
                                # 所有集合都使用file字段
                                source_key = doc.get('file', '')
                                if source_key and source_key not in source_files:
                                    file_type = 'image' if _infer_collection_type(collection_name) == 'image' else 'document'
                                    source_files[source_key] = {
                                        'source_file': source_key,
                                        'file_type': file_type,
                                        'chunk_count': 1
                                    }
                                elif source_key in source_files:
                                    source_files[source_key]['chunk_count'] += 1
                    except:
                        pass
                break  # 停止查询，使用已获取的数据
        
        # 转换为列表格式
        all_source_files = list(source_files.values())
        
        # 对源文件列表应用分页
        total_source_files = len(all_source_files)
        paginated_source_files = all_source_files[offset:offset + limit]
        
        # 生成统计信息
        stats_info = f'已处理{total_processed}条数据块，发现{total_source_files}个源文件'
        if total_processed >= 50000:
            stats_info += '（已达到查询上限，可能还有更多源文件）'
        
        return _create_standard_response(
            success=True,
            data={
                'documents': paginated_source_files,
                'total': total_source_files,  # 返回源文件总数
                'offset': offset,
                'limit': limit,
                'stats': {
                    'processed_chunks': total_processed,
                    'source_files_found': total_source_files,
                    'note': stats_info
                }
            }
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取集合源文件列表失败: {str(e)}"
        )


@router.get("/{collection_name}/files", response_model=StandardResponse)
async def get_collection_file_list(collection_name: str):
    """
    获取集合中所有file字段的去重值列表
    
    Args:
        collection_name: 集合名称
        
    Returns:
        文件列表和统计信息
    """
    try:
        # 使用安全创建的管理器（减少重复代码）
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 获取file字段的去重值
        file_values = manager.get_distinct_file_values()
        
        return _create_standard_response(
            success=True,
            message=f"成功获取集合 {collection_name} 的file列表，共 {len(file_values)} 个文件",
            data={
                "collection_name": collection_name,
                "collection_type": _infer_collection_type(collection_name),
                "total_files": len(file_values),
                "file_list": file_values
            }
        )
    except Exception as e:
        return _handle_collection_operation_error("获取文件列表", collection_name, e)


@router.get("/{collection_name}/files/statistics", response_model=StandardResponse)
async def get_collection_file_statistics(collection_name: str):
    """获取集合中file字段的统计信息 - 优化版本"""
    try:
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 获取file字段的统计信息（使用优化版本）
        stats = manager.get_file_statistics()
        
        if "error" in stats:
            return _create_standard_response(
                success=False,
                error=f"获取file统计信息失败: {stats['error']}"
            )
        
        return _create_standard_response(
            success=True,
            data=stats,
            message=f"成功获取集合 {collection_name} 的file统计信息"
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取file统计信息失败: {str(e)}"
        )


@router.get("/{collection_name}/info/fast", response_model=StandardResponse)
async def get_collection_info_fast(collection_name: str):
    """快速获取集合信息 - 不遍历数据，只获取元数据"""
    try:
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 获取集合快速信息
        info = manager.get_collection_info_fast()
        
        if "error" in info:
            return _create_standard_response(
                success=False,
                error=f"获取集合信息失败: {info['error']}"
            )
        
        return _create_standard_response(
            success=True,
            data=info,
            message=f"成功获取集合 {collection_name} 的快速信息"
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取集合信息失败: {str(e)}"
        )


@router.get("/{collection_name}/stats/optimized", response_model=StandardResponse)
async def get_collection_stats_optimized(collection_name: str):
    """获取集合统计信息 - 高效版本，不遍历数据"""
    try:
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 获取集合统计信息
        stats = manager.get_collection_stats()
        
        if "error" in stats:
            return _create_standard_response(
                success=False,
                error=f"获取集合统计信息失败: {stats['error']}"
            )
        
        return _create_standard_response(
            success=True,
            data=stats,
            message=f"成功获取集合 {collection_name} 的统计信息"
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取集合统计信息失败: {str(e)}"
        )


@router.delete("/{collection_name}/files", response_model=StandardResponse)
async def delete_collection_file(collection_name: str, file_value: str = Query(..., description="要删除的文件标识符")):
    """
    根据file字段值删除集合中的所有相关记录
    
    Args:
        collection_name: 集合名称
        file_value: 要删除的文件标识符
        
    Returns:
        删除操作结果，包含删除的记录数量
    """
    try:
        # 使用安全创建的管理器
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 根据file字段值删除记录
        success, count = manager.delete_by_file(file_value)
        
        if success:
            return _create_standard_response(
                success=True,
                message=f"成功删除文件 {file_value} 的所有记录，共删除 {count} 条记录",
                data={
                    "file_value": file_value,
                    "deleted_count": count,
                    "collection_name": collection_name
                }
            )
        else:
            return _create_standard_response(
                success=False,
                error=f"删除文件 {file_value} 的记录失败"
            )
    except Exception as e:
        return _handle_collection_operation_error("删除文件", collection_name, e)


@router.post("/{collection_name}/files/batch-delete", response_model=StandardResponse)
async def batch_delete_collection_files(collection_name: str, file_values: List[str]):
    """批量根据file字段值删除记录"""
    try:
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 批量删除
        results = manager.batch_delete_by_files(file_values)
        
        # 统计结果
        total_deleted = 0
        success_count = 0
        failed_files = []
        
        for file_val, (success, count) in results.items():
            if success:
                success_count += 1
                total_deleted += count
            else:
                failed_files.append(file_val)
        
        return _create_standard_response(
            success=len(failed_files) == 0,
            data={
                "total_files": len(file_values),
                "success_count": success_count,
                "failed_count": len(failed_files),
                "total_deleted": total_deleted,
                "failed_files": failed_files,
                "detailed_results": results
            },
            message=f"批量删除完成：成功 {success_count}/{len(file_values)} 个文件，共删除 {total_deleted} 条记录"
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"批量删除文件记录失败: {str(e)}"
        )
