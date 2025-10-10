"""
UltrasoundRAG API文档管理模块
提供文档的上传、更新、删除等管理接口
"""

import json
import asyncio
import tempfile
import os
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Form, File, UploadFile

from .models import StandardResponse
from .utils import (
    _create_standard_response, _detect_document_type, _validate_file_for_type,
    _process_markdown_document, _process_image_document, _process_pdf_document
)

# 创建路由器
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/{collection_name}", response_model=StandardResponse)
async def add_document_to_collection(
    collection_name: str,
    file: UploadFile = File(...),
    doc_type: str = Form("md", description="文档类型: md, image, pdf"),
    metadata: Optional[str] = Form(None, description="文档元数据（JSON格式）"),
    auto_detect_type: bool = Form(False, description="自动检测文档类型")
):
    """
    向集合添加单个文档（增强版）
    
    功能特性：
    1. 支持多种文档类型的智能处理
    2. 自动类型检测
    3. 元数据支持
    4. 增强的错误处理
    5. 文件验证和安全检查
    
    Args:
        collection_name: 集合名称
        file: 上传的文件
        doc_type: 文档类型（md, image, pdf）
        metadata: 文档元数据（JSON格式）
        auto_detect_type: 是否自动检测文档类型
    """
    try:
        # 1. 文件基础验证
        if not file.filename:
            return _create_standard_response(
                success=False,
                error="文件名不能为空"
            )
        
        # 2. 自动检测文档类型（如果启用）
        if auto_detect_type:
            detected_type = _detect_document_type(file.filename, file.content_type)
            if detected_type:
                doc_type = detected_type
                
        # 3. 文件类型验证
        validation_result = _validate_file_for_type(file, doc_type)
        if not validation_result['valid']:
            return _create_standard_response(
                success=False,
                error=f"文件验证失败: {validation_result['error']}"
            )
        
        # 4. 解析元数据
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                return _create_standard_response(
                    success=False,
                    error=f"元数据JSON格式错误: {str(e)}"
                )
        
        # 5. 根据文档类型选择处理策略
        if doc_type == "image":
            result = await _process_image_document(file, collection_name, parsed_metadata)
        elif doc_type == "md":
            result = await _process_markdown_document(file, collection_name, parsed_metadata)
        elif doc_type == "pdf":
            result = await _process_pdf_document(file, collection_name, parsed_metadata)
        else:
            return _create_standard_response(
                success=False,
                error=f"不支持的文档类型: {doc_type}"
            )
        
        if result['success']:
            return _create_standard_response(
                success=True,
                message=f"文档已成功添加到集合 {collection_name}",
                data={
                    'collection': collection_name,
                    'filename': file.filename,
                    'type': doc_type,
                    'size': file.size if hasattr(file, 'size') else len(await file.read()),
                    'metadata': parsed_metadata,
                    'processing_details': result.get('details', {})
                }
            )
        else:
            return _create_standard_response(
                success=False,
                error=result.get('error', '文档处理失败')
            )
            
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"添加文档失败: {str(e)}"
        )


@router.post("/{collection_name}/batch", response_model=StandardResponse)
async def batch_add_documents_to_collection(
    collection_name: str,
    files: List[UploadFile] = File(...),
    doc_types: Optional[str] = Form(None, description="文档类型列表（JSON格式），如果为空则自动检测"),
    metadata_list: Optional[str] = Form(None, description="元数据列表（JSON格式）"),
    auto_detect_types: bool = Form(True, description="自动检测文档类型"),
    continue_on_error: bool = Form(True, description="遇到错误时是否继续处理其他文件")
):
    """
    批量向集合添加文档（增强版）
    
    功能特性：
    1. 支持多文件并行处理
    2. 智能类型检测和验证
    3. 灵活的错误处理策略
    4. 详细的处理结果报告
    5. 支持混合文档类型上传
    
    Args:
        collection_name: 集合名称
        files: 上传的文件列表
        doc_types: 文档类型列表（JSON格式）
        metadata_list: 元数据列表（JSON格式）
        auto_detect_types: 自动检测文档类型
        continue_on_error: 遇到错误时是否继续处理
    """
    try:
        # 1. 基础验证
        if not files:
            return _create_standard_response(
                success=False,
                error="未提供文件"
            )
        
        if len(files) > 50:  # 限制批量上传数量
            return _create_standard_response(
                success=False,
                error="批量上传文件数量不能超过50个"
            )
        
        # 2. 解析文档类型列表
        parsed_doc_types = []
        if doc_types:
            try:
                parsed_doc_types = json.loads(doc_types)
                if len(parsed_doc_types) != len(files):
                    return _create_standard_response(
                        success=False,
                        error="文档类型列表长度与文件数量不匹配"
                    )
            except json.JSONDecodeError as e:
                return _create_standard_response(
                    success=False,
                    error=f"文档类型列表JSON格式错误: {str(e)}"
                )
        
        # 3. 解析元数据列表
        parsed_metadata_list = []
        if metadata_list:
            try:
                parsed_metadata_list = json.loads(metadata_list)
                if len(parsed_metadata_list) != len(files):
                    return _create_standard_response(
                        success=False,
                        error="元数据列表长度与文件数量不匹配"
                    )
            except json.JSONDecodeError as e:
                return _create_standard_response(
                    success=False,
                    error=f"元数据列表JSON格式错误: {str(e)}"
                )
        
        # 4. 准备处理任务
        processing_tasks = []
        for i, file in enumerate(files):
            # 确定文档类型
            if parsed_doc_types and i < len(parsed_doc_types):
                doc_type = parsed_doc_types[i]
            elif auto_detect_types:
                doc_type = _detect_document_type(file.filename, file.content_type)
                if not doc_type:
                    doc_type = "md"  # 默认类型
            else:
                doc_type = "md"  # 默认类型
            
            # 获取元数据
            metadata = {}
            if parsed_metadata_list and i < len(parsed_metadata_list):
                metadata = parsed_metadata_list[i]
            
            processing_tasks.append({
                'file': file,
                'doc_type': doc_type,
                'metadata': metadata,
                'index': i
            })
        
        # 5. 并行处理文件
        results = await _batch_process_documents(processing_tasks, collection_name, continue_on_error)
        
        # 6. 统计结果
        successful_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - successful_count
        
        overall_success = failed_count == 0 or (continue_on_error and successful_count > 0)
        
        return _create_standard_response(
            success=overall_success,
            message=f"批量处理完成：成功 {successful_count}/{len(files)} 个文件",
            data={
                'collection': collection_name,
                'total_files': len(files),
                'successful_count': successful_count,
                'failed_count': failed_count,
                'continue_on_error': continue_on_error,
                'results': results
            }
        )
        
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"批量添加文档失败: {str(e)}"
        )


async def _batch_process_documents(tasks: List[Dict], collection_name: str, continue_on_error: bool) -> List[Dict[str, Any]]:
    """
    批量处理文档任务
    
    Args:
        tasks: 处理任务列表
        collection_name: 集合名称
        continue_on_error: 遇到错误时是否继续
        
    Returns:
        处理结果列表
    """
    results = []
    
    async def process_single_task(task):
        """处理单个任务"""
        try:
            file = task['file']
            doc_type = task['doc_type']
            metadata = task['metadata']
            index = task['index']
            
            # 文件验证
            validation_result = _validate_file_for_type(file, doc_type)
            if not validation_result['valid']:
                return {
                    'index': index,
                    'filename': file.filename,
                    'success': False,
                    'error': f"文件验证失败: {validation_result['error']}",
                    'doc_type': doc_type
                }
            
            # 根据文档类型处理
            if doc_type == "image":
                result = await _process_image_document(file, collection_name, metadata)
            elif doc_type == "md":
                result = await _process_markdown_document(file, collection_name, metadata)
            elif doc_type == "pdf":
                result = await _process_pdf_document(file, collection_name, metadata)
            else:
                result = {
                    'success': False,
                    'error': f"不支持的文档类型: {doc_type}"
                }
            
            return {
                'index': index,
                'filename': file.filename,
                'success': result['success'],
                'error': result.get('error'),
                'details': result.get('details', {}),
                'doc_type': doc_type
            }
            
        except Exception as e:
            return {
                'index': task['index'],
                'filename': task['file'].filename,
                'success': False,
                'error': f"处理异常: {str(e)}",
                'doc_type': task.get('doc_type', 'unknown')
            }
    
    # 并行处理（限制并发数）
    semaphore = asyncio.Semaphore(5)  # 最多5个并发任务
    
    async def process_with_semaphore(task):
        async with semaphore:
            return await process_single_task(task)
    
    # 如果设置了continue_on_error，使用gather返回所有结果
    if continue_on_error:
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'index': i,
                    'filename': tasks[i]['file'].filename if i < len(tasks) else 'unknown',
                    'success': False,
                    'error': f"处理异常: {str(result)}",
                    'doc_type': tasks[i].get('doc_type', 'unknown') if i < len(tasks) else 'unknown'
                })
            else:
                processed_results.append(result)
        
        return processed_results
    else:
        # 顺序处理，遇到错误立即停止
        for task in tasks:
            result = await process_single_task(task)
            results.append(result)
            if not result['success']:
                break
        
        return results


@router.delete("/{collection_name}/{document_key}", response_model=StandardResponse)
async def delete_document_from_collection(collection_name: str, document_key: str):
    """
    从集合删除文档（按文档键删除所有相关块）
    
    Args:
        collection_name: 集合名称
        document_key: 文档键
        
    Returns:
        删除操作结果
    """
    try:
        from .utils import _create_milvus_manager_safely
        
        # 使用安全创建的管理器（减少重复代码）
        manager = _create_milvus_manager_safely(collection_name)
        if not manager:
            return _create_standard_response(
                success=False,
                error=f"无法连接到集合 {collection_name}"
            )
        
        # 使用新的file字段进行删除
        filter_expr = f'file == "{document_key}"'
        
        # 查找所有相关的文档块
        related_docs = manager.search_with_filter(filter_expr, limit=1000)
        
        if not related_docs:
            return _create_standard_response(
                success=True,
                message=f"文档 {document_key} 不存在或已被删除"
            )
        
        # 获取所有文档块的ID
        doc_ids = [doc.get('id') for doc in related_docs if doc.get('id') is not None]
        
        if doc_ids:
            # 执行软删除
            success = manager.mark_deleted(doc_ids)
            
            if success:
                return _create_standard_response(
                    success=True,
                    message=f"文档 {document_key} 已从集合 {collection_name} 删除（共删除 {len(doc_ids)} 个块）"
                )
            else:
                return _create_standard_response(
                    success=False,
                    error=f"文档 {document_key} 从集合 {collection_name} 删除失败"
                )
        else:
            return _create_standard_response(
                success=True,
                message=f"文档 {document_key} 没有找到可删除的块"
            )
            
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"删除文档失败: {str(e)}"
        )


@router.put("/{collection_name}/{document_name}", response_model=StandardResponse)
async def update_document_in_collection(
    collection_name: str, 
    document_name: str,
    file: UploadFile = File(...),
    doc_type: str = Form("md", description="文档类型: md, image, pdf")
):
    """更新集合中的文档"""
    try:
        from ...core.indexing import update_single_document
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc_type}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 更新文档
            success = update_single_document(tmp_file_path, collection_name)
            
            if success:
                return _create_standard_response(
                    success=True,
                    message=f"文档 {document_name} 在集合 {collection_name} 中更新成功",
                    data={
                        'collection': collection_name,
                        'document': document_name,
                        'type': doc_type
                    }
                )
            else:
                return _create_standard_response(
                    success=False,
                    error=f"文档 {document_name} 在集合 {collection_name} 中更新失败"
                )
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"更新文档失败: {str(e)}"
        )
