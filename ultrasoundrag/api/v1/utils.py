"""
UltrasoundRAG API工具函数
提供通用的工具函数和辅助方法
"""

import os
import time
import tempfile
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

from .models import StandardResponse


# ==================== 通用工具函数 ====================

def _serialize_rr(rr) -> Dict[str, Any]:
    """序列化检索结果"""
    try:
        return {
            'doc_id': getattr(rr, 'doc_id', None),
            'score': getattr(rr, 'score', None),
            'content': getattr(rr, 'content', None),
            'metadata': getattr(rr, 'metadata', {}),
            'resource_collection': getattr(rr, 'resource_collection', ''),
            'retrieval_type': getattr(rr, 'retrieval_type', ''),
        }
    except Exception:
        return {'raw': str(rr)}


def _create_standard_response(success: bool, message: Optional[str] = None, 
                            data: Optional[Dict[str, Any]] = None, 
                            error: Optional[str] = None) -> StandardResponse:
    """
    创建标准API响应（统一响应格式，减少重复代码）
    
    Args:
        success: 是否成功
        message: 成功消息
        data: 响应数据
        error: 错误消息
        
    Returns:
        标准响应对象
    """
    return StandardResponse(
        success=success,
        message=message,
        data=data,
        error=error,
        timestamp=time.time()
    )


def _handle_collection_operation_error(operation: str, collection_name: str, error: Exception) -> StandardResponse:
    """
    统一处理集合操作错误（减少重复的错误处理代码）
    
    Args:
        operation: 操作名称
        collection_name: 集合名称  
        error: 异常对象
        
    Returns:
        错误响应
    """
    error_message = f"{operation}集合 {collection_name} 失败: {str(error)}"
    
    # 根据异常类型提供更详细的错误信息
    if "connection" in str(error).lower():
        error_message += " (数据库连接异常)"
    elif "permission" in str(error).lower():
        error_message += " (权限不足)"
    elif "not found" in str(error).lower():
        error_message += " (集合不存在)"
    
    return _create_standard_response(
        success=False,
        error=error_message
    )


def _infer_collection_type(collection_name: str) -> str:
    """
    从集合名称推断集合类型
    
    基于命名约定自动识别集合类型，支持灵活的命名方式
    
    Args:
        collection_name: 集合名称
        
    Returns:
        推断的集合类型 ("md", "image", "pdf")
    """
    if not collection_name:
        return 'md'
    
    name_lower = collection_name.lower()
    
    # 图像类型标识符
    image_indicators = ['image', 'img', 'pic', 'photo', 'picture']
    if any(indicator in name_lower for indicator in image_indicators):
        return 'image'
    
    # PDF类型标识符
    pdf_indicators = ['pdf', 'doc', 'paper', 'thesis']
    if any(indicator in name_lower for indicator in pdf_indicators):
        return 'pdf'
    
    # 默认为markdown类型
    return 'md'


def _create_milvus_manager_safely(collection_name: str, collection_type: Optional[str] = None) -> Optional[Any]:
    """
    安全创建Milvus管理器（统一错误处理）
    
    Args:
        collection_name: 集合名称
        collection_type: 集合类型（可选，自动推断）
        
    Returns:
        MilvusManager实例或None（如果创建失败）
    """
    try:
        from ...data.stores.milvus_store import MilvusManager
        
        if not collection_type:
            collection_type = _infer_collection_type(collection_name)
        
        return MilvusManager(
            collection_type=collection_type, 
            collection_name=collection_name
        )
    except Exception as e:
        # 记录错误但不抛出异常，让调用者决定如何处理
        print(f"创建MilvusManager失败: {e}")
        return None


def _find_dataset_name_by_collection(collection_name: str, collection_type: str) -> Optional[str]:
    """根据集合名称找到对应的数据集名称"""
    try:
        from ...config import config
        
        # 获取对应类型的数据集配置
        dataset_type = 'markdown' if collection_type == 'md' else collection_type
        datasets_config = config['indexing'][dataset_type].get('datasets', {})
        
        # 遍历数据集配置，找到collection_name匹配的数据集
        for dataset_name, dataset_cfg in datasets_config.items():
            if dataset_cfg.get('collection_name') == collection_name:
                return dataset_name
        
        return None
        
    except Exception as e:
        print(f"查找数据集名称失败: {e}")
        return None


# ==================== 文件处理工具函数 ====================

def _detect_document_type(filename: str, content_type: Optional[str] = None) -> Optional[str]:
    """
    自动检测文档类型
    
    Args:
        filename: 文件名
        content_type: MIME类型
        
    Returns:
        检测到的文档类型 ("md", "image", "pdf") 或 None
    """
    if not filename:
        return None
    
    filename_lower = filename.lower()
    
    # 基于文件扩展名检测
    if filename_lower.endswith(('.md', '.markdown', '.txt')):
        return "md"
    elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp')):
        return "image"
    elif filename_lower.endswith('.pdf'):
        return "pdf"
    
    # 基于MIME类型检测
    if content_type:
        if content_type.startswith('image/'):
            return "image"
        elif content_type == 'application/pdf':
            return "pdf"
        elif content_type.startswith('text/'):
            return "md"
    
    return None


def _validate_file_for_type(file, doc_type: str) -> Dict[str, Any]:
    """
    验证文件是否符合指定类型的要求
    
    Args:
        file: 上传的文件
        doc_type: 文档类型
        
    Returns:
        验证结果字典
    """
    result = {"valid": True, "error": None}
    
    try:
        # 文件大小限制
        max_sizes = {
            "md": 50 * 1024 * 1024,    # 50MB
            "image": 20 * 1024 * 1024,  # 20MB  
            "pdf": 100 * 1024 * 1024    # 100MB
        }
        
        if hasattr(file, 'size') and file.size:
            if file.size > max_sizes.get(doc_type, 50 * 1024 * 1024):
                result["valid"] = False
                result["error"] = f"{doc_type}文件大小超过限制（{max_sizes.get(doc_type, 50)//1024//1024}MB）"
                return result
        
        # 文件扩展名验证
        allowed_extensions = {
            "md": {'.md', '.markdown', '.txt'},
            "image": {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'},
            "pdf": {'.pdf'}
        }
        
        if file.filename:
            file_ext = os.path.splitext(file.filename.lower())[1]
            if file_ext not in allowed_extensions.get(doc_type, set()):
                result["valid"] = False
                result["error"] = f"{doc_type}类型不支持{file_ext}扩展名"
                return result
        
        # MIME类型验证
        expected_mime_prefixes = {
            "md": ['text/', 'application/octet-stream'],  # 添加对默认MIME类型的支持
            "image": ['image/'],
            "pdf": ['application/pdf']
        }
        
        if file.content_type:
            mime_prefixes = expected_mime_prefixes.get(doc_type, [])
            if mime_prefixes and not any(file.content_type.startswith(prefix) for prefix in mime_prefixes):
                result["valid"] = False
                result["error"] = f"{doc_type}类型MIME类型不匹配：{file.content_type}"
                return result
        
        return result
        
    except Exception as e:
        result["valid"] = False
        result["error"] = f"文件验证异常: {str(e)}"
        return result


def _is_safe_file_path(file_path: str) -> bool:
    """
    检查文件路径是否安全
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否安全
    """
    # 基础安全检查
    dangerous_patterns = ['../', '..\\', '/etc/', '/proc/', 'c:\\windows\\']
    file_path_lower = file_path.lower()
    
    return not any(pattern in file_path_lower for pattern in dangerous_patterns)


# ==================== 文档处理函数 ====================

async def _process_markdown_document(file, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理Markdown文档
    
    Args:
        file: 上传的文件
        collection_name: 集合名称
        metadata: 元数据
        
    Returns:
        处理结果
    """
    try:
        from ...core.indexing import add_data_to_collection
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.md') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 添加到集合，传递原始文件名
            success = add_data_to_collection(tmp_file_path, collection_name, "md", original_filename=file.filename)
            
            return {
                'success': success,
                'details': {
                    'type': 'markdown',
                    'size': len(content),
                    'temp_path': tmp_file_path,
                    'metadata': metadata
                },
                'error': None if success else '添加Markdown文档到集合失败'
            }
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        return {
            'success': False,
            'details': {},
            'error': f"Markdown文档处理失败: {str(e)}"
        }


async def _process_image_document(file, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理图像文档
    
    图像处理需要特殊处理：
    1. 验证图像文件完整性
    2. 生成缩略图（可选）
    3. 提取图像元数据
    4. 保存到指定目录结构
    
    Args:
        file: 上传的文件
        collection_name: 集合名称
        metadata: 元数据
        
    Returns:
        处理结果
    """
    try:
        from PIL import Image
        from ...core.indexing import add_data_to_collection
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 验证图像文件
            with Image.open(tmp_file_path) as img:
                image_info = {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode
                }
            
            # 添加到集合，传递原始文件名
            success = add_data_to_collection(tmp_file_path, collection_name, "image", original_filename=file.filename)
            
            return {
                'success': success,
                'details': {
                    'type': 'image',
                    'size': len(content),
                    'image_info': image_info,
                    'temp_path': tmp_file_path,
                    'metadata': metadata
                },
                'error': None if success else '添加图像文档到集合失败'
            }
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        return {
            'success': False,
            'details': {},
            'error': f"图像文档处理失败: {str(e)}"
        }


async def _process_pdf_document(file, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理PDF文档
    
    Args:
        file: 上传的文件
        collection_name: 集合名称
        metadata: 元数据
        
    Returns:
        处理结果
    """
    try:
        from ...core.indexing import add_data_to_collection
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 添加到集合，传递原始文件名
            success = add_data_to_collection(tmp_file_path, collection_name, "pdf", original_filename=file.filename)
            
            return {
                'success': success,
                'details': {
                    'type': 'pdf',
                    'size': len(content),
                    'temp_path': tmp_file_path,
                    'metadata': metadata
                },
                'error': None if success else '添加PDF文档到集合失败'
            }
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except Exception as e:
        return {
            'success': False,
            'details': {},
            'error': f"PDF文档处理失败: {str(e)}"
        }
