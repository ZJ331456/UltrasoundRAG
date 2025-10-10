"""
UltrasoundRAG API文件管理模块
提供增强的文件管理功能，特别是图片文件的处理
"""

import os
import json
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from .models import StandardResponse
from .utils import _create_standard_response, _is_safe_file_path

# 创建路由器
router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload-image-folder", response_model=StandardResponse)
async def upload_image_folder(
    collection_name: str = Form(...),
    folder_name: str = Form(..., description="文件夹名称"),
    files: List[UploadFile] = File(...),
    annotations: Optional[str] = Form(None, description="图片注释JSON文件内容"),
    create_thumbnails: bool = Form(False, description="是否创建缩略图"),
    auto_organize: bool = Form(True, description="是否自动按类型组织文件")
):
    """
    上传图片文件夹（增强版图片管理）
    
    专门用于处理图片数据集上传，支持：
    1. 批量图片上传
    2. JSON注释文件处理
    3. 自动缩略图生成
    4. 智能文件组织
    5. 图片元数据提取
    
    Args:
        collection_name: 目标集合名称
        folder_name: 文件夹名称（用于组织文件）
        files: 上传的文件列表（图片+可选JSON文件）
        annotations: 图片注释JSON内容
        create_thumbnails: 是否创建缩略图
        auto_organize: 是否自动组织文件结构
    """
    try:
        # 1. 验证输入
        if len(files) > 1000:  # 限制文件数量
            return _create_standard_response(
                success=False,
                error="单次上传文件数量不能超过1000个"
            )
        
        # 2. 分离图片文件和注释文件
        image_files = []
        annotation_data = {}
        
        for file in files:
            if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                image_files.append(file)
            elif file.filename.lower().endswith('.json'):
                # 读取JSON注释文件
                json_content = await file.read()
                try:
                    annotation_data.update(json.loads(json_content.decode('utf-8')))
                except json.JSONDecodeError as e:
                    return _create_standard_response(
                        success=False,
                        error=f"JSON注释文件格式错误: {str(e)}"
                    )
        
        # 3. 处理外部注释数据
        if annotations:
            try:
                external_annotations = json.loads(annotations)
                annotation_data.update(external_annotations)
            except json.JSONDecodeError as e:
                return _create_standard_response(
                    success=False,
                    error=f"注释数据JSON格式错误: {str(e)}"
                )
        
        # 4. 处理图片文件
        processing_results = await _process_image_folder(
            image_files, annotation_data, collection_name, folder_name,
            create_thumbnails, auto_organize
        )
        
        # 5. 统计结果
        successful_images = sum(1 for r in processing_results if r['success'])
        failed_images = len(processing_results) - successful_images
        
        return _create_standard_response(
            success=failed_images == 0,
            message=f"图片文件夹上传完成：成功 {successful_images}/{len(image_files)} 张图片",
            data={
                'collection_name': collection_name,
                'folder_name': folder_name,
                'total_images': len(image_files),
                'successful_images': successful_images,
                'failed_images': failed_images,
                'annotations_count': len(annotation_data),
                'thumbnails_created': create_thumbnails,
                'auto_organized': auto_organize,
                'processing_results': processing_results
            }
        )
        
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"图片文件夹上传失败: {str(e)}"
        )


async def _process_image_folder(
    image_files: List[UploadFile], 
    annotations: Dict[str, Any],
    collection_name: str,
    folder_name: str,
    create_thumbnails: bool,
    auto_organize: bool
) -> List[Dict[str, Any]]:
    """
    处理图片文件夹上传
    
    Args:
        image_files: 图片文件列表
        annotations: 注释数据
        collection_name: 集合名称
        folder_name: 文件夹名称
        create_thumbnails: 是否创建缩略图
        auto_organize: 是否自动组织
        
    Returns:
        处理结果列表
    """
    from PIL import Image
    from ...core.indexing import add_data_to_collection
    
    results = []
    
    # 创建临时工作目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_folder = Path(temp_dir) / folder_name
        temp_folder.mkdir(exist_ok=True)
        
        for file in image_files:
            try:
                # 保存图片文件
                file_path = temp_folder / file.filename
                content = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # 验证图片
                try:
                    with Image.open(file_path) as img:
                        image_info = {
                            'width': img.width,
                            'height': img.height,
                            'format': img.format,
                            'mode': img.mode
                        }
                except Exception as img_error:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': f"图片验证失败: {str(img_error)}"
                    })
                    continue
                
                # 创建缩略图（如果需要）
                thumbnail_path = None
                if create_thumbnails:
                    try:
                        thumbnail_path = _create_thumbnail(file_path, temp_folder)
                    except Exception as thumb_error:
                        # 缩略图创建失败不影响主流程
                        pass
                
                # 获取注释信息
                annotation = annotations.get(file.filename, {})
                
                # 构建元数据
                metadata = {
                    'folder_name': folder_name,
                    'image_info': image_info,
                    'annotation': annotation,
                    'thumbnail_created': thumbnail_path is not None,
                    'auto_organized': auto_organize
                }
                
                # 添加到集合
                success = add_data_to_collection(str(file_path), collection_name, "image")
                
                results.append({
                    'filename': file.filename,
                    'success': success,
                    'metadata': metadata,
                    'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
                    'error': None if success else '添加到集合失败'
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': f"处理失败: {str(e)}"
                })
    
    return results


def _create_thumbnail(image_path, output_dir, size: tuple = (128, 128)):
    """
    创建图片缩略图
    
    Args:
        image_path: 原图片路径
        output_dir: 输出目录
        size: 缩略图尺寸
        
    Returns:
        缩略图路径或None
    """
    try:
        from PIL import Image
        
        if isinstance(image_path, str):
            image_path = Path(image_path)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
            
        thumbnail_name = f"thumb_{image_path.name}"
        thumbnail_path = output_dir / thumbnail_name
        
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumbnail_path, "JPEG", quality=85)
        
        return thumbnail_path
        
    except Exception:
        return None


@router.get("/image/{image_hash}")
async def get_image_file(image_hash: str, size: str = "original"):
    """
    获取图片文件（增强版图片访问）
    
    支持多种尺寸和格式的图片访问：
    - original: 原始图片
    - thumbnail: 缩略图
    - medium: 中等尺寸
    - small: 小尺寸
    
    Args:
        image_hash: 图片哈希值
        size: 图片尺寸 ("original", "thumbnail", "medium", "small")
    """
    try:
        # 查找图片文件
        image_path = _find_image_by_hash(image_hash, size)
        
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"图片 {image_hash} ({size}) 不存在")
        
        # 检查文件安全性
        if not _is_safe_file_path(image_path):
            raise HTTPException(status_code=403, detail="文件路径不安全")
        
        # 返回图片文件
        return FileResponse(
            image_path,
            media_type=f"image/{_get_image_format(image_path)}",
            headers={
                "Cache-Control": "public, max-age=86400",  # 24小时缓存
                "ETag": f'"{image_hash}_{size}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片失败: {str(e)}")


def _find_image_by_hash(image_hash: str, size: str) -> Optional[str]:
    """
    根据哈希值查找图片文件
    
    Args:
        image_hash: 图片哈希值
        size: 图片尺寸
        
    Returns:
        图片文件路径或None
    """
    # 可能的图片目录
    possible_dirs = [
        "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image",
        "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/images",
        "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/ultrasoundrag/data/images"
    ]
    
    # 可能的文件扩展名
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    # 根据尺寸确定文件名前缀
    size_prefixes = {
        'original': '',
        'thumbnail': 'thumb_',
        'medium': 'med_',
        'small': 'small_'
    }
    
    prefix = size_prefixes.get(size, '')
    filename_base = f"{prefix}{image_hash}"
    
    # 搜索文件
    for base_dir in possible_dirs:
        if os.path.exists(base_dir):
            for ext in extensions:
                file_path = os.path.join(base_dir, f"{filename_base}{ext}")
                if os.path.isfile(file_path):
                    return file_path
    
    return None


def _get_image_format(file_path: str) -> str:
    """
    获取图片格式
    
    Args:
        file_path: 图片文件路径
        
    Returns:
        图片格式字符串
    """
    ext = os.path.splitext(file_path)[1].lower()
    format_map = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.png': 'png',
        '.bmp': 'bmp',
        '.tiff': 'tiff',
        '.tif': 'tiff',
        '.gif': 'gif',
        '.webp': 'webp'
    }
    return format_map.get(ext, 'jpeg')


@router.get("/storage-info", response_model=StandardResponse)
async def get_storage_info():
    """
    获取文件存储信息
    
    Returns:
        存储统计信息和配置
    """
    try:
        storage_info = _analyze_storage_usage()
        
        return _create_standard_response(
            success=True,
            message="存储信息获取成功",
            data=storage_info
        )
        
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取存储信息失败: {str(e)}"
        )


def _analyze_storage_usage() -> Dict[str, Any]:
    """
    分析存储使用情况
    
    Returns:
        存储分析结果
    """
    storage_info = {
        'image_directories': [],
        'total_images': 0,
        'total_size_mb': 0,
        'available_thumbnails': 0,
        'storage_paths': []
    }
    
    # 检查图片目录
    possible_dirs = [
        "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image",
        "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/images",
        "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/ultrasoundrag/data/images"
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            try:
                dir_info = _analyze_directory(dir_path)
                storage_info['image_directories'].append(dir_info)
                storage_info['total_images'] += dir_info['image_count']
                storage_info['total_size_mb'] += dir_info['size_mb']
                storage_info['available_thumbnails'] += dir_info['thumbnail_count']
                storage_info['storage_paths'].append(dir_path)
            except Exception as e:
                # 如果某个目录分析失败，记录但继续
                storage_info['image_directories'].append({
                    'path': dir_path,
                    'error': str(e),
                    'accessible': False
                })
    
    return storage_info


def _analyze_directory(dir_path: str) -> Dict[str, Any]:
    """
    分析目录存储情况
    
    Args:
        dir_path: 目录路径
        
    Returns:
        目录分析结果
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    image_count = 0
    thumbnail_count = 0
    total_size = 0
    
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in image_extensions:
                    image_count += 1
                    
                    # 检查是否为缩略图
                    if file.startswith('thumb_'):
                        thumbnail_count += 1
                    
                    # 计算文件大小
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass  # 文件可能已删除或无法访问
        
        return {
            'path': dir_path,
            'image_count': image_count,
            'thumbnail_count': thumbnail_count,
            'size_mb': round(total_size / (1024 * 1024), 2),
            'accessible': True
        }
        
    except Exception as e:
        return {
            'path': dir_path,
            'error': str(e),
            'accessible': False
        }
