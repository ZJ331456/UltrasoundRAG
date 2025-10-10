"""
UltrasoundRAG API搜索检索模块
提供多模态检索接口：t2t, t2i, i2t, i2i, auto, multimodal
"""

import os
import time
import tempfile
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Form, File, UploadFile

from .models import SearchRequest, MultiDatabaseSearchRequest, CollectionSearchRequest, SearchResponse
from .utils import _is_safe_file_path

# 创建路由器
router = APIRouter(prefix="/search", tags=["search"])

# 全局变量，用于检查模块是否可用
try:
    from ...main import get_service
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
    def get_service():
        return None


def _validate_search_request(req: SearchRequest) -> Optional[str]:
    """
    验证搜索请求参数
    
    Args:
        req: 搜索请求
        
    Returns:
        错误消息（如果验证失败）或None（验证通过）
    """
    # 基础模式验证
    valid_modes = {"t2t", "t2i", "i2t", "i2i", "auto", "multimodal"}
    if req.mode not in valid_modes:
        return f"不支持的检索模式: {req.mode}，支持的模式: {', '.join(valid_modes)}"
    
    # 输入参数验证
    if req.mode in {"t2t", "t2i"} and not req.query:
        return f"模式 {req.mode} 需要提供文本查询参数"
    
    if req.mode in {"i2t", "i2i"} and not req.image_path:
        return f"模式 {req.mode} 需要提供图片路径参数"
    
    # auto和multimodal模式需要至少一个输入
    if req.mode in {"auto", "multimodal"} and not req.query and not req.image_path:
        return f"模式 {req.mode} 需要提供文本查询或图片路径参数"
    
    # 文本查询长度限制
    if req.query and len(req.query) > 1000:
        return "查询文本长度不能超过1000个字符"
    
    # 文件路径安全检查
    if req.image_path:
        if not _is_safe_file_path(req.image_path):
            return "图片路径包含不安全字符"
    
    return None


@router.post("", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    统一检索端点（增强版）
    
    支持多种检索模式的统一入口：
    - t2t: 文本到文本检索
    - t2i: 文本到图像检索  
    - i2t: 图像到文本检索
    - i2i: 图像到图像检索
    - auto: 智能自动选择模式
    - multimodal: 多模态融合检索
    
    Args:
        req: 搜索请求参数
        
    Returns:
        检索结果，包含相关文档/图像和元数据
        
    Raises:
        HTTPException: 当服务不可用或参数验证失败时
    """
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="检索服务不可用")
    
    start_time = time.time()
    
    try:
        # 增强的输入验证
        validation_error = _validate_search_request(req)
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)
        
        # 执行搜索
        result = service.search(
            query=req.query,
            image_path=req.image_path,
            mode=req.mode,
            db_name=req.db_name,
            top_k=req.top_k
        )
        
        response_time = time.time() - start_time
        
        # 增强的响应元数据
        metadata = {
            "mode": req.mode,
            "db_name": req.db_name,
            "top_k": req.top_k,
            "response_time": response_time,
            "cache_enabled": req.enable_cache,
            "timeout": req.timeout,
            "actual_results": len(result.get('results', {}).get('results', [])) if result.get('results') else 0
        }
        
        return SearchResponse(
            success=result['success'],
            data=result.get('results'),
            metadata=metadata,
            error=result.get('error')
        )
        
    except HTTPException:
        raise  # 重新抛出HTTP异常
    except Exception as e:
        # 记录详细错误信息用于调试
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "mode": req.mode,
            "db_name": req.db_name,
            "response_time": time.time() - start_time
        }
        
        return SearchResponse(
            success=False,
            error=f"检索失败: {str(e)}",
            metadata=error_details
        )


@router.post("/multi-database", response_model=SearchResponse)
async def multi_database_search(req: MultiDatabaseSearchRequest):
    """多数据库搜索端点"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="检索服务不可用")
    
    start_time = time.time()
    
    try:
        result = service.multi_database_search(
            query=req.query,
            image_path=req.image_path,
            mode=req.mode,
            databases=req.databases,
            top_k=req.top_k
        )
        
        response_time = time.time() - start_time
        
        return SearchResponse(
            success=result['success'],
            data=result.get('data'),
            metadata={
                "mode": req.mode,
                "databases": req.databases,
                "top_k": req.top_k,
                "response_time": response_time
            },
            error=result.get('error')
        )
        
    except Exception as e:
        return SearchResponse(
            success=False,
            error=str(e),
            metadata={"response_time": time.time() - start_time}
        )


@router.post("/upload", response_model=SearchResponse)
async def search_with_upload(
    file: UploadFile = File(...),
    mode: str = Form("i2t"),
    db_name: str = Form("default"),
    top_k: int = Form(10),
    query: Optional[str] = Form(None)
):
    """带文件上传的搜索端点"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="检索服务不可用")
    
    # 验证文件类型
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    start_time = time.time()
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 执行搜索
            result = service.search(
                query=query,
                image_path=tmp_file_path,
                mode=mode,
                db_name=db_name,
                top_k=top_k
            )
            
            response_time = time.time() - start_time
            
            return SearchResponse(
                success=result['success'],
                data=result.get('results'),
                metadata={
                    "mode": mode,
                    "db_name": db_name,
                    "top_k": top_k,
                    "filename": file.filename,
                    "response_time": response_time
                },
                error=result.get('error')
            )
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except Exception as e:
        return SearchResponse(
            success=False,
            error=str(e),
            metadata={"response_time": time.time() - start_time}
        )


@router.post("/collections", response_model=SearchResponse)
async def collection_search(req: CollectionSearchRequest):
    """基于集合的搜索端点（新版本推荐）"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="检索服务不可用")
    
    start_time = time.time()
    
    try:
        # 验证集合配置
        validation_error = req.validate_collections()
        if validation_error:
            raise HTTPException(status_code=400, detail=validation_error)
        
        # 执行基于集合的搜索
        result = service.collection_search(
            text_collections=req.text_collections,
            image_collections=req.image_collections,
            query=req.query,
            image_path=req.image_path,
            mode=req.mode,
            top_k=req.top_k
        )
        
        response_time = time.time() - start_time
        
        return SearchResponse(
            success=result['success'],
            data=result.get('data'),
            metadata={
                "mode": req.mode,
                "text_collections": req.text_collections,
                "image_collections": req.image_collections,
                "top_k": req.top_k,
                "response_time": response_time,
                "cache_enabled": req.enable_cache,
                "timeout": req.timeout
            },
            error=result.get('error')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return SearchResponse(
            success=False,
            error=str(e),
            metadata={"response_time": time.time() - start_time}
        )


@router.get("/collections/info")
async def get_collections_info():
    """获取所有集合的详细信息，包括类型"""
    try:
        service = get_service()
        if not service:
            raise HTTPException(status_code=503, detail="检索服务不可用")
        
        # 获取集合信息
        from ...core.indexing import list_collections
        collection_names = list_collections()
        
        collections_info = []
        if hasattr(service, '_multi_db_manager') and service._multi_db_manager:
            for collection_name in collection_names:
                collection_type = service._multi_db_manager._detect_collection_type(collection_name)
                collections_info.append({
                    'name': collection_name,
                    'type': collection_type,
                    'suitable_for': ['t2t', 'i2t'] if collection_type == 'text' else ['t2i', 'i2i']
                })
        else:
            # 如果管理器不存在，根据名称推断
            for collection_name in collection_names:
                if 'image' in collection_name.lower():
                    collection_type = 'image'
                    suitable_for = ['t2i', 'i2i']
                else:
                    collection_type = 'text'
                    suitable_for = ['t2t', 'i2t']
                
                collections_info.append({
                    'name': collection_name,
                    'type': collection_type,
                    'suitable_for': suitable_for
                })
        
        return {
            'success': True,
            'data': {
                'collections': collections_info,
                'total_count': len(collections_info),
                'text_collections': [c['name'] for c in collections_info if c['type'] == 'text'],
                'image_collections': [c['name'] for c in collections_info if c['type'] == 'image']
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@router.post("/collections/upload", response_model=SearchResponse)
async def collection_search_with_upload(
    file: UploadFile = File(...),
    mode: str = Form("i2t"),
    text_collections: List[str] = Form(default_factory=list),
    image_collections: List[str] = Form(default_factory=list),
    top_k: int = Form(10),
    query: Optional[str] = Form(None)
):
    """基于集合的文件上传搜索端点"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="检索服务不可用")
    
    # 验证文件类型
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    start_time = time.time()
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 执行基于集合的搜索
            result = service.collection_search(
                text_collections=text_collections,
                image_collections=image_collections,
                query=query,
                image_path=tmp_file_path,
                mode=mode,
                top_k=top_k
            )
            
            response_time = time.time() - start_time
            
            return SearchResponse(
                success=result["success"],
                data=result.get("data"),
                metadata={
                    "mode": mode,
                    "text_collections": text_collections,
                    "image_collections": image_collections,
                    "top_k": top_k,
                    "filename": file.filename,
                    "response_time": response_time
                },
                error=result.get("error")
            )
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except Exception as e:
        return SearchResponse(
            success=False,
            error=str(e),
            metadata={"response_time": time.time() - start_time}
        )


@router.post("/upload-multi", response_model=SearchResponse)
async def multi_collection_search_with_upload(
    file: UploadFile = File(...),
    mode: str = Form("i2t"),
    databases: List[str] = Form(...),
    top_k: int = Form(10),
    query: Optional[str] = Form(None)
):
    """多集合文件上传搜索端点（向后兼容）"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="检索服务不可用")
    
    # 验证文件类型
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    start_time = time.time()
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 执行多集合搜索
            result = service.multi_database_search(
                query=query,
                image_path=tmp_file_path,
                mode=mode,
                databases=databases,
                top_k=top_k
            )
            
            response_time = time.time() - start_time
            
            return SearchResponse(
                success=result["success"],
                data=result.get("data"),
                metadata={
                    "mode": mode,
                    "databases": databases,
                    "top_k": top_k,
                    "filename": file.filename,
                    "response_time": response_time
                },
                error=result.get("error")
            )
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except Exception as e:
        return SearchResponse(
            success=False,
            error=str(e),
            metadata={"response_time": time.time() - start_time}
        )
