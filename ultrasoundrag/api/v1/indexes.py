"""
UltrasoundRAG API索引管理模块
提供索引构建、重建等管理接口
"""

import threading
from fastapi import APIRouter

from .models import IndexBuildRequest, TaskResponse, TaskStatus, StandardResponse
from .tasks import task_manager
from .utils import _find_dataset_name_by_collection, _infer_collection_type, _create_standard_response

# 创建路由器
router = APIRouter(prefix="/indexes", tags=["indexes"])

# 全局变量，用于检查模块是否可用
try:
    from ...main import get_service
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
    def get_service():
        return None


def _run_build_indexes_task(task_id: str, target: str, recreate: bool):
    """后台运行索引构建任务"""
    try:
        # 更新任务状态为运行中
        task_manager.update_task(
            task_id, 
            status=TaskStatus.RUNNING,
            progress=10.0,
            message="开始构建索引..."
        )
        
        service = get_service()
        if not service:
            raise Exception("服务不可用")
        
        # 模拟进度更新
        task_manager.update_task(
            task_id,
            progress=30.0,
            message="正在初始化索引构建器..."
        )
        
        # 执行实际的索引构建
        result = service.build_indexes(target, recreate)
        
        if result.get('success'):
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100.0,
                message="索引构建完成",
                result=result.get('result')
            )
        else:
            raise Exception(result.get('error', '索引构建失败'))
            
    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=f"索引构建失败: {str(e)}",
            error=str(e)
        )


def _run_rebuild_collection_task(task_id: str, collection_name: str):
    """后台运行集合重建任务"""
    try:
        # 更新任务状态为运行中
        task_manager.update_task(
            task_id, 
            status=TaskStatus.RUNNING,
            progress=20.0,
            message=f"开始重建集合 {collection_name}..."
        )
        
        from ...core.indexing import delete_collection, build_markdown_index, build_image_index
        
        # 删除现有集合
        task_manager.update_task(
            task_id,
            progress=40.0,
            message=f"正在删除集合 {collection_name}..."
        )
        
        delete_success = delete_collection(collection_name)
        if not delete_success:
            raise Exception(f"删除集合 {collection_name} 失败")
        
        # 重建指定集合的索引
        task_manager.update_task(
            task_id,
            progress=60.0,
            message=f"正在重建集合 {collection_name} 的索引..."
        )
        
        # 根据集合类型只重建对应的索引
        collection_type = _infer_collection_type(collection_name)
        
        # 根据集合名称找到对应的数据集名称
        dataset_name = _find_dataset_name_by_collection(collection_name, collection_type)
        
        if collection_type == 'image':
            build_result = build_image_index(recreate=True, only_datasets=[dataset_name] if dataset_name else None)
        else:
            build_result = build_markdown_index(recreate=True, only_datasets=[dataset_name] if dataset_name else None)
        
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            message=f"集合 {collection_name} 重建完成",
            result={'build_result': build_result}
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message=f"重建集合失败: {str(e)}",
            error=str(e)
        )


@router.post("/build", response_model=TaskResponse)
async def build_indexes_async(request: IndexBuildRequest):
    """异步构建索引"""
    try:
        # 创建异步任务
        task_name = f"构建索引 - {request.target}"
        if request.recreate:
            task_name += " (重建)"
        
        task_id = task_manager.create_task(task_name)
        
        # 在后台线程中执行任务
        thread = threading.Thread(
            target=_run_build_indexes_task,
            args=(task_id, request.target, request.recreate)
        )
        thread.daemon = True
        thread.start()
        
        return TaskResponse(
            success=True,
            task_id=task_id,
            message=f"索引构建任务已启动，任务ID: {task_id}"
        )
        
    except Exception as e:
        return TaskResponse(
            success=False,
            task_id="",
            message=f"启动索引构建任务失败: {str(e)}"
        )


@router.post("/rebuild/{collection_name}", response_model=TaskResponse)
async def rebuild_collection_async(collection_name: str):
    """异步重建集合索引"""
    try:
        # 创建异步任务
        task_name = f"重建集合 - {collection_name}"
        task_id = task_manager.create_task(task_name)
        
        # 在后台线程中执行任务
        thread = threading.Thread(
            target=_run_rebuild_collection_task,
            args=(task_id, collection_name)
        )
        thread.daemon = True
        thread.start()
        
        return TaskResponse(
            success=True,
            task_id=task_id,
            message=f"集合 {collection_name} 重建任务已启动，任务ID: {task_id}"
        )
        
    except Exception as e:
        return TaskResponse(
            success=False,
            task_id="",
            message=f"启动集合重建任务失败: {str(e)}"
        )


@router.post("/documents/update", response_model=StandardResponse)
async def update_documents(docs: list, incremental: bool = True):
    """批量更新文档"""
    service = get_service()
    if not service:
        return _create_standard_response(
            success=False,
            error="服务不可用"
        )
    
    try:
        result = service.update_documents(docs, incremental)
        return _create_standard_response(
            success=result['success'],
            message=result.get('message'),
            data=result.get('result'),
            error=result.get('error')
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"批量更新文档失败: {str(e)}"
        )
