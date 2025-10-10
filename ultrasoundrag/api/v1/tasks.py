"""
UltrasoundRAG API任务管理模块
提供异步任务的创建、查询、删除等管理接口
"""

import uuid
import threading
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException

from .models import Task, TaskStatus, StandardResponse, TaskResponse
from .utils import _create_standard_response

# 创建路由器
router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskManager:
    """全局任务管理器"""
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
    
    def create_task(self, name: str) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        with self._lock:
            self.tasks[task_id] = Task(
                id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                created_at=now,
                updated_at=now
            )
        
        return task_id
    
    def update_task(self, task_id: str, status: TaskStatus = None, 
                   progress: float = None, message: str = None, 
                   result: Dict = None, error: str = None):
        """更新任务状态"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if status is not None:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message is not None:
                    task.message = message
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                task.updated_at = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务信息"""
        with self._lock:
            return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[Task]:
        """列出所有任务"""
        with self._lock:
            return list(self.tasks.values())
    
    def delete_task(self, task_id: str):
        """删除任务"""
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]


# 全局任务管理器实例
task_manager = TaskManager()


@router.get("", response_model=StandardResponse)
async def list_tasks():
    """列出所有任务"""
    try:
        tasks = task_manager.list_tasks()
        return _create_standard_response(
            success=True,
            data={
                'tasks': [task.dict() for task in tasks],
                'total': len(tasks)
            }
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取任务列表失败: {str(e)}"
        )


@router.get("/{task_id}", response_model=StandardResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return _create_standard_response(
            success=True,
            data=task.dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"获取任务状态失败: {str(e)}"
        )


@router.delete("/{task_id}", response_model=StandardResponse)
async def delete_task(task_id: str):
    """删除任务"""
    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task_manager.delete_task(task_id)
        return _create_standard_response(
            success=True,
            message=f"任务 {task_id} 删除成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"删除任务失败: {str(e)}"
        )
