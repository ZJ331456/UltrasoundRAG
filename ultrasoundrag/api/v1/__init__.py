"""
UltrasoundRAG API v1模块
模块化的API接口实现
"""

from fastapi import APIRouter

# 导入所有模块路由
from .health import router as health_router
from .databases import router as databases_router
from .collections import router as collections_router
from .documents import router as documents_router
from .search import router as search_router
from .tasks import router as tasks_router
from .files import router as files_router
from .indexes import router as indexes_router
from .admin import router as admin_router

# 创建v1主路由器（不设置prefix，由主应用设置）
v1_router = APIRouter()

# 注册所有子路由
v1_router.include_router(health_router)
v1_router.include_router(databases_router)
v1_router.include_router(collections_router)
v1_router.include_router(documents_router)
v1_router.include_router(search_router)
v1_router.include_router(tasks_router)
v1_router.include_router(files_router)
v1_router.include_router(indexes_router)
v1_router.include_router(admin_router)

# 导出主要组件
__all__ = [
    "v1_router",
    "health_router",
    "databases_router", 
    "collections_router",
    "documents_router",
    "search_router",
    "tasks_router",
    "files_router",
    "indexes_router",
    "admin_router"
]
