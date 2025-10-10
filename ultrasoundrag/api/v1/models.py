"""
UltrasoundRAG API数据模型
定义所有API接口的请求和响应模型
"""

import time
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ==================== 枚举类型 ====================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================== 请求模型 ====================

class SearchRequest(BaseModel):
    """搜索请求模型"""
    db_name: str = Field(default="default", description="数据库名称")
    mode: str = Field(default="t2t", description="检索模式: t2t, t2i, i2t, i2i, multimodal, auto")
    query: Optional[str] = Field(None, description="查询文本")
    image_path: Optional[str] = Field(None, description="图片路径")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="请求超时时间(秒)")


class MultiDatabaseSearchRequest(BaseModel):
    """多数据库搜索请求模型（向后兼容）"""
    databases: List[str] = Field(description="数据库列表")
    mode: str = Field(default="t2t", description="检索模式")
    query: Optional[str] = Field(None, description="查询文本")
    image_path: Optional[str] = Field(None, description="图片路径")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")


class CollectionSearchRequest(BaseModel):
    """基于集合的搜索请求模型（新版本）"""
    text_collections: List[str] = Field(default_factory=list, description="文本集合列表")
    image_collections: List[str] = Field(default_factory=list, description="图片集合列表")
    mode: str = Field(default="t2t", description="检索模式: t2t, t2i, i2t, i2i, auto, multimodal")
    query: Optional[str] = Field(None, description="查询文本")
    image_path: Optional[str] = Field(None, description="图片路径")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    enable_cache: bool = Field(default=True, description="是否启用缓存")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="请求超时时间(秒)")
    
    def validate_collections(self) -> str:
        """验证集合配置"""
        if not self.text_collections and not self.image_collections:
            return "必须至少指定一个文本集合或图片集合"
        
        # 根据检索模式验证集合
        if self.mode == "t2t" and not self.text_collections:
            return "t2t模式需要指定文本集合"
        elif self.mode == "t2i" and not self.image_collections:
            return "t2i模式需要指定图片集合"
        elif self.mode == "i2t" and not self.text_collections:
            return "i2t模式需要指定文本集合"
        elif self.mode == "i2i" and not self.image_collections:
            return "i2i模式需要指定图片集合"
        
        return ""


class DatabaseConfig(BaseModel):
    """数据库配置模型"""
    db_name: str = Field(description="数据库名称")
    enabled: bool = Field(default=True, description="是否启用")
    description: str = Field(default="", description="数据库描述")
    collections: Dict[str, Any] = Field(default_factory=dict, description="集合配置")


class DatabaseCreateRequest(BaseModel):
    """数据库创建请求模型"""
    name: str = Field(description="数据库名称")
    config: DatabaseConfig = Field(description="数据库配置")


class DocumentRequest(BaseModel):
    """文档请求模型"""
    db_name: str = Field(description="数据库名称")
    collection: str = Field(description="集合名称")
    document: Dict[str, Any] = Field(description="文档内容")


class DocumentUpdateRequest(BaseModel):
    """文档更新请求模型"""
    db_name: str = Field(description="数据库名称")
    collection: str = Field(description="集合名称")
    document_id: str = Field(description="文档ID")
    document: Dict[str, Any] = Field(description="文档内容")


class IndexBuildRequest(BaseModel):
    """索引构建请求模型"""
    target: str = Field(default="all", description="构建目标: all, markdown, image")
    recreate: bool = Field(default=False, description="是否重新创建")


class TestRequest(BaseModel):
    """测试请求模型"""
    test_type: str = Field(default="retrieval", description="测试类型")
    params: Dict[str, Any] = Field(default_factory=dict, description="测试参数")


# ==================== 响应模型 ====================

class SearchResponse(BaseModel):
    """搜索响应模型"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: float
    version: str
    details: Dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
    cache_stats: Dict[str, Any]
    model_stats: Dict[str, Any]


class StandardResponse(BaseModel):
    """标准响应模型"""
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class TaskResponse(BaseModel):
    """任务响应模型"""
    success: bool
    task_id: str
    message: str
    timestamp: float = Field(default_factory=time.time)


# ==================== 任务管理模型 ====================

class Task(BaseModel):
    """任务模型"""
    id: str
    name: str
    status: TaskStatus
    progress: float = 0.0  # 0-100
    message: str = ""
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
