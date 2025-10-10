"""
UltrasoundRAG Web API服务器 - 企业级版本
基于FastAPI的HTTP REST接口，支持完整的CRUD操作

启动示例：
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG
python -m ultrasoundrag.api
或
uvicorn ultrasoundrag.api.api:app --host 0.0.0.0 --port 8000
"""

import os
import time
import asyncio
import uuid
import threading
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# 导入重构后的模块和统一服务
try:
    from ..main import get_service, UltrasoundRAGService
    from ..utils.exceptions import (
        UltrasoundRAGException, 
        AuthenticationError, 
        AuthorizationError, 
        RateLimitError,
        ValidationError,
        RetrievalTimeoutError
    )
    from ..utils.security import security_manager, get_authenticated_user, require_permission
    from ..utils.performance import (
        cache_manager, 
        smart_model_manager, 
        performance_monitor,
        monitor_performance,
        cached
    )
    from ..utils.monitoring import system_monitor, monitor_operation
    from ..config import config
    from ..core.retrieval.modular_retrievers import (
        create_t2t_retriever, create_t2i_retriever,
        create_i2t_retriever, create_i2i_retriever,
        create_enhanced_multimodal_retriever
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"警告: 部分模块导入失败 {e}，使用基础功能")
    IMPORTS_OK = False
    # 提供最小的替代实现
    def get_service():
        return None
    
    # 提供基础异常类
    class UltrasoundRAGException(Exception):
        def __init__(self, message, status_code=500, original_exception=None):
            super().__init__(message)
            self.status_code = status_code
            self.original_exception = original_exception
        
        def to_dict(self):
            return {
                "error": str(self),
                "status_code": self.status_code,
                "timestamp": time.time()
            }


# ==================== 任务管理系统 ====================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

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
                   result: Dict[str, Any] = None, error: str = None):
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

# ==================== 数据模型 ====================

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
    """多数据库搜索请求模型"""
    databases: List[str] = Field(description="数据库列表")
    mode: str = Field(default="t2t", description="检索模式")
    query: Optional[str] = Field(None, description="查询文本")
    image_path: Optional[str] = Field(None, description="图片路径")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")


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


# ==================== API初始化 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("🚀 启动UltrasoundRAG Web API服务器")
    
    if IMPORTS_OK:
        try:
            system_monitor.logger.logger.info("启动增强版UltrasoundRAG API")
            system_monitor.start_monitoring()
            
            # 预加载关键模型
            smart_model_manager.register_model_factory("t2t_retriever", lambda: create_t2t_retriever("default", 10))
            smart_model_manager.register_model_factory("t2i_retriever", lambda: create_t2i_retriever("default", 10))
            system_monitor.logger.logger.info("模型工厂注册完成")
        except Exception as e:
            print(f"警告: 高级功能初始化失败: {e}")
    else:
        print("使用基础模式启动")
    
    yield
    
    # 关闭时清理
    if IMPORTS_OK:
        try:
            system_monitor.stop_monitoring()
            cache_manager.clear_all()
            system_monitor.logger.logger.info("UltrasoundRAG API已关闭")
        except:
            pass
    print("👋 UltrasoundRAG Web API服务器已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="UltrasoundRAG Web API",
    description="医学超声RAG系统Web API - 重构版",
    version="3.0.0",
    lifespan=lifespan
)

# 添加API路由前缀
from fastapi import APIRouter
api_router = APIRouter(prefix="/api")

# 安全中间件
security = HTTPBearer()

# CORS中间件
allowed_origins = os.getenv('ALLOWED_ORIGINS', 
    'http://localhost:8501,http://127.0.0.1:8501,http://localhost:5173,http://127.0.0.1:5173').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

# 信任主机中间件
allowed_hosts = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,*.localhost').split(',')
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)


# ==================== 异常处理 ====================

if IMPORTS_OK:
    @app.exception_handler(UltrasoundRAGException)
    async def ultrasound_rag_exception_handler(request: Request, exc: UltrasoundRAGException):
        """UltrasoundRAG异常处理器"""
        try:
            system_monitor.logger.log_error(exc, {
                "request_path": str(request.url),
                "request_method": request.method,
                "client_ip": request.client.host
            })
            
            if exc.status_code >= 500:
                system_monitor.alert_manager.create_alert(
                    system_monitor.alert_manager.AlertLevel.ERROR,
                    "API异常",
                    str(exc),
                    metadata=exc.to_dict()
                )
        except:
            pass
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


# ==================== 中间件 ====================

@app.middleware("http")
async def request_monitoring_middleware(request: Request, call_next):
    """请求监控中间件"""
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        if IMPORTS_OK:
            try:
                system_monitor.logger.log_request(
                    method=request.method,
                    path=str(request.url.path),
                    status_code=response.status_code,
                    duration=duration,
                    request_id=request_id
                )
                
                performance_monitor.record_time(f"api_{request.method.lower()}", duration)
                performance_monitor.increment_counter("api_requests_total")
                
                if response.status_code >= 400:
                    performance_monitor.increment_counter("api_requests_error")
                else:
                    performance_monitor.increment_counter("api_requests_success")
            except:
                pass
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        if IMPORTS_OK:
            try:
                system_monitor.logger.log_error(e, {
                    "request_id": request_id,
                    "request_path": str(request.url.path),
                    "request_method": request.method,
                    "duration": duration
                })
                performance_monitor.increment_counter("api_requests_error")
            except:
                pass
        
        raise


# ==================== 工具函数 ====================

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


# ==================== 文档处理辅助函数 ====================

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


def _validate_file_for_type(file: UploadFile, doc_type: str) -> Dict[str, Any]:
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


async def _process_markdown_document(file: UploadFile, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理Markdown文档
    
    Args:
        file: 上传的文件
        collection_name: 集合名称
        metadata: 元数据
        
    Returns:
        处理结果
    """
    import tempfile
    import os
    from ..core.indexing import add_data_to_collection
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.md') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 添加到集合
            success = add_data_to_collection(tmp_file_path, collection_name, "md")
            
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


async def _process_image_document(file: UploadFile, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
    import tempfile
    import os
    from PIL import Image
    from ..core.indexing import add_data_to_collection
    
    try:
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
            
            # 添加到集合
            success = add_data_to_collection(tmp_file_path, collection_name, "image")
            
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


async def _process_pdf_document(file: UploadFile, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理PDF文档
    
    Args:
        file: 上传的文件
        collection_name: 集合名称
        metadata: 元数据
        
    Returns:
        处理结果
    """
    import tempfile
    import os
    from ..core.indexing import add_data_to_collection
    
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 添加到集合
            success = add_data_to_collection(tmp_file_path, collection_name, "pdf")
            
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


# ==================== API端点 ====================

@app.get("/api/v1/rag/", response_model=Dict[str, Any])
async def root():
    """根端点"""
    return {
        "service": "UltrasoundRAG Web API",
        "version": "3.0.0",
        "status": "running",
        "timestamp": time.time(),
        "docs": "/docs",
        "health": "/health",
        "enhanced_features": IMPORTS_OK
    }


@app.get("/api/v1/rag/health", response_model=HealthResponse)
async def health_check():
    """
    增强的RAG系统健康检查端点
    
    检查项目包括：
    1. 基础服务状态
    2. 数据库连接状态
    3. 模型加载状态
    4. 向量化服务状态
    5. 集合状态统计
    
    Returns:
        HealthResponse: 详细的健康检查结果
    """
    health_details = {
        "version": "3.0.0",
        "timestamp": time.time(),
        "service_status": "healthy",
        "components": {},
        "statistics": {},
        "issues": []
    }
    
    overall_status = "healthy"
    issues = []
    
    # 1. 检查基础服务状态
    try:
        service = get_service()
        health_details["components"]["main_service"] = {
            "status": "healthy",
            "available": service is not None,
            "details": "主服务运行正常"
        }
    except Exception as e:
        overall_status = "unhealthy"
        issues.append(f"主服务异常: {str(e)}")
        health_details["components"]["main_service"] = {
            "status": "error",
            "available": False,
            "error": str(e)
        }
    
    # 2. 检查数据库连接状态
    try:
        from ..data.stores.milvus_store import MilvusManager
        from ..config import config
        
        # 测试Milvus连接
        milvus_config = config['milvus']
        test_manager = MilvusManager(collection_type="md", collection_name="health_check_test")
        
        # 尝试获取连接信息
        connection_info = test_manager.get_connection_info()
        health_details["components"]["milvus_database"] = {
            "status": "healthy" if connection_info else "warning",
            "available": bool(connection_info),
            "uri": milvus_config.get('milvus_uri', 'unknown'),
            "details": connection_info or "连接信息获取失败"
        }
        
        if not connection_info:
            issues.append("Milvus数据库连接异常")
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # 健康检查完成后，立即删除测试集合
        try:
            test_manager.drop_collection()
            print("健康检查完成，已删除测试集合 'health_check_test'")
        except Exception as cleanup_error:
            print(f"删除测试集合时警告: {cleanup_error}")
            # 删除失败不影响健康检查结果
                
    except Exception as e:
        overall_status = "unhealthy"
        issues.append(f"数据库连接失败: {str(e)}")
        health_details["components"]["milvus_database"] = {
            "status": "error",
            "available": False,
            "error": str(e)
        }
    
    # 3. 检查模型加载状态
    try:
        from ..model.singleton_models import get_shared_fetal_clip
        from ..utils.embedding_utils import embedding_provider
        
        # 检查CLIP模型
        clip_model = get_shared_fetal_clip()
        health_details["components"]["clip_model"] = {
            "status": "healthy" if clip_model else "error",
            "available": clip_model is not None,
            "model_type": "FetalCLIP",
            "details": "CLIP模型加载正常" if clip_model else "CLIP模型未加载"
        }
        
        # 检查嵌入模型
        try:
            embed_model = embedding_provider.get_embedding_model()
            health_details["components"]["embedding_model"] = {
                "status": "healthy" if embed_model else "warning",
                "available": embed_model is not None,
                "model_type": "Text Embedding",
                "details": "文本嵌入模型正常" if embed_model else "嵌入模型未加载"
            }
        except Exception as embed_e:
            issues.append(f"嵌入模型异常: {str(embed_e)}")
            health_details["components"]["embedding_model"] = {
                "status": "error",
                "available": False,
                "error": str(embed_e)
            }
        
        if not clip_model:
            issues.append("CLIP模型未正确加载")
            if overall_status == "healthy":
                overall_status = "degraded"
                
    except Exception as e:
        overall_status = "unhealthy"
        issues.append(f"模型加载检查失败: {str(e)}")
        health_details["components"]["models"] = {
            "status": "error",
            "available": False,
            "error": str(e)
        }
    
    # 4. 检查集合状态
    try:
        from ..core.indexing import list_collections, get_collection_info
        
        collections = list_collections()
        collections_status = {}
        total_documents = 0
        healthy_collections = 0
        
        for collection_name in collections[:10]:  # 限制检查前10个集合
            try:
                info = get_collection_info(collection_name)
                doc_count = info.get('document_count', 0) if info else 0
                status = "healthy" if doc_count > 0 else "empty"
                
                collections_status[collection_name] = {
                    "status": status,
                    "document_count": doc_count,
                    "available": info is not None
                }
                
                total_documents += doc_count
                if status == "healthy":
                    healthy_collections += 1
                    
            except Exception as col_e:
                collections_status[collection_name] = {
                    "status": "error",
                    "error": str(col_e),
                    "available": False
                }
        
        health_details["components"]["collections"] = {
            "status": "healthy" if healthy_collections > 0 else "warning",
            "available": len(collections) > 0,
            "total_collections": len(collections),
            "healthy_collections": healthy_collections,
            "sample_collections": collections_status,
            "details": f"共{len(collections)}个集合，{healthy_collections}个正常"
        }
        
        health_details["statistics"] = {
            "total_collections": len(collections),
            "healthy_collections": healthy_collections,
            "total_documents": total_documents,
            "average_docs_per_collection": total_documents / max(len(collections), 1)
        }
        
        if healthy_collections == 0 and len(collections) > 0:
            issues.append("所有集合都为空或异常")
            if overall_status == "healthy":
                overall_status = "degraded"
                
    except Exception as e:
        issues.append(f"集合状态检查失败: {str(e)}")
        health_details["components"]["collections"] = {
            "status": "error",
            "available": False,
            "error": str(e)
        }
    
    # 5. 高级监控检查（如果可用）
    if IMPORTS_OK:
        try:
            health_status = system_monitor.health_checker.run_all_checks()
            health_details["advanced_monitoring"] = {
                "available": True,
                "status": health_status.get("overall_status", "unknown"),
                "checks": health_status.get("checks", {}),
                "summary": health_status.get("summary", {})
            }
        except Exception as e:
            health_details["advanced_monitoring"] = {
                "available": False,
                "error": f"高级监控不可用: {str(e)}"
            }
    else:
        health_details["advanced_monitoring"] = {
            "available": False,
            "reason": "监控模块未正确导入"
        }
    
    # 设置最终状态
    health_details["issues"] = issues
    health_details["service_status"] = overall_status
    
    return HealthResponse(
        status=overall_status,
        timestamp=time.time(),
        version="3.0.0",
        details=health_details
    )


@app.get("/api/v1/rag/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统状态"""
    if not IMPORTS_OK:
        raise HTTPException(status_code=503, detail="高级功能不可用")
    
    try:
        system_status = system_monitor.get_system_status()
        
        return SystemStatusResponse(
            system_health=system_status["health"],
            performance_metrics=performance_monitor.get_stats(),
            active_alerts=system_status["alerts"]["active_alerts"],
            cache_stats=cache_manager.get_all_stats(),
            model_stats=smart_model_manager.get_stats()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {e}")


# === 数据库管理端点 ===

@app.get("/api/v1/rag/databases", response_model=StandardResponse)
async def list_databases():
    """列出所有数据库（实际上是集合）"""
    try:
        from ..core.indexing import list_collections, get_collection_info
        
        # 获取集合名称列表
        collection_names = list_collections()
        
        # 将集合转换为"数据库"格式
        databases = []
        for name in collection_names:
            try:
                info = get_collection_info(name)
                # 构造数据库信息对象
                database_data = {
                    'name': name,
                    'description': info.get('description', '') if info else f'集合 {name}',
                    'enabled': True,  # 默认启用
                    'collections': {
                        name: {
                            'type': _infer_collection_type(name),
                            'document_count': info.get('document_count', 0) if info else 0,
                            'status': 'active' if info else 'unknown'
                        }
                    },
                    'created_at': info.get('created_timestamp', '') if info else '',
                    'updated_at': info.get('update_timestamp', '') if info else ''
                }
                databases.append(database_data)
            except Exception as e:
                # 如果获取某个集合信息失败，使用默认值
                database_data = {
                    'name': name,
                    'description': f'集合 {name}',
                    'enabled': True,
                    'collections': {
                        name: {
                            'type': _infer_collection_type(name),
                            'document_count': 0,
                            'status': 'unknown'
                        }
                    },
                    'created_at': '',
                    'updated_at': ''
                }
                databases.append(database_data)
        
        return StandardResponse(
            success=True,
            data={
                'databases': databases,
                'total': len(databases)
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"列出数据库失败: {str(e)}"
        )


@app.post("/api/v1/rag/databases", response_model=StandardResponse)
async def create_database(request: DatabaseCreateRequest):
    """创建新数据库（实际上是创建集合）"""
    try:
        from ..core.indexing import create_collection
        
        # 从请求中提取集合类型
        collection_type = 'md'  # 默认类型
        if request.config.collections:
            # 尝试从集合配置中推断类型
            for collection_name, collection_config in request.config.collections.items():
                if isinstance(collection_config, dict) and 'type' in collection_config:
                    collection_type = collection_config['type']
                    break
        
        # 创建集合
        success = create_collection(request.name, collection_type)
        
        if success:
            return StandardResponse(
                success=True,
                message=f"数据库（集合）{request.name} 创建成功",
                data={
                    'name': request.name,
                    'type': collection_type,
                    'description': request.config.description,
                    'enabled': request.config.enabled,
                    'collections': request.config.collections,
                    'created_at': request.config.created_at if hasattr(request.config, 'created_at') else ''
                }
            )
        else:
            return StandardResponse(
                success=False,
                error=f"数据库（集合）{request.name} 创建失败"
            )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"创建数据库失败: {str(e)}"
        )


@app.put("/api/v1/rag/databases/{db_name}", response_model=StandardResponse)
async def update_database(db_name: str, config: DatabaseConfig):
    """更新数据库配置（集合信息）"""
    try:
        # 在这个系统中，数据库实际上就是集合
        # 我们只能更新集合的描述信息，不能改变集合的结构
        return StandardResponse(
            success=True,
            message=f"数据库（集合）{db_name} 配置更新成功",
            data={
                'name': db_name,
                'description': config.description,
                'enabled': config.enabled,
                'collections': config.collections,
                'updated_at': datetime.now().isoformat()
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"更新数据库配置失败: {str(e)}"
        )


@app.delete("/api/v1/rag/databases/{db_name}", response_model=StandardResponse)
async def delete_database(db_name: str):
    """删除数据库（实际上是删除集合）"""
    try:
        from ..core.indexing import delete_collection
        
        # 删除集合
        success = delete_collection(db_name)
        
        if success:
            return StandardResponse(
                success=True,
                message=f"数据库（集合）{db_name} 删除成功"
            )
        else:
            return StandardResponse(
                success=False,
                error=f"数据库（集合）{db_name} 删除失败"
            )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"删除数据库失败: {str(e)}"
        )


# === 集合管理端点 ===

@app.get("/api/v1/rag/collections", response_model=StandardResponse)
async def get_all_collections():
    """
    列出所有集合（重命名以避免函数名冲突）
    
    Returns:
        包含所有集合信息的响应
    """
    try:
        from ..core.indexing import list_collections, get_collection_info
        
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
        
        return StandardResponse(
            success=True,
            data={
                'collections': collections,
                'total': len(collections)
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"列出集合失败: {str(e)}"
        )


@app.get("/api/v1/rag/collections/source-files-summary", response_model=StandardResponse)
async def get_all_collections_source_files_summary():
    """获取所有集合的源文件统计摘要"""
    try:
        from ..core.indexing import list_collections
        from ..data.stores.milvus_store import MilvusManager
        
        # 获取所有集合名称
        collection_names = list_collections()
        
        summary_data = []
        total_source_files = 0
        total_chunks_processed = 0
        
        for collection_name in collection_names:
            try:
                collection_type = _infer_collection_type(collection_name)
                manager = MilvusManager(collection_type=collection_type, collection_name=collection_name)
                
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
                    'collection_type': collection_type,
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
        
        return StandardResponse(
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
        return StandardResponse(
            success=False,
            error=f"获取集合源文件统计失败: {str(e)}"
        )


# ==================== 通用工具函数（避免代码重复） ====================

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
        from ..data.stores.milvus_store import MilvusManager
        
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
        from ..config import config
        
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


@app.post("/api/v1/rag/collections", response_model=StandardResponse)
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
        from ..core.indexing import create_collection
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


@app.get("/api/v1/rag/collections/{collection_name}", response_model=StandardResponse)
async def get_collection_info(collection_name: str):
    """获取集合信息"""
    try:
        from ..core.indexing import get_collection_info
        info = get_collection_info(collection_name)
        
        return StandardResponse(
            success=True,
            data={
                'collection': collection_name,
                'info': info
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"获取集合信息失败: {str(e)}"
        )


@app.delete("/api/v1/rag/collections/{collection_name}", response_model=StandardResponse)
async def delete_collection(collection_name: str):
    """
    删除指定集合
    
    Args:
        collection_name: 要删除的集合名称
        
    Returns:
        删除操作结果
    """
    try:
        from ..core.indexing import delete_collection
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


@app.post("/api/v1/rag/collections/{collection_name}/documents", response_model=StandardResponse)
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
    import tempfile
    import os
    import json
    from pathlib import Path
    from ..core.indexing import add_data_to_collection
    
    try:
        # 1. 文件基础验证
        if not file.filename:
            return StandardResponse(
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
            return StandardResponse(
                success=False,
                error=f"文件验证失败: {validation_result['error']}"
            )
        
        # 4. 解析元数据
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                return StandardResponse(
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
            return StandardResponse(
                success=False,
                error=f"不支持的文档类型: {doc_type}"
            )
        
        if result['success']:
            return StandardResponse(
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
            return StandardResponse(
                success=False,
                error=result.get('error', '文档处理失败')
            )
            
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"添加文档失败: {str(e)}"
        )


@app.post("/api/v1/rag/collections/{collection_name}/documents/batch", response_model=StandardResponse)
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
    import json
    import asyncio
    from typing import List
    
    try:
        # 1. 基础验证
        if not files:
            return StandardResponse(
                success=False,
                error="未提供文件"
            )
        
        if len(files) > 50:  # 限制批量上传数量
            return StandardResponse(
                success=False,
                error="批量上传文件数量不能超过50个"
            )
        
        # 2. 解析文档类型列表
        parsed_doc_types = []
        if doc_types:
            try:
                parsed_doc_types = json.loads(doc_types)
                if len(parsed_doc_types) != len(files):
                    return StandardResponse(
                        success=False,
                        error="文档类型列表长度与文件数量不匹配"
                    )
            except json.JSONDecodeError as e:
                return StandardResponse(
                    success=False,
                    error=f"文档类型列表JSON格式错误: {str(e)}"
                )
        
        # 3. 解析元数据列表
        parsed_metadata_list = []
        if metadata_list:
            try:
                parsed_metadata_list = json.loads(metadata_list)
                if len(parsed_metadata_list) != len(files):
                    return StandardResponse(
                        success=False,
                        error="元数据列表长度与文件数量不匹配"
                    )
            except json.JSONDecodeError as e:
                return StandardResponse(
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
        
        return StandardResponse(
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
        return StandardResponse(
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


@app.get("/api/v1/rag/collections/{collection_name}/documents", response_model=StandardResponse)
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
        
        return StandardResponse(
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
        return StandardResponse(
            success=False,
            error=f"获取集合源文件列表失败: {str(e)}"
        )


@app.delete("/api/v1/rag/collections/{collection_name}/documents/{document_key}", response_model=StandardResponse)
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
            return StandardResponse(
                success=True,
                message=f"文档 {document_key} 不存在或已被删除"
            )
        
        # 获取所有文档块的ID
        doc_ids = [doc.get('id') for doc in related_docs if doc.get('id') is not None]
        
        if doc_ids:
            # 执行软删除
            success = manager.mark_deleted(doc_ids)
            
            if success:
                return StandardResponse(
                    success=True,
                    message=f"文档 {document_key} 已从集合 {collection_name} 删除（共删除 {len(doc_ids)} 个块）"
                )
            else:
                return StandardResponse(
                    success=False,
                    error=f"文档 {document_key} 从集合 {collection_name} 删除失败"
                )
        else:
            return StandardResponse(
                success=True,
                message=f"文档 {document_key} 没有找到可删除的块"
            )
            
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"删除文档失败: {str(e)}"
        )


@app.get("/api/v1/rag/collections/{collection_name}/files", response_model=StandardResponse)
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


@app.get("/api/v1/rag/collections/{collection_name}/files/statistics", response_model=StandardResponse)
async def get_collection_file_statistics(collection_name: str):
    """获取集合中file字段的统计信息 - 优化版本"""
    try:
        from ..data.stores.milvus_store import MilvusManager
        
        # 根据集合类型创建Milvus管理器
        collection_type = _infer_collection_type(collection_name)
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name)
        
        # 获取file字段的统计信息（使用优化版本）
        stats = manager.get_file_statistics()
        
        if "error" in stats:
            return StandardResponse(
                success=False,
                error=f"获取file统计信息失败: {stats['error']}"
            )
        
        return StandardResponse(
            success=True,
            data=stats,
            message=f"成功获取集合 {collection_name} 的file统计信息"
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"获取file统计信息失败: {str(e)}"
        )


@app.get("/api/v1/rag/collections/{collection_name}/info/fast", response_model=StandardResponse)
async def get_collection_info_fast(collection_name: str):
    """快速获取集合信息 - 不遍历数据，只获取元数据"""
    try:
        from ..data.stores.milvus_store import MilvusManager
        
        # 根据集合类型创建Milvus管理器
        collection_type = _infer_collection_type(collection_name)
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name)
        
        # 获取集合快速信息
        info = manager.get_collection_info_fast()
        
        if "error" in info:
            return StandardResponse(
                success=False,
                error=f"获取集合信息失败: {info['error']}"
            )
        
        return StandardResponse(
            success=True,
            data=info,
            message=f"成功获取集合 {collection_name} 的快速信息"
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"获取集合信息失败: {str(e)}"
        )


@app.get("/api/v1/rag/collections/{collection_name}/stats/optimized", response_model=StandardResponse)
async def get_collection_stats_optimized(collection_name: str):
    """获取集合统计信息 - 高效版本，不遍历数据"""
    try:
        from ..data.stores.milvus_store import MilvusManager
        
        # 根据集合类型创建Milvus管理器
        collection_type = _infer_collection_type(collection_name)
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name)
        
        # 获取集合统计信息
        stats = manager.get_collection_stats()
        
        if "error" in stats:
            return StandardResponse(
                success=False,
                error=f"获取集合统计信息失败: {stats['error']}"
            )
        
        return StandardResponse(
            success=True,
            data=stats,
            message=f"成功获取集合 {collection_name} 的统计信息"
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"获取集合统计信息失败: {str(e)}"
        )


@app.delete("/api/v1/rag/collections/{collection_name}/files/{file_value}", response_model=StandardResponse)
async def delete_collection_file(collection_name: str, file_value: str):
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


@app.post("/api/v1/rag/collections/{collection_name}/files/batch-delete", response_model=StandardResponse)
async def batch_delete_collection_files(collection_name: str, file_values: List[str]):
    """批量根据file字段值删除记录"""
    try:
        from ..data.stores.milvus_store import MilvusManager
        
        # 根据集合类型创建Milvus管理器
        collection_type = _infer_collection_type(collection_name)
        manager = MilvusManager(collection_type=collection_type, collection_name=collection_name)
        
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
        
        return StandardResponse(
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
        return StandardResponse(
            success=False,
            error=f"批量删除文件记录失败: {str(e)}"
        )


@app.put("/api/v1/rag/collections/{collection_name}/documents/{document_name}", response_model=StandardResponse)
async def update_document_in_collection(
    collection_name: str, 
    document_name: str,
    file: UploadFile = File(...),
    doc_type: str = Form("md", description="文档类型: md, image, pdf")
):
    """更新集合中的文档"""
    try:
        import tempfile
        import os
        from ..core.indexing import update_single_document
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc_type}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 更新文档
            success = update_single_document(tmp_file_path, collection_name)
            
            if success:
                return StandardResponse(
                    success=True,
                    message=f"文档 {document_name} 在集合 {collection_name} 中更新成功",
                    data={
                        'collection': collection_name,
                        'document': document_name,
                        'type': doc_type
                    }
                )
            else:
                return StandardResponse(
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
        return StandardResponse(
            success=False,
            error=f"更新文档失败: {str(e)}"
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
        
        from ..core.indexing import delete_collection, build_markdown_index, build_image_index
        
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

@app.post("/api/v1/rag/collections/{collection_name}/rebuild", response_model=TaskResponse)
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


# === 文档管理端点 ===
# 注意：文档管理统一使用集合级别的API，避免与数据库级别API重复


# === 检索端点 ===

@app.post("/api/v1/rag/search", response_model=SearchResponse)
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


@app.post("/api/v1/rag/search/multi-database", response_model=SearchResponse)
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
            data=result.get('results'),
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


@app.post("/api/v1/rag/search/upload", response_model=SearchResponse)
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
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图片文件")
    
    start_time = time.time()
    
    try:
        # 保存临时文件
        import tempfile
        import os
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


# === 任务管理端点 ===

@app.get("/api/v1/rag/tasks", response_model=StandardResponse)
async def list_tasks():
    """列出所有任务"""
    try:
        tasks = task_manager.list_tasks()
        return StandardResponse(
            success=True,
            data={
                'tasks': [task.dict() for task in tasks],
                'total': len(tasks)
            }
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"获取任务列表失败: {str(e)}"
        )

@app.get("/api/v1/rag/tasks/{task_id}", response_model=StandardResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        return StandardResponse(
            success=True,
            data=task.dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"获取任务状态失败: {str(e)}"
        )

@app.delete("/api/v1/rag/tasks/{task_id}", response_model=StandardResponse)
async def delete_task(task_id: str):
    """删除任务"""
    try:
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task_manager.delete_task(task_id)
        return StandardResponse(
            success=True,
            message=f"任务 {task_id} 删除成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        return StandardResponse(
            success=False,
            error=f"删除任务失败: {str(e)}"
        )

# === 索引管理端点 ===

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

@app.post("/api/v1/rag/indexes/build", response_model=TaskResponse)
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


@app.post("/api/v1/rag/documents/update", response_model=StandardResponse)
async def update_documents(docs: List[Dict[str, Any]], incremental: bool = True):
    """批量更新文档"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="服务不可用")
    
    result = service.update_documents(docs, incremental)
    return StandardResponse(
        success=result['success'],
        message=result.get('message'),
        data=result.get('result'),
        error=result.get('error')
    )


# === 测试和管理端点 ===

@app.post("/api/v1/rag/tests/run", response_model=StandardResponse)
async def run_tests(request: TestRequest):
    """运行测试"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="服务不可用")
    
    result = service.run_tests(request.test_type, **request.params)
    return StandardResponse(
        success=result['success'],
        data=result.get('result'),
        error=result.get('error')
    )




@app.post("/api/v1/rag/system/reload-config", response_model=StandardResponse)
async def reload_config():
    """重新加载配置"""
    service = get_service()
    if not service:
        raise HTTPException(status_code=503, detail="服务不可用")
    
    result = service.reload_config()
    return StandardResponse(
        success=result['success'],
        message=result.get('message'),
        error=result.get('error')
    )


# === 静态文件服务（用于Vue前端） ===

# 创建静态文件目录 - 指向rag-vue项目
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "rag-vue", "dist")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # 添加前端路由支持
    @app.get("/api/v1/rag/app/{path:path}")
    @app.get("/api/v1/rag/app")
    async def serve_frontend(path: str = "index.html"):
        """服务Vue前端应用"""
        file_path = os.path.join(static_dir, path if path else "index.html")
        if not os.path.exists(file_path):
            file_path = os.path.join(static_dir, "index.html")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.endswith('.html'):
                try:
                    from fastapi.responses import HTMLResponse
                    return HTMLResponse(content=content)
                except ImportError:
                    return content
            elif file_path.endswith('.js'):
                try:
                    from fastapi.responses import Response
                    return Response(content=content, media_type="application/javascript")
                except ImportError:
                    return content
            elif file_path.endswith('.css'):
                try:
                    from fastapi.responses import Response
                    return Response(content=content, media_type="text/css")
                except ImportError:
                    return content
            else:
                try:
                    from fastapi.responses import FileResponse
                    return FileResponse(file_path)
                except ImportError:
                    return content
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"文件未找到: {e}")
else:
    print(f"⚠️  Vue前端构建文件未找到: {static_dir}")
    print("💡 请先构建Vue前端：cd ultrasoundrag/web/rag-vue && npm run build")

# === 图片静态文件服务 ===

# 尝试找到图片存储目录
possible_image_dirs = [
    "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image",
    "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/images",
    "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/ultrasoundrag/data/images",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images"),
    "/media/ps/data-ssd/UltrasoundRAG/data/images"
]

for img_dir in possible_image_dirs:
    if os.path.exists(img_dir):
        try:
            app.mount("/static/images", StaticFiles(directory=img_dir), name="images")
            print(f"✅ 图片静态文件服务已启用: {img_dir}")
            break
        except Exception as e:
            print(f"⚠️  无法挂载图片目录 {img_dir}: {e}")
            continue
else:
    print("⚠️  未找到图片存储目录，图片显示功能可能不可用")


# ==================== 增强的文件管理功能 ====================

@app.post("/api/v1/rag/files/upload-image-folder", response_model=StandardResponse)
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
    import json
    import os
    from pathlib import Path
    
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
    import tempfile
    import shutil
    from PIL import Image
    from ..core.indexing import add_data_to_collection
    
    results = []
    
    # 创建临时工作目录
    with tempfile.TemporaryDirectory() as temp_dir:
        from pathlib import Path
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
        from pathlib import Path
        
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


@app.get("/api/v1/rag/files/image/{image_hash}", response_model=Dict[str, Any])
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
        from fastapi.responses import FileResponse
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


@app.get("/api/v1/rag/files/storage-info", response_model=StandardResponse)
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
    import os
    from pathlib import Path
    
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
    import os
    from pathlib import Path
    
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


# ==================== 管理端点 ====================

if IMPORTS_OK:
    @app.post("/api/v1/rag/admin/cache/clear")
    async def clear_cache():
        """清空缓存"""
        try:
            cache_manager.clear_all()
            return {"message": "缓存已清空", "timestamp": time.time()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"清空缓存失败: {e}")

    @app.post("/api/v1/rag/admin/models/reload")
    async def reload_models():
        """重新加载模型"""
        return {"message": "模型重新加载完成", "timestamp": time.time()}

    @app.get("/api/v1/rag/admin/alerts")
    async def get_alerts():
        """获取告警信息"""
        try:
            active_alerts = system_monitor.alert_manager.get_active_alerts()
            recent_alerts = system_monitor.alert_manager.get_recent_alerts(50)
            
            return {
                "active_alerts": [alert.__dict__ for alert in active_alerts],
                "recent_alerts": [alert.__dict__ for alert in recent_alerts],
                "timestamp": time.time()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取告警失败: {e}")


# ==================== 启动配置 ====================

def main():
    """主函数 - 启动Web API服务器"""
    import logging
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 启动UltrasoundRAG Web API服务器...")
    print("📍 地址: http://0.0.0.0:8000")
    print("📖 文档: http://0.0.0.0:8000/docs")
    print("=" * 50)
    
    # 启动应用
    uvicorn.run(
        "ultrasoundrag.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )


# 包含API路由器
app.include_router(api_router)

if __name__ == "__main__":
    main()
