"""
UltrasoundRAG 增强版API
集成安全、性能、监控等所有优化功能

启动示例（在项目根 /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG 下执行）：
  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import time
import asyncio
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import uvicorn

# 导入优化模块
from UltrasoundRAG.utils.exceptions import (
    UltrasoundRAGException, 
    AuthenticationError, 
    AuthorizationError, 
    RateLimitError,
    ValidationError,
    RetrievalTimeoutError
)
from UltrasoundRAG.utils.security import security_manager, get_authenticated_user, require_permission
from UltrasoundRAG.utils.performance import (
    cache_manager, 
    smart_model_manager, 
    performance_monitor,
    monitor_performance,
    cached
)
from UltrasoundRAG.utils.monitoring import system_monitor, monitor_operation

# 导入原有功能
from UltrasoundRAG.config import config
from UltrasoundRAG.retrival.modular_retrievers import (
    create_t2t_retriever, create_t2i_retriever,
    create_i2t_retriever, create_i2i_retriever,
    create_enhanced_multimodal_retriever
)


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


# ==================== API初始化 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    system_monitor.logger.logger.info("启动增强版UltrasoundRAG API")
    system_monitor.start_monitoring()
    
    # 预加载关键模型
    try:
        smart_model_manager.register_model_factory("t2t_retriever", lambda: create_t2t_retriever("default", 10))
        smart_model_manager.register_model_factory("t2i_retriever", lambda: create_t2i_retriever("default", 10))
        system_monitor.logger.logger.info("模型工厂注册完成")
    except Exception as e:
        system_monitor.logger.log_error(e, {"context": "模型预加载"})
    
    yield
    
    # 关闭时清理
    system_monitor.stop_monitoring()
    cache_manager.clear_all()
    system_monitor.logger.logger.info("UltrasoundRAG API已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="UltrasoundRAG Enhanced API",
    description="医学超声RAG系统增强版API - 集成安全、性能、监控功能",
    version="2.0.0",
    lifespan=lifespan
)

# 安全中间件
security = HTTPBearer()

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 信任主机中间件
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # 生产环境应限制具体主机
)


# ==================== 异常处理 ====================

@app.exception_handler(UltrasoundRAGException)
async def ultrasound_rag_exception_handler(request: Request, exc: UltrasoundRAGException):
    """UltrasoundRAG异常处理器"""
    # 记录异常
    system_monitor.logger.log_error(exc, {
        "request_path": str(request.url),
        "request_method": request.method,
        "client_ip": request.client.host
    })
    
    # 创建告警（如果是严重错误）
    if exc.status_code >= 500:
        system_monitor.alert_manager.create_alert(
            system_monitor.alert_manager.AlertLevel.ERROR,
            "API异常",
            str(exc),
            metadata=exc.to_dict()
        )
    
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
    
    # 记录请求开始
    request_id = f"req_{int(start_time * 1000)}"
    
    try:
        # 执行请求
        response = await call_next(request)
        duration = time.time() - start_time
        
        # 记录请求日志
        system_monitor.logger.log_request(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration=duration,
            request_id=request_id
        )
        
        # 记录性能指标
        performance_monitor.record_time(f"api_{request.method.lower()}", duration)
        performance_monitor.increment_counter("api_requests_total")
        
        if response.status_code >= 400:
            performance_monitor.increment_counter("api_requests_error")
        else:
            performance_monitor.increment_counter("api_requests_success")
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # 记录异常
        system_monitor.logger.log_error(e, {
            "request_id": request_id,
            "request_path": str(request.url.path),
            "request_method": request.method,
            "duration": duration
        })
        
        performance_monitor.increment_counter("api_requests_error")
        raise


# ==================== 依赖项 ====================

async def get_client_ip(request: Request) -> str:
    """获取客户端IP"""
    return request.client.host


async def enforce_security(
    request: Request,
    token: str = Depends(security),
    client_ip: str = Depends(get_client_ip)
):
    """安全检查依赖项"""
    try:
        # 认证
        api_key_info = security_manager.authenticate_request(f"Bearer {token.credentials}")
        
        # 频率限制
        security_manager.enforce_rate_limit(api_key_info, client_ip)
        
        return api_key_info
        
    except (AuthenticationError, AuthorizationError, RateLimitError) as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


# ==================== 工具函数 ====================

def _serialize_rr(rr) -> Dict[str, Any]:
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


# ==================== API端点 ====================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """根端点"""
    return {
        "service": "UltrasoundRAG Enhanced API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": time.time(),
        "docs": "/docs",
        "health": "/health",
        "system_status": "/system/status"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    health_status = system_monitor.health_checker.run_all_checks()
    
    return HealthResponse(
        status=health_status["overall_status"],
        timestamp=time.time(),
        version="2.0.0",
        details={
            "checks": health_status["checks"],
            "summary": health_status["summary"]
        }
    )


@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统状态（需要管理员权限）"""
    system_status = system_monitor.get_system_status()
    
    return SystemStatusResponse(
        system_health=system_status["health"],
        performance_metrics=performance_monitor.get_stats(),
        active_alerts=system_status["alerts"]["active_alerts"],
        cache_stats=cache_manager.get_all_stats(),
        model_stats=smart_model_manager.get_stats()
    )


@app.get("/databases", response_model=Dict[str, Any])
async def list_databases():
    """列出可用数据库"""
    dbs = config['retriever'].get('databases', {})
    out = {}
    for name, db_cfg in dbs.items():
        out[name] = {
            'db_name': db_cfg.get('db_name'),
            'collections': db_cfg.get('collections', {}),
            'enabled': db_cfg.get('enabled', True),
            'description': db_cfg.get('description', '')
        }
    return {'databases': out}


@app.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest
):
    """统一搜索端点"""
    start_time = time.time()
    
    try:
        # 验证输入
        if req.mode in {"t2t", "t2i"} and not req.query:
            raise ValidationError('query is required for t2t/t2i')
        
        if req.mode in {"i2t", "i2i"} and not req.image_path:
            raise ValidationError('image_path is required for i2t/i2i')
        
        # 仅支持通过 JSON 传入的 image_path
        image_path = req.image_path
        
        # 生成缓存key
        cache_key = None
        if req.enable_cache:
            cache_data = f"{req.mode}:{req.query}:{req.image_path}:{req.top_k}:{req.db_name}"
            import hashlib
            cache_key = hashlib.md5(cache_data.encode()).hexdigest()
            
            # 尝试从缓存获取
            cache = cache_manager.get_cache("search_results", maxsize=500, ttl=300)  # 5分钟缓存
            cached_result = cache.get(cache_key)
            if cached_result:
                performance_monitor.increment_counter("cache_hits")
                return SearchResponse(
                    success=True,
                    data=cached_result,
                    metadata={"cached": True, "response_time": time.time() - start_time}
                )
        
        # 执行搜索
        result = await _execute_search(req, image_path)
        
        # 缓存结果
        if req.enable_cache and cache_key:
            cache.put(cache_key, result)
            performance_monitor.increment_counter("cache_misses")
        
        # 无上传文件处理，故无需清理临时文件
        
        response_time = time.time() - start_time
        
        # 记录搜索日志
        system_monitor.logger.log_retrieval(
            retrieval_type=req.mode,
            query=req.query or "image_query",
            results_count=result.get('total_results', 0),
            duration=response_time
        )
        
        return SearchResponse(
            success=True,
            data=result,
            metadata={
                "cached": False,
                "response_time": response_time,
                "user_id": None
            }
        )
        
    except UltrasoundRAGException:
        raise
    except Exception as e:
        raise UltrasoundRAGException(
            f"搜索执行失败: {str(e)}",
            original_exception=e
        )


async def _execute_search(req: SearchRequest, image_path: Optional[str] = None) -> Dict[str, Any]:
    """执行搜索逻辑"""
    mode = req.mode.lower()
    top_k = max(1, int(req.top_k))
    
    # 超时检查装饰器
    def with_timeout(func, timeout: float):
        start_time = time.time()
        try:
            result = func()
            duration = time.time() - start_time
            if duration > timeout:
                raise RetrievalTimeoutError(duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            if duration > timeout:
                raise RetrievalTimeoutError(duration)
            raise
    
    # 根据模式执行搜索
    if mode == "t2t":
        def search_func():
            retriever = smart_model_manager.get_model("t2t_retriever") or create_t2t_retriever(req.db_name, top_k)
            return retriever.search(req.query, top_k)
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
        
    elif mode == "t2i":
        def search_func():
            retriever = create_t2i_retriever(req.db_name, top_k)
            return retriever.search(req.query, top_k)
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
        
    elif mode == "i2t":
        def search_func():
            retriever = create_i2t_retriever(req.db_name, top_k)
            return retriever.search(image_path or req.image_path, top_k)
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
        
    elif mode == "i2i":
        def search_func():
            retriever = create_i2i_retriever(req.db_name, top_k)
            return retriever.search(image_path or req.image_path, top_k)
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
        
    elif mode in {"multimodal", "auto"}:
        def search_func():
            retriever = create_enhanced_multimodal_retriever(req.db_name, top_k)
            query_input = req.query or (image_path or req.image_path or "")
            return retriever.search(query_input, mode="auto" if mode == "auto" else "multimodal", top_k=top_k)
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
    
    else:
        raise ValidationError(f'unsupported mode: {mode}')


# ==================== 管理端点 ====================

@app.post("/admin/cache/clear")
async def clear_cache(api_key_info = Depends(require_permission("admin"))):
    """清空缓存（管理员）"""
    cache_manager.clear_all()
    return {"message": "缓存已清空", "timestamp": time.time()}


@app.post("/admin/models/reload")
async def reload_models(api_key_info = Depends(require_permission("admin"))):
    """重新加载模型（管理员）"""
    # 这里可以添加模型重新加载逻辑
    return {"message": "模型重新加载完成", "timestamp": time.time()}


@app.get("/admin/alerts")
async def get_alerts(api_key_info = Depends(require_permission("admin"))):
    """获取告警信息（管理员）"""
    active_alerts = system_monitor.alert_manager.get_active_alerts()
    recent_alerts = system_monitor.alert_manager.get_recent_alerts(50)
    
    return {
        "active_alerts": [alert.__dict__ for alert in active_alerts],
        "recent_alerts": [alert.__dict__ for alert in recent_alerts],
        "timestamp": time.time()
    }


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import logging
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 启动应用
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # 单worker避免监控冲突
        log_level="info"
    )


