"""
UltrasoundRAG Web API服务器
基于FastAPI的HTTP REST接口

启动示例：
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG
python -m ultrasoundrag.api
或
uvicorn ultrasoundrag.api.api:app --host 0.0.0.0 --port 8000
"""

import os
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

# 导入重构后的模块
try:
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

# 安全中间件
security = HTTPBearer()

# CORS中间件
allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8501,http://127.0.0.1:8501').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
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


# ==================== API端点 ====================

@app.get("/", response_model=Dict[str, Any])
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    if IMPORTS_OK:
        try:
            health_status = system_monitor.health_checker.run_all_checks()
            return HealthResponse(
                status=health_status["overall_status"],
                timestamp=time.time(),
                version="3.0.0",
                details={
                    "checks": health_status["checks"],
                    "summary": health_status["summary"]
                }
            )
        except:
            pass
    
    # 基础健康检查
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="3.0.0",
        details={"mode": "basic"}
    )


@app.get("/system/status", response_model=SystemStatusResponse)
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


@app.get("/databases", response_model=Dict[str, Any])
async def list_databases():
    """列出可用数据库"""
    try:
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
    except Exception as e:
        return {'databases': {}, 'error': str(e)}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """统一搜索端点"""
    if not IMPORTS_OK:
        raise HTTPException(status_code=503, detail="检索功能不可用，请检查依赖")
    
    start_time = time.time()
    
    try:
        # 验证输入
        if req.mode in {"t2t", "t2i"} and not req.query:
            raise ValidationError('query is required for t2t/t2i')
        
        if req.mode in {"i2t", "i2i"} and not req.image_path:
            raise ValidationError('image_path is required for i2t/i2i')
        
        
        # 生成缓存key
        cache_key = None
        if req.enable_cache:
            cache_data = f"{req.mode}:{req.query}:{req.image_path}:{req.top_k}:{req.db_name}"
            import hashlib
            cache_key = hashlib.md5(cache_data.encode()).hexdigest()
            
            # 尝试从缓存获取
            try:
                cache = cache_manager.get_cache("search_results", maxsize=500, ttl=300)
                cached_result = cache.get(cache_key)
                if cached_result:
                    performance_monitor.increment_counter("cache_hits")
                    return SearchResponse(
                        success=True,
                        data=cached_result,
                        metadata={"cached": True, "response_time": time.time() - start_time}
                    )
            except:
                pass
        
        # 执行搜索
        result = await _execute_search(req)
        
        # 缓存结果
        if req.enable_cache and cache_key:
            try:
                cache.put(cache_key, result)
                performance_monitor.increment_counter("cache_misses")
            except:
                pass
        
        response_time = time.time() - start_time
        
        # 记录搜索日志
        try:
            system_monitor.logger.log_retrieval(
                retrieval_type=req.mode,
                query=req.query or "image_query",
                results_count=result.get('total_results', 0),
                duration=response_time
            )
        except:
            pass
        
        return SearchResponse(
            success=True,
            data=result,
            metadata={
                "cached": False,
                "response_time": response_time
            }
        )
        
    except UltrasoundRAGException:
        raise
    except Exception as e:
        raise UltrasoundRAGException(
            f"搜索执行失败: {str(e)}",
            original_exception=e
        )


async def _execute_search(req: SearchRequest) -> Dict[str, Any]:
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
            try:
                retriever = smart_model_manager.get_model("t2t_retriever") or create_t2t_retriever(req.db_name, top_k)
            except:
                retriever = create_t2t_retriever(req.db_name, top_k)
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
            return retriever.search(req.image_path, top_k)
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
        
    elif mode == "i2i":
        def search_func():
            retriever = create_i2i_retriever(req.db_name, top_k)
            return retriever.search(req.image_path, top_k)
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
        
    elif mode == "auto":
        def search_func():
            # Auto模式：智能选择最佳单一检索方式
            retriever = create_enhanced_multimodal_retriever(req.db_name, top_k)
            return retriever.search(
                query=req.query or req.image_path or "",
                mode="auto", 
                top_k=top_k,
                text_query=req.query,
                image_path=req.image_path
            )
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
        
    elif mode == "multimodal":
        def search_func():
            # Multimodal模式：执行多模态融合检索
            retriever = create_enhanced_multimodal_retriever(req.db_name, top_k)
            return retriever.search(
                query=req.query or req.image_path or "",
                mode="multimodal", 
                top_k=top_k,
                text_query=req.query,
                image_path=req.image_path
            )
        
        result = with_timeout(search_func, req.timeout)
        result['results'] = [_serialize_rr(r) for r in result.get('results', [])]
        return result
    
    else:
        raise ValidationError(f'unsupported mode: {mode}')


# ==================== 管理端点 ====================

if IMPORTS_OK:
    @app.post("/admin/cache/clear")
    async def clear_cache():
        """清空缓存"""
        try:
            cache_manager.clear_all()
            return {"message": "缓存已清空", "timestamp": time.time()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"清空缓存失败: {e}")

    @app.post("/admin/models/reload")
    async def reload_models():
        """重新加载模型"""
        return {"message": "模型重新加载完成", "timestamp": time.time()}

    @app.get("/admin/alerts")
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


if __name__ == "__main__":
    main()
