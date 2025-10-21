"""
UltrasoundRAG Web API服务器 - 重构版
基于FastAPI的HTTP REST接口，采用模块化设计

启动示例：
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG
python -m ultrasoundrag.api
或
uvicorn ultrasoundrag.api.api:app --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse
# from fastapi.responses import HTMLResponse  # 前端相关，暂时注释

# 导入v1模块化路由
from .v1 import v1_router

# 导入重构后的模块和统一服务
try:
    from ..utils.monitoring import system_monitor
    from ..utils.performance import (
        performance_monitor, smart_model_manager, cache_manager
    )
    from ..core.retrieval.modular_retrievers import (
        create_t2t_retriever, create_t2i_retriever
    )
    from ..utils.exceptions import UltrasoundRAGException
    IMPORTS_OK = True
except ImportError as e:
    print(f"警告: 部分模块导入失败 {e}，使用基础功能")
    IMPORTS_OK = False
    
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


# ==================== 应用生命周期管理 ====================

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


# ==================== 创建FastAPI应用 ====================

app = FastAPI(
    title="UltrasoundRAG Web API",
    description="医学超声RAG系统Web API - 重构版",
    version="3.1.0",
    lifespan=lifespan
)


# ==================== 中间件配置 ====================

# CORS中间件 - 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 信任主机中间件 - 允许所有主机
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # 允许所有主机
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


# ==================== 请求监控中间件 ====================

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


# ==================== 路由注册 ====================

# 注册API v1路由
app.include_router(v1_router, prefix="/api/v1/rag")

# 根端点
@app.get("/api/v1/rag/", response_model=dict)
async def root():
    """根端点"""
    return {
        "service": "UltrasoundRAG Web API",
        "version": "3.1.0",
        "status": "running",
        "timestamp": time.time(),
        "docs": "/docs",
        "health": "/api/v1/rag/health",
        "enhanced_features": IMPORTS_OK,
        "architecture": "modular"
    }


# ==================== 静态文件服务 ====================

# Vue前端服务 - 暂时注释掉
# static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "rag-vue", "dist")
# if os.path.exists(static_dir):
#     app.mount("/static", StaticFiles(directory=static_dir), name="static")
#     
#     # 添加前端路由支持
#     @app.get("/api/v1/rag/app/{path:path}")
#     @app.get("/api/v1/rag/app")
#     async def serve_frontend(path: str = "index.html"):
#         """服务Vue前端应用"""
#         file_path = os.path.join(static_dir, path if path else "index.html")
#         if not os.path.exists(file_path):
#             file_path = os.path.join(static_dir, "index.html")
#         
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#             
#             if file_path.endswith('.html'):
#                     return HTMLResponse(content=content)
#             elif file_path.endswith('.js'):
#                     return Response(content=content, media_type="application/javascript")
#             elif file_path.endswith('.css'):
#                     return Response(content=content, media_type="text/css")
#             else:
#                     return FileResponse(file_path)
#         except Exception as e:
#             from fastapi import HTTPException
#             raise HTTPException(status_code=404, detail=f"文件未找到: {e}")
# else:
#     print(f"⚠️  Vue前端构建文件未找到: {static_dir}")
#     print("💡 请先构建Vue前端：cd ultrasoundrag/web/rag-vue && npm run build")


# 图片静态文件服务
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


# ==================== 启动配置 ====================

def main():
    """主函数 - 启动Web API服务器"""
    import logging
    import uvicorn
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 启动UltrasoundRAG Web API服务器...")
    print("📍 地址: http://0.0.0.0:8000")
    print("📖 文档: http://0.0.0.0:8000/docs")
    print("🔧 架构: 模块化设计")
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
