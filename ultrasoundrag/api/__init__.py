"""
UltrasoundRAG API模块

提供基于FastAPI的Web API服务器入口与导出。
"""

# Web API相关
try:
    from .api import app as web_app, main as start_web_api
    WEB_API_AVAILABLE = True
except ImportError:
    WEB_API_AVAILABLE = False
    web_app = None
    start_web_api = None

__all__ = [
    # Web API
    'web_app',
    'start_web_api',
    'WEB_API_AVAILABLE',
]
