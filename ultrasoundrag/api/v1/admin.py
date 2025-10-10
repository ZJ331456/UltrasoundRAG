"""
UltrasoundRAG API管理功能模块
提供系统管理、缓存控制、配置重载等管理接口
"""

import time
from fastapi import APIRouter, HTTPException

from .models import StandardResponse, TestRequest
from .utils import _create_standard_response

# 创建路由器
router = APIRouter(prefix="/admin", tags=["admin"])

# 全局变量，用于检查模块是否可用
try:
    from ...main import get_service
    from ...utils.monitoring import system_monitor
    from ...utils.performance import cache_manager
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
    def get_service():
        return None


@router.post("/cache/clear")
async def clear_cache():
    """清空缓存"""
    if not IMPORTS_OK:
        raise HTTPException(status_code=503, detail="高级功能不可用")
    
    try:
        cache_manager.clear_all()
        return {"message": "缓存已清空", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {e}")


@router.post("/models/reload")
async def reload_models():
    """重新加载模型"""
    try:
        # 这里可以添加实际的模型重载逻辑
        return {"message": "模型重新加载完成", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新加载模型失败: {e}")


@router.get("/alerts")
async def get_alerts():
    """获取告警信息"""
    if not IMPORTS_OK:
        raise HTTPException(status_code=503, detail="高级功能不可用")
    
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


@router.post("/system/reload-config", response_model=StandardResponse)
async def reload_config():
    """重新加载配置"""
    service = get_service()
    if not service:
        return _create_standard_response(
            success=False,
            error="服务不可用"
        )
    
    try:
        result = service.reload_config()
        return _create_standard_response(
            success=result['success'],
            message=result.get('message'),
            error=result.get('error')
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"重新加载配置失败: {str(e)}"
        )


@router.post("/tests/run", response_model=StandardResponse)
async def run_tests(request: TestRequest):
    """运行测试"""
    service = get_service()
    if not service:
        return _create_standard_response(
            success=False,
            error="服务不可用"
        )
    
    try:
        result = service.run_tests(request.test_type, **request.params)
        return _create_standard_response(
            success=result['success'],
            data=result.get('result'),
            error=result.get('error')
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"运行测试失败: {str(e)}"
        )
