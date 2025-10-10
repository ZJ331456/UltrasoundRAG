"""
UltrasoundRAG API健康检查模块
提供系统健康检查和状态监控接口
"""

import time
from fastapi import APIRouter, HTTPException

from .models import HealthResponse, SystemStatusResponse, StandardResponse
from .utils import _create_standard_response

# 创建路由器
router = APIRouter(prefix="/health", tags=["health"])

# 全局变量，用于检查模块是否可用
try:
    from ...main import get_service
    from ...utils.performance import performance_monitor
    from ...utils.monitoring import system_monitor
    from ...utils.performance import cache_manager, smart_model_manager
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False
    def get_service():
        return None


@router.get("/", response_model=HealthResponse)
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
        from ...data.stores.milvus_store import MilvusManager
        from ...config import config
        
        # 测试Milvus连接
        milvus_config = config['milvus']
        test_manager = MilvusManager(collection_type="md", collection_name="health_check_test")
        
        # 尝试获取集合统计信息来测试连接
        collection_stats = test_manager.get_collection_stats()
        is_connection_ok = "error" not in collection_stats
        health_details["components"]["milvus_database"] = {
            "status": "healthy" if is_connection_ok else "error",
            "available": is_connection_ok,
            "uri": milvus_config.get('milvus_uri', 'unknown'),
            "details": "数据库连接正常" if is_connection_ok else collection_stats.get("error", "连接异常")
        }
        
        if not is_connection_ok:
            issues.append(f"Milvus数据库连接失败: {collection_stats.get('error', '未知错误')}")
            overall_status = "unhealthy"
        
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
        from ...model.singleton_models import get_shared_fetal_clip
        from ...utils.embedding_utils import embedding_provider
        
        # 检查CLIP模型
        clip_model = get_shared_fetal_clip()
        health_details["components"]["clip_model"] = {
            "status": "healthy" if clip_model else "error",
            "available": clip_model is not None,
            "model_type": "FetalCLIP",
            "details": "CLIP模型加载正常" if clip_model else "CLIP模型未加载"
        }
        
        # 检查嵌入模型 - 尝试获取默认嵌入提供者
        try:
            # 尝试访问默认嵌入提供者来测试是否可用
            from ...config import config
            embedding_config = config.get('embedding_providers', {})
            default_provider = next(iter(embedding_config.keys())) if embedding_config else None
            
            if default_provider:
                embed_model = embedding_provider[default_provider]
                health_details["components"]["embedding_model"] = {
                    "status": "healthy" if embed_model else "warning",
                    "available": embed_model is not None,
                    "model_type": "Text Embedding",
                    "provider": default_provider,
                    "details": f"文本嵌入模型正常 ({default_provider})" if embed_model else "嵌入模型未加载"
                }
            else:
                health_details["components"]["embedding_model"] = {
                    "status": "warning",
                    "available": False,
                    "details": "未配置嵌入提供者"
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
        from ...core.indexing import list_collections, get_collection_info
        
        collections = list_collections()
        collections_status = {}
        total_documents = 0
        healthy_collections = 0
        
        for collection_name in collections[:10]:  # 限制检查前10个集合
            try:
                info = get_collection_info(collection_name)
                doc_count = info.get('document_count', 0) if info else 0
                
                # 检测测试集合，它们可以为空
                is_test_collection = "test" in collection_name.lower() or "health_check" in collection_name.lower()
                
                if info is not None:
                    if doc_count > 0:
                        status = "healthy"
                        healthy_collections += 1
                    elif is_test_collection:
                        status = "ready"  # 测试集合可以为空
                        healthy_collections += 1
                    else:
                        status = "empty"
                else:
                    status = "unavailable"
                
                collections_status[collection_name] = {
                    "status": status,
                    "document_count": doc_count,
                    "available": info is not None,
                    "is_test": is_test_collection
                }
                
                total_documents += doc_count
                    
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
        
        # 只有当没有任何可用集合时才报告问题
        if healthy_collections == 0 and len(collections) > 0:
            # 检查是否有非测试集合
            non_test_collections = [c for c in collections if not ("test" in c.lower() or "health_check" in c.lower())]
            if non_test_collections:
                issues.append("生产集合为空或异常")
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


@router.get("/system/status", response_model=SystemStatusResponse)
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


# === 根端点 ===
@router.get("/info", response_model=dict)
async def get_api_info():
    """获取API基本信息"""
    return {
        "service": "UltrasoundRAG Web API",
        "version": "3.0.0",
        "status": "running",
        "timestamp": time.time(),
        "docs": "/docs",
        "health": "/api/v1/health",
        "enhanced_features": IMPORTS_OK
    }
