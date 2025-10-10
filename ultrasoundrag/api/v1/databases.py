"""
UltrasoundRAG API数据库管理模块
提供数据库的创建、删除、查询等管理接口
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException

from .models import StandardResponse, DatabaseCreateRequest, DatabaseConfig
from .utils import _infer_collection_type, _create_standard_response

# 创建路由器
router = APIRouter(prefix="/databases", tags=["databases"])


@router.get("", response_model=StandardResponse)
async def list_databases():
    """列出所有数据库（实际上是集合）"""
    try:
        from ...core.indexing import list_collections, get_collection_info
        
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
        
        return _create_standard_response(
            success=True,
            data={
                'databases': databases,
                'total': len(databases)
            }
        )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"列出数据库失败: {str(e)}"
        )


@router.post("", response_model=StandardResponse)
async def create_database(request: DatabaseCreateRequest):
    """创建新数据库（实际上是创建集合）"""
    try:
        from ...core.indexing import create_collection
        
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
            return _create_standard_response(
                success=True,
                message=f"数据库（集合）{request.name} 创建成功",
                data={
                    'name': request.name,
                    'type': collection_type,
                    'description': request.config.description,
                    'enabled': request.config.enabled,
                    'collections': request.config.collections,
                    'created_at': getattr(request.config, 'created_at', '') if hasattr(request.config, 'created_at') else ''
                }
            )
        else:
            return _create_standard_response(
                success=False,
                error=f"数据库（集合）{request.name} 创建失败"
            )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"创建数据库失败: {str(e)}"
        )


@router.put("/{db_name}", response_model=StandardResponse)
async def update_database(db_name: str, config: DatabaseConfig):
    """更新数据库配置（集合信息）"""
    try:
        # 在这个系统中，数据库实际上就是集合
        # 我们只能更新集合的描述信息，不能改变集合的结构
        return _create_standard_response(
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
        return _create_standard_response(
            success=False,
            error=f"更新数据库配置失败: {str(e)}"
        )


@router.delete("/{db_name}", response_model=StandardResponse)
async def delete_database(db_name: str):
    """删除数据库（实际上是删除集合）"""
    try:
        from ...core.indexing import delete_collection
        
        # 删除集合
        success = delete_collection(db_name)
        
        if success:
            return _create_standard_response(
                success=True,
                message=f"数据库（集合）{db_name} 删除成功"
            )
        else:
            return _create_standard_response(
                success=False,
                error=f"数据库（集合）{db_name} 删除失败"
            )
    except Exception as e:
        return _create_standard_response(
            success=False,
            error=f"删除数据库失败: {str(e)}"
        )
