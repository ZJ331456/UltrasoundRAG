"""
ChromaDB统一管理模块

该模块提供了一个统一的ChromaDB管理接口，用于：
1. 管理ChromaDB客户端连接
2. 管理不同类型的集合（文档、图像等）
3. 提供统一的配置和初始化
4. 避免重复的客户端创建

使用方式：
```python
from MedicalRAG.utils.chroma_manager import chroma_manager

# 获取文档集合
doc_collection = chroma_manager.get_document_collection()

# 获取图像集合
image_collection = chroma_manager.get_image_collection()

# 获取自定义集合
custom_collection = chroma_manager.get_collection('custom_name', 'custom/path')
```
"""

import os
import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any
from MedicalRAG.utils.logger import setup_logger


class ChromaManager:
    """ChromaDB统一管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化ChromaDB管理器
        
        Args:
            config: 全局配置字典
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # 缓存客户端和集合
        self._clients: Dict[str, chromadb.PersistentClient] = {}
        self._collections: Dict[str, Any] = {}
        
        self.logger.info("ChromaDB管理器初始化完成")
    
    def get_client(self, db_path: str) -> chromadb.PersistentClient:
        """
        获取或创建ChromaDB客户端
        
        Args:
            db_path: 数据库路径
            
        Returns:
            ChromaDB客户端实例
        """
        # 使用绝对路径作为键
        abs_path = os.path.abspath(db_path)
        
        if abs_path not in self._clients:
            self.logger.info(f"创建新的ChromaDB客户端: {abs_path}")
            
            # 确保目录存在
            os.makedirs(abs_path, exist_ok=True)
            
            # 创建客户端
            self._clients[abs_path] = chromadb.PersistentClient(
                path=abs_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.logger.info(f"ChromaDB客户端创建成功: {abs_path}")
        
        return self._clients[abs_path]
    
    def get_collection(
        self, 
        collection_name: str, 
        db_path: str, 
        distance_metric: str = 'cosine',
        create_if_not_exists: bool = True
    ):
        """
        获取或创建集合
        
        Args:
            collection_name: 集合名称
            db_path: 数据库路径
            distance_metric: 距离度量方式
            create_if_not_exists: 如果集合不存在是否创建
            
        Returns:
            ChromaDB集合实例
        """
        # 使用路径和集合名称作为唯一键
        abs_path = os.path.abspath(db_path)
        collection_key = f"{abs_path}::{collection_name}"
        
        if collection_key not in self._collections:
            client = self.get_client(abs_path)
            
            try:
                if create_if_not_exists:
                    # 尝试获取现有集合，如果不存在则创建
                    collection = client.get_or_create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": distance_metric}
                    )
                    self.logger.info(f"获取或创建集合: {collection_name} (路径: {abs_path})")
                else:
                    # 只获取现有集合
                    collection = client.get_collection(name=collection_name)
                    self.logger.info(f"获取现有集合: {collection_name} (路径: {abs_path})")
                
                self._collections[collection_key] = collection
                
            except Exception as e:
                self.logger.error(f"获取集合失败: {collection_name} (路径: {abs_path}), 错误: {e}")
                raise
        
        return self._collections[collection_key]
    
    def get_document_collection(self):
        """
        获取文档集合
        
        Returns:
            文档集合实例
        """
        doc_config = self.config['indexing']['document']
        return self.get_collection(
            collection_name=doc_config['collection_name'],
            db_path=doc_config['vectorstore_path'],
            distance_metric=self.config.get('embedding', {}).get('distance_metric', 'cosine')
        )
    
    def get_image_collection(self):
        """
        获取图像集合
        
        Returns:
            图像集合实例
        """
        image_config = self.config['indexing']['image']
        return self.get_collection(
            collection_name=image_config['collection_name'],
            db_path=image_config['vectorstore_path'],
            distance_metric='cosine'  # 图像通常使用cosine距离
        )
    
    def get_custom_collection(self, collection_name: str, db_path: str, distance_metric: str = 'cosine'):
        """
        获取自定义集合
        
        Args:
            collection_name: 集合名称
            db_path: 数据库路径
            distance_metric: 距离度量方式
            
        Returns:
            自定义集合实例
        """
        return self.get_collection(collection_name, db_path, distance_metric)
    
    def delete_collection(self, collection_name: str, db_path: str) -> bool:
        """
        删除集合
        
        Args:
            collection_name: 集合名称
            db_path: 数据库路径
            
        Returns:
            是否删除成功
        """
        try:
            abs_path = os.path.abspath(db_path)
            collection_key = f"{abs_path}::{collection_name}"
            
            client = self.get_client(abs_path)
            client.delete_collection(name=collection_name)
            
            # 从缓存中移除
            if collection_key in self._collections:
                del self._collections[collection_key]
            
            self.logger.info(f"成功删除集合: {collection_name} (路径: {abs_path})")
            return True
            
        except Exception as e:
            self.logger.error(f"删除集合失败: {collection_name} (路径: {abs_path}), 错误: {e}")
            return False
    
    def get_collection_info(self, collection_name: str, db_path: str) -> Dict[str, Any]:
        """
        获取集合信息
        
        Args:
            collection_name: 集合名称
            db_path: 数据库路径
            
        Returns:
            集合信息字典
        """
        try:
            collection = self.get_collection(collection_name, db_path, create_if_not_exists=False)
            return {
                "collection_name": collection_name,
                "document_count": collection.count(),
                "vectorstore_path": os.path.abspath(db_path)
            }
        except Exception as e:
            self.logger.error(f"获取集合信息失败: {collection_name} (路径: {db_path}), 错误: {e}")
            return {"error": str(e)}
    
    def get_document_collection_info(self) -> Dict[str, Any]:
        """
        获取文档集合信息
        
        Returns:
            文档集合信息字典
        """
        doc_config = self.config['indexing']['document']
        return self.get_collection_info(
            collection_name=doc_config['collection_name'],
            db_path=doc_config['vectorstore_path']
        )
    
    def get_image_collection_info(self) -> Dict[str, Any]:
        """
        获取图像集合信息
        
        Returns:
            图像集合信息字典
        """
        image_config = self.config['indexing']['image']
        return self.get_collection_info(
            collection_name=image_config['collection_name'],
            db_path=image_config['vectorstore_path']
        )
    
    def clear_cache(self):
        """清除所有缓存的客户端和集合"""
        self._clients.clear()
        self._collections.clear()
        self.logger.info("ChromaDB缓存已清除")
    
    def close_all_connections(self):
        """关闭所有连接"""
        # ChromaDB的PersistentClient没有显式的close方法
        # 只需要清除缓存即可
        self.clear_cache()
        self.logger.info("所有ChromaDB连接已关闭")


# 全局ChromaDB管理器实例
_chroma_manager_instance: Optional[ChromaManager] = None


def get_chroma_manager(config: Optional[Dict[str, Any]] = None) -> ChromaManager:
    """
    获取全局ChromaDB管理器实例
    
    Args:
        config: 配置字典，仅在首次调用时需要
        
    Returns:
        ChromaDB管理器实例
    """
    global _chroma_manager_instance
    
    if _chroma_manager_instance is None:
        if config is None:
            # 如果没有提供配置，尝试从全局配置获取
            from MedicalRAG.config.config import config as global_config
            config = global_config
        
        _chroma_manager_instance = ChromaManager(config)
    
    return _chroma_manager_instance


# 便捷函数
def get_document_collection():
    """获取文档集合"""
    return get_chroma_manager().get_document_collection()


def get_image_collection():
    """获取图像集合"""
    return get_chroma_manager().get_image_collection()


def get_collection(collection_name: str, db_path: str, distance_metric: str = 'cosine'):
    """获取自定义集合"""
    return get_chroma_manager().get_collection(collection_name, db_path, distance_metric)


def get_chroma_client(db_path: str):
    """获取ChromaDB客户端"""
    return get_chroma_manager().get_client(db_path)