# """
# 全局配置管理模块
# """
# import yaml
# import os

# class ConfigManager:
#     _instance = None
#     _config = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(ConfigManager, cls).__new__(cls)
#         return cls._instance
    
#     def load_config(self, config_path=None):
#         """加载配置文件"""
#         if self._config is None:
#             if config_path is None:
#                 # 获取当前文件所在目录（config目录）
#                 current_dir = os.path.dirname(os.path.abspath(__file__))
#                 config_path = os.path.join(current_dir, "config.yaml")
            
#             print(f"尝试加载配置文件: {config_path}")  # 调试信息
            
#             if not os.path.exists(config_path):
#                 print(f"配置文件不存在: {config_path}")
#                 print(f"当前目录文件列表: {os.listdir(os.path.dirname(config_path))}")
#                 raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
#             with open(config_path, 'r', encoding='utf-8') as file:
#                 self._config = yaml.safe_load(file)
#                 print("配置文件加载成功")  # 调试信息
#         return self._config
    
#     @property
#     def config(self):
#         """获取配置"""
#         if self._config is None:
#             self.load_config()
#         return self._config

# # 创建全局配置实例
# config = ConfigManager().config

"""
全局配置管理模块
"""
import yaml
import os
import chromadb
from typing import Optional

class ConfigManager:
    _instance = None
    _config = None
    _chroma_client = None
    _chroma_collection = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def load_config(self, config_path=None):
        """加载配置文件"""
        if self._config is None:
            if config_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, "config.yaml")
            
            print(f"尝试加载配置文件: {config_path}")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
                print("配置文件加载成功")
        return self._config
    
    @property
    def config(self):
        """获取配置"""
        if self._config is None:
            self.load_config()
        return self._config
    
    @property
    def chroma_client(self):
        """获取全局ChromaDB客户端"""
        if self._chroma_client is None:
            print("初始化全局ChromaDB客户端...")
            
            # 从配置获取ChromaDB路径
            chroma_config = self.config.get('chroma', {})
            chroma_db_path = chroma_config.get('db_path', "./data/chroma_vectorstore")
            
            # 处理相对路径
            if not os.path.isabs(chroma_db_path):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                chroma_db_path = os.path.join(project_root, chroma_db_path)
            
            # 确保目录存在
            os.makedirs(chroma_db_path, exist_ok=True)
            
            # 创建ChromaDB客户端
            self._chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            print(f"ChromaDB客户端初始化成功: {chroma_db_path}")
        
        return self._chroma_client
    
    @property
    def chroma_collection(self):
        """获取全局ChromaDB集合"""
        if self._chroma_collection is None and self.chroma_client is not None:
            collection_name = self.config['document']['collection_name']
            
            try:
                # 尝试获取现有集合
                self._chroma_collection = self.chroma_client.get_collection(name=collection_name)
                print(f"加载现有集合: {collection_name}")
            except Exception:
                # 创建新集合
                distance_metric = self.config.get('chroma', {}).get('distance_metric', 'cosine')
                self._chroma_collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": distance_metric}
                )
                print(f"创建新集合: {collection_name}")
        
        return self._chroma_collection

# 创建全局配置实例
config_manager = ConfigManager()
config = config_manager.config

# 简化的全局访问函数
def get_chroma_client():
    """获取ChromaDB客户端"""
    return config_manager.chroma_client

def get_chroma_collection():
    """获取ChromaDB集合"""
    return config_manager.chroma_collection