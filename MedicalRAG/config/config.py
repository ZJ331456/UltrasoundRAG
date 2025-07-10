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
from typing import Optional

class ConfigManager:
    _instance = None
    _config = None
    
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
    
    def get_chroma_manager(self):
        """获取ChromaDB管理器"""
        from MedicalRAG.utils.chroma_manager import get_chroma_manager
        return get_chroma_manager(self.config)

# 创建全局配置实例
config_manager = ConfigManager()
config = config_manager.config

# 简化的全局访问函数
def get_chroma_client(db_path: Optional[str] = None):
    """获取ChromaDB客户端"""
    if db_path is None:
        # 使用默认的文档数据库路径
        db_path = config_manager.config['indexing']['document']['vectorstore_path']
    return config_manager.get_chroma_manager().get_client(db_path)

def get_chroma_collection():
    """获取默认文档集合"""
    return config_manager.get_chroma_manager().get_document_collection()

def get_document_collection():
    """获取文档集合"""
    return config_manager.get_chroma_manager().get_document_collection()

def get_image_collection():
    """获取图像集合"""
    return config_manager.get_chroma_manager().get_image_collection()