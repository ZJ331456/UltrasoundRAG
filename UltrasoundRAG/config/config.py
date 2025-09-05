"""
全局配置管理模块
"""
import yaml
import os
from typing import Optional, Dict, Any

class ConfigManager:
    _instance = None
    _config = None
    _config_path = None
    
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
            
            self._config_path = config_path
            print(f"尝试加载配置文件: {config_path}")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
                print("配置文件加载成功")
        return self._config

    @property
    def config_path(self):
        """获取配置文件路径"""
        if self._config_path is None:
            self.load_config()
        return self._config_path
    
    @property
    def config(self):
        """获取配置"""
        if self._config is None:
            self.load_config()
        return self._config

# 创建全局配置实例
config_manager = ConfigManager()

# 创建一个支持字典访问的配置对象
class ConfigDict(dict):
    """支持属性访问和字典访问的配置类"""
    def __init__(self, data: Dict[Any, Any]):
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            else:
                self[key] = value
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value

# 创建全局配置字典，支持 config['milvus']['milvus_uri'] 访问方式
config = ConfigDict(config_manager.config)

# 便捷的配置访问函数
def get_milvus_config():
    """获取Milvus配置"""
    return config['milvus']

def get_embedding_config():
    """获取embedding配置"""
    return config['embedding']

def get_llm_providers():
    """获取LLM提供者配置"""
    return config['llm_providers']

def get_embedding_providers():
    """获取embedding提供者配置"""
    return config['embedding_providers']

def get_indexing_config():
    """获取索引配置"""
    return config['indexing']

def get_retriever_config():
    """获取检索器配置"""
    return config['retriever']

def get_generator_config():
    """获取生成器配置"""
    return config['generator']

def get_rerank_config():
    """获取重排序配置"""
    return config['rerank']

def get_evaluator_config():
    """获取评估器配置"""
    return config['evaluator']

def get_prompt_engineer_config():
    """获取提示工程配置"""
    return config['prompt_engineer']

def get_document_collection():
    """获取文档集合名称"""
    return config['indexing']['markdown']['collections']

def get_image_collection():
    """获取图像集合名称"""
    return config['indexing']['image']['collections']

def get_book_text_collection():
    """获取书籍文本集合名称"""
    return config['indexing']['markdown']['collections']

def get_guides_collection():
    """获取指南集合名称"""
    return config['indexing']['markdown']['collections']

def get_book_image_collection():
    """获取书籍图像集合名称"""
    return config['indexing']['image']['collections']

def get_ultrasound_image_collection():
    """获取超声图像集合名称"""
    return config['indexing']['image']['collections']