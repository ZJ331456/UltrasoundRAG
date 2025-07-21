#!/usr/bin/env python3
"""
MedicalRAG - 医疗RAG系统
包含文档索引、检索、模型加载等功能模块
"""

# 导入主要模块
from . import config
from . import utils
from . import index
from . import retrival
from . import eval
from . import model

# 导入常用类和函数
from .model import FetalCLIPModel, load_fetal_clip_model

__version__ = "1.0.0"
__author__ = "MedicalRAG Team"

__all__ = [
    'config',
    'utils', 
    'index',
    'retrival',
    'eval',
    'model',
    'FetalCLIPModel',
    'load_fetal_clip_model'
]