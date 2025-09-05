#!/usr/bin/env python3
"""
包含FetalCLIP模型加载器和相关工具
"""

from .fetal_clip_model import FetalCLIPModel, load_fetal_clip_model

__all__ = [
    'FetalCLIPModel',
    'load_fetal_clip_model'
]