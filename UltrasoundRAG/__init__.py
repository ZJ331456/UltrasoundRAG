#!/usr/bin/env python3
"""
超声RAG系统
包含文档索引、检索、模型加载等功能模块
"""

# 统一设置 jieba 缓存目录到项目根目录，避免 /tmp 权限问题
import os
from pathlib import Path
import tempfile

try:
    # 将缓存统一放到项目根（包目录的上一级）
    ROOT_DIR = Path(__file__).resolve().parent.parent
    default_tmp_dir = ROOT_DIR / ".tmp"

    # 1) 重定向临时目录，避免写入系统 /tmp
    os.makedirs(default_tmp_dir, exist_ok=True)
    for var in ("TMPDIR", "TMP", "TEMP"):
        if not os.environ.get(var):
            os.environ[var] = str(default_tmp_dir)

    # 触发tempfile模块重新读取环境变量
    tempfile.tempdir = None
    tempfile.gettempdir()

    # 2) 统一Jieba缓存目录到 .tmp 下
    if not os.environ.get("JIEBA_CACHE_DIR"):
        os.environ["JIEBA_CACHE_DIR"] = str(default_tmp_dir / "jieba_cache")
    os.makedirs(os.environ["JIEBA_CACHE_DIR"], exist_ok=True)
except Exception:
    # 环境异常时忽略，不影响主流程
    pass
