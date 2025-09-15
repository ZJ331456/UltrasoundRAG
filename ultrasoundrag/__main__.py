#!/usr/bin/env python3
"""
UltrasoundRAG 包主入口文件
支持 python -m ultrasoundrag 命令执行
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
