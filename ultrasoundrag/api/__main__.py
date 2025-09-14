#!/usr/bin/env python3
"""
UltrasoundRAG API模块入口
支持直接启动Web API服务器

使用方法:
python -m ultrasoundrag.api [--port PORT] [--host HOST]
"""

import sys
import argparse

def main():
    """API模块主入口"""
    parser = argparse.ArgumentParser(description="UltrasoundRAG Web API 服务")
    parser.add_argument('--port', type=int, default=8000, help='端口号 (默认: 8000)')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--reload', action='store_true', help='启用自动重载')
    parser.add_argument('--workers', type=int, default=1, help='工作进程数')

    args = parser.parse_args()

    try:
        from .api import app
        import uvicorn

        print(f"🚀 启动UltrasoundRAG Web API服务器...")
        print(f"📍 地址: http://{args.host}:{args.port}")
        print(f"📖 文档: http://{args.host}:{args.port}/docs")
        print("=" * 50)

        uvicorn.run(
            "ultrasoundrag.api.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ 启动失败: 缺少依赖 {e}")
        print("请安装: pip install fastapi uvicorn")
        return 1
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
