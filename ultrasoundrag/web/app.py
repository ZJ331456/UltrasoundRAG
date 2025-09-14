#!/usr/bin/env python3
"""
UltrasoundRAG Web应用启动脚本
便于快速启动Streamlit前端界面

使用方法:
python -m ultrasoundrag.web.app [--port PORT] [--host HOST]
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def main():
    """启动Web界面"""
    parser = argparse.ArgumentParser(description="启动UltrasoundRAG Web界面")
    parser.add_argument("--port", type=int, default=8501, help="端口号 (默认: 8501)")
    parser.add_argument("--host", default="localhost", help="主机地址 (默认: localhost)")
    parser.add_argument("--open-browser", action="store_true", help="自动打开浏览器")
    
    args = parser.parse_args()
    
    # 获取frontend.py的路径
    current_dir = Path(__file__).parent
    frontend_path = current_dir / "frontend.py"
    
    if not frontend_path.exists():
        print(f"错误: 找不到前端文件 {frontend_path}")
        return 1
    
    # 获取项目根目录
    project_root = current_dir.parent.parent

    # 构建streamlit命令，直接在命令中设置PYTHONPATH
    pythonpath_value = str(project_root)
    if 'PYTHONPATH' in os.environ:
        pythonpath_value = pythonpath_value + os.pathsep + os.environ['PYTHONPATH']

    cmd = [
        "env", f"PYTHONPATH={pythonpath_value}",
        "streamlit", "run", str(frontend_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true"
    ]
    
    if not args.open_browser:
        cmd.extend(["--server.runOnSave", "false"])
    
    print(f"🚀 启动UltrasoundRAG Web界面...")
    print(f"📍 地址: http://{args.host}:{args.port}")
    print(f"📁 前端文件: {frontend_path}")
    print(f"⚡ 执行命令: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # 启动Streamlit
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在退出...")
        return 0
    except FileNotFoundError:
        print("❌ 错误: 未找到 streamlit 命令")
        print("请先安装 streamlit: pip install streamlit")
        return 1

if __name__ == "__main__":
    exit(main())
