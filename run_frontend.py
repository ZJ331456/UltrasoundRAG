#!/usr/bin/env python3
"""
UltrasoundRAG 前端启动脚本
启动Streamlit前端界面用于检索测试

使用方法：
python run_frontend.py
"""

import subprocess
import sys
import os

def main():
    """启动前端界面"""
    print("🚀 启动 UltrasoundRAG 前端界面...")
    print("=" * 50)
    
    # 检查streamlit是否安装
    try:
        import streamlit
        print("✅ Streamlit 已安装")
    except ImportError:
        print("❌ Streamlit 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit 安装完成")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_file = os.path.join(current_dir, "frontend.py")
    
    if not os.path.exists(frontend_file):
        print(f"❌ 找不到前端文件: {frontend_file}")
        return
    
    print(f"📁 前端文件: {frontend_file}")
    print("🌐 启动Web服务器...")
    print("=" * 50)
    print("💡 提示：")
    print("   - 浏览器将自动打开 http://localhost:8501")
    print("   - 如果没有自动打开，请手动访问上述地址")
    print("   - 按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        # 启动streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            frontend_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 前端界面已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
