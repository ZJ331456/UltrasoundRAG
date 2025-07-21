"""
文件操作工具模块

提供各种文件和目录操作的工具函数。
"""

import os
import time
import subprocess
import shutil


def force_remove_directory(path, logger, max_retries=3):
    """
    强制删除目录，处理文件被占用的情况
    
    Args:
        path (str): 要删除的目录路径
        logger: 日志记录器对象
        max_retries (int): 最大重试次数，默认为3
    
    Returns:
        bool: 删除成功返回True，失败返回False
    """
    if not os.path.exists(path):
        return True
    
    for attempt in range(max_retries):
        try:
            # 尝试正常删除
            shutil.rmtree(path)
            logger.info(f"成功删除目录: {path}")
            return True
        except PermissionError as e:
            if "另一个程序正在使用此文件" in str(e) or "being used by another process" in str(e):
                logger.warning(f"目录被占用，尝试强制删除 (尝试 {attempt + 1}/{max_retries}): {path}")
                
                # 尝试终止可能占用文件的进程
                try:
                    # 查找占用ChromaDB文件的进程
                    result = subprocess.run(
                        ['powershell', '-Command', f'Get-Process | Where-Object {{$_.Path -like "*streamlit*" -or $_.ProcessName -eq "python"}} | Select-Object Id, ProcessName, Path'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.stdout:
                        logger.info(f"发现可能占用文件的进程:\n{result.stdout}")
                except Exception as proc_e:
                    logger.warning(f"无法查询进程信息: {proc_e}")
                
                # 等待一段时间后重试
                time.sleep(2)
                
                # 最后一次尝试：使用Windows的rmdir命令强制删除
                if attempt == max_retries - 1:
                    try:
                        subprocess.run(['rmdir', '/s', '/q', path], shell=True, check=True)
                        logger.info(f"使用系统命令强制删除成功: {path}")
                        return True
                    except subprocess.CalledProcessError as cmd_e:
                        logger.error(f"系统命令删除失败: {cmd_e}")
            else:
                logger.error(f"删除目录失败: {e}")
                return False
        except Exception as e:
            logger.error(f"删除目录时发生未知错误: {e}")
            if attempt == max_retries - 1:
                return False
            time.sleep(1)
    
    return False


def ensure_directory_exists(path):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path (str): 目录路径
    """
    os.makedirs(path, exist_ok=True)


def is_file_accessible(file_path):
    """
    检查文件是否可访问（存在且可读）
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        bool: 文件可访问返回True，否则返回False
    """
    return os.path.exists(file_path) and os.access(file_path, os.R_OK)


def get_file_size(file_path):
    """
    获取文件大小
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        int: 文件大小（字节），文件不存在返回-1
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return -1 