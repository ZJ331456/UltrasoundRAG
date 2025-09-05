"""
文件操作工具模块

提供各种文件和目录操作的工具函数。
"""

import os
import time
import subprocess
import shutil
import json
from pathlib import Path


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

def generate_book_image_nodes():
    # 使用相对路径，便于迁移
    index_path = os.path.join('data', 'book', 'index.json')
    if not os.path.exists(index_path):
        return
    with open(index_path, 'r', encoding='utf-8') as f:
        book_index = json.load(f)
    
    nodes = []
    image_root = os.path.join('data', 'book', 'image')
    for md_file, image_folder in book_index.items():
        if not image_folder:
            continue
        image_dir = os.path.join(image_root, image_folder)
        if not os.path.isdir(image_dir):
            continue
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                caption = f"{os.path.splitext(md_file)[0]}: {os.path.splitext(img_file)[0]}"
                nodes.append({
                    'image_path': f'{image_folder}/{img_file}',  # 相对路径，基于base_path
                    'caption': caption,
                    'source': md_file
                })
    
    output_file = os.path.join('data', 'book', 'image_nodes.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + '\n')

def generate_ultrasound_nodes():
    """
    从 meta-data 目录下的JSON聚合生成 nodes：
    - 解析每个 JSON 的 DataInfo 映射，读取 data_path 与 caption
    - 将 image_path 统一写为 'image-data/{data_path}'，以匹配当前 base_path: 'data/ultrasound-image'
    - 仅写入真实存在的图片
    """
    base_root = os.path.join('data', 'ultrasound-image')
    image_root = os.path.join(base_root, 'image-data')
    meta_dir = os.path.join(base_root, 'meta-data')
    if not os.path.isdir(meta_dir):
        return
    os.makedirs(image_root, exist_ok=True)

    nodes = []
    json_files = [os.path.join(meta_dir, fn) for fn in os.listdir(meta_dir) if fn.lower().endswith('.json')]
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        data_info = data.get('DataInfo') or {}
        if not isinstance(data_info, dict):
            continue
        for _, item in data_info.items():
            rel_path = item.get('data_path')
            caption = item.get('caption', '')
            if not rel_path:
                continue
            image_rel = f"image-data/{rel_path}"
            abs_path = os.path.join(base_root, rel_path if rel_path.startswith('image-data/') else image_rel)
            if not os.path.exists(abs_path):
                continue
            nodes.append({
                'image_path': image_rel,
                'caption': caption or os.path.splitext(os.path.basename(rel_path))[0],
                'source': 'ultrasound-image'
            })

    output_file = os.path.join(base_root, 'image_nodes.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + '\n')

def load_image_mapping():
    """
    加载书籍图片映射信息，建立图片路径与caption的关联
    用于在文本索引时建立图文关联
    """
    project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    index_file = project_root / 'data' / 'book' / 'index.json'
    
    if not index_file.exists():
        print(f"Warning: Book index file not found: {index_file}")
        return {}
    
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            book_index = json.load(f)
        
        image_mapping = {}
        for book_name, image_name in book_index.items():
            if image_name:  # 跳过空值
                # 构建图片相对路径
                image_path = f"image/{image_name}"
                image_mapping[image_name] = {
                    'id': image_name,
                    'path': image_path,
                    'book_name': book_name,
                    'type': 'book_image'
                }
        
        print(f"Loaded {len(image_mapping)} image mappings from book index")
        return image_mapping
        
    except Exception as e:
        print(f"Error loading book index: {e}")
        return {}