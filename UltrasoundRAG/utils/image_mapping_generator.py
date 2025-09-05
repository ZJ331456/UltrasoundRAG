#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片映射生成工具
生成image_id到图片路径的映射表
"""

import json
import os
from pathlib import Path
from typing import Dict

class ImageMappingGenerator:
    """图片映射生成器"""
    
    def __init__(self):
        self.image_roots = {
            'book': 'data/book/image',
            'ultrasound': 'data/ultrasound-image/image-data'
        }
    
    def generate_image_id(self, image_path: str, source_type: str) -> str:
        """生成唯一的图片ID"""
        import hashlib
        import base64
        
        # 基于路径和源类型生成唯一ID
        unique_string = f"{source_type}:{image_path}"
        hash_obj = hashlib.sha256(unique_string.encode())
        
        # 生成短ID（8位）
        short_id = base64.urlsafe_b64encode(hash_obj.digest()[:6]).decode('ascii')
        return short_id
    
    def get_image_info(self, image_path: str, source_type: str) -> Dict:
        """获取图片完整信息"""
        # 首先尝试从现有的image_mapping.json中查找
        mapping = self.load_mapping()
        
        # 在映射中查找匹配的图片路径
        for image_id, info in mapping.items():
            if info.get('image_path') == image_path:
                return info
        
        # 如果没找到，则生成新的ID
        image_id = self.generate_image_id(image_path, source_type)
        
        return {
            'image_id': image_id,
            'image_path': image_path,
            'source_type': source_type,
            'access_urls': {
                'info': f'/api/v1/images/{image_id}',
                'download': f'/api/v1/images/{image_id}/download',
                'view': f'/api/v1/images/{image_id}/view',
                'thumbnail': f'/api/v1/images/{image_id}/thumbnail'
            }
        }
    
    def generate_mapping(self) -> Dict:
        """生成图片ID到路径的映射表"""
        mapping = {}
        
        print("开始生成图片映射...")
        
        # 扫描book图片
        book_image_dir = Path('data/book/image')
        if book_image_dir.exists():
            print(f"扫描book图片目录: {book_image_dir}")
            for image_file in book_image_dir.rglob('*'):
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    rel_path = str(image_file.relative_to(book_image_dir))
                    image_info = self.get_image_info(rel_path, 'book')
                    mapping[image_info['image_id']] = image_info
                    print(f"  ✓ {rel_path} -> {image_info['image_id']}")
        else:
            print(f"book图片目录不存在: {book_image_dir}")
        
        # 扫描ultrasound图片
        ultrasound_image_dir = Path('data/ultrasound-image/image-data')
        if ultrasound_image_dir.exists():
            print(f"扫描ultrasound图片目录: {ultrasound_image_dir}")
            for image_file in ultrasound_image_dir.rglob('*'):
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    rel_path = str(image_file.relative_to(ultrasound_image_dir))
                    image_info = self.get_image_info(rel_path, 'ultrasound')
                    mapping[image_info['image_id']] = image_info
                    print(f"  ✓ {rel_path} -> {image_info['image_id']}")
        else:
            print(f"ultrasound图片目录不存在: {ultrasound_image_dir}")
        
        return mapping
    
    def save_mapping(self, mapping: Dict, output_file: str = 'data/image_mapping.json') -> str:
        """保存映射表到文件"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        print(f"\n图片映射生成完成！")
        print(f"总共生成 {len(mapping)} 个图片映射")
        print(f"映射文件保存到: {output_file}")
        
        return output_file
    
    def load_mapping(self, mapping_file: str = 'data/image_mapping.json') -> Dict:
        """加载图片映射表"""
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"映射文件不存在: {mapping_file}")
            return {}

def generate_image_mapping():
    """便捷函数：生成图片映射"""
    generator = ImageMappingGenerator()
    mapping = generator.generate_mapping()
    return generator.save_mapping(mapping)

if __name__ == '__main__':
    generate_image_mapping()
