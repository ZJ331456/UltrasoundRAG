#!/usr/bin/env python3
"""
检查数据库中document_name字段的实际值
"""

import sys
import os
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG')

from ultrasoundrag.data.stores.milvus_store import MilvusManager

def check_document_names():
    """检查数据库中document_name字段的值"""
    try:
        # 连接到Milvus
        manager = MilvusManager(collection_type="md", collection_name="normal_book_md")
        
        # 查询所有数据，只获取document_name字段
        print("正在查询数据库中的document_name字段...")
        
        # 使用search_with_filter查询所有数据
        results = manager.search_with_filter("", limit=100)
        
        print(f"找到 {len(results)} 条记录")
        
        # 先检查第一条记录的结构
        if results:
            print("\n第一条记录的字段:")
            first_result = results[0]
            for key, value in first_result.items():
                print(f"  {key}: {value}")
        
        # 收集所有唯一的md_file值
        md_files = set()
        for result in results:
            md_file = result.get('md_file', '')
            if md_file:
                md_files.add(md_file)
        
        print(f"找到 {len(md_files)} 个唯一的md_file:")
        for i, md_file in enumerate(sorted(md_files), 1):
            print(f"{i}. {md_file}")
            
        # 检查是否包含我们要查找的文档
        target_doc = "12_超声医学第七版下册.md"
        if target_doc in md_files:
            print(f"\n✅ 找到目标文档: {target_doc}")
        else:
            print(f"\n❌ 未找到目标文档: {target_doc}")
            print("可能的原因:")
            print("1. 文档名称格式不匹配")
            print("2. 文档不存在于数据库中")
            print("3. 文档已被删除")
            
    except Exception as e:
        print(f"查询失败: {e}")

if __name__ == "__main__":
    check_document_names()
