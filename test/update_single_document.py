#!/usr/bin/env python3
"""
更新单个文档的脚本
用于删除集合中特定文档的旧内容并重新添加新内容
"""

import sys
import os
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG')

from ultrasoundrag.core.indexing import update_single_document
from ultrasoundrag.data.stores.milvus_store import MilvusManager
from ultrasoundrag.utils import setup_logger

def update_document_in_collection(document_name: str, dataset_name: str = "ultrasound_book", collection_name: str = "normal_book_md"):
    """
    更新集合中的单个文档
    
    Args:
        document_name: 文档名称（如 "12_超声医学第七版下册.md"）
        dataset_name: 数据集名称
        collection_name: 集合名称
    """
    logger = setup_logger("DocumentUpdater")
    
    try:
        logger.info(f"开始更新文档: {document_name}")
        
        # 方法1：直接使用系统的单个文档更新功能
        success = update_single_document(document_name, dataset_name)
        
        if success:
            logger.info(f"文档 {document_name} 更新成功！")
            return True
        else:
            logger.error(f"文档 {document_name} 更新失败！")
            return False
            
    except Exception as e:
        logger.error(f"更新文档时发生错误: {e}")
        return False

def manual_delete_and_add(document_name: str, collection_name: str = "normal_book_md"):
    """
    手动删除并重新添加文档内容
    
    Args:
        document_name: 文档名称
        collection_name: 集合名称
    """
    logger = setup_logger("ManualUpdater")
    
    try:
        logger.info(f"手动处理文档: {document_name}")
        
        # 1. 连接到Milvus
        manager = MilvusManager(collection_type="md", collection_name=collection_name)
        
        # 2. 删除旧内容
        logger.info(f"删除文档 {document_name} 的旧内容...")
        success, deleted_count = manager.delete_document_chunks(document_name)
        
        if success:
            logger.info(f"成功删除 {deleted_count} 个chunk")
        else:
            logger.error("删除操作失败")
            return False
        
        # 3. 重新解析并添加新内容
        logger.info(f"重新解析并添加文档 {document_name}...")
        
        # 使用MarkdownParser重新解析
        from ultrasoundrag.data.loaders.markdown_parser import MarkdownParser
        
        parser = MarkdownParser(dataset_name="ultrasound_book")
        
        # 解析特定文件
        file_path = f"/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/markdown/{document_name}"
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        
        # 解析单个文件
        parsed_data = parser.parse_single_file(file_path)
        
        if not parsed_data:
            logger.error("文件解析失败")
            return False
        
        # 生成嵌入向量
        from ultrasoundrag.core.indexing.builders import IndexBuilder
        builder = IndexBuilder()
        embeddings_qwen, embeddings_clip = builder._generate_text_embeddings(parsed_data)
        
        # 插入新数据
        success = builder._batch_insert_data(manager, parsed_data, embeddings_qwen, embeddings_clip)
        
        if success:
            logger.info(f"成功添加 {len(parsed_data)} 个新chunk")
            return True
        else:
            logger.error("添加新内容失败")
            return False
            
    except Exception as e:
        logger.error(f"手动处理时发生错误: {e}")
        return False

if __name__ == "__main__":
    document_name = "12_超声医学第七版下册.md"
    
    print("=" * 60)
    print(f"更新文档: {document_name}")
    print("=" * 60)
    
    # 方法1：使用系统内置功能（推荐）
    print("\n方法1：使用系统内置的单个文档更新功能")
    success = update_document_in_collection(document_name)
    
    if not success:
        print("\n方法1失败，尝试方法2：手动删除和添加")
        success = manual_delete_and_add(document_name)
    
    if success:
        print(f"\n✅ 文档 {document_name} 更新完成！")
    else:
        print(f"\n❌ 文档 {document_name} 更新失败！")
    
    print("=" * 60)
