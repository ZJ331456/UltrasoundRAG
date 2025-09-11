#!/usr/bin/env python3
"""
PDF解析器使用示例
展示如何使用PDF解析器处理PDF文档并存储到Milvus数据库
"""

import os
import sys
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG')

from UltrasoundRAG.index.pdf_parse import PDFParser
from UltrasoundRAG.milvus.milvus_manager import MilvusManager
from UltrasoundRAG.config import config

def main():
    """主函数：演示PDF处理完整流程"""
    print("=== PDF文档处理完整流程示例 ===\n")
    
    try:
        # 1. 初始化PDF解析器
        print("1. 初始化PDF解析器...")
        pdf_parser = PDFParser(dataset_name="thesis_papers")
        print("✓ PDF解析器初始化成功\n")
        
        # 2. 解析PDF文档
        print("2. 解析PDF文档...")
        pdf_chunks = pdf_parser.parse_pdfs()
        print(f"✓ 成功解析 {len(pdf_chunks)} 个文本块\n")
        
        if not pdf_chunks:
            print("⚠ 没有找到PDF文档或解析失败")
            return
        
        # 3. 初始化Milvus管理器
        print("3. 初始化Milvus管理器...")
        milvus_manager = MilvusManager(collection_type="pdf")
        print("✓ Milvus管理器初始化成功\n")
        
        # 4. 显示解析结果示例
        print("4. 解析结果示例:")
        for i, chunk in enumerate(pdf_chunks[:2]):  # 显示前2个块
            print(f"\n块 {i+1}:")
            print(f"  ID: {chunk['id']}")
            print(f"  标题: {chunk['title']}")
            print(f"  文档: {chunk['document_name']}")
            print(f"  页面: {chunk['page_start']}-{chunk['page_end']}")
            print(f"  内容长度: {len(chunk['content'])} 字符")
            print(f"  内容预览: {chunk['content'][:150]}...")
        
        # 5. 模拟向量生成（实际使用时需要调用embedding模型）
        print(f"\n5. 准备数据插入...")
        print("注意: 实际使用时需要调用embedding模型生成向量")
        print("这里使用零向量作为示例")
        
        # 生成示例向量（实际使用时应该调用embedding模型）
        embeddings_qwen = [[0.0] * 1024 for _ in pdf_chunks]
        embeddings_clip = [[0.0] * 768 for _ in pdf_chunks]
        
        # 6. 插入数据到Milvus（注释掉，避免实际插入）
        print(f"\n6. 数据插入到Milvus...")
        print("注意: 以下代码被注释，避免实际插入数据")
        print("取消注释以下代码来实际插入数据:")
        print("""
        success = milvus_manager.insert_data(
            data_list=pdf_chunks,
            embeddings_qwen=embeddings_qwen,
            embeddings_clip=embeddings_clip
        )
        if success:
            print("✓ 数据插入成功")
        else:
            print("✗ 数据插入失败")
        """)
        
        # 7. 显示统计信息
        print(f"\n7. 处理统计:")
        total_chars = sum(len(chunk['content']) for chunk in pdf_chunks)
        doc_count = len(set(chunk['document_name'] for chunk in pdf_chunks))
        
        print(f"  处理文档数: {doc_count}")
        print(f"  生成文本块数: {len(pdf_chunks)}")
        print(f"  总字符数: {total_chars:,}")
        print(f"  平均块大小: {total_chars/len(pdf_chunks):.0f} 字符")
        
        # 8. 按文档统计
        doc_stats = {}
        for chunk in pdf_chunks:
            doc_name = chunk['document_name']
            if doc_name not in doc_stats:
                doc_stats[doc_name] = {'chunks': 0, 'chars': 0}
            doc_stats[doc_name]['chunks'] += 1
            doc_stats[doc_name]['chars'] += len(chunk['content'])
        
        print(f"\n8. 按文档统计:")
        for doc_name, stats in doc_stats.items():
            print(f"  {doc_name}: {stats['chunks']} 块, {stats['chars']:,} 字符")
        
        print(f"\n✓ PDF处理流程示例完成！")
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        if "pdfplumber" in str(e):
            print("请安装pdfplumber: pip install pdfplumber")
        elif "pymilvus" in str(e):
            print("请安装pymilvus: pip install pymilvus")
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()

def test_single_pdf():
    """测试单个PDF文件处理"""
    print("=== 单个PDF文件处理测试 ===\n")
    
    try:
        # 初始化解析器
        pdf_parser = PDFParser(dataset_name="thesis_papers")
        
        # 获取PDF文件列表
        pdf_files = pdf_parser.get_pdf_files()
        if not pdf_files:
            print("⚠ 没有找到PDF文件")
            return
        
        # 处理第一个PDF文件
        test_file = pdf_files[0]
        print(f"处理文件: {os.path.basename(test_file)}")
        
        chunks = pdf_parser.process_document(test_file)
        print(f"✓ 生成 {len(chunks)} 个文本块")
        
        # 显示详细信息
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n块 {i+1}:")
            print(f"  标题: {chunk['title']}")
            print(f"  页面: {chunk['page_start']}-{chunk['page_end']}")
            print(f"  内容: {chunk['content'][:200]}...")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")

if __name__ == "__main__":
    # 运行完整流程示例
    main()
    
    print("\n" + "="*50 + "\n")
    
    # 运行单个文件测试
    test_single_pdf()
