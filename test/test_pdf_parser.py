#!/usr/bin/env python3
"""
PDF解析器测试脚本
测试指定的PDF文件解析功能
"""

import os
import sys
import json

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ultrasoundrag.data.loaders.pdf_parser import PDFParser


def test_pdf_parser():
    """测试PDF解析器功能"""
    print("=== PDF解析器功能测试 ===")
    
    # 测试文件路径
    test_pdf_path = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/thesis/Advancing Fetal Ultrasound.pdf"
    
    print(f"测试PDF文件: {test_pdf_path}")
    
    # 检查文件是否存在
    if not os.path.exists(test_pdf_path):
        print(f"❌ 错误：PDF文件不存在: {test_pdf_path}")
        return
    
    print("✅ PDF文件存在，开始解析...")
    
    try:
        # 初始化PDF解析器
        print("\n1. 初始化PDF解析器...")
        parser = PDFParser(dataset_name="thesis_papers")
        
        # 直接处理指定的PDF文件
        print("\n2. 开始处理PDF文件...")
        chunks = parser.process_document(test_pdf_path)
        
        print(f"\n3. 解析完成！总共生成 {len(chunks)} 个文档块")
        
        # 显示解析结果
        print("\n=== 解析结果详情 ===")
        for i, chunk in enumerate(chunks):
            print(f"\n--- 文档块 {i+1} ---")
            print(f"标题: {chunk['title']}")
            print(f"内容长度: {len(chunk['content'])} 字符")
            print(f"内容预览: {chunk['content'][:200]}...")
            print(f"图片数量: {len(chunk['image_paths'])}")
            if chunk['image_captions']:
                print(f"图片OCR文本: {chunk['image_captions'][:2]}")  # 显示前2个OCR结果
            print(f"块索引: {chunk['chunk_index']}")
        
        # 统计信息
        total_content_length = sum(len(chunk['content']) for chunk in chunks)
        chunks_with_images = sum(1 for chunk in chunks if chunk['image_paths'])
        total_images = sum(len(chunk['image_paths']) for chunk in chunks)
        
        print(f"\n=== 统计信息 ===")
        print(f"总文档块数: {len(chunks)}")
        print(f"总文本长度: {total_content_length} 字符")
        print(f"包含图片的块数: {chunks_with_images}")
        print(f"总图片数: {total_images}")
        
        # 保存结果到JSON文件
        output_file = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/pdf_parser_test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 解析结果已保存到: {output_file}")
        
        # 显示前几个块的完整内容
        print(f"\n=== 前3个文档块的完整内容 ===")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n{'='*50}")
            print(f"文档块 {i+1} - 标题: {chunk['title']}")
            print(f"{'='*50}")
            print(chunk['content'])
            if chunk['image_captions']:
                print(f"\n[图片OCR文本]:")
                for j, caption in enumerate(chunk['image_captions']):
                    print(f"  图片{j+1}: {caption}")
        
    except Exception as e:
        print(f"❌ 解析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pdf_parser()
