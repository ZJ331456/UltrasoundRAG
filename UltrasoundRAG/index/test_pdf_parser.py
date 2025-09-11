#!/usr/bin/env python3
"""
PDF解析器测试脚本
用于测试PDF文本提取和分块功能
"""

import os
import sys
import json
from datetime import datetime
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG')

from UltrasoundRAG.index.pdf_parse import PDFParser

def save_results_to_file(all_chunks, pdf_files, output_file="pdf_parser_results.txt"):
    """将解析结果保存到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入文件头信息
            f.write("=" * 80 + "\n")
            f.write("PDF解析器测试结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理文件数: {len(pdf_files)}\n")
            f.write(f"生成文本块数: {len(all_chunks)}\n")
            f.write("=" * 80 + "\n\n")
            
            # 写入PDF文件列表
            f.write("处理的PDF文件列表:\n")
            f.write("-" * 40 + "\n")
            for i, pdf_file in enumerate(pdf_files, 1):
                f.write(f"{i:2d}. {os.path.basename(pdf_file)}\n")
            f.write("\n")
            
            # 统计信息
            total_chars = sum(len(chunk['content']) for chunk in all_chunks)
            avg_chunk_size = total_chars / len(all_chunks) if all_chunks else 0
            
            f.write("统计信息:\n")
            f.write("-" * 40 + "\n")
            f.write(f"总文本块数: {len(all_chunks)}\n")
            f.write(f"总字符数: {total_chars:,}\n")
            f.write(f"平均块大小: {avg_chunk_size:.0f} 字符\n\n")
            
            # 按文档统计
            doc_stats = {}
            for chunk in all_chunks:
                doc_name = chunk['document_name']
                if doc_name not in doc_stats:
                    doc_stats[doc_name] = {'chunks': 0, 'chars': 0}
                doc_stats[doc_name]['chunks'] += 1
                doc_stats[doc_name]['chars'] += len(chunk['content'])
            
            f.write("按文档统计:\n")
            f.write("-" * 40 + "\n")
            for doc_name, stats in doc_stats.items():
                f.write(f"{doc_name}: {stats['chunks']} 块, {stats['chars']:,} 字符\n")
            f.write("\n")
            
            # 写入详细的文本块内容
            f.write("详细文本块内容:\n")
            f.write("=" * 80 + "\n")
            
            for i, chunk in enumerate(all_chunks, 1):
                f.write(f"\n块 {i}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"ID: {chunk['id']}\n")
                f.write(f"标题: {chunk['title']}\n")
                f.write(f"文档: {chunk['document_name']}\n")
                f.write(f"PDF文件: {chunk['pdf_file']}\n")
                f.write(f"页面范围: {chunk['page_start']}-{chunk['page_end']}\n")
                f.write(f"块索引: {chunk['chunk_index']}\n")
                f.write(f"内容长度: {len(chunk['content'])} 字符\n")
                f.write(f"内容:\n{chunk['content']}\n")
                f.write("-" * 40 + "\n")
            
            f.write(f"\n\n文件生成完成，共 {len(all_chunks)} 个文本块\n")
        
        print(f"✓ 结果已保存到文件: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ 保存文件失败: {e}")
        return False

def test_pdf_parser():
    """测试PDF解析器功能"""
    print("开始测试PDF解析器...")
    
    try:
        # 初始化PDF解析器
        parser = PDFParser(dataset_name="thesis_papers")
        print("✓ PDF解析器初始化成功")
        
        # 获取PDF文件列表
        pdf_files = parser.get_pdf_files()
        print(f"✓ 找到 {len(pdf_files)} 个PDF文件")
        
        if not pdf_files:
            print("⚠ 没有找到PDF文件，请检查路径配置")
            return
        
        # 显示找到的PDF文件
        for i, pdf_file in enumerate(pdf_files[:3]):  # 只显示前3个
            print(f"  {i+1}. {os.path.basename(pdf_file)}")
        
        # 测试单个PDF文件处理
        test_file = pdf_files[0]
        print(f"\n测试处理文件: {os.path.basename(test_file)}")
        
        chunks = parser.process_document(test_file)
        print(f"✓ 成功处理，生成 {len(chunks)} 个文本块")
        
        # 显示前几个块的信息
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n块 {i+1}:")
            print(f"  标题: {chunk['title']}")
            print(f"  内容长度: {len(chunk['content'])} 字符")
            print(f"  页面范围: {chunk['page_start']}-{chunk['page_end']}")
            print(f"  内容预览: {chunk['content'][:100]}...")
        
        # 测试批量处理
        print(f"\n开始批量处理所有PDF文件...")
        all_chunks = parser.parse_pdfs()
        print(f"✓ 批量处理完成，总共生成 {len(all_chunks)} 个文本块")
        
        # 统计信息
        total_chars = sum(len(chunk['content']) for chunk in all_chunks)
        avg_chunk_size = total_chars / len(all_chunks) if all_chunks else 0
        
        print(f"\n统计信息:")
        print(f"  总文本块数: {len(all_chunks)}")
        print(f"  总字符数: {total_chars:,}")
        print(f"  平均块大小: {avg_chunk_size:.0f} 字符")
        
        # 按文档统计
        doc_stats = {}
        for chunk in all_chunks:
            doc_name = chunk['document_name']
            if doc_name not in doc_stats:
                doc_stats[doc_name] = {'chunks': 0, 'chars': 0}
            doc_stats[doc_name]['chunks'] += 1
            doc_stats[doc_name]['chars'] += len(chunk['content'])
        
        print(f"\n按文档统计:")
        for doc_name, stats in doc_stats.items():
            print(f"  {doc_name}: {stats['chunks']} 块, {stats['chars']:,} 字符")
        
        # 保存结果到文件
        print(f"\n保存结果到文件...")
        output_file = f"pdf_parser_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_results_to_file(all_chunks, pdf_files, output_file)
        
        # 同时保存JSON格式的详细数据
        json_file = f"pdf_parser_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            print(f"✓ 详细数据已保存到JSON文件: {json_file}")
        except Exception as e:
            print(f"✗ 保存JSON文件失败: {e}")
        
        print("\n✓ PDF解析器测试完成！")
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请安装pdfplumber: pip install pdfplumber")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_parser()
