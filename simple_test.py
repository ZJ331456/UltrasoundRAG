#!/usr/bin/env python3
"""
UltrasoundRAG 简单命令行测试界面
支持交互式检索测试

使用方法：
python simple_test.py
"""

import os
import sys
import time
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "UltrasoundRAG"))

try:
    from UltrasoundRAG.retrival.modular_retrievers import (
        create_t2t_retriever, create_t2i_retriever,
        create_i2t_retriever, create_i2i_retriever
    )
    from UltrasoundRAG.retrival.caption_to_image_retriever import create_caption_retriever
    from UltrasoundRAG.utils.logger import setup_logger
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    sys.exit(1)

# 初始化日志
logger = setup_logger("SimpleTest")

def print_separator():
    """打印分隔线"""
    print("=" * 60)

def print_results(results: List[Dict], result_type: str = "结果", max_results: int = 5):
    """打印检索结果"""
    if not results:
        print(f"❌ 没有找到{result_type}")
        return
    
    print(f"✅ 找到 {len(results)} 个{result_type}")
    print("-" * 40)
    
    for i, item in enumerate(results[:max_results]):
        print(f"\n{i+1}. 分数: {getattr(item, 'score', 0):.4f}")
        print(f"   内容: {getattr(item, 'content', '')[:100]}{'...' if len(getattr(item, 'content', '')) > 100 else ''}")
        
        metadata = getattr(item, 'metadata', {})
        if 'relative_path' in metadata:
            print(f"   路径: {metadata['relative_path']}")
        if 'document_name' in metadata:
            print(f"   文档: {metadata['document_name']}")

def test_t2t_retrieval(retriever, query: str, top_k: int = 5):
    """测试T2T检索"""
    print(f"\n🔍 T2T检索: '{query}'")
    print("-" * 30)
    
    start_time = time.time()
    result = retriever.search(query, top_k=top_k)
    end_time = time.time()
    
    print(f"⏱️  耗时: {end_time - start_time:.3f} 秒")
    print_results(result.get('results', []), "文本结果", top_k)

def test_t2i_retrieval(retriever, query: str, top_k: int = 5):
    """测试T2I检索"""
    print(f"\n🖼️  T2I检索: '{query}'")
    print("-" * 30)
    
    start_time = time.time()
    result = retriever.search(query, top_k=top_k)
    end_time = time.time()
    
    print(f"⏱️  耗时: {end_time - start_time:.3f} 秒")
    print_results(result.get('results', []), "图片结果", top_k)

def test_caption_retrieval(retriever, caption: str, top_k: int = 5):
    """测试Caption检索"""
    print(f"\n🏷️  Caption检索: '{caption}'")
    print("-" * 30)
    
    start_time = time.time()
    result = retriever.search_single_caption(caption, top_k=top_k)
    end_time = time.time()
    
    print(f"⏱️  耗时: {end_time - start_time:.3f} 秒")
    print(f"✅ 找到 {result.get('total_results', 0)} 个匹配图片")
    
    for i, res in enumerate(result.get('results', [])[:top_k]):
        print(f"\n{i+1}. 分数: {getattr(res, 'score', 0):.4f}")
        metadata = getattr(res, 'metadata', {})
        if 'relative_path' in metadata:
            print(f"   图片: {metadata['relative_path']}")
        if 'caption' in metadata:
            print(f"   标题: {metadata['caption']}")

def test_text_chunk_retrieval(retriever, text_chunk: str, top_k: int = 5):
    """测试从文本块提取Caption检索"""
    print(f"\n📝 文本块Caption提取检索")
    print("-" * 30)
    print(f"文本: {text_chunk[:100]}{'...' if len(text_chunk) > 100 else ''}")
    
    start_time = time.time()
    result = retriever.search_from_text_chunk(text_chunk, top_k=top_k)
    end_time = time.time()
    
    print(f"⏱️  耗时: {end_time - start_time:.3f} 秒")
    
    extracted_captions = result.get('extracted_captions', [])
    if extracted_captions:
        print(f"✅ 提取到Caption: {', '.join(extracted_captions)}")
    else:
        print("❌ 未提取到Caption")
    
    print(f"✅ 找到 {result.get('total_results', 0)} 个相关图片")
    
    for i, res in enumerate(result.get('results', [])[:top_k]):
        print(f"\n{i+1}. 分数: {getattr(res, 'score', 0):.4f}")
        metadata = getattr(res, 'metadata', {})
        if 'relative_path' in metadata:
            print(f"   图片: {metadata['relative_path']}")

def main():
    """主函数"""
    print_separator()
    print("🔍 UltrasoundRAG 简单检索测试界面")
    print_separator()
    
    # 初始化检索器
    print("🚀 正在初始化检索器...")
    try:
        retrievers = {
            't2t': create_t2t_retriever("default", top_k=10),
            't2i': create_t2i_retriever("default", top_k=10),
            'caption': create_caption_retriever("default", "hybrid_match")
        }
        print("✅ 检索器初始化完成")
    except Exception as e:
        print(f"❌ 检索器初始化失败: {e}")
        return
    
    print_separator()
    
    # 预定义测试用例
    test_cases = {
        "1": {
            "name": "T2T检索测试",
            "queries": [
                "心脏超声检查方法",
                "肝脏病变诊断",
                "胎儿发育评估"
            ]
        },
        "2": {
            "name": "T2I检索测试", 
            "queries": [
                "心脏四腔心切面图",
                "肝脏超声图像",
                "胎儿发育图"
            ]
        },
        "3": {
            "name": "Caption检索测试",
            "captions": [
                "图2-3 心脏超声横切面",
                "图1-1 肝脏超声检查",
                "Figure 3.2 胎儿发育图"
            ]
        },
        "4": {
            "name": "文本块Caption提取测试",
            "text_chunks": [
                "心脏超声检查包括多个切面。图2-1显示四腔心切面，图2-2为心脏短轴切面。",
                "肝脏检查中，图1-1显示正常肝脏，图1-2显示肝脏病变。",
                "胎儿发育评估包括图3-1的头部测量和图3-2的四肢发育。"
            ]
        }
    }
    
    while True:
        print("\n📋 请选择测试类型:")
        print("1. T2T检索测试 (文本→文本)")
        print("2. T2I检索测试 (文本→图片)")
        print("3. Caption检索测试 (标题→图片)")
        print("4. 文本块Caption提取测试")
        print("5. 自定义查询")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice in test_cases:
            test_case = test_cases[choice]
            print(f"\n🧪 开始{test_case['name']}")
            print_separator()
            
            if choice == "1":
                for i, query in enumerate(test_case['queries']):
                    test_t2t_retrieval(retrievers['t2t'], query)
                    if i < len(test_case['queries']) - 1:
                        try:
                            input("\n按回车键继续下一个测试...")
                        except EOFError:
                            print("\n继续下一个测试...")
            
            elif choice == "2":
                for i, query in enumerate(test_case['queries']):
                    test_t2i_retrieval(retrievers['t2i'], query)
                    if i < len(test_case['queries']) - 1:
                        try:
                            input("\n按回车键继续下一个测试...")
                        except EOFError:
                            print("\n继续下一个测试...")
            
            elif choice == "3":
                for i, caption in enumerate(test_case['captions']):
                    test_caption_retrieval(retrievers['caption'], caption)
                    if i < len(test_case['captions']) - 1:
                        try:
                            input("\n按回车键继续下一个测试...")
                        except EOFError:
                            print("\n继续下一个测试...")
            
            elif choice == "4":
                for i, text_chunk in enumerate(test_case['text_chunks']):
                    test_text_chunk_retrieval(retrievers['caption'], text_chunk)
                    if i < len(test_case['text_chunks']) - 1:
                        try:
                            input("\n按回车键继续下一个测试...")
                        except EOFError:
                            print("\n继续下一个测试...")
        
        elif choice == "5":
            print("\n🔧 自定义查询")
            print("1. T2T查询")
            print("2. T2I查询") 
            print("3. Caption查询")
            print("4. 文本块查询")
            
            sub_choice = input("请选择查询类型 (1-4): ").strip()
            query = input("请输入查询内容: ").strip()
            
            if not query:
                print("❌ 查询内容不能为空")
                continue
            
            if sub_choice == "1":
                test_t2t_retrieval(retrievers['t2t'], query)
            elif sub_choice == "2":
                test_t2i_retrieval(retrievers['t2i'], query)
            elif sub_choice == "3":
                test_caption_retrieval(retrievers['caption'], query)
            elif sub_choice == "4":
                test_text_chunk_retrieval(retrievers['caption'], query)
            else:
                print("❌ 无效选择")
        
        else:
            print("❌ 无效选择，请重新输入")
        
        print_separator()

if __name__ == "__main__":
    main()
