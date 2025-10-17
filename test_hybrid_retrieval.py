#!/usr/bin/env python3
"""
BM25混合检索功能测试脚本

测试流程：
1. 创建测试集合
2. 插入测试数据（带BM25元数据）
3. 执行混合检索
4. 对比不同检索策略的结果
"""

import sys
import time
from ultrasoundrag.core.retrieval.modular_retrievers import T2TRetriever, RetrievalContext
from ultrasoundrag.data.stores.milvus_store import MilvusManager
from ultrasoundrag.data.processors.bm25_encoder import get_bm25_encoder
from pymilvus import MilvusClient

def create_test_collection():
    """创建测试集合"""
    print("\n" + "="*60)
    print("步骤1: 创建测试集合")
    print("="*60)
    
    # 使用本地数据库URI
    milvus_uri = './milvus_ultrasound.db'
    collection_name = 'hybrid_test_collection'
    
    client = MilvusClient(uri=milvus_uri)
    
    # 删除旧集合
    if collection_name in client.list_collections():
        client.drop_collection(collection_name)
        print(f"✅ 已删除旧集合: {collection_name}")
    
    # 创建新集合（使用相同的URI）
    manager = MilvusManager(
        milvus_uri=milvus_uri,
        milvus_token=None,
        db_name='default',
        collection_type='md',
        collection_name=collection_name
    )
    
    print(f"✅ 测试集合创建成功: {collection_name}")
    print(f"   数据库URI: {milvus_uri}")
    return manager, collection_name, milvus_uri

def prepare_test_data():
    """准备测试数据"""
    print("\n" + "="*60)
    print("步骤2: 准备测试数据")
    print("="*60)
    
    test_docs = [
        {
            'id': 1,
            'title': '胎儿超声检查指南',
            'content': '胎儿超声检查是产前诊断的重要手段，可以观察胎儿的生长发育情况，评估胎儿的健康状况。常规检查包括胎儿头部、胸部、腹部、四肢和脊柱等部位。',
            'document_name': '产前诊断指南',
            'chunk_index': 0,
            'file': 'guide_prenatal.md',
            'image_paths': [],
            'image_captions': [],
            'domain': '产科',
            'is_deleted': 0
        },
        {
            'id': 2,
            'title': '胎儿心脏超声',
            'content': '胎儿心脏超声检查是评估胎儿心血管系统的重要方法。通过超声可以观察心脏四腔心、大动脉、房间隔、室间隔等结构，早期发现先天性心脏病。',
            'document_name': '心脏超声手册',
            'chunk_index': 0,
            'file': 'cardiac_ultrasound.md',
            'image_paths': [],
            'image_captions': [],
            'domain': '产科',
            'is_deleted': 0
        },
        {
            'id': 3,
            'title': '超声设备使用',
            'content': '超声设备包括探头、主机、显示器等部分。使用前需要检查设备状态，调整增益、深度、焦点等参数，以获得清晰的超声图像。',
            'document_name': '设备操作手册',
            'chunk_index': 0,
            'file': 'equipment_manual.md',
            'image_paths': [],
            'image_captions': [],
            'domain': '技术',
            'is_deleted': 0
        },
        {
            'id': 4,
            'title': '妊娠早期超声',
            'content': '妊娠早期超声检查主要用于确认宫内妊娠、估算孕周、观察胚胎发育。可以测量头臀长（CRL）、观察胎心搏动、评估卵黄囊等。',
            'document_name': '早期妊娠诊断',
            'chunk_index': 0,
            'file': 'early_pregnancy.md',
            'image_paths': [],
            'image_captions': [],
            'domain': '产科',
            'is_deleted': 0
        },
        {
            'id': 5,
            'title': '超声图像诊断',
            'content': '超声图像诊断需要综合分析图像的灰阶、回声强度、边界清晰度等特征。常见病理表现包括囊性病变、实性肿块、钙化灶等。',
            'document_name': '影像诊断学',
            'chunk_index': 0,
            'file': 'image_diagnosis.md',
            'image_paths': [],
            'image_captions': [],
            'domain': '诊断',
            'is_deleted': 0
        }
    ]
    
    print(f"✅ 准备了 {len(test_docs)} 条测试文档")
    return test_docs

def insert_test_data(manager, test_docs):
    """插入测试数据"""
    print("\n" + "="*60)
    print("步骤3: 插入测试数据")
    print("="*60)
    
    try:
        result = manager.insert_data(test_docs)
        # insert_data返回布尔值
        if result:
            print(f"✅ 成功插入 {len(test_docs)} 条数据")
            return True
        else:
            print(f"❌ 插入失败")
            return False
    except Exception as e:
        print(f"❌ 插入异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_strategies(collection_name, milvus_uri):
    """测试不同检索策略"""
    print("\n" + "="*60)
    print("步骤4: 测试不同检索策略")
    print("="*60)
    
    # 创建检索上下文（使用相同的URI）
    context = RetrievalContext(
        db_name='default',
        text_collection=collection_name,
        image_collection=None,
        top_k=3,
        milvus_uri=milvus_uri,
        milvus_token=None
    )
    
    # 创建检索器
    retriever = T2TRetriever(context)
    
    # 测试查询
    test_queries = [
        "胎儿心脏",  # 短查询：BM25权重应该更高
        "如何使用超声设备进行胎儿检查",  # 长查询：向量权重应该更高
        "妊娠早期超声检查"  # 中等查询：平衡权重
    ]
    
    strategies = ['enhanced', 'hybrid']
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}\n")
        
        for strategy in strategies:
            print(f"--- 策略: {strategy} ---")
            start_time = time.time()
            
            try:
                result = retriever.search(query, top_k=3, strategy=strategy)
                elapsed = time.time() - start_time
                
                print(f"检索耗时: {elapsed*1000:.2f}ms")
                print(f"返回结果数: {result['total_results']}")
                
                if 'weights_used' in result:
                    weights = result['weights_used']
                    print(f"权重配置: {weights}")
                
                if result['results']:
                    print("\n结果列表:")
                    for i, res in enumerate(result['results'][:3], 1):
                        print(f"  {i}. {res.title} (分数: {res.score:.4f})")
                        if hasattr(res, 'metadata') and res.metadata:
                            if 'bm25_score' in res.metadata:
                                print(f"     - 向量分数: {res.metadata.get('vector_score', 0):.4f}")
                                print(f"     - BM25分数: {res.metadata.get('bm25_score', 0):.4f}")
                else:
                    print("  无结果")
                
                print()
                
            except Exception as e:
                print(f"❌ 检索失败: {e}")
                import traceback
                traceback.print_exc()
                print()

def main():
    """主测试流程"""
    print("\n" + "="*70)
    print(" "*20 + "BM25混合检索测试")
    print("="*70)
    
    try:
        # 1. 创建测试集合
        manager, collection_name, milvus_uri = create_test_collection()
        
        # 2. 准备测试数据
        test_docs = prepare_test_data()
        
        # 3. 插入测试数据
        if not insert_test_data(manager, test_docs):
            print("\n❌ 数据插入失败，测试终止")
            return
        
        # 等待索引构建
        print("\n⏳ 等待索引构建...")
        time.sleep(2)
        
        # 4. 测试检索策略
        test_retrieval_strategies(collection_name, milvus_uri)
        
        print("\n" + "="*70)
        print(" "*25 + "测试完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

