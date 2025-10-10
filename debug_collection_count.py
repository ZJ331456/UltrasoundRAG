#!/usr/bin/env python3
"""
调试集合文档数量获取问题
"""

def test_direct_milvus_connection():
    """直接测试Milvus连接和数据查询"""
    print("🔧 直接测试Milvus数据库连接...")
    
    try:
        from ultrasoundrag.data.stores.milvus_store import MilvusManager
        
        collections = [
            ("normal_book_md", "md"),
            ("normal_book_image", "image"), 
            ("thyroid_agent_md", "md"),
            ("health_check_test", "md")
        ]
        
        for collection_name, collection_type in collections:
            print(f"\n📊 测试集合: {collection_name} (类型: {collection_type})")
            
            try:
                manager = MilvusManager(
                    collection_type=collection_type, 
                    collection_name=collection_name
                )
                
                # 测试1: 使用get_collection_info_fast
                print("  方法1: get_collection_info_fast")
                fast_info = manager.get_collection_info_fast()
                print(f"    结果: {fast_info}")
                
                # 测试2: 使用修复后的get_collection_info
                print("  方法2: get_collection_info (修复后)")
                info = manager.get_collection_info()
                print(f"    结果: {info}")
                
                # 测试3: 直接使用Milvus客户端
                print("  方法3: 直接查询")
                try:
                    count = manager.client.num_entities(collection_name=collection_name)
                    print(f"    实体数量: {count}")
                except Exception as e:
                    print(f"    num_entities失败: {e}")
                    
                    try:
                        stats = manager.client.get_collection_statistics(collection_name=collection_name)
                        print(f"    统计信息: {stats}")
                    except Exception as e2:
                        print(f"    get_collection_statistics失败: {e2}")
                
                # 测试4: 查询一条记录
                print("  方法4: 查询测试")
                try:
                    result = manager.client.query(
                        collection_name=collection_name,
                        filter="",
                        output_fields=["id"],
                        limit=5
                    )
                    print(f"    查询结果: 找到 {len(result) if result else 0} 条记录")
                    if result:
                        print(f"    样本数据: {result[:2]}")
                except Exception as e:
                    print(f"    查询失败: {e}")
                    
            except Exception as e:
                print(f"  ❌ 创建管理器失败: {e}")
                
    except Exception as e:
        print(f"❌ 导入失败: {e}")

def test_collection_listing():
    """测试集合列表获取"""
    print("\n🗂️ 测试集合列表获取...")
    
    try:
        from ultrasoundrag.core.indexing import list_collections
        collections = list_collections()
        print(f"找到集合: {collections}")
        
        from ultrasoundrag.core.indexing import get_collection_info
        for collection_name in collections:
            print(f"\n📋 获取 {collection_name} 信息:")
            info = get_collection_info(collection_name)
            print(f"  信息: {info}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    print("🚀 调试集合文档数量获取问题")
    print("=" * 60)
    
    test_direct_milvus_connection()
    test_collection_listing()
    
    print("\n" + "=" * 60)
    print("🎯 调试完成")
