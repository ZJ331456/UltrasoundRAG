#!/usr/bin/env python3
"""
清除除default之外的所有Milvus数据库
保留default数据库，删除其他所有数据库和集合
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "UltrasoundRAG"))

from UltrasoundRAG.config import config
from pymilvus import connections, db, utility


def clear_non_default_databases():
    """清除除default之外的所有数据库"""
    print("🧹 清除除default之外的所有Milvus数据库")
    print("=" * 60)
    
    try:
        # 连接Milvus
        milvus_config = config['milvus']
        milvus_uri = milvus_config['milvus_uri']
        milvus_token = milvus_config['milvus_token']
        
        # 解析URI
        if "://" in milvus_uri:
            host = milvus_uri.split("://")[1].split(":")[0]
            port = int(milvus_uri.split("://")[1].split(":")[1]) if ":" in milvus_uri.split("://")[1] else 19530
        else:
            host = milvus_uri.split(":")[0] if ":" in milvus_uri else milvus_uri
            port = int(milvus_uri.split(":")[1]) if ":" in milvus_uri else 19530
        
        print(f"🔗 连接到Milvus: {host}:{port}")
        connections.connect(
            alias="default",
            host=host,
            port=port,
            user="root",
            password=milvus_token if milvus_token else ""
        )
        print("✅ 连接成功")
        
        # 获取所有数据库
        all_databases = db.list_database()
        print(f"\n📋 发现 {len(all_databases)} 个数据库: {all_databases}")
        
        # 过滤出非default数据库
        non_default_databases = [db_name for db_name in all_databases if db_name != "default"]
        
        if not non_default_databases:
            print("ℹ️  除了default数据库外，没有其他数据库需要清理")
            return
        
        print(f"🎯 将清理 {len(non_default_databases)} 个非default数据库: {non_default_databases}")
        
        # 确认操作
        print("\n⚠️  警告：此操作将删除以下数据库及其所有集合：")
        for db_name in non_default_databases:
            print(f"   - {db_name}")
        
        confirm = input("\n确认继续？(输入 'YES' 确认): ").strip().upper()
        if confirm != "YES":
            print("❌ 操作已取消")
            return
        
        # 开始清理
        print(f"\n🚀 开始清理 {len(non_default_databases)} 个数据库...")
        
        success_count = 0
        total_collections_deleted = 0
        total_entities_deleted = 0
        
        for i, database in enumerate(non_default_databases, 1):
            print(f"\n[{i}/{len(non_default_databases)}] 🧹 清理数据库: {database}")
            print("-" * 40)
            
            try:
                # 切换到目标数据库
                db.using_database(database)
                
                # 获取所有集合
                collections = utility.list_collections()
                print(f"   📁 发现 {len(collections)} 个集合: {collections}")
                
                # 删除所有集合
                collections_deleted = 0
                entities_in_db = 0
                
                for collection in collections:
                    try:
                        # 获取集合统计信息（如果可能）
                        try:
                            stats = utility.get_collection_statistics(collection)
                            if isinstance(stats, dict) and 'row_count' in stats:
                                entity_count = stats['row_count']
                                entities_in_db += entity_count
                                print(f"      📊 集合 '{collection}' 包含 {entity_count:,} 个实体")
                        except:
                            print(f"      📊 集合 '{collection}' 统计信息不可用")
                        
                        # 删除集合
                        utility.drop_collection(collection)
                        print(f"      ✅ 删除集合: {collection}")
                        collections_deleted += 1
                        
                    except Exception as e:
                        print(f"      ❌ 删除集合 '{collection}' 失败: {e}")
                
                # 删除数据库
                try:
                    db.drop_database(database)
                    print(f"   ✅ 删除数据库: {database}")
                    print(f"   📊 删除了 {collections_deleted} 个集合，约 {entities_in_db:,} 个实体")
                    
                    success_count += 1
                    total_collections_deleted += collections_deleted
                    total_entities_deleted += entities_in_db
                    
                except Exception as e:
                    print(f"   ❌ 删除数据库 '{database}' 失败: {e}")
                
            except Exception as e:
                print(f"   ❌ 清理数据库 '{database}' 时发生错误: {e}")
        
        # 显示清理结果
        print("\n" + "=" * 60)
        print("🎉 清理完成！")
        print("=" * 60)
        print(f"✅ 成功清理数据库: {success_count}/{len(non_default_databases)}")
        print(f"📁 删除集合总数: {total_collections_deleted}")
        print(f"📊 删除实体总数: {total_entities_deleted:,}")
        
        # 显示剩余数据库
        remaining_databases = db.list_database()
        print(f"\n📋 剩余数据库: {remaining_databases}")
        
        if "default" in remaining_databases:
            print("✅ default数据库已保留")
        
    except Exception as e:
        print(f"❌ 清理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            connections.disconnect("default")
            print("\n🔌 已断开Milvus连接")
        except:
            pass


def show_current_status():
    """显示当前数据库状态"""
    print("🔍 当前Milvus数据库状态")
    print("=" * 40)
    
    try:
        # 连接Milvus
        milvus_config = config['milvus']
        milvus_uri = milvus_config['milvus_uri']
        milvus_token = milvus_config['milvus_token']
        
        # 解析URI
        if "://" in milvus_uri:
            host = milvus_uri.split("://")[1].split(":")[0]
            port = int(milvus_uri.split("://")[1].split(":")[1]) if ":" in milvus_uri.split("://")[1] else 19530
        else:
            host = milvus_uri.split(":")[0] if ":" in milvus_uri else milvus_uri
            port = int(milvus_uri.split(":")[1]) if ":" in milvus_uri else 19530
        
        connections.connect(
            alias="default",
            host=host,
            port=port,
            user="root",
            password=milvus_token if milvus_token else ""
        )
        
        # 获取所有数据库
        databases = db.list_database()
        print(f"📋 发现 {len(databases)} 个数据库:")
        
        for i, db_name in enumerate(databases, 1):
            status = "🔒 默认" if db_name == "default" else "🗑️  将被删除"
            print(f"  {i}. {db_name} {status}")
            
            if db_name != "default":
                try:
                    db.using_database(db_name)
                    collections = utility.list_collections()
                    print(f"     📁 集合: {len(collections)} 个 {collections}")
                except:
                    print(f"     ❌ 无法访问")
        
        print(f"\n🎯 将清理 {len([d for d in databases if d != 'default'])} 个非default数据库")
        
    except Exception as e:
        print(f"❌ 无法获取状态: {e}")
    finally:
        try:
            connections.disconnect("default")
        except:
            pass


def main():
    """主函数"""
    print("🧹 Milvus数据库清理工具 (保留default)")
    print("=" * 50)
    
    while True:
        print("\n请选择操作:")
        print("1. 查看当前数据库状态")
        print("2. 清除除default之外的所有数据库")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-2): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            show_current_status()
        elif choice == "2":
            clear_non_default_databases()
        else:
            print("❌ 无效选择，请重试")


if __name__ == "__main__":
    main()
