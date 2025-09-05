"""
UltrasoundRAG主入口 - 索引构建和检索系统
支持索引重构、多种检索方式、多数据库管理

主要功能：
1. 索引构建和重构
2. 四种检索模式：T2T、T2I、I2T、I2I
3. Caption检索图片
4. 多数据库支持
5. 检索性能测试
"""

import argparse
import random
import torch
import numpy as np
import time
from typing import List, Dict, Any, Optional
from UltrasoundRAG.config import config
from UltrasoundRAG.utils.embedding_utils import embedding_provider
from UltrasoundRAG.index.markdown_parse import MarkdownParser
from UltrasoundRAG.index.image_parse import ImageParser
from UltrasoundRAG.milvus.milvus_manager import MilvusManager
from UltrasoundRAG.retrival.modular_retrievers import (
    create_t2t_retriever, create_t2i_retriever,
    create_i2t_retriever, create_i2i_retriever
)
from UltrasoundRAG.retrival.caption_to_image_retriever import create_caption_retriever
from UltrasoundRAG.retrival.multi_database_manager import (
    create_multi_database_manager, create_retrieval_strategy
)
from UltrasoundRAG.utils.logger import setup_logger
from tqdm import tqdm

def set_seed(seed: int) -> None:
    """设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_markdown_index(recreate: bool = False, only_datasets: list | None = None) -> None:
    """构建Markdown索引，可选重建集合与数据集过滤"""
    print("=" * 50)
    print("开始构建 Markdown 索引")
    markdown_cfg = config['indexing']['markdown']

    manager = MilvusManager(collection_type="md")
    if recreate:
        print("重建 Markdown 集合: 先删除再创建")
        manager.drop_collection()
        # 重新创建集合
        manager._setup_collection()

    built_count = 0
    skipped_count = 0
    global_id = 1  # 跨数据集统一分配主键，避免冲突

    datasets_items = list(markdown_cfg['datasets'].items())
    for dataset_name, dataset_cfg in tqdm(datasets_items, desc="Markdown 数据集", unit="ds"):
        if only_datasets and dataset_name not in only_datasets:
            continue
        if dataset_cfg.get('enabled', False):
            print(f"\n处理 Markdown 数据集: {dataset_name}")
            parser = MarkdownParser(dataset_name=dataset_name)
            parsed_data = parser.parse_markdowns()
            if not parsed_data:
                print(f"数据集 {dataset_name} 没有找到数据，跳过")
                skipped_count += 1
                continue
            # 统一重写 id，避免多个数据集启用时的主键冲突
            for item in parsed_data:
                item['id'] = int(global_id)
                global_id += 1

            texts = [chunk['content'] for chunk in parsed_data]
            # 生成两路向量：Qwen3(1024) 与 FetalCLIP 文本通道(768)
            embedder_qwen = embedding_provider[config['embedding']['provider']]
            embeddings_qwen = []
            batch_size_embed = 64
            for i in tqdm(range(0, len(texts), batch_size_embed), desc="Qwen 文本嵌入", unit="batch"):
                batch_texts = texts[i:i + batch_size_embed]
                try:
                    batch_emb = embedder_qwen.embed_documents(batch_texts)
                except Exception as e:
                    print(f"Qwen 嵌入失败批次 {i // batch_size_embed}: {e}")
                    batch_emb = [[0.0] * 1024 for _ in batch_texts]
                embeddings_qwen.extend(batch_emb)

            # 使用 FetalCLIP 的文本编码通道生成 768 维向量（与图像向量空间一致）
            try:
                from UltrasoundRAG.model.fetal_clip_model import FetalCLIPModel
                clip_model = FetalCLIPModel(
                    model_path=config['indexing']['image_parse']['model_path'],
                    config_path=config['indexing']['image_parse']['model_config_path']
                )
                embeddings_clip = []
                for i in tqdm(range(0, len(texts), batch_size_embed), desc="CLIP 文本嵌入", unit="batch"):
                    batch_texts = texts[i:i + batch_size_embed]
                    tokens = clip_model.tokenize_text(batch_texts)
                    feats = clip_model.encode_text(tokens).cpu().numpy()
                    for row in feats:
                        vec = row.tolist()
                        if len(vec) != 768:
                            vec = [0.0] * 768
                        embeddings_clip.append(vec)
            except Exception as e:
                print(f"生成CLIP文本向量失败: {e}，使用零向量")
                embeddings_clip = [[0.0]*768 for _ in texts]

            # 分批写入，显示进度
            insert_chunk = 1000
            insert_ok = True
            for i in tqdm(range(0, len(parsed_data), insert_chunk), desc="写入 Milvus", unit="chunk"):
                part_data = parsed_data[i:i + insert_chunk]
                part_qwen = embeddings_qwen[i:i + insert_chunk]
                part_clip = embeddings_clip[i:i + insert_chunk]
                ok = manager.insert_data(part_data, embeddings_qwen=part_qwen, embeddings_clip=part_clip)
                insert_ok = insert_ok and ok
            success = insert_ok
            if success:
                built_count += 1
                print(f"✓ 数据集 {dataset_name} 索引构建成功")
            else:
                print(f"✗ 数据集 {dataset_name} 索引构建失败")
        else:
            skipped_count += 1

    print(f"Markdown 构建完成：成功 {built_count}，跳过 {skipped_count}")
    try:
        manager.get_collection_info()
    except Exception:
        pass

def build_image_index(recreate: bool = False, only_datasets: list | None = None) -> None:
    """构建图片索引，可选重建集合与数据集过滤"""
    print("=" * 50)
    print("开始构建图片索引")
    image_cfg = config['indexing']['image']

    manager = MilvusManager(collection_type="image")
    if recreate:
        print("重建 图片 集合: 先删除再创建")
        manager.drop_collection()
        manager._setup_collection()

    built_count = 0
    skipped_count = 0
    global_id = 1  # 跨数据集统一分配主键，避免冲突

    datasets_items = list(image_cfg['datasets'].items())
    for dataset_name, dataset_cfg in tqdm(datasets_items, desc="图片 数据集", unit="ds"):
        if only_datasets and dataset_name not in only_datasets:
            continue
        if dataset_cfg.get('enabled', False):
            print(f"\n处理图片数据集: {dataset_name}")
            parser = ImageParser(dataset_name=dataset_name)
            parsed_data = parser.parse_images()
            if not parsed_data:
                print(f"数据集 {dataset_name} 没有找到数据，跳过")
                skipped_count += 1
                continue
            # 统一重写 id，避免多个数据集启用时的主键冲突
            for item in parsed_data:
                item['id'] = int(global_id)
                global_id += 1

            embeddings = [item['image_vector'] for item in parsed_data]
            insert_chunk = 1000
            insert_ok = True
            for i in tqdm(range(0, len(parsed_data), insert_chunk), desc="写入 Milvus(图片)", unit="chunk"):
                part_data = parsed_data[i:i + insert_chunk]
                part_emb = embeddings[i:i + insert_chunk]
                ok = manager.insert_data(part_data, part_emb)
                insert_ok = insert_ok and ok
            success = insert_ok
            if success:
                built_count += 1
                print(f"✓ 数据集 {dataset_name} 索引构建成功")
            else:
                print(f"✗ 数据集 {dataset_name} 索引构建失败")
        else:
            skipped_count += 1

    print(f"图片 构建完成：成功 {built_count}，跳过 {skipped_count}")
    try:
        manager.get_collection_info()
    except Exception:
        pass

def test_retrieval_modes(db_name: str = "default", top_k: int = 3) -> None:
    """测试四种检索模式"""
    logger = setup_logger("RetrievalTest")
    
    print("=" * 60)
    print("开始测试多种检索模式")
    print("=" * 60)
    
    # 1. T2T检索测试
    print("\n1. 文本到文本检索（T2T）测试:")
    print("-" * 40)
    try:
        t2t_retriever = create_t2t_retriever(db_name, top_k)
        t2t_result = t2t_retriever.search("心脏超声检查方法", top_k=top_k)
        
        print(f"查询: '{t2t_result.get('query', '')}'")
        print(f"策略: {t2t_result.get('strategy', 'basic')}")
        print(f"找到 {t2t_result.get('total_results', 0)} 个文本结果")
        
        for i, result in enumerate(t2t_result.get('results', [])[:3]):
            print(f"  {i+1}. 分数: {result.score:.4f}")
            print(f"      内容: {result.content[:80]}...")
            print(f"      来源: {result.metadata.get('document_name', 'N/A')}")
            print()
    except Exception as e:
        logger.error(f"T2T检索测试失败: {e}")
    
    # 2. T2I检索测试
    print("\n2. 文本到图片检索（T2I）测试:")
    print("-" * 40)
    try:
        t2i_retriever = create_t2i_retriever(db_name, top_k)
        t2i_result = t2i_retriever.search("心脏四腔心切面图", top_k=top_k)
        
        print(f"查询: '{t2i_result.get('query', '')}'")
        print(f"找到 {t2i_result.get('total_results', 0)} 个图片结果")
        
        for i, result in enumerate(t2i_result.get('results', [])[:3]):
            print(f"  {i+1}. 分数: {result.score:.4f}")
            print(f"      图片: {result.metadata.get('relative_path', 'N/A')}")
            print(f"      标题: {result.content[:60]}...")
            print()
    except Exception as e:
        logger.error(f"T2I检索测试失败: {e}")
    
    # 3. Caption检索图片测试
    print("\n3. Caption检索图片测试:")
    print("-" * 40)
    try:
        caption_retriever = create_caption_retriever(db_name, "hybrid_match")
        
        # 单个caption检索
        caption_result = caption_retriever.search_single_caption("图2-3 心脏超声横切面", top_k=top_k)
        print(f"Caption查询: '{caption_result.get('caption', '')}'")
        print(f"找到 {caption_result.get('total_results', 0)} 个匹配图片")
        
        for i, result in enumerate(caption_result.get('results', [])[:3]):
            print(f"  {i+1}. 分数: {result.score:.4f}")
            print(f"      图片: {result.metadata.get('relative_path', 'N/A')}")
            print(f"      匹配类型: {result.metadata.get('search_type', 'N/A')}")
            print()
        
        # 从文本块提取caption检索
        text_chunk = "图2-1显示心脏四腔心切面，图2-2为心脏短轴切面"
        text_result = caption_retriever.search_from_text_chunk(text_chunk, top_k=2)
        print(f"\n从文本提取检索:")
        print(f"提取到caption: {text_result.get('extracted_captions', [])}")
        print(f"找到 {text_result.get('total_results', 0)} 个相关图片")
        
    except Exception as e:
        logger.error(f"Caption检索测试失败: {e}")
    
    print("\n注意: I2T和I2I检索需要提供实际的图片路径进行测试")


def test_multi_database_retrieval(query: str = "心脏超声诊断", top_k: int = 5) -> None:
    """测试多数据库检索功能"""
    logger = setup_logger("MultiDBTest")
    
    print("=" * 60)
    print("开始测试多数据库检索功能")
    print("=" * 60)
    
    try:
        # 创建多数据库管理器
        manager = create_multi_database_manager()
        
        # 获取可用数据库
        available_dbs = manager.get_available_databases()
        print(f"\n可用数据库 ({len(available_dbs)} 个):")
        for db in available_dbs:
            print(f"  - {db.name}: {db.description} (优先级: {db.priority})")
        
        if not available_dbs:
            print("没有可用的数据库配置")
            return
        
        # 单数据库检索测试
        print(f"\n1. 单数据库检索测试 (数据库: {available_dbs[0].name}):")
        print("-" * 50)
        single_result = manager.search_single_database(
            available_dbs[0].name, "t2t", query, top_k=top_k
        )
        print(f"查询: '{query}'")
        print(f"数据库: {single_result.get('database')}")
        print(f"找到结果: {single_result.get('total_results', 0)}")
        
        # 自动选择数据库测试
        print(f"\n2. 自动选择数据库测试:")
        print("-" * 50)
        auto_selected = manager.auto_select_databases(query, "t2t", num_databases=2)
        print(f"为查询 '{query}' 自动选择的数据库: {auto_selected}")
        
        # 多数据库检索测试（如果有多个数据库）
        if len(available_dbs) >= 2:
            print(f"\n3. 多数据库检索测试:")
            print("-" * 50)
            strategy = create_retrieval_strategy(
                [db.name for db in available_dbs[:2]], 
                aggregation_method="weighted",
                max_results=top_k
            )
            multi_result = manager.search_multiple_databases(
                strategy, "t2t", query, top_k=top_k//2
            )
            print(f"检索策略: {strategy.aggregation_method}")
            print(f"搜索数据库: {multi_result.get('total_databases_searched', 0)}")
            print(f"聚合结果: {multi_result.get('total_results', 0)}")
            print(f"响应时间: {multi_result.get('response_time', 0):.3f}秒")
        
        # 数据库统计信息
        print(f"\n4. 数据库统计信息:")
        print("-" * 50)
        stats = manager.get_database_stats()
        print(f"总数据库数: {stats['total_databases']}")
        print(f"启用数据库数: {stats['enabled_databases']}")
        print(f"总查询次数: {stats['usage_stats']['total_queries']}")
        print(f"平均响应时间: {stats['usage_stats']['avg_response_time']:.3f}秒")
        
    except Exception as e:
        logger.error(f"多数据库检索测试失败: {e}")


def benchmark_retrieval_performance(queries: List[str], top_k: int = 10, 
                                  db_name: str = "default") -> None:
    """检索性能基准测试"""
    logger = setup_logger("BenchmarkTest")
    
    print("=" * 60)
    print("开始检索性能基准测试")
    print("=" * 60)
    
    if not queries:
        queries = [
            "心脏超声检查方法",
            "肝脏病变诊断",
            "胎儿发育评估",
            "血管多普勒检查",
            "肾脏结石诊断"
        ]
    
    retriever_types = ["t2t", "t2i"]
    results = {}
    
    for retriever_type in retriever_types:
        print(f"\n测试 {retriever_type.upper()} 检索性能:")
        print("-" * 40)
        
        # 创建检索器
        if retriever_type == "t2t":
            retriever = create_t2t_retriever(db_name, top_k)
        elif retriever_type == "t2i":
            retriever = create_t2i_retriever(db_name, top_k)
        else:
            continue
        
        times = []
        total_results = []
        
        for i, query in enumerate(queries):
            start_time = time.time()
            
            try:
                result = retriever.search(query, top_k=top_k)
                response_time = time.time() - start_time
                
                times.append(response_time)
                total_results.append(result.get('total_results', 0))
                
                print(f"  查询 {i+1}: {response_time:.3f}s, 结果数: {result.get('total_results', 0)}")
                
            except Exception as e:
                logger.error(f"查询 '{query}' 失败: {e}")
                times.append(0)
                total_results.append(0)
        
        # 计算统计信息
        avg_time = sum(times) / len(times) if times else 0
        avg_results = sum(total_results) / len(total_results) if total_results else 0
        max_time = max(times) if times else 0
        min_time = min(times) if times else 0
        
        results[retriever_type] = {
            'avg_time': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'avg_results': avg_results,
            'total_queries': len(queries)
        }
        
        print(f"  平均响应时间: {avg_time:.3f}s")
        print(f"  最长响应时间: {max_time:.3f}s")
        print(f"  最短响应时间: {min_time:.3f}s")
        print(f"  平均结果数: {avg_results:.1f}")
    
    # 性能对比
    print(f"\n性能对比总结:")
    print("-" * 40)
    for retriever_type, stats in results.items():
        print(f"{retriever_type.upper()}检索:")
        print(f"  平均响应时间: {stats['avg_time']:.3f}s")
        print(f"  平均结果数: {stats['avg_results']:.1f}")
        print(f"  查询吞吐量: {1/stats['avg_time']:.1f} 查询/秒")
        print()


def test_search(top_k: int = 3) -> None:
    """兼容性测试搜索功能（保留原有功能）"""
    print("=" * 50)
    print("开始基础搜索功能测试")
    
    # 调用新的检索测试
    test_retrieval_modes("default", top_k)
    
    print("\n=" * 50)
    print("基础搜索功能测试完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="UltrasoundRAG 索引构建与多种检索系统")
    
    # 索引构建相关参数
    indexing_group = parser.add_argument_group('索引构建选项')
    indexing_group.add_argument("--build-md", action="store_true", help="构建 Markdown 索引")
    indexing_group.add_argument("--build-image", action="store_true", help="构建图片索引")
    indexing_group.add_argument("--recreate", action="store_true", help="重建集合（先删除后创建）")
    indexing_group.add_argument("--md-datasets", type=str, nargs='*', default=None, 
                               help="仅构建指定的 Markdown 数据集名（空则按配置 enabled）")
    indexing_group.add_argument("--image-datasets", type=str, nargs='*', default=None, 
                               help="仅构建指定的图片数据集名（空则按配置 enabled）")
    
    # 检索测试相关参数
    retrieval_group = parser.add_argument_group('检索测试选项')
    retrieval_group.add_argument("--test", action="store_true", help="执行基础检索测试")
    retrieval_group.add_argument("--test-modes", action="store_true", help="测试四种检索模式 (T2T, T2I, I2T, I2I)")
    retrieval_group.add_argument("--test-multidb", action="store_true", help="测试多数据库检索功能")
    retrieval_group.add_argument("--test-caption", action="store_true", help="测试Caption检索图片功能")
    retrieval_group.add_argument("--benchmark", action="store_true", help="执行检索性能基准测试")
    
    # 通用参数
    general_group = parser.add_argument_group('通用选项')
    general_group.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    general_group.add_argument("--top-k", type=int, default=3, help="测试返回 TopK，默认 3")
    general_group.add_argument("--db-name", type=str, default="default", help="使用的数据库名称，默认 'default'")
    general_group.add_argument("--query", type=str, help="自定义查询文本（用于测试）")
    general_group.add_argument("--image-path", type=str, help="图片路径（用于I2T和I2I测试）")
    
    # 性能测试参数
    performance_group = parser.add_argument_group('性能测试选项')
    performance_group.add_argument("--benchmark-queries", type=str, nargs='*', 
                                  help="自定义基准测试查询列表")
    performance_group.add_argument("--benchmark-count", type=int, default=10, 
                                  help="基准测试查询数量，默认 10")

    args = parser.parse_args()

    try:
        set_seed(args.seed)
        logger = setup_logger("UltrasoundRAG")
        
        # 确定是否需要构建索引
        has_build_flags = args.build_md or args.build_image
        has_test_flags = (args.test or args.test_modes or args.test_multidb or 
                         args.test_caption or args.benchmark)
        
        # 如果没有任何标志，默认执行构建
        if not has_build_flags and not has_test_flags:
            has_build_flags = True
            args.build_md = True
            args.build_image = True

        # 执行索引构建
        if has_build_flags:
            logger.info("开始索引构建...")
            
            if args.build_md or (not args.build_image and has_build_flags):
                build_markdown_index(recreate=args.recreate, only_datasets=args.md_datasets)
                
            if args.build_image or (not args.build_md and has_build_flags):
                build_image_index(recreate=args.recreate, only_datasets=args.image_datasets)

        # 执行检索测试
        if has_test_flags:
            logger.info("开始检索测试...")
            
            # 基础检索测试
            if args.test:
                test_search(top_k=args.top_k)
            
            # 四种检索模式测试
            if args.test_modes:
                test_retrieval_modes(db_name=args.db_name, top_k=args.top_k)
            
            # 多数据库检索测试
            if args.test_multidb:
                query = args.query or "心脏超声诊断"
                test_multi_database_retrieval(query=query, top_k=args.top_k)
            
            # Caption检索测试
            if args.test_caption:
                print("=" * 60)
                print("开始Caption检索图片专项测试")
                print("=" * 60)
                
                try:
                    caption_retriever = create_caption_retriever(args.db_name, "hybrid_match")
                    
                    # 测试单个caption
                    test_captions = [
                        "图2-3 心脏超声横切面",
                        "图1-1 肝脏超声检查",
                        "Figure 3.2 胎儿发育图"
                    ]
                    
                    for caption in test_captions:
                        print(f"\n测试Caption: '{caption}'")
                        result = caption_retriever.search_single_caption(caption, top_k=args.top_k)
                        print(f"找到 {result.get('total_results', 0)} 个匹配图片")
                    
                    # 测试从文本提取
                    text_chunk = """
                    心脏超声检查包括多个切面。图2-1显示四腔心切面，
                    图2-2为心脏短轴切面，图2-3展示了心尖四腔心切面。
                    """
                    print(f"\n从文本块提取caption测试:")
                    text_result = caption_retriever.search_from_text_chunk(text_chunk, top_k=2)
                    print(f"提取到 {len(text_result.get('extracted_captions', []))} 个caption")
                    print(f"找到 {text_result.get('total_results', 0)} 个相关图片")
                    
                except Exception as e:
                    logger.error(f"Caption检索测试失败: {e}")
            
            # 性能基准测试
            if args.benchmark:
                queries = args.benchmark_queries
                if not queries:
                    # 使用默认查询集
                    queries = [
                        "心脏超声检查方法",
                        "肝脏病变诊断",
                        "胎儿发育评估",
                        "血管多普勒检查",
                        "肾脏结石诊断",
                        "超声引导穿刺",
                        "腹部超声检查",
                        "妇科超声诊断",
                        "甲状腺超声检查",
                        "乳腺超声检查"
                    ]
                
                # 限制查询数量
                queries = queries[:args.benchmark_count]
                benchmark_retrieval_performance(queries, top_k=args.top_k, db_name=args.db_name)
            
            # 图片相关检索测试（如果提供了图片路径）
            if args.image_path:
                print("=" * 60)
                print("开始图片相关检索测试")
                print("=" * 60)
                
                try:
                    # I2T测试
                    print("\n1. 图片到文本检索（I2T）测试:")
                    i2t_retriever = create_i2t_retriever(args.db_name, args.top_k)
                    i2t_result = i2t_retriever.search(args.image_path, top_k=args.top_k)
                    print(f"找到 {i2t_result.get('total_results', 0)} 个相关文本")
                    
                    # I2I测试
                    print("\n2. 图片到图片检索（I2I）测试:")
                    i2i_retriever = create_i2i_retriever(args.db_name, args.top_k)
                    i2i_result = i2i_retriever.search(args.image_path, top_k=args.top_k)
                    print(f"找到 {i2i_result.get('total_results', 0)} 个相似图片")
                    
                except Exception as e:
                    logger.error(f"图片检索测试失败: {e}")

        print("=" * 60)
        print("UltrasoundRAG 系统运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"系统运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()