"""
UltrasoundRAG增强主入口 - 索引构建和智能检索系统
支持索引重构、多种检索方式、多数据库管理和所有优化功能

主要功能：
1. 索引构建和重构（支持领域分区和Caption增强）
2. 四种检索模式：T2T、T2I、I2T、I2I（集成动态权重和智能融合）
3. Caption检索图片（增强版本）
4. 多数据库支持
5. 检索性能测试
6. 查询自适应权重调整
7. 领域分区和智能路由
8. 多模态重排序
9. 智能融合策略
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
    create_i2t_retriever, create_i2i_retriever, 
    create_enhanced_multimodal_retriever
)
from UltrasoundRAG.retrival.caption_to_image_retriever import create_caption_retriever
from UltrasoundRAG.retrival.multi_database_manager import (
    create_multi_database_manager, create_retrieval_strategy
)

# 导入增强功能
from UltrasoundRAG.utils.query_adaptive_weights import (
    get_weight_calculator, calculate_query_adaptive_weights, get_fusion_weights
)
from UltrasoundRAG.utils.domain_partition import (
    get_partition_manager, classify_query_domain, get_search_domains
)
from UltrasoundRAG.utils.caption_enhancement import create_caption_enhancer
from UltrasoundRAG.retrival.fusion_strategy_retrival import (
    FusionRetrievalManager, create_enhanced_fusion_manager, FusionConfig, FusionStrategy
)
from UltrasoundRAG.retrival.enhanced_reranker import (
    create_multimodal_reranker, MultimodalRerankConfig, RerankMode
)

from UltrasoundRAG.utils.logger import setup_logger
from tqdm import tqdm
import json
import os

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

    built_count = 0
    skipped_count = 0
    global_id = 1  # 跨数据集统一分配主键，避免冲突

    datasets_items = list(markdown_cfg['datasets'].items())
    for dataset_name, dataset_cfg in tqdm(datasets_items, desc="Markdown 数据集", unit="ds"):
        if only_datasets and dataset_name not in only_datasets:
            continue
        if dataset_cfg.get('enabled', False):
            print(f"\n处理 Markdown 数据集: {dataset_name}")
            # 为该数据集确定集合名（优先用数据集级别的 collection_name，否则回退到全局默认）
            target_collection = dataset_cfg.get('collection_name', markdown_cfg.get('collections', 'md_documents'))
            # 为该数据集实例化独立的 MilvusManager，指向专属集合
            manager = MilvusManager(collection_type="md", collection_name=target_collection)
            if recreate:
                print(f"重建 Markdown 集合: {target_collection} (先删除再创建)")
                manager.drop_collection()
                manager._setup_collection()
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
                print(f"数据集 {dataset_name} 索引构建成功 -> 集合: {target_collection}")
            else:
                print(f"数据集 {dataset_name} 索引构建失败 -> 集合: {target_collection}")
        else:
            skipped_count += 1

    print(f"Markdown 构建完成：成功 {built_count}，跳过 {skipped_count}")
    # 结尾不再访问单一 manager（因每个数据集使用独立 manager）

def build_image_index(recreate: bool = False, only_datasets: list | None = None) -> None:
    """构建图片索引，可选重建集合与数据集过滤"""
    print("=" * 50)
    print("开始构建图片索引")
    image_cfg = config['indexing']['image']

    built_count = 0
    skipped_count = 0
    global_id = 1  # 跨数据集统一分配主键，避免冲突

    datasets_items = list(image_cfg['datasets'].items())
    for dataset_name, dataset_cfg in tqdm(datasets_items, desc="图片 数据集", unit="ds"):
        if only_datasets and dataset_name not in only_datasets:
            continue
        if dataset_cfg.get('enabled', False):
            print(f"\n处理图片数据集: {dataset_name}")
            # 为该数据集确定集合名（优先用数据集级别的 collection_name，否则回退到全局默认）
            target_collection = dataset_cfg.get('collection_name', image_cfg.get('collections', 'images'))
            # 为该数据集实例化独立的 MilvusManager，指向专属集合
            manager = MilvusManager(collection_type="image", collection_name=target_collection)
            if recreate:
                print(f"重建 图片 集合: {target_collection} (先删除再创建)")
                manager.drop_collection()
                manager._setup_collection()
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
                # 现在parsed_data中已经包含了caption_vector_clip_768，直接传入即可
                ok = manager.insert_data(part_data, part_emb)
                insert_ok = insert_ok and ok
            success = insert_ok
            if success:
                built_count += 1
                print(f"数据集 {dataset_name} 索引构建成功 -> 集合: {target_collection}")
            else:
                print(f"数据集 {dataset_name} 索引构建失败 -> 集合: {target_collection}")
        else:
            skipped_count += 1

    print(f"图片 构建完成：成功 {built_count}，跳过 {skipped_count}")
    # 结尾不再访问单一 manager（因每个数据集使用独立 manager）

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _save_json(path: str, obj: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print(f"写入结果失败 {path}: {e}")


def _rr_to_dict(rr) -> dict:
    try:
        return {
            'doc_id': getattr(rr, 'doc_id', None),
            'score': getattr(rr, 'score', None),
            'content': getattr(rr, 'content', None),
            'metadata': getattr(rr, 'metadata', {}),
            'resource_collection': getattr(rr, 'resource_collection', ''),
            'retrieval_type': getattr(rr, 'retrieval_type', ''),
        }
    except Exception:
        return {'raw': str(rr)}


def _container_to_serializable(container: dict) -> dict:
    if not isinstance(container, dict):
        return container
    out = dict(container)
    for key in ['results', 'all_results', 'matched_images']:
        if key in out and isinstance(out[key], list):
            out[key] = [_rr_to_dict(x) if not isinstance(x, dict) else x for x in out[key]]
    return out


def test_retrieval_modes(db_name: str = "default", top_k: int = 3) -> None:
    """测试四种检索模式 + Caption 匹配，并将原始返回保存到 result/test 目录。"""
    logger = setup_logger("RetrievalTest")
    
    print("=" * 60)
    print("开始测试多种检索模式")
    print("=" * 60)
    
    # 准备输出目录
    result_dir = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/result/test"
    _ensure_dir(result_dir)

    # 1. T2T检索测试
    print("\n1. 文本到文本检索（T2T）测试:")
    print("-" * 40)
    try:
        t2t_retriever = create_t2t_retriever(db_name, top_k)
        # 测试禁用领域分区，避免 domain 过滤影响召回
        try:
            t2t_retriever.context.enable_domain_partition = False
        except Exception:
            pass
        t2t_query = "心脏超声检查方法"
        t2t_result = t2t_retriever.search(t2t_query, top_k=top_k)
        
        print(f"查询: '{t2t_result.get('query', '')}'")
        print(f"策略: {t2t_result.get('strategy', 'basic')}")
        print(f"找到 {t2t_result.get('total_results', 0)} 个文本结果")
        
        # 保存原始返回
        _save_json(os.path.join(result_dir, "t2t.json"), _container_to_serializable(t2t_result))

        for i, result in enumerate(t2t_result.get('results', [])[:3]):
            print(f"  {i+1}. 分数: {result.score:.4f}")
            print(f"      内容: {result.content[:80]}...")
            print(f"      来源: {result.metadata.get('document_name', 'N/A')}")
            print()
    except Exception as e:
        logger.error(f"T2T检索测试失败: {e}")
    
    # 2. T2I检索测试（文本→图片）
    print("\n2. 文本到图片检索（T2I）测试:")
    print("-" * 40)
    try:
        t2i_retriever = create_t2i_retriever(db_name, top_k)
        try:
            t2i_retriever.context.enable_domain_partition = False
        except Exception:
            pass
        t2i_query = "妊娠中晚期正常乳腺切面"
        t2i_result = t2i_retriever.search(t2i_query, top_k=top_k)
        
        print(f"查询: '{t2i_result.get('query', '')}'")
        print(f"找到 {t2i_result.get('total_results', 0)} 个图片结果")
        
        # 保存原始返回
        _save_json(os.path.join(result_dir, "t2i.json"), _container_to_serializable(t2i_result))

        for i, result in enumerate(t2i_result.get('results', [])[:3]):
            # 原始结果尽量完整输出（保持可读性）
            raw = {
                'doc_id': getattr(result, 'doc_id', None),
                'score': getattr(result, 'score', None),
                'content': getattr(result, 'content', None),
                'metadata': getattr(result, 'metadata', {}),
                'resource_collection': getattr(result, 'resource_collection', ''),
                'retrieval_type': getattr(result, 'retrieval_type', ''),
            }
            print(f"\n—— T2I 原始结果 {i+1} ——")
            print(json.dumps(raw, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        logger.error(f"T2I检索测试失败: {e}")
    
    # 3. Caption检索图片测试（精简版接口）
    print("\n3. Caption检索图片测试:")
    print("-" * 40)
    try:
        caption_retriever = create_caption_retriever(db_name)
        
        # 单条/整段 MD image_captions 输入（会自动拆分并 OR-like 匹配）
        md_caps = ["图6-6 大血管短轴切面 （主动脉瓣水平）；图6-6 大血管短轴切面 （主动脉瓣水平）"]
        caption_result = caption_retriever.search_from_md_image_captions(md_caps, top_k_per_caption=top_k, use_like=True)
        print(f"Caption候选: {caption_result.get('normalized_candidates', [])}")
        print(f"找到 {caption_result.get('total_results', 0)} 个匹配图片")

        # 保存原始返回
        _save_json(os.path.join(result_dir, "caption.json"), _container_to_serializable(caption_result))
        
        for i, result in enumerate(caption_result.get('results', [])[:3]):
            raw = {
                'doc_id': getattr(result, 'doc_id', None),
                'score': getattr(result, 'score', None),
                'content': getattr(result, 'content', None),
                'metadata': getattr(result, 'metadata', {}),
                'resource_collection': getattr(result, 'resource_collection', ''),
                'retrieval_type': getattr(result, 'retrieval_type', ''),
            }
            print(f"\n—— Caption 原始结果 {i+1} ——")
            print(json.dumps(raw, ensure_ascii=False, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Caption检索测试失败: {e}")
    
    # 4. I2T（图片→文本）与 5. I2I（图片→图片）测试，使用指定图片路径
    print("\n4. 图片到文本检索（I2T）测试:")
    print("-" * 40)
    try:
        i2t_retriever = create_i2t_retriever(db_name, top_k)
        try:
            i2t_retriever.context.enable_domain_partition = False
        except Exception:
            pass
        test_image_path = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg"
        i2t_result = i2t_retriever.search(test_image_path, top_k=top_k)
        _save_json(os.path.join(result_dir, "i2t.json"), _container_to_serializable(i2t_result))
        print(f"I2T 结果数: {i2t_result.get('total_results', 0)}")
    except Exception as e:
        logger.error(f"I2T检索失败: {e}")

    print("\n5. 图片到图片检索（I2I）测试:")
    print("-" * 40)
    try:
        i2i_retriever = create_i2i_retriever(db_name, top_k)
        try:
            i2i_retriever.context.enable_domain_partition = False
        except Exception:
            pass
        test_image_path = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg"
        i2i_result = i2i_retriever.search(test_image_path, top_k=top_k)
        _save_json(os.path.join(result_dir, "i2i.json"), _container_to_serializable(i2i_result))
        print(f"I2I 结果数: {i2i_result.get('total_results', 0)}")
    except Exception as e:
        logger.error(f"I2I检索失败: {e}")

    print("\n结果已保存到: /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/result/test")


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
    retrieval_group.add_argument("--multi-db-keys", type=str, nargs='*', help="多数据库键列表，如 normal md_thyroid")
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
                if args.multi_db_keys and len(args.multi_db_keys) >= 2:
                    # 走多数据库融合示例
                    print("=" * 60)
                    print("使用多数据库融合示例")
                    print("=" * 60)
                    try:
                        manager = create_multi_database_manager()
                        from UltrasoundRAG.retrival.multi_database_manager import create_retrieval_strategy
                        strategy = create_retrieval_strategy(args.multi_db_keys, aggregation_method="weighted", max_results=args.top_k)
                        result = manager.search_multiple_databases(strategy, "t2t", query, top_k=args.top_k)
                        print(f"融合数据库: {args.multi_db_keys}")
                        print(f"聚合结果数: {result.get('total_results', 0)}")
                    except Exception as e:
                        print(f"多数据库融合失败: {e}")
                else:
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

# === 新增增强功能函数 ===

def test_enhanced_features(query: str = "心脏超声图像", db_name: str = "default", top_k: int = 5) -> None:
    """测试所有增强功能"""
    logger = setup_logger("EnhancedFeaturesTest")
    
    print("\n" + "=" * 80)
    print("🚀 UltrasoundRAG 增强功能测试")
    print("=" * 80)
    
    # 1. 测试查询自适应权重
    print(f"\n1. 📊 查询自适应权重测试")
    print(f"   查询: '{query}'")
    try:
        qwen_weight, clip_weight = calculate_query_adaptive_weights(query)
        fusion_weights = get_fusion_weights(query)
        print(f"   Qwen权重: {qwen_weight:.3f}, CLIP权重: {clip_weight:.3f}")
        print(f"   融合权重: text={fusion_weights.get('text_weight', 0):.3f}, image={fusion_weights.get('image_weight', 0):.3f}")
    except Exception as e:
        logger.error(f"权重计算失败: {e}")
    
    # 2. 测试领域分区
    print(f"\n2. 🏥 领域分区测试")
    try:
        domain_matches = classify_query_domain(query)
        search_domains, domain_weights = get_search_domains(query)
        print(f"   识别领域: {[f'{m.domain.value}(置信度:{m.confidence:.3f})' for m in domain_matches[:3]]}")
        print(f"   检索分区: {search_domains}")
    except Exception as e:
        logger.error(f"领域分区失败: {e}")
    
    # 3. 测试增强检索器
    print(f"\n3. 🔍 增强多模态检索测试")
    try:
        enhanced_retriever = create_enhanced_multimodal_retriever(
            db_name=db_name, 
            top_k=top_k,
            enable_dynamic_weights=True,
            enable_domain_partition=True,
            enable_caption_enhancement=True,
            enable_multimodal_rerank=True
        )
        
        # 自动模式检索
        auto_result = enhanced_retriever.search(query, mode="auto", top_k=top_k)
        print(f"   自动模式: {auto_result.get('retrieval_type', 'unknown')}")
        print(f"   找到结果: {auto_result.get('total_results', 0)} 个")
        print(f"   响应时间: {auto_result.get('unified_response_time', 0):.3f}s")
        
        # 多模态融合模式
        multimodal_result = enhanced_retriever.search(query, mode="multimodal", top_k=top_k)
        print(f"   多模态模式: 文本={multimodal_result.get('text_count', 0)}, 图像={multimodal_result.get('image_count', 0)}")
        
    except Exception as e:
        logger.error(f"增强检索失败: {e}")
    
    # 4. 测试增强融合策略
    print(f"\n4. 🔄 增强融合策略测试")
    try:
        fusion_config = FusionConfig(
            enable_adaptive_weights=True,
            enable_smart_fusion=True,
            default_fusion_strategy=FusionStrategy.ADAPTIVE_SMART
        )
        fusion_manager = create_enhanced_fusion_manager(fusion_config)
        print(f"   融合配置: 自适应权重={fusion_config.enable_adaptive_weights}, 智能融合={fusion_config.enable_smart_fusion}")
        print(f"   默认策略: {fusion_config.default_fusion_strategy.value}")
        
    except Exception as e:
        logger.error(f"融合策略测试失败: {e}")
    
    # 5. 测试多模态重排序
    print(f"\n5. 📈 多模态重排序测试")
    try:
        rerank_config = MultimodalRerankConfig(
            mode=RerankMode.MULTIMODAL,
            enable_rule_based=True,
            enable_model_based=True
        )
        reranker = create_multimodal_reranker(rerank_config)
        print(f"   重排序模式: {rerank_config.mode.value}")
        print(f"   规则重排: {rerank_config.enable_rule_based}, 模型重排: {rerank_config.enable_model_based}")
        
    except Exception as e:
        logger.error(f"重排序测试失败: {e}")
    
    print(f"\n✅ 增强功能测试完成!")


def benchmark_enhanced_performance(queries: List[str], db_name: str = "default", top_k: int = 10) -> None:
    """对比增强前后的性能"""
    logger = setup_logger("EnhancedBenchmark")
    
    print("\n" + "=" * 80)
    print("⚡ UltrasoundRAG 增强性能对比测试")
    print("=" * 80)
    
    results = {
        'traditional': {'total_time': 0, 'avg_time': 0, 'results_count': 0},
        'enhanced': {'total_time': 0, 'avg_time': 0, 'results_count': 0}
    }
    
    for i, query in enumerate(queries[:5]):  # 测试前5个查询
        print(f"\n🔍 查询 {i+1}: '{query[:30]}...'")
        
        # 传统T2T检索
        try:
            start_time = time.time()
            traditional_retriever = create_t2t_retriever(db_name, top_k)
            trad_result = traditional_retriever.search(query, top_k=top_k)
            trad_time = time.time() - start_time
            
            results['traditional']['total_time'] += trad_time
            results['traditional']['results_count'] += trad_result.get('total_results', 0)
            print(f"   传统检索: {trad_time:.3f}s, {trad_result.get('total_results', 0)} 结果")
            
        except Exception as e:
            print(f"   传统检索失败: {e}")
        
        # 增强多模态检索
        try:
            start_time = time.time()
            enhanced_retriever = create_enhanced_multimodal_retriever(
                db_name=db_name, 
                top_k=top_k,
                enable_dynamic_weights=True,
                enable_domain_partition=True,
                enable_multimodal_rerank=True
            )
            enhanced_result = enhanced_retriever.search(query, mode="auto", top_k=top_k)
            enhanced_time = time.time() - start_time
            
            results['enhanced']['total_time'] += enhanced_time
            results['enhanced']['results_count'] += enhanced_result.get('total_results', 0)
            print(f"   增强检索: {enhanced_time:.3f}s, {enhanced_result.get('total_results', 0)} 结果")
            
            # 显示使用的模式和策略
            print(f"   使用模式: {enhanced_result.get('retrieval_type', 'unknown')}")
            if 'fusion_strategy' in enhanced_result:
                print(f"   融合策略: {enhanced_result.get('fusion_strategy', 'unknown')}")
            
        except Exception as e:
            print(f"   增强检索失败: {e}")
    
    # 计算平均性能
    num_queries = min(len(queries), 5)
    if num_queries > 0:
        results['traditional']['avg_time'] = results['traditional']['total_time'] / num_queries
        results['enhanced']['avg_time'] = results['enhanced']['total_time'] / num_queries
        
        print(f"\n📊 性能对比总结:")
        print(f"   传统检索: 平均 {results['traditional']['avg_time']:.3f}s/查询")
        print(f"   增强检索: 平均 {results['enhanced']['avg_time']:.3f}s/查询")
        
        improvement = (results['traditional']['avg_time'] - results['enhanced']['avg_time']) / results['traditional']['avg_time'] * 100
        if improvement > 0:
            print(f"   ⬆️  性能提升: {improvement:.1f}%")
        else:
            print(f"   ⬇️  性能开销: {abs(improvement):.1f}%")


def quick_demo() -> None:
    """快速演示所有增强功能"""
    print("\n" + "🌟" * 20)
    print("UltrasoundRAG 增强功能快速演示")
    print("🌟" * 20)
    
    demo_queries = [
        "心脏超声图像分析",
        "胎儿发育异常检查", 
        "肝脏超声诊断要点",
        "图2-3显示的病变特征"
    ]
    
    for query in demo_queries:
        print(f"\n🔍 演示查询: '{query}'")
        test_enhanced_features(query, top_k=3)
        time.sleep(1)  # 短暂停顿以便观察


if __name__ == "__main__":
    main()