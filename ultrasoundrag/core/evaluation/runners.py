"""
UltrasoundRAG 测试运行器模块
集中处理各种检索功能的测试逻辑
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

from ultrasoundrag.core.retrieval.modular_retrievers import (
    create_t2t_retriever, create_t2i_retriever,
    create_i2t_retriever, create_i2i_retriever
)
from ultrasoundrag.core.retrieval.caption_to_image_retriever import create_caption_retriever
from ultrasoundrag.core.retrieval.multi_database_manager import create_multi_database_manager
from ultrasoundrag.utils.logger import setup_logger


class TestRunner:
    """统一的测试运行器"""
    
    def __init__(self, result_dir: str = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/result/test"):
        self.logger = setup_logger("TestRunner")
        self.result_dir = result_dir
        self._ensure_dir(result_dir)
    
    def run_retrieval_modes_test(self, db_name: str = "default", top_k: int = 3) -> dict:
        """测试四种检索模式 + Caption 匹配"""
        self.logger.info("=" * 60)
        self.logger.info("开始测试多种检索模式")
        self.logger.info("=" * 60)
        
        results = {}
        
        # 1. T2T检索测试
        results['t2t'] = self._test_t2t_retrieval(db_name, top_k)
        
        # 2. T2I检索测试
        results['t2i'] = self._test_t2i_retrieval(db_name, top_k)
        
        # 3. Caption检索测试
        results['caption'] = self._test_caption_retrieval(db_name, top_k)
        
        # 4. I2T检索测试
        results['i2t'] = self._test_i2t_retrieval(db_name, top_k)
        
        # 5. I2I检索测试
        results['i2i'] = self._test_i2i_retrieval(db_name, top_k)
        
        # 保存汇总结果
        summary = {
            'test_time': time.time(),
            'db_name': db_name,
            'top_k': top_k,
            'results': results,
            'summary': {
                'total_tests': len(results),
                'successful_tests': sum(1 for r in results.values() if r.get('success', False)),
                'failed_tests': sum(1 for r in results.values() if not r.get('success', False))
            }
        }
        
        self._save_json(os.path.join(self.result_dir, "test_summary.json"), summary)
        self.logger.info(f"测试结果已保存到: {self.result_dir}")
        
        return summary
    
    def run_multi_database_test(self, query: str = "心脏超声诊断", top_k: int = 5) -> dict:
        """测试多数据库检索功能"""
        self.logger.info("=" * 60)
        self.logger.info("开始测试多数据库检索功能")
        self.logger.info("=" * 60)
        
        try:
            manager = create_multi_database_manager()
            
            # 获取可用数据库
            available_dbs = manager.get_available_databases()
            self.logger.info(f"可用数据库 ({len(available_dbs)} 个):")
            for db in available_dbs:
                self.logger.info(f"  - {db.name}: {db.description} (优先级: {db.priority})")
            
            if not available_dbs:
                return {"success": False, "error": "没有可用的数据库配置"}
            
            results = {}
            
            # 单数据库检索测试
            if available_dbs:
                single_result = manager.search_single_database(
                    available_dbs[0].name, "t2t", query, top_k=top_k
                )
                results['single_database'] = {
                    'database': single_result.get('database'),
                    'total_results': single_result.get('total_results', 0),
                    'success': True
                }
            
            # 自动选择数据库测试
            auto_selected = manager.auto_select_databases(query, "t2t", num_databases=2)
            results['auto_selection'] = {
                'selected_databases': auto_selected,
                'success': True
            }
            
            # 多数据库检索测试（如果有多个数据库）
            if len(available_dbs) >= 2:
                from ultrasoundrag.core.retrieval.multi_database_manager import create_retrieval_strategy
                strategy = create_retrieval_strategy(
                    [db.name for db in available_dbs[:2]], 
                    aggregation_method="weighted",
                    max_results=top_k
                )
                multi_result = manager.search_multiple_databases(
                    strategy, "t2t", query, top_k=top_k//2
                )
                results['multi_database'] = {
                    'aggregation_method': strategy.aggregation_method,
                    'databases_searched': multi_result.get('total_databases_searched', 0),
                    'total_results': multi_result.get('total_results', 0),
                    'response_time': multi_result.get('response_time', 0),
                    'success': True
                }
            
            # 数据库统计信息
            stats = manager.get_database_stats()
            results['statistics'] = {
                'total_databases': stats['total_databases'],
                'enabled_databases': stats['enabled_databases'],
                'total_queries': stats['usage_stats']['total_queries'],
                'avg_response_time': stats['usage_stats']['avg_response_time'],
                'success': True
            }
            
            return {
                'success': True,
                'query': query,
                'top_k': top_k,
                'available_databases': len(available_dbs),
                'results': results
            }
            
        except ImportError as e:
            self.logger.error(f"多数据库检索模块导入失败: {e}")
            return {'success': False, 'error': f'模块导入失败: {e}'}
        except ConnectionError as e:
            self.logger.error(f"多数据库检索连接失败: {e}")
            return {'success': False, 'error': f'数据库连接失败: {e}'}
        except Exception as e:
            self.logger.error(f"多数据库检索测试失败: {type(e).__name__}: {e}")
            return {'success': False, 'error': f'{type(e).__name__}: {str(e)}'}
    
    def run_caption_test(self, db_name: str = "default", top_k: int = 3) -> dict:
        """专项Caption检索测试"""
        self.logger.info("=" * 60)
        self.logger.info("开始Caption检索图片专项测试")
        self.logger.info("=" * 60)
        
        try:
            caption_retriever = create_caption_retriever(db_name, "hybrid_match")
            
            results = {}
            
            # 测试单个caption
            test_captions = [
                "图2-3 心脏超声横切面",
                "图1-1 肝脏超声检查",
                "Figure 3.2 胎儿发育图"
            ]
            
            results['single_captions'] = []
            for caption in test_captions:
                self.logger.info(f"测试Caption: '{caption}'")
                result = caption_retriever.search_single_caption(caption, top_k=top_k)
                results['single_captions'].append({
                    'caption': caption,
                    'total_results': result.get('total_results', 0),
                    'success': result.get('total_results', 0) > 0
                })
            
            # 测试从文本提取
            text_chunk = """
            心脏超声检查包括多个切面。图2-1显示四腔心切面，
            图2-2为心脏短轴切面，图2-3展示了心尖四腔心切面。
            """
            self.logger.info("从文本块提取caption测试:")
            text_result = caption_retriever.search_from_text_chunk(text_chunk, top_k=2)
            results['text_extraction'] = {
                'extracted_captions': len(text_result.get('extracted_captions', [])),
                'total_results': text_result.get('total_results', 0),
                'success': text_result.get('total_results', 0) > 0
            }
            
            return {
                'success': True,
                'db_name': db_name,
                'top_k': top_k,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Caption检索测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_t2t_retrieval(self, db_name: str, top_k: int) -> dict:
        """T2T检索测试"""
        self.logger.info("1. 文本到文本检索（T2T）测试:")
        self.logger.info("-" * 40)
        
        try:
            t2t_retriever = create_t2t_retriever(db_name, top_k)
            # 禁用领域分区避免过滤影响
            try:
                t2t_retriever.context.enable_domain_partition = False
            except Exception:
                pass
            
            t2t_query = "心脏超声检查方法"
            t2t_result = t2t_retriever.search(t2t_query, top_k=top_k)
            
            self.logger.info(f"查询: '{t2t_result.get('query', '')}'")
            self.logger.info(f"找到 {t2t_result.get('total_results', 0)} 个文本结果")
            
            # 保存结果
            self._save_json(os.path.join(self.result_dir, "t2t.json"), self._serialize_result(t2t_result))
            
            return {
                'success': True,
                'query': t2t_query,
                'total_results': t2t_result.get('total_results', 0),
                'strategy': t2t_result.get('strategy', 'basic')
            }
            
        except Exception as e:
            self.logger.error(f"T2T检索测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_t2i_retrieval(self, db_name: str, top_k: int) -> dict:
        """T2I检索测试"""
        self.logger.info("2. 文本到图片检索（T2I）测试:")
        self.logger.info("-" * 40)
        
        try:
            t2i_retriever = create_t2i_retriever(db_name, top_k)
            try:
                t2i_retriever.context.enable_domain_partition = False
            except Exception:
                pass
            
            t2i_query = "妊娠中晚期正常乳腺切面"
            t2i_result = t2i_retriever.search(t2i_query, top_k=top_k)
            
            self.logger.info(f"查询: '{t2i_result.get('query', '')}'")
            self.logger.info(f"找到 {t2i_result.get('total_results', 0)} 个图片结果")
            
            # 保存结果
            self._save_json(os.path.join(self.result_dir, "t2i.json"), self._serialize_result(t2i_result))
            
            return {
                'success': True,
                'query': t2i_query,
                'total_results': t2i_result.get('total_results', 0)
            }
            
        except Exception as e:
            self.logger.error(f"T2I检索测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_caption_retrieval(self, db_name: str, top_k: int) -> dict:
        """Caption检索测试"""
        self.logger.info("3. Caption检索图片测试:")
        self.logger.info("-" * 40)
        
        try:
            caption_retriever = create_caption_retriever(db_name)
            
            md_caps = ["图6-6 大血管短轴切面 （主动脉瓣水平）；图6-6 大血管短轴切面 （主动脉瓣水平）"]
            caption_result = caption_retriever.search_from_md_image_captions(
                md_caps, top_k_per_caption=top_k, use_like=True
            )
            
            self.logger.info(f"Caption候选: {caption_result.get('normalized_candidates', [])}")
            self.logger.info(f"找到 {caption_result.get('total_results', 0)} 个匹配图片")
            
            # 保存结果
            self._save_json(os.path.join(self.result_dir, "caption.json"), self._serialize_result(caption_result))
            
            return {
                'success': True,
                'candidates': caption_result.get('normalized_candidates', []),
                'total_results': caption_result.get('total_results', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Caption检索测试失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_i2t_retrieval(self, db_name: str, top_k: int) -> dict:
        """I2T检索测试"""
        self.logger.info("4. 图片到文本检索（I2T）测试:")
        self.logger.info("-" * 40)
        
        try:
            i2t_retriever = create_i2t_retriever(db_name, top_k)
            try:
                i2t_retriever.context.enable_domain_partition = False
            except Exception:
                pass
            
            test_image_path = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg"
            i2t_result = i2t_retriever.search(test_image_path, top_k=top_k)
            
            # 保存结果
            self._save_json(os.path.join(self.result_dir, "i2t.json"), self._serialize_result(i2t_result))
            
            return {
                'success': True,
                'image_path': test_image_path,
                'total_results': i2t_result.get('total_results', 0)
            }
            
        except Exception as e:
            self.logger.error(f"I2T检索失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _test_i2i_retrieval(self, db_name: str, top_k: int) -> dict:
        """I2I检索测试"""
        self.logger.info("5. 图片到图片检索（I2I）测试:")
        self.logger.info("-" * 40)
        
        try:
            i2i_retriever = create_i2i_retriever(db_name, top_k)
            try:
                i2i_retriever.context.enable_domain_partition = False
            except Exception:
                pass
            
            test_image_path = "/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image/07_腹部超声/图片00008.jpg"
            i2i_result = i2i_retriever.search(test_image_path, top_k=top_k)
            
            # 保存结果
            self._save_json(os.path.join(self.result_dir, "i2i.json"), self._serialize_result(i2i_result))
            
            return {
                'success': True,
                'image_path': test_image_path,
                'total_results': i2i_result.get('total_results', 0)
            }
            
        except Exception as e:
            self.logger.error(f"I2I检索失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _serialize_result(self, result: dict) -> dict:
        """序列化结果，处理不可序列化的对象"""
        if not isinstance(result, dict):
            return result
        
        serialized = dict(result)
        for key in ['results', 'all_results', 'matched_images']:
            if key in serialized and isinstance(serialized[key], list):
                serialized[key] = [self._serialize_rr(x) if not isinstance(x, dict) else x 
                                 for x in serialized[key]]
        return serialized
    
    def _serialize_rr(self, rr) -> dict:
        """序列化检索结果对象"""
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
    
    def _ensure_dir(self, path: str) -> None:
        """确保目录存在"""
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    
    def _save_json(self, path: str, obj: dict) -> None:
        """保存JSON文件"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"写入结果失败 {path}: {e}")


# 便捷函数
def run_retrieval_test(db_name: str = "default", top_k: int = 3) -> dict:
    """运行检索模式测试的便捷函数"""
    runner = TestRunner()
    return runner.run_retrieval_modes_test(db_name, top_k)


def run_multi_database_test(query: str = "心脏超声诊断", top_k: int = 5) -> dict:
    """运行多数据库测试的便捷函数"""
    runner = TestRunner()
    return runner.run_multi_database_test(query, top_k)


def run_caption_test(db_name: str = "default", top_k: int = 3) -> dict:
    """运行Caption测试的便捷函数"""
    runner = TestRunner()
    return runner.run_caption_test(db_name, top_k)
