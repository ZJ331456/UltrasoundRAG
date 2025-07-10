"""
RAG评估系统 - 简化版本
专注于核心的RAGAS指标评估，使用utils模块中的组件

重构改进：
1. 简化评估流程
2. 使用utils模块中的LLM工具
3. 减少冗余代码
4. 优化错误处理
"""

import os
import json
import asyncio
import sys
from typing import List, Dict
from datetime import datetime
import torch

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.embedding_utils import LocalEmbedding
from MedicalRAG.utils.llm_utils import llm_provider, LLMError
# RAGAS相关导入
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    ContextRelevance
)


class RAGEvaluator:
    """RAG评估器 - 专注核心功能"""
    
    def __init__(self, input_file_path: str = None):
        """
        初始化评估器
        
        Args:
            input_file_path: 输入文件路径，如果为None则使用配置文件中的路径
        """
        self.logger = setup_logger(__name__)
        self.input_file_path = input_file_path or config['evaluator']['input_eval_file']
        self.start_time = datetime.now()
        
        # 初始化评估用的LLM和嵌入模型
        self._setup_evaluator_models()
        
        # 加载和准备数据
        self.raw_data = self._load_data()
        self.evaluation_samples = self._prepare_data()
        
        self.logger.info(f"RAG评估器初始化完成，准备评估 {len(self.evaluation_samples)} 个样本")
    
    def _setup_evaluator_models(self):
        """设置评估用的模型"""
        # 评估用LLM
        # 使用配置中的LLM提供者
        # llm_config = config['llm_providers']['medical_vllm_api']['params']
        llm_config = config['llm_providers']['xiaohumini_llm_api']['params']
        # 为RAGAS评估优化LLM配置
        self.evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=llm_config['model_name'],
                openai_api_base=llm_config['api_url'],
                openai_api_key=llm_config['api_key'],
                temperature=0.1,  # 降低温度以获得更稳定的输出
                max_tokens=10480,  # 增加max_tokens以确保RAGAS评估能够完成
                request_timeout=120,  # 增加超时时间
                max_retries=3  # 增加重试次数
            )
        )
        
        # 评估用嵌入模型
        # 从配置中获取嵌入模型提供者
        embedding_provider_name = config['embedding']['provider']
        embedding_config = config['embedding_providers'][embedding_provider_name]
        
        if embedding_config['type'] == 'local':
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(
                LocalEmbedding(
                    model_name=embedding_config['params']['model_name'],
                    device=embedding_config['params'].get('device', "cuda" if torch.cuda.is_available() else "cpu")
                )
            )
        else:
            # 如果是API类型，使用APIEmbedding
            from MedicalRAG.utils.embedding_utils import APIEmbedding
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(
                APIEmbedding(
                    api_url=embedding_config['params']['api_url'],
                    model_name=embedding_config['params']['model_name'],
                    api_key=embedding_config['params'].get('api_key'),
                    batch_size=embedding_config['params'].get('batch_size', 32)
                )
            )
    
    def _load_data(self) -> List[Dict]:
        """加载RAG查询结果数据"""
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否是新的批量结果格式（包含summary和results）
            if isinstance(data, dict) and 'results' in data:
                results = data['results']
                self.logger.info(f"成功加载批量结果格式，共 {len(results)} 条查询结果")
                return results
            elif isinstance(data, list):
                self.logger.info(f"成功加载 {len(data)} 条查询结果")
                return data
            else:
                self.logger.error(f"不支持的数据格式: {type(data)}")
                return []
                
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return []
    
    def _truncate_text(self, text: str, max_length: int = 500) -> str:
        """智能截断文本，保持完整性"""
        if len(text) <= max_length:
            return text
        
        # 尝试按句号截断
        sentences = text.split('。')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + '。') <= max_length:
                truncated += sentence + '。'
            else:
                break
        
        # 如果第一句话就太长，直接截断
        if not truncated:
            truncated = text[:max_length-3] + "..."
        
        return truncated
    
    def _prepare_data(self) -> List[Dict]:
        """准备评估数据"""
        evaluation_samples = []
        
        for item in self.raw_data:
            # 适配新的批量结果格式
            query = item.get("original_question", item.get("query", ""))
            answer = item.get("generated_answer", item.get("answer", ""))
            retrieval_results = item.get("retrieval_results", item.get("retrieval", {}))
            
            # 清理和验证输入
            if not query.strip() or not answer.strip():
                self.logger.warning(f"跳过无效样本: 问题或答案为空")
                continue
            
            # 提取和处理上下文
            contexts = []
            if isinstance(retrieval_results, dict):
                for doc_id, doc_info in retrieval_results.items():
                    content = doc_info.get("content", "")
                    if content.strip():
                        # 为评估截断过长的上下文，避免API限制
                        truncated_content = self._truncate_text(content, max_length=600)
                        # 清理特殊字符
                        cleaned_content = self._clean_text(truncated_content)
                        if cleaned_content:
                            contexts.append(cleaned_content)
            
            # 过滤无效样本
            if not contexts:
                self.logger.warning(f"跳过无效样本: 没有有效上下文 - {query[:50]}...")
                continue
            
            # 限制上下文数量避免API限制
            contexts = contexts[:2]  # 减少到2个上下文
            
            # 清理答案文本
            cleaned_answer = self._clean_text(answer[:800])  # 限制答案长度
            
            sample = {
                "question": query[:200],  # 限制问题长度
                "answer": cleaned_answer,
                "contexts": contexts
            }
            
            evaluation_samples.append(sample)
        
        self.logger.info(f"准备了 {len(evaluation_samples)} 个有效评估样本")
        return evaluation_samples
    
    def _clean_text(self, text: str) -> str:
        """清理文本，移除可能导致解析错误的字符"""
        if not text:
            return ""
        
        # 移除或替换可能导致JSON解析问题的字符
        cleaned = text.replace('\n', ' ').replace('\r', ' ')
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")
        cleaned = cleaned.replace('…', '...')
        
        # 移除多余的空格
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def _create_metrics(self):
        """创建RAGAS评估指标"""
        # 配置各项指标
        faithfulness_metric = faithfulness
        faithfulness_metric.llm = self.evaluator_llm
        
        answer_relevancy_metric = answer_relevancy
        answer_relevancy_metric.llm = self.evaluator_llm
        answer_relevancy_metric.embeddings = self.evaluator_embeddings
        
        context_relevancy_metric = ContextRelevance()
        context_relevancy_metric.llm = self.evaluator_llm
        
        return {
            "faithfulness": faithfulness_metric,
            "answer_relevancy": answer_relevancy_metric,
            "context_relevancy": context_relevancy_metric
        }
    
    async def _evaluate_single_sample(self, sample_data: Dict, metrics: Dict) -> Dict:
        """评估单个样本"""
        sample = SingleTurnSample(
            user_input=sample_data["question"],
            response=sample_data["answer"],
            retrieved_contexts=sample_data["contexts"]
        )
        
        results = {"question": sample_data["question"]}
        
        # 评估各项指标
        for metric_name, metric in metrics.items():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    score = metric.single_turn_score(sample)
                    results[metric_name] = round(float(score), 3) if score is not None else 0.0
                    self.logger.debug(f"{metric_name}: {results[metric_name]:.3f}")
                    break  # 成功则跳出重试循环
                except Exception as e:
                    error_msg = str(e)
                    self.logger.warning(f"{metric_name} 评估失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                    
                    # 如果是输出解析错误，尝试简化输入
                    if "failed to parse output" in error_msg.lower() or "output parser failed" in error_msg.lower():
                        if attempt < max_retries - 1:
                            # 尝试简化上下文长度
                            simplified_contexts = [ctx[:500] for ctx in sample_data["contexts"]]
                            simplified_sample = SingleTurnSample(
                                user_input=sample_data["question"],
                                response=sample_data["answer"][:1000],  # 限制答案长度
                                retrieved_contexts=simplified_contexts
                            )
                            sample = simplified_sample
                            continue
                    
                    # 最后一次尝试失败
                    if attempt == max_retries - 1:
                        self.logger.error(f"{metric_name} 评估最终失败: {error_msg}")
                        results[metric_name] = 0.0
        
        return results
    
    async def evaluate_all(self) -> Dict:
        """评估所有样本"""
        self.logger.info("开始RAGAS评估...")
        
        if not self.evaluation_samples:
            return {"error": "没有可评估的数据"}
        
        # 清除代理设置（避免网络问题）
        for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if var in os.environ:
                del os.environ[var]
        
        metrics = self._create_metrics()
        results = {
            "evaluation_time": datetime.now().isoformat(),
            "total_queries": len(self.evaluation_samples),
            "metrics_used": list(metrics.keys()),
            "query_results": [],
            "summary": {}
        }
        
        # 逐个评估样本
        for i, sample_data in enumerate(self.evaluation_samples):
            self.logger.info(f"评估样本 {i+1}/{len(self.evaluation_samples)}: '{sample_data['question'][:50]}...'")
            
            try:
                query_result = await self._evaluate_single_sample(sample_data, metrics)
                results["query_results"].append(query_result)
            except Exception as e:
                self.logger.error(f"样本 {i+1} 评估失败: {e}")
                # 添加失败的结果
                failed_result = {
                    "question": sample_data["question"],
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_relevancy": 0.0,
                    "error": str(e)
                }
                results["query_results"].append(failed_result)
        
        # 计算汇总统计
        self._calculate_summary(results)
        
        return results
    
    def _calculate_summary(self, results: Dict):
        """计算汇总统计"""
        if not results["query_results"]:
            return
        
        summary = {}
        metric_names = ["faithfulness", "answer_relevancy", "context_relevancy"]
        
        for metric_name in metric_names:
            scores = [
                r[metric_name] for r in results["query_results"] 
                if metric_name in r and isinstance(r[metric_name], (int, float))
            ]
            
            if scores:
                summary[metric_name] = {
                    "average": round(sum(scores) / len(scores), 3),
                    "max": round(max(scores), 3),
                    "min": round(min(scores), 3),
                    "count": len(scores)
                }
        
        results["summary"] = summary
    
    def save_results(self, results: Dict, output_file_path: str = None) -> str:
        """保存评估结果"""
        try:
            if output_file_path is None:
                timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
                input_filename = os.path.basename(self.input_file_path).replace('.json', '')
                output_filename = f"rag_evaluation_{input_filename}_{timestamp}.json"
                output_file_path = os.path.join("results/eval", output_filename)
            
            # 添加元信息
            results.update({
                "evaluation_start_time": self.start_time.isoformat(),
                "evaluation_end_time": datetime.now().isoformat(),
                "input_file": self.input_file_path,
                "output_file": output_file_path
            })
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"评估结果已保存到: {output_file_path}")
            return output_file_path
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            return None
    
    def print_summary(self, results: Dict):
        """打印评估结果摘要"""
        if "summary" not in results:
            return
            
        print("\n评估结果摘要:")
        print("-" * 40)
        
        summary = results["summary"]
        for metric_name, stats in summary.items():
            print(f"{metric_name}:")
            print(f"平均分: {stats['average']:.3f}")
            print(f"最高分: {stats['max']:.3f}")
            print(f"最低分: {stats['min']:.3f}")
            print(f"样本数: {stats['count']}")
            print()
    
    async def run_evaluation(self, output_file: str = None) -> Dict:
        """运行完整评估流程"""
        self.logger.info("开始RAG评估流程...")
        
        try:
            # 执行评估
            results = await self.evaluate_all()
            
            if "error" in results:
                self.logger.error(f"评估失败: {results['error']}")
                return results
            
            # 打印摘要
            self.print_summary(results)

            # 生成输出文件名
            if output_file is None:
                timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
                input_filename = os.path.basename(self.input_file_path).replace('.json', '')
                output_file = f"rag_evaluation_{input_filename}_{timestamp}.json"

            # 确保results目录存在
            os.makedirs("results/eval", exist_ok=True)
            full_path = os.path.join("results/eval", output_file)

            # 保存结果
            saved_file = self.save_results(results, full_path)
            
            self.logger.info("评估完成!")
            return {"results": results, "saved_file": saved_file}
            
        except Exception as e:
            self.logger.error(f"评估过程出错: {e}")
            return {"error": str(e)}


async def main():
    """主函数"""
    evaluator = RAGEvaluator()
    result = await evaluator.run_evaluation()
    
    if "error" in result:
        print(f"评估失败: {result['error']}")
    else:
        print(f"评估成功! 结果保存到: {result['saved_file']}")


if __name__ == "__main__":
    asyncio.run(main())
