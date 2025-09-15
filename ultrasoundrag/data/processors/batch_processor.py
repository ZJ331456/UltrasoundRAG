"""
批量处理器

负责数据的批量处理管道：
- 批量数据加载
- 并行处理管道
- 进度跟踪和错误处理
- 结果聚合和输出
"""

import os
import time
from typing import List, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    data: Any
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict = None

@dataclass
class BatchConfig:
    """批量处理配置"""
    batch_size: int = 32
    max_workers: int = 4
    use_multiprocessing: bool = False
    progress_bar: bool = True
    error_handling: str = "continue"  # "continue", "stop", "retry"

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        初始化批量处理器
        
        Args:
            config: 批量处理配置
        """
        self.config = config or BatchConfig()
        self.results = []
        self.errors = []
    
    def process_batch(self, 
                     data: List[Any],
                     processor_func: Callable,
                     **kwargs) -> List[ProcessingResult]:
        """
        批量处理数据
        
        Args:
            data: 数据列表
            processor_func: 处理函数
            **kwargs: 处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        if not data:
            return []
        
        self.results = []
        self.errors = []
        
        # 分批处理
        batches = self._create_batches(data, self.config.batch_size)
        
        if self.config.use_multiprocessing and len(batches) > 1:
            return self._process_parallel(batches, processor_func, **kwargs)
        else:
            return self._process_sequential(batches, processor_func, **kwargs)
    
    def _create_batches(self, data: List[Any], batch_size: int) -> List[List[Any]]:
        """创建批次"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _process_sequential(self, 
                           batches: List[List[Any]],
                           processor_func: Callable,
                           **kwargs) -> List[ProcessingResult]:
        """顺序处理批次"""
        all_results = []
        
        # 创建进度条
        if self.config.progress_bar:
            pbar = tqdm(total=sum(len(batch) for batch in batches), 
                       desc="处理数据")
        
        for batch in batches:
            batch_results = self._process_single_batch(batch, processor_func, **kwargs)
            all_results.extend(batch_results)
            
            if self.config.progress_bar:
                pbar.update(len(batch))
        
        if self.config.progress_bar:
            pbar.close()
        
        return all_results
    
    def _process_parallel(self, 
                         batches: List[List[Any]],
                         processor_func: Callable,
                         **kwargs) -> List[ProcessingResult]:
        """并行处理批次"""
        all_results = []
        
        # 选择执行器类型
        if self.config.use_multiprocessing:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self._process_single_batch, batch, processor_func, **kwargs): batch
                for batch in batches
            }
            
            # 创建进度条
            if self.config.progress_bar:
                pbar = tqdm(total=sum(len(batch) for batch in batches), 
                           desc="并行处理数据")
            
            # 收集结果
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    if self.config.progress_bar:
                        pbar.update(len(future_to_batch[future]))
                except Exception as e:
                    print(f"批次处理失败: {e}")
                    self.errors.append(str(e))
            
            if self.config.progress_bar:
                pbar.close()
        
        return all_results
    
    def _process_single_batch(self, 
                             batch: List[Any],
                             processor_func: Callable,
                             **kwargs) -> List[ProcessingResult]:
        """处理单个批次"""
        results = []
        
        for item in batch:
            start_time = time.time()
            
            try:
                # 调用处理函数
                processed_data = processor_func(item, **kwargs)
                processing_time = time.time() - start_time
                
                result = ProcessingResult(
                    success=True,
                    data=processed_data,
                    processing_time=processing_time,
                    metadata={'item_type': type(item).__name__}
                )
                results.append(result)
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                
                result = ProcessingResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    processing_time=processing_time,
                    metadata={'item_type': type(item).__name__}
                )
                results.append(result)
                self.errors.append(error_msg)
                
                # 根据错误处理策略决定是否继续
                if self.config.error_handling == "stop":
                    break
        
        return results
    
    def process_files(self, 
                     file_paths: List[str],
                     processor_func: Callable,
                     **kwargs) -> List[ProcessingResult]:
        """
        批量处理文件
        
        Args:
            file_paths: 文件路径列表
            processor_func: 文件处理函数
            **kwargs: 处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        # 过滤存在的文件
        valid_files = [f for f in file_paths if os.path.exists(f)]
        
        if not valid_files:
            print("没有找到有效的文件")
            return []
        
        print(f"找到 {len(valid_files)} 个有效文件，开始处理...")
        
        return self.process_batch(valid_files, processor_func, **kwargs)
    
    def process_texts(self, 
                     texts: List[str],
                     processor_func: Callable,
                     **kwargs) -> List[ProcessingResult]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            processor_func: 文本处理函数
            **kwargs: 处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        return self.process_batch(texts, processor_func, **kwargs)
    
    def process_images(self, 
                      image_paths: List[str],
                      processor_func: Callable,
                      **kwargs) -> List[ProcessingResult]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            processor_func: 图像处理函数
            **kwargs: 处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        return self.process_files(image_paths, processor_func, **kwargs)
    
    def get_successful_results(self) -> List[ProcessingResult]:
        """获取成功的结果"""
        return [r for r in self.results if r.success]
    
    def get_failed_results(self) -> List[ProcessingResult]:
        """获取失败的结果"""
        return [r for r in self.results if not r.success]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        if not self.results:
            return {}
        
        total_count = len(self.results)
        success_count = len(self.get_successful_results())
        failed_count = len(self.get_failed_results())
        
        total_time = sum(r.processing_time for r in self.results)
        avg_time = total_time / total_count if total_count > 0 else 0
        
        return {
            'total_count': total_count,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'total_time': total_time,
            'average_time': avg_time,
            'error_count': len(self.errors)
        }
    
    def save_results(self, 
                    output_path: str,
                    format: str = "json") -> bool:
        """
        保存处理结果
        
        Args:
            output_path: 输出路径
            format: 输出格式 ("json", "csv", "pickle")
            
        Returns:
            是否保存成功
        """
        try:
            import json
            import pandas as pd
            import pickle
            
            # 准备数据
            data = []
            for result in self.results:
                item = {
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'error': result.error,
                    'metadata': result.metadata
                }
                data.append(item)
            
            # 根据格式保存
            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif format == "csv":
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif format == "pickle":
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            print(f"结果已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False
