"""
文档更新管理器
处理文档的增量更新和变更管理
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import os
import hashlib
import time
import json

from ..utils import setup_logger
from ..data.stores.milvus_store import MilvusManager
from ..utils.embedding_utils import embedding_provider


@dataclass
class UpdateStats:
    """更新统计信息"""
    total_files: int = 0
    new_files: int = 0
    updated_files: int = 0
    deleted_files: int = 0
    unchanged_files: int = 0
    failed_files: int = 0
    total_documents: int = 0
    added_documents: int = 0
    failed_documents: int = 0
    start_time: float = 0
    end_time: float = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 1.0
        return (self.total_files - self.failed_files) / self.total_files


class DocumentUpdateManager:
    """文档更新管理器"""
    
    def __init__(self, dataset_name: str, dataset_config: Dict[str, Any]):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.logger = setup_logger(f"DocumentUpdater_{dataset_name}")
        
        # 初始化Milvus管理器
        self.milvus_manager = MilvusManager()
        
        # 获取集合名称
        self.collection_name = dataset_config.get('collection_name', f"{dataset_name}_collection")
        
        # 状态跟踪文件
        self.state_file = f".{dataset_name}_update_state.json"
        
        # 加载之前的状态
        self.previous_state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """加载之前的更新状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"加载状态文件失败: {e}")
            return {}
    
    def _save_state(self, state: Dict[str, Any]):
        """保存更新状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存状态文件失败: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取文件信息"""
        try:
            stat = os.stat(file_path)
            return {
                'path': file_path,
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'hash': self._calculate_file_hash(file_path)
            }
        except Exception as e:
            self.logger.error(f"获取文件信息失败 {file_path}: {e}")
            return {'path': file_path, 'size': 0, 'mtime': 0, 'hash': ''}
    
    def scan_files(self, base_path: str, extensions: List[str]) -> Dict[str, Dict[str, Any]]:
        """扫描指定目录下的文件"""
        files_info = {}
        base_path = Path(base_path)
        
        if not base_path.exists():
            self.logger.warning(f"基础路径不存在: {base_path}")
            return files_info
        
        for ext in extensions:
            pattern = f"**/*.{ext.lstrip('.')}"
            for file_path in base_path.glob(pattern):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(base_path))
                    files_info[relative_path] = self._get_file_info(str(file_path))
        
        return files_info
    
    def detect_changes(self, current_files: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
        """检测文件变更"""
        changes = {
            'new': set(),
            'updated': set(),
            'deleted': set(),
            'unchanged': set()
        }
        
        previous_files = self.previous_state.get('files', {})
        
        # 检测新增和更新的文件
        for file_path, file_info in current_files.items():
            if file_path not in previous_files:
                changes['new'].add(file_path)
            elif file_info['hash'] != previous_files[file_path].get('hash', ''):
                changes['updated'].add(file_path)
            else:
                changes['unchanged'].add(file_path)
        
        # 检测删除的文件
        for file_path in previous_files:
            if file_path not in current_files:
                changes['deleted'].add(file_path)
        
        return changes
    
    def update_documents(self, base_path: str, extensions: List[str] = None) -> UpdateStats:
        """更新文档"""
        if extensions is None:
            extensions = ['md', 'txt']
        
        stats = UpdateStats()
        stats.start_time = time.time()
        
        self.logger.info(f"开始更新文档: {self.dataset_name}")
        
        try:
            # 扫描当前文件
            current_files = self.scan_files(base_path, extensions)
            stats.total_files = len(current_files)
            
            # 检测变更
            changes = self.detect_changes(current_files)
            
            stats.new_files = len(changes['new'])
            stats.updated_files = len(changes['updated'])
            stats.deleted_files = len(changes['deleted'])
            stats.unchanged_files = len(changes['unchanged'])
            
            self.logger.info(f"文件变更统计: 新增={stats.new_files}, 更新={stats.updated_files}, "
                           f"删除={stats.deleted_files}, 未变更={stats.unchanged_files}")
            
            # 处理新增和更新的文件
            for file_path in changes['new'] | changes['updated']:
                try:
                    full_path = os.path.join(base_path, file_path)
                    self._process_file(full_path, file_path, current_files[file_path])
                except Exception as e:
                    self.logger.error(f"处理文件失败 {file_path}: {e}")
                    stats.failed_files += 1
            
            # 处理删除的文件
            for file_path in changes['deleted']:
                try:
                    self._delete_file_from_collection(file_path)
                except Exception as e:
                    self.logger.error(f"删除文件失败 {file_path}: {e}")
                    stats.failed_files += 1
            
            # 保存当前状态
            new_state = {
                'files': current_files,
                'last_update': time.time(),
                'dataset_name': self.dataset_name
            }
            self._save_state(new_state)
            
        except Exception as e:
            self.logger.error(f"文档更新过程失败: {e}")
            stats.failed_files = stats.total_files
        
        stats.end_time = time.time()
        
        self.logger.info(f"文档更新完成: 耗时={stats.duration:.2f}s, "
                        f"成功率={stats.success_rate:.2%}")
        
        return stats
    
    def _process_file(self, full_path: str, relative_path: str, file_info: Dict[str, Any]):
        """处理单个文件"""
        self.logger.info(f"处理文件: {relative_path}")
        
        try:
            # 根据文件类型选择处理器
            if relative_path.endswith('.md'):
                self._process_markdown_file(full_path, relative_path)
            else:
                self.logger.warning(f"不支持的文件类型: {relative_path}")
                
        except Exception as e:
            self.logger.error(f"处理文件失败 {relative_path}: {e}")
            raise
    
    def _process_markdown_file(self, full_path: str, relative_path: str):
        """处理Markdown文件"""
        from ..data.loaders.markdown_parser import MarkdownParser
        
        # 创建Markdown解析器
        parser = MarkdownParser(self.dataset_name)
        
        # 解析文件
        chunks = parser.parse_single_file(full_path)
        
        if not chunks:
            self.logger.warning(f"文件解析结果为空: {relative_path}")
            return
        
        # 删除旧数据（如果存在）
        self._delete_file_from_collection(relative_path)
        
        # 插入新数据
        self._insert_chunks_to_collection(chunks, relative_path)
        
        self.logger.info(f"成功处理文件: {relative_path}, 生成 {len(chunks)} 个chunks")
    
    def _insert_chunks_to_collection(self, chunks: List[Dict], relative_path: str):
        """将chunks插入到集合中"""
        try:
            # 准备数据
            data = []
            for chunk in chunks:
                data.append({
                    'content': chunk['content'],
                    'metadata': {
                        'md_file': relative_path,
                        'chunk_id': chunk.get('chunk_id', ''),
                        'section': chunk.get('section', ''),
                        'page': chunk.get('page', 0)
                    }
                })
            
            # 插入到Milvus
            result = self.milvus_manager.insert_documents(self.collection_name, data)
            self.logger.info(f"插入 {len(data)} 个chunks到集合 {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"插入chunks失败: {e}")
            raise
    
    def batch_update_documents(self, max_documents: Optional[int] = None) -> UpdateStats:
        """批量更新文档（增量更新）"""
        base_path = self.dataset_config.get('base_path', '')
        extensions = self.dataset_config.get('file_extensions', ['.md'])
        
        # 转换扩展名格式
        ext_list = [ext.lstrip('.') for ext in extensions]
        
        return self.update_documents(base_path, ext_list)
    
    def _delete_file_from_collection(self, file_path: str):
        """从集合中删除文件"""
        try:
            # 使用现有的删除方法
            success, count = self.milvus_manager.delete_document_chunks(file_path)
            
            if success:
                self.logger.info(f"删除文件 {file_path}: 成功删除 {count} 个chunks")
            else:
                self.logger.warning(f"删除文件 {file_path}: 未找到相关数据")
            
        except Exception as e:
            self.logger.error(f"删除文件失败 {file_path}: {e}")
            raise


def update_document_by_name(dataset_name: str, base_path: str = None, extensions: List[str] = None) -> UpdateStats:
    """根据数据集名称更新文档的便捷函数"""
    try:
        from ..config import config
        
        # 获取数据集配置
        datasets = config.get('indexing', {}).get('markdown', {}).get('datasets', {})
        if dataset_name not in datasets:
            raise ValueError(f"未找到数据集配置: {dataset_name}")
        
        dataset_config = datasets[dataset_name]
        
        # 使用配置中的路径
        if base_path is None:
            base_path = dataset_config.get('base_path', '')
        
        if extensions is None:
            extensions = dataset_config.get('extensions', ['md'])
        
        # 创建更新管理器并执行更新
        updater = DocumentUpdateManager(dataset_name, dataset_config)
        return updater.update_documents(base_path, extensions)
        
    except Exception as e:
        logger = setup_logger("DocumentUpdater")
        logger.error(f"更新文档失败 {dataset_name}: {e}")
        # 返回失败统计
        stats = UpdateStats()
        stats.failed_files = 1
        stats.total_files = 1
        return stats
