"""图像检索模块

主要功能:
本模块提供图像检索的核心功能，支持三种主要的检索方式：
1. **以文搜图(标题匹配)**: 通过文本描述在图像标题中进行语义搜索
2. **以文搜图(CLIP内容)**: 通过CLIP模型直接检索图像内容
3. **以图搜图**: 通过输入图像在向量数据库中查找相似图像

核心组件:
- `BaseImageSearcher`: 图像搜索基类，提供公共功能
- `ImageSearcher`: 图像到图像的相似性搜索
- `CaptionImageMatcher`: 文本到图像标题的语义匹配
- `CLIPTextImageSearcher`: 基于CLIP的文本到图像内容检索
- `UnifiedImageSearcher`: 统一的图像搜索接口
"""
import os
import sys
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import chromadb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPImageProcessor
from chromadb.config import Settings

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MedicalRAG.config.config import get_image_collection, config_manager
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.image_utils import get_image_embedding, load_clip_model


def find_chinese_font() -> Optional[str]:
    """找到一个可用的中文字体"""
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    font_check_order = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'Heiti', 'Arial Unicode MS']
    
    for font_name_check in font_check_order:
        for font_path in font_paths:
            try:
                font_name = fm.FontProperties(fname=font_path).get_name()
                if font_name_check in font_name:
                    return font_path
            except Exception:
                continue
    
    return None


class BaseImageSearcher(ABC):
    """图像搜索基类，提供公共功能"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.initialized = False
        self.config = config_manager.config.get('image_search', {})
    
    def _validate_config(self, required_keys: List[str]) -> bool:
        """验证配置是否包含必需的键"""
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"配置中缺少必需的键: {key}")
                return False
        return True
    
    def _handle_error(self, operation: str, error: Exception) -> None:
        """统一的错误处理"""
        self.logger.error(f"{operation}失败: {error}", exc_info=True)
        self.initialized = False
    
    @abstractmethod
    def search(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """抽象搜索方法"""
        pass

class ImageSearcher(BaseImageSearcher):
    """图像到图像的相似性搜索器"""
    
    def __init__(self, image_model_path: str, config_path: str, chroma_persist_dir: str = None, collection_name: str = None):
        super().__init__()
        self.image_model_path = image_model_path
        self.config_path = config_path
        self.image_model = None
        self.image_processor = None
        self.collection = None
        
        try:
            self._load_models()
            self._connect_to_db()
            self.initialized = True
            self.logger.info("ImageSearcher 初始化成功")
        except Exception as e:
            self._handle_error("ImageSearcher 初始化", e)
    
    def _load_models(self) -> None:
        """加载图像模型"""
        self.image_model, self.image_processor = load_clip_model(self.image_model_path, self.config_path)
        if not self.image_model:
            raise RuntimeError(f"无法加载模型: {self.image_model_path}")
    
    def _connect_to_db(self) -> None:
        """连接到数据库"""
        self.logger.info("正在连接到 ChromaDB")
        self.collection = get_image_collection()
        self.logger.info("成功连接到图像集合")
    
    def search(self, image_path: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据图像路径搜索相似图像"""
        return self.search_by_image(image_path, top_n)
    
    def search_by_image(self, image_path: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据给定的图片路径，在索引中搜索最相似的图片"""
        if not self.initialized:
            self.logger.warning("ImageSearcher 未正确初始化")
            return None
        
        if not os.path.exists(image_path):
            self.logger.error(f"图像文件不存在: {image_path}")
            return None
        
        try:
            self.logger.info(f"正在编码查询图像: {image_path}")
            query_embedding = get_image_embedding(image_path, self.image_model, self.image_processor)
            if query_embedding is None:
                return None
            
            self.logger.info("正在数据库中搜索相似图像")
            results = self.collection.query(
                query_embeddings=[query_embedding.flatten().tolist()],
                n_results=top_n + 1  # 请求 N+1 个结果，以防查询本身被过滤
            )
            
            # 截断结果到指定数量
            if results and results.get('ids') and len(results['ids'][0]) > top_n:
                for key in results:
                    if results[key] and isinstance(results[key], list) and len(results[key]) > 0:
                        results[key] = [lst[:top_n] for lst in results[key]]
            
            return results
            
        except Exception as e:
            self._handle_error("图像搜索", e)
            return None


class CaptionImageMatcher(BaseImageSearcher):
    """文本到图像的语义匹配搜索器"""
    
    def __init__(self, text_embedding_model_path: str, chroma_persist_dir: str = None, collection_name: str = None):
        super().__init__()
        self.text_embedding_model_path = text_embedding_model_path
        self.text_model = None
        self.caption_collection = None
        self.captions_with_metadata = []
        
        try:
            self._load_model()
            self._build_caption_index()
            self.logger.info("CaptionImageMatcher 初始化成功")
        except Exception as e:
            self._handle_error("CaptionImageMatcher 初始化", e)
    
    def _load_model(self) -> None:
        """加载文本嵌入模型"""
        self.logger.info(f"正在加载文本嵌入模型: {self.text_embedding_model_path}")
        self.text_model = SentenceTransformer(self.text_embedding_model_path)
    
    def _build_caption_index(self) -> None:
        """构建图片标题索引"""
        self.logger.info("正在从 ChromaDB 加载图像元数据")
        
        image_collection = get_image_collection()
        all_data = image_collection.get(include=["metadatas", "documents"])
        
        # 提取标题和元数据
        if all_data and all_data.get('metadatas'):
            for meta, doc in zip(all_data['metadatas'], all_data['documents']):
                caption = meta.get('full_caption', doc)
                image_path = meta.get('image_path')
                if caption and image_path:
                    self.captions_with_metadata.append({
                        'caption': caption,
                        'image_path': image_path
                    })
        
        if not self.captions_with_metadata:
            self.logger.warning("数据库中未找到有效的图片标题或图片路径，文本搜图功能将不可用")
            self.initialized = False
            return
        
        # 创建内存中的标题索引
        all_captions = [item['caption'] for item in self.captions_with_metadata]
        self.logger.info(f"为 {len(all_captions)} 个图片标题创建内存中的语义索引")
        
        in_memory_client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
        self.caption_collection = in_memory_client.create_collection(
            name="image_captions_temp",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 编码标题并添加到集合
        caption_embeddings = self.text_model.encode(
            all_captions, 
            convert_to_tensor=False, 
            show_progress_bar=True
        )
        
        self.caption_collection.add(
            embeddings=caption_embeddings.tolist(),
            documents=all_captions,
            metadatas=[{'image_path': item['image_path']} for item in self.captions_with_metadata],
            ids=[str(i) for i in range(len(all_captions))]
        )
        
        self.logger.info("图片标题索引构建成功")
        self.initialized = True
    
    def search(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据文本查询搜索相关图像"""
        if not self.initialized or not self.caption_collection:
            self.logger.warning("CaptionImageMatcher 未成功初始化，无法执行搜索")
            return None
        
        try:
            self.logger.info(f"正在编码查询: {query_text}")
            query_embedding = self.text_model.encode(query_text, convert_to_tensor=False)
            
            self.logger.info("正在搜索相似的图片标题")
            results = self.caption_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_n
            )
            return results
            
        except Exception as e:
            self._handle_error("文本搜索图像", e)
            return None


class CLIPTextImageSearcher(BaseImageSearcher):
    """基于CLIP的文本到图像内容检索器"""
    
    def __init__(self, image_model_path: str, config_path: str, chroma_persist_dir: str = None, collection_name: str = None):
        super().__init__()
        self.image_model_path = image_model_path
        self.config_path = config_path
        self.clip_model = None
        self.clip_processor = None
        self.collection = None
        
        try:
            self._load_clip_model()
            self._connect_to_db()
            self.initialized = True
            self.logger.info("CLIPTextImageSearcher 初始化成功")
        except Exception as e:
            self._handle_error("CLIPTextImageSearcher 初始化", e)
    
    def _load_clip_model(self) -> None:
        """加载CLIP模型"""
        self.logger.info(f"正在加载FetalCLIP模型: {self.image_model_path}")
        from ..model.fetal_clip_model import load_fetal_clip_model
        self.clip_model = load_fetal_clip_model(self.image_model_path, self.config_path)
        if not self.clip_model:
            raise RuntimeError(f"无法加载FetalCLIP模型: {self.image_model_path}")
        self.logger.info("FetalCLIP模型加载成功")
    
    def _connect_to_db(self) -> None:
        """连接到数据库"""
        self.logger.info("正在连接到 ChromaDB 图像集合")
        self.collection = get_image_collection()
        self.logger.info("成功连接到图像集合")
    
    def search(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据文本查询直接检索图像内容"""
        return self.search_by_clip_text(query_text, top_n)
    
    def search_by_clip_text(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """使用CLIP模型根据文本直接检索图像内容"""
        if not self.initialized:
            self.logger.warning("CLIPTextImageSearcher 未正确初始化")
            return None
        
        try:
            self.logger.info(f"正在使用FetalCLIP编码查询文本: {query_text}")
            
            # 使用FetalCLIP模型编码文本
            import torch
            
            # 使用FetalCLIP的tokenizer对文本进行分词
            text_tokens = self.clip_model.tokenize_text([query_text])
            
            # 使用FetalCLIP模型编码文本
            text_features = self.clip_model.encode_text(text_tokens)
            
            # 转换为列表格式用于ChromaDB查询
            query_embedding = text_features.squeeze().cpu().numpy().tolist()
            
            self.logger.info("正在数据库中搜索相似图像")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_n
            )
            
            self.logger.info(f"CLIP文本搜索完成，找到 {len(results.get('ids', [[]])[0])} 个结果")
            return results
            
        except Exception as e:
            self._handle_error("CLIP文本搜索图像", e)
            return None


class ImageDisplayer:
    """图像结果显示器"""
    
    def __init__(self, image_base_path: str):
        self.image_base_path = image_base_path
        self.font = self._setup_font()
        self.logger = setup_logger(self.__class__.__name__)
    
    def _setup_font(self) -> fm.FontProperties:
        """设置中文字体"""
        chinese_font_path = find_chinese_font()
        if chinese_font_path:
            return fm.FontProperties(fname=chinese_font_path)
        else:
            print("警告: 未找到中文字体，标题可能无法正确显示")
            return fm.FontProperties()
    
    def display_results(self, results: Optional[Dict[str, Any]]) -> None:
        """显示搜索结果"""
        if not results or not results.get('ids') or not results['ids'][0]:
            print("没有找到相关的图片")
            return
        
        for i in range(len(results['ids'][0])):
            self._display_single_result(results, i)
    
    def _display_single_result(self, results: Dict[str, Any], index: int) -> None:
        """显示单个搜索结果"""
        distance = results['distances'][0][index]
        metadata = results['metadatas'][0][index]
        caption = metadata.get('full_caption', metadata.get('caption', 'N/A'))
        image_path = os.path.join(self.image_base_path, metadata['image_path'])
        
        print(f"\n结果 {index + 1}:")
        print(f"  相似度分数: {1 - distance:.4f}")
        print(f"  图片描述: {caption}")
        print(f"  图片路径: {image_path}")
        
        self._show_image(image_path, caption)
    
    def _show_image(self, image_path: str, caption: str) -> None:
        """显示图片"""
        try:
            img = Image.open(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(caption, fontproperties=self.font, pad=20)
            plt.axis('off')
            plt.figtext(0.5, 0.02, image_path, wrap=True, 
                       horizontalalignment='center', fontsize=8, 
                       fontproperties=self.font)
            plt.tight_layout()
            plt.show()
        except FileNotFoundError:
            self.logger.error(f"图片文件未找到: {image_path}")
        except Exception as e:
            self.logger.error(f"显示图片时出错: {e}")


class ImageSearchCLI:
    """图像搜索命令行界面"""
    
    def __init__(self):
        self.config = config_manager.config.get('image_search', {})
        self.logger = setup_logger(self.__class__.__name__)
        self.displayer = ImageDisplayer(self.config.get('image_base_path', ''))
        
        # 从配置加载参数
        self.text_model_path = self.config.get('text_to_image', {}).get('embedding_model')
        self.image_model_path = self.config.get('image_to_image', {}).get('embedding_model')
        self.image_config_path = self.config.get('image_to_image', {}).get('config_path')
        self.default_top_n = self.config.get('top_n', 10)
    
    def run(self) -> None:
        """运行交互式搜索界面"""
        print("\n=== 图像搜索系统 ===")
        
        while True:
            try:
                mode = self._get_search_mode()
                if mode == 'exit':
                    break
                elif mode == '1':
                    self._text_caption_search_mode()
                elif mode == '2':
                    self._text_clip_search_mode()
                elif mode == '3':
                    self._image_search_mode()
                else:
                    print("无效的选项，请输入 1, 2, 3, 或 'exit'")
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                self.logger.error(f"运行时错误: {e}", exc_info=True)
                print(f"发生错误: {e}")
    
    def _get_search_mode(self) -> str:
        """获取搜索模式"""
        print("\n--- 请选择搜索模式 ---")
        print("1. 按文本描述搜索图片标题")
        print("2. 按文本描述搜索图片内容(CLIP)")
        print("3. 按图片搜索")
        return input("请输入选项 (1, 2, 3, 输入 'exit' 退出): ").strip().lower()
    
    def _get_top_n(self) -> int:
        """获取返回结果数量"""
        top_n_str = input(f"您希望返回多少个结果? (默认: {self.default_top_n}): ").strip()
        return int(top_n_str) if top_n_str.isdigit() else self.default_top_n
    
    def _text_caption_search_mode(self) -> None:
        """基于图片标题的文本搜索模式"""
        try:
            matcher = CaptionImageMatcher(self.text_model_path)
            if not matcher.initialized:
                print("文本搜索器初始化失败")
                return
            
            query_text = input("\n请输入您想查询的图片描述: ").strip()
            if not query_text:
                print("查询文本不能为空")
                return
            
            top_n = self._get_top_n()
            results = matcher.search(query_text, top_n=top_n)
            
            print("\n--- 文本搜索结果 ---")
            self.displayer.display_results(results)
            
        except Exception as e:
            self.logger.error(f"文本搜索失败: {e}", exc_info=True)
            print(f"文本搜索失败: {e}")
    
    def _text_clip_search_mode(self) -> None:
        """基于CLIP模型的文本搜索模式"""
        try:
            clip_searcher = CLIPTextImageSearcher(self.image_model_path, self.image_config_path)
            if not clip_searcher.initialized:
                print("CLIP文本搜索器初始化失败")
                return
            
            query_text = input("\n请输入您想查询的图片内容描述: ").strip()
            if not query_text:
                print("查询文本不能为空")
                return
            
            top_n = self._get_top_n()
            results = clip_searcher.search_by_clip_text(query_text, top_n=top_n)
            
            print("\n--- CLIP文本搜索结果 ---")
            self.displayer.display_results(results)
            
        except Exception as e:
            self.logger.error(f"CLIP文本搜索失败: {e}", exc_info=True)
            print(f"CLIP文本搜索失败: {e}")
    
    def _image_search_mode(self) -> None:
        """图片搜索模式"""
        try:
            searcher = ImageSearcher(self.image_model_path, self.image_config_path)
            if not searcher.initialized:
                print("图像搜索器初始化失败")
                return
            
            query_image_path = input("\n请输入查询图片的完整路径: ").strip()
            if not query_image_path:
                print("图片路径不能为空")
                return
            
            top_n = self._get_top_n()
            results = searcher.search_by_image(query_image_path, top_n=top_n)
            
            print("\n--- 图片搜索结果 ---")
            self.displayer.display_results(results)
            
        except Exception as e:
            self.logger.error(f"图片搜索失败: {e}", exc_info=True)
            print(f"图片搜索失败: {e}")


class UnifiedImageSearcher:
    """统一的图像搜索器，整合了以图搜图和多种以文搜图功能"""
    
    def __init__(self, image_model_path: str, config_path: str, text_embedding_model_path: str):
        self.logger = setup_logger(self.__class__.__name__)
        self.initialized = False
        
        try:
            # 初始化三个搜索器
            self.image_searcher = ImageSearcher(image_model_path, config_path)
            self.caption_matcher = CaptionImageMatcher(text_embedding_model_path)
            self.clip_text_searcher = CLIPTextImageSearcher(image_model_path, config_path)
            
            # 只有当所有搜索器都初始化成功时，才认为统一搜索器初始化成功
            self.initialized = (
                self.image_searcher.initialized and 
                self.caption_matcher.initialized and
                self.clip_text_searcher.initialized
            )
            
            if self.initialized:
                self.logger.info("UnifiedImageSearcher 初始化成功")
            else:
                self.logger.warning("UnifiedImageSearcher 部分组件初始化失败")
                
        except Exception as e:
            self.logger.error(f"UnifiedImageSearcher 初始化失败: {e}", exc_info=True)
            self.initialized = False
    
    def search_by_image(self, image_path: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据图像搜索相似图像"""
        if not self.initialized or not self.image_searcher.initialized:
            self.logger.warning("图像搜索器未正确初始化")
            return None
        
        return self.image_searcher.search_by_image(image_path, top_n)
    
    def search_by_text_caption(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据文本在图像标题中搜索相关图像"""
        if not self.initialized or not self.caption_matcher.initialized:
            self.logger.warning("文本标题搜索器未正确初始化")
            return None
        
        return self.caption_matcher.search(query_text, top_n)
    
    def search_by_text_clip(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """使用CLIP模型根据文本直接检索图像内容"""
        if not self.initialized or not self.clip_text_searcher.initialized:
            self.logger.warning("CLIP文本搜索器未正确初始化")
            return None
        
        return self.clip_text_searcher.search_by_clip_text(query_text, top_n)
    
    # 保持向后兼容性的方法
    def search_by_text(self, query_text: str, top_n: int = 10) -> Optional[Dict[str, Any]]:
        """根据文本搜索相关图像（默认使用标题匹配，保持向后兼容性）"""
        return self.search_by_text_caption(query_text, top_n)
    
    def is_ready(self) -> bool:
        """检查搜索器是否准备就绪"""
        return self.initialized
    
    def get_status(self) -> Dict[str, bool]:
        """获取各组件的状态"""
        return {
            'unified_searcher': self.initialized,
            'image_searcher': getattr(self.image_searcher, 'initialized', False),
            'caption_matcher': getattr(self.caption_matcher, 'initialized', False),
            'clip_text_searcher': getattr(self.clip_text_searcher, 'initialized', False)
        }


if __name__ == "__main__":
    cli = ImageSearchCLI()
    cli.run()