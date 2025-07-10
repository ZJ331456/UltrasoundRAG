import sys
import os
from MedicalRAG.config.config import get_image_collection
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.image_utils import get_image_embedding
from MedicalRAG.config.config import config_manager


def find_chinese_font():
    """找到一个可用的中文字体"""
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    # 按顺序检查，找到第一个就返回
    font_check_order = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'Heiti', 'Arial Unicode MS']
    
    for font_name_check in font_check_order:
        for font_path in font_paths:
            try:
                font_name = fm.FontProperties(fname=font_path).get_name()
                if font_name_check in font_name:
                    return font_path
            except:
                continue
    
    # 如果以上都没找到，返回None
    return None

class ImageSearcher:
    """
    一个通过输入图片来查找相似图片的类。
    """
    def __init__(self, image_model_path, chroma_persist_dir, collection_name):
        self.IMAGE_MODEL_PATH = image_model_path
        self.CHROMA_PERSIST_DIR = chroma_persist_dir
        self.COLLECTION_NAME = collection_name
        
        self._load_models()
        self._connect_to_db()

    def _load_models(self):
        print(f"Loading image model from {self.IMAGE_MODEL_PATH}...")
        try:
            self.image_model = CLIPModel.from_pretrained(self.IMAGE_MODEL_PATH)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.IMAGE_MODEL_PATH)
        except Exception as e:
            print(f"Error loading image model: {e}")
            raise

    def _connect_to_db(self):
        print(f"Connecting to ChromaDB...")
        # 直接获取图像集合
        self.collection = get_image_collection()
        print(f"Successfully connected to collection '{self.COLLECTION_NAME}'.")

    def search_by_image(self, image_path: str, top_n: int = 5):
        """
        根据给定的图片路径，在索引中搜索最相似的图片。
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return None

        print(f"\nEncoding query image: '{image_path}'")
        query_embedding = get_image_embedding(image_path, self.image_model, self.image_processor)
        if query_embedding is None:
            return None

        print("Searching for similar images in the database...")
        results = self.collection.query(
            query_embeddings=[query_embedding.flatten().tolist()],
            n_results=top_n + 1 # 请求 N+1 个结果，以防查询本身被过滤
        )
        
        # 如果结果不为空，并且返回的数量超过了请求的数量，则截断
        if results and results['ids'] and len(results['ids'][0]) > top_n:
            for key in results:
                if results[key] and isinstance(results[key], list) and len(results[key]) > 0:
                    results[key] = [lst[:top_n] for lst in results[key]]

        return results


class CaptionImageMatcher:
    """
    一个通过语义搜索图片标题(caption)来查找图片的类。
    它在初始化时加载模型和数据，以提供快速的后续查询。
    """
    def __init__(self, text_embedding_model_path, chroma_persist_dir, collection_name):
        self.TEXT_EMBEDDING_MODEL = text_embedding_model_path
        self.CHROMA_PERSIST_DIR = chroma_persist_dir
        self.COLLECTION_NAME = collection_name
        self.logger = setup_logger(__name__)
        self.initialized = False
        
        self._load_model()
        self._build_caption_index()

    def _load_model(self):
        print(f"Loading text embedding model: {self.TEXT_EMBEDDING_MODEL}...")
        self.text_model = SentenceTransformer(self.TEXT_EMBEDDING_MODEL)

    def _build_caption_index(self):
        self.logger.info(f"Loading image metadata from ChromaDB collection...")
        try:
            image_collection = get_image_collection()
            all_data = image_collection.get(include=["metadatas", "documents"])

            captions_with_metadata = []
            if all_data and all_data.get('metadatas'):
                for meta, doc in zip(all_data['metadatas'], all_data['documents']):
                    # full_caption 优先，兼容旧数据
                    caption = meta.get('full_caption', doc)
                    image_path = meta.get('image_path')
                    if caption and image_path:
                        captions_with_metadata.append({
                            'caption': caption,
                            'image_path': image_path
                        })

            self.captions_with_metadata = captions_with_metadata
            
            if not self.captions_with_metadata:
                self.logger.warning("数据库中未找到有效的图片标题或图片路径，文本搜图功能将不可用。")
                self.caption_collection = None
                self.initialized = False
                return

            all_captions = [item['caption'] for item in self.captions_with_metadata]

            self.logger.info(f"为 {len(all_captions)} 个图片标题创建内存中的语义索引...")
            # 创建临时内存集合用于标题搜索
            import chromadb
            from chromadb.config import Settings
            in_memory_client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
            self.caption_collection = in_memory_client.create_collection(
                name="image_captions_temp",
                metadata={"hnsw:space": "cosine"}
            )

            caption_embeddings = self.text_model.encode(all_captions, convert_to_tensor=False, show_progress_bar=True)
            
            self.caption_collection.add(
                embeddings=caption_embeddings.tolist(),
                documents=all_captions,
                metadatas=[{'image_path': item['image_path']} for item in self.captions_with_metadata],
                ids=[str(i) for i in range(len(all_captions))]
            )
            self.logger.info("图片标题索引构建成功。")
            self.initialized = True

        except Exception as e:
            self.logger.error(f"构建图片标题索引时出错: {e}", exc_info=True)
            self.caption_collection = None
            self.initialized = False

    def search(self, query_text: str, top_n: int = 5):
        """
        根据给定的文本查询，在标题索引中搜索最相似的图片。
        """
        if not self.initialized or not self.caption_collection:
            self.logger.warning("CaptionImageMatcher 未成功初始化，无法执行搜索。")
            return None

        self.logger.info(f"\n编码查询: '{query_text}'")
        query_embedding = self.text_model.encode(query_text, convert_to_tensor=False)

        self.logger.info("搜索相似的图片标题...")
        results = self.caption_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_n
        )
        return results

def display_results(results, image_base_path):
    """通用函数，用于显示文本或图片搜索的结果。"""
    if not results or not results['ids'] or not results['ids'][0]:
        print("没有找到相关的图片。")
        return

    chinese_font_path = find_chinese_font()
    my_font = fm.FontProperties(fname=chinese_font_path) if chinese_font_path else fm.FontProperties()
    if not chinese_font_path:
         print("警告: 未找到中文字体，标题可能无法正确显示。")

    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        caption = metadata.get('full_caption', metadata.get('caption', 'N/A'))
        image_path = os.path.join(image_base_path, metadata['image_path'])

        print(f"\n结果 {i+1}:")
        print(f"  Distance (similarity score): {distance:.4f}")
        print(f"  Caption: {caption}")
        print(f"  Image Path: {image_path}")

        try:
            img = Image.open(image_path)
            plt.figure()
            plt.imshow(img)
            plt.title(caption, fontproperties=my_font, pad=20)
            plt.axis('off')
            plt.figtext(0.5, 0.05, image_path, wrap=True, horizontalalignment='center', fontsize=10, fontproperties=my_font)
            plt.show()
        except FileNotFoundError:
            print(f"  错误: 图片文件未找到 at {image_path}")
        except Exception as e:
            print(f"  错误: 显示图片时出错: {e}")


def main():
    # --- 从配置加载参数 ---
    config = config_manager.config
    search_config = config['image_search']
    
    TEXT_EMBEDDING_MODEL = search_config['text_to_image']['embedding_model']
    IMAGE_MODEL_PATH = search_config['image_to_image']['embedding_model']
    CHROMA_PERSIST_DIR = search_config['vectorstore_path']
    COLLECTION_NAME = search_config['collection_name']
    IMAGE_BASE_PATH = search_config['image_base_path']
    DEFAULT_TOP_N = search_config['top_n']

    while True:
        print("\n--- 请选择搜索模式 ---")
        print("1. 按文本描述搜索")
        print("2. 按图片搜索")
        mode = input("请输入选项 (1 or 2, 输入 'exit' 退出): ")

        if mode.lower() == 'exit':
            break
        
        if mode == '1':
            # --- 文本搜索模式 ---
            try:
                matcher = CaptionImageMatcher(
                    text_embedding_model_path=TEXT_EMBEDDING_MODEL,
                    chroma_persist_dir=CHROMA_PERSIST_DIR,
                    collection_name=COLLECTION_NAME
                )
            except (ValueError, FileNotFoundError) as e:
                print(f"Failed to initialize CaptionImageMatcher: {e}")
                continue

            query_text = input("\n请输入您想查询的图片描述: ")
            top_n_str = input(f"您希望返回多少个结果? (默认: {DEFAULT_TOP_N}): ")
            top_n = int(top_n_str) if top_n_str.isdigit() else DEFAULT_TOP_N

            results = matcher.search(query_text, top_n=top_n)
            print("\n--- 文本搜索结果 ---")
            display_results(results, IMAGE_BASE_PATH)

        elif mode == '2':
            # --- 图片搜索模式 ---
            try:
                searcher = ImageSearcher(
                    image_model_path=IMAGE_MODEL_PATH,
                    chroma_persist_dir=CHROMA_PERSIST_DIR,
                    collection_name=COLLECTION_NAME
                )
            except (ValueError, FileNotFoundError) as e:
                print(f"Failed to initialize ImageSearcher: {e}")
                continue
            
            query_image_path = input("\n请输入查询图片的完整路径: ").strip()
            top_n_str = input(f"您希望返回多少个结果? (默认: {DEFAULT_TOP_N}): ")
            top_n = int(top_n_str) if top_n_str.isdigit() else DEFAULT_TOP_N

            results = searcher.search_by_image(query_image_path, top_n=top_n)
            print("\n--- 图片搜索结果 ---")
            display_results(results, IMAGE_BASE_PATH)

        else:
            print("无效的选项，请输入 1, 2, 或 'exit'.")


if __name__ == "__main__":
    main()