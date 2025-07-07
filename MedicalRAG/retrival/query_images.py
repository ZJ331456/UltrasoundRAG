import sys
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 移除 get_text_embedding 的导入，因为它在当前逻辑中不再使用
# from MedicalRAG.utils.image_utils import get_text_embedding

def find_chinese_font():
    """找到一个可用的中文字体"""
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    # 移除 'Sans' 以避免错误匹配，优先使用常见的 Windows 字体
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

class CaptionImageMatcher:
    """
    一个通过语义搜索图片标题(caption)来查找图片的类。
    它在初始化时加载模型和数据，以提供快速的后续查询。
    """
    def __init__(self, text_embedding_model_path, chroma_persist_dir, collection_name):
        self.TEXT_EMBEDDING_MODEL = text_embedding_model_path
        self.CHROMA_PERSIST_DIR = chroma_persist_dir
        self.COLLECTION_NAME = collection_name
        
        self._load_model()
        self._build_caption_index()

    def _load_model(self):
        print(f"Loading text embedding model: {self.TEXT_EMBEDDING_MODEL}...")
        self.text_model = SentenceTransformer(self.TEXT_EMBEDDING_MODEL)

    def _build_caption_index(self):
        print(f"Loading image metadata from ChromaDB collection '{self.COLLECTION_NAME}'...")
        persistent_client = chromadb.PersistentClient(path=self.CHROMA_PERSIST_DIR)
        try:
            image_collection = persistent_client.get_collection(name=self.COLLECTION_NAME)
            all_data = image_collection.get(include=["metadatas"])
        except ValueError:
            print(f"Error: Collection '{self.COLLECTION_NAME}' not found at '{self.CHROMA_PERSIST_DIR}'.")
            raise

        captions_with_metadata = [
            {'caption': item.get('full_caption', item.get('caption', '')), 'image_path': item.get('image_path', '')}
            for item in all_data['metadatas']
        ]
        
        self.captions_with_metadata = [item for item in captions_with_metadata if item['caption'] and item['image_path']]
        
        if not self.captions_with_metadata:
            raise ValueError("No valid captions with image paths found in the database.")

        all_captions = [item['caption'] for item in self.captions_with_metadata]

        print(f"Creating in-memory semantic index for {len(all_captions)} captions...")
        in_memory_client = chromadb.Client()
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
        print("Caption index built successfully.")

    def search(self, query_text: str, top_n: int = 5):
        """
        根据给定的文本查询，在标题索引中搜索最相似的图片。
        """
        print(f"\nEncoding query: '{query_text}'")
        query_embedding = self.text_model.encode(query_text, convert_to_tensor=False)

        print("Searching for similar captions...")
        results = self.caption_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_n
        )
        return results

def main():
    # --- 配置参数 ---
    TEXT_EMBEDDING_MODEL = "models/bge-small-zh-v1.5"
    CHROMA_PERSIST_DIR = "data/image_vectorstore_clip_taiyi"
    COLLECTION_NAME = "image_features_clip_vit"
    IMAGE_BASE_PATH = "data/processed/image"

    try:
        # 1. 初始化匹配器 (加载模型和数据)
        matcher = CaptionImageMatcher(
            text_embedding_model_path=TEXT_EMBEDDING_MODEL,
            chroma_persist_dir=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Failed to initialize CaptionImageMatcher: {e}")
        return

    # --- 进入交互式查询循环 ---
    while True:
        query_text = input("\n请输入您想查询的图片描述 (输入 'exit' 退出): ")
        if query_text.lower() == 'exit':
            break
        top_n_str = input("您希望返回多少个结果? (默认: 5): ")
        top_n = int(top_n_str) if top_n_str.isdigit() else 5

        # 2. 执行搜索
        results = matcher.search(query_text, top_n=top_n)

        # 3. 处理并显示结果
        print("\n--- 查询结果 ---")
        if not results['ids'] or not results['ids'][0]:
            print("没有找到相关的图片。")
            continue

        chinese_font_path = find_chinese_font()
        my_font = fm.FontProperties(fname=chinese_font_path) if chinese_font_path else fm.FontProperties()
        if not chinese_font_path:
             print("警告: 未找到中文字体，标题可能无法正确显示。")

        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            caption = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            image_path = os.path.join(IMAGE_BASE_PATH, metadata['image_path'])

            print(f"\n结果 {i+1}:")
            print(f"  Distance (text similarity): {distance:.4f}")
            print(f"  Matched Caption: {caption}")
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

if __name__ == "__main__":
    main() 