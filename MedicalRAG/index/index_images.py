import sys
import os
import json
from tqdm import tqdm
import chromadb
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image

# 将项目根目录添加到 a a a sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MedicalRAG.utils.image_utils import get_image_embedding

def main():
    # --- 配置参数 ---
    IMAGE_MODEL_PATH = "models/clip-vit-large-patch14"
    IMAGE_NODES_FILE = "data/processed/image/image_nodes.jsonl"
    IMAGE_BASE_PATH = "data/processed/image"
    CHROMA_PERSIST_DIR = "data/image_vectorstore_clip_taiyi" # 使用新的DB目录
    COLLECTION_NAME = "image_features_clip_vit"

    print(f"Loading image model from {IMAGE_MODEL_PATH}...")
    try:
        # 加载图像处理模型和处理器
        image_model = CLIPModel.from_pretrained(IMAGE_MODEL_PATH)
        image_processor = CLIPImageProcessor.from_pretrained(IMAGE_MODEL_PATH)
    except Exception as e:
        print(f"Error loading image model: {e}")
        return

    print("Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # 检查集合是否存在，如果存在则删除重建
    existing_collections = [col.name for col in chroma_client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
        chroma_client.delete_collection(name=COLLECTION_NAME)

    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    print(f"ChromaDB collection '{COLLECTION_NAME}' created/reset successfully.")

    print(f"Reading image nodes from {IMAGE_NODES_FILE}...")
    with open(IMAGE_NODES_FILE, 'r', encoding='utf-8') as f:
        image_nodes = [json.loads(line) for line in f]

    print(f"Found {len(image_nodes)} images to process.")

    batch_size = 32
    for i in tqdm(range(0, len(image_nodes), batch_size), desc="Indexing Images"):
        batch_nodes = image_nodes[i:i+batch_size]
        
        embeddings = []
        documents = []
        metadatas = []
        ids = []

        for idx, node in enumerate(batch_nodes):
            image_path = os.path.join(IMAGE_BASE_PATH, node['image_path'])
            
            if not os.path.exists(image_path):
                print(f"Image not found, skipping: {image_path}")
                continue

            caption_text = node.get("caption", "")
            if not caption_text:
                print(f"Caption is empty for image {image_path}, skipping.")
                continue

            # 嵌入图片
            image_embedding = get_image_embedding(image_path, image_model, image_processor)
            if image_embedding is None:
                continue
            
            embeddings.append(image_embedding.flatten().tolist())
            documents.append(caption_text) # 将caption作为document存入
            metadatas.append({
                "image_path": node['image_path'],
                "source": node.get('source', ''),
                "full_caption": caption_text
            })
            ids.append(f"image_{i+idx}")

        if embeddings:
            try:
                collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {e}")

    print("\nImage indexing complete.")
    print(f"Total documents in collection: {collection.count()}")
    print(f"ChromaDB data is persisted in: {CHROMA_PERSIST_DIR}")

if __name__ == "__main__":
    main() 