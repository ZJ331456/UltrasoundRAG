import sys
import os
import json
from tqdm import tqdm
from MedicalRAG.config.config import get_image_collection
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image

# 将项目根目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MedicalRAG.utils.image_utils import get_image_embedding
from MedicalRAG.config.config import config_manager

def main():
    # --- 从配置加载参数 ---
    config = config_manager.config
    image_config = config['indexing']['image']
    
    IMAGE_MODEL_PATH = image_config['model_path']
    IMAGE_NODES_FILE = image_config['nodes_file']
    IMAGE_BASE_PATH = image_config['base_path']
    CHROMA_PERSIST_DIR = image_config['vectorstore_path']
    COLLECTION_NAME = image_config['collection_name']
    BATCH_SIZE = image_config['batch_size']

    print(f"Loading image model from {IMAGE_MODEL_PATH}...")
    try:
        # 加载图像处理模型和处理器
        image_model = CLIPModel.from_pretrained(IMAGE_MODEL_PATH)
        image_processor = CLIPImageProcessor.from_pretrained(IMAGE_MODEL_PATH)
    except Exception as e:
        print(f"Error loading image model: {e}")
        return

    print("Initializing ChromaDB...")
    # 使用统一的ChromaDB管理器
    collection = get_image_collection()
    print(f"ChromaDB collection loaded/created successfully.")
    
    print(f"Current documents in collection: {collection.count()}")

    print(f"Reading image nodes from {IMAGE_NODES_FILE}...")
    with open(IMAGE_NODES_FILE, 'r', encoding='utf-8') as f:
        image_nodes = [json.loads(line) for line in f]

    print(f"Found {len(image_nodes)} images to process.")

    for i in tqdm(range(0, len(image_nodes), BATCH_SIZE), desc="Indexing Images"):
        batch_nodes = image_nodes[i:i+BATCH_SIZE]
        
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