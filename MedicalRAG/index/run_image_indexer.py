"""
图片索引脚本（Medical RAG）

本脚本处理图片节点文件，将图片嵌入向量并索引到 ChromaDB 向量库。
"""

import os
import json
import hashlib
from tqdm import tqdm
from MedicalRAG.config.config import get_image_collection
from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.image_utils import load_clip_model, get_image_embedding
from MedicalRAG.utils.file_util import force_remove_directory, ensure_directory_exists


def run_image_indexer():
    """
    索引 nodes 文件中指定的图片到 ChromaDB 向量库。
    """
    logger = setup_logger(__name__)
    logger.info("开始图片索引...")
    
    # --- 从配置加载参数 ---
    image_config = config['indexing']['image']
    
    IMAGE_MODEL_PATH = image_config['model_path']
    IMAGE_CONFIG_PATH = image_config['config_path']
    IMAGE_NODES_FILE = image_config['nodes_file']
    IMAGE_BASE_PATH = image_config['base_path']
    CHROMA_PERSIST_DIR = image_config['vectorstore_path']
    COLLECTION_NAME = image_config['collection_name']
    BATCH_SIZE = image_config['batch_size']

    logger.info(f"从 {IMAGE_MODEL_PATH} 加载图片模型...")
    try:
        # 使用新的FetalCLIP模型加载器
        # 需要从配置中获取config_path
        image_model, image_processor = load_clip_model(IMAGE_MODEL_PATH, IMAGE_CONFIG_PATH)
        if image_model is None:
            logger.error("Failed to load FetalCLIP model")
            return
    except Exception as e:
        logger.error(f"Error loading image model: {e}")
        return

    # 完全删除旧的向量数据库文件夹以避免重复内容
    logger.info("开始清理旧的图片向量数据库文件夹...")
    
    # 使用强制删除函数处理文件被占用的情况
    image_delete_success = force_remove_directory(CHROMA_PERSIST_DIR, logger)
    if not image_delete_success:
        logger.error(f"无法删除图片向量数据库文件夹: {CHROMA_PERSIST_DIR}")
        raise RuntimeError(f"无法删除图片向量数据库文件夹，请手动关闭占用该文件的程序后重试")
    
    logger.info("图片向量数据库文件夹清理完成")
    
    # 确保目录存在
    ensure_directory_exists(CHROMA_PERSIST_DIR)

    logger.info("初始化 ChromaDB（图片）...")
    
    # 使用统一的ChromaDB管理器
    collection = get_image_collection()
    logger.info(f"成功加载/创建 ChromaDB 图片集合。")

    logger.info(f"从 {IMAGE_NODES_FILE} 读取图片节点...")
    if not os.path.exists(IMAGE_NODES_FILE):
        logger.error(f"未找到图片节点文件: {IMAGE_NODES_FILE}，终止图片索引。")
        return
        
    with open(IMAGE_NODES_FILE, 'r', encoding='utf-8') as f:
        image_nodes = [json.loads(line) for line in f]

    logger.info(f"共需处理 {len(image_nodes)} 张图片。")

    for i in tqdm(range(0, len(image_nodes), BATCH_SIZE), desc="Indexing Images"):
        batch_nodes = image_nodes[i:i+BATCH_SIZE]
        
        # 使用字典来处理批次内的潜在重复项
        batch_data = {}

        for idx, node in enumerate(batch_nodes):
            # 兼容绝对路径和相对路径
            image_path = node['image_path']
            if not os.path.isabs(image_path):
                image_path = os.path.join(IMAGE_BASE_PATH, image_path)
            
            if not os.path.exists(image_path):
                logger.warning(f"未找到图片，跳过: {image_path}")
                continue

            caption_text = node.get("caption", "")
            if not caption_text:
                logger.warning(f"图片 {image_path} 的描述为空，跳过。")
                continue

            # 嵌入图片
            image_embedding = get_image_embedding(image_path, image_model, image_processor)
            if image_embedding is None:
                continue
            
            # 使用 image_path 的哈希值作为唯一ID
            image_path_hash = hashlib.sha256(node['image_path'].encode()).hexdigest()

            # 将数据存入字典，以ID为键，自动去重
            batch_data[image_path_hash] = {
                "embedding": image_embedding.flatten().tolist(),
                "document": caption_text,
                "metadata": {
                    "image_path": node['image_path'], # 存储相对路径
                    "source": node.get('source', ''),
                    "full_caption": caption_text
                }
            }

        if batch_data:
            # 从去重后的字典中提取数据
            ids = list(batch_data.keys())
            embeddings = [data['embedding'] for data in batch_data.values()]
            documents = [data['document'] for data in batch_data.values()]
            metadatas = [data['metadata'] for data in batch_data.values()]

            try:
                collection.upsert(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                logger.error(f"批量写入 ChromaDB 时出错: {e}")

    logger.info("图片索引完成。")
    logger.info(f"图片集合总文档数: {collection.count()}")
    logger.info(f"ChromaDB 图片数据已持久化于: {CHROMA_PERSIST_DIR}")


# def main():
#     """主函数"""
#     try:
#         run_image_indexer()
#         print("图片索引已成功完成！")
#     except Exception as e:
#         print(f"图片索引失败: {e}")
#         raise


# if __name__ == "__main__":
#     main() 