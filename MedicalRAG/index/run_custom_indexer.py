"""
Custom Indexing Script for Medical RAG

This script uses the CustomMarkdownParser to process Markdown files,
extracting image-text blocks and indexing them into a ChromaDB vector store.
"""

import os
import glob
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from MedicalRAG.config.config import get_image_collection, get_document_collection
import torch
import json
import hashlib
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor
from llama_index.core import SimpleDirectoryReader

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.custom_document_parser import get_markdown_parser
from MedicalRAG.utils.embedding_utils import embedding_provider, EmbeddingError
from MedicalRAG.utils.image_utils import get_image_embedding


def run_image_indexer(logger):
    """
    Indexes images specified in the nodes file into a ChromaDB vector store.
    """
    logger.info("Starting image indexing...")
    
    # --- 从配置加载参数 ---
    image_config = config['indexing']['image']
    
    IMAGE_MODEL_PATH = image_config['model_path']
    IMAGE_NODES_FILE = image_config['nodes_file']
    IMAGE_BASE_PATH = image_config['base_path']
    CHROMA_PERSIST_DIR = image_config['vectorstore_path']
    COLLECTION_NAME = image_config['collection_name']
    BATCH_SIZE = image_config['batch_size']

    logger.info(f"Loading image model from {IMAGE_MODEL_PATH}...")
    try:
        # 加载图像处理模型和处理器
        image_model = CLIPModel.from_pretrained(IMAGE_MODEL_PATH)
        image_processor = CLIPImageProcessor.from_pretrained(IMAGE_MODEL_PATH)
    except Exception as e:
        logger.error(f"Error loading image model: {e}")
        return

    logger.info("Initializing ChromaDB for images...")
    
    # 使用统一的ChromaDB管理器
    collection = get_image_collection()
    logger.info(f"ChromaDB image collection loaded/created successfully.")
    
    # 在重建索引时，清空旧的图片集合
    if collection.count() > 0:
        logger.info(f"Clearing existing data from image collection '{COLLECTION_NAME}'...")
        collection.delete(where={}) # 删除所有文档
        logger.info(f"Image collection cleared. Current count: {collection.count()}")

    logger.info(f"Reading image nodes from {IMAGE_NODES_FILE}...")
    if not os.path.exists(IMAGE_NODES_FILE):
        logger.error(f"Image nodes file not found: {IMAGE_NODES_FILE}. Aborting image indexing.")
        return
        
    with open(IMAGE_NODES_FILE, 'r', encoding='utf-8') as f:
        image_nodes = [json.loads(line) for line in f]

    logger.info(f"Found {len(image_nodes)} images to process.")

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
                logger.warning(f"Image not found, skipping: {image_path}")
                continue

            caption_text = node.get("caption", "")
            if not caption_text:
                logger.warning(f"Caption is empty for image {image_path}, skipping.")
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
                logger.error(f"Error adding batch to ChromaDB: {e}")

    logger.info("Image indexing complete.")
    logger.info(f"Total documents in image collection: {collection.count()}")
    logger.info(f"ChromaDB image data is persisted in: {CHROMA_PERSIST_DIR}")


def run_custom_indexer():
    """
    Loads documents using the custom parser and creates a vector index.
    """
    logger = setup_logger(__name__)
    
    # Get configuration parameters from the new structure
    doc_config = config['indexing']['document']
    input_dir = doc_config['input_dir']
    collection_name = doc_config['collection_name']
    chroma_db_path = doc_config['vectorstore_path']
    
    logger.info("Starting custom document indexing...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Vector store path: {chroma_db_path}")
    logger.info(f"Collection name: {collection_name}")
    
    # Ensure directories exist
    os.makedirs(chroma_db_path, exist_ok=True)
    
    # 使用统一的ChromaDB管理器
    chroma_collection = get_document_collection()
    
    # Initialize vector store and storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # --- 全新的文档加载与解析流程 ---
    logger.info("使用 SimpleDirectoryReader 加载所有 Markdown 文档...")
    
    # 递归加载指定目录下所有 .md 文件
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        recursive=True,
        required_exts=[".md"]
    )
    documents = reader.load_data(show_progress=True)
    
    if not documents:
        raise ValueError(f"在目录 {os.path.abspath(input_dir)} 中未找到任何 .md 文件")
        
    logger.info(f"成功加载了 {len(documents)} 个 Markdown 文件。")

    # 获取基于标题的 Markdown 解析器
    logger.info("使用 MarkdownNodeParser 按标题进行分块...")
    parser = get_markdown_parser()
    
    # 从文档中获取节点
    nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    
    if not nodes:
        raise ValueError("未能从文档中解析出任何节点。")

    logger.info(f"成功将文档分割成 {len(nodes)} 个节点。")

    #  使用embedding_provider获取嵌入模型
    try:
        embed_model = embedding_provider[config['embedding']['provider']]
    except EmbeddingError as e:
        logger.error(f"初始化嵌入模型失败: {e}")
        raise

    # 索引创建
    logger.info("根据解析出的节点创建向量索引...")
    index = VectorStoreIndex(
        nodes,  # 注意：这里传递的是 nodes 而不是 documents
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info(f"文本索引创建完成! 在集合 '{collection_name}' 中存储了 {len(nodes)} 个节点。")
    
    # --- 2. 索引图片 ---
    run_image_indexer(logger)
    
    return index


def main():
    """Main function"""
    try:
        index = run_custom_indexer()
        print("Custom document indexing completed successfully!")
    except Exception as e:
        print(f"Custom document indexing failed: {e}")
        raise


if __name__ == "__main__":
    main()