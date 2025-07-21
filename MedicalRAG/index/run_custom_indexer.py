"""
Custom Indexing Script for Medical RAG

This script uses the CustomMarkdownParser to process Markdown files,
extracting image-text blocks and indexing them into a ChromaDB vector store.
"""

import os
import glob
import time
import subprocess
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from MedicalRAG.config.config import get_image_collection, get_document_collection
import torch
import json
import hashlib
from tqdm import tqdm
from MedicalRAG.utils.image_utils import load_clip_model
from llama_index.core import SimpleDirectoryReader

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.custom_document_parser import get_markdown_parser
from MedicalRAG.utils.embedding_utils import embedding_provider, EmbeddingError
from MedicalRAG.utils.image_utils import get_image_embedding


def force_remove_directory(path, logger, max_retries=3):
    """
    强制删除目录，处理文件被占用的情况
    """
    import shutil
    
    if not os.path.exists(path):
        return True
    
    for attempt in range(max_retries):
        try:
            # 尝试正常删除
            shutil.rmtree(path)
            logger.info(f"成功删除目录: {path}")
            return True
        except PermissionError as e:
            if "另一个程序正在使用此文件" in str(e) or "being used by another process" in str(e):
                logger.warning(f"目录被占用，尝试强制删除 (尝试 {attempt + 1}/{max_retries}): {path}")
                
                # 尝试终止可能占用文件的进程
                try:
                    # 查找占用ChromaDB文件的进程
                    result = subprocess.run(
                        ['powershell', '-Command', f'Get-Process | Where-Object {{$_.Path -like "*streamlit*" -or $_.ProcessName -eq "python"}} | Select-Object Id, ProcessName, Path'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.stdout:
                        logger.info(f"发现可能占用文件的进程:\n{result.stdout}")
                except Exception as proc_e:
                    logger.warning(f"无法查询进程信息: {proc_e}")
                
                # 等待一段时间后重试
                time.sleep(2)
                
                # 最后一次尝试：使用Windows的rmdir命令强制删除
                if attempt == max_retries - 1:
                    try:
                        subprocess.run(['rmdir', '/s', '/q', path], shell=True, check=True)
                        logger.info(f"使用系统命令强制删除成功: {path}")
                        return True
                    except subprocess.CalledProcessError as cmd_e:
                        logger.error(f"系统命令删除失败: {cmd_e}")
            else:
                logger.error(f"删除目录失败: {e}")
                return False
        except Exception as e:
            logger.error(f"删除目录时发生未知错误: {e}")
            if attempt == max_retries - 1:
                return False
            time.sleep(1)
    
    return False


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
        # 使用新的FetalCLIP模型加载器
        # 需要从配置中获取config_path
        config_path = image_config.get('config_path', 'E:\\Dolphin\\ht-rag\\models\\fetal-clip\\config.json')
        image_model, image_processor = load_clip_model(IMAGE_MODEL_PATH, config_path)
        if image_model is None:
            logger.error("Failed to load FetalCLIP model")
            return
    except Exception as e:
        logger.error(f"Error loading image model: {e}")
        return

    logger.info("Initializing ChromaDB for images...")
    
    # 使用统一的ChromaDB管理器
    collection = get_image_collection()
    logger.info(f"ChromaDB image collection loaded/created successfully.")

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
    
    # Get image configuration for cleaning image vectorstore
    image_config = config['indexing']['image']
    image_chroma_path = image_config['vectorstore_path']
    
    logger.info("Starting custom document indexing...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Vector store path: {chroma_db_path}")
    logger.info(f"Collection name: {collection_name}")
    
    # 完全删除旧的向量数据库文件夹以避免重复内容
    logger.info("开始清理旧的向量数据库文件夹...")
    
    # 使用强制删除函数处理文件被占用的情况
    doc_delete_success = force_remove_directory(chroma_db_path, logger)
    if not doc_delete_success:
        logger.error(f"无法删除文档向量数据库文件夹: {chroma_db_path}")
        raise RuntimeError(f"无法删除文档向量数据库文件夹，请手动关闭占用该文件的程序后重试")
    
    image_delete_success = force_remove_directory(image_chroma_path, logger)
    if not image_delete_success:
        logger.error(f"无法删除图像向量数据库文件夹: {image_chroma_path}")
        raise RuntimeError(f"无法删除图像向量数据库文件夹，请手动关闭占用该文件的程序后重试")
    
    logger.info("向量数据库文件夹清理完成")
    
    # Ensure directories exist
    os.makedirs(chroma_db_path, exist_ok=True)
    os.makedirs(image_chroma_path, exist_ok=True)
    
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