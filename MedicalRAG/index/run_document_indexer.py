"""
文档索引脚本（Medical RAG）

本脚本使用 CustomMarkdownParser 处理 Markdown 文件，
将文档分块并索引到 ChromaDB 向量库。
"""

import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from MedicalRAG.config.config import get_document_collection
from llama_index.core import SimpleDirectoryReader

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.custom_document_parser import get_markdown_parser
from MedicalRAG.utils.embedding_utils import embedding_provider, EmbeddingError
from MedicalRAG.utils.file_util import force_remove_directory, ensure_directory_exists


def run_document_indexer():
    """
    使用自定义解析器加载文档并创建向量索引。
    """
    logger = setup_logger(__name__)
    
    # 从新结构获取配置参数
    # doc_config = config['indexing']['document']
    # input_dir = doc_config['input_dir']
    # collection_name = doc_config['collection_name']
    # chroma_db_path = doc_config['vectorstore_path']
        
    # --- 从配置加载参数 ---
    doc_config = config['indexing']['document']
    INPUT_DIR = doc_config['input_dir']
    COLLECTION_NAME = doc_config['collection_name']
    CHROMA_PERSIST_DIR = doc_config['vectorstore_path']
    
    logger.info("开始文档索引...")
    logger.info(f"输入目录: {INPUT_DIR}")
    logger.info(f"向量库路径: {CHROMA_PERSIST_DIR}")
    logger.info(f"集合名: {COLLECTION_NAME}")
    
    # 完全删除旧的向量数据库文件夹以避免重复内容
    logger.info("开始清理旧的文档向量数据库文件夹...")
    
    # 使用强制删除函数处理文件被占用的情况
    doc_delete_success = force_remove_directory(CHROMA_PERSIST_DIR, logger)
    if not doc_delete_success:
        logger.error(f"无法删除文档向量数据库文件夹: {CHROMA_PERSIST_DIR}")
        raise RuntimeError(f"无法删除文档向量数据库文件夹，请手动关闭占用该文件的程序后重试")
    
    logger.info("文档向量数据库文件夹清理完成")
    
    # 确保目录存在
    ensure_directory_exists(CHROMA_PERSIST_DIR)
    
    # 使用统一的ChromaDB管理器
    chroma_collection = get_document_collection()
    
    # 初始化向量库和存储上下文
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # --- 全新的文档加载与解析流程 ---
    logger.info("使用 SimpleDirectoryReader 加载所有 Markdown 文档...")
    
    # 递归加载指定目录下所有 .md 文件
    reader = SimpleDirectoryReader(
        input_dir=INPUT_DIR,
        recursive=True,
        required_exts=[".md"]
    )
    documents = reader.load_data(show_progress=True)
    
    if not documents:
        raise ValueError(f"在目录 {os.path.abspath(INPUT_DIR)} 中未找到任何 .md 文件")
        
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

    logger.info(f"文本索引创建完成! 在集合 '{COLLECTION_NAME}' 中存储了 {len(nodes)} 个节点。")
    
    return index


# def main():
#     """主函数"""
#     try:
#         index = run_document_indexer()
#         print("文档索引已成功完成！")
#     except Exception as e:
#         print(f"文档索引失败: {e}")
#         raise


# if __name__ == "__main__":
#     main() 