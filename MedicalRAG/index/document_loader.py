"""
文档加载和索引创建脚本
将文档文件转换为嵌入向量并存储到Chroma向量数据库

重构改进：
1. 移除冗余代码和注释
2. 使用utils模块中的组件
3. 简化错误处理和日志
4. 统一配置管理
"""

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
import chromadb
import torch

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.embedding_utils import embedding_provider, EmbeddingError

def load_and_index_documents():
    """
    加载文档并创建向量索引
    
    Returns:
        VectorStoreIndex: LlamaIndex向量索引对象
    """
    logger = setup_logger(__name__)
    
    # 获取配置参数
    input_dir = config['document']['input_dir']
    collection_name = config['document']['collection_name']
    chroma_db_path = config['document']['vectorstore_path']
    # model_dir = config['embedding']['model_name']
    
    logger.info(f"开始加载文档并创建索引...")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"向量数据库路径: {chroma_db_path}")
    logger.info(f"集合名称: {collection_name}")
    
    # 确保目录存在
    os.makedirs(chroma_db_path, exist_ok=True)
    
    # 初始化Chroma客户端
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    
    # 创建或获取Chroma集合
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 初始化向量存储
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 加载文档
    logger.info(f"从目录加载文档: {os.path.abspath(input_dir)}")
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        required_exts=[".json", ".md"],
        recursive=True
    ).load_data()

    if not documents:
        raise ValueError(f"在目录 {os.path.abspath(input_dir)} 中没有找到 .json 或 .md 文件")

    logger.info(f"成功加载 {len(documents)} 个文档")

    # # 初始化嵌入模型
    # embed_model = CustomEmbedding(
    #     model_name=model_dir,
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    embed_model = embedding_provider['bge_zh_local_embedding']
    
    # 初始化Markdown节点解析器，按标题进行分块
    logger.info("使用 MarkdownNodeParser 按标题进行分块...")
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    logger.info(f"文档被分割成 {len(nodes)} 个节点")

    # 创建向量索引
    logger.info("开始创建向量索引...")
    # 移除旧注释，因为我们现在有了明确的分块策略
    index = VectorStoreIndex(
        nodes, # 使用解析后的节点创建索引
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # 持久化存储
    storage_context.persist(persist_dir=os.path.join(chroma_db_path, "storage"))

    logger.info(f"索引创建完成! 已将 {len(documents)} 个文档存储到集合 '{collection_name}'")
    return index


def main():
    """主函数"""
    try:
        index = load_and_index_documents()
        print("文档索引创建成功!")
        return index
    except Exception as e:
        print(f"文档索引创建失败: {e}")
        raise


if __name__ == "__main__":
    main()
