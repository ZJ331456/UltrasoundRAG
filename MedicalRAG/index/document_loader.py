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
from MedicalRAG.config.config import get_document_collection
import torch

from MedicalRAG.config.config import config
from MedicalRAG.utils.embedding_utils import LocalEmbedding
from MedicalRAG.utils.logger import setup_logger
def load_and_index_documents():
    """
    加载文档并创建向量索引
    
    Returns:
        VectorStoreIndex: LlamaIndex向量索引对象
    """
    logger = setup_logger(__name__)
    
    # --- 从配置结构获取参数 ---
    doc_config = config['indexing']['document']
    embedding_provider_name = config['embedding']['provider']
    embedding_config = config['embedding_providers'][embedding_provider_name]

    input_dir = doc_config['input_dir']
    collection_name = doc_config['collection_name']
    chroma_db_path = doc_config['vectorstore_path']
    
    # 确保嵌入模型提供者是 'local' 类型
    if embedding_config.get('type') != 'local':
        raise ValueError(f"文档索引仅支持 'local' 类型的嵌入模型, 但配置的是 '{embedding_config.get('type')}'")
    model_dir = embedding_config.get('params', {}).get('model_name')
    if not model_dir:
         raise ValueError(f"在配置中找不到 '{embedding_provider_name}' 的 model_name")

    logger.info(f"开始加载文档并创建索引...")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"向量数据库路径: {chroma_db_path}")
    logger.info(f"集合名称: {collection_name}")
    
    # 确保目录存在
    os.makedirs(chroma_db_path, exist_ok=True)
    
    # 初始化Chroma客户端
    # 使用统一的ChromaDB管理器
    collection = get_document_collection()
    
    # 创建或获取Chroma集合
    chroma_collection = collection
    
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

    # 初始化嵌入模型
    embed_model = LocalEmbedding(
        model_name=model_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 创建向量索引
    logger.info("开始创建向量索引...")
    # 调用VectorStoreIndex.from_documents建立索引时，不指定具体的分块策略会默认使用SentenceSplitter这个按句子拆分
    # 目前策略：按句子分割，组合成1024个token,重叠为20个token文本
    # llama_index里面的文本分割有MarkdownHeaderTextSplitter这个！后续可以尝试一下！
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
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
