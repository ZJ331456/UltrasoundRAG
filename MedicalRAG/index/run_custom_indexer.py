"""
Custom Indexing Script for Medical RAG

This script uses the CustomMarkdownParser to process Markdown files,
extracting image-text blocks and indexing them into a ChromaDB vector store.
"""

import os
import glob
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import torch

from MedicalRAG.config.config import config
from MedicalRAG.utils.embedding_utils import CustomEmbedding
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.index.custom_document_parser import CustomMarkdownParser
from MedicalRAG.utils.embedding_utils import embedding_provider, EmbeddingError

def run_custom_indexer():
    """
    Loads documents using the custom parser and creates a vector index.
    """
    logger = setup_logger(__name__)
    
    # Get configuration parameters
    input_dir = config['document']['input_dir']
    # Use a new collection for the custom index
    collection_name = "medical_rag_custom" 
    chroma_db_path = config['document']['vectorstore_path']
    # model_dir = config['embedding']['model_name']
    
    logger.info("Starting custom document indexing...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Vector store path: {chroma_db_path}")
    logger.info(f"Collection name: {collection_name}")
    
    # Ensure directories exist
    os.makedirs(chroma_db_path, exist_ok=True)
    
    # Initialize Chroma client and collection
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize vector store and storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # --- Custom Document Loading ---
    logger.info("Loading documents with CustomMarkdownParser...")
    parser = CustomMarkdownParser()
    all_documents = []
    
    # Find all markdown files in the input directory recursively
    md_files = glob.glob(os.path.join(input_dir, '**', '*.md'), recursive=True)
    if not md_files:
        raise ValueError(f"No .md files found in {os.path.abspath(input_dir)}")
        
    logger.info(f"Found {len(md_files)} Markdown files to process.")

    for file_path in md_files:
        logger.info(f"Parsing file: {file_path}")
        try:
            file_documents = parser.parse_file(file_path)
            all_documents.extend(file_documents)
            logger.info(f"  -> Extracted {len(file_documents)} documents (blocks).")
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")

    if not all_documents:
        raise ValueError("No documents were extracted from the source files.")

    logger.info(f"Successfully loaded a total of {len(all_documents)} document blocks.")

    # Initialize embedding model
    # embed_model = CustomEmbedding(
    #     model_name=model_dir,
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )
    embed_model = embedding_provider['bge_zh_local_embedding']
    # --- Index Creation ---
    # We feed the already-parsed documents directly to the index.
    # We do NOT use a node parser here, as our custom parser has already
    # created the semantically meaningful chunks (Documents).
    logger.info("Creating vector index from custom-parsed documents...")
    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    logger.info(f"Index creation complete! Stored {len(all_documents)} blocks in collection '{collection_name}'.")
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