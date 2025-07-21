"""
索引控制器脚本（Medical RAG）

本脚本是索引功能的主控制器，可以分别调用文档索引和图片索引，
或者同时运行两者。
"""

import argparse
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.index.run_document_indexer import run_document_indexer
from MedicalRAG.index.run_image_indexer import run_image_indexer


def run_all_indexers():
    """
    运行文档和图片索引器
    """
    logger = setup_logger(__name__)
    
    logger.info("开始运行完整索引流程（文档 + 图片）...")
    
    try:
        # 1. 运行文档索引
        logger.info("=" * 50)
        logger.info("步骤 1: 开始文档索引")
        logger.info("=" * 50)
        index = run_document_indexer()
        logger.info("文档索引完成！")
        
        # 2. 运行图片索引
        logger.info("=" * 50)
        logger.info("步骤 2: 开始图片索引")
        logger.info("=" * 50)
        run_image_indexer()
        logger.info("图片索引完成！")
        
        logger.info("=" * 50)
        logger.info("所有索引任务完成！")
        logger.info("=" * 50)
        
        return index
        
    except Exception as e:
        logger.error(f"索引流程失败: {e}")
        raise


# def main():
#     """主函数"""
#     parser = argparse.ArgumentParser(description="Medical RAG 索引控制器")
#     parser.add_argument(
#         "--type", 
#         choices=["all", "document", "image"],
#         default="all",
#         help="选择要运行的索引类型: all(全部), document(仅文档), image(仅图片)"
#     )
    
#     args = parser.parse_args()
    
#     try:
#         if args.type == "all":
#             run_all_indexers()
#             print("所有索引已成功完成！")
#         elif args.type == "document":
#             run_document_indexer()
#             print("文档索引已成功完成！")
#         elif args.type == "image":
#             run_image_indexer()
#             print("图片索引已成功完成！")
#     except Exception as e:
#         print(f"索引失败: {e}")
#         raise


# if __name__ == "__main__":
#     main()