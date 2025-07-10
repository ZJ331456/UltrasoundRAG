
from llama_index.core.node_parser import MarkdownNodeParser

def get_markdown_parser() -> MarkdownNodeParser:
    """
    获取一个配置好的Markdown节点解析器。
    该解析器能够根据Markdown的标题（如 #, ##, ###）将文档分割成独立的节点，
    确保每个节点都具有逻辑上的独立性和完整的上下文。

    Returns:
        MarkdownNodeParser: LlamaIndex的Markdown节点解析器实例。
    """
    return MarkdownNodeParser(
        include_metadata=True,  # 在节点中包含元数据
        include_prev_next_rel=True  # 包含前后节点的关联信息
    ) 