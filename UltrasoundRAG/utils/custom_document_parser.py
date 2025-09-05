
import re
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode

def extract_image_info(text: str) -> dict:
    """
    从Markdown文本中提取图片信息
    支持 ![alt](path) 和 ![alt](path "caption") 格式
    """
    image_info = []
    
    # 匹配 ![alt](path) 和 ![alt](path "caption") 格式
    pattern = r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)'
    matches = re.findall(pattern, text)
    
    for alt, path, caption in matches:
        image_info.append({
            'alt': alt,
            'path': path,
            'caption': caption or alt,  # 如果没有caption，使用alt作为caption
            'type': 'markdown_image'
        })
    
    return image_info

def get_markdown_parser() -> MarkdownNodeParser:
    """
    获取一个配置好的Markdown节点解析器。
    该解析器能够根据Markdown的标题（如 #, ##, ###）将文档分割成独立的节点，
    确保每个节点都具有逻辑上的独立性和完整的上下文。
    """
    return MarkdownNodeParser(
        include_metadata=True,  # 在节点中包含元数据
        include_prev_next_rel=False  # 禁用前后节点的关联信息以提升性能
    )

def create_enhanced_nodes(nodes, image_mapping: dict = None) -> list:
    """
    增强节点信息，添加图片关联
    """
    enhanced_nodes = []
    
    for node in nodes:
        # 提取文本中的图片信息
        image_info = extract_image_info(node.text)
        
        # 创建增强的元数据
        enhanced_metadata = node.metadata.copy() if hasattr(node, 'metadata') else {}
        
        if image_info:
            enhanced_metadata['images'] = image_info
            # 如果有图片映射，添加对应的图片路径
            if image_mapping:
                for img in image_info:
                    if img['path'] in image_mapping:
                        img['full_path'] = image_mapping[img['path']]
                        img['image_id'] = image_mapping[img['path']].get('id')
        
        # 创建增强的节点
        enhanced_node = TextNode(
            text=node.text,
            metadata=enhanced_metadata
        )
        
        # 复制其他属性
        if hasattr(node, 'id_'):
            enhanced_node.id_ = node.id_
        if hasattr(node, 'node_id'):
            enhanced_node.node_id = node.node_id
            
        enhanced_nodes.append(enhanced_node)
    
    return enhanced_nodes 