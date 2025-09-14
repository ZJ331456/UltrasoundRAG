"""
内容处理器模块
处理检索结果中的图片链接转换和内容格式化
集成caption_to_image_retriever来匹配真实的图片路径
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from .caption_to_image_retriever import create_caption_retriever


def process_content_with_images(content: str, image_paths: List[str], image_captions: List[str], 
                               db_name: str = "default") -> Tuple[str, str]:
    """
    处理包含图片的内容，将图片链接转换为HTML格式
    使用caption_to_image_retriever匹配真实的图片路径
    
    Args:
        content: 原始内容
        image_paths: 图片路径列表（可能包含URL）
        image_captions: 图片标题列表
        db_name: 数据库名称，用于caption匹配
        
    Returns:
        (processed_content, original_content): 处理后的内容和原始内容
    """
    original_content = content
    
    if not image_captions:
        return content, original_content
    
    # 确保image_captions是列表
    if isinstance(image_captions, str):
        try:
            import json
            image_captions = json.loads(image_captions)
        except:
            image_captions = [image_captions]
    
    # 确保image_paths是列表
    if isinstance(image_paths, str):
        try:
            import json
            image_paths = json.loads(image_paths)
        except:
            image_paths = [image_paths]
    
    processed_content = content
    
    # 使用caption_to_image_retriever匹配真实的图片路径
    try:
        caption_retriever = create_caption_retriever(db_name)
        search_result = caption_retriever.search_from_md_image_captions(
            image_captions, 
            top_k_per_caption=1,  # 每个caption只取第一个匹配
            use_like=True
        )
        
        # 构建caption到图片路径的映射
        matched_images = {}
        for result in search_result.get('results', []):
            caption = result.content
            image_path = result.metadata.get('image_path', '')
            if caption and image_path:
                matched_images[caption] = image_path
    except Exception as e:
        print(f"Caption匹配失败: {e}")
        matched_images = {}
    
    # 先找到所有markdown图片链接
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    img_matches = list(re.finditer(img_pattern, processed_content))
    
    # 按顺序处理每个图片链接
    for i, match in enumerate(img_matches):
        alt_text = match.group(1)  # 图片的alt text
        img_url = match.group(2)   # 图片的URL
        
        # 确定对应的caption
        if i < len(image_captions):
            img_caption = image_captions[i]
        else:
            # 如果没有对应的caption，使用alt text或默认
            img_caption = alt_text if alt_text else f"图片{i+1}"
        
        # 查找匹配的图片路径
        matched_path = matched_images.get(img_caption)
        
        if matched_path:
            # 使用匹配到的真实图片路径
            filename = matched_path.split('/')[-1] if '/' in matched_path else matched_path
            new_img_tag = f'<image src="/api/images/{filename}" caption="{img_caption}"/>'
        else:
            # 如果没有匹配到，使用原始路径或默认处理
            img_path = image_paths[i] if i < len(image_paths) else img_url
            if img_path:
                filename = img_path.split('/')[-1] if '/' in img_path else img_path
                new_img_tag = f'<image src="/api/images/{filename}" caption="{img_caption}"/>'
            else:
                # 如果没有路径信息，只保留caption
                new_img_tag = f'<image src="" caption="{img_caption}"/>'
        
        # 替换这个图片链接
        processed_content = processed_content.replace(match.group(0), new_img_tag, 1)
    
    return processed_content, original_content




def extract_image_info_from_content(content: str) -> Tuple[List[str], List[str]]:
    """
    从内容中提取图片信息
    
    Args:
        content: 内容文本
        
    Returns:
        (image_paths, image_captions): 图片路径和标题列表
    """
    image_paths = []
    image_captions = []
    
    # 匹配 ![](url) 格式
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = re.findall(img_pattern, content)
    
    for caption, path in matches:
        image_captions.append(caption.strip())
        image_paths.append(path.strip())
    
    return image_paths, image_captions


def process_retrieval_result(result: Dict[str, Any], retrieval_type: str) -> Dict[str, Any]:
    """
    处理单个检索结果，添加原始内容和处理图片链接
    
    Args:
        result: 原始检索结果
        retrieval_type: 检索类型
        
    Returns:
        处理后的结果
    """
    content = result.get('content', '')
    image_paths = result.get('image_paths', [])
    image_captions = result.get('image_captions', [])
    
    # 处理图片链接
    processed_content, original_content = process_content_with_images(
        content, image_paths, image_captions
    )
    
    # 更新结果
    result['content'] = processed_content
    result['original_content'] = original_content
    
    return result
