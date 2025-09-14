from PIL import Image
import torch
from transformers import CLIPProcessor
import logging
import os
from ultrasoundrag.models.model.fetal_clip_model import FetalCLIPModel

logger = logging.getLogger(__name__)

def load_clip_model(checkpoint_path: str, config_path: str):
    """
    加载FetalCLIP模型。

    Args:
        checkpoint_path (str): 模型权重文件的路径。
        config_path (str): 模型配置文件路径。

    Returns:
        tuple: (FetalCLIPModel, image_processor) 或 (None, None) 如果加载失败。
    """
    logger.info(f"开始加载CLIP模型: {checkpoint_path}")
    try:
        # 使用新的FetalCLIPModel加载模型
        fetal_clip_model = FetalCLIPModel(
            model_path=checkpoint_path,
            config_path=config_path
        )
        
        # 返回模型和图像处理器
        logger.info("CLIP模型加载成功。")
        return fetal_clip_model, fetal_clip_model.image_processor
    except Exception as e:
        logger.error(f"加载模型失败: {checkpoint_path}. 错误: {e}", exc_info=True)
        return None, None


def get_image_embedding(image_path, image_model, image_processor):
    """
    获取单张图片的向量
    """
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
            
        # 使用图像处理器预处理图像
        processed_image = image_processor(image).unsqueeze(0)  # 添加batch维度
        
        with torch.no_grad():
            # 使用FetalCLIPModel的encode_image方法
            image_features = image_model.encode_image(processed_image)
            
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_image_embeddings_batch(image_paths, image_model, image_processor):
    """
    批量获取图片向量（显著加速）
    """
    images = []
    valid_indices = []
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            valid_indices.append(idx)
        except Exception:
            continue
    if not images:
        return []
    try:
        processed_list = [image_processor(img) for img in images]
        batch = torch.stack(processed_list, dim=0)
        with torch.no_grad():
            feats = image_model.encode_image(batch)
        return feats.cpu().numpy(), valid_indices
    except Exception as e:
        print(f"Error processing image batch: {e}")
        return [], []

def get_text_embedding(text, text_model, tokenizer=None):
    """
    获取文本的向量, 使用与CLIP模型匹配的方法
    """
    try:
        # 如果没有提供tokenizer，使用模型自带的tokenizer
        if tokenizer is None:
            tokenizer = text_model.tokenizer
        
        # 使用tokenizer对文本进行分词
        if isinstance(text, str):
            text = [text]  # 转换为列表格式
        
        text_tokens = tokenizer(text)
        
        with torch.no_grad():
            # 使用FetalCLIPModel的encode_text方法
            text_features = text_model.encode_text(text_tokens)
            
        return text_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing text: {e}")
        return None