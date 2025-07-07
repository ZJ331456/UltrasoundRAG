from PIL import Image
import torch
from transformers import BertForSequenceClassification, BertTokenizer, CLIPModel, CLIPProcessor, CLIPImageProcessor

def get_image_embedding(image_path, image_model, image_processor):
    """
    获取单张图片的向量
    """
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
            
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = image_model.get_image_features(**inputs)
            # 归一化
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_text_embedding(text, text_model, processor):
    """
    获取文本的向量, 使用与CLIP模型匹配的方法
    """
    # 使用 processor 对文本进行分词和预处理
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # 调用 get_text_features 来提取与图像向量空间对齐的文本特征
        text_features = text_model.get_text_features(**inputs)
        # 归一化
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    return text_features.cpu().numpy() 