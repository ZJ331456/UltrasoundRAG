"""
图像解析器是图像解析器，将原始数据处理成对应的格式并存入原始的，milvus数据库
功能：解析图片，生成图片向量
使用模型：目前是fetalclip，后续会更新这个
数据源：一般是存图文对之前需要处理一个索引的jsonl文件出来没通过读取这个文件，获取相关数据
"""
import json
import os
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from UltrasoundRAG.model.fetal_clip_model import FetalCLIPModel
from tqdm import tqdm
from UltrasoundRAG.config import config

class ImageParser:
    """
    图片解析器：负责读取 image_index.jsonl，处理图片路径、生成向量等。
    """
    
    def __init__(self, dataset_name: str = "ultrasound_book"):
        self.dataset_name = dataset_name
        
        # 从 config 加载变量
        image_cfg = config['indexing']['image']
        image_parse_cfg = config['indexing']['image_parse']
        
        dataset_cfg = image_cfg['datasets'][dataset_name]
        self.base_image_path = dataset_cfg['base_image_path']
        self.image_index_path = os.path.join(self.base_image_path, dataset_cfg['image_index_path'])
        self.model_path = image_parse_cfg['model_path']
        self.model_config_path = image_parse_cfg['model_config_path']
        
        print(f"初始化图片解析器，数据集: {dataset_name}")
        print(f"基础路径: {self.base_image_path}")
        print(f"索引文件: {self.image_index_path}")
        
        try:
            self.fetal_model = FetalCLIPModel(model_path=self.model_path, config_path=self.model_config_path)
            print("FetalCLIP 模型加载成功")
        except Exception as e:
            print(f"FetalCLIP 模型加载失败: {e}")
            raise
    
    def load_image_index(self) -> List[Dict]:
        """
        读取 image_index.jsonl 文件，返回图片数据列表。
        每个条目包含：image_path, caption, source。
        """
        image_data = []
        with open(self.image_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                image_data.append(data)
        return image_data
    
    def get_absolute_path(self, relative_path: str) -> str:
        """
        根据相对路径拼接绝对路径。
        """
        return os.path.join(self.base_image_path, relative_path)
    
    def generate_image_vector(self, image_path: str) -> Optional[List[float]]:
        """
        生成图片向量（使用 FetalCLIP 模型）。
        如果图片不存在，返回 None（跳过）。
        如果处理失败，返回随机向量（768 维，与 FetalCLIP 输出匹配）。
        """
        absolute_path = self.get_absolute_path(image_path)
        if not os.path.exists(absolute_path):
            print(f"警告：图片 {absolute_path} 不存在，跳过")
            return None  # 返回 None 表示跳过
        
        try:
            # 打开并预处理图片
            image = Image.open(absolute_path).convert("RGB")
            
            # 使用 FetalCLIP 的预处理方法
            processed_image = self.fetal_model.preprocess_image(image)  # [C, H, W]
            
            # 添加批次维度
            processed_image = processed_image.unsqueeze(0)  # [1, C, H, W]
            
            # 编码图片
            features = self.fetal_model.encode_image(processed_image)  # [1, D]
            
            # 移除批次维度并转换为列表
            vector = features.squeeze().cpu().tolist()  # [D]
            
            # 确保向量维度为 768（FetalCLIP 的输出维度）
            if len(vector) != 768:
                print(f"警告：向量维度为 {len(vector)}，期望 768，使用零向量")
                return [0.0] * 768  # 使用零向量而非随机向量，避免引入噪声
            
            return vector
            
        except Exception as e:
            print(f"生成向量失败 {absolute_path}: {e}，使用零向量")
            return [0.0] * 768  # 使用零向量而非随机向量，避免引入噪声
    
    def generate_caption_vector(self, caption: str) -> List[float]:
        """
        生成caption文本向量（使用 FetalCLIP 文本编码器）。
        """
        try:
            # 使用 FetalCLIP 的文本编码器
            tokens = self.fetal_model.tokenize_text([caption])
            features = self.fetal_model.encode_text(tokens).cpu().numpy()
            
            # 转换为列表并确保维度为 768
            vector = features[0].tolist()
            if len(vector) != 768:
                print(f"警告：caption向量维度为 {len(vector)}，期望 768，使用零向量")
                return [0.0] * 768
            
            return vector
            
        except Exception as e:
            print(f"生成caption向量失败: {e}，使用零向量")
            return [0.0] * 768

    def parse_images(self) -> List[Dict]:
        """
        解析所有图片数据，返回包含所需字段的列表。
        如果图片不存在，跳过该条目。
        每个条目：{
            'id': int,
            'image_path': str,  # 相对路径（统一命名）
            'caption': str,     # Caption
            'source': str,      # 来源（统一命名，原 folder_name）
            'image_vector': List[float],
            'caption_vector_clip_768': List[float]
        }
        """
        image_data = self.load_image_index()
        parsed_data = []
        image_id = 1
        
        for item in tqdm(image_data, desc="处理图片"):
            image_path = item['image_path']
            source = item['source']
            caption = item['caption']
            image_vector = self.generate_image_vector(image_path)
            
            if image_vector is None:
                continue  # 跳过不存在的图片
            
            # 生成caption向量
            caption_vector = self.generate_caption_vector(caption)
            
            parsed_data.append({
                'id': image_id,
                'image_path': image_path,
                'caption': caption,
                'source': source,
                'image_vector': image_vector,
                'caption_vector_clip_768': caption_vector
            })
            image_id += 1
            
        return parsed_data

# 示例使用
if __name__ == "__main__":
    parser = ImageParser()
    parsed_images = parser.parse_images()
    print(f"解析了 {len(parsed_images)} 张图片")
    for img in parsed_images[:3]:  # 打印前3个
        print(f"ID: {img['id']}, 路径: {img['image_path']}, Caption: {img['caption'][:50]}...")