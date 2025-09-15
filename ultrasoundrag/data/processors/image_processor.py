"""
图像处理器

负责图像的预处理、特征提取和质量评估：
- 图像格式转换和标准化
- 图像质量评估
- 特征提取和增强
- 批量图像处理
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torchvision import transforms

@dataclass
class ImageInfo:
    """图像信息数据结构"""
    image_path: str
    width: int
    height: int
    channels: int
    file_size: int
    format: str
    quality_score: float = 0.0
    features: Optional[np.ndarray] = None

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 quality_threshold: float = 0.5):
        """
        初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸
            quality_threshold: 质量阈值
        """
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        
        # 定义图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            PIL图像对象
        """
        try:
            if not os.path.exists(image_path):
                print(f"图像文件不存在: {image_path}")
                return None
            
            image = Image.open(image_path)
            
            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None
    
    def get_image_info(self, image_path: str) -> Optional[ImageInfo]:
        """
        获取图像信息
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像信息对象
        """
        try:
            image = self.load_image(image_path)
            if image is None:
                return None
            
            # 获取文件信息
            file_size = os.path.getsize(image_path)
            
            # 计算质量分数
            quality_score = self.calculate_quality_score(image)
            
            return ImageInfo(
                image_path=image_path,
                width=image.width,
                height=image.height,
                channels=len(image.getbands()),
                file_size=file_size,
                format=image.format or 'Unknown',
                quality_score=quality_score
            )
        except Exception as e:
            print(f"获取图像信息失败 {image_path}: {e}")
            return None
    
    def calculate_quality_score(self, image: Image.Image) -> float:
        """
        计算图像质量分数
        
        Args:
            image: PIL图像对象
            
        Returns:
            质量分数 (0-1)
        """
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 计算清晰度（拉普拉斯方差）
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 计算亮度分布
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # 计算对比度
            contrast = np.std(gray)
            contrast_score = min(contrast / 64, 1.0)
            
            # 计算尺寸分数
            size_score = min(image.width * image.height / (224 * 224), 1.0)
            
            # 综合质量分数
            quality_score = (
                min(laplacian_var / 1000, 1.0) * 0.4 +  # 清晰度
                brightness_score * 0.2 +                 # 亮度
                contrast_score * 0.2 +                   # 对比度
                size_score * 0.2                         # 尺寸
            )
            
            return min(quality_score, 1.0)
        except Exception as e:
            print(f"计算图像质量分数失败: {e}")
            return 0.0
    
    def preprocess_image(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        预处理图像
        
        Args:
            image: PIL图像对象
            
        Returns:
            预处理后的张量
        """
        try:
            # 应用变换
            tensor = self.transform(image)
            return tensor
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None
    
    def enhance_image(self, image: Image.Image, 
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     sharpness: float = 1.0) -> Image.Image:
        """
        图像增强
        
        Args:
            image: 输入图像
            brightness: 亮度调整因子
            contrast: 对比度调整因子
            sharpness: 锐度调整因子
            
        Returns:
            增强后的图像
        """
        try:
            # 亮度调整
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            
            # 对比度调整
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
            
            # 锐度调整
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpness)
            
            return image
        except Exception as e:
            print(f"图像增强失败: {e}")
            return image
    
    def resize_image(self, image: Image.Image, 
                    size: Tuple[int, int],
                    keep_aspect_ratio: bool = True) -> Image.Image:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            size: 目标尺寸
            keep_aspect_ratio: 是否保持宽高比
            
        Returns:
            调整后的图像
        """
        try:
            if keep_aspect_ratio:
                image.thumbnail(size, Image.Resampling.LANCZOS)
            else:
                image = image.resize(size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            print(f"图像尺寸调整失败: {e}")
            return image
    
    def batch_process_images(self, image_paths: List[str]) -> List[ImageInfo]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            图像信息列表
        """
        results = []
        
        for image_path in image_paths:
            image_info = self.get_image_info(image_path)
            if image_info:
                results.append(image_info)
        
        return results
    
    def filter_quality_images(self, image_infos: List[ImageInfo]) -> List[ImageInfo]:
        """
        过滤高质量图像
        
        Args:
            image_infos: 图像信息列表
            
        Returns:
            高质量图像信息列表
        """
        return [
            info for info in image_infos 
            if info.quality_score >= self.quality_threshold
        ]
    
    def process_image_for_model(self, image_path: str) -> Optional[torch.Tensor]:
        """
        为模型处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            模型输入张量
        """
        try:
            # 加载图像
            image = self.load_image(image_path)
            if image is None:
                return None
            
            # 检查质量
            quality_score = self.calculate_quality_score(image)
            if quality_score < self.quality_threshold:
                print(f"图像质量过低，跳过: {image_path}")
                return None
            
            # 预处理
            tensor = self.preprocess_image(image)
            return tensor
        except Exception as e:
            print(f"模型图像处理失败 {image_path}: {e}")
            return None
