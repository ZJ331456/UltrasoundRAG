#!/usr/bin/env python3
"""
FetalCLIP模型使用示例
展示如何加载和使用FetalCLIP模型进行图像-文本匹配
"""

import torch
from PIL import Image
from MedicalRAG.model import load_fetal_clip_model

def main():
    """
    FetalCLIP模型使用示例
    """
    print("=== FetalCLIP模型使用示例 ===")
    
    try:
        # 1. 加载模型
        print("\n1. 加载FetalCLIP模型...")
        model = load_fetal_clip_model(
            model_path="E:\\Dolphin\\ht-rag\\models\\fetal-clip\\dataset1-pretrain\\final_model.pt",
            config_path="E:\\Dolphin\\ht-rag\\models\\fetal-clip\\config.json"
        )
        
        # 2. 准备测试文本
        print("\n2. 准备测试文本...")
        texts = [
            "胎儿超声图像显示正常发育",
            "超声检查显示胎儿头部结构",
            "胎儿心脏超声图像",
            "正常的胎儿脊柱超声"
        ]
        
        # 3. 文本编码
        print("\n3. 对文本进行编码...")
        text_tokens = model.tokenize_text(texts)
        print(f"文本token形状: {text_tokens.shape}")
        
        # 编码文本特征
        text_features = model.encode_text(text_tokens)
        print(f"文本特征形状: {text_features.shape}")
        
        # 4. 模拟图像编码（如果有图像的话）
        print("\n4. 模拟图像编码...")
        # 创建一个模拟的图像张量 (batch_size=2, channels=3, height=224, width=224)
        dummy_images = torch.randn(2, 3, 224, 224)
        
        # 编码图像特征
        image_features = model.encode_image(dummy_images)
        print(f"图像特征形状: {image_features.shape}")
        
        # 5. 计算相似度
        print("\n5. 计算图像-文本相似度...")
        # 只取前2个文本与2个图像计算相似度
        similarity = model.compute_similarity(dummy_images, text_tokens[:2])
        print(f"相似度矩阵形状: {similarity.shape}")
        print(f"相似度矩阵:\n{similarity}")
        
        # 6. 找到最匹配的文本
        print("\n6. 找到最匹配的图像-文本对...")
        for i in range(similarity.shape[0]):
            best_text_idx = similarity[i].argmax().item()
            best_score = similarity[i, best_text_idx].item()
            print(f"图像 {i} 最匹配的文本: '{texts[best_text_idx]}' (相似度: {best_score:.4f})")
        
        print("\n✓ FetalCLIP模型测试完成！")
        
    except FileNotFoundError as e:
        print(f"\n 文件未找到错误: {e}")
        print("请确保模型文件和配置文件路径正确")
    except Exception as e:
        print(f"\n 运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()