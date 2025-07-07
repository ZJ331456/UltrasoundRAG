"""
答案生成器工具模块
基于检索结果生成答案的核心组件

主要功能：
1. 基于检索上下文生成答案
2. 智能上下文长度控制
3. 支持API和本地模型两种生成方式
"""

from typing import List, Dict, Any
from PIL import Image
import os
import re

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.llm_utils import llm_provider, LLMError
from MedicalRAG.utils.prompt import get_optimized_prompt, generate_contextual_prompt, guide_generate_contextual_prompt



class AnswerGenerator:
    """答案生成器 - 基于检索结果生成高质量答案"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化答案生成器
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict
        self.logger = setup_logger(__name__)
        
        # 通过LLM Provider获取生成器实例
        try:
            self.llm = llm_provider['medical_vllm_api']
        except ValueError as e:
            self.logger.error(f"初始化LLM失败: {e}")
            raise e
    
    def generate_answer(self, query: str, retrieval_results: List[Any]) -> str:
        """
        基于检索结果生成答案，支持多模态输入
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果列表，可能包含文本和图像
            
        Returns:
            生成的答案字符串
        """
        if not retrieval_results:
            return "抱歉，我没有找到相关信息来回答您的问题。"

        # 1. 分离文本和图像结果
        text_results = [r for r in retrieval_results if r.retrieval_type != 'image_from_text']
        image_results = [r for r in retrieval_results if r.retrieval_type == 'image_from_text']
        
        # 2. 加载图像
        loaded_images = self._load_images(image_results)

        # 3. 构建文本上下文 (现在知道有多少图片，可以动态调整上下文长度)
        context = self._build_context(text_results, num_images=len(loaded_images))
        
        # 4. 生成提示词
        prompt = guide_generate_contextual_prompt(query, context)
        
        try:
            # 5. 调用LLM生成答案（传入文本和图像）
            # 检查self.llm是否是一个视觉模型
            if getattr(self.llm, 'is_vision_model', False) and loaded_images:
                 answer = self.llm.generate(prompt, images=loaded_images)
            else:
                # 如果LLM不支持图像或没有图像，则回退到纯文本模式
                if loaded_images:
                    self.logger.warning("LLM不是视觉模型，但检测到了图像结果。图像将被忽略。")
                answer = self.llm.generate(prompt)

            self.logger.info(f"成功生成答案，长度: {len(answer)}")
            return answer
            
        except Exception as e:
            self.logger.error(f"答案生成失败: {e}", exc_info=True)
            return f"抱歉，生成答案时出现错误: {str(e)}"

    def _load_images(self, image_results: List[Any]) -> List[Image.Image]:
        """从图像检索结果中加载图像文件"""
        loaded_images = []
        path_pattern = re.compile(r"图片路径: (.+)")

        for result in image_results:
            content = getattr(result, 'content', '')
            match = path_pattern.search(content)
            if not match:
                self.logger.warning(f"无法从内容中解析图片路径: {content}")
                continue

            # 从匹配中提取并清理路径
            image_path = match.group(1).strip()

            # 确保路径在当前操作系统下是有效的
            image_path = os.path.normpath(image_path)
            
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    loaded_images.append(image)
                    self.logger.info(f"成功加载图片: {image_path}")
                except Exception as e:
                    self.logger.error(f"加载图片失败 '{image_path}': {e}")
            else:
                self.logger.warning(f"图片路径不存在: {image_path}")
                
        return loaded_images

    def _build_context(self, retrieval_results: List[Any], num_images: int = 0) -> str:
        """
        仅从文本检索结果构建上下文，并根据图片数量动态调整长度。
        
        Args:
            retrieval_results: 文本检索结果列表
            num_images: 本次请求包含的图片数量
            
        Returns:
            构建的上下文字符串
        """
        base_max_length = self.config['generator'].get('max_context_length', 2000)
        
        # 为图片预留空间。这是一个启发式估计，因为不同图片token数不同。
        # 假设每张图片大约消耗800个字符的上下文预算。
        image_context_cost = num_images * 800
        max_text_length = base_max_length - image_context_cost
        
        # 确保我们至少为文本保留一些最小空间
        max_text_length = max(max_text_length, 500)
        
        self.logger.info(f"构建上下文: {num_images}张图片，文本上下文最大长度调整为: {max_text_length} (基础值: {base_max_length})")

        context_parts = []
        current_length = 0
        
        for i, result in enumerate(retrieval_results, 1):
            # 获取内容，兼容不同的结果格式
            content = getattr(result, 'content', '') if hasattr(result, 'content') else str(result.get('content', ''))
            
            if not content.strip():
                continue
            
            # 计算添加这个片段后的长度
            part = f"参考资料{i}:\n{content.strip()}"
            part_length = len(part)
            
            # 检查是否会超出调整后的长度限制
            if current_length + part_length > max_text_length:
                # 如果超出，尝试截断当前片段
                available_length = max_text_length - current_length - 20  # 预留一些空间
                if available_length > 100:  # 至少保留100字符才有意义
                    truncated_content = content[:available_length] + "..."
                    part = f"参考资料{i}:\n{truncated_content}"
                    context_parts.append(part)
                break
            
            context_parts.append(part)
            current_length += part_length + 2  # +2 for newlines
        
        context = "\n\n".join(context_parts)
        self.logger.debug(f"构建上下文完成，总长度: {len(context)}")
        return context
