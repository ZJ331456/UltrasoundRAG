"""
答案生成器工具模块
基于检索结果生成答案的核心组件

主要功能：
1. 基于检索上下文生成答案
2. 智能上下文长度控制
3. 支持API和本地模型两种生成方式
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import os
import re

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger
from MedicalRAG.utils.llm_utils import llm_provider, LLMError
from MedicalRAG.utils.prompt import generate_strict_medical_prompt
from MedicalRAG.utils.retrival_utils import RetrievalResult



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
        
        # 获取生成器特定的配置
        self.generator_config = self.config.get('generator', {})
        llm_provider_name = self.generator_config.get('llm_provider', 'medical_vllm_api')
        
        # 通过LLM Provider获取生成器实例
        try:
            self.llm = llm_provider[llm_provider_name]
        except ValueError as e:
            self.logger.error(f"初始化LLM provider '{llm_provider_name}' 失败: {e}")
            raise e
    
    def generate_answer(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult],
        image_path: Optional[str] = None
    ) -> str:
        """
        根据检索结果生成最终答案
        
        Args:
            query: 用户原始查询
            retrieval_results: 检索结果列表
            image_path: 用户上传的图片路径 (可选)
        
        Returns:
            生成的答案字符串
        """
        if not retrieval_results and not image_path:
            return "抱歉，我没有找到相关信息来回答您的问题。"

        # --- 2. 准备图像 ---
        loaded_images = []
        image_processing_config = self.generator_config.get('image_processing', {})
        max_images = image_processing_config.get('max_images', 3)

        # 首先加载用户直接上传的图片
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                loaded_images.append(image)
                self.logger.info(f"已将用户提供的图片加入生成上下文: {image_path}")
            except Exception as e:
                self.logger.error(f"加载用户上传的图片失败 '{image_path}': {e}")
        
        # 计算还能加载多少张检索到的图片
        remaining_slots = max_images - len(loaded_images)
        
        # 然后从检索结果中加载图片
        if remaining_slots > 0:
            image_results_from_retrieval = [r for r in retrieval_results if r.retrieval_type in ['image_from_text', 'image_to_image']]
            loaded_images.extend(self._load_images(image_results_from_retrieval, limit=remaining_slots))

        image_count = len(loaded_images)
        
        # 3. 构建文本上下文 (现在知道有多少图片，可以动态调整上下文长度)
        context = self._build_context(retrieval_results, num_images=image_count)
        # print(f"context: {context}")
        # 4. 生成提示词 - 使用优化的prompt选择器
        prompt = generate_strict_medical_prompt(query, context)
        
        try:
            # 5. 调用LLM生成答案（传入文本和图像）
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

    def _load_images(self, image_results: List[Any], limit: int) -> List[Image.Image]:
        """
        从图像检索结果中加载图像文件
        
        Args:
            image_results: 图像检索结果列表
            limit: 本次调用最多加载的图片数量
        
        Returns:
            PIL.Image.Image 对象的列表
        """
        loaded_images = []
        if limit <= 0:
            return loaded_images
            
        path_pattern = re.compile(r"图片路径: (.+)")
        
        loaded_count = 0
        for result in image_results:
            if loaded_count >= limit:
                self.logger.warning(f"已达到本次加载的图像数量限制({limit})，跳过剩余图像。")
                break

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
                    loaded_count += 1
                    self.logger.info(f"成功加载图片: {image_path} ({loaded_count}/{limit})")
                except Exception as e:
                    self.logger.error(f"加载图片失败 '{image_path}': {e}")
            else:
                self.logger.warning(f"图片路径不存在: {image_path}")
                
        return loaded_images

    def _build_context(self, retrieval_results: List[Any], num_images: int = 0, force_reduce: bool = False) -> str:
        """
        仅从文本检索结果构建上下文，并根据图片数量动态调整长度。
        
        Args:
            retrieval_results: 文本检索结果列表
            num_images: 本次请求包含的图片数量
            force_reduce: 是否强制减少上下文长度
            
        Returns:
            构建的上下文字符串
        """
        # 从配置中获取上下文控制参数
        context_control_config = self.generator_config.get('context_control', {})
        model_max_len = context_control_config.get('model_max_len', 10240)
        image_token_cost = context_control_config.get('image_token_cost', 2500)
        prompt_overhead_cost = context_control_config.get('prompt_overhead_cost', 1500)
        answer_cost = context_control_config.get('answer_cost', 1000)
        chars_per_token = context_control_config.get('chars_per_token', 1.8)

        # 计算文本上下文可用的token数
        available_tokens = model_max_len - (num_images * image_token_cost) - prompt_overhead_cost - answer_cost
        
        # 将可用tokens转换为字符数
        max_text_length = int(available_tokens * chars_per_token)

        if force_reduce:
            max_text_length = min(max_text_length, 1500) # 强制截断时限制更小
        
        if max_text_length <= 0:
            self.logger.warning(f"没有足够的空间用于文本上下文 (需要 {max_text_length} 字符)。返回空上下文。")
            return ""

        self.logger.info(f"构建上下文: {num_images}张图片，文本上下文最大长度调整为: {max_text_length} 字符")

        context_parts = []
        current_length = 0
        
        for i, result in enumerate(retrieval_results, 1):
            # 获取内容，兼容不同的结果格式
            content = getattr(result, 'content', '') if hasattr(result, 'content') else str(result.get('content', ''))
            
            if not content.strip():
                continue
            
            # 计算添加这个片段后的长度
            part = content.strip()
            part_length = len(part)
            
            # 检查是否会超出调整后的长度限制
            if current_length + part_length > max_text_length:
                # 如果超出，尝试截断当前片段
                available_length = max_text_length - current_length - 20  # 预留一些空间
                if available_length > 100:  # 至少保留100字符才有意义
                    truncated_content = content[:available_length] + "..."
                    part = truncated_content
                    context_parts.append(part)
                break
            
            context_parts.append(part)
            current_length += part_length + 2  # +2 for newlines
        
        context = "\n\n".join(context_parts)
        self.logger.debug(f"构建上下文完成，总长度: {len(context)}")
        return context
