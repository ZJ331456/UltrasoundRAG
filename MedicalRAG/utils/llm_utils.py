"""
大语言模型工具模块，支持API和本地模型两种调用方式

主要功能：
- 支持OpenAI格式API调用（如vLLM、OpenAI等）
- 支持本地模型加载和推理（如transformers）
- 统一的接口设计，方便切换不同模型
- 支持流式输出和批量处理
- 自动处理模型prompt模板
- 健壮的错误处理机制

使用示例：
    from MedicalRAG.utils.llm_utils import llm_provider
    
    # API方式
    try:
        llm = llm_provider['generator']
        response = llm.generate("你好，世界！")
    except Exception as e:
        print(f"调用失败: {e}")
    
    # 本地模型方式 (假设在config.yaml中定义了名为'local_model'的provider)
    # try:
    #     llm_local = llm_provider['local_model']
    #     response = llm_local.generate("你好，世界！")
    # except Exception as e:
    #     print(f"调用失败: {e}")
"""

import os
import json
import torch
import requests
import base64
from io import BytesIO
from threading import Thread
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, AutoProcessor
from PIL import Image

from MedicalRAG.config.config import config
from MedicalRAG.utils.logger import setup_logger


class LLMError(Exception):
    """自定义LLM模块异常基类"""
    pass

class APIError(LLMError):
    """API相关异常"""
    pass

class ModelLoadError(LLMError):
    """本地模型加载异常"""
    pass

class GenerationError(LLMError):
    """文本生成异常"""
    pass


def pil_to_base64(image: Image.Image, format="JPEG") -> str:
    """将Pillow Image对象转换为Base64字符串"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


class BaseLLM(ABC):
    """大语言模型基类"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.is_vision_model = False # 默认为非视觉模型
    
    @abstractmethod
    def generate(self, prompt: str, images: Optional[List[Image.Image]] = None, **kwargs) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, images: Optional[List[Image.Image]] = None, **kwargs) -> Generator[str, None, None]:
        """流式生成文本"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], images: Optional[List[Image.Image]] = None, **kwargs) -> str:
        """对话模式"""
        pass

    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, str]], images: Optional[List[Image.Image]] = None, **kwargs) -> Generator[str, None, None]:
        """流式对话模式"""
        pass


class APILLM(BaseLLM):
    """API方式调用大语言模型"""
    
    def __init__(
        self, 
        api_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 30,
        is_vision_model: bool = False,
        endpoint_path: Optional[str] = None # 新增：允许自定义端点路径
    ):
        """
        初始化API调用器
        
        Args:
            api_url: API服务地址
            model_name: 模型名称
            api_key: API密钥
            max_tokens: 最大生成长度
            temperature: 采样温度
            timeout: 请求超时时间
            is_vision_model: 此API是否为视觉模型
            endpoint_path: (可选) 自定义的API端点路径, None则默认'/chat/completions'
        """
        super().__init__()
        
        self.api_url = api_url.rstrip('/') if api_url else None
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.is_vision_model = is_vision_model
        self.endpoint_path = endpoint_path # 存储自定义端点
        
        if not self.api_url or not self.model_name:
            raise ValueError("APILLM缺少必要的参数: api_url 和 model_name")
        
        self.logger.info(f"初始化API LLM - URL: {self.api_url}, Model: {self.model_name}, Vision: {self.is_vision_model}")
    
    def _make_request(self, messages: List[Dict[str, str]], images: Optional[List[Image.Image]] = None, stream: bool = False, **kwargs) -> requests.Response:
        """发送API请求 (支持多模态)"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        if not self.api_key:
            headers.pop('Authorization') # 如果没有key，则不发送
            
        final_messages = [msg.copy() for msg in messages]

        # 如果是视觉模型且有图像，则构建多模态消息
        if self.is_vision_model and images:
            user_message = final_messages[-1]
            if user_message.get("role") == "user":
                new_content = [{"type": "text", "text": user_message["content"]}]
                for image in images:
                    base64_image = pil_to_base64(image)
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_image}
                    })
                user_message["content"] = new_content
            else:
                self.logger.warning("图像已提供给API LLM，但最后一条消息不是来自用户。图像将被忽略。")
            
        data = {
            'model': self.model_name,
            'messages': final_messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'stream': stream
        }
        
        # 添加其他参数
        for key, value in kwargs.items():
            if key not in ['max_tokens', 'temperature'] and value is not None:
                data[key] = value
        
        try:
            # 如果未定义endpoint_path，则默认为/chat/completions以实现向后兼容
            # 如果定义为""(空字符串), 则不拼接任何路径
            path_to_append = self.endpoint_path if self.endpoint_path is not None else "/chat/completions"
            request_url = f"{self.api_url}{path_to_append}"
            self.logger.debug(f"向API发送请求: URL={request_url}")

            # 使用配置的超时时间
            response = requests.post(
                request_url, 
                headers=headers, 
                json=data,
                stream=stream,
                timeout=self.timeout
            )
            response.raise_for_status()  # 如果状态码不是2xx，则抛出HTTPError
            
            # 处理流式和非流式响应
            if stream:
                return response
            else:
                result = response.json()
                
                if not result.get('choices'):
                    self.logger.error(f"API返回无效数据: {result}")
                    raise APIError(f"API返回无效数据: {result}")

                return result['choices'][0]['message']['content']
                
        except (APIError, json.JSONDecodeError) as e:
            self.logger.error(f"对话生成失败: {e}")
            raise GenerationError(f"对话生成失败: {e}") from e
    
    def chat(self, messages: List[Dict[str, str]], images: Optional[List[Image.Image]] = None, **kwargs) -> str:
        """对话模式"""
        try:
            response = self._make_request(messages, images=images, stream=False, **kwargs)
            return response # 修复：返回从 _make_request 获取的响应
            
        except (APIError, GenerationError) as e:
            self.logger.error(f"对话生成失败: {e}")
            raise GenerationError(f"对话生成失败: {e}") from e

    def chat_stream(self, messages: List[Dict[str, str]], images: Optional[List[Image.Image]] = None, **kwargs) -> Generator[str, None, None]:
        """流式对话模式"""
        try:
            response = self._make_request(messages, images=images, stream=True, **kwargs)
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        line_text = line_text[6:]
                        if line_text.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(line_text)
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            self.logger.warning(f"无法解析流式数据中的JSON行: {line_text}")
                            continue
            
        except APIError as e:
            self.logger.error(f"流式对话生成失败: {e}")
            raise GenerationError(f"流式对话生成失败: {e}") from e

    def generate(self, prompt: str, images: Optional[List[Image.Image]] = None, **kwargs) -> str:
        """生成文本"""
        messages = [{'role': 'user', 'content': prompt}]
        return self.chat(messages, images=images, **kwargs)
    
    def generate_stream(self, prompt: str, images: Optional[List[Image.Image]] = None, **kwargs) -> Generator[str, None, None]:
        """流式生成文本"""
        messages = [{'role': 'user', 'content': prompt}]
        yield from self.chat_stream(messages, images=images, **kwargs)


class LocalLLM(BaseLLM):
    """本地模型方式调用大语言模型 (已扩展支持多模态)"""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype_str: Optional[str] = None, # e.g. "float16"
        max_length: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs # 捕获多余的参数
    ):
        """
        初始化本地模型
        
        Args:
            model_path: 模型路径
            device: 计算设备
            torch_dtype_str: torch数据类型字符串
            max_length: 最大生成长度
            do_sample: 是否采样
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            repetition_penalty: 重复惩罚
        """
        super().__init__()
        
        if not model_path:
            raise ValueError("LocalLLM缺少必要的参数: model_path")
            
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = getattr(torch, torch_dtype_str) if torch_dtype_str else torch.float16
        
        self.generation_params = {
            "max_length": max_length,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }
        self.logger.info(f"LocalLLM generation params: {self.generation_params}")

        self.tokenizer = None
        self.model = None
        self.processor = None
        self.is_vision_model = False # 标记是否为多模态模型
        
        self._load_model()

    def _load_model(self):
        """加载模型和分词器/处理器"""
        self.logger.info(f"开始从 '{self.model_path}' 加载本地模型...")
        try:
            # 优先尝试加载多模态处理器
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer
                self.is_vision_model = True
                self.logger.info("成功加载多模态 AutoProcessor，模型被视为视觉模型。")
            except Exception:
                self.logger.info("加载 AutoProcessor 失败，回退到加载普通 AutoTokenizer。")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.is_vision_model = False

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()
            self.logger.info(f"模型成功加载到设备: {self.device}")
            
        except ImportError as e:
            self.logger.error(f"模型加载失败，缺少必要的库: {e}")
            raise ModelLoadError(f"模型加载失败，缺少必要的库: {e}") from e
        except Exception as e:
            self.logger.error(f"从 '{self.model_path}' 加载模型时发生未知错误: {e}")
            raise ModelLoadError(f"从 '{self.model_path}' 加载模型时发生未知错误: {e}") from e

    def _prepare_chat_input(
        self, 
        messages: List[Dict[str, str]], 
        images: Optional[List[Image.Image]] = None,
        **kwargs
    ):
        """
        准备模型输入，支持多模态
        """
        if self.is_vision_model and images:
            # 多模态输入处理
            if not self.processor:
                raise ModelLoadError("模型被标记为视觉模型，但多模态处理器 'processor' 未被加载。")
            
            # 从messages中提取文本
            text_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 使用processor处理文本和图像
            inputs = self.processor(
                text=text_prompt, 
                images=images, 
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)
            
            return inputs
        else:
            # 纯文本输入处理
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
            return {"input_ids": input_ids}
    
    def chat(self, messages: List[Dict[str, str]], images: Optional[List[Image.Image]] = None, **kwargs) -> str:
        """
        对话模式，支持图像输入
        
        Args:
            messages: OpenAI格式的对话消息列表
            images: Pillow Image对象列表 (可选)
            **kwargs: 传递给模型generate方法的额外参数
        """
        if not messages:
            return ""

        # 合并生成参数
        gen_kwargs = {**self.generation_params, **kwargs}
        
        try:
            # 准备模型输入
            inputs = self._prepare_chat_input(messages, images=images, **kwargs)
            
            # 生成输出
            gen_outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # 解码输出
            # 对于多模态输入，需要从原始输入的长度之后开始解码，以避免重复显示prompt
            input_length = inputs['input_ids'].shape[1]
            output_ids = gen_outputs[0][input_length:]
            
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            self.logger.error(f"本地模型生成失败: {e}", exc_info=True)
            raise GenerationError(f"本地模型生成失败: {e}") from e
    
    def chat_stream(self, messages: List[Dict[str, str]], images: Optional[List[Image.Image]] = None, **kwargs) -> Generator[str, None, None]:
        """
        流式对话模式 (多模态的流式输出较为复杂，此处简化实现)
        注意：多模态的流式输出可能无法像纯文本一样完美工作，因为它需要更复杂的处理逻辑。
        """
        if not messages:
            return

        # 合并生成参数
        gen_kwargs = {**self.generation_params, **kwargs}

        try:
            # 准备模型输入
            inputs = self._prepare_chat_input(messages, images=images, **kwargs)
            input_length = inputs['input_ids'].shape[1]

            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )

            # 在单独的线程中运行生成，以允许流式读取
            generation_kwargs = dict(inputs, streamer=streamer, **gen_kwargs)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # 从streamer中获取生成的文本
            for new_text in streamer:
                yield new_text
        
        except Exception as e:
            self.logger.error(f"本地模型流式生成失败: {e}", exc_info=True)
            raise GenerationError(f"本地模型流式生成失败: {e}") from e

    def generate(self, prompt: str, images: Optional[List[Image.Image]] = None, **kwargs) -> str:
        """生成文本 (支持图像)"""
        messages = [{'role': 'user', 'content': prompt}]
        return self.chat(messages, images=images, **kwargs)

    def generate_stream(self, prompt: str, images: Optional[List[Image.Image]] = None, **kwargs) -> Generator[str, None, None]:
        """流式生成文本 (支持图像)"""
        messages = [{'role': 'user', 'content': prompt}]
        yield from self.chat_stream(messages, images=images, **kwargs)


class LLMProvider:
    """
    大语言模型提供者（工厂类）
    根据配置创建并缓存LLM实例
    """
    _instances = {}

    def __init__(self):
        self.logger = setup_logger(__name__)

    def __getitem__(self, name: str) -> BaseLLM:
        """
        通过名称获取LLM实例，实现字典式访问
        e.g., llm_provider['generator']
        """
        if name not in self._instances:
            self.logger.info(f"LLM Provider: 缓存中未找到 '{name}'，正在创建新实例...")
            
            provider_config = config.get('llm_providers', {}).get(name)
            if not provider_config:
                raise ValueError(f"在config.yaml的'llm_providers'下未找到名为'{name}'的配置")

            provider_type = provider_config.get('type')
            params = provider_config.get('params', {})
            
            if provider_type == 'api':
                self._instances[name] = APILLM(**params)
            elif provider_type == 'local':
                self._instances[name] = LocalLLM(**params)
            else:
                raise ValueError(f"未知的LLM提供者类型: '{provider_type}'")
        
        return self._instances[name]


llm_provider = LLMProvider()


# 示例使用
if __name__ == "__main__":
    # API方式示例
    print("=== API方式测试 (generator) ===")
    try:
        # 假设config.yaml中有 'llm_providers.generator' 或 'medical_vllm_api'
        # 正确的用法是直接通过provider名称获取实例, 例如:
        # llm_api = llm_provider['generator']
        
        # 修复: 直接使用provider名称'medical_vllm_api'获取LLM实例
        # 之前的代码 llm_provider['llm_providers']... 是错误的用法
        # llm_api = llm_provider['medical_vllm_api']
        llm_api = llm_provider['xiaohumini_llm_api']
        # --- 非流式测试 ---
        print("\n--- 非流式 ---")
        response = llm_api.generate("你好，请介绍一下你自己，并用3句话总结。")
        print(f"API响应: {response}")
        
        # --- 流式测试 ---
        print("\n--- 流式 ---")
        stream_response = llm_api.generate_stream("你好，请介绍一下你自己，并用3句话总结。")
        full_response = ""
        for chunk in stream_response:
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n流式传输结束.")
        
    except Exception as e:
        print(f"\nAPI测试失败: {e}")
    
    # 本地模型方式示例
    # print("\n\n=== 本地模型测试 (需要预先配置) ===")
    # print("注意: 本地模型测试需要你在config.yaml中配置一个名为'local_model'的provider")
    # try:
    #     # 假设config.yaml中有 'llm_providers.local_model'
    #     llm_local = llm_provider['local_model']
    #     # 
    #     print("\n--- 非流式 ---")
    #     response_local = llm_local.generate("你好，请介绍一下你自己，并用3句话总结。")
    #     print(f"本地模型响应: {response_local}")
    #     # 
    #     print("\n--- 流式 ---")
    #     stream_local = llm_local.generate_stream("你好，请介绍一下你自己，并用3句话总结。")
    #     for chunk in stream_local:
    #         print(chunk, end="", flush=True)
    #     print("\n流式传输结束.")
    #     # 
    #     print("跳过本地模型测试，如需测试请取消注释并配置。")
    except Exception as e:
        print(f"\n本地模型测试失败: {e}")