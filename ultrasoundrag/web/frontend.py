"""
UltrasoundRAG 检索测试前端界面
支持文本查询、图片查询和混合查询的检索测试界面

使用方法：
cd /media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG
python -m ultrasoundrag.web.frontend
或
streamlit run ultrasoundrag/web/frontend.py --server.port 8501
"""

import streamlit as st
import os
import time
import sys
from typing import List, Dict, Any, Optional
from PIL import Image
import io
from contextlib import contextmanager
import logging
import pathlib

# 解决相对导入问题：手动添加项目根目录到Python路径
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置包路径，确保相对导入能够正常工作
if hasattr(sys, 'path') and current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 设置页面配置
st.set_page_config(
    page_title="UltrasoundRAG 检索测试",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入重构后的模块 
try:
    from ultrasoundrag.core.retrieval.modular_retrievers import (
        create_t2t_retriever, create_t2i_retriever,
        create_i2t_retriever, create_i2i_retriever
    )
    from ultrasoundrag.core.retrieval.caption_to_image_retriever import create_caption_retriever
    from ultrasoundrag.core.retrieval.multi_database_manager import create_multi_database_manager
    from ultrasoundrag.utils import setup_logger
    from ultrasoundrag.config import config
    IMPORTS_OK = True
except ImportError as e:
    print(f"致命错误: 核心模块导入失败，无法启动应用程序")
    print(f"错误详情: {e}")
    print("请检查项目结构和依赖是否正确安装")
    import sys
    sys.exit(1)


# 初始化日志
logger = setup_logger("Frontend")

# 配置类
class FrontendConfig:
    """前端配置管理"""
    def __init__(self):
        # 显示配置
        self.max_display_results = 5
        self.default_top_k = 5
        self.max_query_length = 1000
        self.supported_image_types = ['png', 'jpg', 'jpeg']
        
        # 性能配置
        self.max_image_file_size = 10 * 1024 * 1024  # 10MB
        self.max_image_display_size = (800, 600)  # 显示图片最大尺寸
        self.max_image_upload_size = (2048, 2048)  # 上传图片最大尺寸
        self.image_quality = 85  # JPEG质量
        
        # 缓存配置
        self.cache_ttl = 300  # 5分钟缓存
        
        # UI配置
        self.max_columns_for_images = 3  # 图片显示最大列数
        # 使用配置系统获取图片根目录，避免硬编码路径
        try:
            from ultrasoundrag.config import get_config_value
            image_base_path = get_config_value('indexing.image.datasets.ultrasound_book.base_path', 'data/book/image')
            project_root = get_config_value('paths.project_root', '.')
            if not os.path.isabs(image_base_path):
                self.default_image_root = os.path.join(project_root, image_base_path)
            else:
                self.default_image_root = image_base_path
        except (ImportError, ModuleNotFoundError) as e:
            print(f"致命错误: 无法导入配置管理器，无法启动应用程序")
            print(f"错误详情: {e}")
            print("请检查项目结构和依赖是否正确安装")
            import sys
            sys.exit(1)
        except (KeyError, AttributeError) as e:
            print(f"致命错误: 配置项不存在，无法启动应用程序")
            print(f"错误详情: {e}")
            print("请检查配置文件是否正确")
            import sys
            sys.exit(1)
        except Exception as e:
            print(f"致命错误: 配置加载失败，无法启动应用程序")
            print(f"错误详情: {e}")
            print("请检查配置文件和系统环境")
            import sys
            sys.exit(1)
        # 使用跨平台兼容的临时目录
        import tempfile
        self.temp_dir = os.getenv('TEMP_DIR', tempfile.gettempdir())
        self.cache_ttl = 300  # 5分钟缓存

# 全局配置实例
config_instance = FrontendConfig()

def _get(obj: Any, key: str, default: Any = None):
    """统一安全取值，兼容 dict 与对象。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

# 检索处理类
class RetrievalHandler:
    """统一的检索处理类"""
    
    def __init__(self, retrievers: Optional[Dict[str, Any]], multi_db_manager: Optional[Any] = None) -> None:
        self.retrievers = retrievers
        self.multi_db_manager = multi_db_manager
    
    def execute_search(self, mode: str, query: str, top_k: int, is_multi: bool, selected_dbs: Optional[List[str]] = None, 
                      text_query: str = None, image_path: str = None) -> Dict[str, Any]:
        """执行检索的统一接口"""
        try:
            if is_multi and selected_dbs:
                return self._multi_db_search(mode, query, top_k, selected_dbs, text_query, image_path)
            else:
                return self._single_db_search(mode, query, top_k, text_query, image_path)
        except Exception as e:
            logger.error(f"检索执行失败 - 模式: {mode}, 错误: {e}")
            raise RetrievalError(f"检索失败: {str(e)}")
    
    def _single_db_search(self, mode: str, query: str, top_k: int, text_query: str = None, image_path: str = None) -> Dict[str, Any]:
        """单数据库检索"""
        # 对于auto和multimodal模式，分别处理
        if mode == "auto":
            # Auto模式：智能选择最佳单一检索方式
            from ultrasoundrag.core.retrieval.modular_retrievers import create_enhanced_multimodal_retriever
            retriever = create_enhanced_multimodal_retriever("default", top_k)
            return retriever.search(
                query=query,
                mode="auto",
                top_k=top_k,
                text_query=text_query,
                image_path=image_path
            )
        elif mode == "multimodal":
            # Multimodal模式：执行多模态融合检索
            from ultrasoundrag.core.retrieval.modular_retrievers import create_enhanced_multimodal_retriever
            retriever = create_enhanced_multimodal_retriever("default", top_k)
            return retriever.search(
                query=query,
                mode="multimodal",
                top_k=top_k,
                text_query=text_query,
                image_path=image_path
            )
        
        # 传统单模式检索
        if not self.retrievers or mode not in self.retrievers:
            raise ValueError(f"不支持的检索模式: {mode}")
        
        retriever = self.retrievers[mode]
        return retriever.search(query, top_k=top_k)
    
    def _multi_db_search(self, mode: str, query: str, top_k: int, selected_dbs: List[str], text_query: str = None, image_path: str = None) -> Dict[str, Any]:
        """多数据库检索"""
        if not self.multi_db_manager:
            raise ValueError("多数据库管理器未初始化")
        
        # 对于auto和multimodal模式，需要特殊处理
        if mode in ["auto", "multimodal"]:
            # 在多数据库环境中，auto和multimodal需要在每个数据库中分别执行
            # 这里简化处理：使用第一个数据库执行增强检索，其他数据库执行t2t检索
            from ultrasoundrag.core.retrieval.multi_database_manager import create_retrieval_strategy
            
            # 创建检索策略
            strategy = create_retrieval_strategy(selected_dbs, aggregation_method="weighted", max_results=top_k)
            
            # 对于auto/multimodal，统一使用t2t模式在多数据库中检索
            # 因为多数据库管理器目前不直接支持复杂的多模态检索
            fallback_mode = "t2t"  # 回退到文本检索
            return self.multi_db_manager.search_multiple_databases(strategy, fallback_mode, text_query or query)
        
        # 传统单模式的多数据库检索
        from ultrasoundrag.core.retrieval.multi_database_manager import create_retrieval_strategy
        strategy = create_retrieval_strategy(selected_dbs, aggregation_method="weighted", max_results=top_k)
        return self.multi_db_manager.search_multiple_databases(strategy, mode, query)

# 自定义异常类
class RetrievalError(Exception):
    """检索相关异常"""
    pass

class ConfigurationError(Exception):
    """配置相关异常"""
    pass

class ModelInitializationError(Exception):
    """模型初始化异常"""
    pass

class ImageProcessingError(Exception):
    """图片处理异常"""
    pass

class ValidationError(Exception):
    """输入验证异常"""
    pass

# 缓存多数据库管理器
@st.cache_resource(ttl=3600)  # 1小时缓存
def get_multi_db_manager() -> Optional[Any]:
    """获取多数据库管理器实例"""
    try:
        return create_multi_database_manager()
    except ImportError as e:
        logger.error(f"导入多数据库管理器模块失败: {e}")
        return None
    except ConnectionError as e:
        logger.error(f"数据库连接失败: {e}")
        return None
    except ConfigurationError as e:
        logger.error(f"数据库配置错误: {e}")
        return None
    except Exception as e:
        logger.error(f"多数据库管理器初始化失败: {type(e).__name__}: {e}")
        return None

# 缓存检索器实例
@st.cache_resource(ttl=3600)  # 1小时缓存
def get_retrievers(db_key: str) -> Optional[Dict[str, Any]]:
    """按数据库键获取所有检索器实例（优化版：使用共享模型）"""
    try:
        # 导入模型管理器，使用智能预加载
        from ultrasoundrag.model.model_manager import model_manager
        
        # 智能预加载：只加载必要的模型
        model_manager.preload_models(['fetal_clip', 'embedding'])
        
        retrievers = {
            't2t': create_t2t_retriever(db_key, top_k=10),
            't2i': create_t2i_retriever(db_key, top_k=10),
            'i2t': create_i2t_retriever(db_key, top_k=10),
            'i2i': create_i2i_retriever(db_key, top_k=10),
            'caption': create_caption_retriever(db_key)
        }
        return retrievers
    except ImportError as e:
        logger.error(f"导入检索器模块失败: {e}")
        st.error(f"❌ 检索器模块导入失败，请检查依赖是否正确安装")
        return None
    except ModelInitializationError as e:
        logger.error(f"模型初始化失败: {e}")
        st.error(f"❌ 模型初始化失败，请检查模型文件是否存在: {e}")
        return None
    except ConnectionError as e:
        logger.error(f"数据库连接失败: {e}")
        st.error(f"❌ 数据库连接失败，请检查Milvus服务是否正常运行")
        return None
    except Exception as e:
        logger.error(f"检索器初始化失败: {type(e).__name__}: {e}")
        st.error(f"❌ 检索器初始化失败: {type(e).__name__}: {str(e)[:100]}...")
        return None

def display_text_results(results: List[Dict], title: str = "文本结果"):
    """显示文本检索结果"""
    if not results:
        st.info("没有找到相关文本结果")
        return
    
    st.subheader(f"{title} ({len(results)} 个)")
    
    for i, result in enumerate(results):
        score = _get(result, 'score', 0.0) or 0.0
        with st.expander(f"结果 {i+1} (分数: {float(score):.4f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**内容:**")
                content = _get(result, 'content', '') or ''
                st.write(content)
                
                st.write("**元数据:**")
                metadata = _get(result, 'metadata', {}) or {}
                # 先专门展示图片相关聚合字段
                if isinstance(metadata, dict):
                    img_paths = metadata.get('image_paths')
                    if img_paths:
                        st.write("- image_paths:")
                        for p in (img_paths if isinstance(img_paths, list) else [img_paths]):
                            st.code(str(p))
                    img_caps = metadata.get('image_captions')
                    if img_caps:
                        st.write("- image_captions:")
                        for c in (img_caps if isinstance(img_caps, list) else [img_caps]):
                            st.write(f"  • {str(c)[:120]}{'...' if len(str(c))>120 else ''}")

                for key, value in (metadata.items() if isinstance(metadata, dict) else []):
                    if key not in ['relative_path', 'image_path', 'file_size', 'image_paths', 'image_captions']:
                        st.write(f"- {key}: {value}")
                
                # 显示资源集合信息
                resource_collection = _get(result, 'resource_collection', '')
                if resource_collection:
                    st.write(f"- 数据来源: {resource_collection}")
            
            with col2:
                if isinstance(metadata, dict) and ('image_path' in metadata or 'relative_path' in metadata):
                    st.write("**文件路径:**")
                    st.code(metadata.get('image_path', metadata.get('relative_path', '')))

# 输入验证函数
def validate_query(query: str, mode: str) -> tuple[bool, str]:
    """验证查询输入"""
    if not query or not query.strip():
        return False, "请输入查询内容"
    
    if len(query.strip()) > config_instance.max_query_length:
        return False, f"查询内容过长，请缩短至{config_instance.max_query_length}字符以内"
    
    return True, ""

def validate_path_security(file_path: str, base_path: str) -> tuple[bool, str]:
    """验证文件路径安全性，防止路径遍历攻击"""
    try:
        # 规范化路径
        file_path = os.path.normpath(file_path)
        base_path = os.path.normpath(base_path)
        
        # 检查是否包含危险字符
        dangerous_patterns = ['../', '..\\', '../', '..\\\\']
        for pattern in dangerous_patterns:
            if pattern in file_path:
                return False, f"路径包含不安全字符: {pattern}"
        
        # 使用pathlib检查路径是否在基础目录内
        try:
            # 修复：先将相对路径与基础路径拼接，再解析
            if not os.path.isabs(file_path):
                # 相对路径：与base_path拼接后解析
                full_path = os.path.join(base_path, file_path.lstrip("/"))
                resolved_file = pathlib.Path(full_path).resolve()
            else:
                # 绝对路径：直接解析
                resolved_file = pathlib.Path(file_path).resolve()
            
            resolved_base = pathlib.Path(base_path).resolve()
            
            # 检查文件路径是否在基础路径内
            if not str(resolved_file).startswith(str(resolved_base)):
                return False, "文件路径超出允许范围"
        except (OSError, ValueError) as e:
            return False, f"路径解析失败: {e}"
        
        return True, ""
    except Exception as e:
        logger.error(f"路径安全验证异常: {e}")
        return False, "路径验证失败"

def validate_image_file(uploaded_file) -> tuple[bool, str]:
    """验证上传的图片文件"""
    if uploaded_file is None:
        return False, "请上传图片文件"
    
    if uploaded_file.type not in [f"image/{ext}" for ext in config_instance.supported_image_types]:
        return False, f"不支持的图片格式，请使用: {', '.join(config_instance.supported_image_types)}"
    
    return True, ""

# 上下文管理器用于临时文件 - 增强版本
@contextmanager
def temporary_image_file(uploaded_file):
    """临时图片文件上下文管理器 - 增强版本，确保资源正确释放"""
    temp_path = None
    image = None
    try:
        # 生成唯一的临时文件名
        import uuid
        temp_filename = f"uploaded_image_{uuid.uuid4().hex[:8]}_{int(time.time())}.jpg"
        temp_path = os.path.join(config_instance.temp_dir, temp_filename)
        
        # 确保临时目录存在
        os.makedirs(config_instance.temp_dir, exist_ok=True)
        
        # 处理图片
        image = Image.open(uploaded_file)
        # 限制图片大小，防止内存溢出
        if image.size[0] > config_instance.max_image_upload_size[0] or image.size[1] > config_instance.max_image_upload_size[1]:
            image.thumbnail(config_instance.max_image_upload_size, Image.Resampling.LANCZOS)
        
        # 保存临时文件
        image.save(temp_path, format='JPEG', quality=config_instance.image_quality, optimize=True)
        yield temp_path
    except Exception as e:
        logger.error(f"处理临时图片文件时发生错误: {e}")
        raise ImageProcessingError(f"图片处理失败: {e}")
    finally:
        # 确保图片对象被正确关闭
        if image:
            try:
                image.close()
            except Exception as e:
                logger.warning(f"关闭图片对象失败: {e}")
        
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"临时文件已清理: {temp_path}")
            except OSError as e:
                logger.warning(f"清理临时文件失败: {temp_path}, 错误: {e}")

# 缓存图片路径解析
@st.cache_data(ttl=config_instance.cache_ttl)
def resolve_image_path_cached(relative_path: str, base_dir: str) -> Optional[str]:
    """缓存版本的图片路径解析"""
    return _resolve_image_path(relative_path, base_dir)

def _resolve_image_path(relative_path: str, base_dir: str) -> Optional[str]:
    """根据侧边栏的图片根目录解析并返回可读路径，带安全验证。"""
    if not relative_path:
        return None
    
    # 路径安全预检查
    if base_dir:
        is_safe, error_msg = validate_path_security(relative_path, base_dir)
        if not is_safe:
            logger.warning(f"不安全的图片路径: {relative_path}, 错误: {error_msg}")
            return None
    
    # 已是绝对路径且存在
    if os.path.isabs(relative_path) and os.path.exists(relative_path):
        # 对绝对路径也要进行安全检查
        if base_dir:
            is_safe, error_msg = validate_path_security(relative_path, base_dir)
            if not is_safe:
                logger.warning(f"不安全的绝对图片路径: {relative_path}")
                return None
        return relative_path
    
    # 尝试使用用户提供的根目录拼接
    if base_dir:
        candidate = os.path.join(base_dir, relative_path.lstrip("/"))
        if os.path.exists(candidate):
            # 再次验证拼接后的路径安全性
            is_safe, error_msg = validate_path_security(candidate, base_dir)
            if is_safe:
                return candidate
            else:
                logger.warning(f"拼接后路径不安全: {candidate}, 错误: {error_msg}")
    
    # 回退：尝试以当前工作目录为根（同样需要安全检查）
    cwd = os.getcwd()
    candidate = os.path.join(cwd, relative_path.lstrip("/"))
    if os.path.exists(candidate):
        is_safe, error_msg = validate_path_security(candidate, cwd)
        if is_safe:
            return candidate
        else:
            logger.warning(f"回退路径不安全: {candidate}, 错误: {error_msg}")
    
    return None


def display_image_with_fallback(image_path: str, caption: str) -> bool:
    """显示图片，带错误处理和内存优化"""
    img = None
    try:
        if not os.path.exists(image_path):
            st.warning(f"图片文件不存在: {image_path}")
            return False
        
        # 检查文件大小，避免加载过大的图片
        file_size = os.path.getsize(image_path)
        if file_size > config_instance.max_image_file_size:
            st.warning(f"图片文件过大: {file_size / 1024 / 1024:.1f}MB，请使用小于{config_instance.max_image_file_size / 1024 / 1024:.0f}MB的图片")
            return False
        
        img = Image.open(image_path)
        
        # 内存优化：限制显示图片的大小
        if img.size[0] > config_instance.max_image_display_size[0] or img.size[1] > config_instance.max_image_display_size[1]:
            # 创建缩略图
            img.thumbnail(config_instance.max_image_display_size, Image.Resampling.LANCZOS)
        
        st.image(img, caption=caption, use_container_width=True)
        return True
    except FileNotFoundError:
        st.warning(f"图片文件不存在: {image_path}")
        return False
    except (OSError, IOError) as e:
        st.warning(f"图片文件读取失败: {e}")
        st.code(f"路径: {image_path}")
        return False
    except Exception as e:
        logger.error(f"图片显示异常: {type(e).__name__}: {e}")
        st.warning(f"图片显示失败: {type(e).__name__}")
        st.code(f"路径: {image_path}")
        return False
    finally:
        # 确保图片对象被正确关闭，释放内存
        if img:
            try:
                img.close()
            except Exception as e:
                logger.warning(f"关闭图片对象失败: {e}")

def display_image_results(results: List[Dict], title: str = "图片结果", base_image_root: str = ""):
    """显示图片检索结果"""
    if not results:
        st.info("没有找到相关图片结果")
        return
    
    st.subheader(f"{title} ({len(results)} 个)")
    
    # 按列显示图片
    cols = st.columns(min(config_instance.max_columns_for_images, len(results)))
    
    for i, result in enumerate(results):
        col_idx = i % config_instance.max_columns_for_images
        with cols[col_idx]:
            with st.container():
                score = _get(result, 'score', 0.0) or 0.0
                st.write(f"**结果 {i+1}** (分数: {float(score):.4f})")
                
                # 显示图片路径信息
                metadata = _get(result, 'metadata', {}) or {}
                img_path = None
                if isinstance(metadata, dict):
                    img_path = metadata.get('image_path') or metadata.get('relative_path')
                # 顶层兜底（某些旧结果可能把路径放在顶层）
                img_path = img_path or _get(result, 'image_path') or _get(result, 'relative_path')
                if img_path:
                    resolved = resolve_image_path_cached(img_path, base_image_root)
                    if resolved and os.path.isfile(resolved):
                        display_image_with_fallback(resolved, os.path.basename(resolved))
                    else:
                        st.write(f"路径: {img_path}")
                
                # 显示描述（直接使用content字段）
                content = _get(result, 'content', '') or ''
                if content:
                    st.write(f"描述: {content[:100]}{'...' if len(content) > 100 else ''}")
                
                # 显示其他元数据
                for key, value in (metadata.items() if isinstance(metadata, dict) else []):
                    if key not in ['relative_path', 'file_size']:
                        st.write(f"- {key}: {value}")
                
                # 显示资源集合信息
                resource_collection = _get(result, 'resource_collection', '')
                if resource_collection:
                    st.write(f"- 数据来源: {resource_collection}")

def get_result_items(result_dict: Dict[str, Any], is_multi: bool) -> List[Dict]:
    """统一的结果项提取函数"""
    if not isinstance(result_dict, dict):
        return []
    key = 'aggregated_results' if is_multi else 'results'
    return result_dict.get(key, [])

def log_search_operation(mode: str, query: str, duration: float, result_count: int):
    """记录检索操作日志"""
    logger.info(f"Search completed - Mode: {mode}, Duration: {duration:.3f}s, Results: {result_count}")

def display_caption_results(results: Dict, title: str = "Caption检索结果"):
    """显示Caption检索结果"""
    st.subheader(title)
    
    if results.get('total_results', 0) == 0:
        st.info("没有找到匹配的图片")
        return
    
    st.write(f"**查询:** {results.get('caption', '')}")
    st.write(f"**找到 {results['total_results']} 个匹配图片**")
    
    for i, result in enumerate(results.get('results', [])[:config_instance.max_display_results]):  # 最多显示配置的结果数
        score = _get(result, 'score', 0.0) or 0.0
        with st.expander(f"图片 {i+1} (分数: {float(score):.4f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**图片信息:**")
                metadata = _get(result, 'metadata', {}) or {}
                if 'relative_path' in metadata:
                    st.write(f"路径: {metadata['relative_path']}")
                if 'caption' in metadata:
                    st.write(f"标题: {metadata['caption']}")
                if 'match_type' in metadata:
                    st.write(f"匹配类型: {metadata['match_type']}")
            
            with col2:
                st.write("**匹配详情:**")
                st.write(f"分数: {float(score):.4f}")
                if 'matched_variant' in metadata:
                    st.write(f"匹配变体: {metadata['matched_variant']}")
                
                # 显示资源集合信息
                resource_collection = _get(result, 'resource_collection', '')
                if resource_collection:
                    st.write(f"数据来源: {resource_collection}")

def render_sidebar() -> tuple[str, int, List[str], str]:
    """渲染侧边栏配置"""
    st.header("⚙️ 检索配置")
    
    # 检索模式选择
    retrieval_mode = st.selectbox(
        "选择检索模式",
        ["T2T (文本→文本)", "T2I (文本→图片)", "I2T (图片→文本)", 
         "I2I (图片→图片)", "Multimodal (多模态)", "Auto (自动模式)"],
        help="选择要测试的检索模式"
    )
    
    # Top-K设置
    top_k = st.slider("返回结果数量", 1, 20, config_instance.default_top_k, help="设置返回的最大结果数量")
    
    # 数据库选择：从配置动态读取可用键（支持多选融合检索）
    try:
        db_options = list(config.get('retriever', {}).get('databases', {}).keys()) or ["default"]
    except (AttributeError, KeyError) as e:
        logger.warning(f"无法读取数据库配置: {e}，使用默认选项")
        db_options = ["default"]
    except Exception as e:
        logger.error(f"配置读取异常: {e}，使用默认选项")
        db_options = ["default"]
    selected_dbs = st.multiselect("数据库(可多选)", db_options, default=[db_options[0]], help="可多选以进行多数据库融合检索")
    
    st.markdown("---")
    # 图片根目录配置
    base_image_root = st.text_input("图片根目录", value=config_instance.default_image_root, help="用于解析结果中的 relative_path 以显示图片")
    
    st.header("ℹ️ 系统信息")
    
    return retrieval_mode, top_k, selected_dbs, base_image_root

def initialize_retrievers(selected_dbs: List[str]) -> tuple[Optional[Dict[str, Any]], Optional[Any]]:
    """初始化检索器（优化版）"""
    retrievers = None
    multi_db_manager = None
    
    # 显示模型状态
    try:
        from ultrasoundrag.model.model_manager import model_manager
        model_info = model_manager.get_model_info()
        
        with st.expander("🔧 模型状态", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("FetalCLIP模型", "已加载" if model_info['fetal_clip_initialized'] else "未加载")
            with col2:
                st.metric("缓存模型数", model_info['total_cached'])
    except ImportError as e:
        logger.warning(f"无法导入模型管理器: {e}")
    except AttributeError as e:
        logger.warning(f"模型管理器接口不兼容: {e}")
    except Exception as e:
        logger.warning(f"获取模型状态异常: {type(e).__name__}: {e}")
    
    if len(selected_dbs) == 1:
        retrievers = get_retrievers(selected_dbs[0])
        if retrievers:
            st.success(f"✅ 检索器已初始化（数据库: {selected_dbs[0]}）")
        else:
            st.error("❌ 检索器初始化失败")
            return None, None
    else:
        multi_db_manager = get_multi_db_manager()
        if multi_db_manager:
            st.success(f"✅ 多数据库管理器已就绪（数据库: {', '.join(selected_dbs)}）")
        else:
            st.error("❌ 多数据库管理器初始化失败")
            return None, None
    
    return retrievers, multi_db_manager

def render_t2t_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str]):
    """渲染文本到文本检索界面"""
    st.header("📝 文本到文本检索")
    query = st.text_area("输入查询文本", placeholder="例如：心脏超声检查方法", height=100)
    
    if st.button("🔍 开始检索", type="primary"):
        is_valid, error_msg = validate_query(query, "t2t")
        if not is_valid:
            st.warning(error_msg)
            return
        
        with st.spinner("正在检索..."):
            start_time = time.time()
            try:
                result = retrieval_handler.execute_search("t2t", query, top_k, is_multi, selected_dbs)
                end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                result_items = get_result_items(result, is_multi)
                log_search_operation("t2t", query, end_time - start_time, len(result_items))
                display_text_results(result_items, "文本检索结果")
            except RetrievalError as e:
                st.error(f"🚫 检索失败: {str(e)}")
            except ValidationError as e:
                st.warning(f"⚠️ 输入验证失败: {str(e)}")
            except ConnectionError as e:
                logger.error(f"T2T检索连接异常: {e}")
                st.error("🔌 数据库连接失败，请检查网络连接或联系管理员")
            except TimeoutError as e:
                logger.error(f"T2T检索超时: {e}")
                st.error("⏰ 检索请求超时，请稍后重试")
            except Exception as e:
                logger.error(f"T2T检索异常: {type(e).__name__}: {e}")
                st.error("❌ 检索过程中发生未知错误，请稍后重试或联系技术支持")

def render_t2i_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str], base_image_root: str):
    """渲染文本到图片检索界面"""
    st.header("🖼️ 文本到图片检索")
    query = st.text_area("输入查询文本", placeholder="例如：心脏四腔心切面图", height=100)
    
    if st.button("🔍 开始检索", type="primary"):
        is_valid, error_msg = validate_query(query, "t2i")
        if not is_valid:
            st.warning(error_msg)
            return
        
        with st.spinner("正在检索..."):
            start_time = time.time()
            try:
                result = retrieval_handler.execute_search("t2i", query, top_k, is_multi, selected_dbs)
                end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                result_items = get_result_items(result, is_multi)
                log_search_operation("t2i", query, end_time - start_time, len(result_items))
                display_image_results(result_items, "图片检索结果", base_image_root)
            except RetrievalError as e:
                st.error(f"🚫 检索失败: {str(e)}")
            except ValidationError as e:
                st.warning(f"⚠️ 输入验证失败: {str(e)}")
            except ConnectionError as e:
                logger.error(f"T2I检索连接异常: {e}")
                st.error("🔌 数据库连接失败，请检查网络连接或联系管理员")
            except TimeoutError as e:
                logger.error(f"T2I检索超时: {e}")
                st.error("⏰ 检索请求超时，请稍后重试")
            except Exception as e:
                logger.error(f"T2I检索异常: {type(e).__name__}: {e}")
                st.error("❌ 检索过程中发生未知错误，请稍后重试或联系技术支持")

def render_i2t_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str]):
    """渲染图片到文本检索界面"""
    st.header("📷 图片到文本检索")
    
    # 图片上传
    uploaded_file = st.file_uploader("上传图片", type=config_instance.supported_image_types, help="支持PNG、JPG、JPEG格式")
    
    if uploaded_file is not None:
        is_valid, error_msg = validate_image_file(uploaded_file)
        if not is_valid:
            st.warning(error_msg)
            return
        
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        if st.button("🔍 开始检索", type="primary"):
            with temporary_image_file(uploaded_file) as temp_path:
                with st.spinner("正在检索..."):
                    start_time = time.time()
                    try:
                        # 多库 I2T 暂不支持图片输入融合，回退首库
                        if is_multi:
                            single_retrievers = get_retrievers(selected_dbs[0])
                            result = single_retrievers['i2t'].search(temp_path, top_k=top_k)
                        else:
                            result = retrieval_handler.execute_search("i2t", temp_path, top_k, False, selected_dbs)
                        
                        end_time = time.time()
                        
                        st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                        result_items = get_result_items(result, False)
                        log_search_operation("i2t", "image_upload", end_time - start_time, len(result_items))
                        display_text_results(result_items, "相关文本结果")
                    except RetrievalError as e:
                        st.error(f"🚫 检索失败: {str(e)}")
                    except ValidationError as e:
                        st.warning(f"⚠️ 图片验证失败: {str(e)}")
                    except ImageProcessingError as e:
                        st.error(f"🖼️ 图片处理失败: {str(e)}")
                    except ConnectionError as e:
                        logger.error(f"I2T检索连接异常: {e}")
                        st.error("🔌 数据库连接失败，请检查网络连接或联系管理员")
                    except Exception as e:
                        logger.error(f"I2T检索异常: {type(e).__name__}: {e}")
                        st.error("❌ 检索过程中发生未知错误，请稍后重试或联系技术支持")
    else:
        st.info("请上传一张图片进行检索")

def render_i2i_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str], base_image_root: str):
    """渲染图片到图片检索界面"""
    st.header("🖼️ 图片到图片检索")
    
    # 图片上传
    uploaded_file = st.file_uploader("上传图片", type=config_instance.supported_image_types, help="支持PNG、JPG、JPEG格式")
    
    if uploaded_file is not None:
        is_valid, error_msg = validate_image_file(uploaded_file)
        if not is_valid:
            st.warning(error_msg)
            return
        
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
        
        if st.button("🔍 开始检索", type="primary"):
            with temporary_image_file(uploaded_file) as temp_path:
                with st.spinner("正在检索..."):
                    start_time = time.time()
                    try:
                        # 多库 I2I 暂不支持图片输入融合，回退首库
                        if is_multi:
                            single_retrievers = get_retrievers(selected_dbs[0])
                            result = single_retrievers['i2i'].search(temp_path, top_k=top_k)
                        else:
                            result = retrieval_handler.execute_search("i2i", temp_path, top_k, False, selected_dbs)
                        
                        end_time = time.time()
                        
                        st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                        result_items = get_result_items(result, False)
                        log_search_operation("i2i", "image_upload", end_time - start_time, len(result_items))
                        display_image_results(result_items, "相似图片结果", base_image_root)
                    except RetrievalError as e:
                        st.error(f"🚫 检索失败: {str(e)}")
                    except ValidationError as e:
                        st.warning(f"⚠️ 图片验证失败: {str(e)}")
                    except ImageProcessingError as e:
                        st.error(f"🖼️ 图片处理失败: {str(e)}")
                    except ConnectionError as e:
                        logger.error(f"I2I检索连接异常: {e}")
                        st.error("🔌 数据库连接失败，请检查网络连接或联系管理员")
                    except Exception as e:
                        logger.error(f"I2I检索异常: {type(e).__name__}: {e}")
                        st.error("❌ 检索过程中发生未知错误，请稍后重试或联系技术支持")
    else:
        st.info("请上传一张图片进行检索")

def _collect_and_display_linked_images_from_text(
    t2t_items: List[Dict],
    selected_dbs: List[str],
    is_multi: bool,
    top_k: int,
    base_image_root: str,
):
    """从文本结果元数据中发现 image_captions，并按需触发 caption 匹配以展示关联图片。"""
    try:
        # 选择 caption 检索器
        if is_multi and selected_dbs:
            single_retrievers = get_retrievers(selected_dbs[0])
            caption_retriever = single_retrievers.get('caption') if single_retrievers else None
        else:
            caption_retriever = get_retrievers(selected_dbs[0]).get('caption') if selected_dbs else None

        if not caption_retriever:
            return

        any_shown = False
        for idx, item in enumerate(t2t_items or []):
            metadata = _get(item, 'metadata', {}) or {}
            image_captions = metadata.get('image_captions') if isinstance(metadata, dict) else None
            if not image_captions:
                continue
            any_shown = True
            with st.expander(f"来源文本 {idx+1} 的关联图片"):
                res = caption_retriever.search_from_md_image_captions(
                    image_captions,
                    top_k_per_caption=max(1, top_k // 2),
                    use_like=True,
                )
                display_image_results(res.get('results', []), "关联图片结果", base_image_root)
        if not any_shown:
            st.info("文本结果中未发现可用的图片标题字段")
    except ImportError as e:
        logger.warning(f"无法导入caption检索器: {e}")
    except ConnectionError as e:
        logger.warning(f"caption检索器连接失败: {e}")
    except Exception as e:
        logger.warning(f"关联图片展示失败: {type(e).__name__}: {e}")

def _execute_hybrid_retrieval(retrieval_handler: RetrievalHandler, query: str, top_k: int, is_multi: bool, selected_dbs: List[str]) -> tuple[Dict, Dict]:
    """执行混合检索的核心逻辑"""
    # 执行核心四类检索（此处示例仅执行 t2t/t2i；i2t/i2i 需图片输入，这里跳过）
    t2t_result = retrieval_handler.execute_search("t2t", query, max(1, top_k//2), is_multi, selected_dbs)
    t2i_result = retrieval_handler.execute_search("t2i", query, max(1, top_k//2), is_multi, selected_dbs)
    return t2t_result, t2i_result

def _display_hybrid_results(t2t_result: Dict, t2i_result: Dict, is_multi: bool, selected_dbs: List[str], top_k: int, base_image_root: str) -> None:
    """显示混合检索结果"""
    st.subheader("📊 检索结果汇总")
    
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📝 文本检索结果**")
        t2t_items = get_result_items(t2t_result, is_multi)
        if t2t_items:
            display_text_results(t2t_items[:config_instance.max_display_results], "文本检索结果")
        else:
            st.info("无文本检索结果")
    
    with col2:
        st.write("**🖼️ 图片检索结果**")
        t2i_items = get_result_items(t2i_result, is_multi)
        if t2i_items:
            display_image_results(t2i_items[:config_instance.max_display_results], "图片检索结果", base_image_root)
        else:
            st.info("无图片检索结果")
    
    with col3:
        st.write("**📎 文本关联图片**")
        _collect_and_display_linked_images_from_text(
            t2t_items=t2t_items,
            selected_dbs=selected_dbs,
            is_multi=is_multi,
            top_k=top_k,
            base_image_root=base_image_root,
        )

def _display_hybrid_statistics(t2t_result: Dict, t2i_result: Dict, is_multi: bool) -> None:
    """显示混合检索统计信息"""
    st.markdown("---")
    st.subheader("📈 检索统计")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("文本结果", len(get_result_items(t2t_result, is_multi)))
    with col2:
        st.metric("图片结果", len(get_result_items(t2i_result, is_multi)))
    with col3:
        st.metric("关联图片", "N/A")
    with col4:
        total_results = len(get_result_items(t2t_result, is_multi)) + len(get_result_items(t2i_result, is_multi))
        st.metric("总结果数", total_results)

def render_multimodal_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str], base_image_root: str) -> None:
    """渲染多模态检索界面"""
    st.header("🔄 多模态检索")
    st.info("多模态检索支持文本、图片或图文混合输入，智能融合多种检索策略获取最佳结果。")
    
    # 输入选项
    input_type = st.radio(
        "选择输入类型",
        ["仅文本", "仅图片", "图文混合"],
        horizontal=True
    )
    
    text_query = None
    image_path = None
    
    if input_type in ["仅文本", "图文混合"]:
        text_query = st.text_area("输入查询文本", placeholder="例如：甲状腺结节在哪？", height=100)
    
    if input_type in ["仅图片", "图文混合"]:
        image_file = st.file_uploader("选择图片", type=['jpg', 'jpeg', 'png'])
        if image_file:
            # 临时保存上传的图片
            image_path = f"/tmp/{image_file.name}"
            with open(image_path, "wb") as f:
                f.write(image_file.read())
            st.image(image_file, caption="上传的图片", width=300)
    
    if st.button("🔍 开始多模态检索", type="primary"):
        # 验证输入
        if not text_query and not image_path:
            st.warning("请至少提供文本查询或上传图片")
            return
        
        with st.spinner("正在执行多模态检索..."):
            start_time = time.time()
            
            try:
                # 使用增强多模态检索器
                result = retrieval_handler.execute_search("multimodal", text_query or image_path, top_k, is_multi, selected_dbs, text_query, image_path)
                
                end_time = time.time()
                st.success(f"多模态检索完成！耗时 {end_time - start_time:.3f} 秒")
                log_search_operation("multimodal", text_query or "图片查询", end_time - start_time, result.get('total_results', 0))
                
                # 显示结果
                results = get_result_items(result, is_multi)
                if results:
                    display_text_results(results, "多模态检索结果")
                
                # 显示检索策略统计
                if 'retrieval_breakdown' in result:
                    with st.expander("🔍 检索策略详情", expanded=False):
                        breakdown = result['retrieval_breakdown']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("T2T检索", "✓" if breakdown.get('t2t_executed') else "✗")
                        with col2:
                            st.metric("T2I检索", "✓" if breakdown.get('t2i_executed') else "✗")
                        with col3:
                            st.metric("I2T检索", "✓" if breakdown.get('i2t_executed') else "✗")
                        with col4:
                            st.metric("I2I检索", "✓" if breakdown.get('i2i_executed') else "✗")
                    
            except Exception as e:
                logger.error(f"多模态检索异常: {type(e).__name__}: {e}")
                st.error(f"❌ 多模态检索失败: {str(e)}")


def render_auto_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str], base_image_root: str) -> None:
    """渲染自动模式检索界面"""
    st.header("🤖 自动模式检索")
    st.info("自动模式会根据您的输入智能选择最合适的检索策略，无需手动指定检索类型。")
    
    # 输入区域
    st.subheader("📝 输入查询")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_query = st.text_area("输入查询文本（可选）", placeholder="例如：甲状腺结节的诊断特征", height=100)
    
    with col2:
        image_file = st.file_uploader("上传图片（可选）", type=['jpg', 'jpeg', 'png'])
        image_path = None
        if image_file:
            # 临时保存上传的图片
            image_path = f"/tmp/{image_file.name}"
            with open(image_path, "wb") as f:
                f.write(image_file.read())
            st.image(image_file, caption="上传的图片", width=200)
    
    if st.button("🚀 开始智能检索", type="primary"):
        # 验证输入
        if not text_query and not image_path:
            st.warning("请至少提供文本查询或上传图片")
            return
        
        with st.spinner("正在智能分析并执行检索..."):
            start_time = time.time()
            
            try:
                # 使用auto模式
                result = retrieval_handler.execute_search("auto", text_query or image_path, top_k, is_multi, selected_dbs, text_query, image_path)
                
                end_time = time.time()
                st.success(f"智能检索完成！耗时 {end_time - start_time:.3f} 秒")
                log_search_operation("auto", text_query or "图片查询", end_time - start_time, result.get('total_results', 0))
                
                # 显示自动选择的模式
                if 'selected_mode' in result:
                    st.info(f"🎯 系统自动选择了 **{result['selected_mode'].upper()}** 检索模式")
                
                # 显示结果
                results = get_result_items(result, is_multi)
                if results:
                    display_text_results(results, "智能检索结果")
                
                # 显示智能分析结果
                if 'parsed_input' in result:
                    with st.expander("🧠 智能分析详情", expanded=False):
                        parsed = result['parsed_input']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**检测到的输入类型:**")
                            if parsed.get('text_query'):
                                st.write("✓ 文本查询")
                            if parsed.get('image_path'):
                                st.write("✓ 图片输入")
                        with col2:
                            st.write("**选择的检索策略:**")
                            st.write(f"📋 {result.get('selected_mode', 'unknown').upper()}")
                
            except Exception as e:
                logger.error(f"自动检索异常: {type(e).__name__}: {e}")
                st.error(f"❌ 自动检索失败: {str(e)}")

def main():
    """主界面"""
    st.title("🔍 UltrasoundRAG 检索测试界面")
    st.markdown("---")
    
    # 侧边栏配置
    with st.sidebar:
        retrieval_mode, top_k, selected_dbs, base_image_root = render_sidebar()
        retrievers, multi_db_manager = initialize_retrievers(selected_dbs)
    
    # 主界面
    if len(selected_dbs) == 1 and not retrievers:
        print("致命错误: 检索器初始化失败，无法启动应用程序")
        print("请检查系统配置和依赖是否正确安装")
        import sys
        sys.exit(1)
    elif len(selected_dbs) > 1 and not multi_db_manager:
        print("致命错误: 多数据库管理器初始化失败，无法启动应用程序")
        print("请检查系统配置和数据库连接")
        import sys
        sys.exit(1)

    # 创建检索处理器
    retrieval_handler = RetrievalHandler(retrievers, multi_db_manager)
    is_multi = len(selected_dbs) > 1
    
    # 根据检索模式显示不同的输入界面
    if retrieval_mode == "T2T (文本→文本)":
        render_t2t_interface(retrieval_handler, top_k, is_multi, selected_dbs)
    elif retrieval_mode == "T2I (文本→图片)":
        render_t2i_interface(retrieval_handler, top_k, is_multi, selected_dbs, base_image_root)
    elif retrieval_mode == "I2T (图片→文本)":
        render_i2t_interface(retrieval_handler, top_k, is_multi, selected_dbs)
    elif retrieval_mode == "I2I (图片→图片)":
        render_i2i_interface(retrieval_handler, top_k, is_multi, selected_dbs, base_image_root)
    elif retrieval_mode == "Multimodal (多模态)":
        render_multimodal_interface(retrieval_handler, top_k, is_multi, selected_dbs, base_image_root)
    elif retrieval_mode == "Auto (自动模式)":
        render_auto_interface(retrieval_handler, top_k, is_multi, selected_dbs, base_image_root)
    
    # 页脚信息
    st.markdown("---")
    st.markdown("**UltrasoundRAG 检索系统** | 支持多种检索模式和混合查询")

if __name__ == "__main__":
    main()
