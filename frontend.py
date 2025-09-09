"""
UltrasoundRAG 简单前端界面
支持文本查询、图片查询和混合查询的检索测试界面

使用方法：
python frontend.py
然后在浏览器中访问 http://localhost:8501

streamlit run frontend.py --server.port 8502
"""

import streamlit as st
import os
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import io
from contextlib import contextmanager
import logging

# 设置页面配置
st.set_page_config(
    page_title="UltrasoundRAG 检索测试",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目路径
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "UltrasoundRAG"))

# 导入检索模块
from UltrasoundRAG.retrival.modular_retrievers import (
    create_t2t_retriever, create_t2i_retriever,
    create_i2t_retriever, create_i2i_retriever
)
from UltrasoundRAG.retrival.caption_to_image_retriever import create_caption_retriever
from UltrasoundRAG.retrival.multi_database_manager import create_multi_database_manager
from UltrasoundRAG.utils.logger import setup_logger
from UltrasoundRAG.config import config


# 初始化日志
logger = setup_logger("Frontend")

# 配置类
class FrontendConfig:
    """前端配置管理"""
    def __init__(self):
        self.max_display_results = 5
        self.default_top_k = 5
        self.max_query_length = 1000
        self.supported_image_types = ['png', 'jpg', 'jpeg']
        self.default_image_root = os.getenv('IMAGE_ROOT', '/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/image')
        self.temp_dir = os.getenv('TEMP_DIR', '/tmp')
        self.cache_ttl = 300  # 5分钟缓存

# 全局配置实例
config_instance = FrontendConfig()

# 检索处理类
class RetrievalHandler:
    """统一的检索处理类"""
    
    def __init__(self, retrievers: Dict, multi_db_manager=None):
        self.retrievers = retrievers
        self.multi_db_manager = multi_db_manager
    
    def execute_search(self, mode: str, query: str, top_k: int, is_multi: bool, selected_dbs: List[str] = None):
        """执行检索的统一接口"""
        try:
            if is_multi and selected_dbs:
                return self._multi_db_search(mode, query, top_k, selected_dbs)
            else:
                return self._single_db_search(mode, query, top_k)
        except Exception as e:
            logger.error(f"检索执行失败 - 模式: {mode}, 错误: {e}")
            raise RetrievalError(f"检索失败: {str(e)}")
    
    def _single_db_search(self, mode: str, query: str, top_k: int):
        """单数据库检索"""
        if mode not in self.retrievers:
            raise ValueError(f"不支持的检索模式: {mode}")
        
        retriever = self.retrievers[mode]
        if mode == 'caption':
            return retriever.search_single_caption(query, top_k=top_k)
        else:
            return retriever.search(query, top_k=top_k)
    
    def _multi_db_search(self, mode: str, query: str, top_k: int, selected_dbs: List[str]):
        """多数据库检索"""
        if not self.multi_db_manager:
            raise ValueError("多数据库管理器未初始化")
        
        strategy = self.multi_db_manager.create_strategy if hasattr(self.multi_db_manager, 'create_strategy') else None
        if strategy is None:
            from UltrasoundRAG.retrival.multi_database_manager import create_retrieval_strategy
            strat = create_retrieval_strategy(selected_dbs, aggregation_method="weighted", max_results=top_k)
        else:
            strat = strategy(selected_dbs, aggregation_method="weighted", max_results=top_k)
        
        return self.multi_db_manager.search_multiple_databases(strat, mode, query)

# 自定义异常类
class RetrievalError(Exception):
    """检索相关异常"""
    pass

# 缓存多数据库管理器
@st.cache_resource
def get_multi_db_manager():
    """获取多数据库管理器实例"""
    try:
        return create_multi_database_manager()
    except Exception as e:
        logger.error(f"多数据库管理器初始化失败: {e}")
        return None

# 缓存检索器实例
@st.cache_resource
def get_retrievers(db_key: str):
    """按数据库键获取所有检索器实例（优化版：使用共享模型）"""
    try:
        # 导入模型管理器，使用智能预加载
        from UltrasoundRAG.model.model_manager import model_manager
        
        # 智能预加载：只加载必要的模型
        model_manager.preload_models(['fetal_clip', 'embedding'])
        
        retrievers = {
            't2t': create_t2t_retriever(db_key, top_k=10),
            't2i': create_t2i_retriever(db_key, top_k=10),
            'i2t': create_i2t_retriever(db_key, top_k=10),
            'i2i': create_i2i_retriever(db_key, top_k=10),
            'caption': create_caption_retriever(db_key, "hybrid_match")
        }
        return retrievers
    except Exception as e:
        st.error(f"初始化检索器失败: {e}")
        return None

def display_text_results(results: List[Dict], title: str = "文本结果"):
    """显示文本检索结果"""
    if not results:
        st.info("没有找到相关文本结果")
        return
    
    st.subheader(f"{title} ({len(results)} 个)")
    
    for i, result in enumerate(results):
        with st.expander(f"结果 {i+1} (分数: {getattr(result, 'score', 0):.4f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**内容:**")
                content = getattr(result, 'content', '')
                st.write(content[:500] + "..." if len(content) > 500 else content)
                
                st.write("**元数据:**")
                metadata = getattr(result, 'metadata', {})
                for key, value in metadata.items():
                    if key not in ['relative_path', 'file_size']:
                        st.write(f"- {key}: {value}")
                
                # 显示资源集合信息
                resource_collection = getattr(result, 'resource_collection', '')
                if resource_collection:
                    st.write(f"- 数据来源: {resource_collection}")
            
            with col2:
                if 'relative_path' in getattr(result, 'metadata', {}):
                    st.write("**文件路径:**")
                    st.code(getattr(result, 'metadata', {}).get('relative_path', ''))

# 输入验证函数
def validate_query(query: str, mode: str) -> tuple[bool, str]:
    """验证查询输入"""
    if not query or not query.strip():
        return False, "请输入查询内容"
    
    if len(query.strip()) > config_instance.max_query_length:
        return False, f"查询内容过长，请缩短至{config_instance.max_query_length}字符以内"
    
    return True, ""

def validate_image_file(uploaded_file) -> tuple[bool, str]:
    """验证上传的图片文件"""
    if uploaded_file is None:
        return False, "请上传图片文件"
    
    if uploaded_file.type not in [f"image/{ext}" for ext in config_instance.supported_image_types]:
        return False, f"不支持的图片格式，请使用: {', '.join(config_instance.supported_image_types)}"
    
    return True, ""

# 上下文管理器用于临时文件
@contextmanager
def temporary_image_file(uploaded_file):
    """临时图片文件上下文管理器"""
    temp_path = None
    try:
        temp_path = os.path.join(config_instance.temp_dir, f"uploaded_image_{int(time.time())}.jpg")
        image = Image.open(uploaded_file)
        image.save(temp_path)
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(f"清理临时文件失败: {e}")

# 缓存图片路径解析
@st.cache_data(ttl=config_instance.cache_ttl)
def resolve_image_path_cached(relative_path: str, base_dir: str) -> Optional[str]:
    """缓存版本的图片路径解析"""
    return _resolve_image_path(relative_path, base_dir)

def _resolve_image_path(relative_path: str, base_dir: str) -> Optional[str]:
    """根据侧边栏的图片根目录解析并返回可读路径。"""
    if not relative_path:
        return None
    # 已是绝对路径且存在
    if os.path.isabs(relative_path) and os.path.exists(relative_path):
        return relative_path
    # 尝试使用用户提供的根目录拼接
    if base_dir:
        candidate = os.path.join(base_dir, relative_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate
    # 回退：尝试以当前工作目录为根
    candidate = os.path.join(os.getcwd(), relative_path.lstrip("/"))
    if os.path.exists(candidate):
        return candidate
    return None


def display_image_with_fallback(image_path: str, caption: str) -> bool:
    """显示图片，带错误处理"""
    try:
        if not os.path.exists(image_path):
            st.warning(f"图片文件不存在: {image_path}")
            return False
        
        img = Image.open(image_path)
        st.image(img, caption=caption, use_container_width=True)
        return True
    except Exception as e:
        st.warning(f"无法显示图片: {e}")
        st.code(f"路径: {image_path}")
        return False

def display_image_results(results: List[Dict], title: str = "图片结果", base_image_root: str = ""):
    """显示图片检索结果"""
    if not results:
        st.info("没有找到相关图片结果")
        return
    
    st.subheader(f"{title} ({len(results)} 个)")
    
    # 按列显示图片
    cols = st.columns(min(3, len(results)))
    
    for i, result in enumerate(results):
        col_idx = i % 3
        with cols[col_idx]:
            with st.container():
                st.write(f"**结果 {i+1}** (分数: {getattr(result, 'score', 0):.4f})")
                
                # 显示图片路径信息
                metadata = getattr(result, 'metadata', {})
                rel_path = metadata.get('relative_path')
                if rel_path:
                    resolved = resolve_image_path_cached(rel_path, base_image_root)
                    if resolved and os.path.isfile(resolved):
                        display_image_with_fallback(resolved, os.path.basename(resolved))
                    else:
                        st.write(f"路径: {rel_path}")
                
                # 显示caption
                content = getattr(result, 'content', '')
                if content:
                    st.write(f"描述: {content[:100]}{'...' if len(content) > 100 else ''}")
                
                # 显示其他元数据
                for key, value in metadata.items():
                    if key not in ['relative_path', 'file_size']:
                        st.write(f"- {key}: {value}")
                
                # 显示资源集合信息
                resource_collection = getattr(result, 'resource_collection', '')
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
    
    for i, result in enumerate(results.get('results', [])[:5]):  # 最多显示5个
        with st.expander(f"图片 {i+1} (分数: {getattr(result, 'score', 0):.4f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**图片信息:**")
                metadata = getattr(result, 'metadata', {})
                if 'relative_path' in metadata:
                    st.write(f"路径: {metadata['relative_path']}")
                if 'caption' in metadata:
                    st.write(f"标题: {metadata['caption']}")
                if 'match_type' in metadata:
                    st.write(f"匹配类型: {metadata['match_type']}")
            
            with col2:
                st.write("**匹配详情:**")
                st.write(f"分数: {getattr(result, 'score', 0):.4f}")
                if 'matched_variant' in metadata:
                    st.write(f"匹配变体: {metadata['matched_variant']}")
                
                # 显示资源集合信息
                resource_collection = getattr(result, 'resource_collection', '')
                if resource_collection:
                    st.write(f"数据来源: {resource_collection}")

def render_sidebar():
    """渲染侧边栏配置"""
    st.header("⚙️ 检索配置")
    
    # 检索模式选择
    retrieval_mode = st.selectbox(
        "选择检索模式",
        ["T2T (文本→文本)", "T2I (文本→图片)", "I2T (图片→文本)", 
         "I2I (图片→图片)", "Caption (标题→图片)", "混合检索"],
        help="选择要测试的检索模式"
    )
    
    # Top-K设置
    top_k = st.slider("返回结果数量", 1, 20, config_instance.default_top_k, help="设置返回的最大结果数量")
    
    # 数据库选择：从配置动态读取可用键（支持多选融合检索）
    try:
        db_options = list(config.get('retriever', {}).get('databases', {}).keys()) or ["default"]
    except Exception:
        db_options = ["default"]
    selected_dbs = st.multiselect("数据库(可多选)", db_options, default=[db_options[0]], help="可多选以进行多数据库融合检索")
    
    st.markdown("---")
    # 图片根目录配置
    base_image_root = st.text_input("图片根目录", value=config_instance.default_image_root, help="用于解析结果中的 relative_path 以显示图片")
    
    st.header("ℹ️ 系统信息")
    
    return retrieval_mode, top_k, selected_dbs, base_image_root

def initialize_retrievers(selected_dbs: List[str]):
    """初始化检索器（优化版）"""
    retrievers = None
    multi_db_manager = None
    
    # 显示模型状态
    try:
        from UltrasoundRAG.model.model_manager import model_manager
        model_info = model_manager.get_model_info()
        
        with st.expander("🔧 模型状态", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("FetalCLIP模型", "已加载" if model_info['fetal_clip_initialized'] else "未加载")
            with col2:
                st.metric("缓存模型数", model_info['total_cached'])
    except Exception as e:
        logger.warning(f"无法获取模型状态: {e}")
    
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
                st.error(str(e))
            except Exception as e:
                logger.error(f"T2T检索异常: {e}")
                st.error("检索过程中发生错误，请稍后重试")

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
                st.error(str(e))
            except Exception as e:
                logger.error(f"T2I检索异常: {e}")
                st.error("检索过程中发生错误，请稍后重试")

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
                        st.error(str(e))
                    except Exception as e:
                        logger.error(f"I2T检索异常: {e}")
                        st.error("检索过程中发生错误，请稍后重试")
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
                        st.error(str(e))
                    except Exception as e:
                        logger.error(f"I2I检索异常: {e}")
                        st.error("检索过程中发生错误，请稍后重试")
    else:
        st.info("请上传一张图片进行检索")

def render_caption_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str], base_image_root: str):
    """渲染标题到图片检索界面"""
    st.header("🏷️ 标题到图片检索")
    
    # 单个caption检索
    st.subheader("单个Caption检索")
    caption = st.text_input("输入图片标题", placeholder="例如：图2-3 心脏超声横切面")
    
    if st.button("🔍 检索单个Caption", type="primary"):
        is_valid, error_msg = validate_query(caption, "caption")
        if not is_valid:
            st.warning(error_msg)
            return
        
        with st.spinner("正在检索..."):
            start_time = time.time()
            try:
                if is_multi:
                    result = retrieval_handler.execute_search("t2i", caption, top_k, is_multi, selected_dbs)
                else:
                    result = retrieval_handler.execute_search("caption", caption, top_k, is_multi, selected_dbs)
                
                end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                result_items = get_result_items(result, is_multi)
                log_search_operation("caption", caption, end_time - start_time, len(result_items))
                
                # 多库时复用图片结果渲染
                if is_multi:
                    display_image_results(result_items, "Caption检索结果", base_image_root)
                else:
                    display_caption_results(result, "Caption检索结果")
            except RetrievalError as e:
                st.error(str(e))
            except Exception as e:
                logger.error(f"Caption检索异常: {e}")
                st.error("检索过程中发生错误，请稍后重试")
    
    st.markdown("---")
    
    # 从文本块提取caption检索
    st.subheader("从文本块提取Caption检索")
    text_chunk = st.text_area("输入包含图片引用的文本", 
                            placeholder="例如：图2-1显示心脏四腔心切面，图2-2为心脏短轴切面", 
                            height=100)
    
    if st.button("🔍 提取并检索", type="secondary"):
        is_valid, error_msg = validate_query(text_chunk, "text_chunk")
        if not is_valid:
            st.warning(error_msg)
            return
        
        with st.spinner("正在提取Caption并检索..."):
            start_time = time.time()
            try:
                if not is_multi:
                    result = retrieval_handler.retrievers['caption'].search_from_text_chunk(text_chunk, top_k=top_k)
                else:
                    st.warning("多数据库模式下暂不支持文本块提取功能")
                    return
                
                end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                
                # 显示提取的captions
                extracted_captions = result.get('extracted_captions', [])
                if extracted_captions:
                    st.write(f"**提取到的Caption:** {', '.join(extracted_captions)}")
                
                # 显示检索结果
                display_image_results(result.get('results', []), "匹配的图片结果", base_image_root)
            except RetrievalError as e:
                st.error(str(e))
            except Exception as e:
                logger.error(f"文本块Caption检索异常: {e}")
                st.error("检索过程中发生错误，请稍后重试")

def render_hybrid_interface(retrieval_handler: RetrievalHandler, top_k: int, is_multi: bool, selected_dbs: List[str], base_image_root: str):
    """渲染混合检索界面"""
    st.header("🔄 混合检索")
    st.info("混合检索将同时执行多种检索模式并融合结果")
    
    query = st.text_area("输入查询文本", placeholder="例如：心脏超声检查", height=100)
    
    if st.button("🔍 开始混合检索", type="primary"):
        is_valid, error_msg = validate_query(query, "hybrid")
        if not is_valid:
            st.warning(error_msg)
            return
        
        with st.spinner("正在执行混合检索..."):
            start_time = time.time()
            
            try:
                # 执行多种检索
                if not is_multi:
                    t2t_result = retrieval_handler.execute_search("t2t", query, top_k//2, False, selected_dbs)
                    t2i_result = retrieval_handler.execute_search("t2i", query, top_k//2, False, selected_dbs)
                    caption_result = retrieval_handler.execute_search("caption", query, top_k//2, False, selected_dbs)
                else:
                    t2t_result = retrieval_handler.execute_search("t2t", query, top_k//2, True, selected_dbs)
                    t2i_result = retrieval_handler.execute_search("t2i", query, top_k//2, True, selected_dbs)
                    caption_result = retrieval_handler.execute_search("t2i", query, top_k//2, True, selected_dbs)
                
                end_time = time.time()
                
                st.success(f"混合检索完成！耗时 {end_time - start_time:.3f} 秒")
                log_search_operation("hybrid", query, end_time - start_time, 0)
                
                # 显示各种结果
                col1, col2 = st.columns(2)
                
                with col1:
                    display_text_results(get_result_items(t2t_result, is_multi), "文本检索结果")
                
                with col2:
                    display_image_results(get_result_items(t2i_result, is_multi), "图片检索结果", base_image_root)
                
                # Caption检索结果
                if not is_multi and caption_result.get('total_results', 0) > 0:
                    display_caption_results(caption_result, "Caption检索结果")
            except RetrievalError as e:
                st.error(str(e))
            except Exception as e:
                logger.error(f"混合检索异常: {e}")
                st.error("检索过程中发生错误，请稍后重试")

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
        st.error("检索器初始化失败，请检查系统配置")
        return

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
    elif retrieval_mode == "Caption (标题→图片)":
        render_caption_interface(retrieval_handler, top_k, is_multi, selected_dbs, base_image_root)
    elif retrieval_mode == "混合检索":
        render_hybrid_interface(retrieval_handler, top_k, is_multi, selected_dbs, base_image_root)
    
    # 页脚信息
    st.markdown("---")
    st.markdown("**UltrasoundRAG 检索系统** | 支持多种检索模式和混合查询")

if __name__ == "__main__":
    main()
