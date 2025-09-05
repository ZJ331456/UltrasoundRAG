"""
UltrasoundRAG 简单前端界面
支持文本查询、图片查询和混合查询的检索测试界面

使用方法：
python frontend.py
然后在浏览器中访问 http://localhost:8501
"""

import streamlit as st
import os
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import io

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
try:
    from UltrasoundRAG.retrival.modular_retrievers import (
        create_t2t_retriever, create_t2i_retriever,
        create_i2t_retriever, create_i2i_retriever
    )
    from UltrasoundRAG.retrival.caption_to_image_retriever import create_caption_retriever
    from UltrasoundRAG.retrival.multi_database_manager import create_multi_database_manager
    from UltrasoundRAG.utils.logger import setup_logger
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.stop()

# 初始化日志
logger = setup_logger("Frontend")

# 缓存检索器实例
@st.cache_resource
def get_retrievers():
    """获取所有检索器实例"""
    try:
        retrievers = {
            't2t': create_t2t_retriever("default", top_k=10),
            't2i': create_t2i_retriever("default", top_k=10),
            'i2t': create_i2t_retriever("default", top_k=10),
            'i2i': create_i2i_retriever("default", top_k=10),
            'caption': create_caption_retriever("default", "hybrid_match")
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
            
            with col2:
                if 'relative_path' in getattr(result, 'metadata', {}):
                    st.write("**文件路径:**")
                    st.code(getattr(result, 'metadata', {}).get('relative_path', ''))

def display_image_results(results: List[Dict], title: str = "图片结果"):
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
                if 'relative_path' in metadata:
                    st.write(f"路径: {metadata['relative_path']}")
                
                # 显示caption
                content = getattr(result, 'content', '')
                if content:
                    st.write(f"描述: {content[:100]}{'...' if len(content) > 100 else ''}")
                
                # 显示其他元数据
                for key, value in metadata.items():
                    if key not in ['relative_path', 'file_size']:
                        st.write(f"- {key}: {value}")

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

def main():
    """主界面"""
    st.title("🔍 UltrasoundRAG 检索测试界面")
    st.markdown("---")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 检索配置")
        
        # 检索模式选择
        retrieval_mode = st.selectbox(
            "选择检索模式",
            ["T2T (文本→文本)", "T2I (文本→图片)", "I2T (图片→文本)", 
             "I2I (图片→图片)", "Caption (标题→图片)", "混合检索"],
            help="选择要测试的检索模式"
        )
        
        # Top-K设置
        top_k = st.slider("返回结果数量", 1, 20, 5, help="设置返回的最大结果数量")
        
        # 数据库选择
        db_name = st.selectbox("数据库", ["default"], help="选择要查询的数据库")
        
        st.markdown("---")
        st.header("ℹ️ 系统信息")
        
        # 显示检索器状态
        retrievers = get_retrievers()
        if retrievers:
            st.success("✅ 检索器已初始化")
        else:
            st.error("❌ 检索器初始化失败")
            return
    
    # 主界面
    if not retrievers:
        st.error("检索器初始化失败，请检查系统配置")
        return
    
    # 根据检索模式显示不同的输入界面
    if retrieval_mode == "T2T (文本→文本)":
        st.header("📝 文本到文本检索")
        query = st.text_area("输入查询文本", placeholder="例如：心脏超声检查方法", height=100)
        
        if st.button("🔍 开始检索", type="primary"):
            if query.strip():
                with st.spinner("正在检索..."):
                    start_time = time.time()
                    result = retrievers['t2t'].search(query, top_k=top_k)
                    end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                display_text_results(result.get('results', []), "文本检索结果")
            else:
                st.warning("请输入查询文本")
    
    elif retrieval_mode == "T2I (文本→图片)":
        st.header("🖼️ 文本到图片检索")
        query = st.text_area("输入查询文本", placeholder="例如：心脏四腔心切面图", height=100)
        
        if st.button("🔍 开始检索", type="primary"):
            if query.strip():
                with st.spinner("正在检索..."):
                    start_time = time.time()
                    result = retrievers['t2i'].search(query, top_k=top_k)
                    end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                display_image_results(result.get('results', []), "图片检索结果")
            else:
                st.warning("请输入查询文本")
    
    elif retrieval_mode == "I2T (图片→文本)":
        st.header("📷 图片到文本检索")
        
        # 图片上传
        uploaded_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'], help="支持PNG、JPG、JPEG格式")
        
        if uploaded_file is not None:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            
            # 保存临时文件
            temp_path = f"/tmp/uploaded_image_{int(time.time())}.jpg"
            image.save(temp_path)
            
            if st.button("🔍 开始检索", type="primary"):
                with st.spinner("正在检索..."):
                    start_time = time.time()
                    result = retrievers['i2t'].search(temp_path, top_k=top_k)
                    end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                display_text_results(result.get('results', []), "相关文本结果")
                
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
        else:
            st.info("请上传一张图片进行检索")
    
    elif retrieval_mode == "I2I (图片→图片)":
        st.header("🖼️ 图片到图片检索")
        
        # 图片上传
        uploaded_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'], help="支持PNG、JPG、JPEG格式")
        
        if uploaded_file is not None:
            # 显示上传的图片
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            
            # 保存临时文件
            temp_path = f"/tmp/uploaded_image_{int(time.time())}.jpg"
            image.save(temp_path)
            
            if st.button("🔍 开始检索", type="primary"):
                with st.spinner("正在检索..."):
                    start_time = time.time()
                    result = retrievers['i2i'].search(temp_path, top_k=top_k)
                    end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                display_image_results(result.get('results', []), "相似图片结果")
                
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
        else:
            st.info("请上传一张图片进行检索")
    
    elif retrieval_mode == "Caption (标题→图片)":
        st.header("🏷️ 标题到图片检索")
        
        # 单个caption检索
        st.subheader("单个Caption检索")
        caption = st.text_input("输入图片标题", placeholder="例如：图2-3 心脏超声横切面")
        
        if st.button("🔍 检索单个Caption", type="primary"):
            if caption.strip():
                with st.spinner("正在检索..."):
                    start_time = time.time()
                    result = retrievers['caption'].search_single_caption(caption, top_k=top_k)
                    end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                display_caption_results(result, "Caption检索结果")
            else:
                st.warning("请输入图片标题")
        
        st.markdown("---")
        
        # 从文本块提取caption检索
        st.subheader("从文本块提取Caption检索")
        text_chunk = st.text_area("输入包含图片引用的文本", 
                                placeholder="例如：图2-1显示心脏四腔心切面，图2-2为心脏短轴切面", 
                                height=100)
        
        if st.button("🔍 提取并检索", type="secondary"):
            if text_chunk.strip():
                with st.spinner("正在提取Caption并检索..."):
                    start_time = time.time()
                    result = retrievers['caption'].search_from_text_chunk(text_chunk, top_k=top_k)
                    end_time = time.time()
                
                st.success(f"检索完成！耗时 {end_time - start_time:.3f} 秒")
                
                # 显示提取的captions
                extracted_captions = result.get('extracted_captions', [])
                if extracted_captions:
                    st.write(f"**提取到的Caption:** {', '.join(extracted_captions)}")
                
                # 显示检索结果
                display_image_results(result.get('results', []), "匹配的图片结果")
            else:
                st.warning("请输入包含图片引用的文本")
    
    elif retrieval_mode == "混合检索":
        st.header("🔄 混合检索")
        st.info("混合检索将同时执行多种检索模式并融合结果")
        
        query = st.text_area("输入查询文本", placeholder="例如：心脏超声检查", height=100)
        
        if st.button("🔍 开始混合检索", type="primary"):
            if query.strip():
                with st.spinner("正在执行混合检索..."):
                    start_time = time.time()
                    
                    # 执行多种检索
                    t2t_result = retrievers['t2t'].search(query, top_k=top_k//2)
                    t2i_result = retrievers['t2i'].search(query, top_k=top_k//2)
                    caption_result = retrievers['caption'].search_single_caption(query, top_k=top_k//2)
                    
                    end_time = time.time()
                
                st.success(f"混合检索完成！耗时 {end_time - start_time:.3f} 秒")
                
                # 显示各种结果
                col1, col2 = st.columns(2)
                
                with col1:
                    display_text_results(t2t_result.get('results', []), "文本检索结果")
                
                with col2:
                    display_image_results(t2i_result.get('results', []), "图片检索结果")
                
                # Caption检索结果
                if caption_result.get('total_results', 0) > 0:
                    display_caption_results(caption_result, "Caption检索结果")
            else:
                st.warning("请输入查询文本")
    
    # 页脚信息
    st.markdown("---")
    st.markdown("**UltrasoundRAG 检索系统** | 支持多种检索模式和混合查询")

if __name__ == "__main__":
    main()
