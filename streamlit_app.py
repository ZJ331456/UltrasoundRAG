import streamlit as st
import os
import sys
from PIL import Image
import json
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple

# python -m streamlit run streamlit_app.py
# streamlit run streamlit_app.py

# 确保项目根目录在 sys.path 中，以便 streamlit 能找到 MedicalRAG 模块
if '.' not in sys.path:
    sys.path.append('.')

try:
    from MedicalRAG.medicalrag import MedicalRAGSystem
except ImportError as e:
    st.error(f"无法导入MedicalRAG模块: {e}")
    st.stop()

# --- 缓存和模型加载 ---

@st.cache_resource
def load_rag_system():
    """
    加载并初始化MedicalRAGSystem。
    这个函数会被缓存，所以模型只在第一次加载时耗时。
    """
    try:
        with st.spinner("正在加载RAG系统模型，请稍候..."):
            system = MedicalRAGSystem()
            if not system.initialize_components():
                st.error("系统组件初始化失败，请检查日志。")
                return None
            st.success("系统加载完成！")
            return system
    except Exception as e:
        st.error(f"加载RAG系统时发生错误: {str(e)}")
        return None


def get_database_status(rag_system) -> Dict:
    """获取数据库状态信息"""
    try:
        return rag_system.check_database_status()
    except Exception as e:
        return {"error": f"检查数据库状态时发生错误: {str(e)}"}


def handle_file_upload(uploaded_file) -> Optional[str]:
    """处理文件上传，返回临时文件路径"""
    if not uploaded_file:
        return None
    
    try:
        # 使用临时文件而不是固定目录
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_path
    except Exception as e:
        st.error(f"文件上传失败: {str(e)}")
        return None


def cleanup_temp_files():
    """清理临时文件"""
    temp_dirs = [d for d in os.listdir(tempfile.gettempdir()) 
                 if d.startswith('tmp') and os.path.isdir(os.path.join(tempfile.gettempdir(), d))]
    
    for temp_dir in temp_dirs:
        try:
            full_path = os.path.join(tempfile.gettempdir(), temp_dir)
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
        except Exception:
            pass  # 忽略清理错误

# --- UI 布局和逻辑 ---

def render_sidebar(rag_system) -> Tuple[str, Optional[str], bool]:
    """渲染侧边栏，返回查询文本、图片路径和提交状态"""
    with st.sidebar:
        st.header("查询输入")
        query_text = st.text_area("请输入您的问题:", height=150)
        uploaded_image = st.file_uploader(
            "上传一张图片进行检索 (可选)", 
            type=['jpg', 'png', 'jpeg'],
            help="支持JPG、PNG、JPEG格式的图片"
        )

        submit_button = st.button("生成答案", type="primary")
        
        # 显示上传的图片预览
        image_path = None
        if uploaded_image:
            st.image(uploaded_image, caption="您上传的图片", use_container_width=True)
            image_path = handle_file_upload(uploaded_image)

        # 显示数据库状态
        render_database_status(rag_system)
        
        # 高级管理选项
        render_admin_options(rag_system)
        
        return query_text, image_path, submit_button


def render_database_status(rag_system):
    """渲染数据库状态信息"""
    with st.expander("📊 数据库状态", expanded=False):
        db_info = get_database_status(rag_system)
        
        if "error" in db_info:
            st.error(f"数据库状态检查失败: {db_info['error']}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("集合名称", db_info.get('collection_name', 'N/A'))
            with col2:
                st.metric("文档数量", db_info.get('document_count', 'N/A'))
            
            if db_info.get('document_count', 0) == 0:
                st.warning("⚠️ 数据库为空，建议重建索引")


def render_admin_options(rag_system):
    """渲染管理选项"""
    with st.expander("⚙️ 高级管理选项"):
        st.warning("⚠️ 重建索引会删除所有现有文本和图片索引，并从头开始构建。此过程可能需要几分钟。")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重建索引", help="重新构建所有索引"):
                handle_index_rebuild(rag_system)
        
        with col2:
            if st.button("🧹 清理缓存", help="清理临时文件和缓存"):
                cleanup_temp_files()
                st.cache_resource.clear()
                st.success("缓存清理完成！")


def handle_index_rebuild(rag_system):
    """处理索引重建"""
    try:
        with st.spinner("正在重建索引... 请耐心等待，此过程可能需要几分钟。"):
            success = rag_system.rebuild_index()
        
        if success:
            st.success("✅ 索引重建成功！应用将自动重新加载。")
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("❌ 索引重建失败，请检查控制台日志获取详细信息。")
    except Exception as e:
        st.error(f"索引重建过程中发生错误: {str(e)}")


def main():
    st.set_page_config(
        page_title="Medical RAG System", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "医疗RAG问答系统 - 基于超声图像的智能检索与问答"
        }
    )
    
    st.title("🏥 Ultrasound RAG Q&A Demo")
    st.markdown("---")
    
    # 加载系统
    rag_system = load_rag_system()
    if not rag_system:
        st.error("❌ 系统加载失败，请检查配置和日志。")
        return

    # 渲染侧边栏
    query_text, image_path, submit_button = render_sidebar(rag_system)
    
    # 处理查询
    if submit_button:
        if not query_text.strip():
            st.warning("⚠️ 请输入问题！")
        else:
            process_query(rag_system, query_text, image_path)


def process_query(rag_system, query_text: str, image_path: Optional[str]):
    """处理用户查询"""
    try:
        with st.spinner("🔍 正在检索并生成答案..."):
            # 调用RAG系统核心方法
            result = rag_system.search_and_generate(
                query=query_text,
                image_path=image_path
            )

            if "error" in result:
                st.error(f"❌ 处理时发生错误: {result['error']}")
                return
            
            # 显示结果
            display_results(result)
            
    except Exception as e:
        st.error(f"❌ 查询处理过程中发生错误: {str(e)}")
    finally:
        # 清理临时文件
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass


def display_results(result: Dict):
    """显示查询结果"""
    # 显示答案
    st.subheader("📝 生成的答案")
    answer = result.get("answer", "未能生成答案。")
    st.markdown(answer)
    
    # 添加复制按钮
    if st.button("📋 复制答案"):
        st.write("答案已复制到剪贴板！")
    
    st.markdown("---")
    st.subheader("📊 检索到的上下文信息")
    
    # 分类检索结果
    categorized_results = categorize_retrieval_results(result.get('retrieval', {}))
    
    # 显示统计信息
    display_retrieval_stats(categorized_results)
    
    # 显示各类结果
    display_text_results(categorized_results['text'])
    display_image_results(categorized_results)


def categorize_retrieval_results(retrieval_data: Dict) -> Dict[str, List]:
    """分类检索结果"""
    categorized = {
        'text': [],
        'image_from_text': [],
        'image_to_image': [],
        'clip_text_to_image': []
    }
    
    for key, doc in retrieval_data.items():
        retrieval_type = doc.get('retrieval_type', 'text')
        if retrieval_type in categorized:
            categorized[retrieval_type].append(doc)
        else:
            categorized['text'].append(doc)
    
    return categorized


def display_retrieval_stats(categorized_results: Dict[str, List]):
    """显示检索统计信息"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 文本块", len(categorized_results['text']))
    with col2:
        st.metric("🖼️ 关联图片", len(categorized_results['image_from_text']))
    with col3:
        st.metric("🔍 CLIP检索", len(categorized_results['clip_text_to_image']))
    with col4:
        st.metric("📷 相似图片", len(categorized_results['image_to_image']))


def display_text_results(text_results: List[Dict]):
    """显示文本检索结果"""
    if not text_results:
        return
        
    with st.expander(f"📚 文本块检索结果 ({len(text_results)}条)", expanded=False):
        for i, doc in enumerate(text_results):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**文档 {i+1}**")
                with col2:
                    st.metric("相关度", f"{doc['score']:.3f}")
                
                st.text_area(
                    label=f"内容 (ID: {doc.get('doc_id', 'N/A')})",
                    value=doc['content'],
                    height=120,
                    disabled=True,
                    key=f"text_{i}"
                )
                st.caption(f"检索类型: {doc.get('retrieval_type', 'N/A')}")
                st.markdown("---")


def extract_image_path(content: str) -> Optional[str]:
    """从内容中提取图片路径"""
    try:
        image_path_lines = [line for line in content.split('\n') if "图片路径:" in line]
        if image_path_lines:
            return image_path_lines[0].replace("图片路径: ", "").strip()
    except Exception:
        pass
    return None


def display_image_grid(images_data: List[Dict], title: str, score_label: str):
    """显示图片网格"""
    if not images_data:
        return
        
    with st.expander(f"{title} ({len(images_data)}张)", expanded=True):
        cols = st.columns(3)
        for i, doc in enumerate(images_data):
            col = cols[i % 3]
            
            try:
                image_path = extract_image_path(doc['content'])
                if image_path and os.path.exists(image_path):
                    col.image(
                        image_path, 
                        caption=f"{score_label}: {doc['score']:.3f}", 
                        use_container_width=True
                    )
                    
                    # 显示图片元数据
                    caption = doc.get('metadata', {}).get('full_caption', '')
                    if caption:
                        col.caption(caption[:100] + "..." if len(caption) > 100 else caption)
                else:
                    col.warning(f"❌ 图片未找到: {image_path}")
                    
            except Exception as e:
                col.error(f"❌ 显示图片时出错: {str(e)}")


def display_image_results(categorized_results: Dict[str, List]):
    """显示图像检索结果"""
    display_image_grid(
        categorized_results['image_from_text'],
        "🖼️ 从文本中关联到的图片",
        "关联度"
    )
    
    display_image_grid(
        categorized_results['clip_text_to_image'],
        "🔍 CLIP文本检索到的图片",
        "相关度"
    )
    
    display_image_grid(
        categorized_results['image_to_image'],
        "📷 与您上传图片相似的图片",
        "相似度"
    )

if __name__ == "__main__":
    main()