import streamlit as st
import os
import sys
from PIL import Image
import json
# streamlit run streamlit_app.py
# 确保项目根目录在 sys.path 中，以便 streamlit 能找到 MedicalRAG 模块
# 当从项目根目录运行 `streamlit run streamlit_app.py` 时，这通常是必需的
if '.' not in sys.path:
    sys.path.append('.')

from MedicalRAG.medicalrag import MedicalRAGSystem

# --- 缓存和模型加载 ---

@st.cache_resource  # 使用 Streamlit 的新缓存装饰器
def load_rag_system():
    """
    加载并初始化MedicalRAGSystem。
    这个函数会被缓存，所以模型只在第一次加载时耗时。
    """
    st.write("正在加载RAG系统模型，请稍候...")
    system = MedicalRAGSystem()
    if not system.initialize_components():
        st.error("系统组件初始化失败，请检查日志。")
        return None
    st.write("系统加载完成！")
    return system

# --- UI 布局和逻辑 ---

def main():
    st.set_page_config(page_title="Medical RAG System", layout="wide")
    st.title("多模态医疗RAG问答系统")
    
    # 加载系统
    rag_system = load_rag_system()
    if not rag_system:
        return

    # --- 输入侧边栏 ---
    with st.sidebar:
        st.header("查询输入")
        query_text = st.text_area("请输入您的问题:", height=150)
        uploaded_image = st.file_uploader("上传一张图片进行检索 (可选)", type=['jpg', 'png', 'jpeg'])

        submit_button = st.button("生成答案")

        # --- 高级管理选项 ---
        with st.expander("高级管理选项"):
            st.warning("重建索引会删除所有现有文本和图片索引，并从头开始构建。此过程可能需要几分钟。")
            if st.button("确认重建索引"):
                with st.spinner("正在重建索引... 请耐心等待，此过程可能需要几分钟。"):
                    success = rag_system.rebuild_index()
                
                if success:
                    st.success("索引重建成功！应用将自动重新加载以应用更改。")
                    # 清理缓存，以便下次能加载新索引
                    st.cache_resource.clear()
                    # 强制重新运行整个应用
                    st.rerun()
                else:
                    st.error("索引重建失败，请检查控制台日志获取详细信息。")


    # --- 主显示区域 ---
    if submit_button and query_text:
        with st.spinner("正在检索并生成答案..."):
            temp_image_path = None
            if uploaded_image:
                # 创建一个临时目录来存放上传的图片
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_image_path = os.path.join(temp_dir, uploaded_image.name)
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                st.sidebar.image(uploaded_image, caption="您上传的图片", use_column_width=True)

            # 调用RAG系统核心方法
            result = rag_system.search_and_generate(
                query=query_text,
                image_path=temp_image_path
            )

            if "error" in result:
                st.error(f"处理时发生错误: {result['error']}")
            else:
                st.subheader("📝 生成的答案")
                st.markdown(result.get("answer", "未能生成答案。"))

                st.subheader("📊 检索到的上下文信息")
                
                # 分类检索结果
                text_results = []
                image_from_text_results = []
                image_to_image_results = []

                retrieval_data = result.get('retrieval', {})
                for key, doc in retrieval_data.items():
                    retrieval_type = doc.get('retrieval_type')
                    if retrieval_type == 'image_from_text':
                        image_from_text_results.append(doc)
                    elif retrieval_type == 'image_to_image':
                        image_to_image_results.append(doc)
                    else: # 'dense', 'sparse', etc.
                        text_results.append(doc)
                
                # --- 显示分类结果 ---
                
                # 1. 文本块检索结果
                with st.expander(f"📚 文本块检索结果 ({len(text_results)}条)", expanded=False):
                    for doc in text_results:
                        st.markdown(f"**分数: `{doc['score']:.3f}` | 类型: `{doc['retrieval_type']}`**")
                        st.text_area(label=f"Doc ID: {doc['doc_id']}", value=doc['content'], height=150, disabled=True)

                # 2. 文本中关联到的图片
                if image_from_text_results:
                    with st.expander(f"🖼️ 从文本中关联到的图片 ({len(image_from_text_results)}张)", expanded=True):
                        cols = st.columns(3) # 每行显示3张图片
                        for i, doc in enumerate(image_from_text_results):
                            col = cols[i % 3]
                            try:
                                # 从 content 中解析图片路径 (修正了换行符)
                                image_path_line = [line for line in doc['content'].split('\n') if "图片路径:" in line]
                                if image_path_line:
                                    full_path = image_path_line[0].replace("图片路径: ", "").strip()
                                    if os.path.exists(full_path):
                                        col.image(full_path, caption=f"分数: {doc['score']:.3f}", use_container_width=True)
                                        col.caption(doc['metadata'].get('full_caption', ''))
                                    else:
                                        col.warning(f"图片未找到:\n{full_path}")
                            except Exception as e:
                                col.error(f"显示图片时出错: {e}")

                # 3. 与上传图片相似的图片
                if image_to_image_results:
                    with st.expander(f"🖼️ 与您上传图片相似的图片 ({len(image_to_image_results)}张)", expanded=True):
                        cols = st.columns(3)
                        for i, doc in enumerate(image_to_image_results):
                            col = cols[i % 3]
                            try:
                                image_path_line = [line for line in doc['content'].split('\n') if "图片路径:" in line]
                                if image_path_line:
                                    full_path = image_path_line[0].replace("图片路径: ", "").strip()
                                    if os.path.exists(full_path):
                                        col.image(full_path, caption=f"相似度: {doc['score']:.3f}", use_container_width=True)
                                        col.caption(doc['metadata'].get('full_caption', ''))
                                    else:
                                        col.warning(f"图片未找到:\n{full_path}")
                            except Exception as e:
                                col.error(f"显示图片时出错: {e}")

    elif submit_button:
        st.warning("请输入问题！")

if __name__ == "__main__":
    main() 