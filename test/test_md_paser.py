import sys
import os
from io import StringIO

# 添加项目路径到 Python 路径（使用Linux路径格式）
sys.path.append('/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/UltrasoundRAG/index')

# 使用相对导入
from .markdown_parse import MarkdownParser

class MultiWriter:
    """同时写入多个输出流的类"""
    def __init__(self, *writers):
        self.writers = writers
    
    def write(self, text):
        for writer in self.writers:
            writer.write(text)
    
    def flush(self):
        for writer in self.writers:
            if hasattr(writer, 'flush'):
                writer.flush()

def test_markdown_parser():
    """测试 Markdown 解析器"""
    
    # 初始化解析器
    parser = MarkdownParser()
    
    # 测试文档路径
    test_file = r'/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/markdown/09_肌骨.md'
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"错误：文件不存在 - {test_file}")
        return
    
    print(f"正在处理文档: {test_file}")
    print("=" * 60)
    
    try:
        # 处理文档
        chunks = parser.process_document(test_file)
        
        print(f"文档解析完成！共生成 {len(chunks)} 个文档块")
        print("=" * 60)
        
        # 显示每个块的信息
        for i, chunk in enumerate(chunks):
            print(f"\n【块 {i+1}】")
            print(f"标题: {chunk['title']}")
            print(f"内容长度: {len(chunk['content'])} 字符")
            print(f"块索引: {chunk['chunk_index']}")
            print(f"图片标题: {chunk['origin_image_caption']}")
            print(f"图片URL: {chunk['image_url']}")
            print(f"内容预览: {chunk['content']}")
            print("-" * 40)
        
        # 统计信息
        print("\n统计信息:")
        print(f"总块数: {len(chunks)}")
        print(f"包含图片的块数: {sum(1 for chunk in chunks if chunk['image_url'])}")
        
        # 显示所有唯一标题
        unique_titles = list(set(chunk['title'] for chunk in chunks))
        print(f"唯一标题数: {len(unique_titles)}")
        print("标题列表:")
        for title in unique_titles:
            print(f"  - {title}")
            
    except Exception as e:
        print(f"处理文档时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def test_individual_functions():
    """测试各个函数功能"""
    
    parser = MarkdownParser()
    test_file = r'/media/ps/data-ssd/UltrasoundRAG/UltrasoundRAG/data/book/markdown/09_肌骨.md'
    
    if not os.path.exists(test_file):
        print(f"错误：文件不存在 - {test_file}")
        return
    
    # 读取文件内容
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("测试标题和内容提取...")
    chunks = parser.extract_title_and_content(content)
    print(f"提取到 {len(chunks)} 个内容块")
    
    print("\n测试图片信息提取...")
    image_infos = parser.extract_image_info(content)
    print(f"提取到 {len(image_infos)} 个图片信息:")
    for caption, url in image_infos:
        print(f"  标题: {caption}")
        print(f"  URL: {url}")
        print()

if __name__ == "__main__":
    print("开始测试 Markdown 解析器...")
    print()
    
    # 打开result.txt文件进行写入
    with open('result.txt', 'w', encoding='utf-8') as result_file:
        # 创建多重写入器，同时写入控制台和文件
        multi_writer = MultiWriter(sys.stdout, result_file)
        
        # 保存原始stdout
        original_stdout = sys.stdout
        
        # 重定向stdout
        sys.stdout = multi_writer
        
        try:
            # 完整测试
            test_markdown_parser()
            
            print("\n" + "=" * 60)
            print("单独功能测试:")
            
            # 单独功能测试
            test_individual_functions()
            
            print("\n测试完成！结果已保存到 result.txt")
        finally:
            # 恢复原始stdout
            sys.stdout = original_stdout