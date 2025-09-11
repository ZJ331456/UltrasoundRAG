"""
功能: 处理PDF文档，进行智能分块
分块策略: 按页面和段落分块，支持重叠窗口
文本提取: 使用pdfplumber提取文本内容
智能分割: 优先按段落分割，其次按句子分割，保持语义完整性
"""
import os
import re
import json
from typing import List, Dict, Optional, Tuple
from UltrasoundRAG.config import config
import math

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("警告: pdfplumber未安装，请运行: pip install pdfplumber")

class PDFParser:
    """
    PDF文档解析器
    
    专注于PDF文档的解析、分块和文本提取
    不包含向量数据库相关功能
    """
    
    def __init__(self, dataset_name: str = "thesis_papers"):
        """初始化PDF解析器"""
        self.dataset_name = dataset_name
        
        # 从 config 加载变量
        pdf_cfg = config['indexing']['pdf']
        pdf_parse_cfg = config['indexing']['pdf_parse']
        
        dataset_cfg = pdf_cfg['datasets'][dataset_name]
        self.base_pdf_path = dataset_cfg['base_pdf_path']
        self.max_chunk_size = pdf_parse_cfg['max_chunk_size']
        self.overlap_ratio = pdf_parse_cfg.get('overlap_ratio', 0.1)  # 重叠比例，默认10%
        self.min_chunk_size = pdf_parse_cfg.get('min_chunk_size', 100)  # 最小块大小
        
        print(f"初始化 PDF 解析器，数据集: {dataset_name}")
        print(f"基础路径: {self.base_pdf_path}")
        
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber未安装，无法处理PDF文件")
    
    def get_pdf_files(self) -> List[str]:
        """获取所有PDF文件"""
        pdf_files = []
        for root, dirs, files in os.walk(self.base_pdf_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        从PDF文件中提取文本内容
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含文本内容和元数据的字典
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                page_texts = []
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # 清理页面文本
                        page_text = self._clean_text(page_text)
                        page_texts.append({
                            'page_num': page_num,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                        full_text += page_text + "\n\n"
                
                return {
                    'full_text': full_text.strip(),
                    'page_texts': page_texts,
                    'total_pages': total_pages,
                    'file_name': os.path.basename(pdf_path),
                    'file_path': pdf_path
                }
                
        except Exception as e:
            print(f"提取PDF文本失败 {pdf_path}: {e}")
            return {
                'full_text': "",
                'page_texts': [],
                'total_pages': 0,
                'file_name': os.path.basename(pdf_path),
                'file_path': pdf_path
            }
    
    def _clean_text(self, text: str) -> str:
        """清理提取的文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符和格式标记
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()（）【】""''""''\-\n]', '', text)
        
        # 规范化换行符
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def extract_sections_from_text(self, text: str) -> List[Dict]:
        """
        从文本中提取章节结构
        
        策略：识别标题模式，按章节分块
        """
        lines = text.split('\n')
        sections = []
        current_section = {
            'title': '文档开始',
            'content': [],
            'page_start': 1,
            'page_end': 1
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测标题模式
            if self._is_title(line):
                # 保存当前章节
                if current_section['content']:
                    sections.append({
                        'title': current_section['title'],
                        'content': '\n'.join(current_section['content']),
                        'page_start': current_section['page_start'],
                        'page_end': current_section['page_end']
                    })
                
                # 开始新章节
                current_section = {
                    'title': line,
                    'content': [],
                    'page_start': current_section['page_end'],
                    'page_end': current_section['page_end']
                }
            else:
                current_section['content'].append(line)
        
        # 添加最后一个章节
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'content': '\n'.join(current_section['content']),
                'page_start': current_section['page_start'],
                'page_end': current_section['page_end']
            })
        
        return sections
    
    def _is_title(self, line: str) -> bool:
        """判断是否为标题"""
        # 标题特征：
        # 1. 长度较短（通常不超过50字符）
        # 2. 包含数字编号（如1.1, 2.3等）
        # 3. 全大写或首字母大写
        # 4. 不包含句号结尾
        
        if len(line) > 50 or line.endswith('。'):
            return False
        
        # 检查数字编号模式
        number_patterns = [
            r'^\d+\.\d+',  # 1.1, 2.3
            r'^\d+\.',     # 1., 2.
            r'^第\d+章',   # 第1章
            r'^第\d+节',   # 第1节
            r'^\d+、',     # 1、2、
        ]
        
        for pattern in number_patterns:
            if re.match(pattern, line):
                return True
        
        # 检查大写模式
        if line.isupper() and len(line) > 3:
            return True
        
        # 检查标题关键词
        title_keywords = ['摘要', '引言', '方法', '结果', '讨论', '结论', '参考文献', 'Abstract', 'Introduction', 'Method', 'Result', 'Discussion', 'Conclusion', 'Reference']
        for keyword in title_keywords:
            if keyword in line and len(line) < 30:
                return True
        
        return False
    
    def _split_long_content(self, content: str, title: str, base_index: int) -> List[Dict]:
        """
        智能分割长内容为小块，支持重叠窗口和语义完整性
        
        策略：
        1. 优先按段落(\n\n)分割
        2. 其次按句号(。)分割  
        3. 支持重叠窗口，保持语义连续性
        4. 智能选择分割点，避免截断重要信息
        """
        max_chunk_size = self.max_chunk_size
        overlap_size = int(max_chunk_size * self.overlap_ratio)
        chunks = []
        
        if len(content) <= max_chunk_size:
            chunks.append({
                'title': title,
                'content': content,
                'chunk_index': base_index
            })
            return chunks
        
        # 智能分割策略
        text_units = self._get_text_units(content)
        current_chunk = ""
        chunk_count = 0
        
        i = 0
        while i < len(text_units):
            unit = text_units[i]
            
            # 检查添加当前单元后是否超出限制
            if len(current_chunk + unit) > max_chunk_size and current_chunk:
                # 当前块已满，保存并开始新块
                chunk_content = current_chunk.strip()
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append({
                        'title': title,
                        'content': chunk_content,
                        'chunk_index': base_index + chunk_count
                    })
                    chunk_count += 1
                
                # 计算重叠窗口的起始位置
                overlap_start = self._get_overlap_start(chunks, overlap_size, text_units, i)
                current_chunk = overlap_start + unit
                
            else:
                current_chunk += unit
                
            i += 1
        
        # 处理最后一个块
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append({
                'title': title,
                'content': current_chunk.strip(),
                'chunk_index': base_index + chunk_count
            })
        
        return chunks
    
    def _get_text_units(self, content: str) -> List[str]:
        """
        将文本分解为基本单元，优先级：段落 > 句子 > 子句
        """
        # 1. 首先按段落分割
        paragraphs = content.split('\n\n')
        units = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果段落本身就很短，直接作为一个单元
            if len(paragraph) <= self.max_chunk_size * 0.3:
                units.append(paragraph + '\n\n')
            else:
                # 段落较长，按句子分割
                sentences = self._split_paragraph_to_sentences(paragraph)
                for sentence in sentences:
                    if sentence.strip():
                        units.append(sentence)
                
                # 段落结束标记
                if units:
                    units[-1] += '\n\n'
                    
        return units
    
    def _split_paragraph_to_sentences(self, paragraph: str) -> List[str]:
        """将段落按句子分割，保持语义完整性"""
        # 按句号分割，但保持特殊情况的完整性
        sentences = []
        current_sentence = ""
        
        # 分割句子，考虑特殊情况（如数字、公式等）
        parts = paragraph.split('。')
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            current_sentence += part
            
            # 检查是否应该结束当前句子
            if i < len(parts) - 1:  # 不是最后一部分
                current_sentence += '。'
                
                # 检查是否是特殊情况（如数字、图表标题等）
                if not self._should_break_sentence(current_sentence, parts[i + 1]):
                    continue
                    
                sentences.append(current_sentence)
                current_sentence = ""
            else:
                # 最后一部分
                if current_sentence:
                    sentences.append(current_sentence)
        
        return sentences
    
    def _should_break_sentence(self, current_sentence: str, next_part: str) -> bool:
        """判断是否应该在此处分割句子"""
        # 如果当前句子很短，可能是图表标题或数字，不分割
        if len(current_sentence) < 10:
            return False
            
        # 如果下一部分以数字开头，可能是连续的数据，不分割
        next_part_stripped = next_part.strip()
        if next_part_stripped and (next_part_stripped[0].isdigit() or next_part_stripped.startswith('(')):
            return False
            
        return True
    
    def _get_overlap_start(self, existing_chunks: List[Dict], overlap_size: int, 
                          text_units: List[str], current_index: int) -> str:
        """
        获取重叠窗口的起始内容
        """
        if not existing_chunks or overlap_size <= 0:
            return ""
            
        last_chunk_content = existing_chunks[-1]['content']
        
        # 从最后一个块的末尾取重叠内容
        if len(last_chunk_content) <= overlap_size:
            overlap_content = last_chunk_content
        else:
            # 找到合适的分割点（优先在句子边界）
            sentences = last_chunk_content.split('。')
            overlap_content = ""
            
            # 从后往前取句子，直到接近目标长度
            for i in range(len(sentences) - 1, -1, -1):
                candidate = sentences[i] + ('。' if i < len(sentences) - 1 else '') + overlap_content
                if len(candidate) <= overlap_size:
                    overlap_content = candidate
                else:
                    break
            
            # 如果没有找到合适的句子边界，直接截取
            if not overlap_content:
                overlap_content = last_chunk_content[-overlap_size:]
        
        return overlap_content
    
    def parse_pdfs(self) -> List[Dict]:
        """
        解析所有PDF文件
        
        Returns:
            处理后的文档块列表
        """
        pdf_files = self.get_pdf_files()
        all_chunks = []
        chunk_id = 1
        
        for pdf_file in pdf_files:
            print(f"处理PDF文件: {pdf_file}")
            
            # 提取PDF文本
            pdf_data = self.extract_text_from_pdf(pdf_file)
            if not pdf_data['full_text']:
                print(f"跳过空PDF文件: {pdf_file}")
                continue
            
            # 提取章节结构
            sections = self.extract_sections_from_text(pdf_data['full_text'])
            
            # 处理每个章节
            for section in sections:
                # 将长内容进一步分割
                sub_chunks = self._split_long_content(
                    section['content'], 
                    section['title'], 
                    chunk_id - 1
                )
                
                # 为每个块添加必要信息
                for chunk in sub_chunks:
                    chunk['id'] = chunk_id
                    chunk['pdf_file'] = os.path.relpath(pdf_file, self.base_pdf_path)
                    chunk['document_name'] = os.path.basename(pdf_file).replace('.pdf', '')
                    chunk['page_start'] = section['page_start']
                    chunk['page_end'] = section['page_end']
                    
                    # PDF没有图片信息，设置为空
                    chunk['image_paths'] = []
                    chunk['image_captions'] = []
                    
                    chunk_id += 1
                    all_chunks.append(chunk)
        
        print(f"总共解析了 {len(all_chunks)} 个PDF文档块")
        return all_chunks
    
    def process_document(self, file_path: str) -> List[Dict]:
        """
        处理单个PDF文档并返回处理后的数据
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            处理后的文档块列表
        """
        # 提取PDF文本
        pdf_data = self.extract_text_from_pdf(file_path)
        if not pdf_data['full_text']:
            return []
        
        # 提取章节结构
        sections = self.extract_sections_from_text(pdf_data['full_text'])
        
        # 处理每个章节
        processed_chunks = []
        chunk_id = 1
        
        for section in sections:
            # 将长内容进一步分割
            sub_chunks = self._split_long_content(
                section['content'], 
                section['title'], 
                chunk_id - 1
            )
            
            # 为每个块添加必要信息
            for chunk in sub_chunks:
                chunk['id'] = chunk_id
                chunk['pdf_file'] = os.path.relpath(file_path, self.base_pdf_path)
                chunk['document_name'] = os.path.basename(file_path).replace('.pdf', '')
                chunk['page_start'] = section['page_start']
                chunk['page_end'] = section['page_end']
                
                # PDF没有图片信息，设置为空
                chunk['image_paths'] = []
                chunk['image_captions'] = []
                
                chunk_id += 1
                processed_chunks.append(chunk)
        
        return processed_chunks

# 示例使用
if __name__ == "__main__":
    parser = PDFParser()
    parsed_chunks = parser.parse_pdfs()
    print(f"解析了 {len(parsed_chunks)} 个PDF文档块")
    for chunk in parsed_chunks[:3]:  # 打印前3个
        print(f"ID: {chunk['id']}, 标题: {chunk['title']}, 内容: {chunk['content'][:100]}...")
