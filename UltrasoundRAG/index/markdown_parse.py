import os
import re
from typing import List, Dict, Optional, Tuple
from UltrasoundRAG.config import config
import math

class MarkdownParser:
    """
    Markdown文档解析器
    
    专注于Markdown文档的解析、分块和图片信息提取
    不包含向量数据库相关功能
    """
    
    def __init__(self, dataset_name: str = "ultrasound_book"):
        """初始化Markdown解析器"""
        self.dataset_name = dataset_name
        
        # 从 config 加载变量
        markdown_cfg = config['indexing']['markdown']
        markdown_parse_cfg = config['indexing']['markdown_parse']
        
        dataset_cfg = markdown_cfg['datasets'][dataset_name]
        self.base_md_path = dataset_cfg['base_md_path']
        self.max_chunk_size = markdown_parse_cfg['max_chunk_size']
        self.overlap_ratio = markdown_parse_cfg.get('overlap_ratio', 0.1)  # 重叠比例，默认10%
        self.min_chunk_size = markdown_parse_cfg.get('min_chunk_size', 100)  # 最小块大小
        
        print(f"初始化 Markdown 解析器，数据集: {dataset_name}")
        print(f"基础路径: {self.base_md_path}")
    
    def get_markdown_files(self) -> List[str]:
        """获取所有Markdown文件"""
        md_files = []
        for root, dirs, files in os.walk(self.base_md_path):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        return md_files
    
    def parse_markdowns(self) -> List[Dict]:
        """
        解析所有Markdown文件
        
        Returns:
            处理后的文档块列表
        """
        md_files = self.get_markdown_files()
        all_chunks = []
        chunk_id = 1
        
        for md_file in md_files:
            print(f"处理文件: {md_file}")
            chunks = self.process_document(md_file)
            
            # 为每个块添加必要信息
            for chunk in chunks:
                chunk['id'] = chunk_id
                chunk['md_file'] = os.path.relpath(md_file, self.base_md_path)
                chunk['document_name'] = os.path.basename(md_file).replace('.md', '')
                
                # 处理图片信息
                if chunk.get('origin_image_caption'):
                    chunk['image_links'] = [chunk.get('image_url', '')]
                    chunk['image_captions'] = [chunk.get('origin_image_caption', '')]
                else:
                    chunk['image_links'] = []
                    chunk['image_captions'] = []
                
                chunk_id += 1
                all_chunks.append(chunk)
        
        print(f"总共解析了 {len(all_chunks)} 个文档块")
        return all_chunks
    
    def extract_title_and_content(self, text: str) -> List[Dict]:
        """
        提取文档的标题和内容块
        
        策略：按标题分块，碰到#/##/###/####等就分块，每个块分配对应的一级标题
        
        Args:
            text: 原始文档文本
            
        Returns:
            包含标题和内容的字典列表
        """
        lines = text.split('\n')
        chunks = []
        current_title = "未知标题"
        current_content = []
        chunk_index = 0
        
        # 第一遍：收集所有一级标题和位置
        title_positions = []
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#') and not line.startswith('##'):
                title_text = line.lstrip('#').strip()
                title_positions.append((line_idx, title_text))
        
        # 第二遍：按标题分块
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 检查是否为任何级别的标题（#/##/###/####等）
            if line.startswith('#'):
                # 保存之前的内容块
                if current_content:
                    content_text = '\n'.join(current_content)
                    
                    # 为这个块分配对应的一级标题
                    block_title = self._get_title_for_position(line_idx - 1, title_positions)
                    
                    # 将长内容进一步分割
                    sub_chunks = self._split_long_content(content_text, block_title, chunk_index)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                
                # 更新当前标题
                current_title = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 添加最后一个内容块
        if current_content:
            content_text = '\n'.join(current_content)
            block_title = self._get_title_for_position(len(lines) - 1, title_positions)
            sub_chunks = self._split_long_content(content_text, block_title, chunk_index)
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _get_title_for_position(self, position: int, title_positions: List[Tuple[int, str]]) -> str:
        """为指定位置获取对应的一级标题"""
        if not title_positions:
            return "未知标题"
        
        current_title = "未知标题"
        for title_pos, title_text in title_positions:
            if title_pos <= position:
                current_title = title_text
            else:
                break
        
        return current_title
    
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
    
    def extract_image_info(self, text: str) -> List[Tuple[str, str]]:
        """提取图片链接和标题信息"""
        url_pattern = r'!\[\]\((https?://[^\s)]+)\)'
        url_matches = re.findall(url_pattern, text)
        
        if not url_matches:
            return []
        
        image_infos = []
        lines = text.split('\n')
        
        for i, image_url in enumerate(url_matches):
            image_url = image_url.strip()
            
            for line_idx, line in enumerate(lines):
                if image_url in line:
                    if line_idx + 1 < len(lines):
                        next_line = lines[line_idx + 1].strip()
                        
                        caption = None
                        
                        # 中文标题模式：图X-X 标题描述
                        chinese_match = re.match(r'^(图\d+[-~]\d+[^。\n\r！？；，]*)', next_line)
                        if chinese_match:
                            caption = chinese_match.group(1).strip()
                        
                        # 英文标题模式：Figure X.X 标题描述
                        if not caption:
                            english_match = re.match(r'^(Figure\s+\d+\.\d+[^。\n\r！？；，]*)', next_line)
                            if english_match:
                                caption = english_match.group(1).strip()
                        
                        if caption:
                            caption = re.sub(r'\s+', ' ', caption)
                            if caption.endswith('。'):
                                caption = caption[:-1]
                            caption = re.sub(r'[，。！？；、]+$', '', caption)
                            
                            if caption and len(caption) > 3:
                                image_infos.append((caption, image_url))
                                break
                    
                    if not any(caption == caption_tuple[0] for caption_tuple in image_infos if caption_tuple[1] == image_url):
                        image_infos.append(("", image_url))
                    break
        
        return image_infos
    
    def process_document(self, file_path: str) -> List[Dict]:
        """
        处理文档并返回处理后的数据
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            处理后的文档块列表
        """
        # 读取文档
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取标题和内容块
        chunks = self.extract_title_and_content(content)
        
        # 处理每个块
        processed_chunks = []
        for chunk in chunks:
            # 检查是否包含图片
            image_infos = self.extract_image_info(chunk['content'])
            
            if image_infos:
                # 合并所有图片标题和URL
                all_captions = []
                all_urls = []
                for image_caption, image_url in image_infos:
                    if image_caption:
                        all_captions.append(image_caption)
                    if image_url:
                        all_urls.append(image_url)
                
                combined_caption = "；".join(all_captions) if all_captions else ""
                combined_url = "；".join(all_urls) if all_urls else ""
                
                processed_chunks.append({
                    'title': chunk['title'],
                    'content': chunk['content'],
                    'origin_image_caption': combined_caption,
                    'image_url': combined_url,
                    'chunk_index': chunk['chunk_index']
                })
            else:
                processed_chunks.append({
                    'title': chunk['title'],
                    'content': chunk['content'],
                    'origin_image_caption': "",
                    'image_url': "",
                    'chunk_index': chunk['chunk_index']
                })
        
        return processed_chunks