"""
PDF解析器：处理PDF文档，提取文本和图片，进行智能分块
功能：解析PDF文档，提取文本内容，识别图片中的文字，生成结构化数据
使用模型：paddleocr进行OCR文本识别
数据源：PDF文件目录
"""
import os
import json
import re
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io
import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm
import math
from ultrasoundrag.config import config


class PDFParser:
    """
    PDF文档解析器
    
    专注于PDF文档的解析、文本提取、图片识别和智能分块
    使用paddleocr进行OCR文本识别
    """
    
    def __init__(self, dataset_name: str = "ultrasound_book"):
        """初始化PDF解析器"""
        self.dataset_name = dataset_name
        
        # 从 config 加载变量
        pdf_cfg = config['indexing']['pdf']
        pdf_parse_cfg = config['indexing']['pdf_parse']
        
        dataset_cfg = pdf_cfg['datasets'][dataset_name]
        self.base_pdf_path = dataset_cfg['base_path']
        self.max_chunk_size = pdf_parse_cfg['max_chunk_size']
        self.overlap_ratio = pdf_parse_cfg.get('overlap_ratio', 0.1)
        self.min_chunk_size = pdf_parse_cfg.get('min_chunk_size', 100)
        
        # 获取PaddleOCR配置
        paddleocr_cfg = pdf_parse_cfg.get('paddleocr', {})
        self.model_dir = paddleocr_cfg.get('model_dir', 'models/paddleocr')
        self.use_angle_cls = paddleocr_cfg.get('use_angle_cls', True)
        self.lang = paddleocr_cfg.get('lang', 'ch')
        
        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初始化PaddleOCR，指定模型下载目录
        print(f"初始化PaddleOCR，模型目录: {self.model_dir}")
        
        # 使用环境变量强制指定模型下载位置
        os.environ['PADDLEOCR_HOME'] = os.path.abspath(self.model_dir)
        
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=self.use_angle_cls, 
            lang=self.lang
        )
        
        print(f"PaddleOCR初始化完成，使用语言: {self.lang}")
        
        print(f"初始化 PDF 解析器，数据集: {dataset_name}")
        print(f"基础路径: {self.base_pdf_path}")
    
    def get_pdf_files(self) -> List[str]:
        """获取所有PDF文件"""
        pdf_files = []
        for root, dirs, files in os.walk(self.base_pdf_path):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从PDF提取文本内容
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            每页的文本内容列表
        """
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # 清理文本
            cleaned_text = self._clean_text(text)
            
            pages_text.append({
                'page_num': page_num + 1,
                'text': cleaned_text,
                'page_size': page.rect
            })
        
        doc.close()
        return pages_text
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从PDF提取图片
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            图片信息列表
        """
        doc = fitz.open(pdf_path)
        images = []
        image_index = 1
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # 获取图片数据
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # 转换为PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        # 保存图片到临时文件
                        temp_path = f"/tmp/pdf_image_{page_num}_{img_index}.png"
                        img_pil.save(temp_path)
                        
                        images.append({
                            'image_path': temp_path,
                            'page_num': page_num + 1,
                            'image_index': image_index,
                            'bbox': img[1:5] if len(img) > 4 else None
                        })
                        
                        image_index += 1
                    
                    pix = None
                    
                except Exception as e:
                    print(f"提取图片失败 {pdf_path} 第{page_num+1}页: {e}")
                    continue
        
        doc.close()
        return images
    
    def ocr_image_text(self, image_path: str) -> str:
        """
        使用PaddleOCR识别图片中的文字
        
        Args:
            image_path: 图片路径
            
        Returns:
            识别出的文本
        """
        try:
            result = self.ocr.ocr(image_path)
            
            if not result or not result[0]:
                return ""
            
            # 提取所有文本 - 处理不同的数据结构
            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
                    try:
                        # 尝试不同的数据结构
                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                            text = line[1][0]  # 文本内容
                            confidence = line[1][1]  # 置信度
                        elif isinstance(line[1], str):
                            text = line[1]
                            confidence = 1.0  # 默认高置信度
                        else:
                            continue
                        
                        # 过滤低置信度文本和低质量文本
                        if confidence > 0.5 and len(text.strip()) > 2:
                            # 过滤掉重复字符和无意义文本
                            if not self._is_low_quality_text(text):
                                texts.append(text)
                    except (IndexError, TypeError) as e:
                        print(f"OCR结果解析错误 {image_path}: {e}, line: {line}")
                        continue
            
            return '\n'.join(texts)
            
        except Exception as e:
            print(f"OCR识别失败 {image_path}: {e}")
            return ""
    
    def _is_low_quality_text(self, text: str) -> bool:
        """判断是否为低质量文本"""
        text = text.strip()
        
        # 检查是否为空或太短
        if len(text) < 3:
            return True
        
        # 检查是否大部分是重复字符
        if len(set(text)) <= 2:
            return True
        
        # 检查是否包含太多无意义字符
        meaningful_chars = len([c for c in text if c.isalnum() or c in '，。！？；：""''（）【】《》、'])
        if meaningful_chars < len(text) * 0.5:
            return True
        
        # 检查是否为常见的OCR错误模式
        low_quality_patterns = [
            r'^[a-z\s\n]+$',  # 只有小写字母和空白
            r'^[n\na\no\nt\no\ne]+$',  # 特定的错误模式
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】《》、]', '', text)
        return text.strip()
    
    def extract_title_and_content(self, text: str) -> List[Dict]:
        """
        提取文档的标题和内容块
        
        策略：按段落和标题分块，保持语义完整性
        """
        lines = text.split('\n')
        chunks = []
        current_title = "未知标题"
        current_content = []
        chunk_index = 0
        
        # 识别标题模式
        title_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节篇]',  # 第X章
            r'^\d+\.?\s+',  # 数字标题
            r'^[一二三四五六七八九十]+[、．]',  # 中文数字标题
            r'^[A-Z]\.?\s+',  # 英文标题
        ]
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 检查是否为标题
            is_title = False
            for pattern in title_patterns:
                if re.match(pattern, line):
                    is_title = True
                    break
            
            if is_title:
                # 保存之前的内容块
                if current_content:
                    content_text = '\n'.join(current_content)
                    sub_chunks = self._split_long_content(content_text, current_title, chunk_index)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                
                # 更新当前标题
                current_title = line
                current_content = []
            else:
                current_content.append(line)
        
        # 添加最后一个内容块
        if current_content:
            content_text = '\n'.join(current_content)
            sub_chunks = self._split_long_content(content_text, current_title, chunk_index)
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_long_content(self, content: str, title: str, base_index: int) -> List[Dict]:
        """
        智能分割长内容为小块，支持重叠窗口
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
        
        # 按段落分割
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 检查添加当前段落后是否超出限制
            if len(current_chunk + paragraph) > max_chunk_size and current_chunk:
                # 当前块已满，保存并开始新块
                chunk_content = current_chunk.strip()
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append({
                        'title': title,
                        'content': chunk_content,
                        'chunk_index': base_index + chunk_count
                    })
                    chunk_count += 1
                
                # 计算重叠窗口
                overlap_start = self._get_overlap_start(chunks, overlap_size)
                current_chunk = overlap_start + paragraph + '\n\n'
            else:
                current_chunk += paragraph + '\n\n'
        
        # 处理最后一个块
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append({
                'title': title,
                'content': current_chunk.strip(),
                'chunk_index': base_index + chunk_count
            })
        
        return chunks
    
    def _get_overlap_start(self, existing_chunks: List[Dict], overlap_size: int) -> str:
        """获取重叠窗口的起始内容"""
        if not existing_chunks or overlap_size <= 0:
            return ""
        
        last_chunk_content = existing_chunks[-1]['content']
        
        if len(last_chunk_content) <= overlap_size:
            return last_chunk_content
        else:
            # 从末尾取重叠内容
            return last_chunk_content[-overlap_size:]
    
    def process_document(self, file_path: str) -> List[Dict]:
        """
        处理PDF文档并返回处理后的数据
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            处理后的文档块列表
        """
        print(f"处理PDF文件: {file_path}")
        
        # 提取文本内容
        pages_text = self.extract_text_from_pdf(file_path)
        
        # 提取图片
        images = self.extract_images_from_pdf(file_path)
        
        # 为每张图片进行OCR识别
        image_ocr_results = {}
        for img_info in images:
            ocr_text = self.ocr_image_text(img_info['image_path'])
            if ocr_text:
                image_ocr_results[img_info['image_index']] = {
                    'text': ocr_text,
                    'page_num': img_info['page_num']
                }
        
        # 合并所有页面的文本
        all_text = ""
        for page in pages_text:
            all_text += page['text'] + '\n\n'
        
        # 提取标题和内容块
        chunks = self.extract_title_and_content(all_text)
        
        # 处理每个块，关联图片信息
        processed_chunks = []
        for chunk in chunks:
            # 查找相关的图片OCR结果
            related_images = []
            related_ocr_texts = []
            
            for img_index, ocr_result in image_ocr_results.items():
                # 简单的关键词匹配来确定图片是否与当前块相关
                if self._is_image_related_to_chunk(chunk['content'], ocr_result['text']):
                    related_images.append(f"image_{img_index}")
                    related_ocr_texts.append(ocr_result['text'])
            
            processed_chunks.append({
                'title': chunk['title'],
                'content': chunk['content'],
                'image_paths': related_images,
                'image_captions': related_ocr_texts,
                'chunk_index': chunk['chunk_index']
            })
        
        return processed_chunks
    
    def _is_image_related_to_chunk(self, chunk_content: str, ocr_text: str) -> bool:
        """判断图片是否与文本块相关"""
        if not ocr_text or not chunk_content:
            return False
        
        # 简单的关键词匹配
        chunk_words = set(re.findall(r'\w+', chunk_content.lower()))
        ocr_words = set(re.findall(r'\w+', ocr_text.lower()))
        
        # 如果有共同词汇，认为相关
        common_words = chunk_words.intersection(ocr_words)
        return len(common_words) > 0
    
    def parse_single_file(self, file_path: str, original_filename: str = None) -> List[Dict]:
        """
        解析单个PDF文件
        
        Args:
            file_path: PDF文件完整路径
            original_filename: 原始文件名（用于file字段）
            
        Returns:
            处理后的文档块列表
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return []
        
        print(f"处理单个PDF文件: {file_path}")
        chunks = self.process_document(file_path)
        all_chunks = []
        chunk_id = 1
        
        for chunk in chunks:
            chunk['id'] = chunk_id
            # 使用原始文件名，如果没有则使用文件路径
            if original_filename:
                chunk['file'] = original_filename
            else:
                chunk['file'] = os.path.relpath(file_path, self.base_pdf_path)
            chunk['document_name'] = os.path.basename(file_path).replace('.pdf', '')
            all_chunks.append(chunk)
            chunk_id += 1
        
        print(f"总共解析了 {len(all_chunks)} 个文档块")
        return all_chunks
    
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
            try:
                chunks = self.process_document(pdf_file)
                
                # 为每个块添加必要信息
                for chunk in chunks:
                    chunk['id'] = chunk_id
                    chunk['pdf_file'] = os.path.relpath(pdf_file, self.base_pdf_path)
                    chunk['document_name'] = os.path.basename(pdf_file).replace('.pdf', '')
                    chunk_id += 1
                    all_chunks.append(chunk)
                    
            except Exception as e:
                print(f"处理PDF文件失败 {pdf_file}: {e}")
                continue
        
        print(f"总共解析了 {len(all_chunks)} 个文档块")
        return all_chunks


# 示例使用
if __name__ == "__main__":
    parser = PDFParser()
    parsed_chunks = parser.parse_pdfs()
    print(f"解析了 {len(parsed_chunks)} 个文档块")
    for chunk in parsed_chunks[:3]:  # 打印前3个
        print(f"ID: {chunk['id']}, 标题: {chunk['title']}, 内容: {chunk['content'][:50]}...")
