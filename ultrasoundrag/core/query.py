#!/usr/bin/env python3
"""
查询处理模块 - 专注于查询预处理和文本分析
基于原始 FulltextQueryer 的查询处理功能适配到 UltrasoundRAG 系统

主要功能：
1. 查询文本预处理和分词
2. 中英文混合处理
3. 同义词扩展和关键词提取
4. 查询特征分析和权重计算
5. 为检索系统提供标准化的查询处理

注意：此模块专注于查询处理，不包含具体的检索逻辑
"""

import logging
import json
import re
import os
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union

from ultrasoundrag.utils.logger import setup_logger
from ultrasoundrag.utils.llm_utils import llm_provider
from ultrasoundrag.utils.prompt import query_rewrite_prompt


class QueryTextProcessor:
    """查询文本处理器 - 专用于查询预处理的文本处理功能"""
    
    @staticmethod
    def sub_special_char(line: str) -> str:
        """转义特殊字符"""
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()
    
    @staticmethod
    def is_chinese(line: str) -> bool:
        """判断是否为中文文本"""
        arr = re.split(r"[ \t]+", line)
        if len(arr) <= 3:
            return True
        e = 0
        for t in arr:
            if not re.match(r"[a-zA-Z]+$", t):
                e += 1
        return e * 1.0 / len(arr) >= 0.7
    
    @staticmethod
    def remove_www(txt: str) -> str:
        """移除常见停用词"""
        patterns = [
            (
                r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ")
        ]
        otxt = txt
        for r, p in patterns:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        if not txt:
            txt = otxt
        return txt
    
    @staticmethod
    def add_space_between_eng_zh(txt: str) -> str:
        """在中英文之间添加空格"""
        # (ENG/ENG+NUM) + ZH
        txt = re.sub(r'([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # ENG + ZH
        txt = re.sub(r'([A-Za-z])([\u4e00-\u9fa5]+)', r'\1 \2', txt)
        # ZH + (ENG/ENG+NUM)
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)', r'\1 \2', txt)
        txt = re.sub(r'([\u4e00-\u9fa5]+)([A-Za-z])', r'\1 \2', txt)
        return txt
    
    @staticmethod
    def traditional_to_simplified(text: str) -> str:
        """繁体转简体（简化版本）"""
        # 这里可以集成更复杂的繁简转换库
        # 目前使用简单的映射
        traditional_map = {
            '學': '学', '習': '习', '醫': '医', '療': '疗', '診': '诊',
            '斷': '断', '檢': '检', '測': '测', '術': '术', '處': '处'
        }
        for trad, simp in traditional_map.items():
            text = text.replace(trad, simp)
        return text
    
    @staticmethod
    def str_q2b(text: str) -> str:
        """全角转半角"""
        result = ""
        for char in text:
            code = ord(char)
            if code == 0x3000:  # 全角空格
                code = 0x0020  # 半角空格
            elif 0xFF01 <= code <= 0xFF5E:  # 全角字符
                code -= 0xFEE0
            result += chr(code)
        return result
    
    @staticmethod
    def tokenize(text: str) -> str:
        """简单分词（可以后续集成更复杂的分词器）"""
        # 基础分词：按空格和标点分割
        tokens = re.split(r'[\s,，。？！；：""''（）【】]+', text)
        return ' '.join([t for t in tokens if t.strip()])
    
    @staticmethod
    def fine_grained_tokenize(text: str) -> str:
        """细粒度分词"""
        # 可以集成jieba等分词库
        return QueryQueryTextProcessor.tokenize(text)


class TermWeightCalculator:
    """词权重计算器"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def weights(self, tokens: List[str], preprocess: bool = True) -> List[Tuple[str, float]]:
        """计算词权重"""
        if preprocess:
            tokens = [QueryQueryTextProcessor.traditional_to_simplified(t) for t in tokens]
        
        # 简单的TF-IDF权重计算
        token_counts = defaultdict(int)
        for token in tokens:
            if token.strip():
                token_counts[token.strip()] += 1
        
        total_tokens = len(tokens)
        weighted_tokens = []
        
        for token, count in token_counts.items():
            # 基础权重：词频 / 总词数
            weight = count / total_tokens
            
            # 长度惩罚：过短的词降低权重
            if len(token) < 2:
                weight *= 0.5
            elif len(token) > 10:
                weight *= 0.8
            
            # 医学术语加权
            medical_terms = {'超声', '检查', '诊断', '病变', '心脏', '肝脏', '肾脏'}
            if any(term in token for term in medical_terms):
                weight *= 1.5
            
            weighted_tokens.append((token, weight))
        
        return weighted_tokens
    
    def split(self, text: str) -> List[str]:
        """文本分割"""
        # 按句子分割
        sentences = re.split(r'[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]


class SynonymLookup:
    """同义词查找器"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        # 简单的同义词词典
        self.synonyms = {
            '超声': ['超声波', 'B超', 'ultrasound'],
            '检查': ['检测', '检验', 'examination'],
            '诊断': ['判断', '确诊', 'diagnosis'],
            '心脏': ['心脏', 'cardiac', 'heart'],
            '肝脏': ['肝', 'liver'],
            '肾脏': ['肾', 'kidney'],
            '病变': ['疾病', '异常', 'lesion'],
            '图像': ['图片', '影像', 'image'],
            '切面': ['截面', 'section'],
            '横切面': ['横截面', 'transverse'],
            '纵切面': ['纵截面', 'longitudinal']
        }
    
    def lookup(self, term: str) -> List[str]:
        """查找同义词"""
        term_lower = term.lower().strip()
        
        # 直接匹配
        if term_lower in self.synonyms:
            return self.synonyms[term_lower]
        
        # 模糊匹配
        for key, values in self.synonyms.items():
            if key in term_lower or any(v in term_lower for v in values):
                return values
        
        return []


class QueryRewriter:
    """查询改写器 - 使用LLM进行查询改写"""
    
    def __init__(self, model_name: str = 'qwen_0_6b_local'):
        self.logger = setup_logger(self.__class__.__name__)
        self.model_name = model_name
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化LLM模型"""
        try:
            # 明确按名称获取provider，避免错误地访问到无效键（如 '0'）
            try:
                self.llm = llm_provider[self.model_name]
                self.logger.info(f"查询改写模型 {self.model_name} 初始化成功")
                return
            except Exception as inner_e:
                self.logger.warning(f"按名称加载失败({self.model_name}): {inner_e}", exc_info=False)
            
            # 回退：尝试加载配置中第一个可用provider
            from ultrasoundrag.config.config import config as _cfg
            providers_dict = _cfg.get('llm_providers', {})
            if providers_dict:
                first_name = next(iter(providers_dict.keys()))
                self.llm = llm_provider[first_name]
                self.logger.info(f"使用回退provider: {first_name}")
            else:
                self.logger.error("配置中未找到任何 llm_providers")
                self.llm = None
        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            self.llm = None
    
    def parse_rewrite_response(self, response: str) -> List[str]:
        """
        解析模型响应，提取改写查询列表
        
        Args:
            response: 模型原始响应
            
        Returns:
            List[str]: 改写查询列表
        """
        # 移除<think>标签及其内容
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 查找编号列表模式：1. 内容 2. 内容 ...
        pattern = r'(\d+)\.\s*([^\n]+)'
        matches = re.findall(pattern, response)
        
        if matches:
            # 提取查询内容并清理
            queries = []
            for _, content in matches:
                # 清理内容：去除首尾空白，移除多余的标点
                content = content.strip()
                if content and not content.startswith('<'):
                    queries.append(content)
            
            # 如果找到的查询少于5个，尝试其他模式
            if len(queries) < 3:
                # 尝试查找其他可能的列表格式
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('<') and not line.startswith('输入') and not line.startswith('输出'):
                        # 检查是否包含查询内容
                        if len(line) > 5 and not line.isdigit():
                            queries.append(line)
            
            return queries[:5]  # 最多返回5个查询
        
        # 如果没有找到编号列表，尝试按行分割
        lines = response.split('\n')
        queries = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<') and not line.startswith('输入') and not line.startswith('输出'):
                if len(line) > 5 and not line.isdigit():
                    queries.append(line)
        
        return queries[:5] if queries else ["解析失败：无法提取改写查询"]
    
    def rewrite_query(self, query: str, context: str = "", 
                     max_queries: int = 5, temperature: float = 0.3) -> List[str]:
        """
        改写查询
        
        Args:
            query: 原始查询
            context: 上下文信息
            max_queries: 最大改写查询数量
            temperature: 生成温度
            
        Returns:
            List[str]: 改写后的查询列表
        """
        if not self.llm:
            self.logger.warning("LLM模型未初始化，返回原始查询")
            return [query]
        
        try:
            # 生成改写提示词
            prompt = query_rewrite_prompt(query, context)
            
            # 调用模型生成改写结果
            start_time = time.time()
            response = self.llm.generate(
                prompt, 
                max_length=10240,
                temperature=temperature,
                do_sample=True,
                top_p=0.8
            )
            end_time = time.time()
            
            self.logger.debug(f"查询改写耗时: {end_time - start_time:.2f}秒")
            
            # 解析响应，提取改写查询
            rewrite_queries = self.parse_rewrite_response(response)
            
            # 确保包含原始查询
            if query not in rewrite_queries:
                rewrite_queries.insert(0, query)
            
            # 限制数量
            return rewrite_queries[:max_queries]
            
        except Exception as e:
            self.logger.error(f"查询改写失败: {e}")
            return [query]
    
    def batch_rewrite_queries(self, queries: List[str], context: str = "") -> Dict[str, List[str]]:
        """
        批量改写查询
        
        Args:
            queries: 查询列表
            context: 上下文信息
            
        Returns:
            Dict[str, List[str]]: 原始查询到改写查询的映射
        """
        results = {}
        for query in queries:
            results[query] = self.rewrite_query(query, context)
        return results


class MatchTextExpr:
    """匹配表达式类 - 替代原始的 MatchTextExpr"""
    
    def __init__(self, fields: List[str], query: str, score: float, 
                 options: Optional[Dict[str, Any]] = None):
        self.fields = fields
        self.query = query
        self.score = score
        self.options = options or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'fields': self.fields,
            'query': self.query,
            'score': self.score,
            'options': self.options
        }


class QueryProcessor:
    """查询处理器 - 专注于查询预处理和文本分析"""
    
    def __init__(self, enable_rewrite: bool = True, rewrite_model: str = 'qwen_0_6b_local'):
        self.logger = setup_logger(self.__class__.__name__)
        
        # 初始化组件
        self.tw = TermWeightCalculator()
        self.syn = SynonymLookup()
        
        # 查询改写器
        self.enable_rewrite = enable_rewrite
        self.rewriter = QueryRewriter(rewrite_model) if enable_rewrite else None
        
        # 查询字段配置（用于生成检索表达式）
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5", 
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]
        
        self.logger.info("查询处理器初始化完成")
    
    def question(self, txt: str, tbl: str = "qa", min_match: float = 0.6, 
                 use_rewrite: bool = True, context: str = "") -> Tuple[Optional[MatchTextExpr], List[str]]:
        """
        处理问题查询
        
        Args:
            txt: 查询文本
            tbl: 表名（兼容参数）
            min_match: 最小匹配度
            use_rewrite: 是否使用查询改写
            context: 上下文信息
            
        Returns:
            (匹配表达式, 关键词列表)
        """
        try:
            # 查询改写（如果启用）
            if use_rewrite and self.rewriter:
                try:
                    rewrite_queries = self.rewriter.rewrite_query(txt, context)
                    self.logger.debug(f"原始查询: {txt}")
                    self.logger.debug(f"改写查询: {rewrite_queries}")
                    
                    # 使用改写后的查询进行后续处理
                    if len(rewrite_queries) > 1:
                        # 如果有多个改写查询，选择最相关的一个
                        txt = self._select_best_rewrite(txt, rewrite_queries)
                    else:
                        txt = rewrite_queries[0] if rewrite_queries else txt
                        
                except Exception as e:
                    self.logger.warning(f"查询改写失败，使用原始查询: {e}")
            
            # 文本预处理
            txt = QueryQueryTextProcessor.add_space_between_eng_zh(txt)
            txt = re.sub(
                r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+",
                " ",
                QueryQueryTextProcessor.traditional_to_simplified(QueryQueryTextProcessor.str_q2b(txt.lower())),
            ).strip()
            
            otxt = txt
            txt = QueryQueryTextProcessor.remove_www(txt)
            
            if not QueryQueryTextProcessor.is_chinese(txt):
                return self._process_english_query(txt, otxt, min_match)
            else:
                return self._process_chinese_query(txt, otxt, min_match)
                
        except Exception as e:
            self.logger.error(f"问题查询处理失败: {e}")
            return None, []
    
    def _process_english_query(self, txt: str, otxt: str, min_match: float) -> Tuple[Optional[MatchTextExpr], List[str]]:
        """处理英文查询"""
        try:
            txt = QueryQueryTextProcessor.remove_www(txt)
            tks = QueryQueryTextProcessor.tokenize(txt).split()
            keywords = [t for t in tks if t]
            
            tks_w = self.tw.weights(tks, preprocess=False)
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]
            
            syns = []
            for tk, w in tks_w[:256]:
                syn = self.syn.lookup(tk)
                syn = QueryQueryTextProcessor.tokenize(" ".join(syn)).split()
                keywords.extend(syn)
                syn = ["\"{}\"^{:.4f}".format(s, w / 4.) for s in syn if s.strip()]
                syns.append(" ".join(syn))
            
            q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if
                 tk and not re.match(r"[.^+\(\)-]", tk)]
            
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,
                    )
                )
            
            if not q:
                q.append(txt)
            
            query = " ".join(q)
            return MatchTextExpr(
                self.query_fields, query, 100
            ), keywords
            
        except Exception as e:
            self.logger.error(f"英文查询处理失败: {e}")
            return None, []
    
    def _process_chinese_query(self, txt: str, otxt: str, min_match: float) -> Tuple[Optional[MatchTextExpr], List[str]]:
        """处理中文查询"""
        try:
            def need_fine_grained_tokenize(tk):
                if len(tk) < 3:
                    return False
                if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                    return False
                return True
            
            txt = QueryQueryTextProcessor.remove_www(txt)
            qs, keywords = [], []
            
            for tt in self.tw.split(txt)[:256]:
                if not tt:
                    continue
                keywords.append(tt)
                twts = self.tw.weights([tt])
                syns = self.syn.lookup(tt)
                if syns and len(keywords) < 32:
                    keywords.extend(syns)
                
                self.logger.debug(json.dumps(twts, ensure_ascii=False))
                tms = []
                
                for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                    sm = (
                        QueryTextProcessor.fine_grained_tokenize(tk).split()
                        if need_fine_grained_tokenize(tk)
                        else []
                    )
                    sm = [
                        re.sub(
                            r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；''【】、！￥……（）——《》？：""-]+",
                            "",
                            m,
                        )
                        for m in sm
                    ]
                    sm = [QueryTextProcessor.sub_special_char(m) for m in sm if len(m) > 1]
                    sm = [m for m in sm if len(m) > 1]
                    
                    if len(keywords) < 32:
                        keywords.append(re.sub(r"[ \\\"']+", "", tk))
                        keywords.extend(sm)
                    
                    tk_syns = self.syn.lookup(tk)
                    tk_syns = [QueryTextProcessor.sub_special_char(s) for s in tk_syns]
                    if len(keywords) < 32:
                        keywords.extend([s for s in tk_syns if s])
                    tk_syns = [QueryTextProcessor.fine_grained_tokenize(s) for s in tk_syns if s]
                    tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]
                    
                    if len(keywords) >= 32:
                        break
                    
                    tk = QueryTextProcessor.sub_special_char(tk)
                    if tk.find(" ") > 0:
                        tk = '"%s"' % tk
                    if tk_syns:
                        tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
                    if sm:
                        tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                    if tk.strip():
                        tms.append((tk, w))
                
                tms = " ".join([f"({t})^{w}" for t, w in tms])
                
                if len(twts) > 1:
                    tms += ' ("%s"~2)^1.5' % QueryTextProcessor.tokenize(tt)
                
                syns = " OR ".join(
                    [
                        '"%s"'
                        % QueryTextProcessor.tokenize(QueryTextProcessor.sub_special_char(s))
                        for s in syns
                    ]
                )
                if syns and tms:
                    tms = f"({tms})^5 OR ({syns})^0.7"
                
                qs.append(tms)
            
            if qs:
                query = " OR ".join([f"({t})" for t in qs if t])
                if not query:
                    query = otxt
                return MatchTextExpr(
                    self.query_fields, query, 100, {"minimum_should_match": min_match}
                ), keywords
            
            return None, keywords
            
        except Exception as e:
            self.logger.error(f"中文查询处理失败: {e}")
            return None, []
    
    def hybrid_similarity(self, avec: List[float], bvecs: List[List[float]], 
                         atks: List[str], btkss: List[List[str]], 
                         tkweight: float = 0.3, vtweight: float = 0.7) -> Tuple[List[float], List[float], List[float]]:
        """混合相似度计算"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            sims = cosine_similarity([avec], bvecs)
            tksim = self.token_similarity(atks, btkss)
            
            if np.sum(sims[0]) == 0:
                return np.array(tksim), tksim, sims[0]
            
            return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]
            
        except ImportError:
            self.logger.warning("sklearn未安装，使用简单相似度计算")
            return self.token_similarity(atks, btkss), self.token_similarity(atks, btkss), [0.0] * len(btkss)
        except Exception as e:
            self.logger.error(f"混合相似度计算失败: {e}")
            return [0.0] * len(btkss), [0.0] * len(btkss), [0.0] * len(btkss)
    
    def token_similarity(self, atks: List[str], btkss: List[List[str]]) -> List[float]:
        """词相似度计算"""
        def to_dict(tks):
            if isinstance(tks, str):
                tks = tks.split()
            d = defaultdict(int)
            wts = self.tw.weights(tks, preprocess=False)
            for i, (t, c) in enumerate(wts):
                d[t] += c
            return d
        
        atks = to_dict(atks)
        btkss = [to_dict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]
    
    def similarity(self, qtwt: Union[Dict[str, float], str], dtwt: Union[Dict[str, float], str]) -> float:
        """计算相似度"""
        if isinstance(dtwt, str):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, str):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}
        
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v
        
        q = 1e-9
        for k, v in qtwt.items():
            q += v
        
        return s / q
    
    def paragraph(self, content_tks: Union[str, List[str]], keywords: List[str] = [], 
                 keywords_topn: int = 30) -> MatchTextExpr:
        """段落处理"""
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]
        
        tks_w = self.tw.weights(content_tks, preprocess=False)
        
        keywords = [f'"{k.strip()}"' for k in keywords]
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            tk_syns = self.syn.lookup(tk)
            tk_syns = [QueryTextProcessor.sub_special_char(s) for s in tk_syns]
            tk_syns = [QueryTextProcessor.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f"\"{s}\"" if s.find(" ") > 0 else s for s in tk_syns]
            tk = QueryTextProcessor.sub_special_char(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
            if tk:
                keywords.append(f"{tk}^{w}")
        
        return MatchTextExpr(
            self.query_fields, " ".join(keywords), 100,
            {"minimum_should_match": min(3, len(keywords) // 10)}
        )
    
    def analyze_query_features(self, query: str) -> Dict[str, Any]:
        """
        分析查询特征
        
        Args:
            query: 查询文本
            
        Returns:
            查询特征分析结果
        """
        try:
            # 基础特征
            length = len(query.strip())
            word_count = len(query.split())
            
            # 语言特征
            is_chinese = QueryTextProcessor.is_chinese(query)
            
            # 关键词提取
            keywords = self._extract_keywords(query)
            
            # 同义词扩展
            expanded_keywords = self._expand_keywords(keywords)
            
            # 查询复杂度
            complexity = self._calculate_complexity(query)
            
            return {
                'original_query': query,
                'length': length,
                'word_count': word_count,
                'is_chinese': is_chinese,
                'keywords': keywords,
                'expanded_keywords': expanded_keywords,
                'complexity': complexity,
                'processed_text': QueryTextProcessor.remove_www(query)
            }
            
        except Exception as e:
            self.logger.error(f"查询特征分析失败: {e}")
            return {
                'original_query': query,
                'error': str(e)
            }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        try:
            # 预处理
            processed = QueryTextProcessor.add_space_between_eng_zh(query)
            processed = re.sub(r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+", " ", 
                             QueryTextProcessor.traditional_to_simplified(QueryTextProcessor.str_q2b(processed.lower()))).strip()
            processed = QueryTextProcessor.remove_www(processed)
            
            # 分词
            if QueryTextProcessor.is_chinese(processed):
                tokens = self.tw.split(processed)
            else:
                tokens = QueryTextProcessor.tokenize(processed).split()
            
            # 计算权重并提取关键词
            weighted_tokens = self.tw.weights(tokens)
            keywords = [token for token, weight in weighted_tokens if weight > 0.1]
            
            return keywords[:20]  # 限制关键词数量
            
        except Exception as e:
            self.logger.error(f"关键词提取失败: {e}")
            return []
    
    def _expand_keywords(self, keywords: List[str]) -> List[str]:
        """扩展关键词（添加同义词）"""
        try:
            expanded = list(keywords)
            for keyword in keywords:
                synonyms = self.syn.lookup(keyword)
                expanded.extend(synonyms)
            
            # 去重并限制数量
            return list(set(expanded))[:30]
            
        except Exception as e:
            self.logger.error(f"关键词扩展失败: {e}")
            return keywords
    
    def _calculate_complexity(self, query: str) -> Dict[str, Any]:
        """计算查询复杂度"""
        try:
            # 基础指标
            length = len(query)
            word_count = len(query.split())
            
            # 特殊字符比例
            special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fa5]', query))
            special_ratio = special_chars / max(length, 1)
            
            # 医学术语密度
            medical_terms = {'超声', '检查', '诊断', '病变', '心脏', '肝脏', '肾脏', 'ultrasound', 'examination'}
            medical_count = sum(1 for term in medical_terms if term.lower() in query.lower())
            medical_ratio = medical_count / max(word_count, 1)
            
            # 图片提示词
            image_patterns = [r'图\s*\d+', r'figure\s*\d+', r'fig\s*\d+', r'切面', r'section']
            image_count = sum(len(re.findall(pattern, query, re.IGNORECASE)) for pattern in image_patterns)
            image_ratio = image_count / max(word_count, 1)
            
            # 综合复杂度分数
            complexity_score = (special_ratio * 0.3 + medical_ratio * 0.4 + image_ratio * 0.3)
            
            return {
                'length': length,
                'word_count': word_count,
                'special_char_ratio': special_ratio,
                'medical_term_ratio': medical_ratio,
                'image_hint_ratio': image_ratio,
                'complexity_score': complexity_score,
                'is_complex': complexity_score > 0.3
            }
            
        except Exception as e:
            self.logger.error(f"复杂度计算失败: {e}")
            return {'error': str(e)}
    
    def _select_best_rewrite(self, original_query: str, rewrite_queries: List[str]) -> str:
        """
        从改写查询中选择最佳的一个
        
        Args:
            original_query: 原始查询
            rewrite_queries: 改写查询列表
            
        Returns:
            str: 最佳查询
        """
        if not rewrite_queries:
            return original_query
        
        if len(rewrite_queries) == 1:
            return rewrite_queries[0]
        
        # 简单的选择策略：选择长度适中且包含关键词的查询
        best_query = original_query
        best_score = 0
        
        for query in rewrite_queries:
            score = 0
            
            # 长度评分：避免过短或过长
            length = len(query)
            if 5 <= length <= 50:
                score += 1
            elif length > 50:
                score += 0.5
            
            # 关键词保持评分
            original_keywords = set(original_query.lower().split())
            query_keywords = set(query.lower().split())
            keyword_overlap = len(original_keywords & query_keywords) / max(len(original_keywords), 1)
            score += keyword_overlap * 2
            
            # 医学术语评分
            medical_terms = {'超声', '检查', '诊断', '心脏', '肝脏', '肾脏', 'ultrasound', 'examination'}
            medical_count = sum(1 for term in medical_terms if term in query.lower())
            score += medical_count * 0.5
            
            if score > best_score:
                best_score = score
                best_query = query
        
        return best_query
    
    def enhanced_question(self, txt: str, context: str = "", 
                         use_rewrite: bool = True, 
                         return_rewrites: bool = False) -> Union[Tuple[Optional[MatchTextExpr], List[str]], 
                                                               Tuple[Optional[MatchTextExpr], List[str], List[str]]]:
        """
        增强的问题查询处理，包含查询改写功能
        
        Args:
            txt: 查询文本
            context: 上下文信息
            use_rewrite: 是否使用查询改写
            return_rewrites: 是否返回改写查询列表
            
        Returns:
            如果return_rewrites=False: (匹配表达式, 关键词列表)
            如果return_rewrites=True: (匹配表达式, 关键词列表, 改写查询列表)
        """
        rewrite_queries = []
        
        try:
            # 查询改写（如果启用）
            if use_rewrite and self.rewriter:
                try:
                    rewrite_queries = self.rewriter.rewrite_query(txt, context)
                    self.logger.debug(f"原始查询: {txt}")
                    self.logger.debug(f"改写查询: {rewrite_queries}")
                    
                    # 使用最佳改写查询
                    if rewrite_queries:
                        txt = self._select_best_rewrite(txt, rewrite_queries)
                        
                except Exception as e:
                    self.logger.warning(f"查询改写失败，使用原始查询: {e}")
            
            # 处理查询
            match_expr, keywords = self.question(txt, use_rewrite=False)
            
            if return_rewrites:
                return match_expr, keywords, rewrite_queries
            else:
                return match_expr, keywords
                
        except Exception as e:
            self.logger.error(f"增强查询处理失败: {e}")
            if return_rewrites:
                return None, [], []
            else:
                return None, []
    
    def batch_process_queries(self, queries: List[str], context: str = "", 
                             use_rewrite: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表
            context: 上下文信息
            use_rewrite: 是否使用查询改写
            
        Returns:
            Dict[str, Dict[str, Any]]: 查询处理结果
        """
        results = {}
        
        for query in queries:
            try:
                match_expr, keywords, rewrite_queries = self.enhanced_question(
                    query, context, use_rewrite, return_rewrites=True
                )
                
                results[query] = {
                    'match_expr': match_expr.to_dict() if match_expr else None,
                    'keywords': keywords,
                    'rewrite_queries': rewrite_queries,
                    'success': match_expr is not None
                }
                
            except Exception as e:
                self.logger.error(f"批量处理查询失败 - 查询: {query}, 错误: {e}")
                results[query] = {
                    'match_expr': None,
                    'keywords': [],
                    'rewrite_queries': [],
                    'success': False,
                    'error': str(e)
                }
        
        return results


# 便捷函数
def create_query_processor(enable_rewrite: bool = True, rewrite_model: str = 'qwen_0_6b_local') -> QueryProcessor:
    """创建查询处理器"""
    return QueryProcessor(enable_rewrite=enable_rewrite, rewrite_model=rewrite_model)


def create_text_processor() -> QueryTextProcessor:
    """创建文本处理器"""
    return QueryTextProcessor()


def create_term_weight_calculator() -> TermWeightCalculator:
    """创建词权重计算器"""
    return TermWeightCalculator()


def create_synonym_lookup() -> SynonymLookup:
    """创建同义词查找器"""
    return SynonymLookup()


def create_query_rewriter(model_name: str = 'qwen_0_6b_local') -> QueryRewriter:
    """创建查询改写器"""
    return QueryRewriter(model_name)


# 向后兼容
def create_fulltext_queryer(db_name: str = "default") -> QueryProcessor:
    """创建查询处理器（向后兼容）"""
    return QueryProcessor()


if __name__ == "__main__":
    # 仅测试 QueryRewriter
    print("=== QueryRewriter 改写测试 ===")

    rewriter = create_query_rewriter(model_name='qwen_0_6b_local')

    test_queries = [
        "心脏超声检查方法",
        "图2-3 胎儿发育",
        "肝脏病变诊断标准"
    ]

    for q in test_queries:
        print(f"\n原始查询: {q}")
        rewrites = rewriter.rewrite_query(q, context="医学超声检查相关", max_queries=5, temperature=0.3)
        print("改写结果:")
        for i, r in enumerate(rewrites, 1):
            print(f"  {i}. {r}")
