####################################
# 通过检索信息让模型生成答案的prompt
####################################

def generate_advanced_medical_prompt(query, context):
    """
    高级医学prompt - 专门针对RAG评估结果优化
    重点提升：答案相关性、上下文忠实性、医学专业性
    """
    prompt = f"""
# 角色定位
你是一位拥有20年临床经验的资深医学超声诊断专家，专精于超声影像诊断和临床指导。

# 核心任务
基于提供的医学资料，为用户提供专业、准确、全面的医学回答。

# 回答原则

## 1. 信息忠实性
- **严格基于资料**：所有信息必须来自提供的医学资料
- **禁止编造**：不得添加任何推测、假设或编造的信息
- **引用明确**：如果资料信息不足，明确说明"根据提供资料，无法获取..."
- **保持客观**：避免主观判断，基于客观医学事实

## 2. 答案相关性
- **直接回答**：开篇即直接回答用户核心问题
- **问题聚焦**：确保每个要点都直接针对用户问题
- **避免冗余**：不重复问题，不添加无关信息
- **逻辑清晰**：按照医学逻辑组织答案结构

## 3. 医学专业性
- **术语准确**：使用标准医学术语，必要时提供解释
- **临床实用**：提供具有临床指导意义的信息
- **结构规范**：采用医学报告的标准结构
- **权威性**：体现专业医学权威性

# 回答结构

## 核心回答（必选）
- 直接回答用户问题的核心要点
- 使用资料中的具体医学信息
- 确保信息准确性和相关性

## 详细说明（可选）
- 补充相关的医学背景知识
- 解释重要的医学术语或概念
- 提供临床应用指导

## 注意事项（可选）
- 临床实践中的关键要点
- 可能的局限性或注意事项
- 进一步检查或咨询建议

# 质量控制检查
在生成答案前，请自问：
1. 答案是否直接回答了用户问题？
2. 所有信息是否来自提供的资料？
3. 是否避免了无关或冗余内容？
4. 医学术语是否准确专业？
5. 答案结构是否清晰易懂？

# 特殊情况处理
- **资料充足**：提供全面、详细的专业回答
- **资料不足**：明确说明信息缺失的具体方面
- **资料矛盾**：指出矛盾之处，建议进一步确认
- **专业建议**：涉及诊断时，提醒咨询专业医生

---

用户问题: {query}

医学资料:
{context}

请提供专业、准确的医学回答:"""
    return prompt

"""精简一下这个prompt"""
def generate_advanced_medical_prompt_simple(query, context):
    """
    精简版医学prompt - 减少token消耗，保持核心功能
    """
    prompt = f"""你是资深医学超声诊断专家。请基于以下【医学资料】，为【用户问题】提供一个全面、详尽且结构清晰的专业回答。

【核心要求】
• **内容全面**：综合所有相关信息，深入挖掘细节，确保回答的完整性。
• **严格循证**：所有结论必须严格基于提供的资料，如果资料不充分，请明确指出。
• **术语精准**：使用准确的医学术语，并对关键概念进行简要解释。
• **逻辑清晰**：回答应条理分明，易于理解。

【建议格式】
1.  **直接回答**: 首先概括性地回答用户的核心问题。
2.  **详细说明**: 对答案进行分点阐述，深入分析细节，并引用资料中的关键信息作为依据。
3.  **补充信息/注意事项**: (如果资料中有) 提供相关的背景知识、临床建议或需要注意的事项。

【用户问题】: {query}

【医学资料】:
{context}

【建议格式】
1.  **直接回答**: 首先概括性地回答用户的核心问题。
2.  **详细说明**: 对答案进行分点阐述，深入分析细节，并引用资料中的关键信息作为依据。
3.  **补充信息/注意事项**: (如果资料中有) 提供相关的背景知识、临床建议或需要注意的事项。

【专业回答】:"""
    return prompt

def generate_strict_medical_prompt(query, context):
    """
    最严格的医学prompt，旨在最大限度地提高忠实度和相关性
    """
    prompt = f"""
# 角色
你是一名严谨的医学信息分析员。

# 核心指令
你的唯一任务是根据下面提供的【医学资料】，为【用户问题】生成一个精准、忠实的答案。

# 行为准则 (必须严格遵守)
1.  **绝对忠实**: 你的回答必须 **完全** 基于【医学资料】。严禁使用任何外部知识或个人推断。
2.  **直接回答**: 回答必须直接针对【用户问题】，不要提供不相关的背景信息。
3.  **引用来源**: (如果可能) 在关键信息后用方括号注明来源，例如 "[来源：文档1]"。
4.  **未知则答未知**: 如果【医学资料】中没有足够信息来回答问题，你 **必须** 回答："根据提供的医学资料，无法回答该问题。" 并且不能添加任何额外信息。
5.  **简洁明了**: 保持答案简洁，只包含回答问题所必需的信息。

---

【用户问题】: {query}

---

【医学资料】:
{context}

---

【分析员回答】:"""
    return prompt
    
####################################
# 基于评估结果优化的prompt
####################################

def generate_optimized_medical_prompt(query, context):
    """
    基于评估结果优化的医学prompt，重点提升答案相关性和忠实性
    """
    prompt = f"""
你是一位资深的医学超声诊断专家。请基于提供的医学资料，直接、准确地回答用户问题。

【核心要求】
1. **直接回答**: 开篇就要直接回答用户的核心问题，不要绕弯子
2. **信息准确**: 严格基于提供的资料，不添加任何推测或编造的信息
3. **结构清晰**: 使用要点式回答，便于理解
4. **专业术语**: 使用准确的医学术语，必要时提供简要解释
5. **信息完整性**: 如果资料不足，明确说明哪些方面信息缺失

【回答格式】
- 开头：直接回答核心问题（1-2句话）
- 主体：详细说明（分点描述）
- 结尾：如有必要，补充注意事项或建议

【质量控制】
- 确保每个要点都来自提供的资料
- 避免重复或冗余信息
- 保持逻辑性和连贯性
- 字数控制在150-200字

用户问题: {query}

医学资料:
{context}

专业回答:"""
    return prompt

def generate_focused_answer_prompt(query, context):
    """
    聚焦式答案生成prompt，专门针对答案相关性优化
    """
    prompt = f"""
作为医学专家，请针对用户的具体问题提供精准回答。

【问题分析】
用户询问: {query}

【回答策略】
1. 首先识别问题的核心要点
2. 从资料中提取最相关的信息
3. 组织成直接、准确的答案
4. 确保答案完全针对用户问题

【资料内容】
{context}

【回答要求】
- 开篇直接回答用户问题
- 使用资料中的具体信息
- 避免无关的扩展内容
- 保持专业性和准确性
- 如果资料不完整，明确说明

请提供精准回答:"""
    return prompt

def generate_enhanced_medical_prompt(query, context):
    """
    增强版医学prompt，结合多种优化策略
    """
    prompt = f"""
你是医学超声诊断专家。请基于提供的资料，精准回答用户问题。

【问题理解】
用户问题: {query}

【回答原则】
✓ 直接回答：开篇即答，不绕弯子
✓ 信息准确：严格基于资料，不编造
✓ 结构清晰：要点式回答，逻辑分明
✓ 专业准确：使用正确医学术语
✓ 完整性：如资料不足，明确说明

【资料内容】
{context}

【回答结构】
1. 直接回答（1-2句话）
2. 详细说明（分点描述）
3. 补充要点（如有必要）

【质量检查】
- 答案是否直接针对用户问题？
- 信息是否来自提供的资料？
- 是否避免了无关内容？
- 是否保持了专业性？

请提供精准、专业的回答:"""
    return prompt


def generate_medical_prompt(query, context):
    """医学专业版prompt"""
    return f"""
你是一位资深的医学超声诊断专家，拥有丰富的临床经验。请基于提供的医学资料，为用户提供专业、准确的回答。

【回答原则】
- 严格基于提供的医学资料，不得编造信息
- 保持医学术语的准确性和专业性
- 重要概念需要简要解释，便于理解
- 涉及诊断建议时，提醒患者咨询专业医生

【格式要求】
- 答案结构清晰，逻辑分明
- 字数控制在200字以内
- 使用中文回答

用户咨询: {query}

参考资料:
{context}

专家解答:"""


def generate_contextual_prompt(query, context):
    """上下文感知prompt"""
    return f"""
作为医学超声专家，请仔细分析用户问题和提供的资料，给出准确回答。

【智能分析】
- 如果问题涉及解剖结构，重点描述位置和特征
- 如果问题涉及检查方法，重点说明操作要点
- 如果问题涉及诊断，重点阐述判断依据
- 如果资料不足，诚实说明并给出建议

【质量要求】
• 答案准确性优先于完整性
• 重要信息优先于次要细节
• 简洁明了优于冗长描述
• 200字以内，使用中文

用户问题: {query}

参考资料:
{context}

专业回答:"""


# 直接生成：不依赖检索资料的医学回答提示词
def generate_direct_medical_prompt(query: str,
                                  role: str = "资深医学超声诊断专家",
                                  language: str = "中文",
                                  ) -> str:
    """
    在无检索上下文时用于直接生成答案的提示词。
    - 避免强制引用资料；聚焦问题本身；控制字数；保持专业性与安全性。
    """
    prompt = f"""
你是一位{role}，请使用{language}回答下列问题。请基于你的通用医学知识与临床常识，给出专业、清晰、结构化的回答。

【用户问题】
{query}

【专业回答】
"""
    return prompt

# def generate_grounded_medical_prompt(query, context):
#     """
#     强资料约束的医学prompt：严格要求仅基于【资料分块】作答，并显式引用【文档分块 i】。
#     """
#     prompt = f"""
# 你是一位资深医学超声诊断专家。请严格基于下面提供的【资料分块】回答用户问题，禁止使用外部知识或主观推断。

# 【回答规则（必须遵守）】
# 1. 尽可能多使用【资料分块】中的信息作答。”
# 2. 开头先直接回答用户问题的核心结论。
# 3. 主体分点说明，逐点引用相应的【文档分块 i】作为依据。
# 4. 引用格式：在每个要点结尾用【参考：文档分块 i(可多项)】。
# 5. 字数尽量不要太少！

# 【用户问题】
# {query}

# 【资料分块】
# {context}

# 【请按以下结构输出】
# - 直接回答：先根据问题给出核心结论。
# - 依据与说明：分点阐述，并在每点末尾标注【参考：文档分块 i,…】。
# """
#     return prompt
def generate_grounded_prompt_simple(query, context,
                                   role="超声资深领域专家",
                                   language="中文",):
    """
    在有检索上下文时用于直接生成答案的提示词。
    """
    prompt = f"""
你是一位{role}，请使用{language}回答下列问题。请基于检索到的资料分块，结合你的通用医学知识与临床常识，给出专业、清晰、结构化的回答。

【用户问题】
{query}

【资料分块】
{context}

【专业回答】
尽可能结合资料分块，给出专业、清晰、结构化的回答，并表明你参考了哪些资料分块。
"""
    return prompt


####################################
# 动态prompt选择器；暂时不做！
####################################



####################################
# 大模型版本的rerank的prompt
####################################

def llm_rerank_prompt(query, candidates):
    """
    创建LLM重排序提示词
    
    Args:
        query: 用户查询
        candidates: 候选文档列表，每个元素应包含content属性
    
    Returns:
        str: 重排序提示词
    """
    prompt = f"""请对以下文档根据与查询的相关性进行重新排序。

查询: {query}

文档列表:
"""
    
    for i, result in enumerate(candidates, 1):
        # 处理不同类型的候选文档
        if hasattr(result, 'content'):
            content = result.content
        elif isinstance(result, dict):
            content = result.get('content', str(result))
        else:
            content = str(result)
            
        content_preview = content[:200] + "..." if len(content) > 200 else content
        prompt += f"{i}. {content_preview}\n\n"
    
    prompt += """
请按照相关性从高到低的顺序，返回文档编号列表（用逗号分隔）。
例如：3,1,5,2,4

排序结果:"""
    
    return prompt

def llm_rerank_advanced_prompt(query, candidates):
    """
    高级LLM重排序提示词（包含评分说明）
    
    Args:
        query: 用户查询
        candidates: 候选文档列表
    
    Returns:
        str: 高级重排序提示词
    """
    prompt = f"""你是一位专业的信息检索专家。请根据查询与文档的相关性对以下文档进行重新排序。

【排序标准】
1. 内容相关性：文档内容与查询的匹配程度
2. 信息完整性：文档是否包含查询所需的完整信息
3. 专业准确性：信息的专业性和准确性
4. 实用价值：对回答查询的实际帮助程度

用户查询: {query}

候选文档:
"""
    
    for i, result in enumerate(candidates, 1):
        if hasattr(result, 'content'):
            content = result.content
        elif isinstance(result, dict):
            content = result.get('content', str(result))
        else:
            content = str(result)
            
        content_preview = content[:300] + "..." if len(content) > 300 else content
        prompt += f"""
文档 {i}:
{content_preview}

---
"""
    
    prompt += f"""
请综合考虑上述排序标准，将文档按相关性从高到低重新排序。
只需返回文档编号的排序列表，用逗号分隔。

排序结果:"""
    
    return prompt

def llm_rerank_with_explanation_prompt(query, candidates):
    """
    带解释的LLM重排序提示词
    
    Args:
        query: 用户查询
        candidates: 候选文档列表
    
    Returns:
        str: 带解释的重排序提示词
    """
    prompt = f"""作为信息检索专家，请对以下文档根据与查询的相关性进行重排序，并简要说明排序理由。

查询: {query}

文档列表:
"""
    
    for i, result in enumerate(candidates, 1):
        if hasattr(result, 'content'):
            content = result.content
        elif isinstance(result, dict):
            content = result.get('content', str(result))
        else:
            content = str(result)
            
        content_preview = content[:200] + "..." if len(content) > 200 else content
        prompt += f"{i}. {content_preview}\n\n"
    
    prompt += """
请按以下格式回答：

排序结果: [文档编号列表，用逗号分隔]
排序理由: [简要说明排序依据，50字以内]

示例：
排序结果: 3,1,5,2,4
排序理由: 文档3最直接回答查询，文档1提供相关背景，其他文档相关性较低。

你的回答:"""
    
    return prompt

def llm_rerank_medical_prompt(query, candidates):
    """
    医学专业领域的LLM重排序提示词
    
    Args:
        query: 医学相关查询
        candidates: 候选文档列表
    
    Returns:
        str: 医学专业重排序提示词
    """
    prompt = f"""你是一位医学信息检索专家。请根据医学专业性和临床相关性对以下文档进行重排序。

【医学排序标准】
- 临床准确性：信息的医学准确性
- 专业相关性：与查询的专业匹配度
- 实用价值：对医学实践的指导意义
- 权威性：信息来源的可靠性

医学查询: {query}

候选医学文档:
"""
    
    for i, result in enumerate(candidates, 1):
        if hasattr(result, 'content'):
            content = result.content
        elif isinstance(result, dict):
            content = result.get('content', str(result))
        else:
            content = str(result)
            
        content_preview = content[:250] + "..." if len(content) > 250 else content
        prompt += f"""
文档 {i}:
{content_preview}

"""
    
    prompt += """
基于医学专业判断，请将文档按相关性和准确性从高到低排序。
返回格式：文档编号列表（用逗号分隔）

专业排序结果:"""
    
    return prompt

def get_rerank_prompt(query, candidates, prompt_type="basic"):
    """
    重排序提示词选择器
    
    Args:
        query: 用户查询
        candidates: 候选文档列表
        prompt_type: 提示词类型 ("basic", "advanced", "explanation", "medical", "auto")
    
    Returns:
        str: 选择的重排序提示词
    """
    
    if prompt_type == "auto":
        # 自动选择提示词类型
        query_lower = query.lower()
        
        # 检查是否为医学相关查询
        medical_keywords = ["医学", "诊断", "治疗", "病", "症状", "检查", "超声", "影像"]
        if any(keyword in query_lower for keyword in medical_keywords):
            prompt_type = "medical"
        # 检查是否需要详细解释
        elif len(candidates) <= 5:
            prompt_type = "explanation"
        # 检查是否为复杂查询
        elif len(query) > 20:
            prompt_type = "advanced"
        else:
            prompt_type = "basic"
    
    if prompt_type == "advanced":
        return llm_rerank_advanced_prompt(query, candidates)
    elif prompt_type == "explanation":
        return llm_rerank_with_explanation_prompt(query, candidates)
    elif prompt_type == "medical":
        return llm_rerank_medical_prompt(query, candidates)
    else:
        return llm_rerank_prompt(query, candidates)

####################################
# 重排序结果解析函数
####################################

# def parse_rerank_result(llm_response, num_candidates):
#     """
#     解析LLM重排序结果
    
#     Args:
#         llm_response: LLM返回的响应
#         num_candidates: 候选文档数量
    
#     Returns:
#         tuple: (排序索引列表, 是否解析成功)
#     """
#     try:
#         # 尝试提取排序结果
#         lines = llm_response.strip().split('\n')
        
#         # 查找包含数字和逗号的行
#         result_line = None
#         for line in lines:
#             if ',' in line and any(char.isdigit() for char in line):
#                 result_line = line
#                 break
        
#         if not result_line:
#             # 如果没找到，取最后一行
#             result_line = lines[-1]
        
#         # 提取数字
#         import re
#         numbers = re.findall(r'\d+', result_line)
        
#         if not numbers:
#             return None, False
        
#         # 转换为索引（减1）
#         indices = [int(num) - 1 for num in numbers]
        
#         # 验证索引有效性
#         valid_indices = [idx for idx in indices if 0 <= idx < num_candidates]
        
#         if len(valid_indices) == num_candidates:
#             return valid_indices, True
#         else:
#             return None, False
            
#     except Exception:
#         return None, False


####################################
# 文档指南答案生成的prompt
####################################

def guide_generate_contextual_prompt(query, context):
    """
    为超声检查指南生成上下文感知prompt的优化版本。

    该prompt指导模型扮演超声专家，根据提供的上下文，
    为用户的查询生成详细、结构化的操作流程。
    """
    prompt = f"""
# 角色
你是一位顶级的超声诊断专家，尤其精通超声检查指南中的每一个细节。你的任务是根据提供的【相关文档分块】，为用户的【查询】生成一份专业、严谨、可执行的超声检查操作流程。

# 指令请严格遵循以下步骤和格式要求，提供专家级的回答，注意只能用检索到的文本块的内容回答，不能捏造任何虚假信息并且参考文献也需要真实有效不能去网上检索，必须来自检索到的文本块。

## 1. 分析与综合
- **核心任务**: 将【相关文档分块】中的信息整合成一个连贯的操作指南。
- **信息处理**: 如果多个分块内容相关，请将其综合成一个步骤。
- **引用**: 在每个包含引用信息的步骤结尾，必须用 `【参考：文档分块 X】` 的格式注明来源。如果一个步骤综合了多个分块，请全部列出，例如 `【参考：文档分块 1, 3】`。

## 2. 内容要求
你的回答必须包含以下结构化内容：

### A. 准备工作 (可选)
- 简要说明检查前的准备，如患者体位、所需探头型号等。

### B. 扫查步骤 (核心部分)
- **编号列表**: 使用 `1.`、`2.`、`3.`... 格式，分步描述操作流程。
- **每步要素**: 每个步骤应至少包含：
    - **探头**: 放置位置、角度、移动方向。
    - **参数**: 建议的机器参数（如频率、焦点、增益）。
    - **要点**: 需要重点观察的解剖结构和判读要点。

### C. 注意事项
- 总结关键的注意事项或常见伪像。

## 3. 信息不足处理
- 如果【相关文档分块】中的信息不足以回答【查询】，请明确指出：“**根据所提供资料，无法获取关于...的完整信息，建议参考原始指南或相关文献。**”
- **禁止编造**：绝对不能虚构或猜测缺失的信息。

# 示例
- **用户问题**: 如何进行肩关节冈上肌肌腱的长轴扫查？
- **专业回答**:
    1.  **患者体位**: 患者正坐，面向检查者，患侧手置于背部后方并内旋（Crass位或改良Crass位），使冈上肌肌腱充分暴露。【参考：文档分块 2】
    2.  **探头放置**: 使用高频线阵探头（>10 MHz），将其放置于肩峰前外侧，探头长轴与冈上肌肌腱长轴保持一致，声束垂直于肌腱纤维。【参考：文档分块 1, 4】
    3.  **图像观察**: 屏幕中应清晰显示喙肩弓下方的冈上肌肌腱，呈均匀的纤维状高回声。需仔细观察肌腱的形态、回声、有无钙化或撕裂。【参考：文档分块 3】

---

# 正式任务

## 【查询】
{query}

## 【相关文档分块】
{context}

## 【你的专业回答】
"""
    return prompt

####################################
# 查询改写prompt
####################################
def query_rewrite_prompt(query, context):
    """
    查询改写prompt
    
    Args:
        query (str): 用户原始查询
        context (str): 上下文信息
        
    Returns:
        str: 格式化的查询改写提示词
    """
    prompt = f"""你是一位查询扩展专家。请对用户的查询进行扩展或改写，返回5个不同版本的查询。

要求：
1. 使用同义词或相关词
2. 将缩写写全
3. 添加澄清性细节
4. 改变表达方式
5. 如果不是英文则翻译成英文

请严格按照示例输出，禁止输出任何解释说明或思考过程，只给出编号列表。

示例：
输入：超声检查
输出：
1. 超声波检查
2. 超声诊断
3. 超声影像检查
4. 超声扫描
5. ultrasound examination

输入：甲状腺超声
输出：
1. 甲状腺超声检查
2. 甲状腺超声诊断
3. 甲状腺超声影像
4. 甲状腺超声扫描
5. thyroid ultrasound

现在请改写以下查询：
输入：{query}
输出："""
    
    return prompt
# def query_rewrite_prompt(query, context):
#     """
#     查询改写prompt
    
#     Args:
#         query (str): 用户原始查询
#         context (str): 上下文信息
        
#     Returns:
#         str: 格式化的查询改写提示词
#     """
#     prompt = f"""请将以下查询改写为5个不同版本，直接输出编号列表：

# {query}

# 输出：
# 1.
# 2.
# 3.
# 4.
# 5."""
    
#     return prompt