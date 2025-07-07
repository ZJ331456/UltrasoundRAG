####################################
# 通过检索信息让模型生成答案的prompt
####################################

def generate_prompt(query, context):
    """生成答案的提示词模板"""
    return f"""
请基于以下相关信息，简明扼要地回答用户的问题。回答要求：
1. 直接回答问题，不要重复问题
2. 答案要准确、简洁，控制在200字以内
3. 如果信息不足，请说明
4. 使用中文回答

用户问题: {query}

相关信息:
{context}

答案:"""

# print(generate_prompt("细胞是什么？", "不知道"))

####################################
# 通过检索信息让模型生成答案的prompt
####################################

def generate_prompt(query, context):
    """生成答案的提示词模板"""
    return f"""
你是一位专业的医学超声专家，请基于以下相关信息，准确回答用户的问题。

回答要求：
1. 基于提供的信息回答，不要编造或猜测
2. 答案要专业准确，条理清晰
3. 使用医学术语时请适当解释
4. 控制在200字以内，重点突出
5. 如果信息不足以完整回答，请明确指出
6. 使用中文回答

用户问题: {query}

相关医学资料:
{context}

专业回答:"""

####################################
# 分层次的prompt优化版本
####################################

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

def generate_step_by_step_prompt(query, context):
    """结构化思维prompt"""
    return f"""
请作为医学专家，基于提供的资料回答问题。请按以下步骤思考：

1. 分析问题核心
2. 提取关键信息
3. 组织专业回答

【回答标准】
✓ 基于资料内容，避免主观推测
✓ 专业术语配合通俗解释
✓ 结构清晰，重点突出
✓ 200字以内，中文回答

问题: {query}

资料内容:
{context}

分析与回答:"""

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

####################################
# 动态prompt选择器
####################################

def get_optimized_prompt(query, context, prompt_type="auto"):
    """
    根据问题类型选择最适合的prompt
    
    Args:
        query: 用户问题
        context: 相关信息
        prompt_type: prompt类型 ("basic", "medical", "step", "contextual", "auto")
    """
    
    if prompt_type == "auto":
        # 自动识别问题类型
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["是什么", "什么是", "定义", "概念"]):
            prompt_type = "step"  # 概念类问题用结构化思维
        elif any(word in query_lower for word in ["如何", "怎么", "方法", "步骤"]):
            prompt_type = "contextual"  # 方法类问题用上下文感知
        elif any(word in query_lower for word in ["诊断", "检查", "治疗"]):
            prompt_type = "medical"  # 医学类问题用专业版
        else:
            prompt_type = "basic"  # 其他用基础版
    
    if prompt_type == "medical":
        return generate_medical_prompt(query, context)
    elif prompt_type == "step":
        return generate_step_by_step_prompt(query, context)
    elif prompt_type == "contextual":
        return generate_contextual_prompt(query, context)
    else:
        return generate_prompt(query, context)

####################################
# 带反思的prompt
####################################

def generate_reflective_prompt(query, context):
    """带自我检查的prompt"""
    return f"""
你是医学超声专家。请基于资料回答问题，并进行自我检查。

用户问题: {query}

参考资料:
{context}

请按以下格式回答：

【分析】简要分析问题要点

【回答】基于资料的专业回答（150字以内）

【检查】
- 回答是否基于提供的资料？
- 专业术语是否准确？
- 是否有遗漏重要信息？

最终答案:"""

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
