参考：https://blog.csdn.net/weixin_42608414/article/details/135355723
# 输入参数要求：
* question：用户输入的问题。
* answer：从 RAG 系统生成的答案(由LLM给出)。
* contexts：根据用户的问题从外部知识源检索的上下文即与问题相关的文档。
* ground_truths： 人类提供的基于问题的真实(正确)答案。 这是唯一的需要人类提供的信息。 
# 评估指标
* 忠实度(faithfulness)  
忠实度(faithfulness)衡量了生成的答案(answer)与给定上下文(context)的事实一致性。它是根据answer和检索到的context计算得出的。并将计算结果缩放到 (0,1) 范围且越高越好。
* 答案相关性(Answer relevancy)  
评估指标“答案相关性”重点评估生成的答案(answer)与用户问题(question)之间相关程度。不完整或包含冗余信息的答案将获得较低分数。该指标是通过计算question和answer获得的，它的取值范围在 0 到 1 之间，其中分数越高表示相关性越好。
* 上下文精度(Context precision)--需要人类提供的ground truth  
上下文精度是一种衡量标准，它评估所有在上下文(contexts)中呈现的与基本事实(ground-truth)相关的条目是否排名较高。理想情况下，所有相关文档块(chunks)必须出现在顶层。该指标使用question和计算contexts，值范围在 0 到 1 之间，其中分数越高表示精度越高。 
* 上下文召回率(Context recall)--需要人类提供的ground truth  
上下文召回率(Context recall)衡量检索到的上下文(Context)与人类提供的真实答案(ground truth)的一致程度。它是根据ground truth和检索到的Context计算出来的，取值范围在 0 到 1 之间，值越高表示性能越好。
* 上下文相关性(Context relevancy)  
该指标衡量检索到的上下文(Context)的相关性，根据用户问题(question)和上下文(Context)计算得到，并且取值范围在 (0, 1)之间，值越高表示相关性越好。理想情况下，检索到的Context应只包含解答question的信息。 我们首先通过识别检索到的Context中与回答question相关的句子数量来估计 |S| 的值。

# 选定指标
由于目前测试阶段需要人工测试，所以选择不需要真实答案的指标
* 忠实度(faithfulness)  
* 答案相关性(Answer relevancy)  
* 上下文相关性(Context relevancy)  