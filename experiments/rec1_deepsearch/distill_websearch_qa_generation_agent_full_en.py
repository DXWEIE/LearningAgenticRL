

QWEN_KEY = 'sk-'
import json
import pandas as pd
import numpy as np
import random
from openai import OpenAI
import math



def get_model_output(text_in,used_model='qwen-plus-2025-09-11',temperature=0):
    attemp_times = 0
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    while attemp_times < 3:
        try:
            completion = client.chat.completions.create(
                # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                model=used_model,
                messages=[
                    {"role": "user", "content": text_in},
                ],
                temperature=temperature,
                timeout=70,
                extra_body={"enable_thinking": False},
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
        attemp_times += 1
    return None


def get_model_output_with_search_tool(text_in,used_model='qwen-plus-2025-09-11',temperature=0,search_strategy="turbo"):
    attemp_times = 0

    client = OpenAI(
            api_key=QWEN_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    while attemp_times < 2:
        try:
            completion = client.chat.completions.create(
                # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                model=used_model,
                messages=[
                    {"role": "user", "content": text_in},
                ],
                temperature=temperature,
                timeout=400,
                    extra_body={
                    "enable_search": True,
                    "search_options": {
                        "forced_search": True,
                        "search_strategy": search_strategy # 配置搜索策略为高性能模式
                    }
                },
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
        attemp_times += 1
    return None


def get_model_output_with_logprobs(text_in, temperature=0):
    """获取模型输出及 logprobs，带重试和统计"""
    
    attemp_times = 0
    client = OpenAI(
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    while attemp_times < 3:
        try:
            completion = client.chat.completions.create(
                model='qwen-plus-2025-09-11',
                messages=[{"role": "user", "content": text_in}],
                timeout=70,
                temperature=temperature,
                logprobs=True,
                top_logprobs=1,  # 只返回实际输出的 token
                extra_body={"enable_thinking": False},
            )
            
            # 提取 token 信息
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            
            content = completion.choices[0].message.content
            logprobs = completion.choices[0].logprobs
            probs_list = []
            if logprobs and logprobs.content:
                for item in logprobs.content:
                    probs_list.append(item.logprob)
                avg_prob = sum(probs_list) / len(probs_list) if probs_list else 0
                avg_prob = math.exp(avg_prob)  # 转换回概率空间
            else:
                avg_prob = None
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "logprobs": avg_prob,
                "error": None
            }
            
        except Exception as e:
            print(f"Error (attempt {attemp_times + 1}/3): {e}")
            attemp_times += 1
    # 降级
    content = get_model_output(text_in, temperature)
    return {
        "content": content,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "logprobs": None,
        "error": "logprobs request failed after 3 attempts"
    }

from openai import OpenAI
import os


def get_model_output_with_search(text_in, used_model='qwen-plus-2025-09-11',search_strategy="max"):
    client = OpenAI(
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=used_model,
        messages=[
            {"role": "user", "content": text_in},
        ],
        extra_body={
            "enable_search": True,
            "search_options": {
                "forced_search": True,
                "search_strategy": search_strategy  # 配置搜索策略为高性能模式
            }
        },
        temperature=0
    )
    return completion.choices[0].message.content


def get_model_output_with_agentic_search(text_in,used_model='qwen-plus-2025-09-11',temperature=0,search_strategy="turbo"):
    # 初始化OpenAI客户端
    client = OpenAI(
        # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复

    # 创建聊天完成请求
    completion = client.chat.completions.create(
        # 此处以qwen-plus为例，可更换为其它支持联网搜索的深度思考模型
        model=used_model,
        messages=[{"role": "user", "content": text_in}],
        extra_body={
            "enable_thinking": True,
            "enable_search": True,  # 开启联网搜索的参数
            "search_options": {
                "forced_search": True,  # 强制联网搜索的参数
                "search_strategy": search_strategy
            },
        },
        stream=True,
        stream_options={"include_usage": True}
    )

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            continue
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, "reasoning_content") and delta.reasoning_content != None:
                #print(delta.reasoning_content, end="", flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering is False:
                    #print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                #print(delta.content, end="", flush=True)
                answer_content += delta.content

    return answer_content


# gen_seed_qa_prompt_template = """# Role
# 你是一位专业的数据标注专家，专门负责构建 GAIA (General AI Assistant) 基准测试数据集。你的任务是根据给定的事实陈述句，生成一组高质量的“种子问答对”。

# # Goal
# 仅基于输入的事实句子，生成 2-3 个不同角度的问题及其对应的精确答案。这些问题应该能够测试 AI 助手的信息提取、推理和事实核查能力。

# # Input Text
# 事实句子：<input_truth>

# # Constraints & Guidelines
# 1. **多样性**：问题必须覆盖句子的不同侧面（如：时间日期、参与实体、法律依据、结果实体），问题发问需清晰明确。
# 2. **精确性**：答案必须简短、明确，直接来自该文本，不要包含多余的修饰词。
# 3. **自然性**：问题应当像人类用户自然提问的方式，避免生硬的填空式提问。
# 4. **难度分级**：主要包含下面2种难度的问题：
#    - Level 1 (基础提取)：直接询问句中显而易见的信息。
#    - Level 2 (逻辑转换)：需要简单的推理或重组信息（例如询问“哪个国家是在...之后建立的？”）。

# # Output Format
# 请严格按照以下 JSON 格式输出：

# ```json
# [
#   {
#     "question": "问题内容",
#     "answer": "标准答案",
#     "type": "问题类型 (如：时间/实体/事件)"
#   },
#   ...
# ]"""


gen_seed_qa_prompt_template = """# Role
You are a professional data annotation expert specializing in constructing the GAIA (General AI Assistant) benchmark dataset. Your task is to generate a set of high-quality "seed question-answer pairs" based on the given factual statement.

# Goal
Based solely on the input factual sentence, generate 2 to 3 questions from different perspectives along with their precise corresponding answers. These questions should test an AI assistant's capabilities in information extraction, reasoning, and fact-checking.

# Input Text
Factual statement: <input_truth>

# Constraints & Guidelines
1. **Diversity**: Questions must cover different aspects of the sentence (e.g., date, involved entities, legal basis, resulting entities) and be clear and explicit.
2. **Precision**: Answers must be brief, clear, and directly derived from the text without redundant modifiers.
3. **Naturalness**: Questions should be phrased like natural human inquiries, avoiding rigid fill-in-the-blank formats.
4. **Difficulty Levels**: The questions should primarily fall into the following two difficulty levels:
   - Level 1 (Basic Extraction): Directly ask for obvious information stated in the sentence.
   - Level 2 (Logical Transformation): Require simple reasoning or information restructuring (e.g., "Which country was established after...?").

# Output Format
Output strictly in the following JSON format:

```json
[
  {
    "question": "Question content",
    "answer": "Standard answer",
    "type": "Question type (e.g., time/entity/event)"
  },
  ...
]"""

# # 尝试回答
# answer_seed_qa_prompt_template = """# Role
# 你是一个专业的 GAIA 基准测试 AI 助手。你拥有广泛的内生知识库，能够精准回答用户的问题。

# # Task
# 回答用户提出的问题。

# # Constraints
# 1. **直接输出**：直接给出问题的最终答案。
# 2. **禁止解释**：严禁输出任何思考过程、推理步骤、前缀（如“答案是：”）或额外说明。
# 3. **精确匹配**：答案应尽可能简洁、准确（例如：如果是年份，只输出数字；如果是人名，只输出全名）。
# 4. **语言一致**：如果题目没有指明，默认使用和题目相同的语言回答

# # Question
# <input_question>

# 请直接输出你的答案，不要解释或拒答"""

answer_seed_qa_prompt_template = """# Role
You are a professional AI assistant for the GAIA benchmark. You possess extensive internal knowledge and can answer user questions accurately.

# Task
Answer the question posed by the user.

# Constraints
1. **Direct output**: Provide only the final answer to the question.
2. **No explanations**: Do not include any thought process, reasoning steps, prefixes (such as "The answer is:"), or additional descriptions.
3. **Exact match**: Answers must be as concise and accurate as possible (e.g., output only the number for a year; output the full name for a person).
4. **Language consistency**: Use the same language as the question by default unless specified otherwise.

# Question
<input_question>

Please output only your answer with no explanations or refusals."""

# multi_hop_gen_template = """你是一个专业的GAIA基准级高难度问题生成专家，专注于将简单种子问题转化为**无任何推理捷径、无实体泄漏、必须多步检索+交叉推理**的超复杂问题，难度对标GAIA Level 4~5。
# 你必须借助联网检索完成多跳知识扩展，且生成的问题**不存在任何单条线索可直接锁定核心实体**的情况。

# ## 核心约束（违反即不合格）
# 1. 【严禁任何shortcut】
#    - 禁止使用**单一标志性属性**直接指向核心实体：如「XX地区最大国家」「XX标志性首都」「唯一获得XX奖的XX人物」「知名殖民建筑群」这类可一步锁定目标的描述
#    - 单条线索必须是**泛化、无唯一指向性**的信息，仅多条线索交叉结合后，才能唯一锁定答案
# 2. 【彻底实体屏蔽】
#    完全不出现种子问题/答案中的**人名、国名、城市名、具体年份、事件名、专有制度名**，也不使用任何可直接暗示该实体的模糊代称
# 3. 【强交叉多跳】
#    至少设计**5跳及以上**的推理链，线索跨**历史+地理+科技/文化/奖项/社会+现代事件**多个领域，跳与跳之间无显性关联，必须逐跳检索后串联
# 4. 【答案唯一性】
#    所有线索组合后仅能指向种子答案，无歧义、无事实错误
# 5. 【无常识直答】
#    问题无法靠内部常识解答，必须依赖外部检索+逻辑推理，不存在「懂的人直接秒答」的空间

# ## 生成规则
# 1. 先通过联网搜索，围绕种子答案的核心实体，挖掘**4类以上无直接关联的附属事实**（冷门关联人物、小众奖项、非标志性地理特征、同期次要历史事件、现代衍生文化/科研信息等）
# 2. 将这些事实拆解为独立泛化线索，打乱时序与逻辑关联，用间接表述串联
# 3. 最终问题只保留线索，不暴露任何推理路径，不指向种子问题的原始场景
# 4. 输出仅包含复杂化后的问题文本，无解释、无答案、无多余内容
# 5. 若线索涉及多语言/地域，无需额外指定回答语言，保证答案与种子完全一致

# ## 合格示例（无shortcut版，对标你要的效果）
# ❌ 错误（含shortcut）：加勒比海地区面积最大的岛国→直接指向古巴
# ✅ 正确（无shortcut）：
# 该国所属群岛被归入大西洋西部热带海域岛链，官方通用语源自伊比利亚半岛古国；1960年代后长期处于西方经贸限制框架下，21世纪后有本土生物化学领域学者获得联合国教科文组织面向全球女性科研工作者的专项奖项；该国曾发生一场由中层军事群体发起的政治变革，变革主导者在执政数十年后将权力移交至亲亲属，该进程深刻影响了美洲中部及南部的地缘格局。请问这场政治变革的核心发起者全名是什么？

# → 推理逻辑：
# 1. 大西洋西部热带岛链+伊比利亚半岛通用语+经贸限制→范围缩小（多个拉美国家）
# 2. 生物化学女性学者获UNESCO女科学家奖+军事变革+亲属接班→**交叉锁定唯一目标**
# 3. 无任何单条线索可直接猜国家/人物，无shortcut

# 现在，请根据以下原始问题和答案，严格遵循上述规则，生成无shortcut、强交叉多跳的复杂问题：
# 原始问题为：<seed_q>
# 原始问题的答案为：<seed_a>
# """

multi_hop_gen_template = """You are an expert question generator specializing in high-difficulty GAIA-benchmark-level problems. Your task is to transform simple seed questions into highly complex questions that **contain no reasoning shortcuts, no entity leakage, and require multi-step retrieval plus cross-domain inference**, with difficulty matching GAIA Level 4–5.
You must expand knowledge across multiple hops via web search, and the generated question **must not allow any single clue to directly identify the core entity**.

## Core Constraints (Violation = Invalid Output)
1. 【Strictly No Shortcuts】
   - Prohibit using **single distinctive attributes** to directly point to the core entity: e.g., "the largest country in region X", "the iconic capital of Y", "the only person to win award Z", "famous colonial architectural complex" — any description that can lock the target in one step.
   - Each individual clue must be **generalized and non-unique**; the target can only be uniquely identified by combining multiple clues.
2. 【Complete Entity Masking】
   Do not include any **person names, country names, city names, specific years, event names, or proper institutional names** from the seed question/answer. Also avoid any vague aliases that directly hint at the entity.
3. 【Strong Cross-Domain Multi-Hop】
   Design a reasoning chain of **at least 5 hops**. Clues must span multiple domains: **history + geography + technology/culture/awards/society + modern events**. Hops must have no explicit logical connection and require sequential retrieval and chaining.
4. 【Unique Answer】
   The combination of all clues must point only to the seed answer, with no ambiguity or factual errors.
5. 【No Commonsense Direct Answer】
   The question cannot be answered using internal knowledge alone; it must rely on external retrieval + logical reasoning, with no room for instant answers by those familiar with basic facts.

## Generation Rules
1. First, via web search, mine **more than 4 unrelated auxiliary facts** around the core entity of the seed answer (e.g., obscure associated figures, niche awards, non-iconic geographic features, secondary contemporary historical events, modern derivative cultural/scientific research information).
2. Decompose these facts into independent generalized clues, disrupt their chronological and logical order, and connect them using indirect phrasing.
3. The final question only retains clues, without exposing any reasoning path or referencing the original scenario of the seed question.
4. Output only the text of the complexified question — no explanations, no answers, no extra content.
5. Your final output language should be English.

## Qualified Example (Shortcut-Free Version, Matching Your Target Standard)
❌ Incorrect (contains shortcut): The largest island country in the Caribbean → directly points to Cuba
✅ Correct (shortcut-free):
This country belongs to an archipelago classified as a tropical island chain in the western Atlantic Ocean, and its official lingua franca originates from an ancient Iberian state. Since the 1960s, it has long been under Western economic and trade restrictions; in the 21st century, a local scholar in biochemistry received a special UNESCO award for female researchers worldwide. A political change was initiated by mid-ranking military groups in this country, whose leader transferred power to a close relative after decades in office — a process that profoundly shaped the geopolitical landscape of Central and South America. What is the full name of the core initiator of this political change?

→ Reasoning Logic:
1. Western Atlantic tropical island chain + Iberian lingua franca + trade restrictions → narrow scope (multiple Latin American countries)
2. UNESCO female scientist award in biochemistry + military-led change + power transfer to relative → **cross-verified unique target**
3. No single clue can directly guess the country/person; no shortcuts

Now, based on the following original question and answer, strictly follow the above rules to generate a shortcut-free, strong cross-domain multi-hop complex question:
Original question: <seed_q>
Answer to the original question: <seed_a>
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

file_list = os.listdir('./wiki_daily_save/')
file_list_zh = ['维基百科英文_历史上的今天_July.jsonl']
# file_list_en = [x for x in file_list if x.startswith('维基百科英文')]

######## 第一步 ########## 读取历史上的今天文件

#file_list_zh = ['维基百科中文_历史上的今天_1月.jsonl']
##### 中文维基 ##### 历史上的今天

for idx in range(len(file_list_zh)):
    wiki_daily_zh = pd.read_json('./wiki_daily_save/'+file_list_zh[idx],lines=True)
    for days in range(len(wiki_daily_zh)):
        for event in wiki_daily_zh.loc[days,'events']:
            try:
                tmp_title_and_url = []
                for link in event['links']:
                    tmp_title_and_url.append({'title':link['title'], 'href':link['url']})
                generated_seed_q = []
                generated_seed_a = []
                generated_seed_type = []
                gen_with_model_list = []
                gen_direct_answer_prob_list = []
                gen_direct_answer_list = []
                gen_seed_qa_prompt = gen_seed_qa_prompt_template.replace('<input_truth>', event['text'])
                gen_response = get_model_output(gen_seed_qa_prompt,used_model='qwen3-max')
                parsed_response = json.loads(gen_response.replace('```json','').replace('```',''))
                for item in parsed_response:
                    generated_seed_q.append(item['question'])
                    generated_seed_a.append(item['answer'])
                    generated_seed_type.append(item['type'])
                    gen_with_model_list.append('qwen3-max')
                    try:
                        answer_seed_prompt = answer_seed_qa_prompt_template.replace('<input_question>', item['question'])
                        answer_seed_response = get_model_output_with_logprobs(answer_seed_prompt)
                        gen_direct_answer_prob_list.append(answer_seed_response['logprobs'])
                        gen_direct_answer_list.append(answer_seed_response['content'])
                    except Exception as e:
                        gen_direct_answer_prob_list.append(1)
                        gen_direct_answer_list.append('模型调用失败')
                one_line_dict = {}
                one_line_dict['date'] = wiki_daily_zh.loc[days,'date']
                one_line_dict['seed_truth'] = event['text']
                one_line_dict['seed_title_and_url'] = tmp_title_and_url
                one_line_dict['generated_seed_q'] = generated_seed_q
                one_line_dict['generated_seed_a'] = generated_seed_a
                one_line_dict['gen_direct_answer_list'] = gen_direct_answer_list
                one_line_dict['gen_direct_answer_prob_list'] = gen_direct_answer_prob_list
                one_line_dict['generated_seed_type'] = generated_seed_type
                one_line_dict['gen_with_model_list'] = gen_with_model_list
                # json dump写入jsonl
                with open('seed_source_question_and_answer_en_7月max.jsonl', 'a') as f:
                    f.write(json.dumps(one_line_dict, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error processing event: {e}")
                continue

###### 第二步  问题复杂化
df_seed_zh = pd.read_json('seed_source_question_and_answer_en_7月max.jsonl',lines=True)
df_seed_zh['complicated_question_q'] = ''
df_seed_zh['complicated_gen_with_model_list'] = ''
df_seed_zh['complicated_try_answer_list'] = ''
df_seed_zh['complicated_try_answer_prob_list'] = ''
for i in range(len(df_seed_zh)):
    try:
        complicated_question_list = ['' for _ in range(len(df_seed_zh.loc[i,'generated_seed_q']))]
        complicated_model_list = ['' for _ in range(len(df_seed_zh.loc[i,'generated_seed_q']))]
        complicated_try_answer_list = ['' for _ in range(len(df_seed_zh.loc[i,'generated_seed_q']))]
        complicated_try_answer_prob_list = ['' for _ in range(len(df_seed_zh.loc[i,'generated_seed_q']))]
        for j in range(len(df_seed_zh.loc[i,'generated_seed_q'])):
            complecated_question_prompt = multi_hop_gen_template.replace('<seed_q>', df_seed_zh.loc[i,'generated_seed_q'][j]).replace('<seed_a>', df_seed_zh.loc[i,'generated_seed_a'][j])
            try:
                if random.random()<0.2:
                    used_model = 'qwen3.6-plus-2026-04-02'
                    if random.random()<0.7:
                        complecated_question_response = get_model_output_with_search(complecated_question_prompt,used_model=used_model,search_strategy="max")
                    else:
                        if random.random()<0.8:
                            complecated_question_response = get_model_output_with_agentic_search(complecated_question_prompt,used_model=used_model,search_strategy="max")
                        else:
                            complecated_question_response = get_model_output_with_agentic_search(complecated_question_prompt,used_model=used_model,search_strategy="agent")
                else:
                    used_model = 'qwen3-max'
                    if random.random()<0.7:
                        complecated_question_response = get_model_output_with_search(complecated_question_prompt,used_model=used_model,search_strategy="max")
                    else:
                        if random.random()<0.8:
                            complecated_question_response = get_model_output_with_agentic_search(complecated_question_prompt,used_model=used_model,search_strategy="max")
                        else:
                            complecated_question_response = get_model_output_with_agentic_search(complecated_question_prompt,used_model=used_model,search_strategy="agent")
            except Exception as e:
                print(f"Error processing question {j} in row {i}: {e}")
            try:
                complicated_question_list[j] = complecated_question_response
                complicated_model_list[j] = used_model
                answer_seed_prompt = answer_seed_qa_prompt_template.replace('<input_question>', complecated_question_response)
                answer_seed_response = get_model_output_with_logprobs(answer_seed_prompt)
                complicated_try_answer_list[j] = answer_seed_response['content']
                complicated_try_answer_prob_list[j] = answer_seed_response['logprobs']
            except Exception as e:
                print(f"Error processing question {j} in row {i}: {e}")
                complicated_try_answer_list[j] = '模型调用失败'
                complicated_try_answer_prob_list[j] = 1
                continue
        df_seed_zh.at[i,'complicated_question_q'] = complicated_question_list
        df_seed_zh.at[i,'complicated_gen_with_model_list'] = complicated_model_list
        df_seed_zh.at[i,'complicated_try_answer_list'] = complicated_try_answer_list
        df_seed_zh.at[i,'complicated_try_answer_prob_list'] = complicated_try_answer_prob_list
        with open('complicated_source_question_and_answer_en_7月max.jsonl', 'a') as f:
            f.write(json.dumps(df_seed_zh.loc[i].to_dict(), ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        continue




