import pandas as pd
import numpy as np
import json

df_question_zh= pd.read_json('./eval_data/data_for_traj_4月.jsonl',lines=True)
df_traj_zh = pd.read_json('./train_traj/traj_for_4月_zh.jsonl',lines=True)

df_question_en= pd.read_json('./eval_data/data_for_traj_7月.jsonl',lines=True)
df_traj_en= pd.read_json('./train_traj/traj_for_7月_en.jsonl',lines=True)

df_question_webshaper = pd.read_json('./eval_data/web_shaper_top200.jsonl',lines=True)
df_traj_webshaper = pd.read_json('./train_traj/traj_webshaper_top200.jsonl',lines=True)

SYSTEM_PROMPT_ORI = """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following principles:

1. **Persistent Actions for Answers**: You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found.

2. **Repeated Verification**: Before presenting a Final Answer, you will **cross-check** and **validate the information** you've gathered to confirm its accuracy and reliability.

3. **Attention to Detail**: You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.

You will engage in a conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools:

<tools>
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Array of query strings. Include multiple complementary search queries in a single call."
      }
    },
    "required": [
      "query"
    ]
    }
},
{
  "name": "visit",
    "description": "Visit webpage(s) and return the summary of the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The specific information goal for visiting webpage(s)."
            }
        },
        "required": [
            "url",
            "goal"
        ]
    }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>"""


SYSTEM_PROMPT = """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions. No matter how complex the query, you will not give up until you find the corresponding information.

As you proceed, adhere to the following **NON-NEGOTIABLE RULES**:

1. **Default to Tool Use, Never Guess**
   Prefer search over your memory, unless the question is about *extremely common, universally known facts* (e.g., "What is the capital of France?", "What is 2+2?", "Is water wet?"), you **MUST NOT answer directly**. Instead, you must initiate a search or visit relevant pages.

2. **Minimum 3 Tool Interaction Rounds**
   - If your tool call steps are FEWER THAN 3, You must add one **final verification search round** to verify the information.

3. **Mandatory Encyclopedia Verification Before Answering**
   If the expected answer is:
   - a specific entity (person, place, organization, work)
   - a year, date, event, number, or factual claim
   You MUST perform a FINAL SEARCH with an explicit encyclopedia query:
   - For ENGLISH entities: include "Wikipedia" in the search query
   - For CHINESE entities: include "百度百科" in the search query
   This is to verify the answer against official, authoritative entries.

4. **Persistent & Deep Searching**
   You will engage in many interactions, delving deeply into the topic to explore all possible aspects until a satisfactory answer is found. Do NOT stop after 1–2 searches.

5. **Repeated Cross-Checking and Self-reflection**
   Before presenting a Final Answer, you will cross-check and validate the information you've gathered to confirm its accuracy and reliability.

6. **Attention to Detail & Credibility**
   You will carefully analyze each information source to ensure that all data is current, relevant, and from credible origins.
   
---
### 【STRICT RULE FOR  CONTENT — NO HALLUCINATION ALLOWED】
1. **NO FABRICATED SEARCH RESULTS IN **
   You are **FORBIDDEN** to write any "obtained information", "search results show", "found that", "learned from search" in  **BEFORE ANY TOOL CALL IS MADE** (especially Round 1).
2. **ONLY REFER TO REAL TOOL RESPONSES**
   You may only mention information in  **AFTER** you receive a real <tool_response> from a tool call. Never invent information that does not exist.  

You will engage in a conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools:

<tools>
{
  "name": "search",
  "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Array of query strings. Include multiple complementary search queries in a single call."
      }
    },
    "required": [
      "query"
    ]
    }
},
{
  "name": "visit",
    "description": "Visit webpage(s) and return the summary of the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The specific information goal for visiting webpage(s)."
            }
        },
        "required": [
            "url",
            "goal"
        ]
    }
}
</tools>

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends with (thinking about the answer -> answer of the question). The thinking processes, tool calls, tool responses, and answer are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<answer> answer here </answer>"""


cnt = 0
for i in range(len(df_traj_zh)):
    if df_traj_zh.loc[i,'messages'][0]['content']==SYSTEM_PROMPT:
        df_traj_zh.loc[i,'messages'][0]['content'] = SYSTEM_PROMPT_ORI
        cnt += 1
print(cnt)
cnt = 0
for i in range(len(df_traj_en)):
    if df_traj_en.loc[i,'messages'][0]['content']==SYSTEM_PROMPT:
        df_traj_en.loc[i,'messages'][0]['content'] = SYSTEM_PROMPT_ORI
        cnt += 1
print(cnt)
cnt = 0
for i in range(len(df_traj_webshaper)):
    if df_traj_webshaper.loc[i,'messages'][0]['content']==SYSTEM_PROMPT:
        df_traj_webshaper.loc[i,'messages'][0]['content'] = SYSTEM_PROMPT_ORI
        cnt += 1
print(cnt)

# 数据集构建
# 拒绝采样，判断答案对不对
quesiton_list = []
answer_list = []
seed_truth_list = []
termination_list = []
response_list = []
messages_list = []
for i in range(len(df_traj_zh)):
    quesiton_list.append(df_traj_zh.loc[i,'question'])
    answer_list.append(df_traj_zh.loc[i,'answer'])
    seed_truth_list.append(df_question_zh.loc[i,'seed_truth'])
    termination_list.append(df_traj_zh.loc[i,'termination'])
    response_list.append(df_traj_zh.loc[i,'prediction'])
    messages_list.append(df_traj_zh.loc[i,'messages'])

for i in range(len(df_traj_en)):
    quesiton_list.append(df_traj_en.loc[i,'question'])
    answer_list.append(df_traj_en.loc[i,'answer'])
    seed_truth_list.append(df_question_en.loc[i,'seed_truth'])
    termination_list.append(df_traj_en.loc[i,'termination'])
    response_list.append(df_traj_en.loc[i,'prediction'])
    messages_list.append(df_traj_en.loc[i,'messages'])

for i in range(len(df_traj_webshaper)):
    quesiton_list.append(df_traj_webshaper.loc[i,'question'])
    answer_list.append(df_traj_webshaper.loc[i,'answer'])
    seed_truth_list.append(df_question_webshaper.loc[i,'answer'])
    termination_list.append(df_traj_webshaper.loc[i,'termination'])
    response_list.append(df_traj_webshaper.loc[i,'prediction'])
    messages_list.append(df_traj_webshaper.loc[i,'messages'])


df_train = pd.DataFrame({'question':quesiton_list,'answer':answer_list,'seed_truth':seed_truth_list,'termination':termination_list,'response':response_list,'messages':messages_list})


JUDGE_PROMPT_BC_en = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Barack Obama's children are Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
    - For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e., just return A or B, without adding any other text.
""".strip()

from openai import OpenAI
import time

QWEN_KEY = 'sk-'
def call_openai(text_in, used_model="qwen-plus-2025-09-11", max_tries=5):
    client = OpenAI(
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    for attempt in range(max_tries):
        try:
            chat_response = client.chat.completions.create(
                model=used_model,
                messages=[{"role": "user", "content": text_in}],
                temperature=0,
                top_p=0.95,
            )
            content = chat_response.choices[0].message.content
            if content:
                return content
        except Exception as e:
            time.sleep(2)
            if attempt == (max_tries - 1):
                print(f"LLM judge server error {e}")
                return None
            continue
    return None


def judge_if_correct(question, response, answer):
    judge_prompt = JUDGE_PROMPT_BC_en.format(
            question=question,
            response=response,
            correct_answer=answer
        )
    llm_response = call_openai(judge_prompt, used_model="qwen-plus-2025-09-11")
    if llm_response is None:
        return 0
    llm_response_lower = llm_response.lower()
    if 'incorrect' in llm_response_lower or 'b' in llm_response_lower:
        return 0
    elif 'correct' in llm_response_lower or 'a' in llm_response_lower:
        return 1
    else:
        return 0
    


from concurrent.futures import ThreadPoolExecutor, as_completed

def concurrent_judge(df, max_workers=2):
    # 初始化 is_correct_judge 列（如果还不存在）
    if 'is_correct_judge' not in df.columns:
        df['is_correct_judge'] = -1
    
    # 找出尚未评估的任务索引
    pending_indices = df[df['is_correct_judge'] == -1].index.tolist()
    
    results = {}
    total = len(pending_indices)
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_idx = {
            executor.submit(
                judge_if_correct, 
                df.loc[idx, 'question'], 
                df.loc[idx, 'response'], 
                df.loc[idx, 'seed_truth']
            ): idx 
            for idx in pending_indices
        }
        
        # 遍历完成的任务
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"Index {idx} generated an exception: {e}")
                results[idx] = -1  # 标记为失败，后续可重试
            
            completed_count += 1
            if completed_count % 20 == 0:
                print(f"[{completed_count}/{total}] Finished processing idx: {idx}")

    # 将结果写回 dataframe
    for idx, res in results.items():
        df.loc[idx, 'is_correct_judge'] = res
    
    return df

# 执行并发评估
df_train['is_correct_judge'] = -1
df_train = concurrent_judge(df_train, max_workers=2)

# 保存结果
df_train.to_csv("df_train_with_judgments.csv", index=False)

# 保存为jsonl，注意使用cls
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

with open('df_train_with_judgments.jsonl', 'w', encoding='utf-8') as f:
    for _, row in df_train.iterrows():
        json_record = row.to_dict()
        f.write(json.dumps(json_record, ensure_ascii=False, cls=MyEncoder) + '\n')

# 查看统计结果
print(df_train['is_correct_judge'].value_counts())