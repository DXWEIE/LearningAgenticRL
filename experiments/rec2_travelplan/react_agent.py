import json
import os
from typing import Dict, List, Optional, Union
from pyparsing import Any
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from summary_utils import summarize_conversation
import re
import time
from tool_search_old import *
from tool_amap import *
from tool_mock_transport import *


MAX_LLM_CALL_PER_RUN = 40
print(f'Running with MAX_LLM_CALL_PER_RUN = {MAX_LLM_CALL_PER_RUN}')
RESUM = os.getenv('RESUM', 'True').lower() == 'true'
print(f'ReSum Mode: {RESUM}')
MAX_CONTEXT = 32
print(f"Maximum Context: {MAX_CONTEXT}k")
QWEN_KEY = 'sk-xxx'

LOG_FILE_PATH = "llm_call_history_traval.jsonl"


TOOL_REGISTRY: Dict[str, Any] = {
    "web_search": web_search,
    "search_flights": search_flights,
    "search_train_tickets": search_train_tickets,
    "search_weather": weather,
    "search_navigation": direction,
    "search_poi": poi_search,
    "search_around": around_search
}

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]

    def call_server(self, msgs, max_tries=10):
        # client = OpenAI(
        #     api_key=QWEN_KEY, 
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     timeout=120, 
        #     max_retries=10
        # )
        client = OpenAI(
            api_key="EMPTY",
            base_url="https://6164w5yt3411mr6makea100.funhpc.com:30499/v1",
        )
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model="qwen3_4b",
                    messages=msgs,
                    #stop=["\n<tool_responseF", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    extra_body={"lora_name": "grpo-lora"}
                )
                # chat_response = client.chat.completions.create(
                #     model=self.model,
                #     messages=msgs,
                #     stop=["\n<tool_responseF", "<tool_response>"],
                #     temperature=self.llm_generate_cfg.get('temperature', 0),
                #     top_p=self.llm_generate_cfg.get('top_p', 0.95),
                # )
                content = chat_response.choices[0].message.content
                if content:
                    log_data = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "model": self.model,
                        "question": self.question,
                        "answer": self.answer,
                        "input_messages": msgs,
                        "output_response": content
                    }
                    # 追加模式写入jsonl，每行一条数据
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    return content
            except Exception as e:
                time.sleep(2)
                if attempt == (max_tries - 1):
                    print(f"SGLang server error {e}")
                    return f"SGLang server error"
                continue
        return "SGLang server empty response"

    def count_tokens(self, messages, model="gpt-4o") -> int:
        """
        快速估算 messages 的 token 数量（适用于中英文混合）
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):  # 多模态消息（如带图片描述）
                for part in content:
                    if part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
        
        # 启发式估算：中文字符权重更高
        # 统计中文字符数量（Unicode 范围）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', ''.join(
            msg.get("content", "") if isinstance(msg.get("content"), str) else ""
            for msg in messages
        )))
        
        # 中文每个字 ≈ 1 token，英文每 4 字符 ≈ 1 token
        english_like_chars = total_chars - chinese_chars
        estimated = chinese_chars + max(0, english_like_chars // 4)
        
        # 加上系统/角色等结构开销（每条消息约 8-10 tokens）
        estimated += len(messages) * 8
        
        return max(1, estimated)

    def _run(self, data: str, model: str, summary_iteration: int, **kwargs) -> List[List[Message]]:
        if_summary = 0
        self.model = model
        question = data['item']['question']
        answer = ''# data['item']['answer'] # 没有answer的
        self.question = question
        self.answer = answer

        messages = [
            {"role": "system", "content": self.system_message}, 
            {"role": "user", "content": question}
        ]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        last_summary, full_trajectory = None, messages.copy() 
        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages) # 是用message来调用
            print(f'round {round}: {content}')

            if '<response>' in content:
                pos = content.find('<response>')
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})
            full_trajectory.append({"role": "assistant", "content": content.strip()}) 

            if '<tool>' in content and '</tool>' in content:
                tool_call = content.split('<tool>')[1].split('</tool>')[0]
                try:
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    result = TOOL_REGISTRY[tool_name](**tool_args)
                    #result = self._call_tool(tool_name, tool_args)
                    print(f"Tool call {tool_name} invocation success with length {len(result)}")
                except Exception as e:
                    print(f"Tool call error: {e}")
                    result = 'Error: Tool call is not valid. Tool call must contain a valid "name" and "arguments" field.'
                result = "<response>" + result + "</response>"
                messages.append({"role": "user", "content": result})
                full_trajectory.append({"role": "user", "content": result}) 

            elif '<answer>' in content and '</answer>' in content:
                answer_content = content.split('<answer>')[1].split('</answer>')[0].strip()
                if len(answer_content):
                    termination = 'answer'
                    prediction = answer_content 
                    break
      
            max_tokens = MAX_CONTEXT * 1024 - 1000
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            should_summarize = ((RESUM and token_count >= max_tokens * 0.9) or round % summary_iteration == 0) and num_llm_calls_available 
            if should_summarize: 
                if_summary = 1
                recent_messages = messages[2:].copy() 

                try:
                    summary_response = summarize_conversation(question, recent_messages, last_summary)
                    print(f"[Summary Tool] ReSum Invocation success (len: {len(summary_response)}): {summary_response}")
                except Exception as e: 
                    print(f"[Summary Tool] ReSum Invocation failed: {e}")
                    summary_response = "" 
                
                if summary_response:  
                    last_summary = summary_response  
                    new_observation = "Question: " + question + "\nBelow is a summary of the previous conversation. This summary condenses key information from earlier steps, so please consider it carefully. Assess whether the summary provides enough information to answer the question and use it as the basis for further reasoning and information gathering to answer the question.\n" \
                                     + "Summary: " + summary_response + "\n"
                    messages = [
                        {"role": "system", "content": self.system_message}, 
                        {"role": "user", "content": new_observation}
                    ] # 后面用message来调用llm的
                    full_trajectory.append({"role": "user", "content": new_observation})  
                    token_count = self.count_tokens(messages) 
                    print(f"round {round}, token count after summary: {token_count}")
            
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            if token_count > max_tokens:
                print(f"Token count exceeds limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop invoking tools and, based on all the information above, think again and provide what you consider the most likely answer in the following format: <reason> your final thinking </reason>\n <answer> your answer </answer>"
                content = self.call_server(messages)

                messages.append({"role": "assistant", "content": content.strip()})
                full_trajectory.append({"role": "assistant", "content": content.strip()}) 

                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "rollout_id": data['rollout_id'],
                    "messages": full_trajectory,
                    "prediction": prediction,
                    "termination": termination,
                    "if_summary": if_summary
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "messages": full_trajectory,
            "prediction": prediction,
            "termination": termination,
            "if_summary": if_summary
        }
        return result