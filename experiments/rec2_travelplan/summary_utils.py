import json 
import requests 
from prompt import QUERY_SUMMARY_PROMPT, QUERY_SUMMARY_PROMPT_LAST
import os 
import re
import time 
from openai import OpenAI

QWEN_KEY = 'sk-xxx'
LOG_FILE_PATH = "llm_call_history_gaia_dft.jsonl"

def call_resum_server(query, used_model='qwen-plus-2025-09-11', max_retries=10):
        client = OpenAI(
            api_key=QWEN_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=used_model,
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    temperature=0,
                    top_p=0.95,
                )
                content = chat_response.choices[0].message.content
                if content: 
                    pattern = r'<reason>.*?</reason>' 
                    content = re.sub(pattern, '', content, flags=re.DOTALL).strip()
                    try:
                        content = content.split("<summary>")[1].split("</summary>")[0]
                    except:
                        content = content 

                    # ===================== 新增：Dump日志（append模式）=====================
                    log_data = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "model": used_model,
                        "question": 'sum',
                        "answer": 'sum',
                        "input_messages": [{"role": "user", "content": query}],
                        "output_response": content
                    }
                    # 追加模式写入jsonl，每行一条数据
                    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                        
                    return "<summary>" + content + "</summary>"
                else: 
                    return ""
            except Exception as e:
                time.sleep(2)
                if attempt == (max_retries - 1):
                    print(f"SGLang server error {e}")
                    
                    return ""
                continue
        return ""


def summarize_conversation(question, recent_history_messages, last_summary, max_retries=10):
    recent_history_str = "\n".join([str(msg) for msg in recent_history_messages])
    
    if not last_summary:
        query_prompt = QUERY_SUMMARY_PROMPT.replace("{{{question}}}", question).replace("{{{recent_history_messages}}}", recent_history_str)
    else:
        query_prompt = QUERY_SUMMARY_PROMPT_LAST.replace("{{{question}}}", question).replace("{{{recent_history_messages}}}", recent_history_str).replace("{{{last_summary}}}", last_summary)
    
    response = call_resum_server(query_prompt, max_retries=max_retries)
    return response


###### Test Code ###### 
if __name__ == "__main__": 
    query = "Please give me a simple three-day travel plan for Kitakyushu, Japan."
    print(call_resum_server(query))
    print(summarize_conversation(query, [], ""))