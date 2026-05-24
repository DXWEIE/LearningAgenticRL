import random
import time
from openai import OpenAI
from qwen_agent.tools.base import BaseTool, register_tool 
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import os
from ddgs import DDGS
import re

# SEARCH_API_URL = os.getenv("SEARCH_API_URL")
# GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY")

IQS_KEY ="xxx"
search_scan_key = "xxxxxx"

def normalize_title(title: str) -> str:
    """
    标准化标题：
    1. 转小写 (针对英文)
    2. 去除所有标点符号、特殊字符、空格、换行
    3. 只保留字母、数字、汉字
    这样 "成龙电影_2024!" 和 "成龙电影 2024" 都会变成 "成龙电影2024"
    """
    if not title:
        return ""
    
    # 转小写
    title = title.lower()
    
    # 正则替换：保留 字母(a-z)、数字(0-9)、汉字(\u4e00-\u9fff)，其他全部删掉
    # \u4e00-\u9fff 覆盖常用汉字
    cleaned = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', title)
    
    return cleaned

def deduplicate_by_url_and_title(results: list) -> list:
    """
    根据 URL 和 标准化后的 Title 对搜索结果列表去重。
    策略：
    1. 跳过百度知道 (zhidao.baidu.com)
    2. URL 完全相同 -> 去重
    3. Title 标准化后完全相同 -> 去重 (解决同文不同链问题)
    """
    seen_urls = set()
    seen_titles = set() # 存储标准化后的 title
    unique_results = []
    
    for item in results:
        url = item.get('url', '')
        title = item.get('title', '')
        
        # 1. 过滤百度知道 (根据你的需求)
        if "zhidao.baidu.com" in url:
            continue
            
        # 2. 处理缺失 URL 的情况 (保留，防止误删)
        if not url:
            # 如果没有 URL，尝试用 Title 去重，如果 Title 也没法判断，则直接保留
            norm_title = normalize_title(title)
            if norm_title and norm_title in seen_titles:
                continue # Title 重复也跳过
            if norm_title:
                seen_titles.add(norm_title)
            unique_results.append(item)
            continue
            
        # 3. URL 去重检查
        if url in seen_urls:
            continue
        
        # 4. Title 去重检查 (核心新增逻辑)
        norm_title = normalize_title(title)
        if norm_title in seen_titles:
            # 标题一样，认为是重复内容，跳过
            continue
            
        # 5. 通过检查，加入结果集
        seen_urls.add(url)
        if norm_title:
            seen_titles.add(norm_title)
        unique_results.append(item)
            
    return unique_results


import re

def count_tokens(self, text: str, model="gpt-4o") -> int:
    """
    纯启发式估算 单个文本字符串 的Token数（无第三方依赖）
    规则：中文字符/中文标点 ≈ 1 Token
         英文/数字/符号/空格 ≈ 4 字符 = 1 Token
    适用于 GPT-4o/GPT-3.5/LLaMA 等主流模型
    """
    if not text or not isinstance(text, str):
        return 1
    
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    
    # 剩余所有字符（英文、数字、标点、空格等）
    other_chars = len(text) - chinese_chars
    
    # 计算Token：中文1字1token，其余4字符1token
    token_count = chinese_chars + (other_chars // 4)
    
    # 保底最小1Token
    return max(1, token_count)

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    按启发式Token数从头截断文本
    规则：中文字符 ≈ 1 token | 英文/数字/标点/空格 ≈ 0.25 token
    :param text: 原始输入字符串
    :param max_tokens: 最大保留的token数
    :return: 截断后的字符串（从头保留）
    """
    # 边界处理
    if not text or max_tokens < 1:
        return ""
    
    current_tokens = 0.0
    truncated_chars = []

    for char in text:
        # 判断是否为中文字符
        if "\u4e00" <= char <= "\u9fff":
            token_cost = 1.0
        else:
            token_cost = 0.25

        # 超过最大token数则停止
        if current_tokens + token_cost > max_tokens:
            break
        
        truncated_chars.append(char)
        current_tokens += token_cost

    # 拼接字符返回结果
    return ''.join(truncated_chars)


def iqs_search(query,top_k=10, type="LiteAdvanced"): # Generic LiteAdvanced
    url = "https://cloud-iqs.aliyuncs.com/search/unified"

    headers = {
        "Authorization": f"Bearer {IQS_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "query": query,
        "engineType": type,
        "contents": {
            "mainText": True,
            "markdownText": True,
            "richMainBody": True,
            "summary": False,
            "rerankScore": True
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        result_json = response.json()
        return result_json['pageItems']
    except Exception as e:
        print(f"Error occurred while searching: {e}")
        return []


def get_iqs_search_result(query,top_k=10):
    try:
        iqs_response = {"results": iqs_search(query, top_k)}
    except Exception as e:
        iqs_response = {"results": []}
        print(f"Error occurred while searching: {e}")
    if len(iqs_response['results']) == 0:
        print("IQS No search results found.Error!")
        return f"No results found for '{query}'. Try with a more general query, or remove the year filter."

    search_result_list = []
    web_snippets = []
    for i in range(min(top_k, len(iqs_response['results']))):
        item = {}
        item['search_engine'] = 'iqs'
        item['query'] = query
        item['url'] = iqs_response['results'][i]['link']
        item['title'] = truncate_text_by_tokens(iqs_response['results'][i]['title'],max_tokens=80)
        item['content'] = truncate_text_by_tokens(iqs_response['results'][i]['snippet'], max_tokens=300)
        item['raw_content'] = iqs_response['results'][i]['mainText']
        search_result_list.append(item.copy())
        redacted_version = f"{i}. [{item['title']}]({item['url']})\n{item['content']}"

        redacted_version = redacted_version.replace("Your browser can't play this video.", "")
        web_snippets.append(redacted_version)

    print(f"获取到的iqs搜索结果数量: {len(search_result_list)}")
    return_content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
    return search_result_list


def search_scan(query, engine='google'):
    attempt = 0
    MAX_RETRIES = 2
    result = {}
    while attempt < MAX_RETRIES:
        url = "https://www.searchcans.com/api/search"
        headers = {
            "Authorization": f"Bearer {search_scan_key}",
            "Content-Type": "application/json"
        }
        data = {
            "s": query,
            "t": engine,  # or "bing"
            "p": 1,
            "d": 30000,  # 30s timeout (production)
            "w": 6000    # 6s wait time
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            result = response.json()
        except Exception as e:
            print(f"Error occurred while searching: {e}")
            result = {"code": -1}

        if result.get('code',-1) == 0:
            break
        else:
            attempt += 1
            print(f"search_scan retry")
            time.sleep(2 * attempt)
    return result.get('data', [])


def get_search_scan_result(query,top_k=10):
    # 为title link snippet mainText rerankScore的格式
    try:
        api_result = search_scan(query)
        if len(api_result)==0:
            print("search scan No valid search results found or API error.")
            return []
        if len(api_result)>top_k:
            api_result = api_result[:top_k]  # 只保留前top_k条结果
        google_response = {"results": api_result}
        
    except Exception as e:
        print(f"Error occurred while parsing search response: {e}")
        return f"No results found for '{query}'. Try with a more general query, or remove the year filter."

    search_result_list = []
    web_snippets = []
    for i in range(min(top_k, len(google_response['results']))):
        item = {}
        item['search_engine'] = 'google'
        item['query'] = query
        item['url'] = google_response['results'][i]['url']
        item['title'] = truncate_text_by_tokens(google_response['results'][i]['title'],max_tokens=80)
        item['content'] = truncate_text_by_tokens(google_response['results'][i]['content'],max_tokens=300)
        item['raw_content'] = ''
        search_result_list.append(item.copy())
        redacted_version = f"{i}. [{item['title']}]({item['url']})\n{item['content']}"
        web_snippets.append(redacted_version)
    print(f"获取到的google搜索结果数量: {len(search_result_list)}")
    return search_result_list


def ddgs_search(query, top_k=10,timeout=40):
    try:
        results = DDGS(timeout=timeout,verify=False).text(query, max_results=top_k, safesearch="on",region="us-en")
    except Exception as e:
        return []
    return results


# 这个需要手动调用qwen-flash来总结一下
def get_ddgs_result(query,top_k=5,timeout=40):
    try:
        ddgs_response = {"results": ddgs_search(query,top_k,timeout)}
    except Exception as e:
        ddgs_response = {"results": []}
        print(f"Error occurred while searching: {e}")
    if len(ddgs_response['results']) == 0:
        print("DDGS No search results found.Error!")
    
    search_result_list = []
    web_snippets = []
    for i in range(len(ddgs_response['results'])):
        item = {}
        item['search_engine'] = 'ddgs'
        item['query'] = query
        item['url'] = ddgs_response['results'][i]['href'] if 'href' in ddgs_response['results'][i] else ''
        item['title'] = ddgs_response['results'][i]['title'] if 'title' in ddgs_response['results'][i] else ''
        item['content'] = ddgs_response['results'][i]['body'] if 'body' in ddgs_response['results'][i] else ''
        item['raw_content'] = ''
        redacted_version = f"{i}. [{item['title']}]({item['url']})\n{item['content']}"
        web_snippets.append(redacted_version)
        search_result_list.append(item.copy())
    print(f"获取到的ddgs搜索结果数量: {len(search_result_list)}")
    #return_content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
    return search_result_list


def detect_query_language(query: str, chinese_threshold: float = 0.5) -> str:

    if not query.strip():
        return 'en'  # default fallback
    
    # Remove spaces, digits, and common punctuation
    cleaned = re.sub(r'[^\w\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', '', query)
    if not cleaned:
        return 'en'
    
    # Count Chinese characters (including CJK extensions)
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', cleaned)
    chinese_ratio = len(chinese_chars) / len(cleaned)
    
    return 'zh' if chinese_ratio >= chinese_threshold else 'en'



SEARCH_SIMULATE_PROMPT = """You act as a real Google search engine.
Generate {num_results} authentic web search results purely based only on the input search query.

Core Rules:
1. Generate real-world, reasonable and highly relevant search results matching the query intent.
2. Simulate genuine webpage titles, urls and natural snippet content.

Format Requirements:
1. Output only a strict JSON List, index starts from 0, exactly {num_results} items.
2. Each item contains 4 fixed fields: "idx", "title", "url", "snippet".
3. snippet keeps about 40 English words, simulate real webpage abstract.
4. No extra explanations, no redundant text, output pure parsable JSON list only.

Query: {query}
Search Output:
"""


def call_server(self, text_in, max_tries=10):
    client = OpenAI(
            api_key="EMPTY",
            base_url="https://0f8pgb12xq2722wmakea100.funhpc.com:30499/v1",
    )
    for attempt in range(max_tries):
        try:
            chat_response = client.chat.completions.create(
                    model="qwen3_4b",
                    messages=[{"role": "user", "content": text_in}],
                    temperature=0.3,
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    extra_body={"lora_name": "search-lora"},
                    timeout=120
                )
            content = chat_response.choices[0].message.content
            search_res = json.loads(content)
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep((attempt + 1))
            search_res = []
    return search_res

def get_simulate_google_result(query, top_k=10):
    prompt = SEARCH_SIMULATE_PROMPT.format(num_results=5, query=query)
    search_result = call_server(prompt)
    return search_result

def simulate_search(query,top_k=10):
    language = detect_query_language(query)
    google_results = []
    unique_results = get_simulate_google_result(query, top_k=5)
    if unique_results and len(unique_results)>0:
        # 重新标一下idx
        idx = 0
        web_snippets = []
        for item in unique_results:
            redacted_version = f"{idx}. [{item['title']}]({item['url']})\n{item['snippet']}"
            idx += 1
            web_snippets.append(redacted_version)
        return_content = f"A Google search for '{query}' found {len(unique_results)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        return return_content
    else:
        return f"No results found for '{query}'. Try with a more general query, or remove the year filter."

def aggregate_search(query,top_k=11):
    language = detect_query_language(query)
    iqs_results = []
    ddgs_results = []
    google_results = []
    if language == 'zh':
        iqs_results = get_iqs_search_result(query, top_k=10)
        #ddgs_results = get_ddgs_result(query, top_k=3)
        combined_results = iqs_results #+ ddgs_results
        unique_results = deduplicate_by_url_and_title(combined_results)
    else:
        google_results = get_search_scan_result(query, top_k=10)
        #ddgs_results = get_ddgs_result(query, top_k=3)
        combined_results = google_results #+ ddgs_results
        unique_results = deduplicate_by_url_and_title(combined_results)

    if unique_results and len(unique_results)>0:
        # 重新标一下idx
        idx = 0
        web_snippets = []
        for item in unique_results:
            redacted_version = f"{idx}. [{item['title']}]({item['url']})\n{item['content']}"
            idx += 1
            web_snippets.append(redacted_version)
        return_content = f"A Google search for '{query}' found {len(unique_results)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        return return_content
    else:
        return f"No results found for '{query}'. Try with a more general query, or remove the year filter."

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {
                    "type": "string"
                    },
                    "description": "Array of query strings. Include multiple complementary search queries in a single call."
                },
            },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            response = aggregate_search(query)
        else:
            assert isinstance(query, List)
            if random.random()<0.1:
                with ThreadPoolExecutor(max_workers=1) as executor: # 一次只能搜索一个，qps限流
                    response = list(executor.map(aggregate_search, query))
            else:
                with ThreadPoolExecutor(max_workers=3) as executor: # 模拟搜索没事
                    response = list(executor.map(simulate_search, query))
            response = "\n=======\n".join(response)
        return response