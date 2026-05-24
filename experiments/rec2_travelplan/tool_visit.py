import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT 
import os 
from openai import OpenAI
import random
import re
from bs4 import BeautifulSoup
from tool_search_old import *

WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000)) # 最大的网页内容长度，是len不是token
IGNORE_JINA = os.getenv("IGNORE_JINA", "false").lower() == "true"
# Visit Tool (Using Jina Reader)
JINA_READER_URL_PREFIX = "https://r.jina.ai/"
JINA_API_KEY = "xxx"

QWEN_KEY = 'sk-xxx'
IQS_KEY ="xxx"
search_scan_key = "vcans_xxx"

#visit_model  = "qwen-plus-2025-09-11"
visit_model  = "qwen-flash"

def preprocess_url(url):
    """
    处理特定的 URL 转换逻辑。
    目前支持：将 arXiv 的 PDF 链接自动转换为 abs (摘要) 链接。
    """
    # arXiv PDF 链接转 abs 链接
    # 匹配模式: https://arxiv.org/pdf/2301.12345.pdf 或 https://arxiv.org/pdf/2301.12345
    arxiv_pdf_pattern = r'(?:https?://)?arxiv\.org/pdf/([0-9]+\.[0-9]+)(?:\.pdf)?$'
    match = re.search(arxiv_pdf_pattern, url)
    if match:
        arxiv_id = match.group(1)
        # 转换为 abs 链接
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        return abs_url
    
    # 如果不是 arXiv PDF 链接，返回原链接
    return url



def get_webpage_content(url, if_text_only=True, timeout=10):
    global web_cache
    """
    静态爬虫获取网页内容（不依赖浏览器渲染）
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }
    try:
        response = requests.get(url, headers=headers, timeout=20, verify=False)
        response.encoding = response.apparent_encoding
        html = response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除不需要的标签
        excluded_tags = ["nav", "footer", "aside", "header", "script", "style", "iframe", "meta"]
        for tag in excluded_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # 提取标题
        title = soup.title.string if soup.title else ""
        
        # 提取正文内容
        if if_text_only:
            # 只提取文本
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
                text = element.get_text(strip=True)
                if len(text) > 5:  # 过滤太短的文本
                    text_parts.append(text)
            webpage_text = '\n\n'.join(text_parts)
        else:
            # 保留链接和结构
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'a']):
                text = element.get_text(strip=True)
                if len(text) > 5:
                    text_parts.append(text)
            webpage_text = '\n\n'.join(text_parts)
        
        # 清理文本
        cleaned_text = webpage_text.replace("undefined", "") if webpage_text else ""
        cleaned_text = re.sub(r'(\n\s*){3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'[\r\t]', '', cleaned_text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        cleaned_text = re.sub(r'^\s+|\s+$', '', cleaned_text, flags=re.MULTILINE)
        
        # 模拟返回结构（兼容原有代码）
        result = {
            'markdown': type('obj', (object,), {'fit_markdown': cleaned_text})(),
            'url': url,
            'title': title,
            'status_code': response.status_code
        }

        return result, cleaned_text.strip()
    
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return None, None
    
def fetch_url_content(url: str):
    global search_cache
    try:
        cache_result = search_cache.get(url)
        if cache_result:
            return cache_result
        else:
            return "[visit] Failed to read page because the page is a PDF file."
    except Exception as e:
        print(f"[fetch_url_content]Error fetching webpage content: {e}")
        return "[visit] Failed to read page."
        
@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
                },
            "minItems": 1,
            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
      },
      "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
      }
    },
    "required": ["url", "goal"]
  }
    # The `call` method is the main function of the tool.
    def call(self, **kwargs) -> str:
        try:
            url = kwargs["url"]
            goal = kwargs["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        if isinstance(url, str):
            response = self.readpage(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in url}
                for future in as_completed(futures):
                    try:
                        response.append(future.result())
                    except Exception as e:
                        response.append(f"Error fetching {futures[future]}: {str(e)}")
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()
    
    # def call_server(self, msgs, max_tries=10):
    #     # 设置 OpenAI 的 API 密钥和 API 基础 URL 使用 vLLM 的 API 服务器。
    #     openai_api_key = "EMPTY"
    #     openai_api_base = "http://127.0.0.1:6002/v1"

    #     client = OpenAI(
    #         api_key=openai_api_key,
    #         base_url=openai_api_base,
    #     )
    #     for attempt in range(max_tries):
    #         try:
    #             chat_response = client.chat.completions.create(
    #                 model='/path/qwen2.5-instruct-72b',
    #                 messages=msgs,
    #                 stop=["\n<tool_response>", "<tool_response>"],
    #                 temperature=0.7
    #             )
    #             content = chat_response.choices[0].message.content
    #             if content:
    #                 try:
    #                     json.loads(content)
    #                 except:
    #                     # extract json from string 
    #                     left = content.find('{')
    #                     right = content.rfind('}') 
    #                     if left != -1 and right != -1 and left <= right: 
    #                         content = content[left:right+1]
    #                 return content
    #         except:
    #             if attempt == (max_tries - 1):
    #                 return ""
    #             continue
    
    
    def call_server(self, msgs, max_tries=10):
        # 设置 OpenAI 的 API 密钥和 API 基础 URL 使用 vLLM 的 API 服务器。
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key=QWEN_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=visit_model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except:
                if attempt == (max_tries - 1):
                    return ""
                continue

    # def jina_readpage(self, url: str) -> str:
    #     """
    #     Read webpage content using Jina service.
        
    #     Args:
    #         url: The URL to read
    #         goal: The goal/purpose of reading the page
            
    #     Returns:
    #         str: The webpage content or error message
    #     """
    #     headers = {
    #         "Authorization": f"Bearer {JINA_API_KEY}",
    #     }
    #     max_retries = 3
    #     timeout = 10
        
    #     for attempt in range(max_retries):
    #         try:
    #             response = requests.get(
    #                 f"https://r.jina.ai/{url}",
    #                 headers=headers,
    #                 timeout=timeout
    #             )
    #             if response.status_code == 200:
    #                 webpage_content = response.text
    #                 return webpage_content
    #             else:
    #                 print(response.text)
    #                 raise ValueError("jina readpage error")
    #         except Exception as e:
    #             if attempt == max_retries - 1:
    #                 return "[visit] Failed to read page."
                
    #     return "[visit] Failed to read page."
    


    def jina_read_page(target_url,timeout=20000,enbable_browser=True):
        url = "https://www.searchcans.com/api/url"
        headers = {
            "Authorization": f"Bearer {search_scan_key}",
            "Content-Type": "application/json"
        }
        data = {
            "s": target_url,
            "t": "url",
            "w": 3000,   # Wait 3s for JavaScript rendering
            "d": timeout,  # 20s timeout for complex pages
            "b": enbable_browser    # Enable browser mode (recommended)
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            result = response.json()
            if result.get('code') == 0:
                return result.get('data', {}).get('markdown', '')
            else:
                return "[visit] Failed to read page."
        except Exception as e:
            print(f"Error jina fetching {url}: {e}")
            return "[visit] Failed to read page."



    def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_attempts = 2
        for attempt in range(max_attempts):
            # Alternate between jina and aidata
            content = fetch_url_content(url)
            sevice = "jina"

            # Check if we got valid content
            print(sevice)
            # print(content)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                # 这里是直接截断了，可以修改一下。。。不然太费token了
                content = content[:WEBCONTENT_MAXLENGTH]
                messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
                parse_retry_times = 0
                raw = self.call_server(messages) # 调用回答子问题的接口，拿到结果

                # 如果网页超长，返回结果是 {\n 这种形式
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    print(status_msg)
                    content = content[:truncate_length] # 超长的直接截断！，截断3次
                    extraction_prompt = EXTRACTOR_PROMPT.format(
                        webpage_content=content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = self.call_server(messages)
                    summary_retries -= 1
                # 说明 raw 的长度大于10或者已经retry 超出了 
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        # 尝试 parse json, "rational", "evidence", "summary"这3个field
                        raw = json.loads(raw)
                        break
                    except:
                        raw = self.call_server(messages)
                        parse_retry_times += 1
                # parse 失败
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                # parse 成功
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                    useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

                    summary_retries -= 1

                if len(useful_information) < 10 and summary_retries < 0:
                    print("[visit] Could not generate valid summary after maximum retries")
                    useful_information = "[visit] Failed to read page"
                return useful_information
                
            # If we're on the last attempt, return the last result
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information

# if __name__=='__main__':
#     tool = Visit()
#     tool.call({
#         "url": "https://example.com",
#         "goal": "Extract key information"
#     })