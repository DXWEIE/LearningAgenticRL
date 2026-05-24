data_classification_prompt_template = """你是旅游用户请求分类专家，请严格根据用户提问内容，划分到下面4个固定类别之一，**只输出标签，不要解释、不要多余文字**。

分类规则：
1. Direction：包含指定途经点的行车/步行/公共交通路线规划，狭义的行程规划，例如“从尚云寺到灵岩禅寺，再到光明禅寺，怎么走最方便”
2. trip plan：单城市一日游规划、多日长途旅行整体行程规划等
3. Transportation compare：单纯搜索或对比飞机、火车、大巴等出行交通方式、耗时、价格等，例如“澳门到南京明天坐飞机和高铁大概多少钱”
4. search poi：单纯搜索某个位置周边的景点、餐厅、酒店、游玩地点等兴趣点，例如“请问在墨尔本市天津代表处附近有没有广汽本田的维修点”

输入用户请求：
{query}

只允许从下面标签里选一个输出，不要有任何解释：
Direction、trip plan、Transportation compare、search poi"""

import os
from openai import OpenAI
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque


class RateLimiter:
    """Simple thread-safe fixed-window rate limiter allowing up to `qps` acquires per second."""
    def __init__(self, qps: int):
        self.qps = max(1, int(qps))
        self.lock = threading.Lock()
        # store timestamps (float seconds) of recent acquisitions
        self.times = deque()

    def acquire(self):
        """Block until a slot is available, then record timestamp and return."""
        while True:
            now = time.time()
            with self.lock:
                # drop timestamps older than 1 second
                while self.times and now - self.times[0] >= 1.0:
                    self.times.popleft()

                if len(self.times) < self.qps:
                    # allow and record
                    self.times.append(now)
                    return

            # not allowed yet; sleep a short while before retrying
            time.sleep(0.05)

QWEN_KEY = 'sk-xxx'

def call_openai(text_in, used_model="qwen-flash", max_tries=5):
    client = OpenAI(
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    for attempt in range(max_tries):
        try:
            # Throttle across threads to respect QPS limits.
            # rate_limiter is a global initialized below (after max_workers is set).
            try:
                rate_limiter.acquire()
            except NameError:
                # If rate_limiter isn't defined, continue without throttling.
                pass
            chat_response = client.chat.completions.create(
                model=used_model,
                messages=[{"role": "user", "content": text_in}],
                temperature=0,
                top_p=0.95,
                timeout=600
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

# Read dataset
all_train = pd.read_json("./dataset/train.jsonl", lines=True)

# Use 2 concurrent workers to call the LLM
max_workers = 2
futures_map = {}
# Throttle to at most `max_workers` requests per second across threads.
rate_limiter = RateLimiter(qps=max_workers)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in range(len(all_train)):
        query = all_train.loc[i, 'query']
        prompt = data_classification_prompt_template.format(query=query)
        fut = executor.submit(call_openai, prompt)
        futures_map[fut] = query
        if (i + 1) % 10 == 0:
            print(f"Submitted {i + 1} tasks")

    # As each future completes, write the result to the output file
    with open("./dataset/filtered_train.jsonl", "a", encoding="utf-8") as f:
        for fut in as_completed(futures_map):
            query = futures_map[fut]
            try:
                label = fut.result()
            except Exception as e:
                label = None
                print(f"Error getting label for query '{query}': {e}")
            # Write even if label is None to keep alignment
            f.write(json.dumps({"query": query, "class": label}, ensure_ascii=False) + "\n")
