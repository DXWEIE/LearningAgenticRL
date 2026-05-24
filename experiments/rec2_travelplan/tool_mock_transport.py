import asyncio
from datetime import datetime, timedelta
import os
import json
import logging

from openai import OpenAI

model = "qwen-plus-2025-09-11"
QWEN_KEY = 'sk-xxxx'

transport_client = OpenAI(
    api_key=QWEN_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=100, 
    max_retries=10
)

def search_flights(date: str, from_city: str, to_city: str) -> str:
    """
    date: YYYY-MM-DD
    from_city: 出发城市中文名
    to_city: 到达城市中文名
    """

    system_prompt = """角色设定
你是一名“航班查询结果模拟专家”，能够根据用户给出的日期、出发城市与到达城市，生成覆盖全天主要时段的机票信息（6–14 条）。所有信息均为模拟数据，但必须符合以下“真实性规则”。

输入格式
用户将以 JSON 形式输入：
{
"date": "YYYY-MM-DD",
"from_city": "出发城市中文名",
"to_city": "到达城市中文名"
}

输出格式
• 以 JSON 数组形式返回，每一条为一段中文字符串；
• 每条字符串遵循：
"航班 {航司代码+航班号}，价格{票价}元，{起飞时刻}从{出发机场}出发，{到达时刻}到达{到达机场}，飞行时长{X小时Y分}"
• 举例：
"航班 CA1847，价格763.0元，09:05从首都国际机场出发，12:25到达浦东国际机场，飞行时长3小时20分"

真实性规则

航司与航班号
• 航司代码：两位大写英文字母（常见：CA/MU/CZ/HU/HO/3U/GF/EK/AF 等）；
• 航班号：3–4 位数字。
机场
• 国内：使用城市主要机场（可带“国际／白塔／天府／首都／虹桥／禄口”等）；
• 国际：如有跨国城市，可使用国际机场（例：Heathrow、Changi、Narita 等）。
时间
• 出发时间覆盖 05:00–23:00，各航班间隔合理；
• 到达时间 = 出发时间 + 合理飞行时长（国内 1–4 小时，国际 2–15 小时）。
• 禁止返回的所有航班的飞行耗时都一样，需要有差异性，符合真实情况
价格
• 国内：200–1500 元波动；
• 国际：800–8000 元波动；
• 同一日期票价从低到高大致递增但可随机。
条数
• 返回 10–15 条航班信息；
• 建议按起飞时间顺序排列，便于用户阅读。
语气
• 仅返回机票数组；不添加任何解释、换行、符号或多余信息。
示例交互
用户输入：
{"date":"2025-07-25","from_city":"呼和浩特市","to_city":"成都市"}

模型输出：
[
"航班 8L9672，价格745.0元，11:00从白塔国际机场出发，13:35到达天府机场，飞行时长2小时35分",
"航班 CA8147，价格763.0元，09:05从白塔国际机场出发，12:00到达天府机场，飞行时长2小时55分",
...
"航班 CA8131，价格965.0元，16:30从白塔国际机场出发，19:15到达天府机场，飞行时长3小时25分"
]
"""

    kwargs = {"date": date, "from_city": from_city, "to_city": to_city}
    query = json.dumps(kwargs, ensure_ascii=False)
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = transport_client.chat.completions.create(
            messages=messages, model=model
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        #logger.error(f"[search_flights] Failed to get response: {e}")
        result = ""

    if not result:
        raise ValueError("两地无航班信息")

    return result


def search_train_tickets(
    date: str,
    from_city: str,
    to_city: str
) -> str:
    """
    date: 查询日期(格式 yyyy-MM-dd)
    from_city / to_city: 中文城市名
    from_city_adcode / to_city_adcode: 行政区划代码
    from_lat、from_lon、to_lat、to_lon: 两地经纬度
    """

    system_prompt = """请扮演“火车票查询结果模拟器”。

输入是一段 JSON，字段包括：
• date：查询日期（格式 yyyy-MM-dd）
• from_city / to_city：中文城市名
任务：基于输入信息，输出 10-15 条该日期“{from_city}→{to_city}”的直达列车信息，覆盖凌晨、上午、下午、傍晚、夜间等大部分时段。
输出格式要求：
• 类型：JSON 数组，每个元素为一条车次信息字符串。
• 字符串内容模板：
“直达车次 {TrainNo}，价格{Price}元，{DepTime}从{DepStation}出发，{ArrTime}到达{ArrStation}，全程约{Duration}。”
• 关键值规范：
TrainNo：在 G / D / Z / K / T / Y / C 等字母+数字中随机选取，避免重复；
Price：综合里程与车种随机生成，动车/高铁 150-600 元，普速 60-300 元，硬卧可 100-420 元（仅普速时可给三档价位），车票价格根据两地距离而定；
DepTime / ArrTime：24h 制，确保 ArrTime ≥ DepTime，合理计算 Duration（四舍五入到分钟）；
DepStation / ArrStation：
• 如果城市内存在多个常见客运站（如“郑州”“郑州东”“郑州西”等），随机挑选符合列车类型的站名；
• 北/南/东/西/站字样请符合真实火车站命名习惯；
• Duration：按实际时间差给出“X时Y分”。
逻辑与随机性：
• 按常见列车运行规律生成时刻表，不要出现荒诞时间（如 03:00-03:20 只跑 20 分钟的普速）。
• 避免完全均匀分布，可略集中在早高峰 (06-09)、午后 (12-15)、晚高峰 (17-21) 等。
其他：
• 不输出与需求无关的文字、解释或注释，仅返回符合格式的 JSON 数组。
• 所有结果仅为模拟数据，非真实票务信息。
"""

    kwargs = {
        "date": date,
        "from_city": from_city,
        "to_city": to_city
    }
    query = json.dumps(kwargs, ensure_ascii=False)
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    # 计算日期时间差是否超过15天
    if (datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days > 15:
        result = "搜索火车票信息的日期超过15天，请选择一个更近的日期，通常车次和价格不会发生变化。"
        return result

    try:
        response = transport_client.chat.completions.create(
                messages=messages, model=model
            )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        #logger.error(f"[search_train_tickets] Failed to get response: {e}")
        result = ""

    if not result:
        raise ValueError("两地无直达火车票")

    return result


# if __name__ == "__main__":
#     # Simple manual tests for the two functions above.
#     # To avoid making real network calls set environment variable MOCK_REMOTE=1
#     mock = 0

#     # use tomorrow's date by default for queries
#     test_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     print(f"TEST MODE: {'MOCK' if mock else 'REAL'}\n")

#     # # Test search_flights
#     # try:
#     #     print("=== search_flights ===")
#     #     if mock:
#     #         sample = [
#     #             "航班 CA1001，价格500.0元，09:00从首都国际机场出发，11:30到达天府机场，飞行时长2小时30分",
#     #             "航班 MU2002，价格720.0元，13:20从首都国际机场出发，15:50到达天府机场，飞行时长2小时30分",
#     #         ]
#     #         print(json.dumps(sample, ensure_ascii=False, indent=2))
#     #     else:
#     #         out = search_flights(test_date, "北京", "成都")
#     #         print(out)
#     # except Exception as e:
#     #     print(f"search_flights raised: {e}")

#     # Test search_train_tickets
#     try:
#         print("\n=== search_train_tickets ===")
#         if mock:
#             sample_trains = [
#                 "直达车次 G101，价格350元，07:00从北京南出发，12:30到达成都东，全程约5时30分。",
#                 "直达车次 D202，价格420元，09:20从北京南出发，14:50到达成都东，全程约5时30分。",
#             ]
#             print(json.dumps(sample_trains, ensure_ascii=False, indent=2))
#         else:
#             # pass empty strings for adcodes/coords if not available
#             out = search_train_tickets(
#                 test_date,
#                 "北京",
#                 "成都",
#             )
#             print(out)
#     except Exception as e:
#         print(f"search_train_tickets raised: {e}")

