
from datetime import datetime, timedelta
import re
import httpx
import json
import time
import random
"""
获取环境变量中的 API 密钥, 用于调用高德地图 API
环境变量名为: AMAP_MAPS_API_KEY, 在客户端侧通过配置环境变量进行设置传入
获取方式请参考: https://lbs.amap.com/api/webservice/create-project-and-key
API 文档: https://lbs.amap.com/api/webservice/summary
"""

def json2md(json_block: dict, depth: int = 1, htag: str = "#") -> str:
    def parseJSON(json_block, depth):
        if isinstance(json_block, dict):
            parseDict(json_block, depth)
        if isinstance(json_block, list):
            parseList(json_block, depth)

    def parseDict(d, depth):
        for k in d:
            if isinstance(d[k], (dict, list)):
                addHeader(k, depth)
                parseJSON(d[k], depth + 1)
            else:
                addValue(k, d[k])

        nonlocal markdown
        markdown += "\n"

    def parseList(l, depth):
        for i, value in enumerate(l):
            addHeader(str(i + 1), depth)

            if not isinstance(value, (dict, list)):
                index = l.index(value)
                addValue(index, value)
            else:
                parseDict(value, depth)

        nonlocal markdown
        markdown += "\n"

    def buildHeaderChain(depth, title):
        chain = "\n" + htag * (depth + 1) + f" {title}\n\n"
        return chain

    def buildValueChain(key, value):
        chain = str(key) + f": {value}\n"
        return chain

    def addHeader(value, depth):
        chain = buildHeaderChain(depth, value.title())
        nonlocal markdown
        markdown += chain

    def addValue(key, value):
        chain = buildValueChain(key, value)
        nonlocal markdown
        markdown += chain

    markdown = ""
    parseJSON(json_block, depth)
    return markdown.strip()

def truncate_text(text: str, max_len: int = 5000) -> str:
    if len(text) <= max_len:
        return text

    head_len = max_len // 2
    tail_len = max_len // 2

    head_part = text[:head_len]
    head_matches = list(re.finditer(r"\s", head_part))
    if head_matches:
        head_end_index = head_matches[-1].start()
    else:
        head_end_index = head_len
    head = text[:head_end_index]

    tail_part = text[-tail_len:]
    tail_match = re.search(r"\s", tail_part)
    if tail_match:
        tail_start_index_in_part = tail_match.start()
        tail_start_index = len(text) - tail_len + tail_start_index_in_part
        tail = text[tail_start_index:].lstrip()
    else:
        tail = tail_part

    truncated_chars = len(text) - len(head) - len(tail)
    ellipsis = f"\n\n... [内容已截断，共省略 {truncated_chars} 字符] ...\n\n"

    return head + ellipsis + tail



def reverse_geocode(location: str):
    time.sleep(random.uniform(0, 12))
    url = "https://restapi.amap.com/v3/geocode/regeo"
    AMAP_MAPS_API_KEY=""
    params = {"key": AMAP_MAPS_API_KEY, "location": location}

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


def get_citycode(location: str):
    result = reverse_geocode(location)

    try:
        citycode = result["regeocode"]["addressComponent"]["citycode"]
    except Exception:
        citycode = None

    return citycode


# def poi_search(address: str, region: str | None = None) -> str:
#     """
#     通过文本搜索地点信息。文本可以是结构化地址，例如：北京市朝阳区望京阜荣街10号；也可以是 POI 名称，例如：首开广场。
#     返回多个可能相关的 POI 信息，包括：
#         - 详细地址，
#         - 经纬度（location 字段，经度和纬度用","分割，经度在前，纬度在后），
#         - 商业信息（Business 字段）。
#     地址结构越完整，返回的结果越准确。

#     Args:
#         address (`str`): 需要被检索的地点文本信息。只支持一个地址，文本总长度不可超过 80 字符。
#             推荐使用标准的结构化地址信息，如北京市海淀区上地十街十号。地址结构越完整，解析精度越高。
#         region (`Optional[str]`): 增加指定区域内数据召回权重，仅支持城市级别和中文，如“北京市”。
#             默认为 None，表示在全国范围内搜索。
#     """
#     time.sleep(random.uniform(0, 12))
#     AMAP_MAPS_API_KEY="xxx"
#     url = "https://restapi.amap.com/v5/place/text"
#     params = {
#         "key": AMAP_MAPS_API_KEY,
#         "keywords": address,
#         "show_fields": "business",
#     }
#     if region:
#         params["region"] = region

#     with httpx.Client() as client:
#         response = client.get(url, params=params)
#         response.raise_for_status()
#         result = response.json()

#     if result.get("status") != "1":
#         msg = result.get("info", "unknown error")
#         raise Exception(f"API response error: {msg}")

#     pois = result.get("pois")
#     if not pois:
#         raise Exception("No POI data available.")

#     return truncate_text(json2md(pois))

def poi_search(address: str, region: str | None = None) -> str:
    """
    通过文本搜索地点信息。文本可以是结构化地址，例如：北京市朝阳区望京阜荣街10号；也可以是 POI 名称，例如：首开广场。
    返回多个可能相关的 POI 信息，包括：
        - 详细地址，
        - 经纬度（location 字段，经度和纬度用","分割，经度在前，纬度在后），
        - 商业信息（Business 字段）。
    地址结构越完整，返回的结果越准确。

    Args:
        address (`str`): 需要被检索的地点文本信息。只支持一个地址，文本总长度不可超过 80 字符。
            推荐使用标准的结构化地址信息，如北京市海淀区上地十街十号。地址结构越完整，解析精度越高。
        region (`Optional[str]`): 增加指定区域内数据召回权重，仅支持城市级别和中文，如“北京市”。
            默认为 None，表示在全国范围内搜索。
    """
    url = "https://restapi.amap.com/v5/place/text"
    params = {
        "key": 'xx',
        "keywords": address,
        "offset": '20',
        "page": '1',
        "show_fields": "business"
    }
    if region:
        params["city"] = region

    url = "https://mapapi.cc/v5/place/text"


    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    pois = result.get("pois")
    if not pois:
        raise Exception("No POI data available.")

    return truncate_text(json2md(pois))


# def around_search(
#     location: str,
#     radius: int = 5000,
#     keyword: str | None = None,
#     region: str | None = None,
# ) -> str:
#     """
#     通过设置圆心和半径，搜索圆形区域内的地点信息。可通过 keyword 设定POI类型或限定返回结果，如“银行”。
#     返回多个可能相关的 POI 信息，包括：
#         - 详细地址，
#         - 经纬度（location 字段，经度和纬度用","分割，经度在前，纬度在后），
#         - 商业信息（Business 字段）。

#     Args:
#         location (`str`): 圆形区域检索的中心点坐标，不支持多个点。经度和纬度用","分割，经度在前，纬度在后，经纬度小数点后不得超过6位
#         radius (`int`): 圆形区域的搜索半径，取值范围:0-50000，大于50000时按默认值，单位：米。
#         keyword (`str`): 需要被检索的地点文本信息。只支持一个关键字，如“银行”。
#         region (`Optional[str]`): 增加指定区域内数据召回权重，仅支持城市级别和中文，如“北京市”。
#             默认为 None，表示在全国范围内搜索。
#     """
#     time.sleep(random.uniform(0, 12))
#     AMAP_MAPS_API_KEY="xxx"
#     url = "https://restapi.amap.com/v5/place/around"
#     params = {
#         "key": AMAP_MAPS_API_KEY,
#         "location": location,
#         "radius": radius,
#         "show_fields": "business",
#     }
#     if keyword:
#         params["keywords"] = keyword
#     if region:
#         params["region"] = region

#     with httpx.Client() as client:
#         response = client.get(url, params=params)
#         response.raise_for_status()
#         result = response.json()

#     if result.get("status") != "1":
#         msg = result.get("info", "unknown error")
#         raise Exception(f"API response error: {msg}")

#     pois = result.get("pois")
#     if not pois:
#         raise Exception("No POI data available.")

#     return truncate_text(json2md(pois))


def around_search(
    location: str,
    radius: int = 5000,
    keyword: str | None = None,
    region: str | None = None,
) -> str:
    """
    通过设置圆心和半径，搜索圆形区域内的地点信息。可通过 keyword 设定POI类型或限定返回结果，如“银行”。
    返回多个可能相关的 POI 信息，包括：
        - 详细地址，
        - 经纬度（location 字段，经度和纬度用","分割，经度在前，纬度在后），
        - 商业信息（Business 字段）。

    Args:
        location (`str`): 圆形区域检索的中心点坐标，不支持多个点。经度和纬度用","分割，经度在前，纬度在后，经纬度小数点后不得超过6位
        radius (`int`): 圆形区域的搜索半径，取值范围:0-50000，大于50000时按默认值，单位：米。
        keyword (`str`): 需要被检索的地点文本信息。只支持一个关键字，如“银行”。
        region (`Optional[str]`): 增加指定区域内数据召回权重，仅支持城市级别和中文，如“北京市”。
            默认为 None，表示在全国范围内搜索。
    """
    #time.sleep(random.uniform(0, 12))
    params = {
        "key": 'xx',
        "location": location,
        "radius": radius,
        "offset": '20',
        "page": '1',
    }
    url = "https://mapapi.cc/v5/place/around"

    if keyword:
        params["keywords"] = keyword
    if region:
        params["region"] = region

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    pois = result.get("pois")
    if not pois:
        return "No POI data available. Need to increase the search radius."
        #raise Exception("No POI data available.")

    return truncate_text(json2md(pois))


def driving_direction(
    origin: str, destination: str, waypoints: str | None = None
):
    AMAP_MAPS_API_KEY="xx"
    url = "https://restapi.amap.com/v5/direction/driving?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    if waypoints:
        params["waypoints"] = waypoints

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


def walking_direction(origin: str, destination: str):
    AMAP_MAPS_API_KEY="xx"
    url = "https://restapi.amap.com/v5/direction/walking?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


def bicycling_direction(origin: str, destination: str):
    AMAP_MAPS_API_KEY="xx"
    url = "https://restapi.amap.com/v5/direction/bicycling?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


def electrobike_direction(origin: str, destination: str):
    AMAP_MAPS_API_KEY="xx"
    url = "https://restapi.amap.com/v5/direction/electrobike?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


def transit_direction(origin: str, destination: str):
    AMAP_MAPS_API_KEY="xx"
    url = "https://restapi.amap.com/v5/direction/transit/integrated?parameters"

    citycode_origin = get_citycode(origin)
    citycode_destination = get_citycode(destination)

    if not citycode_origin:
        raise Exception("City not found for transit origin.")

    if not citycode_destination:
        raise Exception("City not found for transit destination.")

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
        "city1": citycode_origin,
        "city2": citycode_destination,
    }

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    return result


def direction(
    origin: str, destination: str, mode: str = "driving", waypoints: str | None = None
) -> str:
    """
    提供多种路线规划服务。支持驾车、步行、骑行、电动车、公交路线规划。

    Args:
        origin: 起点信息坐标。经度在前，纬度在后，经度和纬度用","分割，经纬度小数点后不得超过6位。
        destination: 目的地信息坐标。经度在前，纬度在后，经度和纬度用","分割，经纬度小数点后不得超过6位。
        mode: 路线规划类型，默认为驾车路线规划。
            - Enum: ["driving", "walking", "bicycling", "electrobike", "transit"]。
        waypoints: 途经点。经度和纬度用","分割，经度在前，纬度在后，小数点后不超过6位，坐标点之间用";"分隔。
            - 最大数目：16个坐标点。
    """
    time.sleep(random.uniform(0, 12))
    if mode == "driving":
        result = driving_direction(origin, destination, waypoints=waypoints)
    elif mode == "walking":
        result = walking_direction(origin, destination)
    elif mode == "bicycling":
        result = bicycling_direction(origin, destination)
    elif mode == "electrobike":
        result = electrobike_direction(origin, destination)
    elif mode == "transit":
        result = transit_direction(origin, destination)

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    route = result.get("route")
    if not route:
        raise Exception("No route available.")

    return truncate_text(json2md(route))


def weather(city: str) -> str:
    """
    根据城市名称查询指定城市的天气

    Args:
        city (`str`): 城市名称
    """
    AMAP_MAPS_API_KEY="xx"
    time.sleep(random.uniform(0, 12))
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": AMAP_MAPS_API_KEY,
        "city": city,
        "extensions": "all",
    }

    with httpx.Client() as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        result = response.json()

    if result.get("status") != "1":
        msg = result.get("info", "unknown error")
        raise Exception(f"API response error: {msg}")

    forecasts = result.get("forecasts")
    if not forecasts:
        raise Exception("No forecast data available.")

    def format_cast(idx, cast):
        # idx=0表示第一天，以此类推
        # 需要返回日期和周几
        current_date = (datetime.now() + timedelta(days=idx)).strftime("%Y-%m-%d")
        current_weekday = (datetime.now() + timedelta(days=idx)).strftime("%A")
        return {
            "date": current_date,
            "week": current_weekday,
            "dayweather": cast["dayweather"],
            "nightweather": cast["nightweather"],
            "daytemp": cast["daytemp"],
            "nighttemp": cast["nighttemp"],
            "daywind": cast["daywind"],
            "nightwind": cast["nightwind"],
            "daypower": cast["daypower"],
            "nightpower": cast["nightpower"],
        }

    def format_forecast(forecast):
        return {
            "city": forecast["city"],
            "province": forecast["province"],
            "casts": [format_cast(idx,cast) for idx, cast in enumerate(forecast["casts"])],
        }

    forecasts = [format_forecast(forecast) for forecast in forecasts]
    return truncate_text(json2md(forecasts))


# if __name__ == "__main__":
#     # Simple synchronous smoke tests for each function.
#     # These are lightweight and wrapped in try/except so the script can run
#     # without failing on network errors. Adjust inputs as needed.

#     def safe_call(name, func, *args, **kwargs):
#         print(f"--- {name} ---")
#         try:
#             res = func(*args, **kwargs)
#             try:
#                 out = json.dumps(res, ensure_ascii=False)
#             except Exception:
#                 out = str(res)
#             print(truncate_text(out, max_len=1000))
#         except Exception as e:
#             print(f"Error: {e}")
#         print()

#     # sample coordinates (Beijing)
#     origin = "116.481488,39.990464"
#     destination = "116.466,39.920"

    # safe_call("reverse_geocode", reverse_geocode, origin)
    # safe_call("get_citycode", get_citycode, origin)
    # safe_call("poi_search (故宫)", poi_search, "故宫", region="北京市")
    # safe_call("around_search (餐饮)", around_search, origin, 1000, keyword="餐饮")
    # safe_call("driving_direction", driving_direction, origin, destination)
    # safe_call("walking_direction", walking_direction, origin, destination)
    # safe_call("bicycling_direction", bicycling_direction, origin, destination)
    # safe_call("electrobike_direction", electrobike_direction, origin, destination)
    # safe_call("transit_direction", transit_direction, origin, destination)
    # safe_call("direction (driving)", direction, origin, destination, "driving")
    #safe_call("weather (北京)", weather, "北京")

