import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

def _check_api_key():
    return os.getenv("OPENAI_API_KEY") and os.getenv("AMAP_API_KEY")

@tool
def search_destinations(query: str, city: Optional[str] = None) -> str:
    """搜索旅游目的地、景点、酒店、餐厅等信息。返回地点名称、地址、类型、距离和电话。"""
    from modules.amap_mcp import get_amap_service

    amap = get_amap_service()
    if not amap.api_key:
        return "高德地图 API Key 未配置，请检查环境变量。"

    try:
        result = amap.search_place(query, city=city)
        pois = result.get("pois", [])

        if not pois:
            return f"未找到与 '{query}' 相关的地点。"

        output = [f"找到 {result.get('count', 0)} 个结果:\n"]
        for poi in pois[:10]:
            output.append(amap.format_place_info(poi))

        return "\n".join(output)
    except Exception as e:
        return f"搜索地点时出错: {str(e)}"

@tool
def get_nearby_places(location: str, keywords: Optional[str] = None,
                      radius: int = 2000) -> str:
    """查询指定位置周围的地点，如餐厅、酒店、景点等。"""
    from modules.amap_mcp import get_amap_service

    amap = get_amap_service()
    if not amap.api_key:
        return "高德地图 API Key 未配置。"

    try:
        result = amap.get_around_places(location, keywords=keywords, radius=radius)
        pois = result.get("pois", [])

        if not pois:
            return f"在指定位置周围未找到相关地点。"

        output = [f"找到 {len(pois)} 个周边地点:\n"]
        for poi in pois[:10]:
            output.append(amap.format_place_info(poi))

        return "\n".join(output)
    except Exception as e:
        return f"查询周边地点时出错: {str(e)}"

@tool
def plan_route(origin: str, destination: str,
              mode: str = "walking") -> str:
    """规划两个地点之间的路线，支持步行、驾车、公交等出行方式。"""
    from modules.amap_mcp import get_amap_service

    amap = get_amap_service()
    if not amap.api_key:
        return "高德地图 API Key 未配置。"

    try:
        result = amap.get_route_directions(origin, destination, mode=mode)
        return amap.format_route_info(result)
    except Exception as e:
        return f"路线规划时出错: {str(e)}"

@tool
def geocode_address(address: str, city: Optional[str] = None) -> str:
    """将地址转换为经纬度坐标，用于后续的周边搜索和路线规划。"""
    from modules.amap_mcp import get_amap_service

    amap = get_amap_service()
    if not amap.api_key:
        return "高德地图 API Key 未配置。"

    try:
        result = amap.get_geocode(address, city=city)
        geocodes = result.get("geocodes", [])

        if not geocodes:
            return f"未能解析地址: {address}"

        gc = geocodes[0]
        return f"""
地址: {gc.get('province', '')}{gc.get('city', '')}{gc.get('district', '')}{gc.get('address', '')}
经纬度: {gc.get('location', 'N/A')}
"""
    except Exception as e:
        return f"地址解析时出错: {str(e)}"

@tool
def search_restaurants(keywords: str, city: str,
                       category: Optional[str] = None) -> str:
    """搜索餐厅信息，包括名称、地址、评分和人均价格。"""
    from modules.meituan_mcp import get_meituan_service

    meituan = get_meituan_service()
    if not meituan.api_key:
        return _get_fallback_restaurants(keywords, city)

    try:
        result = meituan.search_restaurants(keywords, city, category=category)
        restaurants = result.get("restaurants", [])

        if not restaurants:
            return _get_fallback_restaurants(keywords, city)

        output = [f"找到 {result.get('total', 0)} 家餐厅:\n"]
        for r in restaurants[:10]:
            output.append(meituan.format_restaurant_info(r))

        return "\n".join(output)
    except Exception as e:
        return _get_fallback_restaurants(keywords, city)

@tool
def search_hotels(city: str, check_in: str, check_out: str,
                keywords: Optional[str] = None,
                star_level: Optional[int] = None,
                max_price: Optional[int] = None) -> str:
    """搜索酒店信息，包括名称、地址、星级、价格和评分。"""
    from modules.meituan_mcp import get_meituan_service

    meituan = get_meituan_service()
    if not meituan.api_key:
        return _get_fallback_hotels(city)

    try:
        price_range = None
        if max_price:
            price_range = (0, max_price)

        result = meituan.search_hotels(city, check_in, check_out,
                                       keywords=keywords,
                                       star_level=star_level,
                                       price_range=price_range)
        hotels = result.get("hotels", [])

        if not hotels:
            return _get_fallback_hotels(city)

        output = [f"找到 {result.get('total', 0)} 家酒店:\n"]
        for h in hotels[:10]:
            output.append(meituan.format_hotel_info(h))

        return "\n".join(output)
    except Exception as e:
        return _get_fallback_hotels(city)

def _get_fallback_restaurants(keywords: str, city: str) -> str:
    """当美团 API 不可用时，返回基于常见餐饮数据的示例餐厅信息"""
    restaurants_data = {
        "北京": [
            {"name": "全聚德烤鸭店", "address": "北京市东城区前门大街30号", "rating": "4.5", "avgPrice": 200, "category": "烤鸭", "distance": 0},
            {"name": "东来顺涮羊肉", "address": "北京市东城区金宝街8号", "rating": "4.4", "avgPrice": 150, "category": "火锅", "distance": 0},
            {"name": "北京炸酱面馆", "address": "北京市西城区护国寺街52号", "rating": "4.3", "avgPrice": 50, "category": "面食", "distance": 0},
            {"name": "护国寺小吃", "address": "北京市西城区护国寺街1号", "rating": "4.2", "avgPrice": 40, "category": "小吃", "distance": 0},
            {"name": "南锣鼓巷小吃街", "address": "北京市东城区南锣鼓巷", "rating": "4.3", "avgPrice": 60, "category": "小吃", "distance": 0},
            {"name": "簋街夜市", "address": "北京市东城区簋街", "rating": "4.4", "avgPrice": 80, "category": "夜市", "distance": 0},
            {"name": "便宜坊烤鸭店", "address": "北京市崇文区崇文门外大街甲2号", "rating": "4.3", "avgPrice": 180, "category": "烤鸭", "distance": 0},
            {"name": "鸿毛饺子", "address": "北京市朝阳区三里屯太古里", "rating": "4.2", "avgPrice": 60, "category": "饺子", "distance": 0},
            {"name": "旺顺阁鱼头泡饼", "address": "北京市海淀区中关村大街27号", "rating": "4.5", "avgPrice": 120, "category": "海鲜", "distance": 0},
            {"name": "局气北京菜", "address": "北京市朝阳区朝阳大悦城", "rating": "4.4", "avgPrice": 90, "category": "京菜", "distance": 0},
        ]
    }

    city_restaurants = restaurants_data.get(city, restaurants_data["北京"])
    output = [f"找到 {len(city_restaurants)} 家餐厅（基于热门推荐）:\n"]
    for r in city_restaurants[:10]:
        output.append(f"""
餐厅名称: {r['name']}
地址: {r['address']}
评分: {r['rating']}分
人均价格: {r['avgPrice']}元
菜系: {r['category']}
""")
    return "\n".join(output)

def _get_fallback_hotels(city: str) -> str:
    """当美团 API 不可用时，返回基于常见酒店数据的示例信息"""
    hotels_data = {
        "北京": [
            {"name": "北京饭店", "address": "北京市东城区东长安街33号", "star": 5, "price": 800, "rating": "4.6", "tags": ["市中心", "天安门广场附近"], "district": "东城区"},
            {"name": "北京王府半岛酒店", "address": "北京市东城区金宝街8号", "star": 5, "price": 1500, "rating": "4.7", "tags": ["豪华", "近王府井"], "district": "东城区"},
            {"name": "北京jw万豪酒店", "address": "北京市朝阳区大望路甲88号", "star": 5, "price": 1200, "rating": "4.5", "tags": ["商务", "国贸CBD"], "district": "朝阳区"},
            {"name": "北京国际饭店", "address": "北京市东城区建国门内大街9号", "star": 4, "price": 600, "rating": "4.4", "tags": ["交通便利", "近地铁"], "district": "东城区"},
            {"name": "北京希尔顿酒店", "address": "北京市朝阳区东三环北路东方路1号", "star": 5, "price": 900, "rating": "4.5", "tags": ["国际品牌", "近农业展览馆"], "district": "朝阳区"},
            {"name": "长城饭店", "address": "北京市朝阳区东三环北路12号", "star": 5, "price": 850, "rating": "4.4", "tags": ["老牌五星", "近三里屯"], "district": "朝阳区"},
            {"name": "北京粤财jw万豪酒店", "address": "北京市西城区宣武门外大街18号", "star": 5, "price": 1000, "rating": "4.5", "tags": ["近天安门", "西城区"], "district": "西城区"},
            {"name": "北京千禧大酒店", "address": "北京市朝阳区东三环中路辅路", "star": 4, "price": 550, "rating": "4.3", "tags": ["性价比高", "双井"], "district": "朝阳区"},
            {"name": "北京香格里拉饭店", "address": "北京市海淀区紫竹院路29号", "star": 5, "price": 1100, "rating": "4.6", "tags": ["花园式", "近北京动物园"], "district": "海淀区"},
            {"name": "北京万达文华酒店", "address": "北京市朝阳区建国路93号", "star": 5, "price": 950, "rating": "4.5", "tags": ["万达集团", "近国贸"], "district": "朝阳区"},
        ]
    }

    city_hotels = hotels_data.get(city, hotels_data["北京"])
    output = [f"找到 {len(city_hotels)} 家酒店（基于热门推荐）:\n"]
    for h in city_hotels[:10]:
        output.append(f"""
酒店名称: {h['name']}
地址: {h['address']}
星级: {h['star']}星
价格: {h['price']}元/晚
评分: {h['rating']}分
区域: {h['district']}
标签: {', '.join(h['tags'])}
""")
    return "\n".join(output)

@tool
def calculate_trip_budget(budget_items: str) -> str:
    """根据提供的费用明细（如门票、餐饮、住宿等）计算总预算。"""
    import re
    prices = re.findall(r'(\d+(?:\.\d+)?)', budget_items)
    if not prices:
        return "未能解析预算信息，请提供数字格式的费用明细。"

    total = sum(float(p) for p in prices)
    return f"计算出的总预算为: {total:.2f} 元。\n费用明细: {budget_items}"

class TravelAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        if not _check_api_key():
            self.agent = None
            return

        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.tools = [
            search_destinations, get_nearby_places, plan_route,
            geocode_address, search_restaurants, search_hotels,
            calculate_trip_budget
        ]

        self.system_prompt = """你是一个专业的旅游规划助手。你的任务是根据用户的需求生成详细、个性化的旅游行程方案。

【数据来源与工具使用规则】（严格遵守）
- 路线规划、景点位置查询、导航等：仅使用【高德地图 MCP 服务】
- 餐厅搜索、酒店搜索、美食和住宿预订等：仅使用【美团 MCP 服务】
  * 使用 search_restaurants 搜索餐厅
  * 使用 search_hotels 搜索酒店

数据获取策略：
- 使用 search_destinations 搜索景点和目的地信息（高德地图）
- 使用 geocode_address 获取地址的经纬度（高德地图）
- 使用 plan_route 规划景点间的路线（高德地图）
- 使用 search_restaurants 搜索餐厅（美团）
- 使用 search_hotels 搜索酒店（美团）

【重要】输出格式要求：
1. 景点/地点名称（仅用于路线规划、导航相关）必须使用 Markdown 链接格式，格式为：
   [景点名称](https://www.amap.com/search?query=景点名称&city=城市名)
   用户点击后将在高德地图网页版打开，用于查看位置和导航。

2. 餐饮推荐（美团）必须使用 Markdown 链接格式，格式为：
   [餐厅名称](https://www.meituan.com/s/{城市名}/{餐厅名称})
   用户点击后将在美团网页版打开，用于查看详情和订餐。

3. 酒店推荐（美团）必须使用 Markdown 链接格式，格式为：
   [酒店名称](https://hotel.meituan.com/search?q=酒店名称)
   用户点击后将在美团酒店网页版打开，用于查看详情和预订。

4. 餐饮推荐要求（每日三餐，每餐5条推荐）：
   早餐/午餐/晚餐 各需列出5个推荐选项，每个选项包含：
   - 餐厅名称（带美团链接）
   - 人均价格（元）
   - 推荐理由（1句话）
   使用以下格式（每餐独立展示）：
   ### 🍜 [时间段] 餐饮推荐
   | 序号 | 餐厅名称（可点击） | 人均价格 | 推荐理由 |
   |------|----------|----------|----------|

5. 酒店推荐要求（每天5条）：
   每天列出5家推荐酒店，每家包含：
   - 酒店名称（带美团链接）
   - 价格（元/晚）
   - 星级/档次
   - 位置区域
   - 特色简介
   使用以下格式：
   ### 🏨 [Day X] 酒店推荐
   | 序号 | 酒店名称（可点击） | 价格/晚 | 星级 | 区域 | 特色 |
   |------|----------|---------|------|------|------|

6. 行程安排格式：
   ### 📅 Day X - [日期]
   **上午/下午/晚上**：时间安排（地点使用高德地图链接）

回复要求：
- 语气亲切、专业。
- 结构清晰，使用 Markdown 格式展示行程。
- 行程应具备逻辑性，考虑景点之间的地理位置和开放时间。
- 必须包含具体的预算清单。
- 景点/地点使用【高德地图】链接，餐厅使用【美团】链接，酒店使用【美团酒店】链接。
"""

        self.agent = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=self.system_prompt
        )

    def run(self, user_input: str, chat_history: List[Any] = None):
        if not self.agent:
            return "请先配置 OPENAI_API_KEY 和 AMAP_API_KEY 以启用 Agent。"

        if chat_history is None:
            chat_history = []

        messages = chat_history + [HumanMessage(content=user_input)]

        result = self.agent.invoke({"messages": messages})
        return result["messages"][-1].content

    def stream_run(self, user_input: str, chat_history: List[Any] = None):
        """流式运行 Agent，产生中间步骤和最终结果。"""
        if not self.agent:
            yield {"error": "请先配置 API Key"}
            return

        if chat_history is None:
            chat_history = []

        messages = chat_history + [HumanMessage(content=user_input)]
        
        # 使用 stream 模式获取中间步骤
        for chunk in self.agent.stream({"messages": messages}, stream_mode="values"):
            if "messages" in chunk:
                last_msg = chunk["messages"][-1]
                yield last_msg

_travel_agent: Optional[TravelAgent] = None

def get_travel_agent() -> TravelAgent:
    global _travel_agent
    if _travel_agent is None:
        _travel_agent = TravelAgent()
    return _travel_agent
