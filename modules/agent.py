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
    """搜索旅游目的地、景点、餐厅、酒店等信息。返回地点名称、地址、类型、距离和电话。"""
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
def search_restaurants(query: str, city: str) -> str:
    """搜索餐厅信息，包括名称、地址、评分和人均价格。使用高德地图搜索。"""
    from modules.amap_mcp import get_amap_service

    amap = get_amap_service()
    if not amap.api_key:
        return "高德地图 API Key 未配置。"

    try:
        result = amap.search_place(f"{query} 餐厅", city=city)
        pois = result.get("pois", [])

        if not pois:
            return f"未找到与 '{query}' 相关的餐厅。"

        output = [f"找到 {len(pois)} 家餐厅:\n"]
        for poi in pois[:10]:
            output.append(amap.format_place_info(poi))

        return "\n".join(output)
    except Exception as e:
        return f"搜索餐厅时出错: {str(e)}"

@tool
def search_hotels(query: str, city: str) -> str:
    """搜索酒店信息，包括名称、地址、星级、价格和评分。使用高德地图搜索。"""
    from modules.amap_mcp import get_amap_service

    amap = get_amap_service()
    if not amap.api_key:
        return "高德地图 API Key 未配置。"

    try:
        result = amap.search_place(f"{query} 酒店", city=city)
        pois = result.get("pois", [])

        if not pois:
            return f"未找到与 '{query}' 相关的酒店。"

        output = [f"找到 {len(pois)} 家酒店:\n"]
        for poi in pois[:10]:
            output.append(amap.format_place_info(poi))

        return "\n".join(output)
    except Exception as e:
        return f"搜索酒店时出错: {str(e)}"

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
def calculate_trip_budget(budget_items: str) -> str:
    """根据提供的费用明细（如门票、餐饮、住宿等）计算总预算。"""
    import re
    prices = re.findall(r'(\d+(?:\.\d+)?)', budget_items)
    if not prices:
        return "未能解析预算信息，请提供数字格式的费用明细。"

    total = sum(float(p) for p in prices)
    return f"计算出的总预算为: {total:.2f} 元。\n费用明细: {budget_items}"

@tool
def batch_search(query: str, city: str, types: List[str]) -> str:
    """批量搜索景点、餐厅、酒店等多个类型，一次性返回结果。适用于需要同时获取多种类型地点的场景。"""
    import json
    from modules.amap_mcp import get_amap_service

    amap = get_amap_service()
    if not amap.api_key:
        return "高德地图 API Key 未配置。"

    try:
        results = {}
        for search_type in types:
            result = amap.search_place(f"{query} {search_type}", city=city)
            pois = result.get("pois", [])
            results[search_type] = {
                "count": len(pois),
                "places": [amap.format_place_info(poi) for poi in pois[:5]]
            }

        output = [f"批量搜索 '{query}' 的结果:\n"]
        for stype, data in results.items():
            output.append(f"\n【{stype}】共找到 {data['count']} 个:")
            output.extend(data["places"][:3])

        return "\n".join(output)
    except Exception as e:
        return f"批量搜索时出错: {str(e)}"

@tool
def retrieve_attractions_knowledge(query: str, city: Optional[str] = None) -> str:
    """从本地知识库检索相关景点信息，包括景点介绍、评分、开放时间等。用于增强景点推荐的准确性。"""
    from modules.vector_store import get_vector_store

    try:
        vector_store = get_vector_store()
        results = vector_store.search_attractions(query=query, city=city, limit=5)

        if not results:
            return f"知识库中未找到与 '{query}' 相关的景点信息。"

        output = [f"【知识库检索结果】找到 {len(results)} 条相关信息:\n"]
        for i, r in enumerate(results, 1):
            output.append(f"\n{i}. {r['content']}")

        return "\n".join(output)
    except Exception as e:
        return f"知识库检索时出错: {str(e)}"

@tool
def save_trip_preference(
    destination: str,
    days: int,
    budget: int,
    preferences: str,
    selected_places: str
) -> str:
    """保存用户的行程偏好到本地知识库，包括目的地、天数、预算、偏好类型和用户选择的景点。用于下次规划时参考。"""
    from modules.vector_store import get_vector_store
    import json

    try:
        vector_store = get_vector_store()

        prefs_list = [p.strip() for p in preferences.split(",") if p.strip()]

        places_data = json.loads(selected_places) if selected_places else []

        doc_id = vector_store.add_user_preference(
            user_id="default_user",
            destination=destination,
            days=days,
            budget=budget,
            preferences=prefs_list,
            selected_places=places_data
        )

        return f"行程偏好已保存到知识库 (ID: {doc_id})。"
    except json.JSONDecodeError:
        return "景点数据格式错误，未能保存偏好。"
    except Exception as e:
        return f"保存偏好时出错: {str(e)}"

@tool
def get_user_history_preferences(destination: str) -> str:
    """检索用户的历史行程偏好，用于个性化推荐。返回用户之前选择过的景点和偏好设置。"""
    from modules.vector_store import get_vector_store

    try:
        vector_store = get_vector_store()
        results = vector_store.get_user_preferences(user_id="default_user", limit=5)

        filtered = [r for r in results if destination in r.get("destination", "")]

        if not filtered:
            return f"未找到用户前往 {destination} 的历史偏好。"

        output = [f"【用户历史偏好】\n"]
        for i, r in enumerate(filtered, 1):
            output.append(f"\n{i}. 目的地: {r['destination']}")
            output.append(f"   天数: {r['days']}天 | 预算: {r['budget']}元")
            output.append(f"   偏好: {', '.join(r['preferences'])}")

        return "\n".join(output)
    except Exception as e:
        return f"检索偏好时出错: {str(e)}"

class TravelAgent:
    def __init__(self, model_name: str = "qwen3.5-plus"):
        if not _check_api_key():
            self.agent = None
            return

        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.tools = [
            search_destinations, search_restaurants, search_hotels,
            get_nearby_places, plan_route, geocode_address, calculate_trip_budget,
            batch_search, retrieve_attractions_knowledge, save_trip_preference,
            get_user_history_preferences
        ]

        self.system_prompt = """你是一个专业的旅游规划助手。你的任务是根据用户的需求生成详细、个性化的旅游行程方案。

【数据来源与工具使用规则】（严格遵守）
所有数据查询均使用【高德地图 MCP 服务】：
- search_destinations - 搜索景点、餐厅、酒店
- search_restaurants - 搜索餐厅
- search_hotels - 搜索酒店
- get_nearby_places - 查询周边地点
- geocode_address - 获取地址经纬度
- plan_route - 规划路线
- batch_search - 批量搜索多个类型（景点、餐厅、酒店等）
- retrieve_attractions_knowledge - 从知识库检索景点信息
- save_trip_preference - 保存用户行程偏好到知识库
- get_user_history_preferences - 获取用户历史偏好

【RAG 增强规则】（重要）
在规划行程时，应优先：
1. 使用 retrieve_attractions_knowledge 查询知识库中的景点详情
2. 使用 get_user_history_preferences 查询用户历史偏好
3. 结合高德地图实时数据和知识库内容提供更精准的推荐

【输出格式规范 - 必须严格遵守】

所有行程输出必须遵循以下格式，使用 Markdown 渲染：

## 🗺️ 行程概览
- **目的地**: [城市名]
- **出行日期**: XXXX年XX月XX日 - XXXX年XX月XX日（共X天）
- **总预算**: 约XXXX元

---

## 📅 Day 1 - XXXX年XX月XX日（星期X）

### 🕘 上午
| 项目 | 详情 |
|------|------|
| 景点 | [景点名称](https://www.amap.com/search?query=景点名称&city=城市名) |
| 开放时间 | XX:XX - XX:XX |
| 门票 | XX元 |
| 建议时长 | X小时 |

### 🍜 午餐推荐
| 序号 | 餐厅名称 | 人均 | 地址 | 推荐理由 |
|------|----------|------|------|----------|
| 1 | [餐厅名](https://www.amap.com/search?query=餐厅名&city=城市名) | XX元 | 地址 | 理由 |
| ... | ... | ... | ... | ... |

### 🕑 下午
（同上午格式）

### 🍜 晚餐推荐
| 序号 | 餐厅名称 | 人均 | 地址 | 推荐理由 |
|------|----------|------|------|----------|
| 1 | [餐厅名](https://www.amap.com/search?query=餐厅名&city=城市名) | XX元 | 地址 | 理由 |
| ... | ... | ... | ... | ... |

### 🏨 Day 1 酒店推荐
| 序号 | 酒店名称 | 星级 | 价格 | 评分 | 地址 |
|------|----------|------|------|------|------|
| 1 | [酒店名](https://www.amap.com/search?query=酒店名&city=城市名) | X星 | XX元/晚 | X分 | 地址 |
| ... | ... | ... | ... | ... | ... |

---

【Few-shot 示例】

示例输入：我想去北京玩3天，预算3000元

示例输出：
## 🗺️ 行程概览
- **目的地**: 北京
- **出行日期**: 2026年4月21日 - 2026年4月23日（共3天）
- **总预算**: 约3000元

---

## 📅 Day 1 - 2026年4月21日（星期一）

### 🕘 上午
| 项目 | 详情 |
|------|------|
| 景点 | [故宫](https://www.amap.com/search?query=故宫&city=北京) |
| 开放时间 | 08:30 - 17:00 |
| 门票 | 60元 |
| 建议时长 | 3小时 |

### 🍜 午餐推荐
| 序号 | 餐厅名称 | 人均 | 地址 | 推荐理由 |
|------|----------|------|------|----------|
| 1 | [全聚德烤鸭店](https://www.amap.com/search?query=全聚德烤鸭店&city=北京) | 200元 | 前门大街30号 | 经典北京烤鸭 |
| 2 | [东来顺](https://www.amap.com/search?query=东来顺&city=北京) | 150元 | 金宝街8号 | 老字号涮羊肉 |

### 🕑 下午
| 项目 | 详情 |
|------|------|
| 景点 | [天安门广场](https://www.amap.com/search?query=天安门广场&city=北京) |
| 开放时间 | 全天开放 |
| 门票 | 免费 |
| 建议时长 | 1小时 |

### 🍜 晚餐推荐
| 序号 | 餐厅名称 | 人均 | 地址 | 推荐理由 |
|------|----------|------|------|----------|
| 1 | [南门涮肉](https://www.amap.com/search?query=南门涮肉&city=北京) | 120元 | 东城区 | 地道铜锅涮肉 |

### 🏨 Day 1 酒店推荐
| 序号 | 酒店名称 | 星级 | 价格 | 评分 | 地址 |
|------|----------|------|------|------|------|
| 1 | [北京饭店](https://www.amap.com/search?query=北京饭店&city=北京) | 5星 | 800元/晚 | 4.6分 | 东长安街33号 |

---

【链接格式规范 - 所有链接均使用高德地图】
- 景点: [名称](https://www.amap.com/search?query=名称&city=城市名)
- 餐厅: [名称](https://www.amap.com/search?query=名称&city=城市名)
- 酒店: [名称](https://www.amap.com/search?query=名称&city=城市名)

【回复要求】
- 语气亲切，专业
- 行程具备逻辑性，考虑景点地理位置和开放时间
- 必须包含具体预算清单
- 所有地点名称必须是可点击的高德地图导航链接
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
