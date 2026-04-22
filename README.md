# 智能旅游行程规划系统

基于 `LangChain + LangGraph + Streamlit` 的旅行规划 Agent。
当前版本统一使用高德地图能力进行景点/餐厅/酒店检索与路线规划，并输出结构化行程方案。

***

## 目录

1. [系统架构总览](#系统架构总览)
2. [技术栈与依赖](#技术栈与依赖)
3. [LangChain 核心组件详解](#langchain-核心组件详解)
4. [LangGraph 架构设计详解](#langgraph-架构设计详解)
5. [模块职责与数据流转](#模块职责与数据流转)
6. [RAG 增强架构 (Chroma DB)](#rag-增强架构-chroma-db)
7. [快速开始](#快速开始)
8. [代码示例](#代码示例)
9. [性能优化建议](#性能优化建议)
10. [常见问题](#常见问题)

***

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit 前端                           │
│                     (app.py - 用户交互界面)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent 编排层                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    StateGraph (状态图)                    │    │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐        │    │
│  │  │   路由节点  │───▶│  执行节点  │───▶│  输出节点  │        │    │
│  │  └───────────┘    └───────────┘    └───────────┘        │    │
│  │         │              │              │                 │    │
│  │         └──────────────┴──────────────┘                 │    │
│  │                      边 (Edge)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LangChain 核心组件                           │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │LLM 调用 │  │ 工具调用 │  │  提示词  │  │  消息   │    │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       业务工具层 (Tools)                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │  地点搜索   │ │  餐厅搜索   │ │  酒店搜索   │ │  路线规划   │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │  周边搜索   │ │  地址解析   │ │  预算计算   │ │ RAG检索   │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      高德地图 MCP 服务                            │
│                   (amap_mcp.py - API 封装层)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌───────────────────────────┐  ┌───────────────────────────────────┐
│   CacheManager            │  │   Chroma DB 向量存储               │
│   (分层缓存管理)           │  │   (vector_store.py)              │
│                           │  │   - attractions 集合             │
│                           │  │   - user_preferences 集合         │
└───────────────────────────┘  └───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      基础设施层                                    │
│  ┌──────────────────────┐  ┌──────────────────────┐           │
│  │   CacheManager        │  │   ErrorHandler        │           │
│  │   (缓存管理与LRU淘汰)  │  │   (重试与错误处理)     │           │
│  └──────────────────────┘  └──────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 架构分层说明

| 层级        | 组件                   | 职责                        |
| --------- | -------------------- | ------------------------- |
| **前端层**   | Streamlit            | 用户界面、请求收集、响应渲染、对话历史管理     |
| **编排层**   | LangGraph StateGraph | Agent 流程控制、状态管理、条件分支、节点调度 |
| **核心层**   | LangChain            | LLM 调用、提示词模板、工具绑定、消息管理    |
| **工具层**   | LangChain Tools      | 业务工具定义（搜索、规划、计算）          |
| **服务层**   | AmapMCPService       | 高德地图 API 封装、参数组装、响应格式化    |
| **基础设施层** | Cache/ErrorHandler   | 缓存加速、错误重试、日志记录            |

***

## 技术栈与依赖

### 核心依赖

```txt
# requirements.txt
langchain==0.3.0+
langchain-openai==0.2.0+
langchain-community==0.3.0+
langchain-chroma==0.3.0+
langgraph==0.2.0+
chromadb==0.4.0+
streamlit==1.40+
faiss-cpu==1.8+
pandas==2.2+
python-dotenv==1.0+
tiktoken==0.7+
pydantic==2.10+
requests==2.32+
```

### 环境要求

- **Python**: 3.10+
- **模型服务**: OpenAI 兼容 API（默认使用阿里云 DashScope qwen-plus）
- **地图服务**: 高德地图 API Key
- **向量数据库**: Chroma DB (本地持久化存储)

### 版本兼容性说明

| 组件               | 最低版本  | 推荐版本  | 说明                |
| ---------------- | ----- | ----- | ----------------- |
| langchain-core   | 0.3.0 | 0.3.x | 链式调用、消息、提示词核心抽象   |
| langgraph        | 0.2.0 | 0.2.x | 状态图构建、节点边定义       |
| langchain-openai | 0.2.0 | 0.2.x | OpenAI ChatAPI 集成 |

***

## LangChain 核心组件详解

### 1. LLM 集成 (ChatOpenAI)

本系统使用 `ChatOpenAI` 作为 LLM 底层调用组件，通过 `langchain-openai` 适配器接入阿里云 DashScope 服务。

```python
# modules/agent.py
from langchain_openai import ChatOpenAI

self.llm = ChatOpenAI(
    model="qwen-plus",      # 模型名称
    temperature=0.7,         # 创造性参数 (0-1)
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

**核心概念**:

- `model`: 指定使用的模型，支持 `gpt-4`, `gpt-3.5-turbo`, `qwen-plus` 等
- `temperature`: 控制输出随机性，0.0 为确定性输出，1.0 为高随机性
- `openai_api_base`: OpenAI 兼容 API 端点地址

### 2. 提示词模板 (ChatPromptTemplate)

提示词模板定义了 Agent 的系统行为规范，包含角色定义、工具使用规则、输出格式要求。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

self.system_prompt = """你是一个专业的旅游规划助手...

【数据来源与工具使用规则】
所有数据查询均使用【高德地图 MCP 服务】：
- search_destinations - 搜索景点、餐厅、酒店
- search_restaurants - 搜索餐厅
...

【输出格式规范 - 必须严格遵守】
所有行程输出必须遵循以下格式...
"""
```

**模板组件**:

- `MessagesPlaceholder`: 动态插入对话历史消息
- 角色定义 (System/ Human/ AI): 明确各参与方职责
- Few-shot 示例: 通过示例引导模型按指定格式输出

### 3. 工具调用 (Tools)

LangChain 的 `@tool` 装饰器将 Python 函数转换为 Agent 可调用的工具。

```python
from langchain_core.tools import tool

@tool
def search_destinations(query: str, city: Optional[str] = None) -> str:
    """搜索旅游目的地、景点、餐厅、酒店等信息。"""
    # 工具实现逻辑
    result = amap.search_place(query, city=city)
    return formatted_output
```

**工具定义规范**:

- 每个工具必须有清晰的文档字符串（用于模型理解工具用途）
- 使用类型注解定义输入参数
- 返回字符串类型（便于 LLM 解析）

**本系统定义的工具**:

| 工具名称                    | 功能         | 输入参数                             |
| ----------------------- | ---------- | -------------------------------- |
| `search_destinations`   | 搜索景点/餐厅/酒店 | `query`, `city`                  |
| `search_restaurants`    | 搜索餐厅       | `query`, `city`                  |
| `search_hotels`         | 搜索酒店       | `query`, `city`                  |
| `get_nearby_places`     | 周边地点查询     | `location`, `keywords`, `radius` |
| `plan_route`            | 路线规划       | `origin`, `destination`, `mode`  |
| `geocode_address`       | 地址解析       | `address`, `city`                |
| `calculate_trip_budget`  | 预算计算       | `budget_items`                   |
| `batch_search`          | 批量搜索       | `query`, `city`, `types`          |
| `retrieve_attractions_knowledge` | RAG景点检索 | `query`, `city` |
| `save_trip_preference`   | 保存用户偏好    | `destination`, `days`, `budget`, `preferences`, `selected_places` |
| `get_user_history_preferences` | 获取历史偏好 | `destination` |

### 4. 消息管理 (Messages)

LangChain 使用统一的消息类型系统管理对话上下文。

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 用户消息
user_msg = HumanMessage(content="我想去北京玩3天")

# AI 响应
ai_msg = AIMessage(content="好的，我来为您规划...")

# 工具执行结果
tool_msg = ToolMessage(
    content="找到10个景点...",
    tool_call_id="abc123"
)
```

**消息类型说明**:

| 类型              | 说明    | 用途                  |
| --------------- | ----- | ------------------- |
| `HumanMessage`  | 用户消息  | 用户输入                |
| `AIMessage`     | AI 消息 | 模型输出（含 tool\_calls） |
| `ToolMessage`   | 工具结果  | 工具返回值               |
| `SystemMessage` | 系统消息  | 系统级指示               |

### 5. 链式调用 (Chain)

虽然本系统主要使用 LangGraph 进行编排，但 LangChain 的链式调用概念是基础：

```
用户输入 → 提示词组装 → LLM 推理 → 工具选择 → 工具执行 → 响应生成
```

***

## LangGraph 架构设计详解

### 状态图架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         StateGraph                                │
│                                                                   │
│    ┌─────────────┐                                               │
│    │   START      │                                               │
│    └──────┬──────┘                                               │
│           │                                                       │
│           ▼                                                       │
│    ┌─────────────┐     ┌───────────────────────────────────┐    │
│    │  route_node  │────▶│  tool_execution (ReAct 循环)       │    │
│    │   (路由)      │     │                                    │    │
│    └─────────────┘     │  ┌─────────────────────────────┐   │    │
│           │             │  │  while has_action:          │   │    │
│           │             │  │    LLM → Tool Call → Result │   │    │
│           │             │  └─────────────────────────────┘   │    │
│           │             └───────────────┬───────────────────┘    │
│           │                               │                       │
│           ▼                               ▼                       │
│    ┌─────────────┐               ┌─────────────┐                 │
│    │  END        │◀──────────────│ output_node  │                 │
│    │   (结束)     │               │   (输出)      │                 │
│    └─────────────┘               └─────────────┘                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 节点定义 (Nodes)

本系统使用 `create_react_agent` 预构建 Agent，这是 LangGraph 提供的标准 ReAct (Reasoning + Acting) 模式实现：

```python
# modules/agent.py
from langgraph.prebuilt import create_react_agent

self.agent = create_react_agent(
    self.llm,                    # 底层 LLM
    tools=self.tools,            # 可用工具列表
    prompt=self.system_prompt    # 系统提示词
)
```

**ReAct Agent 工作流程**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct 循环                                │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Thought  │───▶│  Action  │───▶│ Observe  │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│       ▲                                    │                │
│       │                                    │                │
│       └────────────────────────────────────┘                │
│                    (循环直到得到答案)                         │
└─────────────────────────────────────────────────────────────┘
```

### 边连接 (Edges)

LangGraph 的边定义了节点之间的连接关系：

```python
# create_react_agent 内部边逻辑
graph = StateGraph(...)
graph.add_edge("start", "agent")        # 起始 → Agent
graph.add_edge("agent", "action")       # Agent → 工具执行
graph.add_edge("action", "agent")       # 工具 → Agent (继续推理)
graph.add_edge("agent", "end")          # Agent → 结束 (无工具调用时)
```

### 条件分支 (Conditional Edges)

ReAct Agent 根据 LLM 输出动态决定下一步：

- **有工具调用** → 执行工具 → 返回 Agent 继续推理
- **无工具调用且有文本输出** → 结束，返回最终结果

### 状态管理 (State)

LangGraph 使用 `State` 字典在节点间传递数据：

```python
# 流式输出的状态结构
{
    "messages": [
        HumanMessage(content="用户输入"),
        AIMessage(content="思考...", tool_calls=[...]),
        ToolMessage(content="搜索结果..."),
        AIMessage(content="最终答案")
    ]
}
```

### 流式执行 (Stream Mode)

```python
# modules/agent.py - stream_run 方法
def stream_run(self, user_input: str, chat_history: List[Any] = None):
    for chunk in self.agent.stream({"messages": messages}, stream_mode="values"):
        if "messages" in chunk:
            yield chunk["messages"][-1]  # 产出中间步骤
```

**stream\_mode 说明**:

| 模式         | 说明       | 用途       |
| ---------- | -------- | -------- |
| `values`   | 输出完整状态字典 | 获取所有中间消息 |
| `messages` | 仅输出消息列表  | 减少数据传输   |
| `updates`  | 输出状态更新增量 | 高效增量更新   |

***

## 模块职责与数据流转

### 核心类图

```
┌─────────────────────────────────────────────────────────────────┐
│                           app.py                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    StreamlitPage                         │    │
│  │  - session_state.messages: List[BaseMessage]           │    │
│  │  + process_agent_response()                             │    │
│  │  + render_chat_history()                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │ TravelAgent
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        modules/agent.py                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      TravelAgent                        │    │
│  │  - llm: ChatOpenAI                                       │    │
│  │  - tools: List[BaseTool]                                │    │
│  │  - system_prompt: str                                    │    │
│  │  - agent: PrebuiltReActAgent                            │    │
│  │  + run(user_input, chat_history)                        │    │
│  │  + stream_run(user_input, chat_history)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │   @tool           │  │   @tool           │                   │
│  │ search_destinations│  │ search_restaurants│                   │
│  └───────────────────┘  └───────────────────┘                   │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │   @tool           │  │   @tool           │                   │
│  │ search_hotels     │  │ get_nearby_places │                   │
│  └───────────────────┘  └───────────────────┘                   │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │   @tool           │  │   @tool           │                   │
│  │ plan_route        │  │ geocode_address   │                   │
│  └───────────────────┘  └───────────────────┘                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ AmapMCPService
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      modules/amap_mcp.py                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    AmapMCPService                       │    │
│  │  - api_key: str                                         │    │
│  │  - base_url: str                                        │    │
│  │  - cache: CacheManager                                  │    │
│  │  + search_place(keywords, city, types)                 │    │
│  │  + get_around_places(location, keywords, radius)        │    │
│  │  + get_route_directions(origin, destination, mode)     │    │
│  │  + get_geocode(address, city)                          │    │
│  │  + format_place_info(poi)                              │    │
│  │  + format_route_info(route_data)                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌───────────────────────────┐  ┌───────────────────────────┐
│    modules/cache.py       │  │    modules/error_handler.py│
│  ┌─────────────────────┐  │  │  ┌─────────────────────┐  │
│  │   CacheManager      │  │  │  │   ErrorHandler      │  │
│  │  - _cache: Dict     │  │  │  │  - max_retries: int │  │
│  │  - _lock: RLock     │  │  │  │  - base_delay: float│  │
│  │  - _ttl: int        │  │  │  │  + exponential_back │  │
│  │  - _access_times    │  │  │  │  + should_retry()   │  │
│  │  + get(key)         │  │  │  │  + handle_error()    │  │
│  │  + set(key, value)  │  │  │  └─────────────────────┘  │
│  │  + _evict_lru()     │  │  │  ┌─────────────────────┐  │
│  │  + clear()          │  │  │  │   @with_retry       │  │
│  └─────────────────────┘  │  │  │   decorator         │  │
│                            │  │  └─────────────────────┘  │
│  Singleton: get_cache()    │  │                           │
└────────────────────────────┘  └───────────────────────────┘
```

### 数据流转机制

```
用户输入 (Streamlit)
       │
       ▼
TravelAgent.run() / stream_run()
       │
       ▼
ChatHistory + HumanMessage 组装
       │
       ▼
create_react_agent.invoke() / stream()
       │
       ├──────────────────┐
       ▼                  ▼
  [LLM 推理]        [工具选择]
       │                  │
       ▼                  ▼
  文本输出           tool_calls
       │                  │
       │          ┌───────┴───────┐
       │          ▼               ▼
       │    工具执行           格式化返回
       │          │               │
       │          ▼               │
       │    ToolMessage           │
       │          │               │
       └──────────┴───────────────┘
                     │
                     ▼
              AIMessage (最终输出)
                     │
                     ▼
           Streamlit 界面渲染
```

***

## RAG 增强架构 (Chroma DB)

### 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG (Retrieval-Augmented Generation)          │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Query      │────▶│   Retrieve   │────▶│    Generate  │   │
│  │   输入查询     │     │   检索向量库   │     │   LLM 生成    │   │
│  └──────────────┘     └──────┬───────┘     └──────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                     ┌──────────────┐                            │
│                     │  Chroma DB   │                            │
│                     │  向量数据库    │                            │
│                     └──────────────┘                            │
│                            │                                    │
│              ┌─────────────┴─────────────┐                       │
│              ▼                           ▼                       │
│    ┌──────────────────┐     ┌──────────────────┐              │
│    │ attractions      │     │ user_preferences  │              │
│    │ 景点知识库         │     │ 用户偏好历史       │              │
│    └──────────────────┘     └──────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Chroma DB 集合设计

| 集合名称 | 用途 | 存储内容 |
|---------|------|---------|
| `attractions` | 景点知识库 | 景点名称、城市、地址、类型、评分、开放时间、门票、简介 |
| `user_preferences` | 用户偏好历史 | 目的地、天数、预算、偏好类型、选择景点 |

### RAG 工具

| 工具名称 | 功能 | 使用场景 |
|---------|------|---------|
| `retrieve_attractions_knowledge` | 检索景点知识库 | 规划前查询景点详情 |
| `save_trip_preference` | 保存用户偏好 | 行程结束后记录选择 |
| `get_user_history_preferences` | 获取历史偏好 | 个性化推荐参考 |

### 数据流转

```
用户请求规划行程
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ 1. retrieve_attractions_knowledge (RAG 检索)        │
│    → 查询 Chroma DB attractions 集合                 │
│    → 返回景点知识: 评分、开放时间、简介               │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ 2. get_user_history_preferences (用户画像)          │
│    → 查询 Chroma DB user_preferences 集合           │
│    → 返回用户历史: 偏好类型、预算范围、选择模式        │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ 3. 高德地图 MCP (实时数据)                          │
│    → search_place / get_route_directions           │
│    → 返回实时 POI 数据                               │
└─────────────────────────────────────────────────────┘
       │
       ▼
   LLM 综合知识库 + 历史 + 实时数据 → 生成个性化行程
```

### 向量存储配置

```env
# .env 新增配置
VECTOR_STORE_DIR=./data/vector_store
```

### 使用示例

```python
from modules.vector_store import get_vector_store

vector_store = get_vector_store()

# 添加景点到知识库
vector_store.add_attraction({
    "name": "故宫",
    "city": "北京",
    "address": "北京市东城区景山前街4号",
    "type": "历史博物馆",
    "rating": 4.8,
    "open_hours": "08:30-17:00",
    "ticket": "60元",
    "description": "中国明清两代的皇家宫殿",
    "highlights": "太和殿、乾清宫、御花园"
})

# 检索景点知识
results = vector_store.search_attractions(query="故宫", city="北京", limit=3)

# 保存用户偏好
vector_store.add_user_preference(
    user_id="user_001",
    destination="北京",
    days=3,
    budget=5000,
    preferences=["历史文化", "美食"],
    selected_places=[{"name": "故宫", "type": "景点"}, {"name": "全聚德", "type": "餐厅"}]
)

# 获取用户历史偏好
prefs = vector_store.get_user_preferences(user_id="user_001", limit=5)
```

***

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# OpenAI 兼容 API 配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 高德地图 API 配置
AMAP_API_KEY=your_amap_api_key
AMAP_BASE_URL=https://restapi.amap.com/v3

# 缓存配置
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
```

### 3. 获取 API Key

**阿里云 DashScope (LLM)**:

1. 访问 <https://dashscope.console.aliyun.com/>
2. 创建 API Key 并充值

**高德地图**:

1. 访问 <https://lbs.amap.com/>
2. 创建应用并获取 Web API Key

### 4. 启动应用

```bash
streamlit run app.py
```

访问 `http://localhost:8501` 开始使用。

***

## 代码示例

### 示例 1: 直接使用 TravelAgent

```python
from modules.agent import get_travel_agent
from langchain_core.messages import HumanMessage

agent = get_travel_agent()

# 同步调用
response = agent.run(
    "我想去北京玩3天，预算3000元",
    chat_history=[]
)
print(response)
```

### 示例 2: 流式调用 (实时展示中间步骤)

```python
from modules.agent import get_travel_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

agent = get_travel_agent()

# 流式调用，实时展示工具调用过程
for msg in agent.stream_run("我想去北京玩3天"):
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for tool_call in msg.tool_calls:
            print(f"调用工具: {tool_call['name']}")
            print(f"参数: {tool_call['args']}")
    elif isinstance(msg, ToolMessage):
        print(f"工具返回: {msg.content[:100]}...")
    elif isinstance(msg, AIMessage) and msg.content:
        print(f"最终答案: {msg.content}")
```

### 示例 3: 自定义 AmapMCPService 查询

```python
from modules.amap_mcp import get_amap_service

amap = get_amap_service()

# 搜索景点
places = amap.search_place("故宫", city="北京")
for poi in places.get("pois", [])[:5]:
    print(amap.format_place_info(poi))

# 周边搜索
around = amap.get_around_places(
    location="116.397428,39.90923",  # 故宫经纬度
    keywords="餐厅",
    radius=1000
)

# 路线规划
route = amap.get_route_directions(
    origin="天安门",
    destination="故宫",
    mode="walking"
)
print(amap.format_route_info(route))
```

### 示例 4: 缓存使用示例

```python
from modules.cache import get_cache

cache = get_cache()

# 直接缓存
cache.set("my_key", {"data": "value"})
value = cache.get("my_key")

# 装饰器缓存
@cache.cached
def expensive_function(arg1, arg2):
    # 耗时计算
    return result
```

### 示例 5: 错误处理与重试

```python
from modules.error_handler import with_retry, APIError

@with_retry("Amap")
def unreliable_api_call():
    # 自动重试机制
    # - 指数退避: 1s, 2s, 4s, 8s...
    # - 最大延迟: 60s
    # - 可重试: RateLimitError, ServiceUnavailableError, timeout
    pass
```

***

## 性能优化建议

### 1. 分层缓存策略 (已实现)

系统实现了基于数据类型的分层缓存，自动识别搜索类型并应用不同 TTL：

```python
# modules/cache.py
CACHE_TTL_CONFIG = {
    "place": 86400,       # 景点: 24小时
    "restaurant": 3600,   # 餐厅: 1小时
    "hotel": 3600,        # 酒店: 1小时
    "route": 1800,        # 路线: 30分钟
    "geocode": 604800,    # 地理编码: 7天
}

def get_cache_ttl(category: str) -> int:
    return CACHE_TTL_CONFIG.get(category, 3600)
```

**自动分类机制**: `AmapMCPService._detect_category()` 根据关键词自动识别数据类型：
- 包含"餐厅"、"美食" → restaurant
- 包含"酒店"、"宾馆" → hotel
- 包含"路线"、"导航" → route
- 包含"地址"、"位置" → geocode
- 其他 → place

### 2. 批量搜索工具 (已实现)

新增 `batch_search` 工具，支持一次调用搜索多个类型：

```python
@tool
def batch_search(query: str, city: str, types: List[str]) -> str:
    """批量搜索景点、餐厅、酒店等多个类型"""
    results = {}
    for search_type in types:
        result = amap.search_place(f"{query} {search_type}", city=city)
        results[search_type] = {...}
    return formatted_output
```

**优势**: 减少 LLM 推理次数，提升多日行程规划效率。

### 3. 对话历史管理 (已实现)

限制对话历史长度，避免内存溢出：

```python
MAX_HISTORY = 20

def trim_history(messages, max_len=MAX_HISTORY):
    if len(messages) > max_len:
        return messages[:1] + messages[-(max_len-1):]
    return messages
```

### 4. 行程导出功能 (已实现)

支持 Markdown 和文本格式导出：

```python
st.download_button(
    label="📥 导出行程 (Markdown)",
    data=itinerary_text,
    file_name="行程规划.md",
    mime="text/markdown"
)
```

### 5. 缓存命中率优化

避免缓存穿透：

```python
# 查询前检查缓存
cache_key = f"amap_search:{keywords}:{city}"
cached = cache.get(cache_key)
if cached:
    return cached  # 命中缓存，直接返回
```

***

## 常见问题

### Q: API Key 未配置

**A**: 请检查 `.env` 文件中的 `OPENAI_API_KEY` 与 `AMAP_API_KEY` 是否正确设置。

### Q: 启动端口冲突

**A**: Streamlit 会自动切换端口，按终端提示访问新地址。也可指定端口：

```bash
streamlit run app.py --server.port 8502
```

### Q: 搜索无结果

**A**: 尝试更具体的关键词，例如 "北京 三里屯 餐厅" 而非仅 "餐厅"。

### Q: LangChain 与 LangGraph 版本不兼容

**A**: 确保使用推荐版本组合：

```bash
pip install langchain==0.3.0 langgraph==0.2.0 langchain-openai==0.2.0
```

### Q: 缓存未生效

**A**: 检查 `CACHE_TTL` 设置，确保缓存未过期。可通过日志确认缓存命中情况。

***

## 安全建议

- 不要将真实密钥提交到 Git
- 建议将 `.env` 加入 `.gitignore`
- 定期轮换 API 密钥
- 生产环境建议使用密钥管理服务（如阿里云 KMS）

***

## 项目结构

```text
.
├── app.py                  # Streamlit 前端入口
├── modules/
│   ├── agent.py            # Agent 编排与提示词
│   ├── amap_mcp.py         # 高德地图能力封装
│   ├── cache.py            # 缓存管理
│   ├── vector_store.py     # Chroma DB 向量存储
│   └── error_handler.py    # 错误处理与重试
├── data/
│   └── vector_store/       # Chroma DB 持久化存储
├── requirements.txt
└── .env
```

