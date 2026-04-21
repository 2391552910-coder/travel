import streamlit as st
from modules.agent import get_travel_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
import json
from datetime import date

st.set_page_config(page_title="智能旅游行程规划助手", page_icon="✈️", layout="wide")

def process_agent_response(user_input, history):
    """处理 Agent 响应并实时展示中间过程"""
    agent = get_travel_agent()
    full_response = ""

    # 创建一个状态容器展示思考过程
    with st.status("🔍 正在规划行程，请稍候...", expanded=True) as status:
        for msg in agent.stream_run(user_input, history):
            # 处理模型思考和工具调用
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    st.write(f"🛠️ **调用工具**: `{tool_call['name']}`")
                    st.json(tool_call['args'])

            # 处理工具执行结果
            elif isinstance(msg, ToolMessage):
                st.write(f"✅ **工具返回结果** (ID: `{msg.tool_call_id}`)")
                with st.expander("查看原始数据"):
                    st.code(msg.content)

            # 处理最终文本输出
            elif isinstance(msg, AIMessage) and msg.content:
                full_response = msg.content

        status.update(label="✨ 行程规划完成！", state="complete", expanded=False)

    return full_response

if "messages" not in st.session_state:
    st.session_state.messages = []

# 默认值
DEFAULT_DESTINATION = "北京"
DEFAULT_START_DATE = date(2026, 4, 21)
DEFAULT_END_DATE = date(2026, 4, 24)

with st.sidebar:
    st.title("🗺️ 规划你的行程")
    st.info("填写基本信息，开始个性化规划")

    with st.form("planning_form"):
        destination = st.text_input("目的地", value=DEFAULT_DESTINATION)

        # 修改为日期范围选择
        date_range = st.date_input("出行日期范围", [DEFAULT_START_DATE, DEFAULT_END_DATE])

        # 修复预算输入倒序问题，改用 text_input
        budget_str = st.text_input("预算范围 (元)", value="3000", help="请输入数字，例如：3000")
        try:
            budget = int(budget_str) if budget_str else 0
        except ValueError:
            st.error("预算请输入纯数字")
            budget = 0

        travelers = st.selectbox("同行人员", ["单人", "情侣/夫妻", "家庭出游", "朋友聚会"])
        preferences = st.multiselect("兴趣偏好", ["历史文化", "自然风光", "美食探店", "休闲度假", "购物逛街", "亲子活动"], default=["历史文化", "美食探店"])
        special_needs = st.text_area("特殊需求", placeholder="例如：轮椅友好、不吃辣、慢节奏等")
        submit_button = st.form_submit_button("生成行程")

if submit_button and destination:
    # 验证日期范围
    if len(date_range) != 2:
        st.warning("请选择完整的出行日期范围（开始和结束日期）")
    else:
        start_date, end_date = date_range
        days = (end_date - start_date).days + 1
        
        initial_input = f"""
        请为我规划一次旅行：
        - 目的地：{destination}
        - 出行日期：从 {start_date} 到 {end_date}（共 {days} 天）
        - 预算：{budget}元
    - 同行人员：{travelers}
    - 兴趣偏好：{', '.join(preferences)}
    - 特殊需求：{special_needs if special_needs else "无"}
    
    请根据以上需求生成一个完整的行程规划。
    """
    
    st.session_state.messages.append(HumanMessage(content=initial_input))
    
    # 使用新定义的流式展示函数
    try:
        history = st.session_state.messages[:-1]
        response = process_agent_response(initial_input, history)
        if response:
            st.session_state.messages.append(AIMessage(content=response))
            st.rerun() # 强制刷新以显示最新消息
    except Exception as e:
        st.error(f"生成行程时出错: {e}")

st.title("✨ 您的个性化旅游行程")
st.caption("💡 提示：🗺️ 景点地点链接指向高德地图 | 🍜 餐厅链接指向美团美食 | 🏨 酒店链接指向美团酒店")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        if "请为我规划一次旅行" in msg.content:
            continue
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            # 使用 unsafe_allow_html 允许链接在新标签页打开
            st.markdown(msg.content, unsafe_allow_html=True)

if user_feedback := st.chat_input("对行程不满意？告诉我想怎么修改"):
    st.session_state.messages.append(HumanMessage(content=user_feedback))
    with st.chat_message("user"):
        st.markdown(user_feedback)
        
    # 使用新定义的流式展示函数
    try:
        history = st.session_state.messages[:-1]
        response = process_agent_response(user_feedback, history)
        if response:
            st.session_state.messages.append(AIMessage(content=response))
            st.rerun()
    except Exception as e:
        st.error(f"更新行程时出错: {e}")
        if st.session_state.messages:
            st.session_state.messages.pop()
