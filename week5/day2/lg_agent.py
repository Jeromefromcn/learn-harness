"""
week5_day2/lg_agent.py
=======================
第五週 Day 2：用 LangGraph StateGraph + checkpointer 重建 agent

對應手冊任務：
  - 把 agent 表達成一張狀態圖（StateGraph）
  - 加一個 checkpointer，讓狀態跨會話持久化
  - 試一次「人在迴路中（human-in-the-loop）中斷」：
    在執行危險操作前暫停，等人工批准後再繼續

LangGraph 比 create_tool_calling_agent 多了什麼？
  - 狀態圖：可以表達複雜的分支邏輯（不只是 model→tools→model 的線性循環）
  - Checkpointer：狀態持久化（中斷後可以從斷點繼續）
  - Human-in-the-loop：可以暫停等人工審批
  - 多 agent：可以有多個節點（多個模型或多個專業 agent）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_OPENAI_BASE_URL, MODEL_FLASH

from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools_lc import TOOLS


# ============================================================
# 狀態定義
# ============================================================

class AgentState(TypedDict):
    """
    Agent 的狀態。LangGraph 用這個類型管理每一輪的數據。

    messages: 消息歷史列表
      Annotated[..., add_messages] 表示：新消息追加到列表末尾，
      而不是覆蓋（這是 LangGraph 的「reducer」機制）
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 模型初始化
# ============================================================

model = ChatOpenAI(
    model=MODEL_FLASH,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_OPENAI_BASE_URL,
    temperature=0,
).bind_tools(TOOLS)   # 把工具綁定到模型，讓它知道有哪些工具可用


# ============================================================
# 節點函數
# ============================================================

def call_model(state: AgentState) -> dict:
    """
    model 節點：調用 LLM，把響應加入消息歷史。
    對應手寫版的：resp = client.messages.create(...)
    """
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    條件邊：決定下一步走 tools 還是 END。
    對應手寫版的：if resp.stop_reason == "tool_use"
    """
    last_message = state["messages"][-1]
    # AIMessage 裡有 tool_calls 就走 tools 節點，否則結束
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ============================================================
# === 第五週第二天新增：構建 StateGraph ===
# ============================================================

def build_graph(use_checkpointer: bool = True):
    """
    構建 LangGraph 狀態圖。

    圖的結構：
      START → model → [工具請求?] → tools → model → ... → END
                                ↓
                              [無工具請求]
                                ↓
                              END

    對比手寫版的 run_agent for 循環：
      手寫：命令式循環，你控制每一步
      LangGraph：聲明式圖，你定義節點和邊，框架執行循環
    """
    builder = StateGraph(AgentState)

    # 添加節點
    builder.add_node("model", call_model)
    # ToolNode 自動路由到對應工具函數（對應手寫版的 dispatch_tools）
    builder.add_node("tools", ToolNode(TOOLS))

    # 添加邊
    builder.add_edge(START, "model")                     # 入口 → model
    builder.add_conditional_edges("model", should_continue)  # model → tools or END
    builder.add_edge("tools", "model")                   # tools → model（循環）

    # Checkpointer：保存每一步的狀態（實現中斷恢復和多輪對話）
    if use_checkpointer:
        memory = MemorySaver()  # 內存版，生產環境可用 SqliteSaver 或 PostgresSaver
        graph = builder.compile(checkpointer=memory)
    else:
        graph = builder.compile()

    return graph


# ============================================================
# 運行函數
# ============================================================

graph = build_graph(use_checkpointer=True)


def run_agent(user_msg: str, thread_id: str = "default") -> str:
    """
    LangGraph 版的 agent 入口。

    thread_id：
      - 同一個 thread_id 的調用共享消息歷史（多輪對話）
      - 不同 thread_id 是獨立的會話
      - 這是手寫版需要自己管理 messages 列表的地方，框架幫你處理了

    Args:
        user_msg:  用戶問題
        thread_id: 會話 ID（同 ID 可接續上一輪對話）
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config=config,
    )
    # 取最後一條 AI 消息的文字內容
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def demo_human_in_the_loop(thread_id: str = "hitl-demo"):
    """
    演示「人在迴路中」中斷。

    真實場景：
      在執行某個危險操作（如「刪除訂單」、「退款」）前，
      先暫停讓人工確認，批准後再繼續。

    LangGraph 實現方式：
      graph.stream() + interrupt_before=["tools"]
      當圖到達 tools 節點前，自動暫停，返回當前狀態。
      人工審批後，用相同 thread_id 繼續執行。
    """
    print("\n=== Human-in-the-Loop 演示 ===")
    config = {"configurable": {"thread_id": thread_id}}

    # 構建帶中斷的圖（在 tools 節點前暫停）
    hitl_graph = build_graph(use_checkpointer=True)
    hitl_graph = StateGraph(AgentState)
    hitl_graph.add_node("model", call_model)
    hitl_graph.add_node("tools", ToolNode(TOOLS))
    hitl_graph.add_edge(START, "model")
    hitl_graph.add_conditional_edges("model", should_continue)
    hitl_graph.add_edge("tools", "model")
    memory = MemorySaver()
    hitl_graph = hitl_graph.compile(
        checkpointer=memory,
        interrupt_before=["tools"],  # 在執行工具前暫停
    )

    print("第一步：發送問題（到 tools 節點前會暫停）")
    for chunk in hitl_graph.stream(
        {"messages": [HumanMessage(content="查一下 SF123 的物流狀態")]},
        config=config,
        stream_mode="values",
    ):
        last = chunk["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            print(f"\n⏸  已暫停！模型想調用工具：{[tc['name'] for tc in last.tool_calls]}")
            print("   → 等待人工批准...")

    # 模擬人工批准
    approved = input("\n是否允許執行工具？(y/n): ").strip().lower() == "y"

    if approved:
        print("\n第二步：人工批准，繼續執行...")
        result = hitl_graph.invoke(None, config=config)  # None = 繼續上次的狀態
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"最終回答：{msg.content}")
                break
    else:
        print("操作已取消。")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Day 2：LangGraph StateGraph + Checkpointer")
    print("=" * 60)

    # 測試 1：基本運行
    print("\n【測試 1】基本運行")
    answer = run_agent("SF123 在哪裡？5kg 到 B 區多少錢？", thread_id="test-001")
    print(f"回答：{answer[:200]}")

    # 測試 2：多輪對話（同一 thread_id 接續上文）
    print("\n【測試 2】多輪對話（接續上文）")
    answer2 = run_agent("SKU-9 有貨嗎？", thread_id="test-001")
    print(f"回答（應該知道上文提到了 B 區）：{answer2[:200]}")

    # 測試 3：與手寫版的對比問題
    print("\n【與手寫版對比】")
    print("LangGraph 額外給了你：")
    print("  ✅ Checkpointer（狀態持久化，中斷可恢復）")
    print("  ✅ 多輪對話（同 thread_id 自動接續）")
    print("  ✅ Human-in-the-loop（interrupt_before）")
    print("  ✅ 聲明式圖（複雜分支比 if/else 更清晰）")
    print("\n你手寫版有、框架沒有的：")
    print("  ✅ 精細的重試策略（@with_retry 裝飾器）")
    print("  ✅ 斷路器（CircuitBreaker）")
    print("  ✅ 成本追蹤（SessionCostTracker）")
    print("  ✅ 數據不出境的可觀測性")

    print("\nDay 2 完成。")
    print("下一步：week5_day3/langsmith_setup.py — 零改動開啟追蹤。")
