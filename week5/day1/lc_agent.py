"""
week5_day1/lc_agent.py
=======================
第五週 Day 1:用 LangChain create_agent 重建 Week 2 的 agent

對應手冊任務:
  - 把三個工具加上 @tool,用 create_agent 創建 agent
  - 記錄行數對比:框架版 vs 手寫版
  - 想一想:框架在幕後幫你做了什麼?

DeepSeek 接入方式:
  DeepSeek 走 OpenAI 兼容格式,
  所以用 langchain-openai 的 ChatOpenAI,把 base_url 指向 DeepSeek.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_OPENAI_BASE_URL, MODEL_FLASH

# LangChain 接入 DeepSeek(OpenAI 兼容接口)
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

from tools_lc import TOOLS


# ============================================================
# 初始化模型(指向 DeepSeek)
# ============================================================

# DeepSeek 走 OpenAI 兼容格式,所以用 ChatOpenAI
# model 名稱和 base_url 指向 DeepSeek 即可
model = ChatOpenAI(
    model=MODEL_FLASH,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_OPENAI_BASE_URL,
    temperature=0,
)

# 系統提示詞(對應 Week 2 Day 5 的 SYSTEM_PROMPT_WITH_JSON_CONSTRAINT)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個物流客服助手.使用提供的工具查詢信息後,給出清晰的回答."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # LangChain 的工具調用記錄佔位符
])


# ============================================================
# === 第五週第一天新增:create_tool_calling_agent ===
# ============================================================

# 創建 agent(這幾行對應 Week 2 Day 3 的整個 run_agent 函數 + dispatch_tools)
agent = create_tool_calling_agent(
    llm=model,
    tools=TOOLS,
    prompt=prompt,
)

# AgentExecutor:執行器(處理循環,工具調用,錯誤)
agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,       # 打印每一步(等同於手寫版的 verbose=True)
    max_iterations=6,   # 等同於 max_turns=6
    handle_parsing_errors=True,  # 捕獲輸出格式錯誤
)


def run_agent(user_msg: str) -> str:
    """LangChain 版的 agent 入口,接口和 Week 2 的 run_agent 一樣."""
    result = agent_executor.invoke({"input": user_msg})
    return result.get("output", "")


# ============================================================
# 主程序:行數對比觀察
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Day 1:LangChain create_tool_calling_agent")
    print("=" * 60)

    print("\n[行數對比]")
    print("  Week 2 Day 3 手寫 run_agent:約 45 行(包含 dispatch_tools)")
    print("  Week 5 Day 1 框架版:約 15 行(不含工具定義)")
    print("  節省:~30 行,但代價是多了 langchain 依賴")

    print("\n[框架幫你做了什麼?]")
    print("  ✅ Agent loop(CLOSED 循環)")
    print("  ✅ Tool dispatch(自動把模型請求路由到對應工具)")
    print("  ✅ 錯誤捕獲(handle_parsing_errors)")
    print("  ✅ max_iterations 護欄")
    print("  ❌ 沒幫你做:重試策略(Week 3 那些你要自己加)")
    print("  ❌ 沒幫你做:成本追蹤(需要 LangSmith 或手寫)")
    print("  ❌ 沒幫你做:結構化輸出校驗(需要 OutputParser)")

    # 實際運行
    print("\n[實際測試]")
    q = "查一下 SF123 的物流狀態,另外 5kg 寄到 B 區多少錢?"
    print(f"問題:{q}\n")
    answer = run_agent(q)
    print(f"\n最終回答:{answer}")

    print("\nDay 1 完成.")
    print("下一步:week5_day2/lg_agent.py - LangGraph StateGraph + checkpointer.")
