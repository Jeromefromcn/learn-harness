"""
week5_day3/langsmith_setup.py
==============================
第五週 Day 3:用 LangSmith 自動追蹤

對應手冊任務:
  - 設置兩個環境變量,跑幾次 agent,到 LangSmith 看自動生成的 trace
  - 找到框架自動記錄的字段(工具調用,token,成本,延遲)
  - 對比:和 Week 3 Day 3 手寫的 log_call() 有什麼差別?

核心發現:
  設置 LANGSMITH_TRACING=true 之後,所有 LangChain/LangGraph 調用
  自動上報,不需要改 agent 代碼任何一行.
  這是框架"開箱即用的可觀測性"--你手寫的 log_call() 是它的雛形.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import setup_langsmith_env, LANGSMITH_PROJECT

# ============================================================
# === 第五週第三天:開啟 LangSmith 自動追蹤 ===
# ============================================================

# 只需要這兩行:設置環境變量,後續所有 LangChain 調用自動被追蹤
setup_langsmith_env()

print(f"LangSmith 追蹤已開啟(project: {LANGSMITH_PROJECT})")
print("每次 agent 調用都會自動上報 trace 到 https://smith.langchain.com")
print()

# 正常 import 和使用 agent(代碼一行不改)
from config import DEEPSEEK_API_KEY, DEEPSEEK_OPENAI_BASE_URL, MODEL_FLASH
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools_lc import TOOLS

model = ChatOpenAI(
    model=MODEL_FLASH,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_OPENAI_BASE_URL,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個物流客服助手.使用提供的工具查詢信息後,給出清晰的回答."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=model, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=TOOLS, verbose=True, max_iterations=6)


def run_agent(user_msg: str) -> str:
    """完全相同的 agent,只是開了 LangSmith 追蹤."""
    result = agent_executor.invoke({"input": user_msg})
    return result.get("output", "")


# ============================================================
# 對比分析:手寫 log_call() vs LangSmith 自動追蹤
# ============================================================

COMPARISON = """
手寫 log_call()(Week 3 Day 3) vs LangSmith 自動追蹤(Week 5 Day 3)

[LangSmith 比手寫好的地方]
  ✅ 零侵入:不改 agent 代碼,只設環境變量
  ✅ 完整 trace 樹:每一步(模型調用,工具調用)都有父子關係
  ✅ 可視化:在 Web UI 裡看 trace,比讀 JSONL 更直觀
  ✅ 時間線:可以看每個步驟花了多少時間
  ✅ Playground:直接在 UI 裡重放和修改某次調用

[手寫比 LangSmith 好的地方]
  ✅ 數據不出境:敏感數據不會發到第三方
  ✅ 完全控制:你決定記錄什麼,不記錄什麼
  ✅ 可自定義:cost_usd 字段是你加的,LangSmith 的不一定有
  ✅ 無外部依賴:LangSmith 掛了你的日誌還在

[什麼時候用哪個]
  - 快速原型,個人項目:LangSmith(省時間)
  - 涉及客戶數據,企業合規:手寫可觀測性(數據在你這)
  - 兩者結合:手寫基礎指標,LangSmith 做調試用的 trace
"""


if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Day 3:LangSmith 自動追蹤")
    print("=" * 60)

    print("\n注意:需要先設置 LANGSMITH_API_KEY 環境變量")
    print("在 https://smith.langchain.com 免費注冊後獲取\n")

    # 跑幾次 agent,到 LangSmith 看 trace
    questions = [
        "SF123 在哪裡?",
        "5kg 包裹到 B 區多少錢?",
        "SKU-9 有沒有庫存?",
    ]

    for q in questions:
        print(f"\n問:{q}")
        answer = run_agent(q)
        print(f"答:{answer[:100]}")
        print("-> 在 LangSmith 應該能看到這次調用的完整 trace")

    print("\n" + "=" * 60)
    print(COMPARISON)

    print("\n關鍵觀察任務:")
    print("1. 到 LangSmith 找到剛才的 3 次調用")
    print("2. 點開一個 trace,找到它自動記錄了哪些字段")
    print("3. 對比 Week 3 的 agent_calls.jsonl,看差了什麼,多了什麼")
    print("\nDay 3 完成.下一步:week5_day4/langsmith_eval.py - 框架版評估.")
