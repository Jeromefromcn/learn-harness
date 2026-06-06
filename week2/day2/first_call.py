"""
week2_day2/first_call.py
=========================
第二週 Day 2:第一次帶 tools 的 API 調用

對應手冊任務:
  - 發出第一個帶 tools 的請求
  - 確認 stop_reason 返回 'tool_use'
  - 從響應裡解析出模型想調哪個工具,帶了哪些參數
  - 手動執行對應的工具函數,看結果對不對

重要概念(Day 2 的核心):
  stop_reason = "tool_use"  -> 模型沒有給出最終答案,
  它在說"我需要調用一個工具,請你幫我執行".
  這就是 harness 存在的意義:模型說想做什麼,harness 替它做.

注意:Day 2 只做"一次"調用,不做循環.
     完整的 loop 在 Day 3 的 agent.py 裡.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEFAULT_MODEL, setup_anthropic_env

# 設置環境變量(讓 Anthropic SDK 指向 DeepSeek)
setup_anthropic_env()
import anthropic  # 必須在 setup_anthropic_env() 之後 import

from tools import TOOLS, dispatch_tools


# ============================================================
# 初始化 Anthropic 客戶端
# ============================================================
# 因為 setup_anthropic_env() 已經設置了 ANTHROPIC_API_KEY 和
# ANTHROPIC_BASE_URL,這裡直接調用無參數版本就好.
client = anthropic.Anthropic()


def single_call_with_tools(user_message: str) -> anthropic.types.Message:
    """
    發出一次帶 tools 的 API 請求(不做循環,只看第一次響應).

    返回完整的 Message 對象,讓你可以探索它的結構.
    """
    print(f"\n用戶問題:{user_message}")
    print(f"使用模型:{DEFAULT_MODEL}")
    print("-" * 40)

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        tools=TOOLS,        # 告訴模型有哪些工具可以用
        messages=[
            {"role": "user", "content": user_message}
        ],
    )
    return response


def inspect_response(resp: anthropic.types.Message):
    """
    解析並打印響應的關鍵信息.
    Day 2 的核心任務:理解 tool_use 響應的結構.
    """
    print(f"\nstop_reason = '{resp.stop_reason}'")
    print(f"  -> {'模型請求工具調用,還沒給最終答案' if resp.stop_reason == 'tool_use' else '模型給出了最終答案'}")

    print(f"\n共 {len(resp.content)} 個 content 塊:")
    for i, block in enumerate(resp.content):
        print(f"  [{i}] type = {block.type}")

        if block.type == "text":
            print(f"      text = {block.text[:100]}...")

        elif block.type == "tool_use":
            # 這是模型說"我要調用這個工具"的結構
            print(f"      id   = {block.id}")
            print(f"      name = {block.name}")
            print(f"      input = {json.dumps(block.input, ensure_ascii=False)}")

    print(f"\nToken 使用:輸入 {resp.usage.input_tokens},輸出 {resp.usage.output_tokens}")


def manually_execute_tool(resp: anthropic.types.Message):
    """
    手動執行模型請求的工具(Day 2 觀察用).
    Day 3 的 dispatch_tools 會自動化這個過程.
    """
    print("\n=== 手動執行工具(Day 2 觀察) ===")
    tool_results = dispatch_tools(resp.content)

    if not tool_results:
        print("響應中沒有 tool_use 塊(模型直接給出了答案)")
        return

    for tr in tool_results:
        print(f"\ntool_use_id: {tr['tool_use_id']}")
        print(f"結果: {tr['content']}")

    print("\n-> 下一步(Day 3 的任務):把這個結果作為 user 消息發回給模型,")
    print("  讓它繼續推理,直到給出最終答案.")


# ============================================================
# 主程序:逐步運行,觀察每一步的輸出
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Week 2 Day 2:第一次帶 tools 的 API 調用")
    print("=" * 50)

    # 測試用例 1:觸發工具調用的問題
    resp = single_call_with_tools("幫我查一下 SF123 的物流狀態")
    inspect_response(resp)
    manually_execute_tool(resp)

    print("\n" + "=" * 50)

    # 測試用例 2:不需要工具的問題(對比)
    resp2 = single_call_with_tools("你好,你是誰?")
    print("\n不需要工具的問題:")
    inspect_response(resp2)

    print("\n\nDay 2 完成.下一步:week2_day3/agent.py - 實現完整的循環.")
