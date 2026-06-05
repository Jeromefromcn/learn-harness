"""
week2_day3/agent.py
====================
第二週 Day 3：完整的 Agent 循環

對應手冊任務：
  - 實現 dispatch_tools()：遍歷工具請求 → 執行 → 包裝成 tool_result
  - 把工具結果加回 messages，讓循環繼續
  - 測試一個「需要多步」的問題，確認它連續調用兩個工具
  - 加上 max_turns 上限，避免死循環

這是本週最重要的文件：一個真正能運作的 agent。
Day 4 會測試它怎麼壞掉，Day 5 會給輸出加護欄。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_MODEL, setup_anthropic_env

setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools

client = anthropic.Anthropic()


# ============================================================
# run_agent：核心 Agent 循環
# ============================================================

def run_agent(user_msg: str, max_turns: int = 6, verbose: bool = True) -> str:
    """
    完整的 Agent 執行循環。

    循環邏輯（harness 的核心）：
      1. 把用戶問題發給模型
      2. 模型返回 stop_reason = "tool_use" → 執行工具 → 把結果加入消息歷史 → 回到步驟 1
      3. 模型返回 stop_reason = "end_turn" → 提取最終文字答案 → 返回

    max_turns 是最基礎的護欄：防止模型陷入無限工具調用循環。

    Args:
        user_msg:  用戶的問題
        max_turns: 最多允許幾輪工具調用（每輪 = 一次模型調用）
        verbose:   是否打印每一輪的調試信息

    Returns:
        模型的最終文字答案；若超過 max_turns 則返回錯誤信息
    """
    # 消息歷史：這是 agent「記憶」的載體
    # 格式必須嚴格遵守 Anthropic API 的要求：
    #   user → assistant → user → assistant → ...
    messages = [{"role": "user", "content": user_msg}]

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Agent 第 {turn + 1} 輪] 調用模型...")

        # 每輪都把完整的消息歷史傳給模型
        # 模型沒有「記憶」，所有上下文必須在 messages 裡
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            print(f"[Agent 第 {turn + 1} 輪] stop_reason = {resp.stop_reason}")

        # 把模型的回應加入歷史（無論是工具請求還是最終答案）
        # 這是 API 的要求：assistant 的每次回應都必須記錄下來
        messages.append({"role": "assistant", "content": resp.content})

        # ── 情況 A：模型給出了最終答案 ──────────────────────────────
        if resp.stop_reason == "end_turn":
            # 從 content 塊中找出文字回答
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text
            return ""   # 理論上不應該發生

        # ── 情況 B：模型請求工具調用 ─────────────────────────────────
        if resp.stop_reason == "tool_use":
            # 執行模型請求的所有工具（可能一次請求多個）
            tool_results = dispatch_tools(resp.content)

            if verbose:
                tool_names = [
                    b.name for b in resp.content if b.type == "tool_use"
                ]
                print(f"[Agent 第 {turn + 1} 輪] 執行工具：{tool_names}")

            # 把工具結果作為 user 角色的消息加入歷史
            # 注意：tool_result 必須緊跟在 tool_use 的 assistant 消息之後
            messages.append({"role": "user", "content": tool_results})
            # 循環繼續，讓模型看到工具結果後繼續推理
            continue

        # ── 情況 C：其他 stop_reason（max_tokens 等）─────────────────
        print(f"[Agent] 意外的 stop_reason: {resp.stop_reason}")
        break

    # 超過 max_turns：harness 的護欄觸發
    return f"[錯誤] 超過最大輪次 {max_turns}，任務未完成。請簡化問題或增加 max_turns。"


# ============================================================
# 主程序：測試多步推理
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Week 2 Day 3：完整 Agent 循環測試")
    print("=" * 55)

    # 測試用例 1：需要連續調用兩個工具的問題
    # 預期行為：先調 calc_shipping_fee，再調 check_inventory
    q1 = "我要寄 5kg 的包裹到 B 區要多少運費？順便查一下 SKU-9 還有沒有庫存"
    print(f"\n【問題 1】{q1}")
    answer1 = run_agent(q1)
    print(f"\n【最終回答】\n{answer1}")

    print("\n" + "=" * 55)

    # 測試用例 2：需要查詢物流狀態
    q2 = "SF123 和 SF456 分別在哪裡？"
    print(f"\n【問題 2】{q2}")
    answer2 = run_agent(q2)
    print(f"\n【最終回答】\n{answer2}")

    print("\n\nDay 3 完成。下一步：week2_day4/failure_modes.py — 故意讓它壞掉。")
