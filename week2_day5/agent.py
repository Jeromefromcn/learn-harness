"""
week2_day5/agent.py
====================
在 week2_day3/agent.py 基礎上，加入輸出校驗護欄。

=== 第二週第五天修改 ===
  - 新增 system_prompt，強制模型輸出 JSON
  - 在最終答案出來後，調用 guardrail.validate_output() 校驗
  - 校驗失敗時拋出 ValidationError（上層可以選擇重試）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_MODEL, setup_anthropic_env

setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools
# === 第二週第五天新增 ===
from guardrail import (
    validate_output,
    ValidationError,
    ShippingQuote,
    ShipmentStatus,
    InventoryCheck,
    SYSTEM_PROMPT_WITH_JSON_CONSTRAINT,
)
from pydantic import BaseModel

client = anthropic.Anthropic()


def run_agent(
    user_msg: str,
    output_schema: type[BaseModel] | None = None,
    max_turns: int = 6,
    verbose: bool = True,
) -> BaseModel | str:
    """
    帶輸出校驗的 Agent 循環。

    Args:
        user_msg:      用戶問題
        output_schema: 期望的輸出 Pydantic 類（None 則返回原始文字）
        max_turns:     最大輪次（護欄）
        verbose:       是否打印調試信息

    Returns:
        校驗成功 → 返回 Pydantic 對象
        校驗失敗 → 拋出 ValidationError
        output_schema=None → 返回原始文字字符串
    """
    messages = [{"role": "user", "content": user_msg}]

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Agent 第 {turn + 1} 輪] 調用模型...")

        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            tools=TOOLS,
            # === 第二週第五天新增：system prompt 強制 JSON 格式 ===
            system=SYSTEM_PROMPT_WITH_JSON_CONSTRAINT if output_schema else None,
            messages=messages,
        )

        if verbose:
            print(f"[Agent 第 {turn + 1} 輪] stop_reason = {resp.stop_reason}")

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            # 提取最終文字
            final_text = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    final_text = block.text
                    break

            # === 第二週第五天新增：護欄校驗 ===
            if output_schema is not None:
                if verbose:
                    print(f"[Agent] 開始校驗輸出（schema: {output_schema.__name__}）...")
                # validate_output 成功返回 Pydantic 對象，失敗拋 ValidationError
                validated = validate_output(final_text, output_schema)
                if verbose:
                    print("[Agent] ✅ 輸出校驗通過")
                return validated
            return final_text

        if resp.stop_reason == "tool_use":
            tool_results = dispatch_tools(resp.content)
            if verbose:
                names = [b.name for b in resp.content if b.type == "tool_use"]
                print(f"[Agent 第 {turn + 1} 輪] 執行工具：{names}")
            messages.append({"role": "user", "content": tool_results})
            continue

        break

    return f"[錯誤] 超過最大輪次 {max_turns}"


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Week 2 Day 5：帶輸出校驗的 Agent")
    print("=" * 55)

    # 測試 1：運費查詢 + ShippingQuote 校驗
    print("\n【測試 1】運費查詢（期望結構化輸出）")
    try:
        result = run_agent(
            "5kg 的包裹寄到 B 區要多少錢？預計幾天？",
            output_schema=ShippingQuote,
        )
        print(f"\n✅ 校驗通過：fee={result.fee_hkd} HKD, eta={result.eta_days}天")
        print(f"   摘要：{result.summary}")
    except ValidationError as e:
        print(f"\n❌ 輸出校驗失敗：{e}")

    # 測試 2：不要求結構化輸出（返回原始文字）
    print("\n\n【測試 2】閒聊（不要求結構化輸出）")
    result2 = run_agent("你好", output_schema=None, verbose=False)
    print(f"原始輸出：{result2[:100]}")

    print("\n\nWeek 2 完成！")
    print("你現在有了：工具 + agent loop + 輸出校驗護欄 + 失敗模式清單")
    print("下一步：week3_day1 — 加上重試、斷路器、可觀測性。")
