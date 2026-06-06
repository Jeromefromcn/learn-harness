"""
week3_day1/agent.py
====================
在 week2_day5/agent.py 基礎上，加入重試 + 指數退避。

=== 第三週第一天修改 ===
  - 用 @with_retry 包裝 client.messages.create() 調用
  - 區分可重試和不可重試的異常
  - 新增 retry_with_llm_feedback：校驗失敗時把錯誤反饋給模型
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEFAULT_MODEL, setup_anthropic_env

setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools
from guardrail import (
    validate_output, ValidationError, ShippingQuote,
    SYSTEM_PROMPT_WITH_JSON_CONSTRAINT
)
from pydantic import BaseModel

# === 第三週第一天新增 ===
from retry import with_retry, retry_with_llm_feedback

client = anthropic.Anthropic()


@with_retry(max_attempts=3, base_delay=1.0)
def _call_model(messages: list, tools: list, system: str | None) -> anthropic.types.Message:
    """
    帶重試的 API 調用。用裝飾器包裝，讓 run_agent 保持簡潔。
    只有這一層知道重試邏輯，上層的 run_agent 不需要關心。
    """
    kwargs = {"model": DEFAULT_MODEL, "max_tokens": 1024,
              "tools": tools, "messages": messages}
    if system:
        kwargs["system"] = system
    return client.messages.create(**kwargs)


def run_agent(
    user_msg: str,
    output_schema: type[BaseModel] | None = None,
    max_turns: int = 6,
    verbose: bool = True,
) -> BaseModel | str:
    """帶重試的 Agent 循環（詳細注釋見 week2_day5/agent.py）"""
    messages = [{"role": "user", "content": user_msg}]
    system = SYSTEM_PROMPT_WITH_JSON_CONSTRAINT if output_schema else None

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Agent W3D1 第 {turn + 1} 輪] 調用模型（帶重試）...")

        # === 第三週第一天：使用帶重試的 API 調用 ===
        resp = _call_model(messages, TOOLS, system)

        if verbose:
            print(f"[Agent] stop_reason = {resp.stop_reason}")

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in resp.content if hasattr(b, "text")), ""
            )
            if output_schema:
                return validate_output(final_text, output_schema)
            return final_text

        if resp.stop_reason == "tool_use":
            tool_results = dispatch_tools(resp.content)
            messages.append({"role": "user", "content": tool_results})
            continue
        break

    return f"[錯誤] 超過最大輪次 {max_turns}"


if __name__ == "__main__":
    print("=" * 55)
    print("Week 3 Day 1：帶重試的 Agent")
    print("=" * 55)
    result = run_agent("5kg 的包裹寄到 B 區要多少錢？", output_schema=ShippingQuote)
    if isinstance(result, ShippingQuote):
        print(f"\n✅ 結果：{result.fee_hkd} HKD，{result.summary}")
    else:
        print(f"\n結果：{result}")
    print("\nDay 1 完成：重試邏輯已接入 agent loop。")
