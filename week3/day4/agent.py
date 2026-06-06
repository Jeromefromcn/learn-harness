"""
week3_day4/agent.py
====================
在 week3_day3/agent.py 基礎上,加入優雅降級.

=== 第三週第四天修改 ===
  - 構建三級降級策略鏈:Pro -> Flash -> 靜態兜底
  - 斷路器開路 or 超時時走降級鏈,而不是直接返回錯誤
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import setup_anthropic_env
setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools
from guardrail import validate_output, ValidationError, SYSTEM_PROMPT_WITH_JSON_CONSTRAINT
from pydantic import BaseModel
from retry import with_retry
from circuit_breaker import CircuitBreaker, CircuitOpenError, call_with_timeout
from observability import log_call

# === 第三週第四天新增 ===
from fallback import FallbackChain, build_default_fallback_chain, static_fallback_strategy

client = anthropic.Anthropic()
llm_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

# 默認降級鏈:Pro -> Flash -> 靜態兜底
_fallback_chain = build_default_fallback_chain()


def _llm_loop(user_msg: str, system: str | None, max_turns: int,
              verbose: bool, conversation_id: str) -> str:
    """
    內部的 LLM 循環邏輯(不含降級).
    降級邏輯在 run_agent 外層處理.
    這樣分離讓降級策略可以替換整個循環,而不是嵌入其中.
    """
    from config import DEFAULT_MODEL
    messages = [{"role": "user", "content": user_msg}]

    for turn in range(max_turns):
        def _call():
            kwargs = {"model": DEFAULT_MODEL, "max_tokens": 1024,
                      "tools": TOOLS, "messages": messages}
            if system:
                kwargs["system"] = system
            return client.messages.create(**kwargs)

        t0 = time.time()
        resp = llm_breaker.call(call_with_timeout, _call, 30.0)
        log_call(messages, resp, t0, conversation_id=conversation_id, turn_number=turn + 1)

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            return next((b.text for b in resp.content if hasattr(b, "text")), "")

        if resp.stop_reason == "tool_use":
            tool_results = dispatch_tools(resp.content)
            messages.append({"role": "user", "content": tool_results})
            continue
        break

    return f"[錯誤] 超過最大輪次 {max_turns}"


def run_agent(
    user_msg: str,
    output_schema: type[BaseModel] | None = None,
    max_turns: int = 6,
    verbose: bool = True,
    conversation_id: str = "",
) -> BaseModel | str:
    """帶降級策略的 Agent 循環"""
    system = SYSTEM_PROMPT_WITH_JSON_CONSTRAINT if output_schema else None
    messages_for_fallback = [{"role": "user", "content": user_msg}]

    if verbose:
        print(f"\n[Agent W3D4] 開始執行(帶降級策略鏈)...")

    try:
        # 嘗試正常的 LLM 循環
        final_text = _llm_loop(user_msg, system, max_turns, verbose, conversation_id)
    except (CircuitOpenError, TimeoutError, Exception) as e:
        # === 第三週第四天新增:失敗時走降級鏈 ===
        print(f"[Fallback] 主循環失敗({type(e).__name__}),啟動降級策略鏈...")
        try:
            # 降級鏈嘗試次級模型,最壞情況返回靜態文字
            final_text, level = _fallback_chain.execute(messages_for_fallback, TOOLS)
            print(f"[Fallback] 第 {level+1} 級策略成功")
        except Exception as fallback_e:
            print(f"[Fallback] 所有策略失敗:{fallback_e}")
            return "系統繁忙,請稍後再試."

    if output_schema:
        try:
            return validate_output(final_text, output_schema)
        except ValidationError as e:
            print(f"[Agent] 輸出校驗失敗:{e}")
            return final_text

    return final_text


if __name__ == "__main__":
    print("=" * 55)
    print("Week 3 Day 4:帶降級策略的 Agent")
    print("=" * 55)
    result = run_agent("查一下 SF456 的物流狀態")
    print(f"\n結果:{result[:150] if isinstance(result, str) else result}")
    print("\nDay 4 完成:優雅降級策略鏈已接入 agent.")
