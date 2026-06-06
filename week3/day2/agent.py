"""
week3_day2/agent.py
====================
在 week3_day1/agent.py 基礎上，加入斷路器 + 超時。

=== 第三週第二天修改 ===
  - 用斷路器包裝 API 調用：連續失敗 3 次就斷開，60 秒後嘗試恢復
  - 斷路器開路時捕獲 CircuitOpenError，走靜態降級響應
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEFAULT_MODEL, setup_anthropic_env
setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools
from guardrail import validate_output, ValidationError, ShippingQuote, SYSTEM_PROMPT_WITH_JSON_CONSTRAINT
from pydantic import BaseModel
from retry import with_retry

# === 第三週第二天新增 ===
from circuit_breaker import CircuitBreaker, CircuitOpenError, call_with_timeout

client = anthropic.Anthropic()

# 全局斷路器（生產環境應該是單例，保持跨請求的狀態）
llm_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)


@with_retry(max_attempts=2, base_delay=1.0)
def _call_model_with_breaker(
    messages: list, tools: list, system: str | None
) -> anthropic.types.Message:
    """
    帶重試 + 斷路器的 API 調用。
    裝飾器從外到內：重試 → 斷路器 → 實際調用。
    斷路器開路時直接拋 CircuitOpenError，不觸發重試（不可重試的錯誤）。
    """
    def _call():
        kwargs = {"model": DEFAULT_MODEL, "max_tokens": 1024,
                  "tools": tools, "messages": messages}
        if system:
            kwargs["system"] = system
        return client.messages.create(**kwargs)

    # 通過斷路器執行（30 秒超時）
    return llm_breaker.call(call_with_timeout, _call, 30.0)


CIRCUIT_OPEN_FALLBACK = "系統暫時不可用，請稍後再試。"


def run_agent(
    user_msg: str,
    output_schema: type[BaseModel] | None = None,
    max_turns: int = 6,
    verbose: bool = True,
) -> BaseModel | str:
    """帶重試 + 斷路器的 Agent 循環"""
    messages = [{"role": "user", "content": user_msg}]
    system = SYSTEM_PROMPT_WITH_JSON_CONSTRAINT if output_schema else None

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Agent W3D2 第 {turn + 1} 輪] 調用模型（帶重試+斷路器）...")
        try:
            resp = _call_model_with_breaker(messages, TOOLS, system)
        except CircuitOpenError as e:
            # === 第三週第二天新增：斷路器開路時的降級處理 ===
            print(f"[Agent] ⚡ 斷路器開路：{e}")
            return CIRCUIT_OPEN_FALLBACK
        except Exception as e:
            print(f"[Agent] 調用失敗：{e}")
            return f"[錯誤] {str(e)[:100]}"

        if verbose:
            print(f"[Agent] stop_reason = {resp.stop_reason}")
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            final_text = next((b.text for b in resp.content if hasattr(b, "text")), "")
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
    print("Week 3 Day 2：帶重試 + 斷路器的 Agent")
    print(f"斷路器當前狀態：{llm_breaker}")
    print("=" * 55)
    result = run_agent("查一下 SF123 的物流狀態")
    print(f"\n結果：{result[:150] if isinstance(result, str) else result}")
    print(f"\n斷路器狀態：{llm_breaker}")
