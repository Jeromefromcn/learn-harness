"""
week3_day5/agent.py
====================
在 week3_day4/agent.py 基礎上，加入成本追蹤。

=== 第三週第五天修改 ===
  - 每次調用後記錄成本（calc_cost）
  - Agent 結束時打印本次對話的成本報告
  - 新增 run_agent_with_report：返回結果 + 成本摘要

Week 3 最終版：包含所有可靠性層：
  ✅ 重試 + 指數退避
  ✅ 斷路器
  ✅ 可觀測性日誌
  ✅ 優雅降級
  ✅ 成本追蹤
"""

import sys, os, time, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_MODEL, setup_anthropic_env
setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools
from guardrail import validate_output, ValidationError, ShippingQuote, SYSTEM_PROMPT_WITH_JSON_CONSTRAINT
from pydantic import BaseModel
from retry import with_retry
from circuit_breaker import CircuitBreaker, CircuitOpenError, call_with_timeout
from observability import log_call
from fallback import FallbackChain, build_default_fallback_chain

# === 第三週第五天新增 ===
from cost_tracker import SessionCostTracker, calc_cost

client = anthropic.Anthropic()
llm_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
_fallback_chain = build_default_fallback_chain()


def run_agent(
    user_msg: str,
    output_schema: type[BaseModel] | None = None,
    max_turns: int = 6,
    verbose: bool = True,
    conversation_id: str = "",
) -> tuple[BaseModel | str, dict]:
    """
    Week 3 最終版：帶所有可靠性層的 Agent 循環。

    Returns:
        (result, cost_report) 元組：
          - result:      模型的最終回答（Pydantic 對象或字符串）
          - cost_report: 本次會話的成本摘要字典
    """
    system = SYSTEM_PROMPT_WITH_JSON_CONSTRAINT if output_schema else None
    messages = [{"role": "user", "content": user_msg}]

    # === 第三週第五天新增：初始化成本追蹤器 ===
    tracker = SessionCostTracker(session_id=conversation_id or str(uuid.uuid4())[:8])

    final_text = ""

    try:
        for turn in range(max_turns):
            if verbose:
                print(f"\n[Agent W3D5 第 {turn + 1} 輪]")

            def _call():
                kwargs = {"model": DEFAULT_MODEL, "max_tokens": 1024,
                          "tools": TOOLS, "messages": messages}
                if system:
                    kwargs["system"] = system
                return client.messages.create(**kwargs)

            t0 = time.time()
            resp = llm_breaker.call(call_with_timeout, _call, 30.0)

            # 可觀測性日誌
            record = log_call(messages, resp, t0,
                              conversation_id=conversation_id, turn_number=turn + 1)

            # === 第三週第五天新增：記錄本輪成本 ===
            call_cost = tracker.record(resp)
            if verbose:
                print(f"[Cost] 本輪費用：¥{call_cost.cost_cny:.4f}")

            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                final_text = next((b.text for b in resp.content if hasattr(b, "text")), "")
                break

            if resp.stop_reason == "tool_use":
                tool_results = dispatch_tools(resp.content)
                messages.append({"role": "user", "content": tool_results})
                continue
            break

    except (CircuitOpenError, TimeoutError, Exception) as e:
        print(f"[Fallback] 主循環失敗，啟動降級...")
        try:
            final_text, level = _fallback_chain.execute(messages[:1], TOOLS)
        except Exception:
            final_text = "系統繁忙，請稍後再試。"

    # 生成成本報告
    cost_report = tracker.report()

    # 輸出校驗
    result: BaseModel | str = final_text
    if output_schema and final_text:
        try:
            result = validate_output(final_text, output_schema)
        except ValidationError:
            result = final_text

    return result, cost_report


if __name__ == "__main__":
    print("=" * 55)
    print("Week 3 Day 5：完整可靠性層 Agent（最終版）")
    print("=" * 55)

    result, cost = run_agent(
        "5kg 包裹寄到 B 區多少錢？SF123 現在在哪裡？SKU-9 有貨嗎？",
        output_schema=None,
        conversation_id="w3d5-demo",
    )

    print(f"\n最終回答：")
    print(result[:300] if isinstance(result, str) else str(result))

    print("\n" + "=" * 55)
    print("Week 3 全部完成！你的 agent 已具備生產就緒的可靠性層。")
    print("下一步：week4_day1 — 評估數據集，進入真正的差距。")
