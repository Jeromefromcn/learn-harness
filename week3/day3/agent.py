"""
week3_day3/agent.py
====================
在 week3_day2/agent.py 基礎上，加入結構化可觀測性日誌。

=== 第三週第三天修改 ===
  - 每次調用前記錄 t0，調用後立即調用 log_call()
  - 每個 turn 都有完整的記錄：輸入 hash、延遲、token、stop_reason
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEFAULT_MODEL, setup_anthropic_env
setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools
from guardrail import validate_output, ValidationError, SYSTEM_PROMPT_WITH_JSON_CONSTRAINT
from pydantic import BaseModel
from retry import with_retry
from circuit_breaker import CircuitBreaker, CircuitOpenError, call_with_timeout

# === 第三週第三天新增 ===
from observability import log_call, SessionCostTracker

client = anthropic.Anthropic()
llm_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

# 簡易成本追蹤（為 Day 5 做鋪墊，這裡先放進來）
class MockCostTracker:
    def record(self, resp): pass

try:
    from cost_tracker import SessionCostTracker as RealCostTracker
    _CostTrackerClass = RealCostTracker
except ImportError:
    _CostTrackerClass = MockCostTracker


def _call_model_raw(messages, tools, system):
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
    conversation_id: str = "",
) -> BaseModel | str:
    """帶重試 + 斷路器 + 可觀測性的 Agent 循環"""
    messages = [{"role": "user", "content": user_msg}]
    system = SYSTEM_PROMPT_WITH_JSON_CONSTRAINT if output_schema else None

    for turn in range(max_turns):
        if verbose:
            print(f"\n[Agent W3D3 第 {turn + 1} 輪] 調用模型...")

        try:
            # === 第三週第三天新增：記錄開始時間 ===
            t0 = time.time()

            resp = llm_breaker.call(call_with_timeout, _call_model_raw,
                                    30.0, messages, TOOLS, system)

            # === 第三週第三天新增：記錄調用詳情 ===
            record = log_call(messages, resp, t0,
                              conversation_id=conversation_id, turn_number=turn + 1)
            if verbose:
                print(f"[Obs] latency={record.latency_ms}ms, "
                      f"tokens={record.input_tokens}+{record.output_tokens}, "
                      f"stop={record.stop_reason}")

        except CircuitOpenError as e:
            print(f"[Agent] 斷路器開路：{e}")
            return "系統暫時不可用，請稍後再試。"
        except Exception as e:
            print(f"[Agent] 調用失敗：{e}")
            return f"[錯誤] {str(e)[:100]}"

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
    import uuid
    print("=" * 55)
    print("Week 3 Day 3：帶可觀測性的 Agent")
    print("=" * 55)
    conv_id = str(uuid.uuid4())[:8]
    result = run_agent("SF123 的物流狀態怎麼樣？", conversation_id=conv_id)
    print(f"\n結果：{result[:150] if isinstance(result, str) else result}")
    print(f"\n提示：查看 agent_calls.jsonl 文件，應該有這次調用的記錄。")
    print("用 `cat agent_calls.jsonl | python -m json.tool` 格式化查看。")
