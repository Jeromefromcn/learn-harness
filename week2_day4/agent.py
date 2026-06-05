"""week2_day4/agent.py — 繼承自 week2_day3/agent.py，本日未修改。
（week2_day4 的重點是 failure_modes.py，agent 本身不變）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_MODEL, setup_anthropic_env
setup_anthropic_env()
import anthropic
from tools import TOOLS, dispatch_tools

client = anthropic.Anthropic()

def run_agent(user_msg: str, max_turns: int = 6, verbose: bool = True) -> str:
    """完整的 Agent 循環（詳細注釋見 week2_day3/agent.py）"""
    messages = [{"role": "user", "content": user_msg}]
    for turn in range(max_turns):
        if verbose:
            print(f"\n[Agent 第 {turn + 1} 輪] 調用模型...")
        resp = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=1024, tools=TOOLS, messages=messages)
        if verbose:
            print(f"[Agent 第 {turn + 1} 輪] stop_reason = {resp.stop_reason}")
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
        if resp.stop_reason == "tool_use":
            tool_results = dispatch_tools(resp.content)
            if verbose:
                names = [b.name for b in resp.content if b.type == "tool_use"]
                print(f"[Agent 第 {turn + 1} 輪] 執行工具：{names}")
            messages.append({"role": "user", "content": tool_results})
            continue
        break
    return f"[錯誤] 超過最大輪次 {max_turns}"
