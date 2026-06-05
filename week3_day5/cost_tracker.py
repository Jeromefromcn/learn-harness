"""
week3_day5/cost_tracker.py
===========================
第三週 Day 5：Token 成本追蹤

對應手冊任務：
  - 用 Day 3 的 token 日誌，加一個 cost_usd（或 cost_cny）字段
  - 寫一個小腳本，能匯總：每次對話的平均成本、總成本、最貴的幾次
  - 想一個可能的成本節點（比如簡單問題路由到小模型）

為什麼成本追蹤很重要？
  對你自己：知道「這個功能每個月要花多少錢」
  對業務：是否能盈利取決於 token 費用 vs 帶來的商業價值
  對架構決策：什麼情況下應該用 Flash 而不是 Pro？
"""

import json
import os
from dataclasses import dataclass
from typing import Optional


# ============================================================
# DeepSeek 定價（截至 2025 年，請以官網為準）
# 單位：美元 / 每百萬 token
# 參考：https://api-docs.deepseek.com/zh-cn/quick_start/pricing
# ============================================================

PRICING = {
    # Flash 模型（快速，便宜）
    "deepseek-v4-flash": {
        "input_per_million": 0.27,   # 輸入 token
        "output_per_million": 1.10,  # 輸出 token（比輸入貴）
        "currency": "USD",
    },
    # Pro 模型（慢，貴）
    "deepseek-v4-pro": {
        "input_per_million": 0.27,
        "output_per_million": 1.10,
        "currency": "USD",
    },
    # 不認識的模型：保守估計
    "default": {
        "input_per_million": 1.0,
        "output_per_million": 3.0,
        "currency": "USD",
    },
}

# USD → CNY 匯率（近似，實際請查實時匯率）
USD_TO_CNY = 7.25


# ============================================================
# === 第三週第五天新增：成本計算函數 ===
# ============================================================

@dataclass
class CallCost:
    """一次 LLM 調用的成本信息。"""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cost_cny: float

    def __str__(self):
        return (
            f"模型={self.model}, "
            f"輸入={self.input_tokens}t, "
            f"輸出={self.output_tokens}t, "
            f"費用=${self.cost_usd:.6f} / ¥{self.cost_cny:.4f}"
        )


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> CallCost:
    """
    根據模型和 token 數計算成本。

    Args:
        model:         模型 ID
        input_tokens:  輸入 token 數
        output_tokens: 輸出 token 數

    Returns:
        CallCost 對象
    """
    # 查定價表，找不到就用 default
    pricing = PRICING.get(model, PRICING["default"])

    cost_usd = (
        input_tokens  / 1_000_000 * pricing["input_per_million"]
        + output_tokens / 1_000_000 * pricing["output_per_million"]
    )
    cost_cny = cost_usd * USD_TO_CNY

    return CallCost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost_usd, 8),
        cost_cny=round(cost_cny, 6),
    )


# ============================================================
# SessionCostTracker：追蹤一個 Agent 會話的累計成本
# ============================================================

class SessionCostTracker:
    """
    追蹤一次 Agent 對話的所有 LLM 調用成本。

    用法：
        tracker = SessionCostTracker()

        # 在 agent loop 裡每次調用後：
        tracker.record(resp)

        # 結束後打印報告：
        tracker.report()
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.calls: list[CallCost] = []

    def record(self, response) -> CallCost:
        """
        從 Anthropic API 響應中提取並記錄成本。
        在每次 client.messages.create() 之後調用。
        """
        cost = calc_cost(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        self.calls.append(cost)
        return cost

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_cost_cny(self) -> float:
        return sum(c.cost_cny for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    def report(self) -> dict:
        """生成本次會話的成本報告。"""
        if not self.calls:
            return {"error": "本次會話沒有任何 API 調用記錄"}

        report = {
            "session_id": self.session_id,
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_cost_cny": round(self.total_cost_cny, 4),
            "avg_cost_usd_per_call": round(self.total_cost_usd / len(self.calls), 8),
            "models_used": list({c.model for c in self.calls}),
        }

        print("\n" + "=" * 50)
        print(f"💰 成本報告（會話 ID：{self.session_id or '未設置'}）")
        print("=" * 50)
        print(f"  調用次數：{report['total_calls']}")
        print(f"  總 Token：輸入 {report['total_input_tokens']}，"
              f"輸出 {report['total_output_tokens']}")
        print(f"  總費用：${report['total_cost_usd']:.6f} "
              f"（約 ¥{report['total_cost_cny']:.4f}）")
        print(f"  均費用：${report['avg_cost_usd_per_call']:.8f} / 次調用")
        print(f"  使用模型：{', '.join(report['models_used'])}")
        print("=" * 50)

        return report


# ============================================================
# 日誌分析：從 observability.py 的 JSONL 日誌計算歷史成本
# ============================================================

def analyze_cost_from_log(log_file: str = "agent_calls.jsonl") -> dict:
    """
    讀取可觀測性日誌，計算歷史成本統計。

    這是把 Day 3 的日誌和 Day 5 的成本計算結合起來的例子：
    日誌記錄了「發生了什麼」，這裡計算「花了多少錢」。
    """
    records = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return {"error": f"日誌文件 {log_file} 不存在"}

    if not records:
        return {"error": "日誌文件為空"}

    total_cost_usd = 0.0
    per_model_costs: dict[str, float] = {}

    for r in records:
        model = r.get("model", "default")
        input_t = r.get("input_tokens", 0)
        output_t = r.get("output_tokens", 0)
        cost = calc_cost(model, input_t, output_t)
        total_cost_usd += cost.cost_usd
        per_model_costs[model] = per_model_costs.get(model, 0) + cost.cost_usd

    return {
        "total_calls": len(records),
        "total_cost_usd": round(total_cost_usd, 6),
        "total_cost_cny": round(total_cost_usd * USD_TO_CNY, 4),
        "per_model_cost_usd": {m: round(c, 6) for m, c in per_model_costs.items()},
        "avg_cost_usd_per_call": round(total_cost_usd / len(records), 8),
    }


# ============================================================
# 成本節點思考（面試常見問題）
# ============================================================

COST_OPTIMIZATION_IDEAS = """
成本優化思路（第五週可以作為面試素材）：

1. 問題路由（Question Routing）
   - 識別簡單問題（如「SF123 在哪裡」）→ 用 Flash 模型
   - 識別複雜問題（如需要多步推理）→ 用 Pro 模型
   - 估計節省：Flash 費用 ≈ Pro 的 1/4，簡單問題佔比通常 60-70%

2. 緩存常見答案（Semantic Cache）
   - 記錄問題的語義向量，相似問題直接返回緩存
   - 物流查詢：同一個追蹤號 5 分鐘內不再調 LLM，直接返回上次結果

3. 壓縮消息歷史（Context Compression）
   - 多輪對話時，舊的消息會佔用大量 token
   - 可以把早期對話壓縮成摘要，節省輸入 token

4. 工具結果過濾
   - 工具返回了很長的 JSON，但模型只需要其中幾個字段
   - 在把結果加入消息歷史之前，先提取關鍵字段
"""


# ============================================================
# 快速驗證
# ============================================================

if __name__ == "__main__":
    print("=== 成本追蹤模塊驗證 ===\n")

    # 模擬一次對話的成本計算
    print("模擬 3 輪對話的成本：")
    tracker = SessionCostTracker(session_id="demo-001")

    # 手動添加模擬記錄（不需要真實 API）
    class MockResponse:
        def __init__(self, model, in_t, out_t):
            self.model = model
            class Usage:
                input_tokens = in_t
                output_tokens = out_t
            self.usage = Usage()

    # 模擬：第1輪（模型決策）+ 第2輪（工具後繼續）+ 第3輪（最終答案）
    calls = [
        MockResponse("deepseek-v4-flash", 350, 80),    # 第1輪
        MockResponse("deepseek-v4-flash", 520, 150),   # 第2輪（帶工具結果）
        MockResponse("deepseek-v4-flash", 680, 200),   # 第3輪（最終答案）
    ]
    for resp in calls:
        cost = tracker.record(resp)
        print(f"  {cost}")

    tracker.report()

    print("\n成本優化思路：")
    print(COST_OPTIMIZATION_IDEAS)

    print("\n讀取歷史日誌成本（需要先運行幾次 agent）：")
    log_stats = analyze_cost_from_log()
    print(json.dumps(log_stats, ensure_ascii=False, indent=2))

    print("\nWeek 3 完成！")
    print("你的 agent 現在有：重試、斷路器、可觀測性、降級、成本追蹤")
    print("這就是手冊說的 harness '80% 核心'。")
    print("下一步：week4_day1 — 評估數據集，進入真正的差距。")
