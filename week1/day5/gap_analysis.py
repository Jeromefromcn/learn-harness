"""
week1_day5/gap_analysis.py
===========================
第一週 Day 5:對自己系統做一次缺口分析(Gap Analysis)

對應手冊任務:
  - 建一張表,列出六大組件:工具/行動,護欄,反饋循環,
    可觀測性,評估,成本/降級
  - 按"已有 / 待改善 / 全新"分類
  - 標記"全新"的部分--那就是後四週要重點攻克的

本文件的用途:
  幫你用結構化的方式評估你後端系統目前的 agent 就緒程度.
  這份分析直接決定後四週的學習重心--
  手冊說"評估(evaluation)幾乎都是全新的",這也是真正的難點.

使用方法:
  修改下面的 MY_SYSTEM_STATUS,填入你自己的評估.
  然後運行本文件,它會打印出優先級建議.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


# ============================================================
# 六大 Harness 組件的定義
# ============================================================

class Component(Enum):
    """
    Harness 的六大組件(來自手冊的"組件視角")

    這是評估一個 agent 系統"就緒程度"的標準框架.
    每個組件都可以獨立評估,也可以組合使用.
    """
    TOOLS = "tools_and_actions"
    # 工具/行動:agent 能調用哪些外部能力?
    # 例子:API 調用,數據庫查詢,文件讀寫,計算函數

    GUARDRAILS = "guardrails"
    # 護欄:有哪些機制防止 agent 做出不可接受的輸出?
    # 例子:輸出格式校驗,敏感詞過濾,金額上限,確認步驟

    FEEDBACK_LOOPS = "feedback_loops"
    # 反饋循環:工具執行失敗時,錯誤信息怎麼回饋給模型?
    # 例子:測試失敗 -> 錯誤信息 -> 模型重試,schema 驗證失敗 -> 提示修正

    OBSERVABILITY = "observability"
    # 可觀測性:每次調用的信息有多少被記錄下來?
    # 例子:延遲,token 數,輸入 hash,stop_reason,工具調用序列

    EVALUATION = "evaluation"
    # 評估:怎麼衡量 agent 的輸出質量?
    # 例子:精確率/召回率(計算式),LLM-as-judge(推斷式)

    COST_AND_FALLBACK = "cost_and_fallback"
    # 成本/降級:Token 費用怎麼追蹤?主模型故障時怎麼降級?
    # 例子:每次對話的 cost_usd,Flash/Pro 自動切換,緩存兜底


# ============================================================
# 評估狀態
# ============================================================

class Status(Enum):
    EXISTING = "already_have"          # 已有:可用,無需大改
    NEEDS_IMPROVEMENT = "improve"      # 待改善:有但不夠健壯
    NEW = "brand_new"                  # 全新:需要從零開始建


@dataclass
class ComponentAssessment:
    """對一個組件的評估結果"""
    component: Component
    status: Status
    current_state: str       # 現狀描述(1-2 句)
    gap: str                 # 缺口:缺少什麼
    priority_week: int | None = None  # 建議哪一週重點攻克(2-5)


# ============================================================
# 填入你自己的評估(這是 Day 5 的核心任務)
# ============================================================
# 說明:下面是一個"典型後端工程師初次評估"的示例,
#       你需要根據自己的實際情況修改每一項.

MY_SYSTEM_STATUS: list[ComponentAssessment] = [

    ComponentAssessment(
        component=Component.TOOLS,
        status=Status.EXISTING,
        current_state="已有封裝好的 API 客戶端,DB query 函數,有完整的錯誤處理",
        gap="工具沒有 JSON schema 描述,模型無法知道如何調用",
        priority_week=2,  # Week 2 補上 schema
    ),

    ComponentAssessment(
        component=Component.GUARDRAILS,
        status=Status.NEEDS_IMPROVEMENT,
        current_state="有基本的輸入校驗(Pydantic),但沒有針對 LLM 輸出的格式校驗",
        gap="缺少對 agent 最終輸出結構的 schema 驗證",
        priority_week=2,  # Week 2 Day 5
    ),

    ComponentAssessment(
        component=Component.FEEDBACK_LOOPS,
        status=Status.NEEDS_IMPROVEMENT,
        current_state="工具會拋異常,但沒有統一的方式把錯誤格式化後回饋給模型",
        gap="缺少 dispatch_tools 層面的錯誤捕獲和格式化",
        priority_week=2,
    ),

    ComponentAssessment(
        component=Component.OBSERVABILITY,
        status=Status.EXISTING,
        current_state="有 Python logging,記錄基本的請求信息",
        gap="沒有 LLM 調用專用的結構化日誌(token 數,延遲,stop_reason)",
        priority_week=3,
    ),

    ComponentAssessment(
        component=Component.EVALUATION,
        status=Status.NEW,  # ← 手冊說:幾乎人人都是全新
        current_state="沒有系統化的評估;靠人工試用感覺結果好不好",
        gap="完全缺失:測試集,精確率/召回率計算,LLM-as-judge",
        priority_week=4,  # ← 第四週的重點
    ),

    ComponentAssessment(
        component=Component.COST_AND_FALLBACK,
        status=Status.NEEDS_IMPROVEMENT,
        current_state="有重試邏輯,但沒有 token 費用追蹤,沒有主/備模型切換",
        gap="缺少 cost_usd 字段,斷路器,備用模型路由",
        priority_week=3,
    ),
]


# ============================================================
# 輸出分析報告
# ============================================================

STATUS_EMOJI = {
    Status.EXISTING: "✅",
    Status.NEEDS_IMPROVEMENT: "🔧",
    Status.NEW: "🆕",
}

STATUS_LABEL = {
    Status.EXISTING: "已有",
    Status.NEEDS_IMPROVEMENT: "待改善",
    Status.NEW: "全新",
}


def print_gap_analysis(assessments: list[ComponentAssessment]):
    print("=" * 60)
    print("Harness 組件缺口分析報告")
    print("=" * 60)

    # 統計
    counts = {s: 0 for s in Status}
    for a in assessments:
        counts[a.status] += 1

    print(f"\n概況:✅ 已有 {counts[Status.EXISTING]} 個  "
          f"🔧 待改善 {counts[Status.NEEDS_IMPROVEMENT]} 個  "
          f"🆕 全新 {counts[Status.NEW]} 個")

    # 詳細列表
    print("\n" + "-" * 60)
    for a in assessments:
        emoji = STATUS_EMOJI[a.status]
        label = STATUS_LABEL[a.status]
        week = f"Week {a.priority_week}" if a.priority_week else "N/A"

        print(f"\n{emoji} [{a.component.value}][{label}] -> {week} 重點攻克")
        print(f"   現狀:{a.current_state}")
        print(f"   缺口:{a.gap}")

    # 優先級建議
    new_items = [a for a in assessments if a.status == Status.NEW]
    print("\n" + "=" * 60)
    print("⚡ 優先級建議(全新組件,後四週重點):")
    for a in sorted(new_items, key=lambda x: x.priority_week or 99):
        print(f"  Week {a.priority_week}: {a.component.value}")

    print("\n手冊說:'評估(evaluation)幾乎都是全新的,這是真正的難點.'")
    print("-> 請確認你的評估組件狀態是否標記為 NEW.")


if __name__ == "__main__":
    print_gap_analysis(MY_SYSTEM_STATUS)
    print("\n" + "=" * 60)
    print('下一步:把"全新"的組件,按 priority_week 的順序攻克.')
    print("Week 2 開始,第一行真正的代碼在 week2_day1/tools.py.")
