"""
week1_day1_2/concepts.py
========================
第一週 Day 1-2:建立核心心智模型(閱讀筆記 + 概念代碼化)

對應手冊任務:
  - 閱讀 Birgitta Böckeler 在 Martin Fowler 網站的文章
  - 用一句話寫下 Agent = Model + Harness 的定義
  - 記下 guides / sensors 兩類控制的區別
  - 閱讀"Humans and Agents in Software Engineering Loops"

本文件的用途:
  把手冊裡的抽象概念用 Python 數據結構表達出來,幫助你把語言
  轉化為可操作的心智模型.這裡沒有真正運行的 API 調用,只是
  "把概念釘在代碼上".
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any


# ============================================================
# 核心定義:Agent = Model + Harness
# ============================================================

class Agent:
    """
    Agent 的本質定義(概念模型,非真實實現).

    關鍵洞察:
      Model  = 決策者(LLM),負責"想什麼"
      Harness = 環境/腳手架,負責"怎麼做"

    Harness 是 model 以外的一切:工具,提示詞,校驗,
    日誌,重試,降級...這些才是工程師真正的工作對象.
    """

    def __init__(self, model: "Model", harness: "Harness"):
        self.model = model      # 決策核心(LLM API 調用)
        self.harness = harness  # 工程支撐層(我們親手建造的)


@dataclass
class Model:
    """
    抽象的模型層.
    工程師對這一層的控制手段有限:選擇供應商,選擇版本,寫提示詞.
    """
    provider: str       # 例如 "DeepSeek", "Anthropic"
    model_id: str       # 例如 "deepseek-v4-flash"
    system_prompt: str  # 系統提示詞:harness 影響模型行為的主要手段之一


@dataclass
class Harness:
    """
    Harness 層:model 以外的所有工程基礎設施.

    Harness 的兩大控制機制:
      guides  = 事前引導(告訴模型該怎麼做)
      sensors = 事後測量(觀察模型做了什麼)
    """
    guides: list["Guide"] = field(default_factory=list)
    sensors: list["Sensor"] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)         # 可調用的工具
    max_turns: int = 6                                       # 防死循環護欄


# ============================================================
# Guides:事前引導機制
# ============================================================

class GuideType(Enum):
    """Guides 的分類(根據手冊)"""
    SYSTEM_PROMPT = "system_prompt"     # 系統提示詞:全局指令
    FEW_SHOT = "few_shot"               # 少樣本示例:教模型風格
    TOOL_DESCRIPTION = "tool_desc"      # 工具描述:告訴模型有什麼可用
    CONTEXT_INJECTION = "context"       # 上下文注入:給模型必要的背景


@dataclass
class Guide:
    """
    Guide = 事前引導,在模型做決策"之前"介入.

    類比:像交通標誌,在駕駛人遇到路口前就給出指示,
    而不是等他轉錯了彎才糾正.

    例子:
      - CLAUDE.md / AGENTS.md 文件就是 system prompt guide
      - 工具的 description 字段是 tool_description guide
    """
    guide_type: GuideType
    content: str
    description: str = ""  # 這個 guide 解決了什麼問題


# ============================================================
# Sensors:事後測量機制
# ============================================================

class SensorType(Enum):
    """
    Sensors 分兩大類(這是手冊最重要的區分之一)

    計算式 (Computational):確定性強,結果明確
      優點:速度快,費用低,結果可重複
      例子:linter 通過/失敗,JSON schema 校驗,測試通過率

    推斷式 (Inferential):需要語義判斷,帶概率性
      優點:能評估無法窮舉的質量維度(語氣,相關性)
      例子:LLM-as-judge,語義相似度
    """
    COMPUTATIONAL = "computational"  # 確定性,例如 lint/test/schema
    INFERENTIAL = "inferential"      # 概率性,例如 LLM-as-judge


@dataclass
class Sensor:
    """
    Sensor = 事後測量,在模型輸出"之後"評估質量.

    類比:像質檢員,生產線末端抽查產品是否合格.
    如果不合格,可以觸發重試或降級.

    設計原則:
      優先用計算式 sensor(便宜,確定);
      只有在語義質量確實無法用規則捕捉時,才用推斷式.
    """
    sensor_type: SensorType
    name: str
    description: str
    # 實際的測量函數:接受模型輸出,返回分數或通過/失敗
    measure: Callable[[Any], Any] | None = None


# ============================================================
# 調性維度(Day 2 的核心概念)
# ============================================================

class MaintenanceDimension(Enum):
    """
    Harness 工程要維護的三個調性維度(來自 Day 2 閱讀).

    Harness 不是一次性配置,而是持續迭代的工程實踐,
    這三個維度提供了評估當前 agent 健康狀況的框架.
    """
    MAINTAINABILITY = "maintainability"
    # 可維護性:代碼/提示詞是否清晰?其他人能接手嗎?
    # 信號:prompt 超過 500 行,工具描述含糊,沒有測試

    ARCHITECTURE = "architecture"
    # 架構健康:模塊邊界清晰嗎?依賴關係合理嗎?
    # 信號:agent 直接調用 DB,工具函數混入業務邏輯

    FUNCTIONAL_BEHAVIOUR = "functional_behaviour"
    # 功能行為:agent 是否做了正確的事?
    # 信號:評估分數下降,用戶反饋變差,新 case 失敗


# ============================================================
# 核心洞察總結
# ============================================================

KEY_INSIGHTS = {
    "agent_equation": (
        "Agent = Model + Harness."
        "Model 是決策者(LLM),Harness 是工程師建造的一切其他東西."
    ),
    "harness_is_not_config": (
        "Harness 不是一次性的配置,而是持續迭代的工程實踐."
        "就像後端的中間件,需要隨業務演化不斷調整."
    ),
    "guides_vs_sensors": (
        "Guides(前置引導)決定模型的行為方向;"
        "Sensors(後置測量)評估模型輸出的質量."
        "兩者共同構成 feedback loop."
    ),
    "sensor_types": (
        "計算式 sensor:確定性強(lint/test),優先使用;"
        "推斷式 sensor:帶概率性(LLM-as-judge),用於語義質量."
    ),
    "human_judgment": (
        "Harness 試圖封裝人類的業務知識和工具,"
        "但無法完全代替人類的判斷--這正是資深工程師的核心價值所在."
    ),
}


if __name__ == "__main__":
    # 打印核心洞察,幫助自我檢查理解深度
    print("=== 第一週 Day 1-2 核心概念 ===\n")
    for key, insight in KEY_INSIGHTS.items():
        print(f"[{key}]")
        print(f"  {insight}\n")

    # 自測問題(對照手冊的 Day 1 任務)
    print("=== 自測問題 ===")
    print("1. 用一句話解釋:什麼是 harness?(不能用技術術語)")
    print("2. 給同事解釋計算式和推斷式 sensor 的區別,各舉一個例子.")
    print('3. 你的系統裡,評估(evaluation)相關的部分是"已有/待建/全新"?')
