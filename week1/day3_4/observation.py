"""
week1_day3_4/observation.py
============================
第一週 Day 3-4:親眼見識 harness 的運作

對應手冊任務:
  - 安裝 Claude Code,在一個真實的現有項目裡跑一次真實任務
  - 只是觀察,不是實驗(不要跑後面幾週你要自己寫的那些實驗)
  - 記錄 harness 行為:用了哪些工具,測試失敗的反饋循環怎麼運作
  - 看 CLAUDE.md / AGENTS.md 這類配置文件的作用

本文件的用途:
  提供一個結構化的觀察記錄模板.邊看 Claude Code 邊填寫,
  填完後你對 harness 各組件的直覺就建立起來了.
  這份記錄第四週還會用到(評估數據集的靈感來源).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


# ============================================================
# 觀察記錄結構
# ============================================================

@dataclass
class ToolCall:
    """記錄一次工具調用的觀察"""
    tool_name: str              # 工具名稱,例如 "Read", "Edit", "Bash"
    input_summary: str          # 輸入參數的簡短描述
    output_summary: str         # 輸出結果的簡短描述
    triggered_by: str           # 為什麼模型選擇調用這個工具
    followed_by: str = ""       # 這次調用之後模型做了什麼


@dataclass
class FeedbackLoop:
    """記錄一次反饋循環的觀察(sensor 觸發 -> 模型反應)"""
    trigger: str                    # 什麼觸發了這個循環(測試失敗/lint 錯誤/...)
    sensor_type: Literal["computational", "inferential"]
    model_response: str             # 模型收到反饋後怎麼做的
    resolved: bool = False          # 最後解決了嗎
    turns_to_resolve: int = 0       # 幾輪之後解決


@dataclass
class HarnessObservation:
    """
    一次完整 Claude Code 任務的 harness 觀察記錄.

    填寫說明:
      不需要記錄所有細節,只記下讓你"哦,原來如此"的那些時刻.
      這些直覺後面幾週在你自己寫 harness 時會反復用到.
    """
    # 任務基本信息
    task_description: str           # 你給 Claude Code 的任務
    project_type: str               # 項目類型,例如 "Python Django"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # 工具使用觀察
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Guide 觀察(事前引導)
    guides_observed: list[str] = field(default_factory=list)
    # 例如:["CLAUDE.md 裡的測試指令影響了它跑 pytest 的方式",
    #         "工具描述讓它知道 Edit 比 Write 更適合修改文件"]

    # Sensor 觀察(事後測量)
    feedback_loops: list[FeedbackLoop] = field(default_factory=list)

    # 人工介入點
    human_interventions: list[str] = field(default_factory=list)
    # 例如:"它要 rm -rf 我叫停了,說明護欄需要覆蓋破壞性命令"

    # 自由筆記
    surprises: list[str] = field(default_factory=list)
    # "出乎意料"的觀察,往往是最有學習價值的地方


def print_observation_summary(obs: HarnessObservation):
    """打印觀察記錄的摘要報告"""
    print(f"\n{'='*50}")
    print(f"任務:{obs.task_description}")
    print(f"項目:{obs.project_type}")
    print(f"{'='*50}")

    print(f"\n[工具調用]共 {len(obs.tool_calls)} 次")
    for i, tc in enumerate(obs.tool_calls, 1):
        print(f"  {i}. {tc.tool_name}: {tc.input_summary}")
        print(f"     原因:{tc.triggered_by}")

    print(f"\n[反饋循環]共 {len(obs.feedback_loops)} 次")
    for fl in obs.feedback_loops:
        status = "✓ 已解決" if fl.resolved else "✗ 未解決"
        print(f"  [{fl.sensor_type}] {fl.trigger} -> {status}")

    print(f"\n[人工介入]共 {len(obs.human_interventions)} 次")
    for hi in obs.human_interventions:
        print(f"  - {hi}")

    print(f"\n[令我意外的觀察]")
    for s in obs.surprises:
        print(f"  ★ {s}")


# ============================================================
# 示例:如何填寫一份觀察記錄
# ============================================================

EXAMPLE_OBSERVATION = HarnessObservation(
    task_description="幫我給 user_service.py 的 get_user 函數補全測試",
    project_type="Python FastAPI",

    tool_calls=[
        ToolCall(
            tool_name="Read",
            input_summary="user_service.py",
            output_summary="讀取到 get_user 函數的實現",
            triggered_by="需要先理解函數邏輯才能寫測試",
        ),
        ToolCall(
            tool_name="Bash",
            input_summary="pytest tests/test_user.py -v",
            output_summary="3 passed, 0 failed",
            triggered_by="寫完測試後自動驗證是否通過",
            followed_by="測試通過,任務完成",
        ),
    ],

    guides_observed=[
        "CLAUDE.md 裡寫了'測試必須用 pytest',它確實只用了 pytest 而非 unittest",
        "工具的 description 字段讓它優先選擇 Edit 而不是 Write(避免覆蓋整個文件)",
    ],

    feedback_loops=[
        FeedbackLoop(
            trigger="第一次 pytest 失敗(assert 方向寫反了)",
            sensor_type="computational",
            model_response="讀取錯誤信息,定位到具體行數,直接修改",
            resolved=True,
            turns_to_resolve=1,
        ),
    ],

    human_interventions=[
        "它要刪除舊測試文件,我叫停了,讓它改為重構",
    ],

    surprises=[
        "它主動在測試文件頂部加了 fixture,我沒要求--說明它從 CLAUDE.md 裡理解了項目慣例",
        "測試失敗的反饋循環幾乎是即時的,說明計算式 sensor 的速度優勢很顯著",
    ],
)


# ============================================================
# 自測問題清單(Day 3-4 的任務核心)
# ============================================================

CHECKLIST = """
觀察完成後,確認你能回答以下問題:

□ Claude Code 用了哪些工具?(Read / Edit / Bash / ...)
□ 測試失敗的反饋循環是怎麼觸發的?模型收到錯誤信息後做了什麼?
□ 你在哪些地方需要人工介入?為什麼?
□ CLAUDE.md(如果有)影響了哪些行為?這就是"事前引導 guide"的具體形態.
□ 有沒有讓你意外的時刻?(往往是最值得記錄的洞察)

完成記錄後,對照 week1_day5/gap_analysis.py 評估你自己系統的缺口.
"""


if __name__ == "__main__":
    # 打印示例記錄
    print("=== 示例觀察記錄 ===")
    print_observation_summary(EXAMPLE_OBSERVATION)

    # 打印自測清單
    print("\n=== 觀察自測清單 ===")
    print(CHECKLIST)

    # 動手:複製下面這個模板,填入你自己的觀察
    print("\n=== 你的觀察記錄模板 ===")
    print("my_observation = HarnessObservation(")
    print('    task_description="",')
    print('    project_type="",')
    print('    tool_calls=[],')
    print('    guides_observed=[],')
    print('    feedback_loops=[],')
    print('    human_interventions=[],')
    print('    surprises=[],')
    print(")")
