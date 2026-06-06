"""
week2_day4/failure_modes.py
============================
第二週 Day 4:親手讓 agent 壞掉,建立失敗模式目錄

對應手冊任務:
  - 傳入邊界參數(空字符串,負數重量,不存在的 SKU)
  - 讓工具主動拋異常,觀察整個循環是否直接崩潰
  - 做一個"工具等一下才答"的問題,看模型是否幻覺出答案
  - 把觀察到的失敗模式記錄下來

為什麼要故意讓它壞?
  你不可能為沒有親眼見過的失敗模式設計防禦.
  這份"失敗清單"是第三,四週設計護欄和評估的直接輸入.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEFAULT_MODEL, setup_anthropic_env

setup_anthropic_env()
import anthropic

from tools import TOOLS, dispatch_tools, calc_shipping_fee, check_inventory
from agent import run_agent


# ============================================================
# 故意壞掉的工具(用於測試 harness 的容錯性)
# ============================================================

def track_shipment_broken(tracking_no: str) -> dict:
    """
    永遠拋異常的工具版本--測試 dispatch_tools 的錯誤捕獲.
    如果 dispatch_tools 沒有 try/except,整個循環會崩潰.
    """
    raise RuntimeError(f"物流系統宕機,無法查詢 {tracking_no}")


def check_inventory_slow(sku: str) -> dict:
    """
    模擬"工具等很久才回答"的場景.
    (本 mock 不真的 sleep,只是返回空結果,觀察模型行為)
    """
    return {}   # 空 dict:模型怎麼處理"工具沒給有用信息"?


# ============================================================
# 失敗場景測試
# ============================================================

def test_bad_inputs():
    """場景 1:邊界/非法輸入"""
    print("\n[場景 1]邊界輸入測試")
    print("-" * 40)

    # 1a. 空字符串
    print("\n1a. 空字符串追蹤號:")
    result = run_agent("幫我查一下 '' 的物流狀態", verbose=False)
    print(f"  -> {result[:150]}")

    # 1b. 負數重量(會觸發 ValueError)
    print("\n1b. 負數重量(-1kg 到 A 區):")
    result = run_agent("寄 -1kg 的包裹到 A 區多少錢?", verbose=False)
    print(f"  -> {result[:150]}")

    # 1c. 不存在的 SKU
    print("\n1c. 不存在的 SKU:")
    result = run_agent("查一下 NONEXISTENT-SKU 的庫存", verbose=False)
    print(f"  -> {result[:150]}")


def test_tool_exception():
    """
    場景 2:工具拋異常
    把宕機版工具臨時替換進去,看 dispatch_tools 能否優雅捕獲.
    """
    print("\n\n[場景 2]工具拋異常測試")
    print("-" * 40)

    # 臨時替換 dispatch_tools 用的工具映射
    # (這是 monkey patching 技術,僅用於測試,生產不推薦)
    import week2_day4.tools as tools_module
    original_fn = tools_module.track_shipment
    tools_module.track_shipment = track_shipment_broken

    print("\n工具主動拋 RuntimeError,看整個循環是否崩潰:")
    try:
        result = run_agent("查一下 SF123 的物流", verbose=True)
        print(f"\n-> agent 沒有崩潰,返回:{result[:150]}")
    except Exception as e:
        print(f"\n-> agent 崩潰了!錯誤:{e}")
        print("  問題在哪?查看 dispatch_tools 裡的 try/except")
    finally:
        # 恢復原始工具(不污染後續測試)
        tools_module.track_shipment = original_fn


def test_tool_returns_nothing():
    """
    場景 3:工具返回空/無用信息
    觀察模型在拿不到有效數據時是否會幻覺出答案.
    """
    print("\n\n[場景 3]工具返回空響應")
    print("-" * 40)

    # 用返回空 dict 的版本替換
    import week2_day4.tools as tools_module
    original_fn = tools_module.check_inventory
    tools_module.check_inventory = check_inventory_slow

    print("\n工具返回空 dict,看模型是否幻覺出庫存數字:")
    result = run_agent("SKU-1 還有多少庫存?", verbose=False)
    print(f"\n-> agent 回答:{result[:200]}")
    print("  觀察:模型說了什麼?是否承認"沒有拿到有用信息"?")

    tools_module.check_inventory = original_fn


def test_ambiguous_question():
    """
    場景 4:問題模糊,工具"等一下才答"(工具答不了)
    觀察模型是否用已知信息幻覺出看似合理的答案.
    """
    print("\n\n[場景 4]模糊問題(工具無法回答)")
    print("-" * 40)
    print("\n問:'幫我查一下最近的訂單狀態'(沒有追蹤號)")
    result = run_agent("幫我查一下最近的訂單狀態", verbose=False)
    print(f"\n-> {result[:200]}")
    print("  觀察:模型是否要求你提供追蹤號,而不是猜一個?")


# ============================================================
# 失敗模式記錄(供後四週參考)
# ============================================================

FAILURE_MODES: list[dict] = [
    {
        "id": "FM-01",
        "name": "工具拋異常 -> 循環崩潰",
        "trigger": "工具函數 raise Exception",
        "current_behavior": "如果 dispatch_tools 沒有 try/except,整個 agent 崩潰",
        "solution": "Week 3 Day 1:在 dispatch_tools 外層加 try/except + 重試",
        "sensor_type": "computational",
    },
    {
        "id": "FM-02",
        "name": "工具返回空/無效數據 -> 模型幻覺",
        "trigger": "工具返回 {} 或 None",
        "current_behavior": "模型可能捏造看似合理的答案",
        "solution": "Week 2 Day 5:護欄強制校驗輸出格式",
        "sensor_type": "inferential",
    },
    {
        "id": "FM-03",
        "name": "非法輸入 -> 工具 ValueError",
        "trigger": "負數重量,空字符串等",
        "current_behavior": "dispatch_tools 捕獲錯誤並返回錯誤信息,模型一般能理解",
        "solution": "Week 2 Day 5:在工具 schema 裡加 minimum/pattern 校驗",
        "sensor_type": "computational",
    },
    {
        "id": "FM-04",
        "name": "問題模糊 -> 模型不請求工具",
        "trigger": "沒有具體追蹤號/SKU 的問題",
        "current_behavior": "模型通常會反問,但偶爾會瞎猜",
        "solution": "Week 4:在評估集裡加入"模糊問題"類別",
        "sensor_type": "inferential",
    },
    {
        "id": "FM-05",
        "name": "無限工具循環(理論風險)",
        "trigger": "工具結果觸發更多工具調用",
        "current_behavior": "max_turns 護欄會在第 6 輪停下",
        "solution": "Week 3:加斷路器(circuit breaker)",
        "sensor_type": "computational",
    },
]


def print_failure_modes():
    print("\n\n" + "=" * 55)
    print("失敗模式目錄(第三,四週設計防禦的依據)")
    print("=" * 55)
    for fm in FAILURE_MODES:
        print(f"\n[{fm['id']}] {fm['name']}")
        print(f"  觸發條件:{fm['trigger']}")
        print(f"  當前行為:{fm['current_behavior']}")
        print(f"  解決方案:{fm['solution']}")
        print(f"  Sensor 類型:{fm['sensor_type']}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Week 2 Day 4:失敗模式測試")
    print("=" * 55)
    print("目標:建立對失敗模式的直覺,而不是修復它們.")
    print("(修復在 Week 3 和 Week 4)")

    test_bad_inputs()
    test_tool_exception()
    test_tool_returns_nothing()
    test_ambiguous_question()
    print_failure_modes()

    print("\n\nDay 4 完成.下一步:week2_day5/guardrail.py - 第一個護欄.")
