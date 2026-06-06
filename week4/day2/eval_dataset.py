"""
week4_day1/eval_dataset.py
===========================
第四週 Day 1:建立評估數據集

對應手冊任務:
  - 寫 20-50 個測試案例,每個包含:輸入問題,期望工具調用,可接受的輸出
  - 把第二週發現的失敗模式(邊界,模糊問題)都做進去
  - 存成 JSONL 或 CSV,讓數據集可以重複使用

沒有數據集就沒有評估.
沒有評估,你就不知道改了提示詞是真的變好了還是只是感覺好了.
這份數據集第四週會用兩次,第五週在 LangSmith 裡還會再用一次.
"""

import json
import os
from pathlib import Path
from typing import Literal


# ============================================================
# 測試案例數據結構
# ============================================================

def make_case(
    case_id: str,
    question: str,
    expected_tools: list[str],
    acceptable_outputs: list[str],
    category: Literal["normal", "edge", "ambiguous", "multi_step"],
    notes: str = "",
) -> dict:
    """
    創建一個評估測試案例.

    Args:
        case_id:          唯一 ID,例如 "TC-001"
        question:         發給 agent 的用戶問題
        expected_tools:   期望調用的工具列表(可以是空列表)
        acceptable_outputs: 可接受的輸出關鍵詞列表(任意一個匹配即通過)
        category:         案例類別(用於分層分析)
        notes:            備注(記錄這個案例要測什麼)

    Returns:
        dict(可直接序列化為 JSONL)
    """
    return {
        "id": case_id,
        "question": question,
        "expected_tools": expected_tools,
        "acceptable_outputs": acceptable_outputs,
        "category": category,
        "notes": notes,
    }


# ============================================================
# === 第四週第一天新增:評估數據集 ===
# ============================================================
# 共 30 個案例,覆蓋:正常流程 / 邊界情況 / 多步問題 / 模糊問題

EVAL_DATASET = [

    # ── 正常流程:物流查詢 ──────────────────────────────────────

    make_case("TC-001", "幫我查一下 SF123 的物流狀態",
              expected_tools=["track_shipment"],
              acceptable_outputs=["in_transit", "在途", "上海", "2天"],
              category="normal",
              notes="最基本的物流查詢"),

    make_case("TC-002", "SF456 送到了嗎?",
              expected_tools=["track_shipment"],
              acceptable_outputs=["delivered", "已送達", "送到了"],
              category="normal",
              notes="已送達狀態"),

    make_case("TC-003", "SF789 什麼時候能到?",
              expected_tools=["track_shipment"],
              acceptable_outputs=["processing", "5天", "處理中"],
              category="normal",
              notes="詢問 ETA"),

    make_case("TC-004", "SF000 的包裹出什麼問題了嗎?",
              expected_tools=["track_shipment"],
              acceptable_outputs=["exception", "異常", "海關"],
              category="normal",
              notes="異常狀態"),

    # ── 正常流程:運費計算 ──────────────────────────────────────

    make_case("TC-005", "5公斤包裹寄到B區要多少錢?",
              expected_tools=["calc_shipping_fee"],
              acceptable_outputs=["150", "HKD", "港幣"],
              category="normal",
              notes="標準運費計算(5kg × 24 + 30 = 150)"),

    make_case("TC-006", "1.5kg 寄到 A 區的運費是多少?",
              expected_tools=["calc_shipping_fee"],
              acceptable_outputs=["57", "HKD"],
              category="normal",
              notes="小包裹,A 區(1.5 × 18 + 30 = 57)"),

    make_case("TC-007", "10kg 的貨物寄到 C 區要多少運費?",
              expected_tools=["calc_shipping_fee"],
              acceptable_outputs=["350", "HKD"],
              category="normal",
              notes="大包裹,遠距離(10 × 32 + 30 = 350)"),

    # ── 正常流程:庫存查詢 ──────────────────────────────────────

    make_case("TC-008", "SKU-1 還有貨嗎?",
              expected_tools=["check_inventory"],
              acceptable_outputs=["120", "有貨", "available"],
              category="normal",
              notes="有庫存的商品"),

    make_case("TC-009", "查一下 SKU-2 的庫存",
              expected_tools=["check_inventory"],
              acceptable_outputs=["0", "缺貨", "無庫存", "沒有"],
              category="normal",
              notes="無庫存的商品"),

    make_case("TC-010", "SKU-3 庫存充足嗎?",
              expected_tools=["check_inventory"],
              acceptable_outputs=["45", "有貨", "充足"],
              category="normal",
              notes="有庫存,措辭評估"),

    # ── 多步問題:需要連續調用多個工具 ─────────────────────────

    make_case("TC-011", "5kg 包裹寄到 B 區多少錢?另外 SKU-9 有貨嗎?",
              expected_tools=["calc_shipping_fee", "check_inventory"],
              acceptable_outputs=["150", "缺貨", "SKU-9"],
              category="multi_step",
              notes="運費 + 庫存,需要兩個工具"),

    make_case("TC-012", "查 SF123 的物流,順便告訴我 3kg 寄 A 區多少錢",
              expected_tools=["track_shipment", "calc_shipping_fee"],
              acceptable_outputs=["in_transit", "84", "HKD"],
              category="multi_step",
              notes="物流 + 運費(3 × 18 + 30 = 84)"),

    make_case("TC-013", "SKU-1 有貨的話,幫我查一下 5kg 寄 B 區的費用",
              expected_tools=["check_inventory", "calc_shipping_fee"],
              acceptable_outputs=["120", "有貨", "150"],
              category="multi_step",
              notes="先查庫存,再計算運費"),

    make_case("TC-014", "SF456 到哪了?另外 SKU-2 和 SKU-3 分別有多少庫存?",
              expected_tools=["track_shipment", "check_inventory"],
              acceptable_outputs=["delivered", "0", "45"],
              category="multi_step",
              notes="物流 + 兩個 SKU 庫存"),

    # ── 邊界情況:來自 Week 2 Day 4 的失敗模式 ────────────────

    make_case("TC-015", "查一下 SF999 的物流",
              expected_tools=["track_shipment"],
              acceptable_outputs=["not_found", "找不到", "不存在", "SF999"],
              category="edge",
              notes="不存在的追蹤號"),

    make_case("TC-016", "NONEXISTENT-SKU 有多少庫存?",
              expected_tools=["check_inventory"],
              acceptable_outputs=["不存在", "找不到", "SKU 不存在"],
              category="edge",
              notes="不存在的 SKU"),

    make_case("TC-017", "查一下 D 區的運費,5kg",
              expected_tools=["calc_shipping_fee"],
              acceptable_outputs=["未知區域", "D", "A/B/C", "錯誤"],
              category="edge",
              notes="不存在的區域"),

    make_case("TC-018", "幫我查 SKU-2 的庫存,如果有貨計算寄 2kg 到 A 區的費用",
              expected_tools=["check_inventory"],
              acceptable_outputs=["0", "缺貨", "無需計算", "無法", "沒有庫存"],
              category="edge",
              notes="庫存為零時,運費計算應跳過"),

    # ── 模糊問題:模型是否要求澄清 ───────────────────────────

    make_case("TC-019", "查一下我最近的訂單",
              expected_tools=[],    # 期望:不調用工具,而是要求澄清
              acceptable_outputs=["追蹤號", "什麼", "哪", "請提供", "無法"],
              category="ambiguous",
              notes="沒有追蹤號,期望模型反問"),

    make_case("TC-020", "幫我計算運費",
              expected_tools=[],
              acceptable_outputs=["重量", "區域", "請告訴", "多少", "哪個"],
              category="ambiguous",
              notes="沒有重量和區域,期望反問"),

    make_case("TC-021", "有沒有貨?",
              expected_tools=[],
              acceptable_outputs=["SKU", "哪個", "什麼商品", "請提供"],
              category="ambiguous",
              notes="沒有 SKU,期望反問"),

    # ── 更多正常流程(湊夠 30 個)────────────────────────────

    make_case("TC-022", "SF123 預計幾天後到?",
              expected_tools=["track_shipment"],
              acceptable_outputs=["2", "兩天", "eta"],
              category="normal",
              notes="只詢問 ETA"),

    make_case("TC-023", "8公斤的包裹寄到C區多少錢?",
              expected_tools=["calc_shipping_fee"],
              acceptable_outputs=["286", "HKD"],
              category="normal",
              notes="8 × 32 + 30 = 286"),

    make_case("TC-024", "SKU-1 的庫存數量是多少?",
              expected_tools=["check_inventory"],
              acceptable_outputs=["120"],
              category="normal",
              notes="精確詢問數量"),

    make_case("TC-025", "SF000 遇到問題了嗎,現在什麼情況?",
              expected_tools=["track_shipment"],
              acceptable_outputs=["exception", "異常", "海關"],
              category="normal",
              notes="異常狀態,措辭不同"),

    make_case("TC-026", "2kg 包裹到 B 區多少運費?",
              expected_tools=["calc_shipping_fee"],
              acceptable_outputs=["78", "HKD"],
              category="normal",
              notes="2 × 24 + 30 = 78"),

    make_case("TC-027", "SKU-3 和 SKU-9 哪個有貨?",
              expected_tools=["check_inventory"],
              acceptable_outputs=["SKU-3", "45", "有貨"],
              category="multi_step",
              notes="比較兩個 SKU 的庫存"),

    make_case("TC-028", "SF789 現在在哪裡?還要等幾天?",
              expected_tools=["track_shipment"],
              acceptable_outputs=["深圳", "5", "processing"],
              category="normal",
              notes="查詢位置和 ETA"),

    make_case("TC-029", "我想寄一個很重的包裹到 C 區",
              expected_tools=[],
              acceptable_outputs=["重量", "多重", "幾公斤", "請告訴"],
              category="ambiguous",
              notes="沒有具體重量,期望反問"),

    make_case("TC-030", "SF123 已經到了嗎?",
              expected_tools=["track_shipment"],
              acceptable_outputs=["in_transit", "尚未", "還沒", "在途"],
              category="normal",
              notes="詢問是否已送達(答案是否)"),
]


# ============================================================
# 數據集 I/O
# ============================================================

def save_dataset(dataset: list[dict], filepath: str = "dataset.jsonl"):
    """把數據集存成 JSONL 文件(每行一個 JSON 對象)."""
    with open(filepath, "w", encoding="utf-8") as f:
        for case in dataset:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"✅ 數據集已保存:{filepath}(共 {len(dataset)} 個案例)")


def load_dataset(filepath: str = "dataset.jsonl") -> list[dict]:
    """從 JSONL 文件讀取數據集."""
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def print_dataset_summary(dataset: list[dict]):
    """打印數據集的分類統計."""
    from collections import Counter
    categories = Counter(c["category"] for c in dataset)
    print(f"\n數據集摘要(共 {len(dataset)} 個案例):")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:12s}: {count} 個")
    print(f"\n  期望工具調用覆蓋:")
    all_tools = []
    for c in dataset:
        all_tools.extend(c["expected_tools"])
    tool_counts = Counter(all_tools)
    for tool, count in sorted(tool_counts.items()):
        print(f"    {tool}: {count} 次")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Week 4 Day 1:創建評估數據集")
    print("=" * 55)

    print_dataset_summary(EVAL_DATASET)

    # 保存到 JSONL 文件
    output_path = os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    save_dataset(EVAL_DATASET, output_path)

    # 驗證讀取
    loaded = load_dataset(output_path)
    print(f"\n讀取驗證:從 {output_path} 讀到 {len(loaded)} 個案例 ✅")
    print(f"\n第一個案例示例:")
    print(json.dumps(loaded[0], ensure_ascii=False, indent=2))

    print("\nDay 1 完成.下一步:week4_day2/metrics.py - 計算精確率和召回率.")
