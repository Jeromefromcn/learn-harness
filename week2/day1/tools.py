"""
week2_day1/tools.py
====================
第二週 Day 1：定義工具 + JSON Schema

對應手冊任務：
  - 先做 mock 工具出來（查物流狀態、算運費、查庫存）
  - 給每個工具寫好 JSON schema 描述
  - 現在不接真系統，用假數據，讓注意力放在 agent 邏輯本身

重要概念：
  JSON schema 是模型「知道」工具存在的方式——
  name + description + input_schema 三件套缺一不可。
  description 尤其關鍵：模型靠它判斷「什麼時候」調哪個工具。
"""

import sys
import os

# 從根目錄讀取全局配置（API key 等）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import setup_anthropic_env  # noqa: F401（供 run.py 使用）


# ============================================================
# 工具實現：Mock 物流系統
# ============================================================

def track_shipment(tracking_no: str) -> dict:
    """
    查詢物流追蹤號的當前狀態。
    真實項目中，這裡會調用順豐/極兔/UPS 的 API。
    現在用 mock 數據，讓你專注在 agent 邏輯上。
    """
    mock_data = {
        "SF123": {"status": "in_transit",  "eta_days": 2, "location": "上海轉運中心"},
        "SF456": {"status": "delivered",   "eta_days": 0, "location": "已送達"},
        "SF789": {"status": "processing",  "eta_days": 5, "location": "深圳倉"},
        "SF000": {"status": "exception",   "eta_days": -1, "location": "海關扣押"},
    }
    if tracking_no not in mock_data:
        # 返回 dict 而不是拋異常：讓模型知道「找不到」，而不是讓整個 loop 崩潰
        return {"status": "not_found", "message": f"找不到追蹤號 {tracking_no}"}
    return mock_data[tracking_no]


def calc_shipping_fee(weight_kg: float, zone: str) -> dict:
    """
    根據重量和目的地區域計算運費（港幣）。
    計費規則：基礎費 30 港幣 + 區域單價 × 重量。
    區域：A（近）= 18/kg，B（中）= 24/kg，C（遠）= 32/kg
    """
    zone_rates = {"A": 18, "B": 24, "C": 32}

    # 對明顯不合理的輸入拋異常（讓 dispatch_tools 捕獲並返回錯誤信息）
    if weight_kg <= 0:
        raise ValueError(f"重量必須為正數，收到：{weight_kg}")
    if zone not in zone_rates:
        # 不拋異常，返回錯誤信息讓模型自行理解
        return {"error": f"未知區域 '{zone}'，可選值：A / B / C"}

    rate = zone_rates[zone]
    fee = round(weight_kg * rate + 30, 2)   # 小數點精度統一到分
    return {
        "fee_hkd": fee,
        "weight_kg": weight_kg,
        "zone": zone,
        "breakdown": f"基礎費 30 + {weight_kg}kg × {rate} = {fee} HKD",
    }


def check_inventory(sku: str) -> dict:
    """
    查詢指定 SKU 的庫存數量。
    SKU-2 和 SKU-9 設計為無庫存，用於測試「庫存不足」的場景。
    """
    mock_inventory = {
        "SKU-1": 120,
        "SKU-2": 0,     # 無庫存
        "SKU-3": 45,
        "SKU-9": 0,     # 無庫存（Day 3 的多步測試用到這個）
    }
    if sku not in mock_inventory:
        return {"sku": sku, "in_stock": None, "error": "SKU 不存在系統中"}

    qty = mock_inventory[sku]
    return {
        "sku": sku,
        "in_stock": qty,
        "available": qty > 0,       # 布爾值方便模型判斷
        "status": "有貨" if qty > 0 else "缺貨",
    }


# ============================================================
# TOOLS：模型能「看到」的工具列表
# ============================================================
# 這是傳給 client.messages.create(tools=TOOLS) 的對象。
# 模型根據 name + description 決定什麼時候調哪個工具，
# 根據 input_schema 知道要傳哪些參數。

TOOLS = [
    {
        "name": "track_shipment",
        # description 要清楚說明「什麼情況下用這個工具」
        "description": "查詢物流追蹤號的當前狀態和預計送達時間。當用戶詢問包裹在哪裡、何時送到時使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "tracking_no": {
                    "type": "string",
                    "description": "物流追蹤號，例如 SF123、SF456",
                }
            },
            "required": ["tracking_no"],    # 必填字段，模型不能省略
        },
    },
    {
        "name": "calc_shipping_fee",
        "description": "根據包裹重量和目的地區域計算運費（港幣）。區域分 A/B/C 三檔。",
        "input_schema": {
            "type": "object",
            "properties": {
                "weight_kg": {
                    "type": "number",
                    "description": "包裹重量，單位：公斤，必須為正數",
                },
                "zone": {
                    "type": "string",
                    "enum": ["A", "B", "C"],    # 枚舉值讓模型不會瞎猜
                    "description": "目的地區域：A（港九）、B（新界）、C（離島）",
                },
            },
            "required": ["weight_kg", "zone"],
        },
    },
    {
        "name": "check_inventory",
        "description": "查詢指定商品 SKU 的庫存數量。在確認是否可以發貨時使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "商品 SKU 編號，例如 SKU-1、SKU-3",
                }
            },
            "required": ["sku"],
        },
    },
]


# ============================================================
# dispatch_tools：工具分發器（harness 的核心執行層）
# ============================================================

def dispatch_tools(content: list) -> list:
    """
    執行模型請求的所有工具調用，把結果格式化為 tool_result 格式返回。

    這個函數是 harness 的「執行引擎」：
      模型說「我要調用 track_shipment(SF123)」
      → dispatch_tools 執行 track_shipment("SF123")
      → 把結果包裝成 API 要求的 tool_result 格式
      → 返回給模型繼續推理

    Args:
        content: 模型回應的 content 列表（可能包含 tool_use 塊）

    Returns:
        tool_result 格式的列表，作為下一輪消息的 user content
    """
    # 工具名稱 → 函數的映射表
    # 新增工具時，在這裡加一行就夠了
    tool_map = {
        "track_shipment":   track_shipment,
        "calc_shipping_fee": calc_shipping_fee,
        "check_inventory":  check_inventory,
    }

    results = []
    for block in content:
        # 只處理 tool_use 類型的塊，忽略文字塊
        if block.type != "tool_use":
            continue

        tool_fn = tool_map.get(block.name)

        if tool_fn is None:
            # 模型請求了不存在的工具（幻覺），返回錯誤信息而不是崩潰
            result = {"error": f"工具 '{block.name}' 不存在"}
        else:
            try:
                # 用 ** 解包：block.input 是 dict，對應工具函數的參數
                result = tool_fn(**block.input)
            except Exception as e:
                # 捕獲工具執行錯誤，格式化後返回讓模型理解
                result = {"error": f"工具執行失敗：{str(e)}"}

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,    # 必須和模型請求的 id 對應
            "content": str(result),     # API 要求 content 是字符串
        })

    return results


# ============================================================
# 快速驗證（直接運行本文件時）
# ============================================================

if __name__ == "__main__":
    print("=== 工具函數驗證 ===\n")

    print("track_shipment('SF123'):", track_shipment("SF123"))
    print("track_shipment('UNKNOWN'):", track_shipment("UNKNOWN"))

    print("\ncalc_shipping_fee(5.0, 'B'):", calc_shipping_fee(5.0, "B"))
    print("calc_shipping_fee(0, 'A'):", end=" ")
    try:
        calc_shipping_fee(0, "A")
    except ValueError as e:
        print(f"ValueError: {e}")

    print("\ncheck_inventory('SKU-1'):", check_inventory("SKU-1"))
    print("check_inventory('SKU-2'):", check_inventory("SKU-2"))   # 缺貨

    print(f"\n共定義了 {len(TOOLS)} 個工具：{[t['name'] for t in TOOLS]}")
    print("Day 1 完成：工具函數 + JSON schema 就緒，Day 2 開始接 API。")
