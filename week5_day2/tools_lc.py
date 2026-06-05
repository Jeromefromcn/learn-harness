"""
week5_day1/tools_lc.py
=======================
第五週 Day 1：把工具改成 LangChain @tool 格式

與 week2_day1/tools.py 的差異：
  - 用 @tool 裝飾器（LangChain 的工具格式）
  - 函數的 docstring 就是工具 description（框架自動提取）
  - 不再需要手寫 JSON schema——LangChain 從類型注解自動生成

第五週核心問題：這省了什麼？要注意什麼？
"""

from langchain.tools import tool


# ============================================================
# === 第五週第一天新增：@tool 裝飾器格式 ===
# ============================================================

@tool
def track_shipment(tracking_no: str) -> str:
    """
    查詢物流追蹤號的當前狀態和預計送達時間。
    當用戶詢問包裹在哪裡、何時送到時使用。

    注意：docstring 的第一行就是工具的 description，
    LangChain 會自動提取它傳給模型。
    """
    mock_data = {
        "SF123": {"status": "in_transit",  "eta_days": 2, "location": "上海轉運中心"},
        "SF456": {"status": "delivered",   "eta_days": 0, "location": "已送達"},
        "SF789": {"status": "processing",  "eta_days": 5, "location": "深圳倉"},
        "SF000": {"status": "exception",   "eta_days": -1, "location": "海關扣押"},
    }
    if tracking_no not in mock_data:
        return str({"status": "not_found", "message": f"找不到追蹤號 {tracking_no}"})
    return str(mock_data[tracking_no])


@tool
def calc_shipping_fee(weight_kg: float, zone: str) -> str:
    """
    根據包裹重量和目的地區域計算運費（港幣）。
    區域分 A（港九）、B（新界）、C（離島）三檔。
    """
    zone_rates = {"A": 18, "B": 24, "C": 32}
    if weight_kg <= 0:
        return str({"error": f"重量必須為正數，收到：{weight_kg}"})
    if zone not in zone_rates:
        return str({"error": f"未知區域 '{zone}'，可選值：A / B / C"})
    rate = zone_rates[zone]
    fee = round(weight_kg * rate + 30, 2)
    return str({"fee_hkd": fee, "weight_kg": weight_kg, "zone": zone})


@tool
def check_inventory(sku: str) -> str:
    """
    查詢指定商品 SKU 的庫存數量。
    在確認是否可以發貨時使用。
    """
    mock_inventory = {"SKU-1": 120, "SKU-2": 0, "SKU-3": 45, "SKU-9": 0}
    if sku not in mock_inventory:
        return str({"sku": sku, "in_stock": None, "error": "SKU 不存在系統中"})
    qty = mock_inventory[sku]
    return str({"sku": sku, "in_stock": qty, "available": qty > 0,
                "status": "有貨" if qty > 0 else "缺貨"})


# 工具列表（傳給 LangChain agent）
TOOLS = [track_shipment, calc_shipping_fee, check_inventory]


if __name__ == "__main__":
    # 直接調用 LangChain tool（不通過 LLM）
    print("=== LangChain @tool 格式驗證 ===\n")
    print(f"track_shipment.name: {track_shipment.name}")
    print(f"track_shipment.description: {track_shipment.description[:80]}")
    print(f"直接調用: {track_shipment.invoke({'tracking_no': 'SF123'})}")
    print(f"\n共 {len(TOOLS)} 個工具：{[t.name for t in TOOLS]}")
    print("\n對比 week2_day1/tools.py：省掉了手寫 JSON schema，")
    print("代價是工具返回值必須是字符串（@tool 的限制）。")
