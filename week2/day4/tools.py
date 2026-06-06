"""week2_day4/tools.py - 繼承自 week2_day1/tools.py,本日未修改."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def track_shipment(tracking_no: str) -> dict:
    mock_data = {
        "SF123": {"status": "in_transit",  "eta_days": 2, "location": "上海轉運中心"},
        "SF456": {"status": "delivered",   "eta_days": 0, "location": "已送達"},
        "SF789": {"status": "processing",  "eta_days": 5, "location": "深圳倉"},
        "SF000": {"status": "exception",   "eta_days": -1, "location": "海關扣押"},
    }
    if tracking_no not in mock_data:
        return {"status": "not_found", "message": f"找不到追蹤號 {tracking_no}"}
    return mock_data[tracking_no]

def calc_shipping_fee(weight_kg: float, zone: str) -> dict:
    zone_rates = {"A": 18, "B": 24, "C": 32}
    if weight_kg <= 0:
        raise ValueError(f"重量必須為正數,收到:{weight_kg}")
    if zone not in zone_rates:
        return {"error": f"未知區域 '{zone}',可選值:A / B / C"}
    rate = zone_rates[zone]
    fee = round(weight_kg * rate + 30, 2)
    return {"fee_hkd": fee, "weight_kg": weight_kg, "zone": zone,
            "breakdown": f"基礎費 30 + {weight_kg}kg × {rate} = {fee} HKD"}

def check_inventory(sku: str) -> dict:
    mock_inventory = {"SKU-1": 120, "SKU-2": 0, "SKU-3": 45, "SKU-9": 0}
    if sku not in mock_inventory:
        return {"sku": sku, "in_stock": None, "error": "SKU 不存在系統中"}
    qty = mock_inventory[sku]
    return {"sku": sku, "in_stock": qty, "available": qty > 0,
            "status": "有貨" if qty > 0 else "缺貨"}

TOOLS = [
    {"name": "track_shipment",
     "description": "查詢物流追蹤號的當前狀態和預計送達時間.",
     "input_schema": {"type": "object",
                      "properties": {"tracking_no": {"type": "string"}},
                      "required": ["tracking_no"]}},
    {"name": "calc_shipping_fee",
     "description": "根據包裹重量和目的地區域計算運費(港幣).",
     "input_schema": {"type": "object",
                      "properties": {
                          "weight_kg": {"type": "number"},
                          "zone": {"type": "string", "enum": ["A", "B", "C"]}},
                      "required": ["weight_kg", "zone"]}},
    {"name": "check_inventory",
     "description": "查詢指定商品 SKU 的庫存數量.",
     "input_schema": {"type": "object",
                      "properties": {"sku": {"type": "string"}},
                      "required": ["sku"]}},
]

def dispatch_tools(content: list) -> list:
    tool_map = {"track_shipment": track_shipment,
                "calc_shipping_fee": calc_shipping_fee,
                "check_inventory": check_inventory}
    results = []
    for block in content:
        if block.type != "tool_use":
            continue
        tool_fn = tool_map.get(block.name)
        if tool_fn is None:
            result = {"error": f"工具 '{block.name}' 不存在"}
        else:
            try:
                result = tool_fn(**block.input)
            except Exception as e:
                result = {"error": f"工具執行失敗:{str(e)}"}
        results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})
    return results
