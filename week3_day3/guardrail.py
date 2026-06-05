"""
week2_day5/guardrail.py
========================
第二週 Day 5：第一個護欄——結構化輸出校驗

對應手冊任務：
  - 用 Pydantic 定義最終輸出的結構
  - 在提示詞裡明確要求模型只輸出 JSON
  - 把 validate_output() 接到 agent 的最終響應上
  - 校驗成功和校驗失敗兩條路都跑通

核心概念：
  護欄（guardrail）= 計算式 sensor 的一種。
  它不是「希望」模型輸出正確格式，而是「強制」校驗。
  校驗失敗 → 拋異常 → 上層可以選擇重試或降級。

  Pydantic 的優勢：
    1. 類型安全（fee_hkd 不會是字符串）
    2. 業務規則校驗（費用必須為正數）
    3. 失敗信息精確（知道是哪個字段、什麼規則）
"""

import json
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal


# ============================================================
# === 第二週第五天新增：輸出 Schema 定義 ===
# ============================================================

class ShipmentStatus(BaseModel):
    """
    物流查詢的結構化輸出。
    模型最終回答必須符合這個結構，否則視為校驗失敗。
    """
    tracking_no: str
    status: Literal["in_transit", "delivered", "processing", "exception", "not_found"]
    eta_days: int
    summary: str    # 給用戶看的自然語言描述（1-2句）

    @field_validator("eta_days")
    @classmethod
    def eta_must_be_non_negative_for_active(cls, v):
        # exception 狀態允許 -1（表示「未知」）
        # 其他狀態的 eta_days 不應為負數
        if v < -1:
            raise ValueError(f"eta_days 不合法：{v}")
        return v


class ShippingQuote(BaseModel):
    """
    運費報價的結構化輸出。
    fee_hkd 必須為正數，eta_days 不能超過 30 天。
    """
    fee_hkd: float
    eta_days: int
    zone: Literal["A", "B", "C"]
    summary: str    # 給用戶看的自然語言描述

    @field_validator("fee_hkd")
    @classmethod
    def fee_must_be_positive(cls, v):
        # 業務規則：運費必須為正數（免費情況用 0 但要明確標記）
        if v <= 0:
            raise ValueError(f"運費必須為正數，得到：{v}")
        return v

    @field_validator("eta_days")
    @classmethod
    def eta_must_be_reasonable(cls, v):
        if v < 0 or v > 30:
            raise ValueError(f"eta_days 不合法（0-30），得到：{v}")
        return v


class InventoryCheck(BaseModel):
    """庫存查詢的結構化輸出。"""
    sku: str
    in_stock: int
    available: bool
    summary: str

    @model_validator(mode="after")
    def available_must_match_stock(self):
        # 業務一致性：available 必須和 in_stock 保持一致
        expected = self.in_stock > 0
        if self.available != expected:
            raise ValueError(
                f"available={self.available} 與 in_stock={self.in_stock} 不一致"
            )
        return self


# ============================================================
# validate_output：護欄入口函數
# ============================================================

class ValidationError(Exception):
    """輸出校驗失敗時拋出，帶有詳細的失敗原因。"""
    pass


def validate_output(raw: str, schema_class: type[BaseModel]) -> BaseModel:
    """
    把 agent 的最終文字輸出解析並校驗為 Pydantic 對象。

    失敗時拋 ValidationError（而不是讓程序崩潰）：
      → 上層可以選擇：把錯誤信息反饋給模型讓它重試
      → 或者：記錄失敗，走降級路徑

    Args:
        raw:          agent 最終輸出的文字（期望是 JSON 字符串）
        schema_class: 要校驗的 Pydantic 模型類

    Returns:
        校驗成功的 Pydantic 對象

    Raises:
        ValidationError: 格式不對或業務規則失敗
    """
    # 步驟 1：嘗試從文字中提取 JSON
    # 模型有時會在 JSON 前後加說明文字，需要先提取
    json_str = _extract_json(raw)
    if json_str is None:
        raise ValidationError(
            f"輸出中找不到有效的 JSON。\n"
            f"原始輸出（前200字）：{raw[:200]}"
        )

    # 步驟 2：Pydantic 校驗（類型 + 業務規則）
    try:
        return schema_class.model_validate_json(json_str)
    except Exception as e:
        raise ValidationError(
            f"JSON 格式正確但業務規則校驗失敗：{e}\n"
            f"原始 JSON：{json_str[:300]}"
        )


def _extract_json(text: str) -> str | None:
    """
    從可能帶有說明文字的輸出中提取 JSON 字符串。
    優先找 ```json ... ``` 代碼塊，其次嘗試直接解析整段文字。
    """
    # 嘗試提取 ```json ... ``` 代碼塊
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 嘗試找第一個 { 到最後一個 } 的範圍
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)   # 驗證是否是合法 JSON
            return candidate
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# 帶校驗的 system prompt（護欄的「事前引導」部分）
# ============================================================

SYSTEM_PROMPT_WITH_JSON_CONSTRAINT = """
你是一個物流客服助手。使用提供的工具查詢信息後，
你的最終回答必須是一個合法的 JSON 對象，格式如下：

查詢物流時：
{"tracking_no": "SF123", "status": "in_transit", "eta_days": 2, "summary": "包裹在途中，預計2天送達"}

計算運費時：
{"fee_hkd": 120.0, "eta_days": 2, "zone": "B", "summary": "寄到B區需要120港幣，預計2天"}

查庫存時：
{"sku": "SKU-1", "in_stock": 120, "available": true, "summary": "SKU-1 有120件庫存"}

規則：
1. 只輸出 JSON，不要加任何說明文字
2. 不要用 Markdown 代碼塊包裹
3. 所有字段都必須填寫
"""


# ============================================================
# 快速驗證
# ============================================================

if __name__ == "__main__":
    print("=== Guardrail 校驗測試 ===\n")

    # 測試 1：合法輸出
    valid_json = '{"fee_hkd": 120.0, "eta_days": 2, "zone": "B", "summary": "需要120港幣"}'
    try:
        result = validate_output(valid_json, ShippingQuote)
        print(f"✅ 校驗成功：fee={result.fee_hkd} HKD, zone={result.zone}")
    except ValidationError as e:
        print(f"❌ 校驗失敗：{e}")

    # 測試 2：費用為負數（業務規則失敗）
    invalid_fee = '{"fee_hkd": -50.0, "eta_days": 2, "zone": "A", "summary": "退款"}'
    try:
        validate_output(invalid_fee, ShippingQuote)
        print("❌ 應該失敗但沒失敗！")
    except ValidationError as e:
        print(f"✅ 業務規則正確攔截：{str(e)[:80]}")

    # 測試 3：輸出帶多餘說明文字
    with_extra_text = '根據計算，運費如下：{"fee_hkd": 90.0, "eta_days": 3, "zone": "C", "summary": "90港幣"}'
    try:
        result = validate_output(with_extra_text, ShippingQuote)
        print(f"✅ 從混合文字中提取 JSON 成功：fee={result.fee_hkd}")
    except ValidationError as e:
        print(f"❌ 提取失敗：{e}")

    # 測試 4：不是 JSON 的輸出
    plain_text = "根據查詢，SF123 目前在上海轉運中心，預計兩天後送達。"
    try:
        validate_output(plain_text, ShipmentStatus)
        print("❌ 應該失敗但沒失敗！")
    except ValidationError as e:
        print(f"✅ 純文字被正確攔截：{str(e)[:60]}")

    print("\nDay 5 完成。Week 2 的物流 agent 已有基本護欄。")
    print("下一步：week3_day1/retry.py — 加上重試和指數退避。")
