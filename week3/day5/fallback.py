"""
week3_day4/fallback.py
=======================
第三週 Day 4：優雅降級（Graceful Degradation）

對應手冊任務：
  - 設計降級策略：主模型失敗 → 切換到更小/更快的模型
  - 再失敗 → 返回緩存結果或「請稍後再試」的安全響應
  - 把上週「等一下才答」的問題接上降級路徑，確認系統不崩潰

核心概念：
  優雅降級 = 系統在部分故障時，能降低服務質量但不完全失去功能。

  類比你做 SRE 時的經驗：
    主數據庫掛了 → 讀緩存（舊數據） → 至少不是 500 錯誤
    CDN 故障 → 直接打源站 → 慢但能用

  在 LLM 系統裡：
    Pro 模型超時 → 切 Flash 模型（便宜但能用）
    Flash 也掛 → 返回緩存的常見問題答案
    緩存也沒有 → 返回「系統繁忙，請稍後重試」的安全響應
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_FLASH, MODEL_PRO, DEFAULT_MODEL, setup_anthropic_env

setup_anthropic_env()
import anthropic


# ============================================================
# === 第三週第四天新增：降級策略鏈 ===
# ============================================================

class FallbackExhaustedError(Exception):
    """所有降級選項都試過了，還是失敗。"""
    pass


class FallbackChain:
    """
    降級策略鏈。

    把多個策略按優先級排列：
      1. 主模型（Pro，能力最強）
      2. 次級模型（Flash，便宜快速）
      3. 緩存響應（離線，無 API 依賴）
      4. 安全兜底（靜態文字，永不失敗）

    每個策略失敗時自動嘗試下一個，直到所有策略耗盡。
    整個過程對調用方透明：它只知道最終得到了一個結果，
    不知道（也不需要知道）用了哪個降級路徑。
    """

    def __init__(self, strategies: list, verbose: bool = True):
        """
        Args:
            strategies: 降級策略列表，按優先級從高到低排列
                        每個策略是一個 callable，返回結果或拋異常
            verbose:    是否打印降級日誌
        """
        self.strategies = strategies
        self.verbose = verbose
        self._last_successful_strategy_index = -1

    def execute(self, *args, **kwargs) -> tuple[any, int]:
        """
        依次嘗試策略鏈中的每個策略。

        Returns:
            (result, strategy_index): 成功的結果和使用的策略序號（0=主策略）
            strategy_index > 0 表示發生了降級

        Raises:
            FallbackExhaustedError: 所有策略都失敗
        """
        last_error = None

        for i, strategy in enumerate(self.strategies):
            try:
                result = strategy(*args, **kwargs)
                self._last_successful_strategy_index = i
                if i > 0 and self.verbose:
                    print(f"[Fallback] ⚠️  使用了第 {i+1} 級降級策略")
                return result, i

            except Exception as e:
                last_error = e
                if self.verbose:
                    print(
                        f"[Fallback] 第 {i+1} 級策略失敗 "
                        f"（{type(e).__name__}: {str(e)[:60]}）"
                        + ("，嘗試下一級..." if i < len(self.strategies) - 1 else "")
                    )

        raise FallbackExhaustedError(
            f"所有 {len(self.strategies)} 個降級策略都失敗。"
            f"最後一個錯誤：{last_error}"
        )


# ============================================================
# 具體降級策略實現
# ============================================================

def make_llm_strategy(model: str, timeout_sec: float = 30.0):
    """
    創建一個使用指定模型的 LLM 調用策略。

    Args:
        model:       模型 ID
        timeout_sec: 超時時間（秒）

    Returns:
        一個 callable，接受 messages 列表，返回響應文字
    """
    client = anthropic.Anthropic()

    def strategy(messages: list, tools: list = None) -> str:
        from circuit_breaker import call_with_timeout

        def _call():
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                messages=messages,
                tools=tools or [],
            )
            # 簡化：只返回文字內容
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        result = call_with_timeout(_call, timeout_sec=timeout_sec)
        print(f"[Fallback] 使用模型：{model}，耗時限制：{timeout_sec}秒")
        return result

    strategy.__name__ = f"llm_{model}"
    return strategy


# 靜態緩存響應（最後的安全兜底）
STATIC_FALLBACK_RESPONSES = {
    "shipping": "抱歉，物流查詢系統暫時不可用，請稍後重試或致電客服 400-XXX-XXXX。",
    "inventory": "庫存系統暫時維護中，請稍後查詢。",
    "fee": "運費計算暫時不可用，標準運費請參考網站費率表。",
    "default": "系統繁忙，請稍後再試。感謝您的耐心等待。",
}


def static_fallback_strategy(messages: list, tools: list = None) -> str:
    """
    靜態兜底策略：永不失敗，但只能返回預設文字。
    這是降級鏈的最後一道防線。
    """
    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = str(m.get("content", ""))

    # 根據用戶問題選擇最相關的靜態回覆
    if "運費" in user_content or "費用" in user_content:
        return STATIC_FALLBACK_RESPONSES["fee"]
    if "庫存" in user_content or "SKU" in user_content:
        return STATIC_FALLBACK_RESPONSES["inventory"]
    if "物流" in user_content or "追蹤" in user_content:
        return STATIC_FALLBACK_RESPONSES["shipping"]
    return STATIC_FALLBACK_RESPONSES["default"]


# 構建默認的降級策略鏈
def build_default_fallback_chain() -> FallbackChain:
    """
    構建生產環境的降級策略鏈（3 個層級）：
      Level 0：Pro 模型（30秒超時）
      Level 1：Flash 模型（10秒超時）
      Level 2：靜態兜底（永不失敗）
    """
    return FallbackChain(
        strategies=[
            make_llm_strategy(MODEL_PRO, timeout_sec=30.0),
            make_llm_strategy(MODEL_FLASH, timeout_sec=10.0),
            static_fallback_strategy,
        ]
    )


# ============================================================
# 快速驗證
# ============================================================

if __name__ == "__main__":
    print("=== 降級策略測試 ===\n")

    # 測試：靜態兜底策略（不需要 API）
    print("測試靜態兜底策略（不需要 API 調用）：")
    messages = [{"role": "user", "content": "查一下 SF123 的物流狀態"}]
    result = static_fallback_strategy(messages)
    print(f"  → {result}")

    # 測試：策略鏈（前兩個策略故意失敗，最後用靜態兜底）
    print("\n測試策略鏈（前兩個策略故意失敗）：")

    def always_fail_strategy(messages, tools=None):
        raise RuntimeError("模擬主模型宕機")

    def always_fail_strategy_2(messages, tools=None):
        raise RuntimeError("模擬備用模型也掛了")

    test_chain = FallbackChain(
        strategies=[
            always_fail_strategy,
            always_fail_strategy_2,
            static_fallback_strategy,
        ]
    )
    result, level = test_chain.execute(messages)
    print(f"  ✅ 最終使用第 {level+1} 級策略（靜態兜底）")
    print(f"  → {result}")

    print("\nDay 4 完成。優雅降級策略鏈就緒。")
    print("下一步：week3_day5/cost_tracker.py — Token 成本追蹤。")
