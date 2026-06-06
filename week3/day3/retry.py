"""
week3_day1/retry.py
====================
第三週 Day 1:重試 + 指數退避

對應手冊任務:
  - 給 API 調用加一層重試,對 RateLimitError 和校驗失敗做指數退避
  - 區分"可重試"(網絡抖動,頻率限制)和"不可重試"(參數錯誤)的異常
  - 校驗失敗時,可以把錯誤信息反饋給模型讓它自我修正

核心概念:
  指數退避(Exponential Backoff)= 第1次重試等1秒,第2次等2秒,第3次等4秒...
  這樣能避免所有客戶端同時重試,壓垮服務端.
  抖動(Jitter)= 在退避時間上加隨機抖動,進一步分散重試.
"""

import time
import random
import functools
import anthropic


# ============================================================
# === 第三週第一天新增:異常分類 ===
# ============================================================

class RetryableError(Exception):
    """可重試的錯誤:網絡抖動,頻率限制,臨時服務不可用."""
    pass


class NonRetryableError(Exception):
    """不可重試的錯誤:參數錯誤,認證失敗,業務邏輯錯誤."""
    pass


def classify_error(error: Exception) -> str:
    """
    判斷一個異常是否可重試.

    設計原則:
      - 5xx(服務端錯誤)-> 可重試(對方出問題,等一下可能好)
      - 429(頻率限制)-> 可重試(等久一點)
      - 4xx(客戶端錯誤)-> 不可重試(我方問題,重試也沒用)
      - 網絡超時 -> 可重試
      - 業務校驗失敗 -> 不可重試(需要修改邏輯)

    Returns:
        "retryable" 或 "non_retryable"
    """
    # Anthropic SDK 的異常層級
    if isinstance(error, anthropic.RateLimitError):
        return "retryable"      # 429:等一下再試

    if isinstance(error, anthropic.APIStatusError):
        if error.status_code >= 500:
            return "retryable"  # 5xx:服務端臨時錯誤
        return "non_retryable"  # 4xx:客戶端問題

    if isinstance(error, anthropic.APIConnectionError):
        return "retryable"      # 網絡問題

    if isinstance(error, anthropic.APITimeoutError):
        return "retryable"      # 超時

    # 未知錯誤:保守起見,不重試
    return "non_retryable"


# ============================================================
# with_retry:重試裝飾器
# ============================================================

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
):
    """
    帶指數退避的重試裝飾器.

    用法:
        @with_retry(max_attempts=3)
        def call_api():
            return client.messages.create(...)

    Args:
        max_attempts: 最多嘗試幾次(含第一次,不是額外重試次數)
        base_delay:   第一次退避的基礎等待時間(秒)
        max_delay:    退避時間的上限(秒)
        jitter:       是否加隨機抖動(推薦開啟,避免雷群效應)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_error = e
                    error_type = classify_error(e)

                    if error_type == "non_retryable":
                        # 不可重試的錯誤,直接拋出,不等待
                        print(f"[Retry] 不可重試的錯誤({type(e).__name__}),直接失敗")
                        raise

                    if attempt == max_attempts - 1:
                        # 最後一次嘗試也失敗了
                        print(f"[Retry] 已用完所有重試次數({max_attempts}),放棄")
                        raise

                    # 計算退避時間:2^attempt * base_delay
                    delay = min(base_delay * (2 ** attempt), max_delay)

                    # 加抖動:±25% 的隨機偏差
                    if jitter:
                        delay *= (0.75 + random.random() * 0.5)
                    delay = round(delay, 2)

                    print(
                        f"[Retry] 第 {attempt + 1}/{max_attempts} 次失敗 "
                        f"({type(e).__name__}),"
                        f"{delay}秒後重試..."
                    )
                    time.sleep(delay)

            # 理論上不會到這裡,但為了靜態分析工具:
            raise last_error

        return wrapper
    return decorator


# ============================================================
# retry_with_feedback:校驗失敗時把錯誤反饋給模型
# ============================================================

def retry_with_llm_feedback(
    run_agent_fn,
    user_msg: str,
    output_schema,
    max_retries: int = 2,
    verbose: bool = True,
):
    """
    特殊的重試策略:校驗失敗時,把錯誤信息加到提示詞裡,
    讓模型看到自己的錯誤並自我修正.

    這是一個 sensor -> feedback -> 重試 的完整循環示例:
      1. Agent 輸出 -> 校驗 -> 失敗
      2. 把失敗原因加入消息歷史
      3. 讓模型重新生成,這次它知道上次哪裡錯了

    這比盲目重試更聰明,因為模型能"看到"自己的錯誤.
    """
    from guardrail import ValidationError

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return run_agent_fn(user_msg, output_schema=output_schema, verbose=verbose)
        except ValidationError as e:
            last_error = e
            if attempt < max_retries:
                if verbose:
                    print(f"\n[LLM Feedback Retry] 第 {attempt + 1} 次校驗失敗,"
                          f"把錯誤信息反饋給模型...")
                # 把錯誤原因附加到用戶消息裡,讓下一輪模型看到
                user_msg = (
                    f"{user_msg}\n\n"
                    f"[注意:上次你的輸出不符合要求,"
                    f"錯誤原因是:{str(e)[:200]}."
                    f"請再試一次,確保輸出是合法的 JSON 格式.]"
                )

    raise last_error


# ============================================================
# 快速驗證(模擬重試行為)
# ============================================================

def _simulate_flaky_api(call_count: list) -> str:
    """模擬不穩定的 API:前兩次失敗,第三次成功."""
    call_count[0] += 1
    if call_count[0] < 3:
        raise anthropic.APIConnectionError(request=None)  # 模擬網絡失敗
    return "成功"


if __name__ == "__main__":
    print("=== Retry 機制測試 ===\n")

    # 測試 1:指數退避裝飾器
    count = [0]

    @with_retry(max_attempts=3, base_delay=0.1)  # 測試用短時間
    def flaky_call():
        return _simulate_flaky_api(count)

    try:
        result = flaky_call()
        print(f"✅ 最終成功(共嘗試 {count[0]} 次):{result}")
    except Exception as e:
        print(f"❌ 最終失敗:{e}")

    # 測試 2:不可重試的錯誤不等待直接失敗
    print("\n測試:不可重試的錯誤(參數錯誤)")
    try:
        error = anthropic.BadRequestError(
            message="invalid param",
            response=None,
            body=None,
        )
        raise error
    except Exception as e:
        category = classify_error(e)
        print(f"  BadRequestError -> 分類為:{category}")

    print("\nDay 1 完成.重試 + 指數退避機制就緒.")
    print("下一步:week3_day2/circuit_breaker.py - 斷路器.")
