"""
week3_day2/circuit_breaker.py
==============================
第三週 Day 2：斷路器（Circuit Breaker）

對應手冊任務：
  - 實現一個簡單的斷路器：連續 N 次失敗後，一段時間內直接走降級路徑
  - 超時（timeout）：給每次調用設一個時間上限

核心概念：
  斷路器模式（Circuit Breaker Pattern）= 電路熔斷器的軟件版。

  三個狀態：
    CLOSED（閉合/正常）：請求正常通過，失敗計數累積
    OPEN（斷開）：連續失敗超過閾值，直接拒絕請求，不等待
    HALF_OPEN（半開）：等待一段時間後，嘗試一次探測請求

  為什麼需要斷路器？
    重試機制解決「偶發性失敗」，
    斷路器解決「持續性故障」：
    如果 API 宕機了，繼續重試只會讓用戶等更久，
    不如直接快速失敗，告訴用戶「系統暫時不可用」。
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field


# ============================================================
# === 第三週第二天新增：斷路器 ===
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"       # 正常：請求通過
    OPEN = "open"           # 斷開：直接拒絕
    HALF_OPEN = "half_open" # 半開：試探性通過一個請求


class CircuitOpenError(Exception):
    """斷路器開路時拋出的異常，區別於一般的 API 錯誤。"""
    pass


@dataclass
class CircuitBreaker:
    """
    帶狀態的斷路器。

    使用方法：
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

        try:
            result = cb.call(some_api_function, arg1, arg2)
        except CircuitOpenError:
            # 走降級路徑
            result = fallback_response()
        except Exception as e:
            # 真正的 API 錯誤
            handle_error(e)

    線程安全：使用 threading.Lock() 保護狀態變量。
    """
    failure_threshold: int = 3      # 連續失敗幾次後斷開
    recovery_timeout: float = 30.0  # 斷開後等多少秒再嘗試恢復
    success_threshold: int = 1      # 半開狀態下成功幾次才恢復閉合

    # 內部狀態（不需要外部設置）
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """讀取當前狀態（自動處理 OPEN → HALF_OPEN 的時間轉換）。"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # 檢查是否已經超過恢復等待時間
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    # 時間到了，進入半開狀態，允許一個探測請求通過
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    print(f"[CircuitBreaker] OPEN → HALF_OPEN（等待了 {elapsed:.1f}秒）")
            return self._state

    def call(self, func, *args, **kwargs):
        """
        通過斷路器執行一個函數調用。

        CLOSED  → 直接調用，失敗時計數
        OPEN    → 直接拋 CircuitOpenError（不調用函數）
        HALF_OPEN → 調用一次：成功則恢復，失敗則重新斷開

        Args:
            func:   要調用的函數
            *args:  函數參數
            **kwargs: 函數關鍵字參數

        Raises:
            CircuitOpenError: 斷路器處於 OPEN 狀態
            Exception:        函數本身拋出的異常
        """
        current_state = self.state  # 讀取最新狀態（可能觸發 OPEN→HALF_OPEN）

        if current_state == CircuitState.OPEN:
            # 快速失敗：不等待，直接告訴調用方「現在不行」
            wait_remaining = self.recovery_timeout - (time.time() - self._last_failure_time)
            raise CircuitOpenError(
                f"斷路器開路，預計 {max(0, wait_remaining):.0f}秒後恢復。"
                f"（連續失敗 {self._failure_count} 次）"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success(current_state)
            return result

        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self, state_before_call: CircuitState):
        """記錄成功（HALF_OPEN 狀態下可能觸發恢復）。"""
        with self._lock:
            self._failure_count = 0  # 成功後重置失敗計數

            if state_before_call == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    # 半開狀態下連續成功，恢復正常
                    self._state = CircuitState.CLOSED
                    print(f"[CircuitBreaker] HALF_OPEN → CLOSED（系統已恢復）")

    def _on_failure(self, error: Exception):
        """記錄失敗（可能觸發斷開）。"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if (self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
                    and self._failure_count >= self.failure_threshold):
                # 超過閾值，斷開
                self._state = CircuitState.OPEN
                print(
                    f"[CircuitBreaker] → OPEN "
                    f"（連續失敗 {self._failure_count} 次，"
                    f"{self.recovery_timeout}秒後嘗試恢復）"
                )

    def reset(self):
        """手動重置斷路器（運維操作用）。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            print("[CircuitBreaker] 手動重置 → CLOSED")

    def __repr__(self):
        return (
            f"CircuitBreaker(state={self._state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )


# ============================================================
# 帶超時的 API 調用包裝
# ============================================================

import concurrent.futures


def call_with_timeout(func, timeout_sec: float, *args, **kwargs):
    """
    給函數調用設置超時時間。
    超時後拋 TimeoutError，而不是讓調用無限等待。

    實現方式：用線程池在後台執行，主線程等待指定時間。
    這種方式的限制：線程不會被強制終止，只是主線程放棄等待。

    Args:
        func:        要執行的函數
        timeout_sec: 超時時間（秒）
        *args, **kwargs: 傳給 func 的參數

    Raises:
        TimeoutError: 超時
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"調用超時（{timeout_sec}秒）")


# ============================================================
# 全局斷路器實例（單例模式，共用狀態）
# ============================================================

# 給 LLM API 調用使用的斷路器
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=3,     # 連續失敗 3 次就斷開
    recovery_timeout=60.0,   # 1 分鐘後嘗試恢復
)


# ============================================================
# 快速驗證（模擬斷路器狀態轉換）
# ============================================================

if __name__ == "__main__":
    print("=== 斷路器狀態機測試 ===\n")

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)
    print(f"初始狀態：{cb}")

    # 模擬連續失敗
    def always_fail():
        raise RuntimeError("模擬 API 宕機")

    def always_succeed():
        return "成功"

    # 失敗 3 次 → 斷開
    for i in range(3):
        try:
            cb.call(always_fail)
        except RuntimeError:
            print(f"第 {i+1} 次失敗，斷路器狀態：{cb.state.value}")

    # 嘗試調用 → 被斷路器拒絕
    print(f"\n斷路器狀態：{cb.state.value}，嘗試調用：")
    try:
        cb.call(always_succeed)
    except CircuitOpenError as e:
        print(f"✅ 被快速拒絕：{e}")

    # 等待恢復時間
    print(f"\n等待 2.1 秒（恢復時間 2.0 秒）...")
    time.sleep(2.1)

    # HALF_OPEN：允許一次探測
    print(f"\n現在狀態：{cb.state.value}，嘗試恢復性調用：")
    result = cb.call(always_succeed)
    print(f"✅ 恢復成功：{result}，斷路器狀態：{cb.state.value}")

    print("\nDay 2 完成。斷路器機制就緒。")
    print("下一步：week3_day3/observability.py — 結構化日誌。")
