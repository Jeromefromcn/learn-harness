"""
week3_day3/observability.py
============================
第三週 Day 3:可觀測性--每次調用都留下痕跡

對應手冊任務:
  - 給每次 LLM 調用記錄:輸入 hash,模型版本,延遲,token 數,stop_reason
  - 把日誌寫成結構化 JSON(每行一條),方便 grep 和後期分析
  - 翻幾條日誌,確認"能重構出這次對話發生了什麼"

為什麼要結構化 JSON 而不是普通文字日誌?
  - 普通:`INFO 2024-01-01 call took 1.2s` -> 沒法自動化分析
  - 結構化:`{"latency_ms": 1200, "model": "...", "tokens": 450}` ->
    可以用 jq,grep,ClickHouse 等工具直接查詢和聚合

  第五週你用 LangSmith 時,你會發現框架自動幫你做了所有這些,
  到時候你就能親身感受"手寫 vs 框架"的差距.
"""

import json
import time
import hashlib
import logging
import sys
import os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


# ============================================================
# 日誌配置:每行一條 JSON,方便機器解析
# ============================================================

def setup_json_logger(log_file: str = "agent_calls.jsonl") -> logging.Logger:
    """
    設置結構化 JSON 日誌記錄器.
    日誌文件格式:JSONL(每行一個 JSON 對象)
    """
    logger = logging.getLogger("agent.observability")
    logger.setLevel(logging.INFO)

    # 避免重複添加 handler(多次 import 時的保護)
    if not logger.handlers:
        # 文件 handler:寫 JSONL 文件
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # 格式器:只輸出消息體(JSON 本身)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        # 控制台 handler:調試時也能看到
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("[LOG] %(message)s"))
        logger.addHandler(console_handler)

    return logger


logger = setup_json_logger()


# ============================================================
# === 第三週第三天新增:調用記錄數據結構 ===
# ============================================================

@dataclass
class LLMCallRecord:
    """
    一次 LLM 調用的完整記錄.
    所有字段都必須可序列化為 JSON.
    """
    # 時間信息
    timestamp: str          # ISO 8601 格式,帶時區
    latency_ms: int         # 從發起請求到收到完整響應的毫秒數

    # 請求標識
    input_hash: str         # 輸入消息的 hash(前12位),用於追蹤同一請求的多次嘗試

    # 模型信息
    model: str              # 實際使用的模型 ID
    stop_reason: str        # 為什麼停下來:end_turn / tool_use / max_tokens

    # Token 使用
    input_tokens: int       # 輸入 token 數(費用計算依據)
    output_tokens: int      # 輸出 token 數(費用計算依據)

    # 工具調用信息(可選)
    tools_called: list[str] = None  # 這輪調用了哪些工具

    # 可選的業務標籤(方便按業務維度篩選日誌)
    conversation_id: str = ""   # 哪次對話
    turn_number: int = 0        # 這是第幾輪


# ============================================================
# log_call:主要的可觀測性接口
# ============================================================

def log_call(
    messages: list[dict],
    response,               # anthropic.types.Message
    t0: float,
    conversation_id: str = "",
    turn_number: int = 0,
) -> LLMCallRecord:
    """
    記錄一次 LLM 調用的關鍵指標.

    設計為"調用完成後立即調用":
        t0 = time.time()
        resp = client.messages.create(...)
        record = log_call(messages, resp, t0)

    Args:
        messages:        發給模型的消息列表
        response:        模型的完整響應對象
        t0:              請求發起時的時間戳(time.time())
        conversation_id: 這次對話的唯一 ID(可選)
        turn_number:     這是第幾輪(可選)

    Returns:
        LLMCallRecord 對象(方便上層做進一步處理)
    """
    # 計算輸入的 hash:用於識別"同樣的輸入請求了幾次"
    # 只取前12位:夠用,不存完整 hash 節省空間
    input_str = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:12]

    # 提取工具調用信息
    tools_called = []
    for block in response.content:
        if hasattr(block, "type") and block.type == "tool_use":
            tools_called.append(block.name)

    record = LLMCallRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=int((time.time() - t0) * 1000),
        input_hash=input_hash,
        model=response.model,
        stop_reason=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        tools_called=tools_called or None,
        conversation_id=conversation_id,
        turn_number=turn_number,
    )

    # 序列化為 JSON 並寫入日誌
    # None 值會被排除,保持日誌簡潔
    record_dict = {k: v for k, v in asdict(record).items() if v is not None and v != ""}
    logger.info(json.dumps(record_dict, ensure_ascii=False))

    return record


# ============================================================
# 日誌分析工具
# ============================================================

def analyze_log(log_file: str = "agent_calls.jsonl") -> dict:
    """
    讀取 JSONL 日誌,計算基本統計信息.
    這是一個輕量版的"LangSmith 儀表盤".
    """
    records = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return {"error": f"日誌文件 {log_file} 不存在,先跑一些 agent 調用"}

    if not records:
        return {"error": "日誌文件為空"}

    latencies = [r["latency_ms"] for r in records if "latency_ms" in r]
    input_tokens = [r["input_tokens"] for r in records if "input_tokens" in r]
    output_tokens = [r["output_tokens"] for r in records if "output_tokens" in r]

    # 工具調用統計
    tool_counts: dict[str, int] = {}
    for r in records:
        for tool in r.get("tools_called", []) or []:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    return {
        "total_calls": len(records),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "stop_reasons": {
            reason: sum(1 for r in records if r.get("stop_reason") == reason)
            for reason in set(r.get("stop_reason") for r in records)
        },
        "tool_call_counts": tool_counts,
    }


# ============================================================
# 快速驗證(打印格式示例,不做真實 API 調用)
# ============================================================

if __name__ == "__main__":
    import anthropic

    print("=== 可觀測性模塊驗證 ===\n")

    # 模擬一條調用記錄
    print("示例日誌條目格式(這是每次 LLM 調用會寫入的 JSON):")
    sample = {
        "timestamp": "2026-06-05T10:30:00+00:00",
        "latency_ms": 1234,
        "input_hash": "a3f8c2d9e1b4",
        "model": "deepseek-v4-flash",
        "stop_reason": "tool_use",
        "input_tokens": 450,
        "output_tokens": 120,
        "tools_called": ["track_shipment"],
        "conversation_id": "conv-001",
        "turn_number": 1,
    }
    print(json.dumps(sample, ensure_ascii=False, indent=2))

    print("\n日誌分析示例(真實調用後運行):")
    stats = analyze_log()
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\nDay 3 完成.可觀測性層就緒.")
    print("提示:先跑幾次 agent,再回來分析日誌--那時候的感受會很不同.")
    print("下一步:week3_day4/fallback.py - 優雅降級.")
