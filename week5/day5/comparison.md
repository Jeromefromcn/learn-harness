# 裸 SDK vs LangChain/LangGraph：一個物流 Agent 的兩種實現

> **寫作說明：** 這是模板。用你的真實數字替換所有 `[X]` 佔位符。
> 完成後，這篇文章本身就是面試時「兩種都會，而且懂 trade-off」的最好證明。
> 目標讀者：有後端背景，正在評估是否引入 LangChain 的工程師。

---

## 背景

我花了五週用裸 Anthropic SDK（接 DeepSeek 後端）從零實現了一個物流查詢 agent，
然後用 LangChain/LangGraph 把它重做了一遍。

這篇文章記錄：框架省了什麼、帶來了什麼，以及在什麼情況下我會選哪種。

---

## 兩種實現的行數對比

| 組件 | 裸 SDK | LangChain | 差值 |
|------|--------|-----------|------|
| 工具定義（含 schema） | [X] 行 | [X] 行 | [X] |
| Agent loop | [X] 行 | [X] 行（AgentExecutor） | [X] |
| Tool dispatch | [X] 行 | 0（框架內置） | [-X] |
| 狀態管理 | [X] 行（messages 列表） | 0（StateGraph 自動） | [-X] |
| 可觀測性 | [X] 行（手寫 log_call）| 0（LangSmith 自動） | [-X] |
| 重試邏輯 | [X] 行（@with_retry） | [X] 行（需自己加） | [~=] |
| 斷路器 | [X] 行（CircuitBreaker） | [X] 行（需自己加） | [~=] |
| 評估流水線 | [X] 行（run_eval.py） | [X] 行（evaluate()） | [X] |

**總計：** 裸 SDK [X] 行，LangChain [X] 行，節省 [X]%。

---

## 框架真正給了什麼

### 1. 零配置可觀測性（最大驚喜）

裸 SDK 版：我花了一個下午寫 `observability.py`——
input_hash、latency_ms、token 計數、stop_reason，逐字段手動記錄。

LangChain 版：`export LANGSMITH_TRACING=true`，完事。
LangSmith 自動給我完整的 trace 樹、時間線、token 明細——
而且可以在 Web UI 裡直接重放和修改某次調用。

[你的觀察：LangSmith 的哪個功能讓你最意外？]

### 2. 狀態持久化（LangGraph Checkpointer）

裸 SDK 版：`messages` 列表在函數返回後消失。
要做多輪對話，我得自己維護一個 `conversation_id → messages` 的字典。

LangGraph 版：
```python
graph = builder.compile(checkpointer=MemorySaver())
graph.invoke(input, config={"configurable": {"thread_id": "order-42"}})
```
同一個 `thread_id` 的調用自動接續，還能從中斷點繼續。

[你的觀察：checkpointer 在什麼場景下特別有用？]

### 3. Human-in-the-Loop

裸 SDK 版：需要自己在循環裡加 `input()` 或消息隊列，侵入性強。

LangGraph 版：
```python
graph.compile(interrupt_before=["tools"])
```
在 `tools` 節點執行前自動暫停，等外部信號後繼續。架構上更乾淨。

---

## 框架沒給的（或給得不好）

### 1. 精細的重試策略

框架的重試是全有或全無。我的 `@with_retry` 裝飾器能區分：
- `RateLimitError` → 指數退避
- `ValidationError` → 不重試（重試也沒用）
- `APIConnectionError` → 重試3次，然後走降級

框架做不到這種粒度，Week 3 Day 1 的 `retry.py` 在框架版裡我還是原樣用著。

### 2. 數據主權

LangSmith 的追蹤數據上傳到 LangChain 的服務器。
對於涉及客戶物流、訂單信息的場景，這是合規紅線。

裸 SDK 版的 `agent_calls.jsonl` 只在你的服務器上。

[你的觀察：在你的業務場景裡，這是否是一個限制因素？]

### 3. 成本控制的能見度

LangSmith 有費用顯示，但不是 real-time 的，也沒有告警。
我在 Week 3 Day 5 手寫的 `SessionCostTracker` 可以在每輪後立即打印費用，
還可以接 Slack 告警。

---

## 評估體系的對比

Week 4 手寫版 (`run_eval.py`):
- 自己讀 JSONL → 自己跑 agent → 自己算指標 → 自己生成 Markdown 報告
- 麻煩，但完全在你的掌控裡

LangSmith evaluate() 版 (`langsmith_eval.py`):
- 數據集在 LangSmith 雲端
- `evaluate()` 自動並發跑、自動存結果、自動生成對比視圖
- 可以把「手寫 agent」和「框架 agent」在同一份數據集上的結果並排比較

**實際結果對比（填入你的數字）：**

| 版本 | 通過率 | 工具 F1 | Judge 均分 |
|------|--------|---------|------------|
| 裸 SDK agent | [X]% | [X.XX] | [X.X]/5 |
| LangChain agent | [X]% | [X.XX] | [X.X]/5 |
| 差異 | [±X%] | [±X.XX] | [±X.X] |

[你的解讀：差異來自哪裡？是框架改善了什麼，還是只是隨機波動？]

---

## 我的選擇框架

```
需求判斷樹：

是否涉及客戶敏感數據？
  → 是 → 裸 SDK（數據主權優先）

是否需要複雜的多 agent 協作或 human-in-loop？
  → 是 → LangGraph（State 管理太繁瑣了手寫）

是否快速原型 / 個人項目？
  → 是 → LangChain（省時間，不糾結）

是否對重試策略、成本控制有精細要求？
  → 是 → 裸 SDK 核心 + LangChain 上層（混合）
  
其他情況 → 看團隊技能棧和維護能力
```

---

## 一句話總結

**我不會選「哪個更好」，我會選「這個場景哪個更合適」。**

裸 SDK 版讓我理解 agent 的每一個環節——
因為我把它們全部手寫過，包括失敗過。
框架版讓我更快、讓我的代碼更簡潔——
但如果我沒有手寫過，我不會知道框架在幕後做了什麼，
也不會知道它在哪裡會讓我失去控制。

---

## 附錄：代碼倉庫結構

```
learn-harness/
├── week2_day1/    工具定義
├── week2_day3/    裸 SDK agent loop
├── week3_day*/    可靠性層（重試/斷路器/可觀測性/降級/成本）
├── week4_day*/    評估體系（數據集/指標/judge/流水線）
├── week5_day1/    LangChain create_agent
├── week5_day2/    LangGraph StateGraph + Checkpointer
├── week5_day3/    LangSmith 追蹤
└── week5_day4/    LangSmith evaluate()
```

*全部代碼開源：[GitHub URL]*
