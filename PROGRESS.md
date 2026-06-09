# Harness Engineering 五週學習進度追蹤

- [ ] = 未開始
- [x] = 已完成

---

## Week 1 — 建立心智模型

> 目標: 讀懂 harness 六大組件，用自己的話能解釋清楚

### Day 1 — 讀懂核心框架 (~90 min)
- [x] 閱讀 Birgitta Böckeler 的 <Harness engineering for coding agent users>
- [x] 寫下 Agent = Model + Harness 的定義
- [x] 記下 guides（前饋）與 sensors（反饋）的區別
- [x] 記住 sensors 分兩種：計算式 vs 推理式

### Day 2 — 讀懂「人在 loop 之上工作」 (~60 min)
- [x] 閱讀 <Humans and Agents in Software Engineering Loops>
- [x] 寫下三個調控維度：可維護性、架構契合、功能行為
- [x] 記下關鍵洞察：harness 無法完全取代人的判斷

### Day 3–4 — 親手感受 harness 的運作 (~2 hr)
- [x] 安裝 Claude Code，在現有專案啟動它
- [x] 給它真實小任務，觀察它讀檔、跑測試、自我修正
- [x] 記錄觀察到的 harness 行為
- [x] 查看 CLAUDE.md / AGENTS.md 配置檔

### Day 5 — 做一次自我盤點 (~60 min)
- [x] 建表列出六大組件並標記掌握程度
- [x] 圈出「全新」項目作為第四週重點
- [x] 產出：一頁筆記 + 六組件自我盤點表

**Week 1 完成標準:** 能解釋什麼是 harness、guides vs sensors、計算式 vs 推理式 sensors，並說出個人最需要補的兩塊。

---

## Week 2 — 第一個工具調用 Agent

> 目標: 從零做出能調用 2–3 個函數的物流 agent，加上第一道護欄

### Day 1 — 搭專案骨架 + 定義工具 (~90 min)
- [ ] 建虛擬環境，安裝 `anthropic` + `pydantic`
- [ ] 寫出三個 mock 工具函數（track_shipment / calc_shipping_fee / check_inventory）
- [ ] 給每個工具寫好 JSON schema 描述

### Day 2 — 接上 tool use (~90 min)
- [ ] 發出第一個帶 tools 的請求，確認 stop_reason = tool_use
- [ ] 從響應解析工具名稱與參數
- [ ] 手動執行工具函數確認結果

### Day 3 — 完成 agent loop (~2 hr)
- [ ] 實現 `dispatch_tools()` 遍歷執行工具
- [ ] 將工具結果喂回 messages 讓循環繼續
- [ ] 測試多步驟問題（查運費 + 庫存）
- [ ] 加上 max_turns 上限（最基本的護欄）

### Day 4 — 故意搞壞它 (~60 min)
- [ ] 傳壞參數（空字串、負重量、不存在 SKU）
- [ ] 讓工具主動拋異常，觀察循環是否崩潰
- [ ] 問工具答不了的問題，觀察是否幻覺
- [ ] 記錄失敗模式清單

### Day 5 — 第一道護欄：結構化輸出校驗 (~90 min)
- [ ] 用 Pydantic 定義輸出結構 + 業務校驗規則
- [ ] 在 prompt 要求模型只輸出 JSON
- [ ] 把 `validate_output()` 接到 agent 最終響應

**Week 2 完成標準:** 能多步調用工具 + 結構化校驗的物流 agent + 失敗模式清單。

---

## Week 3 — 加上可靠性

> 目標: 把 agent 變得「可信」—重試、熔斷、可觀測性、降級、成本

### Day 1 — 重試 + 指數退避 (~60 min)
- [ ] API 調用包重試層，對 RateLimitError 和校驗失敗做指數退避
- [ ] 區分可重試 vs 不可重試異常
- [ ] 校驗失敗時把報錯回饋給模型讓它自行修正

### Day 2 — 超時 + 熔斷器 (~60 min)
- [ ] 給每次調用設合理超時
- [ ] 實現簡單熔斷器：連續 N 次失敗後走降級路徑

### Day 3 — 可觀測性 (~90 min)
- [ ] 記錄每次 LLM 調用：輸入 hash、模型、延遲、token、stop_reason
- [ ] 日誌寫成結構化 JSON（每行一條）
- [ ] 跑幾個請求後翻日誌確認可追溯

### Day 4 — 優雅降級 (~60 min)
- [ ] 設計降級策略：主模型 → 小模型 → 快取/安全回應
- [ ] 接上降級路徑，確認系統不再崩

### Day 5 — 成本追蹤 (~60 min)
- [ ] 在日誌加 cost_usd 字段，按 DeepSeek 定價換算
- [ ] 寫腳本匯總平均成本、總成本、最貴的幾次
- [ ] 想出一個省成本點子記在筆記

**Week 3 完成標準:** agent 具備重試、熔斷、可觀測性、降級、成本追蹤 — 可往生產環境放。

---

## Week 4 — 學會評估

> 目標: 為「沒有標準答案」的系統設計測試

### Day 1 — 建立評估資料集 (~90 min)
- [ ] 寫 20–50 個測試案例（含預期工具調用與判斷標準）
- [ ] 摻入邊界與陷阱案例
- [ ] 存成 JSONL 或 CSV

### Day 2 — 確定性指標 (~90 min)
- [ ] 寫腳本跑完整個資料集
- [ ] 計算工具選擇精確率 / 召回率
- [ ] 列出所有失敗案例並人工檢查

### Day 3 — 推理式 sensor：LLM-as-judge (~2 hr)
- [ ] 寫 judge 給 agent 回答打分（1–5 分 + 理由）
- [ ] 同一批案例跑兩三次，觀察分數波動
- [ ] 挑出 judge 與直覺不一致的案例分析原因

### Day 4 — 自動化評估流水線 (~90 min)
- [ ] 寫 `run_eval.py` 一鍵跑完整份資料集並產出報告
- [ ] 做一次真實迭代：改 prompt，重跑評估，用數字證明好壞
- [ ] 報告輸出成易讀格式（Markdown / HTML）

### Day 5 — 寫成案例 (~90 min)
- [ ] 寫 README：背景、架構、設計取捨
- [ ] 寫「失敗模式與對策」章節
- [ ] 放上評估報告與成本數字
- [ ] 推到 GitHub 公開

**Week 4 完成標準:** 完整評估流水線 + 能說明質量的數字報告 + 公開 GitHub repo。

---

## Week 5 — 遷移到框架（對照組）

> 目標: 用手寫 + LangChain/LangGraph/LangSmith 兩種實作，能講清楚取捨

### Day 1 — 用 create_agent 重建 (~90 min)
- [ ] 安裝 `langchain` / `langgraph` / `langsmith` / `langchain-openai`（鎖 1.x）
- [ ] 用 `@tool` 裝飾器 + `create_agent()` 重建 agent
- [ ] 記錄行數對比：框架版 vs 手寫版
- [ ] 思考框架在背後幫你做了什麼

### Day 2 — LangGraph 狀態圖 (~2 hr)
- [ ] 重構成 StateGraph：model 節點、tools 節點、條件分支
- [ ] 掛上 checkpointer，確認狀態可保存
- [ ] 試 human-in-the-loop 中斷
- [ ] 對照手寫 max_turns 循環：圖模型多給了什麼？

### Day 3 — LangSmith 取代手寫可觀測性 (~60 min)
- [ ] 設環境變數啟用 LangSmith tracing
- [ ] 到 LangSmith 後台看自動產生的 trace
- [ ] 老實評估：框架 vs 手寫各自的優缺點

### Day 4 — 評估搬進框架 (~90 min)
- [ ] 上傳 Week 4 資料集到 LangSmith
- [ ] 跑 `evaluate()` 掛上確定性 + LLM-as-judge evaluator
- [ ] 對比手寫版與框架版分數

### Day 5 — 寫對照案例 (~90 min)
- [ ] 寫「框架幫我省了什麼」
- [ ] 寫「框架藏了什麼」
- [ ] 寫「我會怎麼選」
- [ ] 兩個版本放同一個 GitHub repo（/raw + /framework）

**Week 5 完成標準:** 雙版本 GitHub repo + 「手寫 vs 框架」技術對照文章。

---

## 總進度

| 週次 | 完成度 |
|------|--------|
| Week 1 — 心智模型 | 0% |
| Week 2 — 工具調用 Agent | 0% |
| Week 3 — 可靠性 | 0% |
| Week 4 — 評估 | 0% |
| Week 5 — 框架對照 | 0% |
| **總進度** | **0%** |
