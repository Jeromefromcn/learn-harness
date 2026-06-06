# Harness Engineering 代碼集合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 按照五週 Harness Engineering 手冊,生成一組漸進式,自包含的 Python 代碼文件夾,每個文件夾對應一個學習單元,可獨立運行.

**Architecture:** 根目錄 `config.py` 集中管理 DeepSeek API 配置;每個 weekX_dayY 文件夾自包含(複製繼承文件 + 新增當日文件);新增內容用 `# === 第X週第Y天新增 ===` 標記;所有文件有詳細中文注釋.

**Tech Stack:** Python 3.10+, anthropic SDK (指向 DeepSeek), pydantic, langchain>=1,<2, langgraph>=1,<2, langsmith, langchain-openai

---

## 文件結構

```
learn-harness/
├── config.py                     # 全局 API 配置
├── week1_day1_2/concepts.py      # 核心概念代碼化
├── week1_day3_4/observation.py   # Claude Code 觀察模板
├── week1_day5/gap_analysis.py    # 缺口分析模板
├── week2_day1/tools.py           # 3 個 mock 工具 + JSON schema
├── week2_day2/first_call.py      # 第一次帶 tools 的 API 調用
├── week2_day3/agent.py           # 完整 agent 循環
├── week2_day4/failure_modes.py   # 失敗模式測試
├── week2_day5/guardrail.py       # Pydantic 輸出校驗
├── week3_day1/retry.py           # 重試 + 指數退避
├── week3_day2/circuit_breaker.py # 斷路器
├── week3_day3/observability.py   # 結構化日誌
├── week3_day4/fallback.py        # 優雅降級
├── week3_day5/cost_tracker.py    # Token 成本追蹤
├── week4_day1/eval_dataset.py    # 評估數據集
├── week4_day2/metrics.py         # 精確率/召回率
├── week4_day3/judge.py           # LLM-as-judge
├── week4_day4/run_eval.py        # 完整評估流水線
├── week4_day5/case_study.md      # 案例寫作模板
├── week5_day1/lc_agent.py        # LangChain create_agent
├── week5_day2/lg_agent.py        # LangGraph StateGraph
├── week5_day3/langsmith_setup.py # LangSmith 追蹤
├── week5_day4/langsmith_eval.py  # LangSmith evaluate()
└── week5_day5/comparison.md      # 兩種方案對比模板
```

## 執行狀態

- [x] 創建目錄結構
- [x] config.py
- [x] Week 1
- [x] Week 2
- [x] Week 3
- [x] Week 4
- [x] Week 5
