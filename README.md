# Harness Engineering - 五週學習代碼集

Week 1–5 的練習代碼,使用 DeepSeek API(Anthropic 兼容接口).

## 快速開始

### 1. 建立虛擬環境(避免污染主環境)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows 用 .venv\Scripts\activate
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

額外依賴(Week 5 才需要):

```bash
pip install langchain langchain-openai langgraph langsmith
```

### 2. 配置 API Keys

```bash
cp .env.example .env
```

編輯 `.env`,填入真實的 key:

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
LANGSMITH_API_KEY=ls-xxxxxxxxxxxxxxxx   # 僅 Week 5 Day 3-4 需要
```

- DeepSeek API Key:在 [platform.deepseek.com](https://platform.deepseek.com) 獲取
- LangSmith API Key:在 [smith.langchain.com](https://smith.langchain.com) 免費注冊後獲取(Week 5 才需要)

> `.env` 已在 `.gitignore` 中,不會被提交到 git.

### 3. 運行代碼

直接在對應子目錄執行:

```bash
python week2_day2/first_call.py
python week3_day3/agent.py
```

每個文件開頭都有 `from config import setup_anthropic_env`,會自動讀取根目錄的 `.env`,無需額外配置.

## 目錄結構

```
.
├── config.py          # 全局配置,統一讀取 .env
├── .env.example       # Key 模板(複製為 .env 後填入真實值)
├── week1_day1_2/      # 基礎概念
├── week1_day3_4/      # 觀察與分析
├── week1_day5/        # Gap 分析
├── week2_day1-5/      # Tools,Agent,Guardrail
├── week3_day1-5/      # Retry,Circuit Breaker,Observability
├── week4_day1-5/      # Eval,Metrics,LLM Judge
└── week5_day1-5/      # LangChain,LangGraph,LangSmith
```

## 常見問題

**`DEEPSEEK_API_KEY` 未設置 / 報 401 錯誤**
確認 `.env` 文件存在於項目根目錄,且 key 填寫正確.

**Week 5 代碼報 import 錯誤**
Week 5 使用 LangChain,確認已安裝:
```bash
pip install langchain langchain-openai langgraph langsmith
```
