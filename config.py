"""
全局配置文件 - Harness Engineering 五週學習代碼集
所有 weekX_dayY 文件夾都從這裡讀取 API 配置.

使用方法:在每個子文件夾的代碼開頭加上以下兩行:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import setup_anthropic_env, DEFAULT_MODEL
    setup_anthropic_env()

API Keys 設置:
    cp .env.example .env
    # 然後編輯 .env 填入真實 key(.env 已在 .gitignore,不會被提交)
"""

import os
from dotenv import load_dotenv

# 從項目根目錄的 .env 文件加載環境變量
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ============================================================
# DeepSeek API 配置
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# DeepSeek 提供兩種 API 接口:
# 1. Anthropic 兼容接口:供 Week 1-4 的 anthropic SDK 使用
DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
# 2. OpenAI 兼容接口:供 Week 5 的 LangChain (langchain-openai) 使用
DEEPSEEK_OPENAI_BASE_URL = "https://api.deepseek.com"

# ============================================================
# 模型選擇
# ============================================================
# Flash 版:速度快,費用低,適合開發調試
MODEL_FLASH = "deepseek-v4-flash"
# Pro 版:能力強,費用高,適合生產複雜推理
MODEL_PRO = "deepseek-v4-pro"

# 學習階段預設使用 Flash,節省費用
# 若要測試更強推理能力,把這裡改成 MODEL_PRO
DEFAULT_MODEL = MODEL_FLASH

# ============================================================
# LangSmith 配置(Week 5 Day 3-4 使用,其他週可忽略)
# ============================================================
# 在 https://smith.langchain.com 免費注冊後獲取 API key
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = "harness-engineering-learn"


# ============================================================
# 初始化函數
# ============================================================

def setup_anthropic_env():
    """
    設置 Anthropic SDK 使用 DeepSeek 後端所需的環境變量.
    必須在 import anthropic 並調用 anthropic.Anthropic() 之前調用.

    原理:Anthropic Python SDK 自動讀取兩個環境變量:
      - ANTHROPIC_API_KEY:身份驗證 key
      - ANTHROPIC_BASE_URL:API 服務器地址
    我們把這兩個變量指向 DeepSeek,代碼邏輯完全不變,只換了後端.
    """
    os.environ["ANTHROPIC_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["ANTHROPIC_BASE_URL"] = DEEPSEEK_ANTHROPIC_BASE_URL


def setup_langsmith_env():
    """
    開啟 LangSmith 自動追蹤.設置後,所有 LangChain/LangGraph 調用
    都會自動上報 trace 到 LangSmith,無需修改 agent 代碼.
    (Week 5 Day 3-4 使用)
    """
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
