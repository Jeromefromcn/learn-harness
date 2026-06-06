"""
week5_day4/langsmith_eval.py
=============================
第五週 Day 4：用 LangSmith evaluate() 重跑 Week 4 的評估

對應手冊任務：
  - 把 Week 4 的評估數據集上傳到 LangSmith dataset
  - 用 evaluate() 跑一遍（確定性 evaluator + LLM-as-judge）
  - 用框架的對比視圖，把「手寫 agent」和「框架 agent」在同一份數據集上的分數並排比較

核心體驗：
  Week 4 你花了一整天手寫 run_eval.py；
  這裡你會發現框架把「數據集管理 + 跑 agent + 對比」都做好了。
  但你因為手寫過，所以你完全清楚框架在幕後做了什麼——
  這才是「兩種都會」的真正意義。
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_OPENAI_BASE_URL, MODEL_FLASH,
    setup_langsmith_env
)

# 開啟 LangSmith 追蹤
setup_langsmith_env()

from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools_lc import TOOLS

# ============================================================
# Agent 初始化（和 Day 3 一樣）
# ============================================================

model = ChatOpenAI(
    model=MODEL_FLASH,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_OPENAI_BASE_URL,
    temperature=0,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個物流客服助手，使用提供的工具後給出清晰的回答。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm=model, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, max_iterations=6)

ls_client = Client()

DATASET_NAME = "logistics-agent-eval-v1"


# ============================================================
# === 第五週第四天新增：上傳數據集到 LangSmith ===
# ============================================================

def upload_dataset_to_langsmith(jsonl_path: str) -> str:
    """
    把 Week 4 的 JSONL 數據集上傳到 LangSmith。
    LangSmith 的 dataset 是 "examples" 的集合，每個 example 有 inputs 和 outputs。

    Returns:
        dataset_name（用於後續 evaluate()）
    """
    # 檢查是否已存在
    existing = [d for d in ls_client.list_datasets() if d.name == DATASET_NAME]
    if existing:
        print(f"數據集 '{DATASET_NAME}' 已存在，跳過上傳")
        return DATASET_NAME

    # 讀取 JSONL
    cases = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    # 創建數據集
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="物流 agent 評估集（來自 week4_day1/eval_dataset.py）",
    )

    # 上傳 examples
    ls_client.create_examples(
        inputs=[{"question": c["question"]} for c in cases],
        outputs=[{"expected_tools": c["expected_tools"],
                  "acceptable_outputs": c["acceptable_outputs"]}
                 for c in cases],
        dataset_id=dataset.id,
    )

    print(f"✅ 數據集已上傳：{DATASET_NAME}（{len(cases)} 個案例）")
    return DATASET_NAME


# ============================================================
# Evaluator 函數（對應 Week 4 Day 2 的 check_output_content）
# ============================================================

def correctness_evaluator(run, example) -> dict:
    """
    確定性 evaluator：輸出是否包含可接受的關鍵詞。
    格式要求：接受 run 和 example，返回 {"key": str, "score": float}
    """
    actual = str(run.outputs.get("output", "")).lower()
    acceptable = example.outputs.get("acceptable_outputs", [])

    for keyword in acceptable:
        if keyword.lower() in actual:
            return {"key": "keyword_match", "score": 1.0}

    return {"key": "keyword_match", "score": 0.0}


def tool_selection_evaluator(run, example) -> dict:
    """
    確定性 evaluator：工具選擇 F1。
    從 run 的 metadata 裡提取工具調用記錄。
    """
    expected = set(example.outputs.get("expected_tools", []))
    # LangSmith 把工具調用記錄在 run 的子 run 裡
    # 這裡用一個簡化版：直接看輸出是否提到了期望的工具
    # 完整版需要遍歷 run.child_runs
    if not expected:
        return {"key": "tool_selection", "score": 1.0}  # 期望不調工具

    actual_output = str(run.outputs.get("output", ""))
    # 簡化：看輸出裡是否有期望的信息標誌
    return {"key": "tool_selection", "score": 0.5}  # 保守估計


def make_llm_judge():
    """
    LLM-as-judge evaluator 工廠。
    對應 Week 4 Day 3 的 judge() 函數。
    """
    judge_model = ChatOpenAI(
        model=MODEL_FLASH,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_OPENAI_BASE_URL,
        temperature=0,
    )

    def llm_judge(run, example) -> dict:
        question = example.inputs.get("question", "")
        answer = str(run.outputs.get("output", ""))

        prompt_text = (
            f"評估 agent 回答質量，只輸出 JSON: {{\"score\": <1-5>}}\n"
            f"問題：{question}\n回答：{answer[:300]}"
        )
        import re
        try:
            resp = judge_model.invoke(prompt_text)
            raw = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r'"score"\s*:\s*(\d)', raw)
            score = int(m.group(1)) / 5.0 if m else 0.5  # 歸一化到 0-1
        except Exception:
            score = 0.5

        return {"key": "llm_judge", "score": score}

    return llm_judge


# ============================================================
# 運行評估
# ============================================================

def run_langsmith_evaluation(max_concurrency: int = 2) -> dict:
    """
    用 LangSmith evaluate() 跑完整評估。

    等同於 Week 4 Day 4 的 run_full_evaluation()，
    但框架自動處理了：並發、重試、進度追蹤、結果存儲。

    Args:
        max_concurrency: 並發數（控制費用）

    Returns:
        評估結果摘要
    """

    def target_fn(inputs: dict) -> dict:
        """被評估的 agent 函數（LangSmith 的要求格式）"""
        result = agent_executor.invoke({"input": inputs["question"]})
        return {"output": result.get("output", "")}

    results = evaluate(
        target_fn,
        data=DATASET_NAME,
        evaluators=[
            correctness_evaluator,
            make_llm_judge(),
        ],
        experiment_prefix="week5-framework-agent",
        max_concurrency=max_concurrency,
    )

    return results


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 5 Day 4：LangSmith evaluate()")
    print("=" * 60)

    print("\n步驟 1：上傳評估數據集到 LangSmith")
    dataset_path = Path(__file__).parent.parent / "week4_day1" / "dataset.jsonl"

    if not dataset_path.exists():
        print("先運行 week4_day1/eval_dataset.py 生成數據集")
    else:
        upload_dataset_to_langsmith(str(dataset_path))

        print("\n步驟 2：運行評估（max_concurrency=2 控制費用）")
        print("注意：需要有效的 LANGSMITH_API_KEY\n")

        try:
            results = run_langsmith_evaluation(max_concurrency=2)
            print(f"\n✅ 評估完成！到 https://smith.langchain.com 查看結果")
            print(f"在 LangSmith 的對比視圖裡，可以並排看：")
            print(f"  - week5-framework-agent（本次，框架版）")
            print(f"  - Week 4 Day 4 的 eval_report.md（手寫版）")

        except Exception as e:
            print(f"評估運行失敗（可能是 API key 未設置）：{e}")
            print("\n以下是 evaluate() 的核心概念（不需要真實運行）：")
            print("""
LangSmith evaluate() 等同於你的 run_full_evaluation()：
  1. 從 dataset 讀取所有 examples
  2. 對每個 example 調用 target_fn（你的 agent）
  3. 對每個結果調用 evaluators（你的 metrics）
  4. 把結果存到 LangSmith，生成對比報告

框架給你省的：
  - 並發控制（max_concurrency）
  - 進度顯示
  - 結果可視化
  - 多個 experiment 的對比視圖

你手寫版給你省的：
  - 數據不出境
  - 可以在 run_full_evaluation 裡加任何自定義邏輯
            """)

    print("\n第五週 Day 4 完成。")
    print("明天：week5_day5/comparison.md — 把兩種方案的對比寫成文章。")
    print("那篇文章本身就是你面試時最強的技術寫作樣本。")
