"""
week4_day3/judge.py
====================
第四週 Day 3：推斷式 Sensor——LLM-as-judge

對應手冊任務：
  - 寫一個 judge，給 agent 的回答質量打 1-5 分
  - 對同一個案例跑兩次，觀察 judge 評分的波動（感受概率性）
  - 找幾個「judge 和你直覺不一致」的案例，想想是 judge 不準，
    還是你的 prompt 不夠清晰

重要提示（手冊原文）：
  LLM-as-judge 不是神諭。它本身也會出錯。
  它的價值在於把「無法窮舉的質量」變成「大致可追蹤的數字」，
  讓你能比較不同版本的好壞，而不是給出絕對的裁決。
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEFAULT_MODEL, setup_anthropic_env

setup_anthropic_env()
import anthropic

client = anthropic.Anthropic()


# ============================================================
# === 第四週第三天新增：Judge Prompt ===
# ============================================================

JUDGE_PROMPT = """你是一位嚴格的質量評審。請根據以下標準，給 agent 的回答打分（1-5分）。

評分標準：
  5分：回答完全準確、有幫助、格式清晰，無任何問題
  4分：回答基本正確，有小瑕疵（措辭、冗長等）
  3分：部分正確，但有明顯遺漏或輕微錯誤
  2分：回答有誤或嚴重遺漏，但能理解用戶意圖
  1分：完全錯誤、幻覺、或拒絕回答

評審重點：
  ✓ 準確性：回答是否與工具返回的數據一致？
  ✓ 完整性：是否回答了用戶的所有問題？
  ✓ 誠實性：模型是否承認「沒有相關信息」而非捏造？
  ✓ 有用性：用戶是否能根據回答做出決策？

請只輸出 JSON，格式如下：
{{"score": <1-5的整數>, "reason": "<一句話說明評分原因>"}}

用戶問題：{question}
Agent 回答：{answer}
（背景：工具返回的原始數據——{tool_context}）"""


# ============================================================
# judge 函數
# ============================================================

def judge(
    question: str,
    answer: str,
    tool_context: str = "",
    judge_model: str = None,
) -> dict:
    """
    用 LLM 評估一個 agent 回答的質量。

    Args:
        question:     用戶的原始問題
        answer:       agent 的回答
        tool_context: 工具返回的原始數據（可選，提升 judge 準確性）
        judge_model:  judge 使用的模型（默認和 agent 相同；生產中可以用不同模型）

    Returns:
        {"score": int, "reason": str, "raw": str}
    """
    model = judge_model or DEFAULT_MODEL
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        tool_context=tool_context or "（未提供工具上下文）",
    )

    resp = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()

    # 解析 JSON 輸出
    try:
        # 嘗試提取 JSON（judge 可能在 JSON 前後加說明）
        import re
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = json.loads(raw)

        # 確保 score 是整數，在有效範圍內
        result["score"] = max(1, min(5, int(result.get("score", 3))))
        result["raw"] = raw
        return result

    except (json.JSONDecodeError, ValueError):
        # judge 本身輸出格式不對：保守給 3 分
        return {"score": 3, "reason": "judge 輸出格式無效", "raw": raw}


def judge_twice(question: str, answer: str, tool_context: str = "") -> dict:
    """
    對同一個案例跑兩次 judge，計算評分穩定性。

    這是手冊裡「感受概率性」的具體操作：
    如果兩次評分差距 >= 2，說明這個案例的評估標準不夠清晰，
    需要修改 judge prompt 或者細化評分標準。

    Returns:
        {"score1", "score2", "delta", "stable", "avg_score"}
    """
    r1 = judge(question, answer, tool_context)
    r2 = judge(question, answer, tool_context)
    delta = abs(r1["score"] - r2["score"])

    return {
        "score1": r1["score"],
        "score2": r2["score"],
        "reason1": r1.get("reason", ""),
        "reason2": r2.get("reason", ""),
        "delta": delta,
        "stable": delta <= 1,   # 差距 <= 1 算穩定
        "avg_score": round((r1["score"] + r2["score"]) / 2, 1),
    }


# ============================================================
# 批量 judge
# ============================================================

def judge_batch(eval_results: list[dict]) -> list[dict]:
    """
    對一批評估結果批量運行 judge，加入語義質量分數。

    Args:
        eval_results: metrics.py 的 run_evaluation() 輸出

    Returns:
        加入了 judge_score 字段的結果列表
    """
    judged = []
    for i, result in enumerate(eval_results, 1):
        if "error" in result or not result.get("actual_output"):
            result["judge_score"] = None
            judged.append(result)
            continue

        print(f"[Judge {i}/{len(eval_results)}] {result['id']}...")

        j = judge(
            question=result["question"],
            answer=result.get("actual_output", ""),
            tool_context=str(result.get("tools_called", "")),
        )
        result["judge_score"] = j["score"]
        result["judge_reason"] = j.get("reason", "")
        print(f"  → {j['score']}/5 分：{j.get('reason', '')[:60]}")

        judged.append(result)
        time.sleep(0.5)  # 避免頻率限制

    return judged


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Day 3：LLM-as-judge 測試")
    print("=" * 60)

    # 測試 1：單次評估
    print("\n【測試 1】單次評估")
    result = judge(
        question="SF123 在哪裡？",
        answer="SF123 目前在上海轉運中心，預計2天後送達。",
        tool_context='{"status": "in_transit", "eta_days": 2, "location": "上海轉運中心"}',
    )
    print(f"評分：{result['score']}/5")
    print(f"理由：{result['reason']}")

    # 測試 2：幻覺答案（應該得低分）
    print("\n【測試 2】幻覺答案（期望低分）")
    hallucination_result = judge(
        question="SF999 在哪裡？",
        answer="SF999 目前在廣州倉庫，預計明天送達。",  # 幻覺：SF999 不存在
        tool_context='{"status": "not_found", "message": "找不到追蹤號 SF999"}',
    )
    print(f"評分：{hallucination_result['score']}/5（應該 <= 2）")
    print(f"理由：{hallucination_result['reason']}")

    # 測試 3：兩次評估，觀察波動
    print("\n【測試 3】兩次評估，觀察穩定性")
    stability = judge_twice(
        question="5kg 包裹寄到 B 區多少錢？",
        answer="寄 5kg 包裹到 B 區的運費是 150 港幣。計算方式：基礎費 30 + 5kg × 24 = 150 HKD。",
        tool_context='{"fee_hkd": 150.0, "zone": "B"}',
    )
    print(f"第一次：{stability['score1']}/5")
    print(f"第二次：{stability['score2']}/5")
    print(f"差距：{stability['delta']}，穩定性：{'✅ 穩定' if stability['stable'] else '⚠️ 不穩定'}")
    print(f"平均分：{stability['avg_score']}")

    if not stability["stable"]:
        print("\n⚠️ 兩次評分差距大，說明 judge prompt 不夠清晰。")
        print("思考：是哪個評分標準有歧義？嘗試修改 JUDGE_PROMPT。")

    print("\nDay 3 完成。")
    print("核心洞察：LLM-as-judge 帶概率性，但它讓「語義質量」變得可追蹤。")
    print("下一步：week4_day4/run_eval.py — 串成完整評估流水線。")
