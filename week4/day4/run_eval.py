"""
week4_day4/run_eval.py
=======================
第四週 Day 4：完整評估流水線

對應手冊任務：
  - 串聯前三天的組件：加載數據集 → 跑 agent → 算指標 + judge → 輸出報告
  - 做一次真實迭代：改一個 prompt 或護欄，重跑，用數字判斷是否變好
  - 報告輸出可讀的格式（Markdown 或 HTML 表格）

這是 harness engineering 裡「持續評估」的具體形態：
  每次改動都有數字依據，而不是靠「感覺變好了」。
  這也是你和「只會寫 prompt 不做評估」的人之間的核心差距。
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DEFAULT_MODEL, setup_anthropic_env

setup_anthropic_env()
import anthropic

client = anthropic.Anthropic()


# ============================================================
# 精簡版 agent（評估用）
# ============================================================

def _run_agent_simple(question: str, max_turns: int = 6) -> tuple[str, list[str]]:
    """評估用的精簡 agent，返回 (最終文字, 調用工具列表)"""
    from tools import TOOLS, dispatch_tools
    messages = [{"role": "user", "content": question}]
    tools_called = []

    for _ in range(max_turns):
        resp = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=512, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            text = next((b.text for b in resp.content if hasattr(b, "text")), "")
            return text, tools_called

        if resp.stop_reason == "tool_use":
            for b in resp.content:
                if b.type == "tool_use":
                    tools_called.append(b.name)
            messages.append({"role": "user", "content": dispatch_tools(resp.content)})
            continue
        break
    return "", tools_called


# ============================================================
# 評估工具函數（從 metrics.py 和 judge.py 複製，保持自包含）
# ============================================================

def _check_tool_selection(actual: list[str], expected: list[str]) -> dict:
    actual_set, expected_set = set(actual), set(expected)
    if not expected_set and not actual_set:
        return {"f1": 1.0, "correct": True}
    if not expected_set:
        return {"f1": 0.0, "correct": False}
    if not actual_set:
        return {"f1": 0.0, "correct": False}
    tp = len(actual_set & expected_set)
    p = tp / len(actual_set)
    r = tp / len(expected_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"f1": round(f1, 3), "correct": f1 >= 0.8,
            "actual": list(actual_set), "expected": list(expected_set)}


def _check_output(answer: str, acceptable: list[str]) -> dict:
    answer_lower = answer.lower()
    for kw in acceptable:
        if kw.lower() in answer_lower:
            return {"passed": True, "matched": kw}
    return {"passed": False}


def _judge_answer(question: str, answer: str) -> int:
    """快速 judge：只返回分數（1-5）"""
    import re
    prompt = (
        f"評估 agent 回答質量，只輸出 JSON: {{\"score\": <1-5>}}\n"
        f"問題：{question}\n回答：{answer[:200]}"
    )
    try:
        resp = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=80,
            messages=[{"role": "user", "content": prompt}])
        raw = resp.content[0].text
        m = re.search(r'"score"\s*:\s*(\d)', raw)
        return int(m.group(1)) if m else 3
    except Exception:
        return 3


# ============================================================
# === 第四週第四天新增：完整流水線 ===
# ============================================================

def run_full_evaluation(
    dataset: list[dict],
    run_judge: bool = True,
    max_cases: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    完整評估流水線：加載 → 跑 agent → 算指標 → judge → 返回結果。

    Args:
        dataset:   評估案例列表
        run_judge: 是否運行 LLM-as-judge（耗費額外 API 調用）
        max_cases: 最多評估幾個案例（None = 全部）
        verbose:   是否打印進度

    Returns:
        帶完整評估結果的案例列表
    """
    cases = dataset[:max_cases] if max_cases else dataset
    results = []

    print(f"\n開始評估（{len(cases)} 個案例，judge={'開啟' if run_judge else '關閉'}）")
    print("-" * 60)

    for i, case in enumerate(cases, 1):
        if verbose:
            print(f"[{i:02d}/{len(cases)}] {case['id']}: {case['question'][:35]}...")

        t0 = time.time()
        try:
            answer, tools_called = _run_agent_simple(case["question"])
        except Exception as e:
            results.append({**case, "error": str(e), "passed": False})
            print(f"  ❌ 錯誤：{str(e)[:50]}")
            continue

        elapsed = round(time.time() - t0, 2)
        tool_m = _check_tool_selection(tools_called, case["expected_tools"])
        output_m = _check_output(answer, case["acceptable_outputs"])
        passed = tool_m["correct"] and output_m["passed"]

        result = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": answer[:300],
            "tools_called": tools_called,
            "tool_f1": tool_m["f1"],
            "tool_correct": tool_m["correct"],
            "output_passed": output_m["passed"],
            "output_matched": output_m.get("matched"),
            "passed": passed,
            "latency_sec": elapsed,
        }

        if run_judge:
            result["judge_score"] = _judge_answer(case["question"], answer)

        status = "✅" if passed else "❌"
        judge_str = f", judge={result.get('judge_score', '-')}/5" if run_judge else ""
        if verbose:
            print(f"  {status} tool_f1={tool_m['f1']:.2f}, "
                  f"output={'✅' if output_m['passed'] else '❌'}"
                  f"{judge_str}, {elapsed}s")

        results.append(result)

    return results


def generate_markdown_report(
    results: list[dict],
    run_name: str = "eval",
) -> str:
    """
    把評估結果生成 Markdown 格式的報告。
    這份報告可以貼到 GitHub PR 或 README 裡作為基準線。
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    errors = sum(1 for r in results if "error" in r)

    judge_scores = [r["judge_score"] for r in results
                    if "judge_score" in r and r["judge_score"]]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None

    avg_tool_f1 = sum(r.get("tool_f1", 0) for r in results) / total if total else 0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 評估報告：{run_name}",
        f"**生成時間：** {ts}  **模型：** {DEFAULT_MODEL}",
        "",
        "## 總覽",
        f"| 指標 | 數值 |",
        f"|------|------|",
        f"| 通過率 | {passed}/{total} ({passed/total*100:.1f}%) |",
        f"| 工具選擇 F1 | {avg_tool_f1:.3f} |",
    ]
    if avg_judge is not None:
        lines.append(f"| Judge 平均分 | {avg_judge:.1f}/5 |")
    if errors:
        lines.append(f"| 執行錯誤 | {errors} |")

    # 按類別統計
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r.get("passed", False))

    lines += ["", "## 按類別統計", "| 類別 | 通過 | 總計 | 通過率 |",
              "|------|------|------|--------|"]
    for cat, passes in sorted(by_cat.items()):
        p = sum(passes)
        lines.append(f"| {cat} | {p} | {len(passes)} | {p/len(passes)*100:.0f}% |")

    # 失敗案例
    failures = [r for r in results if not r.get("passed", False) and "error" not in r]
    if failures:
        lines += ["", "## 失敗案例", "| ID | 問題 | 工具F1 | 輸出 |",
                  "|----|------|--------|------|"]
        for f in failures:
            lines.append(
                f"| {f['id']} | {f['question'][:30]} | "
                f"{f.get('tool_f1', 0):.2f} | "
                f"{'✅' if f.get('output_passed') else '❌'} |"
            )

    return "\n".join(lines)


# ============================================================
# 主程序：完整的「改 → 測 → 比較」迭代循環
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Day 4：完整評估流水線")
    print("=" * 60)

    # 加載數據集
    dataset_path = Path(__file__).parent.parent / "week4_day1" / "dataset.jsonl"
    if not dataset_path.exists():
        print("先運行 week4_day1/eval_dataset.py 生成數據集")
        sys.exit(1)

    from eval_dataset import load_dataset
    dataset = load_dataset(str(dataset_path))

    # 運行評估（取前 15 個案例控制費用）
    results = run_full_evaluation(dataset, run_judge=True, max_cases=15, verbose=True)

    # 生成報告
    report = generate_markdown_report(results, run_name="Week4-Day4-Baseline")

    # 打印報告
    print("\n" + report)

    # 保存報告和結果
    report_path = "eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n報告已保存到 {report_path}")

    results_path = "eval_results_full.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n下一步（真實迭代）：")
    print("  1. 分析失敗案例，找一個共同原因")
    print("  2. 修改 tools.py 的 description 或 guardrail.py 的提示詞")
    print("  3. 重新跑這個腳本")
    print("  4. 比較兩次報告的數字——這就是 eval-driven 開發")
    print("\nDay 4 完成。下一步：week4_day5 — 寫成案例，讓別人理解你做了什麼。")
