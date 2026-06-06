"""
week4_day2/metrics.py
======================
第四週 Day 2：確定性指標——精確率 / 召回率

對應手冊任務：
  - 跑整個數據集，記錄每個案例的實際輸出
  - 計算「工具選擇是否正確」、「輸出格式是否通過」的精確率/召回率
  - 把失敗案例單獨列出來，人工看一遍找出原因

為什麼先做確定性指標，不直接上 LLM-as-judge？
  計算式 sensor（確定性）比推斷式 sensor（LLM-as-judge）：
    - 便宜：不用再調一次 LLM
    - 快：本地計算，毫秒級
    - 確定：同樣輸入永遠同樣結果，便於 CI 集成

  先把確定性能測的測完，再用 LLM 評估剩下的語義質量部分。
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import setup_anthropic_env

setup_anthropic_env()

# 複用 Week 3 Day 5 的 agent（帶完整可靠性層）
# 為自包含，把最精簡的 agent 實現內聯進來
import anthropic
from config import DEFAULT_MODEL

client = anthropic.Anthropic()


# ============================================================
# 最精簡的 agent（評估用，不需要完整可靠性層）
# ============================================================

def _make_tools():
    """返回工具列表（從 tools.py 複製以保持自包含）"""
    from tools import TOOLS
    return TOOLS


def _make_dispatch():
    from tools import dispatch_tools
    return dispatch_tools


def run_agent_for_eval(question: str, max_turns: int = 6) -> tuple[str, list[str]]:
    """
    評估專用的 agent 運行器。

    Returns:
        (final_text, tools_called): 最終文字輸出 + 調用過的工具列表
    """
    tools = _make_tools()
    dispatch_tools = _make_dispatch()
    messages = [{"role": "user", "content": question}]
    tools_called = []

    for _ in range(max_turns):
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=512,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            final_text = next((b.text for b in resp.content if hasattr(b, "text")), "")
            return final_text, tools_called

        if resp.stop_reason == "tool_use":
            for block in resp.content:
                if block.type == "tool_use":
                    tools_called.append(block.name)
            tool_results = dispatch_tools(resp.content)
            messages.append({"role": "user", "content": tool_results})
            continue
        break

    return "", tools_called


# ============================================================
# === 第四週第二天新增：確定性指標計算 ===
# ============================================================

def check_tool_selection(actual_tools: list[str], expected_tools: list[str]) -> dict:
    """
    評估工具選擇的正確性。
    指標：
      - Precision（精確率）：調用的工具中，有多少是期望中的？
      - Recall（召回率）：期望的工具中，有多少被調用了？

    Args:
        actual_tools:   實際調用的工具列表（可能有重複）
        expected_tools: 期望調用的工具列表

    Returns:
        包含 precision、recall、f1 的字典
    """
    # 去重比較（只關心調用了哪些，不關心順序和次數）
    actual_set = set(actual_tools)
    expected_set = set(expected_tools)

    if not expected_set and not actual_set:
        # 期望不調工具，實際也不調：完全正確
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "correct": True}

    if not expected_set:
        # 期望不調工具，但實際調了：精確率為 0
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "correct": False,
                "note": "意外調用了工具"}

    if not actual_set:
        # 期望調工具，但實際沒調：召回率為 0
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "correct": False,
                "note": "沒有調用任何期望的工具"}

    true_positives = len(actual_set & expected_set)
    precision = true_positives / len(actual_set) if actual_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "correct": f1 >= 0.8,   # 80% 以上 F1 算通過
        "actual": list(actual_set),
        "expected": list(expected_set),
        "missed": list(expected_set - actual_set),
        "unexpected": list(actual_set - expected_set),
    }


def check_output_content(actual_output: str, acceptable_outputs: list[str]) -> dict:
    """
    檢查輸出是否包含任意一個可接受的關鍵詞。
    這是最簡單的確定性 sensor：字符串包含匹配。

    Args:
        actual_output:     模型的實際輸出文字
        acceptable_outputs: 可接受的輸出關鍵詞列表

    Returns:
        {"passed": bool, "matched": str | None}
    """
    actual_lower = actual_output.lower()
    for keyword in acceptable_outputs:
        if keyword.lower() in actual_lower:
            return {"passed": True, "matched": keyword}
    return {"passed": False, "matched": None,
            "acceptable": acceptable_outputs}


# ============================================================
# 批量評估
# ============================================================

def run_evaluation(dataset: list[dict], verbose: bool = True) -> list[dict]:
    """
    對整個數據集跑一遍 agent，收集結果。

    Args:
        dataset: load_dataset() 讀取的案例列表
        verbose: 是否實時打印每個案例的結果

    Returns:
        每個案例的評估結果列表
    """
    results = []

    print(f"開始評估 {len(dataset)} 個案例...\n")

    for i, case in enumerate(dataset, 1):
        print(f"[{i:02d}/{len(dataset)}] {case['id']}: {case['question'][:40]}...")

        try:
            t0 = time.time()
            actual_output, tools_called = run_agent_for_eval(case["question"])
            elapsed = round(time.time() - t0, 2)

            # 計算指標
            tool_metrics = check_tool_selection(tools_called, case["expected_tools"])
            output_metrics = check_output_content(actual_output, case["acceptable_outputs"])

            passed = tool_metrics["correct"] and output_metrics["passed"]

            result = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "actual_output": actual_output[:200],   # 截斷，節省空間
                "tools_called": tools_called,
                "tool_metrics": tool_metrics,
                "output_metrics": output_metrics,
                "passed": passed,
                "latency_sec": elapsed,
            }

            status = "✅" if passed else "❌"
            if verbose:
                print(f"  {status} 工具: {tool_metrics['f1']:.2f} F1, "
                      f"輸出: {'通過' if output_metrics['passed'] else '失敗'}, "
                      f"{elapsed}s")

        except Exception as e:
            result = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "error": str(e),
                "passed": False,
            }
            print(f"  ❌ 錯誤：{str(e)[:60]}")

        results.append(result)

    return results


def print_evaluation_report(results: list[dict]):
    """打印評估結果的統計報告。"""
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    print("\n" + "=" * 60)
    print("評估報告")
    print("=" * 60)
    print(f"總體通過率：{passed}/{total} = {passed/total*100:.1f}%")

    # 按類別統計
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r.get("passed", False))

    print("\n按類別統計：")
    for cat, passes in sorted(by_cat.items()):
        cat_pass = sum(passes)
        print(f"  {cat:12s}: {cat_pass}/{len(passes)} = {cat_pass/len(passes)*100:.1f}%")

    # 失敗案例
    failures = [r for r in results if not r.get("passed", False)]
    if failures:
        print(f"\n失敗案例（{len(failures)} 個，建議人工分析）：")
        for f in failures:
            print(f"  [{f['id']}] {f['question'][:45]}")
            if "tool_metrics" in f:
                tm = f["tool_metrics"]
                om = f["output_metrics"]
                if not tm["correct"]:
                    print(f"    工具問題：期望 {tm.get('expected')}，"
                          f"實際 {tm.get('actual')}")
                if not om["passed"]:
                    print(f"    輸出問題：'{f['actual_output'][:60]}...'")
            elif "error" in f:
                print(f"    錯誤：{f['error'][:60]}")

    # 計算工具精確率/召回率宏平均
    tool_f1s = [r["tool_metrics"]["f1"]
                for r in results if "tool_metrics" in r]
    if tool_f1s:
        avg_f1 = sum(tool_f1s) / len(tool_f1s)
        print(f"\n工具選擇宏平均 F1：{avg_f1:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Day 2：確定性指標評估")
    print("=" * 60)

    # 加載數據集
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "week4_day1", "dataset.jsonl"
    )
    if not os.path.exists(dataset_path):
        print(f"先運行 week4_day1/eval_dataset.py 生成數據集")
    else:
        from eval_dataset import load_dataset
        dataset = load_dataset(dataset_path)

        # 先跑部分案例（省費用）
        sample = [c for c in dataset if c["category"] in ("normal", "edge")][:10]
        print(f"評估樣本（{len(sample)} 個案例，normal + edge 類別）...")

        results = run_evaluation(sample, verbose=True)
        print_evaluation_report(results)

        # 保存結果
        results_path = "eval_results.jsonl"
        with open(results_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n結果已保存到 {results_path}")
        print("Day 2 完成。下一步：week4_day3/judge.py — LLM-as-judge。")
