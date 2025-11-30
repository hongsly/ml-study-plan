"""Analyze low-performing questions across all metrics."""
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DATA_DIR = PROJECT_ROOT / "data" / "eval"

# Thresholds for identifying failures
THRESHOLDS = {
    "answer_correctness": 0.5,
    "context_precision": 0.7,
    "context_recall": 0.7,
    "answer_relevancy": 0.8,
    "faithfulness": 0.7,
}


def load_mode_data(mode: str):
    """Load evaluation results for a mode."""
    with_ref_path = EVAL_DATA_DIR / f"eval_results_{mode}_with_reference_filtered.json"
    no_ref_path = EVAL_DATA_DIR / f"eval_results_{mode}_no_reference_filtered.json"

    with open(with_ref_path, 'r') as f:
        results_with_ref = json.load(f)

    with open(no_ref_path, 'r') as f:
        results_no_ref = json.load(f)

    # Merge results - focus on 28 ragas questions only
    # results_with_ref: 28 ragas questions (indices 0-27)
    # results_no_ref: 10 manual + 28 ragas (indices 10-37 for ragas)
    merged = []
    for i in range(28):
        item = {
            "index": i,
            "question": results_with_ref[i]["user_input"],
            "reference": results_with_ref[i]["reference"],
            "response": results_with_ref[i]["response"],
            "retrieved_contexts": results_with_ref[i].get("retrieved_contexts", []),
            # Metrics from with_ref (ragas questions)
            "answer_correctness": results_with_ref[i].get("answer_correctness"),
            "context_precision": results_with_ref[i].get("context_precision"),
            "context_recall": results_with_ref[i].get("context_recall"),
            # Metrics from no_ref (offset by 10 for manual questions)
            "answer_relevancy": results_no_ref[10 + i].get("answer_relevancy"),
            "faithfulness": results_no_ref[10 + i].get("faithfulness"),
        }
        if item.get("faithfulness") is None:
            print(f'!!!Warning: No faithfulness for item: {item["index"]}, {item["question"]}')
            continue
        merged.append(item)

    return merged


def categorize_failures(data):
    """Categorize questions by failure patterns."""
    categories = defaultdict(list)

    for item in data:
        failures = []

        # Identify which metrics failed
        if item.get("answer_correctness", 1.0) < THRESHOLDS["answer_correctness"]:
            failures.append("low_answer_correctness")
        if item.get("context_recall", 1.0) < THRESHOLDS["context_recall"]:
            failures.append("low_context_recall")
        if item.get("context_precision", 1.0) < THRESHOLDS["context_precision"]:
            failures.append("low_context_precision")
        if item.get("answer_relevancy", 1.0) < THRESHOLDS["answer_relevancy"]:
            failures.append("low_answer_relevancy")
        if item.get("faithfulness", 1.0) < THRESHOLDS["faithfulness"]:
            failures.append("low_faithfulness")

        # Categorize by primary failure mode
        if not failures:
            categories["success"].append(item)
            continue

        # Pattern 1: Retrieval failure (didn't find relevant chunks)
        if "low_context_recall" in failures:
            if "low_answer_correctness" in failures:
                # Didn't retrieve right chunks → wrong answer
                categories["retrieval_failure"].append(item)
            elif "low_faithfulness" in failures:
                # Didn't retrieve right chunks → imagined correct answer
                categories["retrieval_failure_with_hallucination"].append(item)
            else:
                # Retrieved some but not all relevant chunks, answer still OK
                categories["partial_retrieval"].append(item)
            continue

        # Pattern 2: Generation failure (retrieval OK but answer wrong)
        if "low_answer_correctness" in failures:
            if "low_faithfulness" in failures:
                # Wrong answer + unfaithful → hallucination/fabrication
                categories["hallucination"].append(item)
            else:
                # Wrong answer but faithful to context → reasoning error
                categories["generation_failure"].append(item)
            continue

        # Pattern 3: Ranking issue (retrieved irrelevant chunks ranked high)
        if "low_context_precision" in failures:
            categories["ranking_issue"].append(item)
            continue

        # Pattern 4: Relevancy issue (answer doesn't address question)
        if "low_answer_relevancy" in failures:
            categories["relevancy_issue"].append(item)
            continue

        # Catch-all for other failure combinations
        categories["mixed_failure"].append(item)

    return categories


def print_failure_summary(categories, mode):
    """Print summary of failures."""
    print(f"\n{'='*80}")
    print(f"FAILURE ANALYSIS: {mode.upper()}")
    print(f"{'='*80}")

    total = sum(len(items) for items in categories.values())
    print(f"\nTotal questions: {total}")

    for category, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        count = len(items)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {category:25s}: {count:2d} ({pct:5.1f}%)")


def print_detailed_failures(categories, mode, top_n=3):
    """Print detailed information for worst failures."""
    print(f"\n{'='*80}")
    print(f"DETAILED FAILURES: {mode.upper()} (Top {top_n} per category)")
    print(f"{'='*80}")

    failure_categories = [k for k in categories.keys() if k != "success"]

    for category in failure_categories:
        items = categories[category]
        if not items:
            continue

        # Sort by answer_correctness (lowest first)
        sorted_items = sorted(items, key=lambda x: x.get("answer_correctness", 1.0))[:top_n]

        print(f"\n{'-'*80}")
        print(f"Category: {category.upper()}")
        print(f"{'-'*80}")

        for i, item in enumerate(sorted_items, 1):
            print(f"\n[{i}] Question {item['index']}:")
            print(f"Q: {item['question'][:150]}{'...' if len(item['question']) > 150 else ''}")
            print(f"\nMetrics:")
            print(f"  answer_correctness : {item.get('answer_correctness', 'N/A'):.3f}" if item.get('answer_correctness') is not None else "  answer_correctness : N/A")
            print(f"  context_recall     : {item.get('context_recall', 'N/A'):.3f}" if item.get('context_recall') is not None else "  context_recall     : N/A")
            print(f"  context_precision  : {item.get('context_precision', 'N/A'):.3f}" if item.get('context_precision') is not None else "  context_precision  : N/A")
            print(f"  answer_relevancy   : {item.get('answer_relevancy', 'N/A'):.3f}" if item.get('answer_relevancy') is not None else "  answer_relevancy   : N/A")
            print(f"  faithfulness       : {item.get('faithfulness', 'N/A'):.3f}" if item.get('faithfulness') is not None else "  faithfulness       : N/A")

            print(f"\n<Reference>{item['reference'][:200]}{'...' if len(item['reference']) > 200 else ''}</Reference>")
            print(f"\n<Generated>{item['response'][:200]}{'...' if len(item['response']) > 200 else ''}</Generated>")

            # Print first retrieved context
            if item['retrieved_contexts']:
                print(f"\nTop Retrieved Context (truncated):")
                print(f"<RetrievedContext>{item['retrieved_contexts'][0][:300]}...</RetrievedContext>")


def analyze_mode(mode, top_n=3):
    """Analyze failures for a specific mode."""
    print(f"\nLoading data for mode: {mode.upper()}...")
    data = load_mode_data(mode)

    categories = categorize_failures(data)

    print_failure_summary(categories, mode)
    print_detailed_failures(categories, mode, top_n)

    return categories


def main():
    """Analyze all modes."""
    modes = ["sparse", "hybrid", "dense"]
    # modes = ["dense"]

    all_categories = {}
    for mode in modes:
        all_categories[mode] = analyze_mode(mode, top_n=3)

    # Save analysis results
    output_path = EVAL_DATA_DIR / "error_analysis_summary.json"
    summary = {}
    for mode, categories in all_categories.items():
        summary[mode] = {
            category: len(items) for category, items in categories.items()
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
