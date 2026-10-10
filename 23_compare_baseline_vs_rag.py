import csv
import json
import re
from pathlib import Path

BASELINE_PATH = Path("evaluation_results.json")
RAG_PATH = Path("rag_evaluation_results.json")

CSV_PATH = Path("baseline_vs_rag_comparison.csv")
SUMMARY_PATH = Path("baseline_vs_rag_summary.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_direction(text):
    text = text.lower()

    vertical = None
    horizontal = None

    if "down" in text or "bottom" in text:
        vertical = "down"
    elif "up" in text or "top" in text:
        vertical = "up"

    if "right" in text:
        horizontal = "right"
    elif "left" in text:
        horizontal = "left"

    return vertical, horizontal


def get_number(text):
    text = str(text).lower()

    word_numbers = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twice": 2,
        "thrice": 3,
    }

    match = re.search(r"\b\d+\b", text)

    if match:
        return int(match.group())

    for word, number in word_numbers.items():
        if re.search(r"\b" + word + r"\b", text):
            return number

    return None


def get_action(text):
    text = str(text).lower()

    if "put down" in text:
        return "put down"

    if "closed" in text or "close" in text:
        return "close"

    if "opened" in text or "open" in text:
        return "open"

    if "drank" in text or "drink" in text:
        return "drink"

    if "took" in text or "take" in text:
        return "take"

    return None


def evaluate(category, ground_truth, prediction):
    if category == "moving_direction":
        gt_vertical, gt_horizontal = get_direction(ground_truth)
        pred_vertical, pred_horizontal = get_direction(prediction)

        vertical_correct = gt_vertical == pred_vertical
        horizontal_correct = gt_horizontal == pred_horizontal

        if vertical_correct and horizontal_correct:
            return "Correct"
        elif vertical_correct or horizontal_correct:
            return "Partially Correct"
        else:
            return "Wrong"

    if category == "action_count":
        gt_number = get_number(ground_truth)
        pred_number = get_number(prediction)

        return (
            "Correct"
            if gt_number is not None and gt_number == pred_number
            else "Wrong"
        )

    if category == "action_sequence":
        gt_action = get_action(ground_truth)
        pred_action = get_action(prediction)

        return (
            "Correct"
            if gt_action is not None and gt_action == pred_action
            else "Wrong"
        )

    if category == "action_localization":
        return "Not scored"

    return "Unknown category"


print("Loading baseline and RAG results...")

baseline = load_json(BASELINE_PATH)
rag = load_json(RAG_PATH)

if len(baseline) != len(rag):
    raise ValueError(
        f"Question counts differ: baseline={len(baseline)}, RAG={len(rag)}"
    )

if not baseline:
    raise ValueError("The result files are empty.")

# Verify that each pair refers to the same evaluation example.
rows = []

for index, (base, rag_result) in enumerate(zip(baseline, rag)):
    for field in ("category", "video", "question", "ground_truth"):
        if base.get(field) != rag_result.get(field):
            raise ValueError(
                f"Mismatch in row {index + 1}, field '{field}': "
                f"baseline={base.get(field)!r}, "
                f"RAG={rag_result.get(field)!r}"
            )

    category = base["category"]
    gt = base["ground_truth"]
    baseline_answer = base["qwen_answer"]
    rag_answer = rag_result["prediction"]

    rows.append({
        "question_number": index + 1,
        "category": category,
        "video": base["video"],
        "question": base["question"],
        "ground_truth": gt,
        "baseline_answer": baseline_answer,
        "baseline_score": evaluate(category, gt, baseline_answer),
        "rag_answer": rag_answer,
        "rag_score": evaluate(category, gt, rag_answer),
        "rag_retrieved_chunks": "; ".join(
            f"{chunk['start']}-{chunk['end']}s: {chunk['description']}"
            for chunk in rag_result.get("retrieved_chunks", [])
        ),
    })


# Save detailed question-by-question results.
with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


categories = [
    "moving_direction",
    "action_count",
    "action_localization",
    "action_sequence",
]

summary = {}

print(f"\nQuestions compared: {len(rows)}")
print("All rows match by category, video, question, and ground truth.")

print("\nCATEGORY COMPARISON")
print("-" * 76)

for category in categories:
    category_rows = [
        row for row in rows
        if row["category"] == category
    ]

    if category == "action_localization":
        summary[category] = {
            "baseline": "Not scored",
            "rag": "Not scored",
            "reason": (
                "Numeric ground-truth intervals are unavailable "
                "in the result files."
            ),
        }

        print(f"\n{category}")
        print("Baseline: Not scored")
        print("Video RAG: Not scored")
        print("Reason: numeric ground-truth intervals are unavailable.")
        continue

    base_scores = [row["baseline_score"] for row in category_rows]
    rag_scores = [row["rag_score"] for row in category_rows]

    result = {}

    for name, scores in [
        ("baseline", base_scores),
        ("rag", rag_scores),
    ]:
        result[name] = {
            "correct": scores.count("Correct"),
            "partially_correct": scores.count("Partially Correct"),
            "wrong": scores.count("Wrong"),
            "total": len(scores),
        }

        result[name]["accuracy_percent"] = round(
            100 * result[name]["correct"] / len(scores), 1
        )

        # Direction gets half credit for a partially correct answer.
        weighted_points = (
            result[name]["correct"]
            + 0.5 * result[name]["partially_correct"]
        )

        result[name]["weighted_score_percent"] = round(
            100 * weighted_points / len(scores), 1
        )

    summary[category] = result

    print(f"\n{category} ({len(category_rows)} questions)")

    for name in ("baseline", "rag"):
        data = result[name]

        print(
            f"{name:8} | Correct: {data['correct']} | "
            f"Partial: {data['partially_correct']} | "
            f"Wrong: {data['wrong']} | "
            f"Exact accuracy: {data['accuracy_percent']}% | "
            f"Weighted: {data['weighted_score_percent']}%"
        )


# Overall aggregate excludes localization because it is not scorable.
scored_rows = [
    row for row in rows
    if row["category"] in (
        "moving_direction",
        "action_count",
        "action_sequence",
    )
]

overall = {}

for name, score_field in [
    ("baseline", "baseline_score"),
    ("rag", "rag_score"),
]:
    scores = [row[score_field] for row in scored_rows]

    correct = scores.count("Correct")
    partial = scores.count("Partially Correct")
    wrong = scores.count("Wrong")
    total = len(scores)

    overall[name] = {
        "correct": correct,
        "partially_correct": partial,
        "wrong": wrong,
        "total_scored": total,
        "exact_accuracy_percent": round(100 * correct / total, 1),
        "weighted_score_percent": round(
            100 * (correct + 0.5 * partial) / total,
            1,
        ),
    }

summary["overall_excluding_localization"] = overall

print("\nOVERALL (localization excluded)")
print("-" * 76)

for name in ("baseline", "rag"):
    data = overall[name]

    print(
        f"{name:8} | Exact correct: {data['correct']}/{data['total_scored']} "
        f"({data['exact_accuracy_percent']}%) | "
        f"Weighted score: {data['weighted_score_percent']}%"
    )

with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print(f"\nDetailed report saved to: {CSV_PATH}")
print(f"Summary saved to: {SUMMARY_PATH}")