import json
from pathlib import Path

CATEGORIES = [
    "moving_direction",
    "action_count",
    "action_localization",
    "action_sequence",
]

SAMPLES_PER_CATEGORY = 5

metadata_dir = Path("metadata")
output_path = Path("evaluation_sample.json")

evaluation_data = []

for category in CATEGORIES:
    file_path = metadata_dir / f"{category}.json"

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    selected = data[:SAMPLES_PER_CATEGORY]

    for item in selected:
        evaluation_data.append({
            "category": category,
            "video": item["video"],
            "question": item["question"],
            "answer": item["answer"],
            "candidates": item.get("candidates", []),
            "start": item.get("start"),
            "end": item.get("end"),
        })

print("Evaluation sample created.")
print("Total questions:", len(evaluation_data))

print("\nQuestions per category:")

for category in CATEGORIES:
    count = sum(
        1 for item in evaluation_data
        if item["category"] == category
    )
    print(f"{category}: {count}")

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(evaluation_data, file, indent=2, ensure_ascii=False)

print(f"\nSaved to: {output_path}")