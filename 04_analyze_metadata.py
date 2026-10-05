
import json
from pathlib import Path
from collections import Counter

metadata_dir = Path("metadata")
categories = [
    "moving_direction",
    "action_count",
    "action_localization",
    "action_sequence",
]

summary = {}

for category in categories:
    file_path = metadata_dir / f"{category}.json"

    with open(file_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    video_names = {item["video"] for item in questions}
    question_lengths = [len(item["question"]) for item in questions]

    summary[category] = {
        "question_count": len(questions),
        "unique_videos": len(video_names),
        "average_question_length": round(
            sum(question_lengths) / len(question_lengths), 2
        ),
        "has_candidates": sum(
            "candidates" in item for item in questions
        ),
        "has_timestamps": sum(
            "start" in item and "end" in item for item in questions
        ),
    }

    print(f"\n{category}")
    for key, value in summary[category].items():
        print(f"  {key}: {value}")

output_path = Path("metadata_summary.json")
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print("\nSummary saved to metadata_summary.json")

