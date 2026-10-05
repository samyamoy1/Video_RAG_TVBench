
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_ID = "FunAILab/TVBench"
CATEGORIES = [
    "moving_direction",
    "action_count",
    "action_localization",
    "action_sequence",
]

output_dir = Path("metadata")
output_dir.mkdir(exist_ok=True)

for category in CATEGORIES:
    print(f"\nInspecting: {category}")

    try:
        file_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"json/{category}.json",
            local_dir="."
        )

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        output_path = output_dir / f"{category}.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        print("Data type:", type(data).__name__)

        if isinstance(data, list):
            print("Number of entries:", len(data))
            if data:
                print("First entry:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False)[:2000])

        elif isinstance(data, dict):
            print("Top-level keys:", list(data.keys())[:20])
            print("Preview:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])

    except Exception as error:
        print("Could not inspect this category:", error)

print("\nMetadata inspection finished.")
