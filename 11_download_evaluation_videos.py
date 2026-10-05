import json
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_ID = "FunAILab/TVBench"

with open("evaluation_sample.json", "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)

category_archives = {
    "moving_direction": "video/moving_direction.zip",
    "action_count": "video/action_count.zip",
    "action_localization": "video/action_localization.zip",
    "action_sequence": "video/action_sequence.zip",
}

output_dir = Path("evaluation_videos")
output_dir.mkdir(exist_ok=True)

videos_by_category = {}

for item in evaluation_data:
    category = item["category"]
    video = item["video"]

    videos_by_category.setdefault(category, set()).add(video)

for category, videos in videos_by_category.items():

    print(f"\nCategory: {category}")
    print("Videos needed:", len(videos))

    archive_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=category_archives[category],
        local_dir="."
    )

    print("Archive downloaded.")

    with zipfile.ZipFile(archive_path, "r") as archive:

        archive_names = {
            Path(name).name: name
            for name in archive.namelist()
        }

        for video_name in videos:

            if video_name not in archive_names:
                print("Video not found:", video_name)
                continue

            output_path = output_dir / video_name

            if output_path.exists():
                print("Already exists:", video_name)
                continue

            with archive.open(archive_names[video_name]) as source:
                with open(output_path, "wb") as destination:
                    destination.write(source.read())

            print("Extracted:", video_name)

print("\nEvaluation video preparation complete.")

print(
    "Total videos available:",
    len(list(output_dir.glob("*.mp4")))
)