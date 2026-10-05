
import json
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_ID = "FunAILab/TVBench"

with open("metadata/moving_direction.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

video_name = questions[0]["video"]

print("Sample question:", questions[0]["question"])
print("Expected answer:", questions[0]["answer"])
print("Looking for:", video_name)

print("\nDownloading the movement video archive...")
archive_path = hf_hub_download(
    repo_id=DATASET_ID,
    repo_type="dataset",
    filename="video/moving_direction.zip",
    local_dir="."
)

output_dir = Path("sample_videos")
output_dir.mkdir(exist_ok=True)

with zipfile.ZipFile(archive_path, "r") as archive:
    matches = [
        name for name in archive.namelist()
        if Path(name).name == video_name
    ]

    if not matches:
        print("Video not found in archive.")
        print("Example archive entries:")
        print(archive.namelist()[:20])
    else:
        with archive.open(matches[0]) as source:
            output_path = output_dir / video_name
            with open(output_path, "wb") as destination:
                destination.write(source.read())

        print("\nSample video extracted to:", output_path)
        print("You can now open it with a video player.")
