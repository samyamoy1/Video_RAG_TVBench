
import json
from pathlib import Path

project_dir = Path(__file__).parent

with open(project_dir / "metadata_summary.json", "r", encoding="utf-8") as file:
    summary = json.load(file)

sample_video = project_dir / "sample_videos" / "video_14393.mp4"

report = {
    "project": "Video RAG and Motion Understanding",
    "dataset": "FunAILab/TVBench",
    "python_version": "3.11.3",
    "metadata_categories": summary,
    "sample_video": {
        "filename": sample_video.name,
        "exists": sample_video.exists(),
        "verified_manually": True,
        "question": "Which direction does the purple sphere move in the video?",
        "expected_answer": "Down and to the right.",
        "observed_answer": "Down and to the right."
    },
    "next_phase": [
        "Set up GPU-enabled PyTorch",
        "Load video frames",
        "Build motion understanding",
        "Build video question answering",
        "Evaluate predictions"
    ]
}

output_path = project_dir / "gpu_transfer_report.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("GPU transfer report created!")
print("Report:", output_path)
print("Sample video exists:", sample_video.exists())
print("Metadata categories:", len(summary))
