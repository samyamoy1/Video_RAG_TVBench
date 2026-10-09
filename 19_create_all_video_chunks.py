import cv2
import json
from pathlib import Path

EVALUATION_PATH = Path("evaluation_sample.json")
VIDEO_DIR = Path("evaluation_videos")
OUTPUT_DIR = Path("video_chunks")

CHUNK_DURATION = 2.0
OVERLAP = 1.0
STEP = CHUNK_DURATION - OVERLAP

OUTPUT_DIR.mkdir(exist_ok=True)

with open(EVALUATION_PATH, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)

videos = sorted({
    item["video"]
    for item in evaluation_data
})

print(f"Videos to process: {len(videos)}")

total_chunks = 0

for video_name in videos:
    video_path = VIDEO_DIR / video_name

    if not video_path.exists():
        print(f"SKIPPING missing video: {video_name}")
        continue

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"SKIPPING unreadable video: {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        print(f"SKIPPING invalid video: {video_name}")
        continue

    duration = total_frames / fps

    chunks = []
    start = 0.0
    chunk_id = 0

    while start < duration:
        end = min(start + CHUNK_DURATION, duration)

        chunks.append({
            "chunk_id": chunk_id,
            "video": video_name,
            "start": round(start, 2),
            "end": round(end, 2)
        })

        chunk_id += 1
        start += STEP

    output_path = OUTPUT_DIR / f"{Path(video_name).stem}_chunks.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2)

    total_chunks += len(chunks)

    print(
        f"{video_name}: {len(chunks)} chunks "
        f"({duration:.1f} seconds)"
    )

print("\nChunk preparation complete.")
print(f"Videos processed: {len(videos)}")
print(f"Total chunks created: {total_chunks}")
print(f"Output folder: {OUTPUT_DIR}")