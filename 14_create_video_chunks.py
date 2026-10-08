import cv2
import json
from pathlib import Path

VIDEO_PATH = "evaluation_videos/KQDX6.mp4"

CHUNK_DURATION = 2.0
OVERLAP = 1.0

OUTPUT_DIR = Path("video_chunks")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Reading video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = total_frames / fps

cap.release()

print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Video duration: {duration:.2f} seconds")

step = CHUNK_DURATION - OVERLAP

chunks = []

start = 0.0
chunk_id = 0

while start < duration:

    end = min(start + CHUNK_DURATION, duration)

    chunks.append({
        "chunk_id": chunk_id,
        "start": round(start, 2),
        "end": round(end, 2)
    })

    chunk_id += 1
    start += step

output_path = OUTPUT_DIR / "KQDX6_chunks.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=2)

print(f"\nCreated {len(chunks)} chunks.")

print("\nFirst 10 chunks:")

for chunk in chunks[:10]:
    print(
        f"Chunk {chunk['chunk_id']}: "
        f"{chunk['start']} - {chunk['end']} sec"
    )

print(f"\nSaved to: {output_path}")