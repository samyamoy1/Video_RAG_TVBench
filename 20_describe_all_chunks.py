import cv2
import json
import torch

from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

EVALUATION_PATH = Path("evaluation_sample.json")
VIDEO_DIR = Path("evaluation_videos")
CHUNKS_DIR = Path("video_chunks")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

FRAMES_PER_CHUNK = 4
IMAGE_SIZE = (360, 360)
MAX_NEW_TOKENS = 50

QUESTION = """Analyze these images in chronological order.

Identify the main action happening during this video segment.
Focus on actions, movements, and changes over time.

Do not describe only the person's appearance or the objects.
If no clear action is visible, describe the visible activity
without inventing an action.

Give one short sentence."""


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    temporary_path = path.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temporary_path.replace(path)


def load_valid_cache(output_path, chunks):
    """Reuse descriptions only when their timestamps still match."""

    if not output_path.exists():
        return {}

    existing = load_json(output_path)

    current_by_id = {
        int(chunk["chunk_id"]): chunk
        for chunk in chunks
    }

    cache = {}

    for record in existing:
        if "chunk_id" not in record:
            continue

        chunk_id = int(record["chunk_id"])
        current = current_by_id.get(chunk_id)

        if current is None:
            continue

        same_start = abs(
            float(record["start"]) - float(current["start"])
        ) < 0.02

        same_end = abs(
            float(record["end"]) - float(current["end"])
        ) < 0.02

        if (
            same_start
            and same_end
            and record.get("description", "").strip()
        ):
            cache[chunk_id] = {
                "chunk_id": chunk_id,
                "video": current["video"],
                "start": current["start"],
                "end": current["end"],
                "description": record["description"].strip(),
            }

    return cache


def extract_frames(cap, chunk, fps, total_frames):
    start_frame = min(
        int(float(chunk["start"]) * fps),
        total_frames - 1,
    )

    end_frame = min(
        max(int(float(chunk["end"]) * fps) - 1, start_frame),
        total_frames - 1,
    )

    indices = [
        round(
            start_frame
            + i * (end_frame - start_frame)
            / (FRAMES_PER_CHUNK - 1)
        )
        for i in range(FRAMES_PER_CHUNK)
    ]

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = image.resize(IMAGE_SIZE)

        frames.append(image)

    return frames


print("Loading evaluation videos...")

evaluation = load_json(EVALUATION_PATH)

video_names = sorted({
    item["video"]
    for item in evaluation
})

work = []
total_chunks = 0
cached_chunks = 0

for video_name in video_names:
    stem = Path(video_name).stem

    video_path = VIDEO_DIR / video_name
    chunks_path = CHUNKS_DIR / f"{stem}_chunks.json"
    output_path = CHUNKS_DIR / f"{stem}_descriptions.json"

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video is missing: {video_path}"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk metadata is missing: {chunks_path}"
        )

    chunks = load_json(chunks_path)

    # Attach the video filename to each chunk.
    for chunk in chunks:
        chunk["video"] = video_name

    cache = load_valid_cache(output_path, chunks)

    total_chunks += len(chunks)
    cached_chunks += len(cache)

    work.append({
        "video_name": video_name,
        "video_path": video_path,
        "chunks": chunks,
        "output_path": output_path,
        "cache": cache,
    })

print(f"Videos: {len(video_names)}")
print(f"Total chunks: {total_chunks}")
print(f"Descriptions already available: {cached_chunks}")
print(f"Descriptions still needed: {total_chunks - cached_chunks}")

if total_chunks == cached_chunks:
    print("All chunk descriptions already exist.")
    raise SystemExit(0)


print("\nLoading Qwen 2.5 VL 3B...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Qwen loaded.")


completed = cached_chunks
interrupted = False

try:
    for item in work:
        video_name = item["video_name"]
        video_path = item["video_path"]
        chunks = item["chunks"]
        output_path = item["output_path"]
        cache = item["cache"]

        missing = [
            chunk for chunk in chunks
            if int(chunk["chunk_id"]) not in cache
        ]

        if not missing:
            print(f"\nSkipping {video_name}: already complete.")
            continue

        print(
            f"\nProcessing {video_name}: "
            f"{len(missing)} chunks remaining."
        )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_video = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        try:
            for chunk in missing:
                frames = extract_frames(
                    cap,
                    chunk,
                    fps,
                    total_frames_in_video,
                )

                if not frames:
                    print(
                        f"Skipping unreadable chunk "
                        f"{chunk['chunk_id']}."
                    )
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[
                                {
                                    "type": "image",
                                    "image": frame,
                                }
                                for frame in frames
                            ],
                            {
                                "type": "text",
                                "text": QUESTION,
                            },
                        ],
                    }
                ]

                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = processor(
                    text=[prompt],
                    images=frames,
                    padding=True,
                    return_tensors="pt",
                )

                with torch.inference_mode():
                    generated_ids = model.generate(
                        ** inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )

                trimmed_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(
                        inputs.input_ids,
                        generated_ids,
                    )
                ]

                description = processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                if not description:
                    description = "No clear action identified."

                cache[int(chunk["chunk_id"])] = {
                    "chunk_id": int(chunk["chunk_id"]),
                    "video": video_name,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "description": description,
                }

                # Save immediately after each completed chunk.
                ordered_results = sorted(
                    cache.values(),
                    key=lambda record: record["chunk_id"],
                )

                save_json(output_path, ordered_results)

                completed += 1

                print(
                    f"[{completed}/{total_chunks}] "
                    f"{video_name} | "
                    f"Chunk {chunk['chunk_id']} "
                    f"({chunk['start']}-{chunk['end']}s)"
                )
                print(f"Description: {description}")

        finally:
            cap.release()

except KeyboardInterrupt:
    interrupted = True
    print("\nStopped by user. Completed descriptions are saved.")

print("\n========== SUMMARY ==========")
print(f"Videos: {len(video_names)}")
print(f"Chunks in dataset: {total_chunks}")

if interrupted:
    print("Run this script again to resume.")
else:
    completed_descriptions = 0

    for item in work:
        if item["output_path"].exists():
            completed_descriptions += len(
                load_json(item["output_path"])
            )

    print(
        f"Descriptions currently saved: "
        f"{completed_descriptions}"
    )

    if completed_descriptions == total_chunks:
        print("All video chunk descriptions are ready.")
    else:
        print(
            "Some chunks remain. Run the script again "
            "to resume missing descriptions."
        )