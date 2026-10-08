import cv2
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

VIDEO_PATH = "evaluation_videos/KQDX6.mp4"
CHUNKS_PATH = "video_chunks/KQDX6_chunks.json"

OUTPUT_PATH = "video_chunks/KQDX6_descriptions.json"

FRAMES_PER_CHUNK = 4
IMAGE_SIZE = (360, 360)

QUESTION = """Analyze the frames in chronological order.

Identify the main ACTION that happens during this segment.
Focus on changes or movements performed by the person.

Do NOT describe only what the person is holding or how the scene looks.

Answer with one short sentence describing the action.
"""

print("Loading Qwen 2.5 VL 3B...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Model loaded!")


with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)

results = []


for chunk in chunks:

    start_time = chunk["start"]
    end_time = chunk["end"]

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    frame_indices = [
        int(
            start_frame
            + i * (end_frame - start_frame)
            / (FRAMES_PER_CHUNK - 1)
        )
        for i in range(FRAMES_PER_CHUNK)
    ]

    frames = []

    for index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(frame)
        image = image.resize(IMAGE_SIZE)

        frames.append(image)

    if not frames:
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


    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    inputs = processor(
        text=[text],
        images=frames,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)


    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
        )


    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]


    description = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


    result = {
        "chunk_id": chunk["chunk_id"],
        "start": start_time,
        "end": end_time,
        "description": description.strip(),
    }

    results.append(result)

    print(
        f"Chunk {chunk['chunk_id']}: "
        f"{start_time:.1f}-{end_time:.1f}s"
    )

    print("Description:", description.strip())
    print()


cap.release()


with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False,
    )


print("Finished.")
print(f"Saved to: {OUTPUT_PATH}")