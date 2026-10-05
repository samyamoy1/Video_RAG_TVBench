import json
import cv2
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

EVALUATION_FILE = "evaluation_sample.json"
VIDEO_DIR = Path("evaluation_videos")
OUTPUT_FILE = "evaluation_results.json"

print("Loading Qwen 2.5 VL 3B...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Model loaded!")

with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    evaluation_data = json.load(file)

results = []

for number, item in enumerate(evaluation_data, start=1):

    video_path = VIDEO_DIR / item["video"]

    print(f"\n[{number}/{len(evaluation_data)}]")
    print("Category:", item["category"])
    print("Video:", item["video"])
    print("Question:", item["question"])

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = [
        int(i * (total_frames - 1) / 5)
        for i in range(6)
    ]

    frames = []

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = cap.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = image.resize((360, 360))
            frames.append(image)

    cap.release()

    print("Frames extracted:", len(frames))

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
                    "text": item["question"],
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
            max_new_tokens=64,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("Qwen answer:", answer)
    print("Ground truth:", item["answer"])

    results.append({
        "category": item["category"],
        "video": item["video"],
        "question": item["question"],
        "ground_truth": item["answer"],
        "qwen_answer": answer,
    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

print("\nEvaluation finished.")
print("Results saved to:", OUTPUT_FILE)