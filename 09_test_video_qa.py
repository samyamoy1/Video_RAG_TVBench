import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_PATH = "sample_videos/video_14393.mp4"

QUESTION = "Which direction does the purple sphere move in the video?"

print("Loading Qwen 2.5 VL 3B...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Model loaded!")
print("Reading video...")

cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", total_frames)

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
        frames.append(Image.fromarray(frame))

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

print("Asking Qwen...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

answer = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\nQuestion:", QUESTION)
print("Qwen's answer:", answer)