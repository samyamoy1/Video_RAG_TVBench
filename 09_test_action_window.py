import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_PATH = "evaluation_videos/OY3LS.mp4"

QUESTION = """The images are shown in chronological order.
Focus on the action or change that happens between the frames,
not just what the person is holding.
What action did the person perform first?"""

# Dense window around the actual laptop-opening action
START_TIME = 2.4
END_TIME = 4.2

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
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames:", total_frames)
print("FPS:", fps)

start_frame = int(START_TIME * fps)
end_frame = int(END_TIME * fps)

print(f"Using action window: {START_TIME} - {END_TIME} seconds")

# 16 evenly spaced frames inside the short action window
frame_indices = [
    int(
        start_frame
        + i * (end_frame - start_frame) / 15
    )
    for i in range(16)
]

print("Frame indices:", frame_indices)

frames = []

for index in frame_indices:

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    success, frame = cap.read()

    if success:
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(
            Image.fromarray(frame)
        )

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
    for in_ids, out_ids in zip(
        inputs.input_ids,
        generated_ids
    )
]

answer = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\nQuestion:", QUESTION)
print(
    f"Action window: {START_TIME} - {END_TIME} seconds"
)
print("Qwen's answer:", answer)