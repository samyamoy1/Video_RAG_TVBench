import cv2
import json
import torch

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


VIDEO_PATH = "evaluation_videos/KQDX6.mp4"
CHUNKS_PATH = Path("video_chunks/KQDX6_descriptions.json")

QUESTION = "What action involves drinking?"

TOP_K = 3
FRAMES_PER_CHUNK = 4
IMAGE_SIZE = (360, 360)


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

print("Embedding model loaded.")


print("\nLoading chunk descriptions...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# Create embeddings for all chunk descriptions
descriptions = [
    chunk["description"]
    for chunk in chunks
]

print("Creating chunk embeddings...")

chunk_embeddings = embedding_model.encode(
    descriptions,
    normalize_embeddings=True,
)

question_embedding = embedding_model.encode(
    QUESTION,
    normalize_embeddings=True,
)

# Calculate semantic similarity
scores = chunk_embeddings @ question_embedding

# Retrieve top K chunks
ranked_indices = scores.argsort()[::-1][:TOP_K]


print("\nQuestion:")
print(QUESTION)

print("\nRetrieved chunks:")

retrieved_chunks = []

for rank, index in enumerate(ranked_indices, start=1):

    chunk = chunks[index]

    retrieved_chunks.append(chunk)

    print(
        f"Rank {rank} | "
        f"Score: {scores[index]:.4f}"
    )

    print(
        f"Chunk {chunk['chunk_id']} | "
        f"{chunk['start']} - {chunk['end']} sec"
    )

    print(
        f"Description: {chunk['description']}"
    )

    print()


print("Loading Qwen 2.5 VL 3B...")

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

print("Qwen loaded.")


print("\nExtracting frames from retrieved chunks...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

frames = []


for chunk in retrieved_chunks:

    start_frame = int(chunk["start"] * fps)
    end_frame = int(chunk["end"] * fps)

    frame_indices = [
        int(
            start_frame
            + i * (end_frame - start_frame)
            / (FRAMES_PER_CHUNK - 1)
        )
        for i in range(FRAMES_PER_CHUNK)
    ]

    print(
        f"Chunk {chunk['chunk_id']}: "
        f"{chunk['start']} - {chunk['end']} sec"
    )

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            index
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        image = image.resize(IMAGE_SIZE)

        frames.append(image)


cap.release()

print(f"Total retrieved frames: {len(frames)}")


# Build the multimodal prompt
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
                "text": """Answer the question using only the
retrieved video frames.

Give a concise natural-language answer.

Question:
""" + QUESTION,
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

inputs = inputs.to(qwen_model.device)


print("\nAsking Qwen using retrieved video...")

with torch.inference_mode():

    generated_ids = qwen_model.generate(
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


print("\n========== FINAL RAG ANSWER ==========")
print("Question:", QUESTION)
print("Answer:", answer)
print("======================================")