import cv2
import json
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)


EVALUATION_PATH = Path("evaluation_sample.json")
VIDEO_DIR = Path("evaluation_videos")
CHUNKS_DIR = Path("video_chunks")
OUTPUT_PATH = Path("rag_direction_prompt_test.json")

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

TOP_K = 3
FRAMES_PER_CHUNK = 4
IMAGE_SIZE = (360, 360)
MAX_NEW_TOKENS = 64


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    temporary_path = path.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


def load_video_descriptions(video_name):
    stem = Path(video_name).stem
    path = CHUNKS_DIR / f"{stem}_descriptions.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing descriptions for {video_name}: {path}"
        )

    chunks = load_json(path)

    if not chunks:
        raise RuntimeError(
            f"No chunk descriptions found for {video_name}"
        )

    return chunks


def extract_frames(video_path, retrieved_chunks):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video metadata: {video_path}"
        )

    # Avoid duplicate frames from overlapping chunks.
    selected_indices = set()

    for chunk in retrieved_chunks:
        start_frame = min(
            int(float(chunk["start"]) * fps),
            total_frames - 1,
        )

        end_frame = min(
            max(
                int(float(chunk["end"]) * fps) - 1,
                start_frame,
            ),
            total_frames - 1,
        )

        indices = np.linspace(
            start_frame,
            end_frame,
            num=FRAMES_PER_CHUNK,
            dtype=int,
        )

        selected_indices.update(
            int(index) for index in indices
        )

    frames = []

    # Read selected frames in chronological order.
    for index in sorted(selected_indices):
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

    cap.release()

    return frames


print("Loading evaluation set...")

all_questions = load_json(EVALUATION_PATH)

# Evaluate only moving-direction questions.
evaluation_data = [
    item
    for item in all_questions
    if item["category"] == "moving_direction"
]

print(
    f"Moving-direction questions: {len(evaluation_data)}"
)


print("\nLoading semantic embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_ID,
    device="cpu",
)

print("Embedding model loaded.")


# Cache descriptions and embeddings for each video.
video_cache = {}

for item in evaluation_data:
    video_name = item["video"]

    if video_name in video_cache:
        continue

    chunks = load_video_descriptions(video_name)

    descriptions = [
        chunk["description"]
        for chunk in chunks
    ]

    print(f"Embedding chunks for {video_name}...")

    embeddings = embedding_model.encode(
        descriptions,
        normalize_embeddings=True,
    )

    video_cache[video_name] = {
        "chunks": chunks,
        "embeddings": embeddings,
    }


print(f"Videos indexed: {len(video_cache)}")


print("\nLoading Qwen 2.5 VL 3B...")

qwen_model = (
    Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
)

processor = AutoProcessor.from_pretrained(
    QWEN_MODEL_ID
)

print("Qwen loaded.")


# Resume previously saved results, if present.
if OUTPUT_PATH.exists():
    previous_results = load_json(OUTPUT_PATH)
else:
    previous_results = []

results_by_index = {
    int(result["index"]): result
    for result in previous_results
}

print(
    "Previously completed questions:",
    len(results_by_index),
)


for index, item in enumerate(evaluation_data):

    if index in results_by_index:
        print(f"Skipping completed question {index + 1}.")
        continue

    video_name = item["video"]
    video_path = VIDEO_DIR / video_name
    question = item["question"]

    print("\n" + "=" * 55)
    print(f"Question {index + 1}/{len(evaluation_data)}")
    print("Video:", video_name)
    print("Question:", question)

    cached = video_cache[video_name]

    chunks = cached["chunks"]
    chunk_embeddings = cached["embeddings"]

    # Retrieve semantically relevant chunks from this
    # question's video only.
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    scores = chunk_embeddings @ question_embedding

    ranked_indices = np.argsort(scores)[::-1][:TOP_K]

    retrieved_chunks = []

    for rank, chunk_index in enumerate(
        ranked_indices,
        start=1,
    ):
        chunk = chunks[int(chunk_index)].copy()

        chunk["retrieval_score"] = float(
            scores[chunk_index]
        )

        retrieved_chunks.append(chunk)

        print(
            f"Retrieved {rank}: "
            f"{chunk['start']}-{chunk['end']} seconds | "
            f"Score: {chunk['retrieval_score']:.4f}"
        )

        print("Description:", chunk["description"])

    # Extract frames from the retrieved chunks.
    frames = extract_frames(
        video_path,
        retrieved_chunks,
    )

    print("Frames extracted:", len(frames))

    # Correct indentation: this check belongs inside the loop.
    if not frames:
        raise RuntimeError(
            f"No frames extracted for question {index + 1}"
        )

    retrieved_notes = "\n".join(
        f"- {chunk['start']}-{chunk['end']} seconds: "
        f"{chunk['description']}"
        for chunk in retrieved_chunks
    )

    # Focus the prompt on both directions of movement.
    prompt = f"""Track the specified object across the retrieved
video frames in chronological order.

Determine its horizontal movement (left or right) and
vertical movement (up or down) separately.

Do not replace a direction of movement with a description
of its relationship to another object.

The retrieved descriptions may contain mistakes.
Prioritize the visual evidence in the frames.

Retrieved segment descriptions:
{retrieved_notes}

Question:
{question}

State both horizontal and vertical directions when
the visual evidence supports them. Be concise.
"""

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
                    "text": prompt,
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

    print("Generating answer...")

    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
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

    prediction = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    result = {
        "index": index,
        "category": item["category"],
        "video": video_name,
        "question": question,
        "ground_truth": item["answer"],
        "prediction": prediction,
        "retrieved_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "score": chunk["retrieval_score"],
                "description": chunk["description"],
            }
            for chunk in retrieved_chunks
        ],
        "num_frames": len(frames),
    }

    results_by_index[index] = result

    # Save after each question so completed work is preserved.
    save_json(
        OUTPUT_PATH,
        [
            results_by_index[i]
            for i in sorted(results_by_index)
        ],
    )

    print("Prediction:", prediction)
    print(f"Saved progress to {OUTPUT_PATH}")

    del inputs, generated_ids, trimmed_ids, frames

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\n" + "=" * 55)
print("Direction prompt test finished.")
print(f"Questions completed: {len(results_by_index)}")
print(f"Results saved to: {OUTPUT_PATH}")