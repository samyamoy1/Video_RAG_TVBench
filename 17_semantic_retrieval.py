import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("video_chunks/KQDX6_descriptions.json")

QUESTION = "What action involves drinking?"

TOP_K = 5


print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

print("Embedding model loaded.")


print("Loading chunk descriptions...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


descriptions = [
    chunk["description"]
    for chunk in chunks
]


print("Creating embeddings...")

chunk_embeddings = model.encode(
    descriptions,
    normalize_embeddings=True,
)

question_embedding = model.encode(
    QUESTION,
    normalize_embeddings=True,
)


# Because the embeddings are normalized,
# dot product gives cosine similarity.
scores = chunk_embeddings @ question_embedding


ranked_indices = scores.argsort()[::-1][:TOP_K]


print("\nQuestion:")
print(QUESTION)

print("\nTop retrieved chunks:\n")


for rank, index in enumerate(ranked_indices, start=1):

    chunk = chunks[index]

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