import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = Path("video_chunks/KQDX6_descriptions.json")

QUESTION = "What action involves drinking?"

TOP_K = 5


print("Loading chunk descriptions...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Loaded {len(chunks)} chunks.")


# Extract descriptions for retrieval
descriptions = [
    chunk["description"]
    for chunk in chunks
]


# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()

chunk_vectors = vectorizer.fit_transform(descriptions)

question_vector = vectorizer.transform([QUESTION])


# Compare the question with every chunk
similarities = cosine_similarity(
    question_vector,
    chunk_vectors
)[0]


# Get the most similar chunks
ranked_indices = similarities.argsort()[::-1][:TOP_K]


print("\nQuestion:")
print(QUESTION)

print("\nTop retrieved chunks:\n")


for rank, index in enumerate(ranked_indices, start=1):

    chunk = chunks[index]

    print(
        f"Rank {rank} | "
        f"Score: {similarities[index]:.4f}"
    )

    print(
        f"Chunk {chunk['chunk_id']} | "
        f"{chunk['start']} - {chunk['end']} sec"
    )

    print(
        f"Description: {chunk['description']}"
    )

    print()