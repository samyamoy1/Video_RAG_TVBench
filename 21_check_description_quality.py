import json
import re
from pathlib import Path

CHUNKS_DIR = Path("video_chunks")

description_files = sorted(
    CHUNKS_DIR.glob("*_descriptions.json")
)

total_descriptions = 0
bad_descriptions = []

print(f"Description files found: {len(description_files)}")

for path in description_files:

    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    for record in records:
        total_descriptions += 1

        description = record.get("description", "").strip()
        letters = sum(character.isalpha() for character in description)

        if (
            not description
            or letters == 0
            or letters / max(len(description), 1) < 0.30
        ):
            bad_descriptions.append({
                "file": path.name,
                "chunk_id": record.get("chunk_id"),
                "start": record.get("start"),
                "end": record.get("end"),
                "description": description,
            })

print(f"\nTotal descriptions: {total_descriptions}")
print(f"Suspicious descriptions: {len(bad_descriptions)}")

for record in bad_descriptions:
    print("\nFile:", record["file"])
    print("Chunk:", record["chunk_id"])
    print("Time:", record["start"], "-", record["end"])
    print("Description:", repr(record["description"]))

if not bad_descriptions:
    print("\nNo obvious punctuation-only or empty descriptions found.")