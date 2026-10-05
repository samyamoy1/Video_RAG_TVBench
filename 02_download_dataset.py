
from huggingface_hub import HfApi

DATASET_ID = "FunAILab/TVBench"

api = HfApi()

print("Checking TVBench dataset...")
info = api.dataset_info(DATASET_ID)

print("\nDataset:", info.id)
print("Dataset files:")

files = api.list_repo_files(
    repo_id=DATASET_ID,
    repo_type="dataset"
)

for file in files[:40]:
    print("-", file)

print(f"\nTotal files listed: {len(files)}")
print("\nInspection complete. No video files downloaded.")
